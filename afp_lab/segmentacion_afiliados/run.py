"""Demo end-to-end de `segmentacion_afiliados`.

    python -m afp_lab.segmentacion_afiliados.run --n-afiliados 20000

Genera el universo `cuprum` propio y las tablas externas (proyectos 02/03), construye
features "as-of" en un corte de referencia (y en `corte-6`, `corte+6`, `corte+12`), elige
k y algoritmo por silueta + estabilidad, perfila los segmentos, valida su separacion de
negocio (traspaso/APV/cese), entrena el clasificador explicativo y arma las matrices de
transicion a 6 y 12 meses. Guarda todo en `--salida` (por defecto
`salidas/segmentacion_afiliados/`).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .evaluate import (construir_targets_negocio, entrenar_clasificador, matriz_transicion,
                       perfil_segmentos, separacion_negocio, tasas_por_segmento)
from .features import VENTANA_DENSIDAD, construir_features
from .segmentar import (_silueta_muestreada, ajustar, asignar_fusionado, escalar,
                        fusionar_segmentos_chicos, seleccionar_k)
from .synthetic import add_months, generar_datos, generar_tablas_externas, periodo_to_idx

DESFASE_RIM = 2
HORIZONTE_TRANSICION = 12
HORIZONTE_ESTABILIDAD = 6


def _elegir_corte_referencia(periodo_inicio: int, periodo_fin: int) -> int:
    """Corte de referencia mas reciente que deja: (a) `VENTANA_DENSIDAD` meses de historia
    completa dentro del rango simulado, (b) un corte `corte-6` igual de completo para la
    estabilidad temporal, y (c) `HORIZONTE_TRANSICION` (12) meses libres hacia adelante para
    la matriz de transicion y la validacion de negocio (el horizonte mas largo, 6 meses,
    cabe holgado dentro de eso)."""
    corte_min = add_months(periodo_inicio, VENTANA_DENSIDAD + DESFASE_RIM + HORIZONTE_ESTABILIDAD)
    corte_max = add_months(periodo_fin, -HORIZONTE_TRANSICION)
    if periodo_to_idx(corte_max) < periodo_to_idx(corte_min):
        raise ValueError(f"Rango [{periodo_inicio},{periodo_fin}] insuficiente: se necesitan al menos "
                         f"{VENTANA_DENSIDAD + DESFASE_RIM + HORIZONTE_ESTABILIDAD + HORIZONTE_TRANSICION} "
                         "meses de holgura entre historia y horizonte.")
    return corte_max


def main(n_afiliados: int = 5000, periodo_inicio: int = 202201, periodo_fin: int = 202609,
        seed: int = 42, salida: str = "salidas/segmentacion_afiliados", perfil: str = "cuprum") -> dict:
    t0 = time.time()
    salida_dir = Path(salida)
    (salida_dir / "reportes").mkdir(parents=True, exist_ok=True)

    print(f"Generando universo propio (n_afiliados={n_afiliados}, seed={seed}, perfil={perfil})...")
    tablas = generar_datos(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                           periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    print(f"  listo en {time.time() - t0:.1f}s")

    t1 = time.time()
    print("Generando tablas externas (02 churn_traspaso, 03 propension_apv, sin modificarlas)...")
    externas = generar_tablas_externas(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                       periodo_fin=periodo_fin, seed=seed)
    print(f"  listo en {time.time() - t1:.1f}s")

    corte = _elegir_corte_referencia(periodo_inicio, periodo_fin)
    corte_t6 = add_months(corte, -HORIZONTE_ESTABILIDAD)
    corte_mas6 = add_months(corte, 6)
    corte_mas12 = add_months(corte, 12)
    print(f"Corte de referencia: {corte} (estabilidad vs. {corte_t6}; transicion a {corte_mas6}/{corte_mas12})")

    t2 = time.time()
    feats = construir_features(tablas, externas, corte)
    feats_t6 = construir_features(tablas, externas, corte_t6)
    print(f"Features construidas ({len(feats)} + {len(feats_t6)} filas) en {time.time() - t2:.1f}s")

    t3 = time.time()
    tabla_k, ganador, scaler = seleccionar_k(feats, feats_t6, seed=seed)
    tabla_k.to_csv(salida_dir / "reportes" / "seleccion_k.csv", index=False)
    print(f"\nSeleccion de k y algoritmo [{time.time() - t3:.1f}s]:")
    print(tabla_k.to_string(index=False))
    print(f"Ganador: {ganador.algoritmo} k={ganador.k} silueta={ganador.silueta:.4f} "
         f"ari_resampling={ganador.ari_resampling:.4f} ari_temporal={ganador.ari_temporal:.4f} "
         f"tamanos_ok={ganador.tamanos_ok} cumple_aceptacion={ganador.cumple}")

    labels, modelo, scaler = ajustar(feats, ganador.algoritmo, ganador.k, seed=seed, scaler=scaler)
    X_ref, _ = escalar(feats, scaler)
    labels, mapa_fusion = fusionar_segmentos_chicos(X_ref, labels)
    k_final = len(set(mapa_fusion.values()))
    if k_final < ganador.k:
        silueta_final = _silueta_muestreada(X_ref, labels, seed=seed)
        print(f"\nFusion de segmentos chicos (< 3 %): k baja de {ganador.k} a {k_final}. "
             f"Silueta final (post-fusion): {silueta_final:.4f}")
    else:
        silueta_final = ganador.silueta
        print(f"\nSin fusiones: los {ganador.k} segmentos ya cumplen el piso de 3 %.")
    feats = feats.copy()
    feats["segmento"] = labels
    tam_ok_final = bool((pd.Series(labels).value_counts(normalize=True) >= 0.03).all())
    cumple_final = (silueta_final >= 0.20) and (ganador.ari_resampling >= 0.60) and tam_ok_final
    print(f"Aceptacion final (post-fusion): silueta={silueta_final:.4f} (>=0,20), "
         f"ari_resampling={ganador.ari_resampling:.4f} (>=0,60), segmentos>=3%={tam_ok_final}, "
         f"cumple={cumple_final}")

    perfil_seg = perfil_segmentos(feats)
    perfil_seg.to_csv(salida_dir / "reportes" / "perfil_segmentos.csv", index=False)
    print("\nPerfil de segmentos (medianas / modas):")
    print(perfil_seg.to_string(index=False))

    targets = construir_targets_negocio(tablas, externas, feats["afiliado_id"].to_numpy(), corte)
    tasas = tasas_por_segmento(feats, targets)
    tasas.to_csv(salida_dir / "reportes" / "tasas_negocio_por_segmento.csv", index=False)
    print("\nTasas de negocio por segmento (cese 3m, traspaso 6m, APV 6m):")
    print(tasas.to_string(index=False))
    sep = separacion_negocio(tasas)
    print(f"Separacion de negocio: razones max/min = {sep['razones']}, "
         f"tasas con razon >= 2: {sep['n_tasas_con_razon_2']}/3, cumple_aceptacion={sep['cumple']}")

    t4 = time.time()
    clas = entrenar_clasificador(feats, seed=seed)
    clas["importancias"].to_csv(salida_dir / "reportes" / "importancias_clasificador.csv", index=False)
    print(f"\nClasificador explicativo (LightGBM) [{time.time() - t4:.1f}s]: "
         f"accuracy={clas['accuracy']:.4f} (train={clas['n_train']}, test={clas['n_test']})")
    print(clas["importancias"].head(15).to_string(index=False))

    feats_mas6 = construir_features(tablas, externas, corte_mas6)
    feats_mas12 = construir_features(tablas, externas, corte_mas12)
    labels_mas6 = asignar_fusionado(modelo, scaler, feats_mas6, mapa_fusion)
    labels_mas12 = asignar_fusionado(modelo, scaler, feats_mas12, mapa_fusion)

    trans6, n6 = matriz_transicion(feats["afiliado_id"].to_numpy(), labels,
                                   feats_mas6["afiliado_id"].to_numpy(), labels_mas6)
    trans12, n12 = matriz_transicion(feats["afiliado_id"].to_numpy(), labels,
                                     feats_mas12["afiliado_id"].to_numpy(), labels_mas12)
    trans6.to_csv(salida_dir / "reportes" / "transicion_6m.csv")
    trans12.to_csv(salida_dir / "reportes" / "transicion_12m.csv")
    print(f"\nMatriz de transicion a 6 meses (n comun={n6}):")
    print(trans6.round(3).to_string())
    print(f"\nMatriz de transicion a 12 meses (n comun={n12}):")
    print(trans12.round(3).to_string())

    feats.to_parquet(salida_dir / "dataset_segmentos.parquet", index=False)
    total = time.time() - t0
    print(f"\nRuntime total: {total:.1f}s. Salidas en {salida_dir.resolve()}")
    return {
        "tabla_k": tabla_k, "ganador": ganador, "k_final": k_final, "silueta_final": silueta_final,
        "cumple_final": cumple_final, "mapa_fusion": mapa_fusion, "feats": feats,
        "perfil_segmentos": perfil_seg, "tasas_negocio": tasas, "separacion_negocio": sep,
        "clasificador": clas, "transicion_6m": trans6, "transicion_12m": trans12, "corte": corte,
        "modelo": modelo, "scaler": scaler,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-afiliados", type=int, default=5000)
    p.add_argument("--periodo-inicio", type=int, default=202201)
    p.add_argument("--periodo-fin", type=int, default=202609)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salida", type=str, default="salidas/segmentacion_afiliados")
    p.add_argument("--perfil", type=str, default="cuprum", choices=["sistema", "cuprum"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(n_afiliados=args.n_afiliados, periodo_inicio=args.periodo_inicio, periodo_fin=args.periodo_fin,
        seed=args.seed, salida=args.salida, perfil=args.perfil)
