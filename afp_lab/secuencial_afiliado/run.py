"""Demo end-to-end de `secuencial_afiliado` (proyecto 08).

    PYTHONPATH=<worktree> python3 -m afp_lab.secuencial_afiliado.run --n-afiliados 10000

Genera el universo + tablas de 02/04, construye los datasets y splits de CADA cabeza con las
funciones de origen (sin modificarlas), entrena para cada cabeza: LightGBM "de produccion"
(02/04), LightGBM "tabular con lags" (misma secuencia, aplanada), la red multi-tarea y la red
single-task; evalua todo en el MISMO test de cada cabeza con `common.metrics` + IC bootstrap, y
guarda las tablas en `--salida` (por defecto `salidas/secuencial_afiliado/`).

Alcance de esta entrega (README, "Pendientes"): solo las cabezas de traspaso (02) y cese-6m
(04), que el brief marca como prioritarias; APV (03), RIM (regresion) y la comparacion del
embedding contra la segmentacion de 06 quedan fuera por presupuesto de tiempo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso import HORIZONTE_MESES as HORIZONTE_TRASPASO
from afp_lab.churn_traspaso.features import (columnas_features as columnas_features_traspaso,
                                             construir_dataset as construir_dataset_traspaso,
                                             cortes_evaluables, split_temporal as split_temporal_traspaso)
from afp_lab.churn_traspaso.model import baseline_tasa_base as baseline_tasa_base_traspaso
from afp_lab.churn_traspaso.model import entrenar as entrenar_lgbm_traspaso
from afp_lab.churn_traspaso.model import predecir as predecir_lgbm_traspaso
from afp_lab.riesgo_cese_cotizacion.model import ModeloCese
from afp_lab.riesgo_cese_cotizacion.model import baseline_tasa_base as baseline_tasa_base_cese
from afp_lab.riesgo_cese_cotizacion.model import predecir_regla_negocio_cese, predecir_tasa_base
from afp_lab.riesgo_cese_cotizacion.run import _construir_dataset as construir_dataset_cese
from afp_lab.riesgo_cese_cotizacion.run import _split_temporal as split_temporal_cese

from .evaluate import conclusion_vale_la_pena, tabla_cabeza
from .features import construir_matrices_panel, construir_secuencias
from .model import DatosCabeza, entrenar_lgbm_lags, entrenar_red, obtener_embedding, predecir_lgbm_lags, predecir_red
from .synthetic import PERIODO_FIN, PERIODO_INICIO, generar_tablas

HORIZONTE_CESE = 6
COLUMNA_TARGET_CESE = "y6"
COLUMNA_HORIZ_COMPLETO_CESE = "horizonte6_completo"


def _preparar_cabeza_traspaso(tablas: dict, max_cortes_train: int, n_test: int, n_valid: int) -> dict:
    cortes = cortes_evaluables(tablas, horizonte=HORIZONTE_TRASPASO)
    dataset = construir_dataset_traspaso(tablas, cortes, horizonte=HORIZONTE_TRASPASO)
    split = split_temporal_traspaso(cortes, n_test=n_test, n_valid=n_valid, horizonte=HORIZONTE_TRASPASO,
                                    max_cortes_train=max_cortes_train)
    train = dataset[dataset["periodo_corte"].isin(split["train"])].reset_index(drop=True)
    val = dataset[dataset["periodo_corte"].isin(split["valid"])].reset_index(drop=True)
    test = dataset[dataset["periodo_corte"].isin(split["test"])].reset_index(drop=True)
    return {"dataset": dataset, "cortes": cortes, "split": split, "train": train, "val": val,
           "test": test, "columna_y": "y", "columnas_lgbm": columnas_features_traspaso(dataset)}


def _preparar_cabeza_cese(tablas: dict, periodo_inicio: int, periodo_fin: int,
                          max_cortes_train: int | None) -> dict:
    dataset, cortes = construir_dataset_cese(tablas, periodo_inicio, periodo_fin)
    pob = dataset[(dataset["pop_cese"] == 1) & dataset[COLUMNA_HORIZ_COMPLETO_CESE]].copy()
    split = split_temporal_cese(pob, cortes, horizonte=HORIZONTE_CESE)
    if max_cortes_train is not None and len(split["cortes"]["train"]) > max_cortes_train:
        cortes_train_recortados = split["cortes"]["train"][-max_cortes_train:]
        split["train"] = split["train"][split["train"]["periodo_corte"].isin(cortes_train_recortados)]
        split["cortes"]["train"] = cortes_train_recortados
    return {"dataset": pob, "cortes": cortes, "split": split, "train": split["train"].reset_index(drop=True),
           "val": split["val"].reset_index(drop=True), "test": split["test"].reset_index(drop=True),
           "columna_y": COLUMNA_TARGET_CESE}


def _construir_datos_cabeza(mats, afiliados: pd.DataFrame, cab: dict) -> DatosCabeza:
    pares = lambda df: df[["afiliado_id", "periodo_corte"]]  # noqa: E731
    return DatosCabeza(
        train=construir_secuencias(mats, afiliados, pares(cab["train"])),
        y_train=cab["train"][cab["columna_y"]].to_numpy(dtype=np.float32),
        val=construir_secuencias(mats, afiliados, pares(cab["val"])),
        y_val=cab["val"][cab["columna_y"]].to_numpy(dtype=np.float32),
        test=construir_secuencias(mats, afiliados, pares(cab["test"])),
        y_test=cab["test"][cab["columna_y"]].to_numpy(dtype=np.float32),
    )


def main(n_afiliados: int = 10_000, periodo_inicio: int = PERIODO_INICIO, periodo_fin: int = PERIODO_FIN,
         seed: int = 42, perfil: str = "cuprum", salida: str | None = None,
         max_cortes_train_traspaso: int = 7, max_cortes_train_cese: int = 7,
         epocas: int = 6, paciencia: int = 2, hidden: int = 32, dim_embedding: int = 24,
         batch_size: int = 768, lr: float = 3e-3, pesos: dict | None = None,
         n_boot: int = 200) -> dict:
    t0 = time.time()
    salida_dir = Path(salida or "salidas/secuencial_afiliado")
    salida_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Generando universo + tablas de 02/04 (n_afiliados={n_afiliados}, seed={seed}, "
         f"perfil={perfil})...")
    tablas = generar_tablas(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                            periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    print(f"      listo en {time.time() - t0:.1f}s")

    print("[2/6] Dataset + split de cada cabeza (funciones de 02 y 04, sin modificar)...")
    t1 = time.time()
    cab_tr = _preparar_cabeza_traspaso(tablas, max_cortes_train_traspaso, n_test=3, n_valid=3)
    cab_ce = _preparar_cabeza_cese(tablas, periodo_inicio, periodo_fin, max_cortes_train_cese)
    print(f"      traspaso: train/val/test cortes = {cab_tr['split']['train']}/... "
         f"{len(cab_tr['split']['train'])}/{len(cab_tr['split']['valid'])}/{len(cab_tr['split']['test'])} cortes, "
         f"filas {len(cab_tr['train'])}/{len(cab_tr['val'])}/{len(cab_tr['test'])}, "
         f"test cortes={cab_tr['split']['test']}")
    print(f"      cese-6m:  cortes train/val/test = "
         f"{len(cab_ce['split']['cortes']['train'])}/{len(cab_ce['split']['cortes']['val'])}/"
         f"{len(cab_ce['split']['cortes']['test'])}, filas {len(cab_ce['train'])}/{len(cab_ce['val'])}/"
         f"{len(cab_ce['test'])}, test cortes={cab_ce['split']['cortes']['test']}")
    if sorted(cab_tr["split"]["test"]) != sorted(cab_ce["split"]["cortes"]["test"]):
        print("      AVISO: los cortes de test de traspaso y cese NO coinciden (ver README, limitaciones).")
    print(f"      listo en {time.time() - t1:.1f}s")

    print("[3/6] Secuencias (panel hasta corte-2) para cada cabeza...")
    t2 = time.time()
    mats = construir_matrices_panel(tablas, n_afiliados)
    datos_tr = _construir_datos_cabeza(mats, tablas["afiliados"], cab_tr)
    datos_ce = _construir_datos_cabeza(mats, tablas["afiliados"], cab_ce)
    datos = {"traspaso": datos_tr, "cese": datos_ce}
    print(f"      listo en {time.time() - t2:.1f}s")

    print("[4/6] Baselines LightGBM (produccion de 02/04 y tabular-con-lags)...")
    t3 = time.time()
    modelo_lgbm_tr = entrenar_lgbm_traspaso(cab_tr["train"], cab_tr["val"], seed=seed)
    pred_lgbm_tr = predecir_lgbm_traspaso(modelo_lgbm_tr, cab_tr["test"], uso="ranking")
    pred_tasabase_tr = baseline_tasa_base_traspaso(cab_tr["train"], cab_tr["test"])

    modelo_lgbm_ce = ModeloCese.fit(cab_ce["train"], cab_ce["val"], COLUMNA_TARGET_CESE)
    pred_lgbm_ce = modelo_lgbm_ce.predict_proba_cruda(cab_ce["test"])
    tasas_ce, tasa_global_ce = baseline_tasa_base_cese(cab_ce["train"], COLUMNA_TARGET_CESE)
    pred_tasabase_ce = predecir_tasa_base(tasas_ce, tasa_global_ce, cab_ce["test"])
    pred_regla_ce = predecir_regla_negocio_cese(cab_ce["test"])

    lags_tr = entrenar_lgbm_lags(datos_tr.train, datos_tr.y_train, datos_tr.val, datos_tr.y_val, seed=seed)
    pred_lags_tr = predecir_lgbm_lags(lags_tr, datos_tr.test)
    lags_ce = entrenar_lgbm_lags(datos_ce.train, datos_ce.y_train, datos_ce.val, datos_ce.y_val, seed=seed)
    pred_lags_ce = predecir_lgbm_lags(lags_ce, datos_ce.test)
    print(f"      listo en {time.time() - t3:.1f}s")

    print("[5/6] Red secuencial (multi-tarea y single-task por cabeza)...")
    t4 = time.time()
    pesos = pesos or {"traspaso": 0.6, "cese": 0.4}  # traspaso es la tarea mas dificil (tasa base ~3%);
    # elegido en validacion (n=3000, ver README "Decisiones tomadas"), no con el test.
    kwargs_red = dict(seed=seed, epocas=epocas, paciencia=paciencia, hidden=hidden,
                      dim_embedding=dim_embedding, batch_size=batch_size, lr=lr, pesos=pesos)
    res_multi = entrenar_red(datos, cabezas=["traspaso", "cese"], **kwargs_red)
    res_single_tr = entrenar_red(datos, cabezas=["traspaso"], **kwargs_red)
    res_single_ce = entrenar_red(datos, cabezas=["cese"], **kwargs_red)
    tiempo_red = time.time() - t4
    print(f"      listo en {tiempo_red:.1f}s (epocas: multi={len(res_multi['historia_val'])}, "
         f"single_tr={len(res_single_tr['historia_val'])}, single_ce={len(res_single_ce['historia_val'])})")

    pred_multi_tr = predecir_red(res_multi["modelo"], datos_tr.test, "traspaso")
    pred_multi_ce = predecir_red(res_multi["modelo"], datos_ce.test, "cese")
    pred_single_tr = predecir_red(res_single_tr["modelo"], datos_tr.test, "traspaso")
    pred_single_ce = predecir_red(res_single_ce["modelo"], datos_ce.test, "cese")

    print("[6/6] Metricas + IC bootstrap...")
    t5 = time.time()
    tabla_tr = tabla_cabeza(datos_tr.y_test, {
        "lightgbm_produccion": pred_lgbm_tr, "lightgbm_tabular_lags": pred_lags_tr,
        "red_multitarea": pred_multi_tr, "red_single_task": pred_single_tr,
        "baseline_tasa_base": pred_tasabase_tr,
    }, n_boot=n_boot, seed=seed)
    tabla_ce = tabla_cabeza(datos_ce.y_test, {
        "lightgbm_produccion": pred_lgbm_ce, "lightgbm_tabular_lags": pred_lags_ce,
        "red_multitarea": pred_multi_ce, "red_single_task": pred_single_ce,
        "baseline_tasa_base": pred_tasabase_ce, "baseline_regla_negocio": pred_regla_ce,
    }, n_boot=n_boot, seed=seed)
    conclusion = conclusion_vale_la_pena({"traspaso_6m": tabla_tr, "cese_6m": tabla_ce})
    print(f"      listo en {time.time() - t5:.1f}s")

    print("\n=== Traspaso 6m (02) ===")
    print(tabla_tr.to_string(index=False))
    print("\n=== Cese 6m (04) ===")
    print(tabla_ce.to_string(index=False))
    print(f"\nConclusion (BRIEF.md): {conclusion['veredicto']} -- deltas AUC (red_multitarea - "
         f"lightgbm_produccion): {conclusion['deltas_auc']}")

    tabla_tr.to_csv(salida_dir / "metricas_traspaso_6m.csv", index=False)
    tabla_ce.to_csv(salida_dir / "metricas_cese_6m.csv", index=False)
    (salida_dir / "conclusion.json").write_text(json.dumps(conclusion, indent=2, ensure_ascii=False),
                                                encoding="utf-8")

    emb_ref = obtener_embedding(res_multi["modelo"], datos_tr.test)
    pd.DataFrame(emb_ref, columns=[f"emb_{i}" for i in range(emb_ref.shape[1])]).assign(
        afiliado_id=datos_tr.test.meta["afiliado_id"].to_numpy(),
        periodo_corte=datos_tr.test.meta["periodo_corte"].to_numpy(),
    ).to_csv(salida_dir / "embedding_test_traspaso.csv", index=False)

    total = time.time() - t0
    print(f"\nRuntime total: {total:.1f}s (red: {tiempo_red:.1f}s). Salidas en {salida_dir.resolve()}")
    return {"tabla_traspaso": tabla_tr, "tabla_cese": tabla_ce, "conclusion": conclusion,
           "runtime_total_s": total, "runtime_red_s": tiempo_red}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-afiliados", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perfil", type=str, default="cuprum", choices=("sistema", "cuprum"))
    ap.add_argument("--periodo-inicio", type=int, default=PERIODO_INICIO)
    ap.add_argument("--periodo-fin", type=int, default=PERIODO_FIN)
    ap.add_argument("--salida", type=str, default=None)
    ap.add_argument("--max-cortes-train-traspaso", type=int, default=7)
    ap.add_argument("--max-cortes-train-cese", type=int, default=7)
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--paciencia", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dim-embedding", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=768)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-boot", type=int, default=200)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    main(n_afiliados=a.n_afiliados, periodo_inicio=a.periodo_inicio, periodo_fin=a.periodo_fin,
        seed=a.seed, perfil=a.perfil, salida=a.salida,
        max_cortes_train_traspaso=a.max_cortes_train_traspaso, max_cortes_train_cese=a.max_cortes_train_cese,
        epocas=a.epocas, paciencia=a.paciencia, hidden=a.hidden, dim_embedding=a.dim_embedding,
        batch_size=a.batch_size, lr=a.lr, n_boot=a.n_boot)
