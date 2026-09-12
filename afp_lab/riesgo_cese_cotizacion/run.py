"""Demo end-to-end de `riesgo_cese_cotizacion`.

    python -m afp_lab.riesgo_cese_cotizacion.run --n-afiliados 20000

Genera el universo sintetico, construye features "as-of" para una ventana de cortes
mensuales, entrena los 3 modelos (cese 3 m, cese 6 m, reingreso) con calibracion
isotonica, los compara contra los baselines obligatorios y guarda tablas de resultados,
modelos e importancias en `--salida` (por defecto `salidas/riesgo_cese_cotizacion/`).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import (agregar_tramos, curva_calibracion, evaluar, evaluar_por_segmento,
                       metrica_negocio_agregado, metrica_negocio_captura)
from .features import agregar_targets, construir_features
from .model import (ModeloCese, baseline_tasa_base, predecir_regla_negocio_cese,
                    predecir_regla_negocio_reingreso, predecir_tasa_base)
from .synthetic import add_months, generar_datos, periodo_to_idx

N_CORTES = 24     # cortes mensuales usados (compartidos por los 3 targets)
N_TEST = 3        # ultimos cortes reservados para test (CONVENCIONES: "ultimos 3 cortes")
FRAC_VAL = 0.25   # fraccion (de los cortes de train+val) usada para calibrar

TARGETS = {
    "y3": {"horizonte": 3, "poblacion": "pop_cese", "nombre": "Cese 3 meses"},
    "y6": {"horizonte": 6, "poblacion": "pop_cese", "nombre": "Cese 6 meses"},
    "y_reingreso": {"horizonte": 6, "poblacion": "pop_reingreso", "nombre": "Reingreso"},
}

RUBROS_ESTACIONALES = ["A_AGRICULTURA", "G_COMERCIO", "P_EDUCACION", "I_TURISMO", "F_CONSTRUCCION"]


def _construir_dataset(tablas: dict[str, pd.DataFrame], periodo_fin: int) -> pd.DataFrame:
    corte_max = add_months(periodo_fin, -6)
    cortes = sorted(int(add_months(corte_max, -k)) for k in range(N_CORTES))
    filas = []
    for c in cortes:
        f = construir_features(tablas, c)
        filas.append(f)
    dataset = pd.concat(filas, ignore_index=True)
    return agregar_targets(dataset, tablas["panel_mensual"], periodo_fin), cortes


def _split_temporal(dataset: pd.DataFrame, cortes: list[int], horizonte: int) -> dict[str, pd.DataFrame]:
    """Split temporal con gap >= horizonte entre el ultimo corte de train/val y el primer
    corte de test (CONVENCIONES #3): test = ultimos N_TEST cortes; el resto, salvo el gap,
    se reparte en train/val (val = los ultimos FRAC_VAL cortes disponibles, para calibrar)."""
    cortes_ordenados = sorted(cortes)
    test_cortes = cortes_ordenados[-N_TEST:]
    test_min_idx = periodo_to_idx(min(test_cortes))
    candidatos = [c for c in cortes_ordenados if periodo_to_idx(c) <= test_min_idx - horizonte - 1]
    n_val = max(2, round(len(candidatos) * FRAC_VAL))
    train_cortes, val_cortes = candidatos[:-n_val], candidatos[-n_val:]
    return {
        "train": dataset[dataset["periodo_corte"].isin(train_cortes)],
        "val": dataset[dataset["periodo_corte"].isin(val_cortes)],
        "test": dataset[dataset["periodo_corte"].isin(test_cortes)],
        "cortes": {"train": train_cortes, "val": val_cortes, "test": test_cortes},
    }


def _regla_negocio(target: str, df: pd.DataFrame) -> np.ndarray:
    return predecir_regla_negocio_cese(df) if target != "y_reingreso" else predecir_regla_negocio_reingreso(df)


def _entrenar_y_evaluar(target: str, cfg: dict, dataset: pd.DataFrame, cortes: list[int],
                        salida: Path) -> dict:
    horiz_col = "horizonte3_completo" if cfg["horizonte"] == 3 else "horizonte6_completo"
    pob = dataset[(dataset[cfg["poblacion"]] == 1) & dataset[horiz_col]].copy()
    pob = agregar_tramos(pob)
    splits = _split_temporal(pob, cortes, cfg["horizonte"])
    train, val, test = splits["train"], splits["val"], splits["test"]

    modelo = ModeloCese.fit(train, val, target)
    modelo.save(str(salida / "modelos"))
    test = test.copy()
    test["pred_modelo"] = modelo.predict_proba(test)
    test["pred_modelo_crudo"] = modelo.predict_proba_cruda(test)

    tasas, tasa_global = baseline_tasa_base(pd.concat([train, val]), target)
    test["pred_base_tasa"] = predecir_tasa_base(tasas, tasa_global, test)
    test["pred_base_regla"] = _regla_negocio(target, test)

    metricas = {
        "modelo": evaluar(test, target, "pred_modelo"),
        "baseline_tasa_base": evaluar(test, target, "pred_base_tasa"),
        "baseline_regla_negocio": evaluar(test, target, "pred_base_regla"),
    }
    tabla_metricas = pd.DataFrame(metricas).T.reset_index().rename(columns={"index": "predictor"})

    segmentos = evaluar_por_segmento(test, target, "pred_modelo")
    calib = curva_calibracion(test, target, "pred_modelo")
    negocio_agregado = metrica_negocio_agregado(test, target, "pred_modelo")
    negocio_captura = metrica_negocio_captura(test, target, "pred_modelo", frac=0.10)
    importancias = modelo.importancias(20)

    n_por_split = {k: len(v) for k, v in splits.items() if k != "cortes"}
    return {
        "target": target, "nombre": cfg["nombre"], "tabla_metricas": tabla_metricas,
        "segmentos": segmentos, "calibracion": calib, "negocio_agregado": negocio_agregado,
        "negocio_captura": negocio_captura, "importancias": importancias, "n_por_split": n_por_split,
        "cortes_split": splits["cortes"], "test_df": test, "modelo": modelo,
    }


def _ejemplos_por_rubro(resultado_y3: dict, top: int = 5) -> pd.DataFrame:
    test, modelo = resultado_y3["test_df"], resultado_y3["modelo"]
    filas = []
    for rubro in RUBROS_ESTACIONALES:
        cand = test[test["rubro"] == rubro]
        if cand.empty:
            continue
        fila = cand.sort_values("pred_modelo", ascending=False).iloc[[0]]
        filas.append(fila)
        if len(filas) >= top:
            break
    if not filas:
        return pd.DataFrame()
    ejemplos = pd.concat(filas, ignore_index=True)
    from .model import _preparar_X   # uso interno: misma matriz que ve el modelo
    X = _preparar_X(ejemplos)
    contrib = modelo.clasificador.booster_.predict(X, pred_contrib=True)
    cols = modelo.columnas
    salida = []
    for i, (_, fila) in enumerate(ejemplos.iterrows()):
        c = pd.Series(contrib[i, :-1], index=cols).sort_values(key=np.abs, ascending=False)
        top3 = ", ".join(f"{f}={v:+.3f}" for f, v in c.head(3).items())
        salida.append({"afiliado_id": fila["afiliado_id"], "rubro": fila["rubro"],
                       "periodo_corte": fila["periodo_corte"], "pred_modelo": fila["pred_modelo"],
                       "y3_real": fila["y3"], "top3_contribuciones_shap": top3})
    return pd.DataFrame(salida)


def main(n_afiliados: int = 5000, periodo_inicio: int = 202201, periodo_fin: int = 202609,
        seed: int = 42, salida: str = "salidas/riesgo_cese_cotizacion") -> dict:
    t0 = time.time()
    salida_dir = Path(salida)
    (salida_dir / "reportes").mkdir(parents=True, exist_ok=True)
    (salida_dir / "modelos").mkdir(parents=True, exist_ok=True)

    print(f"Generando universo sintetico (n_afiliados={n_afiliados}, seed={seed})...")
    tablas = generar_datos(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                           periodo_fin=periodo_fin, seed=seed)
    print(f"  listo en {time.time() - t0:.1f}s")

    t1 = time.time()
    dataset, cortes = _construir_dataset(tablas, periodo_fin)
    print(f"Features construidas: {dataset.shape} en {time.time() - t1:.1f}s "
         f"({len(cortes)} cortes, {cortes[0]}..{cortes[-1]})")

    resultados = {}
    for target, cfg in TARGETS.items():
        t2 = time.time()
        r = _entrenar_y_evaluar(target, cfg, dataset, cortes, salida_dir)
        resultados[target] = r
        r["tabla_metricas"].to_csv(salida_dir / "reportes" / f"metricas_{target}.csv", index=False)
        for seg, tabla in r["segmentos"].items():
            tabla.to_csv(salida_dir / "reportes" / f"segmento_{seg}_{target}.csv", index=False)
        r["calibracion"].to_csv(salida_dir / "reportes" / f"calibracion_{target}.csv", index=False)
        r["negocio_agregado"].to_csv(salida_dir / "reportes" / f"negocio_agregado_{target}.csv", index=False)
        r["importancias"].to_csv(salida_dir / "reportes" / f"importancias_{target}.csv", index=False)
        print(f"\n=== {cfg['nombre']} ({target}) === filas train/val/test: {r['n_por_split']} "
             f"[{time.time() - t2:.1f}s]")
        print(r["tabla_metricas"].to_string(index=False))
        print("Negocio - agregado por corte (test):")
        print(r["negocio_agregado"].to_string(index=False))
        print(f"Negocio - top 10% de riesgo: {r['negocio_captura']}")

    ejemplos = _ejemplos_por_rubro(resultados["y3"])
    if not ejemplos.empty:
        ejemplos.to_csv(salida_dir / "reportes" / "ejemplos_interpretados.csv", index=False)
        print("\nEjemplos interpretados (cese 3 m, uno por rubro estacional):")
        print(ejemplos.to_string(index=False))

    dataset.to_parquet(salida_dir / "dataset_features.parquet", index=False)
    total = time.time() - t0
    print(f"\nRuntime total: {total:.1f}s. Salidas en {salida_dir.resolve()}")
    return resultados


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-afiliados", type=int, default=5000)
    p.add_argument("--periodo-inicio", type=int, default=202201)
    p.add_argument("--periodo-fin", type=int, default=202609)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salida", type=str, default="salidas/riesgo_cese_cotizacion")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(n_afiliados=args.n_afiliados, periodo_inicio=args.periodo_inicio, periodo_fin=args.periodo_fin,
        seed=args.seed, salida=args.salida)
