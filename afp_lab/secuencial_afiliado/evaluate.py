"""Metricas del proyecto 08: `afp_lab.common.metrics` (AUC, PR-AUC=average_precision, lift
decil 1, Brier, KS) por cabeza y por modelo, con intervalos bootstrap sobre el test, y la tabla
comparativa red vs baselines que pide el BRIEF (con la conclusion "vale la pena": mejora >= 0,01
AUC en >= 2 cabezas, o igual AUC con menos ingenieria de features)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.common.metrics import metricas_clasificacion

UMBRAL_MEJORA_AUC = 0.01


def evaluar_modelo(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return metricas_clasificacion(y, pred)


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metrica: str = "auc", n_boot: int = 300,
                 seed: int = 42, alfa: float = 0.05) -> tuple[float, float, float]:
    """IC bootstrap (percentil) de `metrica` (una clave de `metricas_clasificacion`) sobre el
    test, remuestreando FILAS con reposicion. Devuelve `(punto, lo, hi)`."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    punto = metricas_clasificacion(y, pred)[metrica]
    valores = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if yi.min() == yi.max():   # remuestreo sin ambas clases: metrica no definida, se omite
            valores[b] = np.nan
            continue
        valores[b] = metricas_clasificacion(yi, pred[idx])[metrica]
    valores = valores[~np.isnan(valores)]
    lo, hi = np.percentile(valores, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return float(punto), float(lo), float(hi)


def tabla_cabeza(y_test: np.ndarray, predicciones: dict[str, np.ndarray], n_boot: int = 300,
                 seed: int = 42) -> pd.DataFrame:
    """Una fila por modelo (`predicciones`: nombre_modelo -> score en test), con AUC/PR-AUC/
    lift decil 1 puntuales + IC bootstrap del AUC."""
    filas = []
    for nombre, pred in predicciones.items():
        m = evaluar_modelo(y_test, pred)
        auc, lo, hi = bootstrap_ci(y_test, pred, "auc", n_boot=n_boot, seed=seed)
        filas.append({"modelo": nombre, "n": m["n"], "tasa_base": m["tasa_base"], "auc": m["auc"],
                     "auc_ic_lo": lo, "auc_ic_hi": hi, "pr_auc": m["average_precision"],
                     "lift_decil_1": m["lift_decil_1"], "brier": m["brier"], "ks": m["ks"]})
    return pd.DataFrame(filas)


def conclusion_vale_la_pena(tablas_por_cabeza: dict[str, pd.DataFrame],
                            modelo_red: str = "red_multitarea",
                            modelo_referencia: str = "lightgbm_produccion") -> dict:
    """Criterio del BRIEF: "vale la pena" si la red mejora >= 0,01 AUC frente al LightGBM de
    produccion en >= 2 cabezas, o si iguala el AUC (dentro de +/-0,01) con menos ingenieria de
    features (la secuencia no usa las tablas operacionales de 02: contactos/reclamos/saldos/
    cambios_fondo/uso_digital, solo el panel comun + estaticas)."""
    deltas = {}
    for cabeza, tabla in tablas_por_cabeza.items():
        t = tabla.set_index("modelo")
        if modelo_red not in t.index or modelo_referencia not in t.index:
            continue
        deltas[cabeza] = float(t.loc[modelo_red, "auc"] - t.loc[modelo_referencia, "auc"])
    n_mejora = sum(1 for d in deltas.values() if d >= UMBRAL_MEJORA_AUC)
    n_iguala = sum(1 for d in deltas.values() if abs(d) < UMBRAL_MEJORA_AUC)
    if n_mejora >= 2:
        veredicto = "reemplaza"
    elif n_mejora >= 1 or n_iguala == len(deltas) and len(deltas) > 0:
        veredicto = "complementa"
    else:
        veredicto = "no_aporta"
    return {"deltas_auc": deltas, "n_cabezas_mejora_0_01": n_mejora, "n_cabezas_iguala": n_iguala,
           "veredicto": veredicto}
