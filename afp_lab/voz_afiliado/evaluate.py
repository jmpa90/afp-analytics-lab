"""Evaluacion del proyecto 11: NPS verdadero/ingenuo/inferido, AUC de detractor, calibracion,
drivers y valor del punto. Las metricas de clasificacion salen de `afp_lab.common.metrics`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.common.metrics import metricas_clasificacion, tabla_metricas_por_segmento

from . import model as M

SEGMENTOS = ["tramo_edad", "region", "tramo_empleador", "mes"]


def agregar_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los tramos de negocio pedidos por el brief: mes, edad, region, tramo de empleador."""
    out = df.copy()
    edad = out["edad"].to_numpy(dtype=float)
    out["tramo_edad"] = pd.cut(edad, [0, 29, 39, 49, 59, 200],
                               labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    out["tramo_empleador"] = out["tamano_empleador"].astype(str)
    out["region"] = out["region"].astype(str)
    return out


def _nps_de_categoria(cat: pd.Series) -> float:
    n = len(cat)
    if n == 0:
        return float("nan")
    return float((cat == "PROMOTOR").sum() / n * 100 - (cat == "DETRACTOR").sum() / n * 100)


def preparar_comparacion(tablas: dict[str, pd.DataFrame], df: pd.DataFrame,
                         scores: dict[str, np.ndarray] | None = None,
                         probs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> pd.DataFrame:
    """`df` = dataset (afiliado x periodo_corte) con `respondio`/`categoria` de la encuesta.

    `scores` = nombre -> score 0-10 (se bucketiza con la regla del generador: promotor/pasivo/
    detractor); pensado para versiones sin "encogimiento" de varianza (reglas de negocio, un
    baseline constante). `probs` = nombre -> `(p_promotor, p_detractor)` (dos clasificadores; el NPS
    se agrega como probabilidad esperada, ver `model.nps_de_probs`) -- la version correcta para un
    modelo de regresion/clasificacion entrenado con pocos respondientes (ver docstring de esa
    funcion). Devuelve `df` con `cat_verdadera` (toda la cartera, desde `satisfaccion_latente`),
    `cat_ingenua` (solo donde `respondio==1`, NaN en el resto), `cat_<nombre>` por cada `scores` y
    `_p_promotor_<nombre>` / `_p_detractor_<nombre>` por cada `probs`.
    """
    scores, probs = scores or {}, probs or {}
    lat = tablas["satisfaccion_latente"].rename(columns={"periodo": "periodo_corte"})
    out = df.merge(lat[["afiliado_id", "periodo_corte", "lat_satisfaccion"]],
                   on=["afiliado_id", "periodo_corte"], how="left")
    out["cat_verdadera"] = M.categoria_de_score(out["lat_satisfaccion"].to_numpy())
    out["cat_ingenua"] = np.where(out["respondio"].to_numpy() == 1, out["categoria"].astype(object), None)
    for nombre, score in scores.items():
        out[f"cat_{nombre}"] = M.categoria_de_score(np.asarray(score, dtype=float))
        out[f"score_{nombre}"] = np.asarray(score, dtype=float)
    for nombre, (p_prom, p_det) in probs.items():
        out[f"_p_promotor_{nombre}"] = np.asarray(p_prom, dtype=float)
        out[f"_p_detractor_{nombre}"] = np.asarray(p_det, dtype=float)
    return out


def nps_por(df_prep: pd.DataFrame, by: str, columnas: list[str] | None = None) -> pd.DataFrame:
    """NPS (verdadero/ingenuo/inferido...) agrupado por `by` (p.ej. `periodo_corte`, `tramo_edad`)."""
    cols_cat = [c for c in df_prep.columns if c.startswith("cat_")] if columnas is None else columnas
    nombres_blandos = sorted({c[len("_p_promotor_"):] for c in df_prep.columns if c.startswith("_p_promotor_")})
    filas = []
    for valor, g in df_prep.groupby(by, observed=True):
        fila = {by: valor, "n": int(len(g))}
        for c in cols_cat:
            sub = g[g[c].notna()] if c == "cat_ingenua" else g
            fila[c.replace("cat_", "nps_")] = _nps_de_categoria(sub[c])
            if c == "cat_ingenua":
                fila["n_respondientes"] = int(len(sub))
        for nombre in nombres_blandos:
            fila[f"nps_{nombre}"] = M.nps_de_probs(g[f"_p_promotor_{nombre}"].to_numpy(),
                                                    g[f"_p_detractor_{nombre}"].to_numpy())
        if "peso_ipw" in g.columns:
            sub = g[g["cat_ingenua"].notna()]
            if len(sub):
                w = sub["peso_ipw"].to_numpy(dtype=float)
                fila["nps_ponderado_ht"] = float(
                    100 * (np.average((sub["cat_ingenua"] == "PROMOTOR").to_numpy(dtype=float), weights=w)
                          - np.average((sub["cat_ingenua"] == "DETRACTOR").to_numpy(dtype=float), weights=w)))
            else:
                fila["nps_ponderado_ht"] = float("nan")
        filas.append(fila)
    return pd.DataFrame(filas).sort_values(by).reset_index(drop=True)


def tabla_errores(tabla_nps: pd.DataFrame, columna_verdadero: str = "nps_verdadera") -> pd.DataFrame:
    """Error absoluto de cada version de NPS contra el verdadero, y reduccion de error vs el ingenuo."""
    out = tabla_nps.copy()
    cols = [c for c in out.columns if c.startswith("nps_") and c != columna_verdadero]
    for c in cols:
        out[f"error_abs_{c}"] = (out[c] - out[columna_verdadero]).abs()
    return out


def resumen_error(tabla_nps: pd.DataFrame, columna_verdadero: str = "nps_verdadera") -> pd.DataFrame:
    """MAE (promedio sobre cortes/segmentos) de cada version de NPS y reduccion de error vs ingenuo."""
    e = tabla_errores(tabla_nps, columna_verdadero)
    cols_err = [c for c in e.columns if c.startswith("error_abs_")]
    mae = {c.replace("error_abs_nps_", ""): float(e[c].mean()) for c in cols_err}
    base = mae.get("ingenua", float("nan"))
    filas = [{"version": k, "mae_puntos_nps": v,
             "reduccion_error_vs_ingenuo": float(1 - v / base) if base else float("nan")}
            for k, v in mae.items()]
    return pd.DataFrame(filas).sort_values("mae_puntos_nps").reset_index(drop=True)


# --------------------------------------------------------------------- AUC de detractor y calibracion
def metricas_detractor(df_resp: pd.DataFrame, prob: np.ndarray) -> dict:
    return metricas_clasificacion(df_resp["es_detractor"].to_numpy(), np.asarray(prob, dtype=float))


def curva_calibracion(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    orden = np.argsort(prob)
    grupos = np.array_split(orden, min(n_bins, max(2, len(y) // 20)))
    filas = []
    for i, g in enumerate(grupos, start=1):
        filas.append({"decil": i, "n": len(g), "prob_media": float(prob[g].mean()),
                     "tasa_observada": float(y[g].mean()), "dif": float(prob[g].mean() - y[g].mean())})
    return pd.DataFrame(filas)
