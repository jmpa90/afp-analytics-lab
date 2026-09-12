"""Metricas compartidas por los proyectos (para que los resultados sean comparables).

Regresion: MAE, RMSE, WAPE (= sum|e| / sum|y|, robusta a ceros), sesgo agregado.
Clasificacion: AUC, average precision, Brier, lift en el decil superior, KS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def metricas_regresion(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    e = pred - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "wape": float(np.sum(np.abs(e)) / max(np.sum(np.abs(y)), 1e-9)),
        "sesgo_agregado": float((pred.sum() - y.sum()) / max(y.sum(), 1e-9)),
    }


def metricas_clasificacion(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y, prob = np.asarray(y, int), np.asarray(prob, float)
    orden = np.argsort(-prob)
    top = orden[: max(1, len(y) // 10)]
    base = y.mean() if y.mean() > 0 else 1e-9
    # KS: distancia maxima entre las distribuciones acumuladas de positivos y negativos
    df = pd.DataFrame({"y": y, "p": prob}).sort_values("p", ascending=False)
    pos = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    neg = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return {
        "n": int(len(y)),
        "tasa_base": float(y.mean()),
        "auc": float(roc_auc_score(y, prob)) if 0 < y.mean() < 1 else float("nan"),
        "average_precision": float(average_precision_score(y, prob)) if 0 < y.mean() < 1 else float("nan"),
        "brier": float(brier_score_loss(y, prob)),
        "lift_decil_1": float(y[top].mean() / base),
        "ks": float(np.max(np.abs(pos.to_numpy() - neg.to_numpy()))),
    }


def tabla_metricas_por_segmento(df: pd.DataFrame, col_segmento: str, fn, col_y: str, col_pred: str) -> pd.DataFrame:
    filas = []
    for seg, g in df.groupby(col_segmento):
        filas.append({col_segmento: seg, **fn(g[col_y].to_numpy(), g[col_pred].to_numpy())})
    return pd.DataFrame(filas)
