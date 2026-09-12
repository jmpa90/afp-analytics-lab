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
        # sin masa real en el grupo el WAPE y el sesgo relativo no estan definidos (NaN, no 1e17)
        "wape": float(np.sum(np.abs(e)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else float("nan"),
        "sesgo_agregado": float((pred.sum() - y.sum()) / y.sum()) if y.sum() > 0 else float("nan"),
    }


def metricas_clasificacion(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y, prob = np.asarray(y, int), np.asarray(prob, float)
    base = y.mean() if y.mean() > 0 else 1e-9
    # lift del decil 1 con empates tratados por su valor esperado: el grupo de score que cruza el
    # borde del decil aporta su tasa media ponderada por la fraccion incluida (un score constante da
    # exactamente 1,0; una regla binaria da la tasa del grupo marcado)
    k = max(1, len(y) // 10)
    grupos = pd.DataFrame({"p": prob, "y": y}).groupby("p")["y"].agg(["sum", "count"]).sort_index(ascending=False)
    restante, positivos = k, 0.0
    for suma, cuenta in zip(grupos["sum"], grupos["count"]):
        toma = min(restante, cuenta)
        positivos += suma * toma / cuenta
        restante -= toma
        if restante <= 0:
            break
    lift = (positivos / k) / base
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
        "lift_decil_1": float(lift),
        "ks": float(np.max(np.abs(pos.to_numpy() - neg.to_numpy()))),
    }


def tabla_metricas_por_segmento(df: pd.DataFrame, col_segmento: str, fn, col_y: str, col_pred: str) -> pd.DataFrame:
    filas = []
    # observed=True: con columnas category no se generan filas vacias (reportado por DS-1 y DS-2)
    for seg, g in df.groupby(col_segmento, observed=True):
        if len(g) == 0:
            continue
        filas.append({col_segmento: seg, **fn(g[col_y].to_numpy(), g[col_pred].to_numpy())})
    return pd.DataFrame(filas)
