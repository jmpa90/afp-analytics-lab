"""Metricas de `riesgo_cese_cotizacion`, siempre con `afp_lab.common.metrics` para que
sean comparables con el resto del laboratorio, mas la metrica de negocio propia del brief.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.common.metrics import metricas_clasificacion, tabla_metricas_por_segmento

# CONVENCIONES #4 exige tipo de afiliado, tramo de renta y tramo de edad siempre; rubro y
# tipo de contrato son los segmentos propios de este proyecto.
SEGMENTOS_COMUNES = ["tipo_afiliado", "tramo_edad", "tramo_renta", "rubro", "tipo_contrato"]


def agregar_tramos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de tramo de edad y de renta (sobre `rim_l1`, la ultima RIM conocida)
    para reportar metricas por segmento (CONVENCIONES #4)."""
    # se castea a str (no categorico) para que groupby por segmento (afp_lab.common.metrics,
    # que no se modifica) no itere categorias vacias y falle al calcular metricas sobre 0 filas.
    out = df.copy()
    out["tramo_edad"] = pd.cut(out["edad"], [0, 24, 34, 44, 54, 64, 120],
                               labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
    out["tramo_renta"] = pd.cut(out["rim_l1"], [-1, 0, 400_000, 700_000, 1_100_000, np.inf],
                                labels=["sin_renta", "<=400k", "400-700k", "700k-1.1M", ">1.1M"]).astype(str)
    return out


def evaluar(df: pd.DataFrame, col_y: str, col_pred: str) -> dict[str, float]:
    return metricas_clasificacion(df[col_y].to_numpy(), df[col_pred].to_numpy())


def evaluar_por_segmento(df: pd.DataFrame, col_y: str, col_pred: str,
                         segmentos: list[str] = SEGMENTOS_COMUNES) -> dict[str, pd.DataFrame]:
    df = agregar_tramos(df) if "tramo_edad" not in df.columns else df
    salida = {}
    for seg in segmentos:
        t = tabla_metricas_por_segmento(df, seg, metricas_clasificacion, col_y, col_pred)
        t["n_filas_pct"] = 100 * t["n"] / t["n"].sum()
        salida[seg] = t.sort_values(seg)
    return salida


def curva_calibracion(df: pd.DataFrame, col_y: str, col_pred: str, n_deciles: int = 10) -> pd.DataFrame:
    """Curva de calibracion por decil de probabilidad predicha: predicho medio vs. tasa
    observada. Deciles por rango (`qcut` con `duplicates="drop"`, robusto a masas en 0)."""
    d = df[[col_y, col_pred]].copy()
    d["decil"] = pd.qcut(d[col_pred].rank(method="first"), n_deciles, labels=False, duplicates="drop")
    g = d.groupby("decil").agg(n=(col_y, "size"), pred_media=(col_pred, "mean"), obs_tasa=(col_y, "mean"))
    g["error_abs"] = (g["pred_media"] - g["obs_tasa"]).abs()
    return g.reset_index()


def metrica_negocio_agregado(df: pd.DataFrame, col_y: str, col_pred: str,
                             col_corte: str = "periodo_corte") -> pd.DataFrame:
    """Numero esperado (suma de probabilidades calibradas) vs. real de casos positivos, por
    mes de corte, con el error relativo (WAPE agregado a nivel de corte)."""
    g = df.groupby(col_corte).agg(n_esperado=(col_pred, "sum"), n_real=(col_y, "sum"), n_filas=(col_y, "size"))
    g["error_relativo"] = (g["n_esperado"] - g["n_real"]) / g["n_real"].clip(lower=1e-9)
    return g.reset_index()


def metrica_negocio_captura(df: pd.DataFrame, col_y: str, col_pred: str, frac: float = 0.10) -> dict[str, float]:
    """Contactando al `frac` (10 %) de mayor riesgo: que fraccion de los casos positivos
    reales se captura y cuantas contactos se necesitan por positivo capturado."""
    d = df.sort_values(col_pred, ascending=False)
    n_top = max(1, int(round(len(d) * frac)))
    top = d.iloc[:n_top]
    total_positivos = d[col_y].sum()
    capturados = top[col_y].sum()
    return {
        "frac_contactados": frac,
        "n_contactados": int(n_top),
        "positivos_totales": float(total_positivos),
        "positivos_capturados": float(capturados),
        "pct_capturado": float(capturados / total_positivos) if total_positivos > 0 else float("nan"),
    }
