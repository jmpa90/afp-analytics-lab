"""Evaluacion del modelo de traspasos: metricas comunes, segmentos, calibracion y negocio.

Todas las metricas de clasificacion salen de `afp_lab.common.metrics` para que los resultados sean
comparables con los otros proyectos del laboratorio. Se agregan dos cosas propias del problema:

* **captura**: contactando al X % de mayor score, que porcentaje de los traspasos y del *saldo en
  riesgo* se alcanza. Es la metrica con la que el equipo comercial decide.
* **saldo en riesgo esperado**: sum(prob calibrada x saldo), que se compara con el saldo realmente
  perdido para verificar que la calibracion sirve para agregar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.common.metrics import metricas_clasificacion, tabla_metricas_por_segmento

SEGMENTOS = ["tramo_saldo", "tramo_edad", "tipo_afiliado", "fondo", "tramo_renta", "estado_laboral"]
FRACCIONES_CONTACTO = (0.05, 0.10, 0.20)


# --------------------------------------------------------------------- segmentos
def agregar_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los tramos de negocio usados para cortar las metricas."""
    out = df.copy()
    saldo = out["saldo_obligatorio"].to_numpy(dtype=float)
    out["tramo_saldo"] = pd.cut(saldo, [-1, 5e6, 15e6, 40e6, np.inf],
                                labels=["0-5M", "5-15M", "15-40M", "40M+"]).astype(str)
    edad = out["edad"].to_numpy()
    out["tramo_edad"] = pd.cut(edad, [0, 29, 39, 49, 59, 200],
                               labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    renta = out["rim_ref"].to_numpy(dtype=float)
    out["tramo_renta"] = pd.cut(np.nan_to_num(renta, nan=-1.0), [-2, 0, 6e5, 1.2e6, 2.5e6, np.inf],
                                labels=["sin_renta", "0-600k", "600k-1,2M", "1,2-2,5M", "2,5M+"]).astype(str)
    # a texto: `common.metrics.tabla_metricas_por_segmento` ya agrupa con observed=True (arreglado
    # por el gerente en la ronda 1), pero dejar los segmentos como texto mantiene estable el join de
    # `comparar_segmentos` y hace legibles los csv
    for c in ("tipo_afiliado", "fondo", "estado_laboral"):
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


# --------------------------------------------------------------------- tablas principales
def tabla_comparativa(df: pd.DataFrame, predicciones: dict[str, np.ndarray]) -> pd.DataFrame:
    """Metricas de `common.metrics.metricas_clasificacion` para el modelo y cada baseline."""
    y = df["y"].to_numpy()
    filas = []
    for nombre, p in predicciones.items():
        filas.append({"predictor": nombre, **metricas_clasificacion(y, np.asarray(p, dtype=float))})
    return pd.DataFrame(filas)


def metricas_por_segmento(df: pd.DataFrame, predicciones: dict[str, np.ndarray],
                          segmentos: list[str] | None = None, min_frac: float = 0.05) -> pd.DataFrame:
    """Metricas por segmento para cada predictor; marca los segmentos grandes (>= `min_frac`)."""
    segmentos = segmentos or SEGMENTOS
    filas = []
    for nombre, p in predicciones.items():
        d = df.assign(_p=np.asarray(p, dtype=float))
        for seg in segmentos:
            if seg not in d.columns:
                continue
            t = tabla_metricas_por_segmento(d, seg, metricas_clasificacion, "y", "_p")
            t = t.rename(columns={seg: "valor"})
            t.insert(0, "segmento", seg)
            t.insert(0, "predictor", nombre)
            filas.append(t)
    out = pd.concat(filas, ignore_index=True)
    out["frac_filas"] = out["n"] / len(df)
    out["segmento_grande"] = out["frac_filas"] >= min_frac
    return out


def comparar_segmentos(tabla_seg: pd.DataFrame, modelo: str, baseline: str,
                       metrica: str = "auc") -> pd.DataFrame:
    """Diferencia modelo - baseline en cada segmento (para el checklist: no perder en segmentos grandes)."""
    m = tabla_seg[tabla_seg["predictor"] == modelo].set_index(["segmento", "valor"])
    b = tabla_seg[tabla_seg["predictor"] == baseline].set_index(["segmento", "valor"])
    out = m[[metrica, "n", "tasa_base", "frac_filas", "segmento_grande"]].join(
        b[[metrica]], rsuffix="_baseline", how="inner").reset_index()
    out["dif"] = out[metrica] - out[f"{metrica}_baseline"]
    # un segmento sin positivos (o sin negativos) devuelve AUC NaN: no puede darse por ganado
    out["comparable"] = out[metrica].notna() & out[f"{metrica}_baseline"].notna()
    out["gana_modelo"] = out["comparable"] & (out["dif"] > 0)
    return out.sort_values(["segmento", "valor"]).reset_index(drop=True)


def segmentos_no_superados(tabla_comp: pd.DataFrame) -> pd.DataFrame:
    """Segmentos grandes donde el modelo NO supera al baseline, incluidos los no comparables."""
    g = tabla_comp[tabla_comp["segmento_grande"]]
    return g[~g["gana_modelo"]].reset_index(drop=True)


# --------------------------------------------------------------------- metrica de negocio
def captura_top(df: pd.DataFrame, predicciones: dict[str, np.ndarray],
                fracciones: tuple[float, ...] = FRACCIONES_CONTACTO,
                col_saldo: str = "saldo_obligatorio", seed: int = 7) -> pd.DataFrame:
    """% de traspasos y % de saldo en riesgo capturados contactando al X % de mayor score.

    Los empates (la regla de negocio binaria genera muchos) se rompen con un desempate aleatorio
    reproducible: es la forma honesta de evaluar un score con pocos valores distintos.
    """
    y = df["y"].to_numpy()
    saldo = df[col_saldo].to_numpy(dtype=float)
    saldo_riesgo_total = float(saldo[y == 1].sum())
    rng = np.random.default_rng(seed)
    desempate = rng.random(len(df))
    filas = []
    for nombre, p in predicciones.items():
        orden = np.lexsort((desempate, -np.asarray(p, dtype=float)))
        for frac in fracciones:
            k = max(1, int(round(frac * len(df))))
            sel = orden[:k]
            filas.append({
                "predictor": nombre,
                "frac_contactada": frac,
                "n_contactados": k,
                "traspasos_capturados": int(y[sel].sum()),
                "pct_traspasos_capturados": float(y[sel].sum() / max(y.sum(), 1)),
                "pct_saldo_riesgo_capturado": float(saldo[sel][y[sel] == 1].sum() / max(saldo_riesgo_total, 1.0)),
                "lift": float((y[sel].mean() / max(y.mean(), 1e-9))),
                "saldo_riesgo_capturado_mm": float(saldo[sel][y[sel] == 1].sum() / 1e6),
            })
    return pd.DataFrame(filas)


def saldo_en_riesgo(df: pd.DataFrame, prob: np.ndarray, col_saldo: str = "saldo_obligatorio") -> pd.DataFrame:
    """Saldo en riesgo esperado (sum p x saldo) contra el saldo efectivamente traspasado."""
    saldo = df[col_saldo].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    p = np.asarray(prob, dtype=float)
    esperado = float((p * saldo).sum())
    real = float(saldo[y == 1].sum())
    return pd.DataFrame([{
        "saldo_riesgo_esperado_mm": esperado / 1e6,
        "saldo_riesgo_real_mm": real / 1e6,
        "sesgo_relativo": (esperado - real) / max(real, 1.0),
        # descomposicion: cuanto del sesgo es error de NIVEL (numero de traspasos) y cuanto es la
        # correlacion entre la probabilidad y el saldo
        "sesgo_conteo": float(p.sum() - y.sum()) / max(float(y.sum()), 1.0),
        "saldo_total_cartera_mm": float(saldo.sum()) / 1e6,
    }])


# --------------------------------------------------------------------- calibracion
def saldo_en_riesgo_por_corte(df: pd.DataFrame, prob: np.ndarray,
                              col_saldo: str = "saldo_obligatorio") -> pd.DataFrame:
    """Saldo en riesgo esperado vs real en cada corte (el sesgo agregado de un solo numero esconde
    que la calibracion se mueve mes a mes)."""
    d = df.assign(_p=np.asarray(prob, dtype=float))
    filas = []
    for corte, g in d.groupby("periodo_corte", observed=True):
        r = saldo_en_riesgo(g, g["_p"].to_numpy(), col_saldo).iloc[0].to_dict()
        filas.append({"periodo_corte": int(corte), **r})
    return pd.DataFrame(filas)


def curva_calibracion(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de calibracion por decil de score (probabilidad media vs tasa observada)."""
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    orden = np.argsort(prob)
    grupos = np.array_split(orden, n_bins)
    filas = []
    for i, g in enumerate(grupos, start=1):
        filas.append({
            "decil": i,
            "n": len(g),
            "prob_media": float(prob[g].mean()),
            "tasa_observada": float(y[g].mean()),
            "dif": float(prob[g].mean() - y[g].mean()),
        })
    return pd.DataFrame(filas)


def resumen_por_corte(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """AUC y tasa base por corte de test (estabilidad temporal)."""
    d = df.assign(_p=np.asarray(prob, dtype=float))
    return tabla_metricas_por_segmento(d, "periodo_corte", metricas_clasificacion, "y", "_p")
