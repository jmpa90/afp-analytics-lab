"""Evaluacion del proyecto 05: metricas comparables, segmentos, calibracion y negocio.

Todo sobre el pool de test (12 cortes mensuales), nunca sobre validacion ni entrenamiento. Las
metricas de clasificacion son las de `afp_lab.common.metrics`.

Las dos metricas de negocio del brief:

1. **Masa imponible en riesgo capturada** gestionando al 10 % y al 20 % de los empleadores del mes,
   ordenados por `masa_en_riesgo_esperada`. Es lo que decide a quien llama cobranza.
2. **Error del agregado mensual**: masa en DNP esperada vs real mes a mes (objetivo: <= 10 % en al
   menos 10 de los 12 meses).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.common.metrics import (metricas_clasificacion, metricas_regresion,
                                    tabla_metricas_por_segmento)

FRACCIONES_GESTION = (0.10, 0.20)
TOLERANCIA_AGREGADO = 0.10


# --------------------------------------------------------------------------- clasificacion
def tabla_clasificacion(df: pd.DataFrame, predicciones: dict[str, pd.DataFrame],
                        target: str = "y_dnp") -> pd.DataFrame:
    filas = []
    for nombre, pred in predicciones.items():
        m = metricas_clasificacion(df[target].to_numpy(), pred[f"prob_{target}"].to_numpy())
        filas.append({"predictor": nombre, "target": target, **m})
    return pd.DataFrame(filas)


def tabla_clasificacion_todos_los_targets(df: pd.DataFrame, pred: pd.DataFrame,
                                          targets=("y_dnp", "y_tardio", "y_dnp_3m")) -> pd.DataFrame:
    filas = []
    for t in targets:
        filas.append({"target": t, **metricas_clasificacion(df[t].to_numpy(),
                                                            pred[f"prob_{t}"].to_numpy())})
    return pd.DataFrame(filas)


def metricas_por_corte(df: pd.DataFrame, pred: pd.DataFrame, target: str = "y_dnp",
                       col_corte: str = "periodo_corte") -> pd.DataFrame:
    filas = []
    for corte, g in df.groupby(col_corte, observed=True):
        p = pred.loc[g.index]
        m = metricas_clasificacion(g[target].to_numpy(), p[f"prob_{target}"].to_numpy())
        real = float(g["masa_en_riesgo"].sum())
        esperada = float(p["masa_en_riesgo_esperada"].sum())
        # volatilidad intrinseca del agregado: incluso con la probabilidad EXACTA, la masa en
        # riesgo realizada de un mes es una suma de Bernoulli ponderada por masas muy desiguales
        # (unos pocos empleadores grandes dominan). Este es el piso de error inevitable.
        prob = p[f"prob_{target}"].to_numpy()
        masa = g["masa_ref_clp"].to_numpy()
        sigma = float(np.sqrt(np.sum(prob * (1 - prob) * masa ** 2)))
        filas.append({"periodo_corte": int(corte), "n": m["n"], "tasa_base": m["tasa_base"],
                      "auc": m["auc"], "lift_decil_1": m["lift_decil_1"], "brier": m["brier"],
                      "masa_en_riesgo_real": real, "masa_en_riesgo_esperada": esperada,
                      "error_relativo": esperada / max(real, 1e-9) - 1.0,
                      "volatilidad_intrinseca_pct": sigma / max(esperada, 1e-9),
                      "dentro_de_1_sigma": abs(esperada - real) <= sigma})
    return pd.DataFrame(filas).sort_values("periodo_corte").reset_index(drop=True)


def comparar_segmentos(df: pd.DataFrame, predicciones: dict[str, pd.DataFrame], segmento: str,
                       target: str = "y_dnp", metrica: str = "auc",
                       min_frac: float = 0.05) -> pd.DataFrame:
    partes = []
    for nombre, pred in predicciones.items():
        d = df[[segmento, target]].copy()
        d["prob"] = pred[f"prob_{target}"].to_numpy()
        t = tabla_metricas_por_segmento(d, segmento, metricas_clasificacion, target, "prob")
        t["predictor"] = nombre
        partes.append(t)
    largo = pd.concat(partes, ignore_index=True)
    ancho = largo.pivot(index=segmento, columns="predictor", values=metrica)
    ancho["n"] = largo.groupby(segmento)["n"].first()
    ancho["positivos"] = df.groupby(segmento, observed=True)[target].sum()
    ancho["frac_filas"] = ancho["n"] / len(df)
    ancho["segmento_grande"] = ancho["frac_filas"] >= min_frac
    ancho["nota"] = np.where(ancho["positivos"] == 0, "sin casos de DNP",
                             np.where(ancho["positivos"] == ancho["n"], "todos DNP", ""))
    return ancho.reset_index()


# --------------------------------------------------------------------------- calibracion
def curva_calibracion(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(prob, float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), prob_media=("p", "mean"), tasa_real=("y", "mean"))
    g["diferencia"] = g["prob_media"] - g["tasa_real"]
    return g.reset_index()


def resumen_calibracion(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    c = curva_calibracion(y, prob, n_bins)
    return {"brier": float(np.mean((np.asarray(prob) - np.asarray(y, float)) ** 2)),
            "ece": float((c["n"] / c["n"].sum() * c["diferencia"].abs()).sum()),
            "sesgo_agregado": float(np.sum(prob) / max(np.sum(y), 1e-9) - 1.0)}


# --------------------------------------------------------------------------- negocio
def metrica_negocio(df: pd.DataFrame, scores: dict[str, np.ndarray],
                    fracciones: tuple[float, ...] = FRACCIONES_GESTION,
                    col_corte: str = "periodo_corte") -> pd.DataFrame:
    """Que masa imponible en riesgo se captura gestionando al X % de los empleadores del mes.

    La seleccion se hace **dentro de cada corte** (es la lista mensual de cobranza).
    """
    filas = []
    total_masa = df["masa_en_riesgo"].sum()
    total_pos = df["y_dnp"].sum()
    for nombre, s in scores.items():
        d = df[[col_corte, "y_dnp", "masa_en_riesgo"]].copy()
        d["score"] = np.asarray(s, float)
        d["rk"] = d.groupby(col_corte)["score"].rank(ascending=False, method="first")
        n_corte = d.groupby(col_corte)["score"].transform("size")
        for f in fracciones:
            sel = d["rk"] <= np.ceil(n_corte * f)
            filas.append({
                "score": nombre, "fraccion_gestionada": f, "n_gestionados": int(sel.sum()),
                "captura_masa_en_riesgo": float(d.loc[sel, "masa_en_riesgo"].sum()
                                                / max(total_masa, 1e-9)),
                "captura_empleadores_dnp": float(d.loc[sel, "y_dnp"].sum() / max(total_pos, 1e-9)),
                "precision_gestionados": float(d.loc[sel, "y_dnp"].mean()),
                "masa_media_gestionada": float(d.loc[sel, "masa_en_riesgo"].mean()),
            })
    return pd.DataFrame(filas)


def resumen_negocio(negocio: pd.DataFrame, score_modelo: str = "modelo_masa_esperada",
                    score_referencia: str = "regla_negocio") -> pd.DataFrame:
    piv = negocio.pivot(index="fraccion_gestionada", columns="score",
                        values="captura_masa_en_riesgo")
    out = piv[[score_modelo, score_referencia]].copy()
    out["ganancia_pp"] = (out[score_modelo] - out[score_referencia]) * 100
    out["lift_vs_regla"] = out[score_modelo] / out[score_referencia].replace(0, np.nan)
    return out.reset_index()


def error_agregado_mensual(por_corte: pd.DataFrame,
                           tolerancia: float = TOLERANCIA_AGREGADO) -> pd.DataFrame:
    """Resumen del error del agregado mensual de masa en DNP (criterio de aceptacion del brief)."""
    err = por_corte["error_relativo"].abs()
    return pd.DataFrame([{
        "n_meses": len(por_corte),
        "meses_dentro_de_tolerancia": int((err <= tolerancia).sum()),
        "meses_dentro_de_1_sigma_intrinseca": int(por_corte["dentro_de_1_sigma"].sum()),
        "volatilidad_intrinseca_media_pct": float(por_corte["volatilidad_intrinseca_pct"].mean()),
        "tolerancia": tolerancia,
        "error_absoluto_medio": float(err.mean()),
        "error_mediano": float(err.median()),
        "peor_mes": int(por_corte.loc[err.idxmax(), "periodo_corte"]),
        "peor_error": float(err.max()),
        "sesgo_agregado_total": float(por_corte["masa_en_riesgo_esperada"].sum()
                                     / max(por_corte["masa_en_riesgo_real"].sum(), 1e-9) - 1.0),
    }])


def masa_por_segmento(df: pd.DataFrame, pred: pd.DataFrame, segmento: str) -> pd.DataFrame:
    """Masa en riesgo real vs esperada por segmento (regresion de `common.metrics`)."""
    d = df[[segmento, "masa_en_riesgo"]].copy()
    d["pred"] = pred["masa_en_riesgo_esperada"].to_numpy()
    return tabla_metricas_por_segmento(d, segmento, metricas_regresion, "masa_en_riesgo", "pred")


# --------------------------------------------------------------------------- aceptacion
def chequear_aceptacion(clasificacion: pd.DataFrame, negocio: pd.DataFrame,
                        por_corte: pd.DataFrame, agregado: pd.DataFrame,
                        nombre_modelo: str = "modelo") -> pd.DataFrame:
    """Criterios de aceptacion del brief 05, con el valor agregado y el rango por corte."""
    fila = clasificacion.set_index("predictor").loc[nombre_modelo]
    cap = negocio.set_index(["score", "fraccion_gestionada"])["captura_masa_en_riesgo"]
    cap_modelo = float(cap.loc[("modelo_masa_esperada", 0.10)])
    cap_regla = float(cap.loc[("regla_negocio", 0.10)])
    dentro = int(agregado["meses_dentro_de_tolerancia"].iloc[0])
    n_meses = int(agregado["n_meses"].iloc[0])
    criterios = [
        ("AUC test >= 0,80", float(fila["auc"]), 0.80, float(fila["auc"]) >= 0.80,
         f"{por_corte['auc'].min():.3f} a {por_corte['auc'].max():.3f}"),
        ("Captura de masa en riesgo al 10 % >= 50 %", cap_modelo, 0.50, cap_modelo >= 0.50, ""),
        ("Captura al 10 % mejor que la regla", cap_modelo, cap_regla, cap_modelo > cap_regla, ""),
        (f"Error del agregado mensual <= 10 % en >= 10 de {n_meses} meses",
         float(dentro), 10.0, dentro >= 10,
         f"error |medio| {float(agregado['error_absoluto_medio'].iloc[0]):.3f}"),
    ]
    return pd.DataFrame(criterios, columns=["criterio", "valor_agregado", "umbral", "cumple",
                                            "rango_por_corte"])
