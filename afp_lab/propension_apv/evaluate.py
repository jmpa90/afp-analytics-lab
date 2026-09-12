"""Evaluacion: metricas comparables (`afp_lab.common.metrics`), segmentos, calibracion y negocio.

Todo se calcula sobre el set de test (12 cortes mensuales), nunca sobre validacion ni
entrenamiento. Las metricas de clasificacion y regresion son las de `common.metrics` para que
los resultados sean comparables con los otros proyectos del laboratorio.

Ademas del agregado se reporta **por corte** (la tasa base es estacional: un test de 3 cortes
contiguos no dice nada) y el **sesgo del monto por decil** (un sesgo agregado chico puede
esconder deciles bajos subestimados y un decil alto sobreestimado que se compensan).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.common.metrics import (metricas_clasificacion, metricas_regresion,
                                    tabla_metricas_por_segmento)

FRACCIONES_CONTACTO = (0.10, 0.20)


# --------------------------------------------------------------------------- clasificacion
def tabla_clasificacion(df: pd.DataFrame, predicciones: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """AUC / AP / Brier / lift decil 1 / KS de cada predictor sobre `df`."""
    filas = []
    for nombre, pred in predicciones.items():
        m = metricas_clasificacion(df["y"].to_numpy(), pred["prob_aporte"].to_numpy())
        filas.append({"predictor": nombre, **m})
    return pd.DataFrame(filas)


def clasificacion_por_segmento(df: pd.DataFrame, prob: np.ndarray, segmento: str) -> pd.DataFrame:
    d = df[[segmento, "y"]].copy()
    d["prob"] = prob
    return tabla_metricas_por_segmento(d, segmento, metricas_clasificacion, "y", "prob")


def comparar_segmentos(df: pd.DataFrame, predicciones: dict[str, pd.DataFrame], segmento: str,
                       metrica: str = "auc", min_frac: float = 0.05) -> pd.DataFrame:
    """Compara los predictores en cada nivel de `segmento` y marca los segmentos grandes.

    CONVENCIONES §4: el modelo no puede perder en ningun segmento con >= 5 % de las filas.
    """
    partes = []
    for nombre, pred in predicciones.items():
        t = clasificacion_por_segmento(df, pred["prob_aporte"].to_numpy(), segmento)
        t["predictor"] = nombre
        partes.append(t)
    largo = pd.concat(partes, ignore_index=True)
    ancho = largo.pivot(index=segmento, columns="predictor", values=metrica)
    ancho["n"] = largo.groupby(segmento)["n"].first()
    ancho["frac_filas"] = ancho["n"] / len(df)
    ancho["segmento_grande"] = ancho["frac_filas"] >= min_frac
    return ancho.reset_index()


# --------------------------------------------------------------------------- monto
def tabla_monto(df: pd.DataFrame, predicciones: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dos vistas del monto.

    * `condicional_en_aportantes`: MAE/RMSE/WAPE/sesgo del `monto_condicional` sobre las filas
      con aporte real (es la calidad del regresor del hurdle).
    * `esperado_en_todos`: mismas metricas del `monto_esperado` sobre toda la poblacion
      (es lo que se agrega para proyectar el flujo de ahorro voluntario).
    """
    filas = []
    pos = df["y"] == 1
    for nombre, pred in predicciones.items():
        m = metricas_regresion(df.loc[pos, "monto"].to_numpy(), pred.loc[pos, "monto_condicional"].to_numpy())
        filas.append({"predictor": nombre, "vista": "condicional_en_aportantes", **m})
        m = metricas_regresion(df["monto"].to_numpy(), pred["monto_esperado"].to_numpy())
        filas.append({"predictor": nombre, "vista": "esperado_en_todos", **m})
    return pd.DataFrame(filas)


def sesgo_monto_por_decil(df: pd.DataFrame, pred: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Sesgo del `monto_esperado` por decil de monto esperado.

    El sesgo agregado de una variable con 95 % de ceros y cola larga se compensa entre deciles:
    esta tabla muestra si el modelo subestima a la mayoria y sobreestima al decil superior.
    """
    d = pd.DataFrame({"monto": df["monto"].to_numpy(),
                      "y": df["y"].to_numpy(),
                      "pred": pred["monto_esperado"].to_numpy()})
    d["decil"] = pd.qcut(d["pred"].rank(method="first"), n_bins, labels=False) + 1
    g = d.groupby("decil").agg(n=("pred", "size"), aportantes=("y", "sum"),
                               monto_real=("monto", "sum"), monto_predicho=("pred", "sum"))
    g["sesgo"] = g["monto_predicho"] / g["monto_real"].replace(0, np.nan) - 1.0
    g["share_monto_real"] = g["monto_real"] / max(d["monto"].sum(), 1e-9)
    return g.reset_index()


def metricas_por_corte(df: pd.DataFrame, pred: pd.DataFrame,
                       col_corte: str = "periodo_corte") -> pd.DataFrame:
    """Clasificacion y monto corte a corte (para ver la estacionalidad de la tasa base)."""
    filas = []
    for corte, g in df.groupby(col_corte, observed=True):
        p = pred.loc[g.index]
        m = metricas_clasificacion(g["y"].to_numpy(), p["prob_aporte"].to_numpy())
        pos = g["y"] == 1
        r = (metricas_regresion(g.loc[pos, "monto"].to_numpy(),
                                p.loc[pos, "monto_condicional"].to_numpy())
             if pos.any() else {"wape": np.nan})
        filas.append({"periodo_corte": int(corte), "n": m["n"], "tasa_base": m["tasa_base"],
                      "auc": m["auc"], "lift_decil_1": m["lift_decil_1"], "brier": m["brier"],
                      "wape_monto_aportantes": r["wape"],
                      "sesgo_monto_esperado": float(p["monto_esperado"].sum()
                                                    / max(g["monto"].sum(), 1e-9) - 1.0)})
    return pd.DataFrame(filas).sort_values("periodo_corte").reset_index(drop=True)


def monto_por_segmento(df: pd.DataFrame, pred: pd.DataFrame, segmento: str,
                       solo_aportantes: bool = True) -> pd.DataFrame:
    d = df[[segmento, "monto", "y"]].copy()
    d["pred"] = pred["monto_condicional"].to_numpy() if solo_aportantes else pred["monto_esperado"].to_numpy()
    if solo_aportantes:
        d = d[d["y"] == 1]
    return tabla_metricas_por_segmento(d, segmento, metricas_regresion, "monto", "pred")


# --------------------------------------------------------------------------- calibracion
def curva_calibracion(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Curva de calibracion por decil de probabilidad predicha."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(prob, float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), prob_media=("p", "mean"), tasa_real=("y", "mean"))
    g["diferencia"] = g["prob_media"] - g["tasa_real"]
    return g.reset_index()


def resumen_calibracion(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    c = curva_calibracion(y, prob, n_bins)
    ece = float((c["n"] / c["n"].sum() * c["diferencia"].abs()).sum())
    return {"brier": float(np.mean((np.asarray(prob) - np.asarray(y, float)) ** 2)),
            "ece": ece,
            "sesgo_agregado": float(np.sum(prob) / max(np.sum(y), 1e-9) - 1.0)}


# --------------------------------------------------------------------------- metrica de negocio
def metrica_negocio(df: pd.DataFrame, scores: dict[str, np.ndarray],
                    fracciones: tuple[float, ...] = FRACCIONES_CONTACTO,
                    col_corte: str = "periodo_corte") -> pd.DataFrame:
    """Que se captura contactando al X % mejor rankeado de cada corte.

    Para cada score y cada fraccion: porcentaje del **monto real** aportado y porcentaje de
    los **aportantes** que quedan dentro de la lista de contacto, y la tasa de aporte de los
    contactados. La seleccion se hace dentro de cada `periodo_corte` (asi se opera el mes a mes).
    """
    filas = []
    total_monto = df["monto"].sum()
    total_pos = df["y"].sum()
    for nombre, s in scores.items():
        d = df[[col_corte, "y", "monto"]].copy()
        d["score"] = np.asarray(s, float)
        d["rk"] = d.groupby(col_corte)["score"].rank(ascending=False, method="first")
        n_corte = d.groupby(col_corte)["score"].transform("size")
        for f in fracciones:
            sel = d["rk"] <= np.ceil(n_corte * f)
            filas.append({
                "score": nombre, "fraccion_contactada": f,
                "n_contactados": int(sel.sum()),
                "captura_monto": float(d.loc[sel, "monto"].sum() / max(total_monto, 1e-9)),
                "captura_aportantes": float(d.loc[sel, "y"].sum() / max(total_pos, 1e-9)),
                "tasa_aporte_contactados": float(d.loc[sel, "y"].mean()),
                "monto_medio_contactado": float(d.loc[sel, "monto"].mean()),
            })
    return pd.DataFrame(filas)


def contexto_negocio(df: pd.DataFrame, tablas: dict[str, pd.DataFrame],
                     horizonte: int = 6) -> pd.DataFrame:
    """Que fraccion del flujo total de ahorro voluntario es el que este modelo puede capturar.

    El target cubre solo la poblacion de **adquisicion / reactivacion** (sin aporte en los 6
    meses previos): el resto del flujo lo aportan los aportantes mensuales vigentes, que estan
    excluidos por definicion del problema. Sirve para no presentar la captura del modelo como si
    fuera sobre el flujo total de la AFP.
    """
    from afp_lab.rim_nowcasting.calendario import add_months

    meses = sorted({add_months(int(c), k) for c in df["periodo_corte"].unique()
                    for k in range(1, horizonte + 1)})
    ap = tablas["aportes_voluntarios"]
    flujo_total = float(ap[ap["periodo"].isin(meses)]["monto"].sum())
    # el monto del target se cuenta una vez por corte; para comparar con el flujo total se usa
    # el promedio por corte escalado a la cantidad de meses cubiertos
    monto_target = float(df["monto"].sum())
    n_cortes = df["periodo_corte"].nunique()
    return pd.DataFrame([{
        "meses_cubiertos": len(meses),
        "n_cortes_test": n_cortes,
        "flujo_total_aportes_en_esos_meses": flujo_total,
        "monto_target_sumado_sobre_cortes": monto_target,
        "monto_target_promedio_por_corte": monto_target / max(n_cortes, 1),
        "share_adquisicion_sobre_flujo_total": (monto_target / max(n_cortes, 1))
                                               / max(flujo_total / max(len(meses), 1) * horizonte, 1e-9),
    }])


def resumen_negocio(negocio: pd.DataFrame, score_modelo: str = "modelo_monto_esperado",
                    score_referencia: str = "renta") -> pd.DataFrame:
    """Diferencia de captura de monto entre el modelo y el ordenamiento por renta."""
    piv = negocio.pivot(index="fraccion_contactada", columns="score", values="captura_monto")
    out = piv[[score_modelo, score_referencia]].copy()
    out["ganancia_pp"] = (out[score_modelo] - out[score_referencia]) * 100
    out["lift_vs_renta"] = out[score_modelo] / out[score_referencia].replace(0, np.nan)
    return out.reset_index()


# --------------------------------------------------------------------------- aceptacion
def chequear_aceptacion(clasificacion: pd.DataFrame, monto: pd.DataFrame, negocio: pd.DataFrame,
                        nombre_modelo: str = "modelo") -> pd.DataFrame:
    """Criterios de aceptacion del brief, con el valor real obtenido."""
    fila = clasificacion.set_index("predictor").loc[nombre_modelo]
    wape = monto[(monto["predictor"] == nombre_modelo)
                 & (monto["vista"] == "condicional_en_aportantes")]["wape"].iloc[0]
    cap = negocio.set_index(["score", "fraccion_contactada"])["captura_monto"]
    cap_modelo = float(cap.loc[("modelo_monto_esperado", 0.10)])
    cap_renta = float(cap.loc[("renta", 0.10)])
    criterios = [
        ("AUC test >= 0,75", float(fila["auc"]), 0.75, float(fila["auc"]) >= 0.75),
        ("Lift decil 1 >= 3", float(fila["lift_decil_1"]), 3.0, float(fila["lift_decil_1"]) >= 3.0),
        ("WAPE monto en aportantes <= 0,6", float(wape), 0.6, float(wape) <= 0.6),
        ("Captura al 10 % > orden por renta", cap_modelo, cap_renta, cap_modelo > cap_renta),
    ]
    return pd.DataFrame(criterios, columns=["criterio", "valor", "umbral", "cumple"])
