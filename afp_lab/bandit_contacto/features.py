"""Contexto as-of para el bandit: reutiliza `churn_traspaso.features` y agrega dos cosas propias.

`construir_contexto` delega ENTERAMENTE en `afp_lab.churn_traspaso.features.construir_features_multi`
(mismas reglas as-of, mismo desfase de RIM, mismo recorte por corte: CONVENCIONES §3) para no
duplicar ni arriesgar inconsistencias con 02. Lo propio de este proyecto son dos derivados que las
politicas SI pueden ver (no son `lat_`/`p_`, son reglas/funciones de negocio sobre columnas as-of):

* `score_riesgo`: heuristica de negocio (no el generador latente) usada para (a) la politica de
  logging heuristica y (b) para acotar la poblacion "en riesgo" de cada mes al presupuesto mensual
  de contactos.
* `CONTEXTO_NUMERICO` + `ajustar_escalador`/`transformar_contexto`: subconjunto numerico
  estandarizado (z-score) que usan LinUCB y Thompson lineal (necesitan un vector real, no columnas
  categoricas). El escalador se ajusta UNA vez sobre todo el contexto de la simulacion (no por
  particion train/valid/test: aqui no hay un target que fugue, es solo una transformacion de
  covariables para estabilizar la inversion de matrices; se documenta como decision en el README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso.features import (columnas_features, construir_features_multi,
                                              cortes_evaluables, recortar_tablas)

CONTEXTO_NUMERICO = [
    "rim_l1", "rim_media_12", "rim_cv_12", "densidad_12", "meses_desde_ult_cotiza",
    "log_saldo", "antiguedad_afp_meses", "reclamos_6m", "reclamos_abiertos_12m",
    "contactos_competencia_interesado_6m", "meses_desde_contacto_competencia",
    "brecha_comision_rel", "brecha_rentabilidad_pp", "edad", "meses_hasta_pension", "logins_3m",
]


def construir_contexto(tablas: dict[str, pd.DataFrame], cortes: list[int]) -> pd.DataFrame:
    """Contexto as-of (afiliado x periodo_corte) para los `cortes` de la simulacion.

    Igual que `churn_traspaso.construir_dataset` pero SIN target (aqui la "etiqueta" la produce el
    simulador de brazos, no un traspaso ya ocurrido). Agrega `score_riesgo` y RIM de referencia para
    la comision esperada.
    """
    df = construir_features_multi(tablas, cortes)
    df["score_riesgo"] = calcular_score_riesgo(df)
    return df


def calcular_score_riesgo(df: pd.DataFrame) -> pd.Series:
    """Heuristica de riesgo de negocio (NO el generador latente): mayor = mas prioritario contactar.

    Combina brecha de comision con la competencia, reclamos y senales de interes en la competencia,
    penalizando alta densidad de cotizacion reciente (afiliado "tranquilo"). Deliberadamente
    imperfecta (pesos y variables distintos de `synthetic.calcular_respuesta_latente`): una AFP real
    tampoco conoce la propension verdadera, solo reglas de negocio razonables.
    """
    brecha = df["brecha_comision_rel"].fillna(0.0).clip(-1, 3)
    reclamos = np.log1p(df["reclamos_6m"].fillna(0.0))
    interes_comp = np.log1p(df["contactos_competencia_interesado_6m"].fillna(0.0))
    dens = df["densidad_12"].fillna(0.0)
    meses_comp = df["meses_desde_contacto_competencia"].fillna(99.0)
    antig = np.log1p(df["antiguedad_afp_meses"].fillna(0.0).clip(lower=0))
    return (0.9 * brecha + 0.5 * reclamos + 0.6 * interes_comp - 0.3 * dens
           + 0.6 / (1.0 + meses_comp) - 0.05 * antig)


def ajustar_escalador(df: pd.DataFrame, columnas: list[str] = CONTEXTO_NUMERICO
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviacion estandar (con piso 1e-6) de `columnas`, para estandarizar el contexto."""
    x = df[columnas].fillna(0.0).to_numpy(dtype=np.float64)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu, sigma


def transformar_contexto(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray,
                         columnas: list[str] = CONTEXTO_NUMERICO) -> np.ndarray:
    """Matriz (n, len(columnas)+1) estandarizada con columna de sesgo (bias) al final."""
    x = df[columnas].fillna(0.0).to_numpy(dtype=np.float64)
    z = (x - mu) / sigma
    return np.hstack([z, np.ones((len(df), 1))])


__all__ = ["construir_contexto", "calcular_score_riesgo", "ajustar_escalador",
          "transformar_contexto", "CONTEXTO_NUMERICO", "cortes_evaluables", "recortar_tablas",
          "columnas_features"]
