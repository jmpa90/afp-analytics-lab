"""Campana historica de contacto (confounding real) + holdout aleatorio + efecto causal `lat_cate`.

Reutiliza SIN modificar el universo y el target de `afp_lab.churn_traspaso` (mismas tablas, mismo
`construir_features_multi`, mismo `construir_target`). Este modulo agrega, por encima, una capa
propia de 07: quien fue CONTACTADO (`T`) por una campana de retencion, y el efecto CAUSAL verdadero
de ese contacto sobre el traspaso (`lat_cate`), que en el mundo real nadie observa.

Mecanismo (una fila por afiliado x periodo_corte, sobre las features as-of de 02):

1. **Asignacion de tratamiento `T`** (`asignar_tratamiento`): el 10 % de la cartera (`p_holdout`,
   fijo por afiliado en el tiempo) queda en un **holdout experimental** con asignacion ALEATORIA
   (`p_tratamiento_holdout`, sin relacion con `x`): sirve para medir Qini/AUUC sin sesgo. El 90 %
   restante recibe la **politica historica real**: un ejecutivo contacta mas a quien tiene saldo
   alto, es mayor (cartera premium) y tiene un score heuristico de riesgo alto (`riesgo_latente_heuristico`)
   -- exactamente el tipo de confounding que un uplift model tiene que corregir (correlaciona con el
   riesgo, no con el efecto).
2. **Efecto causal heterogeneo `lat_cate`** (`construir_efecto_causal`): tres segmentos, definidos
   SOLO con features as-of (mas una sensibilidad a comision latente, `lat_sensibilidad_comision`,
   analoga a `lat_switcher` de 02: no observable, solo genera la dinamica):
     * **persuadibles** (reclamo reciente RESUELTO, saldo medio, uso digital medio): el contacto
       REDUCE la probabilidad de traspaso.
     * **sleeping dogs** (uso digital bajo y sensibilidad a comision alta, top 30 %): el contacto la
       AUMENTA (el ejecutivo "despierta" a alguien que no estaba pensando en irse).
     * el resto (mayoria): efecto cero.
   La magnitud escala con `riesgo_latente_heuristico(x)` (mismo score que en la propension: mas
   riesgo => mas para ganar reteniendo a un persuadible; menos riesgo => mas sorpresa si se despierta
   un sleeping dog).
3. **Alteracion del target** (`aplicar_efecto_causal`): el target de CONTROL (`y0`, sale de
   `churn_traspaso.construir_target`, IDENTICO al de 02) se altera solo en las filas TRATADAS: un
   persuadible que iba a irse (`y0=1`) se retiene con probabilidad `p_retencion_persuadible`; un
   sleeping dog que iba a quedarse (`y0=0`) se traspasa con probabilidad `p_despierta_sleeping_dog`.
   Es un flip probabilistico con randomizacion PROPIA (no determinista), lo que equivale a desplazar
   la probabilidad del evento en la magnitud del efecto sin tocar `traspasos` ni ninguna tabla de 02:
   para las filas de CONTROL, `y` queda exactamente igual al target de 02 (coherencia garantizada).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParametrosCampana:
    """Parametros de la campana historica y del efecto causal. Todos documentados en el README."""

    # --- holdout experimental (10 % de la cartera, brief)
    p_holdout: float = 0.10
    p_tratamiento_holdout: float = 0.50

    # --- propension historica confundida (fuera del holdout): sigmoid(z)
    b0_propension: float = -1.6
    b_saldo_percentil: float = 1.3      # percentil de saldo DENTRO del corte (cartera premium primero)
    b_tramo_edad_mayor: float = 0.35    # +edad_umbral_mayor anos: ejecutivo prioriza afiliados mayores
    b_score_riesgo: float = 2.0         # el ejecutivo tambien reacciona al riesgo (confounding real)
    edad_umbral_mayor: int = 50

    # --- score heuristico de riesgo (propio de 07, NO es el hazard real de 02: transparente y simple,
    # solo para fijar la propension y escalar la magnitud de `lat_cate`)
    intercepto_riesgo: float = -1.1
    w_reclamo_6m: float = 1.1
    w_contacto_competencia_3m: float = 1.3
    w_brecha_rentabilidad_mala: float = 0.9
    w_brecha_comision_alta: float = 0.5
    w_antiguedad_baja: float = 0.6
    umbral_brecha_rentabilidad: float = -0.5
    umbral_brecha_comision: float = 0.5
    umbral_antiguedad_meses: int = 24

    # --- efecto causal heterogeneo (segmentos)
    p_retencion_persuadible: float = 0.40     # prob. de retener a un persuadible tratado (y0=1->y=0)
    p_despierta_sleeping_dog: float = 0.12    # prob. de despertar a un sleeping dog tratado (y0=0->y=1)
    frac_sensibilidad_comision_alta: float = 0.30   # top 30 % de lat_sensibilidad_comision
    logins_3m_bajo: float = 1.0               # <= este valor: "uso digital bajo" (sleeping dog)
    logins_3m_medio_lo: float = 2.0           # rango [lo, hi): "uso digital medio" (persuadible)
    logins_3m_medio_hi: float = 10.0
    tramo_saldo_medio: str = "5-15M"          # mismo bin que churn_traspaso.evaluate.agregar_segmentos


# --------------------------------------------------------------------- score heuristico de riesgo
def riesgo_latente_heuristico(df: pd.DataFrame, par: ParametrosCampana | None = None) -> np.ndarray:
    """Score de riesgo en [0,1], simple y transparente (NO es el hazard de 02).

    Combina cuatro senales as-of ya presentes en `construir_features_multi` (reclamo reciente,
    visita de la competencia, mala rentabilidad relativa, comision cara y poca antiguedad). Sirve
    para dos cosas: (a) alimentar la propension historica confundida, y (b) escalar la magnitud de
    `lat_cate` (mas riesgo => mas para ganar reteniendo a un persuadible).
    """
    par = par or ParametrosCampana()
    reclamo = (df["reclamos_6m"].to_numpy(dtype=float) > 0).astype(float)
    competencia = (df["contactos_vendedor_competencia_3m"].to_numpy(dtype=float) > 0).astype(float)
    rent_mala = (df["brecha_rentabilidad_pp"].to_numpy(dtype=float) < par.umbral_brecha_rentabilidad).astype(float)
    com_alta = (df["brecha_comision_pp"].to_numpy(dtype=float) > par.umbral_brecha_comision).astype(float)
    antig_baja = (df["antiguedad_afp_meses"].to_numpy(dtype=float) < par.umbral_antiguedad_meses).astype(float)
    z = (par.intercepto_riesgo
         + par.w_reclamo_6m * reclamo
         + par.w_contacto_competencia_3m * competencia
         + par.w_brecha_rentabilidad_mala * rent_mala
         + par.w_brecha_comision_alta * com_alta
         + par.w_antiguedad_baja * antig_baja)
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------- variables latentes (por afiliado)
def variables_latentes_afiliado(afiliados: pd.DataFrame, seed: int,
                                par: ParametrosCampana | None = None) -> pd.DataFrame:
    """Holdout (fijo en el tiempo) y sensibilidad a comision (latente) por afiliado.

    Ambas se fijan UNA vez por afiliado (no por corte): el holdout porque es un diseno experimental
    persistente, y la sensibilidad porque es un rasgo latente del afiliado (como `lat_switcher` en 02).
    """
    par = par or ParametrosCampana()
    ids = np.sort(afiliados["afiliado_id"].to_numpy())
    rng = np.random.default_rng(seed + 909)
    u_holdout = rng.random(len(ids))
    sensibilidad = rng.random(len(ids))
    return pd.DataFrame({
        "afiliado_id": ids,
        "holdout": u_holdout < par.p_holdout,
        "lat_sensibilidad_comision": sensibilidad,
    })


# --------------------------------------------------------------------- asignacion de tratamiento
def asignar_tratamiento(df: pd.DataFrame, riesgo: np.ndarray, latentes_afiliado: pd.DataFrame,
                        seed: int, par: ParametrosCampana | None = None) -> pd.DataFrame:
    """Asigna `T` por fila: aleatorio en el holdout, confundido (saldo/edad/riesgo) fuera de el.

    Devuelve una tabla (afiliado_id, periodo_corte, T, holdout, e_hist) alineada con `df`. `e_hist`
    es la propension VERDADERA de la politica historica (solo para diagnostico interno: nunca se
    expone como feature, un analista real no la conoce con exactitud).
    """
    par = par or ParametrosCampana()
    d = df[["afiliado_id", "periodo_corte", "saldo_obligatorio", "edad"]].merge(
        latentes_afiliado[["afiliado_id", "holdout"]], on="afiliado_id", how="left")
    saldo_pct = d.groupby("periodo_corte")["saldo_obligatorio"].rank(pct=True).to_numpy()
    edad_mayor = (d["edad"].to_numpy(dtype=float) >= par.edad_umbral_mayor).astype(float)
    z = (par.b0_propension + par.b_saldo_percentil * saldo_pct
         + par.b_tramo_edad_mayor * edad_mayor + par.b_score_riesgo * riesgo)
    e_hist = np.clip(1.0 / (1.0 + np.exp(-z)), 0.01, 0.99)

    rng = np.random.default_rng(seed + 1717)
    u = rng.random(len(d))
    holdout = d["holdout"].to_numpy()
    p_asignacion = np.where(holdout, par.p_tratamiento_holdout, e_hist)
    T = (u < p_asignacion).astype(np.int8)

    out = d[["afiliado_id", "periodo_corte"]].copy()
    out["T"] = T
    out["holdout"] = holdout
    out["e_hist"] = e_hist
    return out


# --------------------------------------------------------------------- efecto causal (verdad latente)
def construir_efecto_causal(df: pd.DataFrame, riesgo: np.ndarray, latentes_afiliado: pd.DataFrame,
                            par: ParametrosCampana | None = None) -> pd.DataFrame:
    """CATE verdadero por fila (`lat_cate`) y el segmento que lo genera. Solo para evaluacion.

    `df` debe traer `tramo_saldo` (de `churn_traspaso.evaluate.agregar_segmentos`) ademas de las
    columnas as-of de siempre. Ningun insumo usado aqui llega a `columnas_features` del dataset de
    entrenamiento: esta tabla vive aparte, en `afiliados_latente_uplift`.
    """
    par = par or ParametrosCampana()
    d = df[["afiliado_id", "periodo_corte", "reclamos_6m", "reclamos_abiertos_12m",
           "reclamos_desfavorables_12m", "logins_3m", "tramo_saldo"]].merge(
        latentes_afiliado[["afiliado_id", "lat_sensibilidad_comision"]], on="afiliado_id", how="left")

    reclamo_reciente_resuelto = ((d["reclamos_6m"].to_numpy(dtype=float) > 0)
                                 & (d["reclamos_abiertos_12m"].to_numpy(dtype=float) == 0)
                                 & (d["reclamos_desfavorables_12m"].to_numpy(dtype=float) == 0))
    uso_medio = ((d["logins_3m"].to_numpy(dtype=float) >= par.logins_3m_medio_lo)
                & (d["logins_3m"].to_numpy(dtype=float) < par.logins_3m_medio_hi))
    saldo_medio = d["tramo_saldo"].to_numpy() == par.tramo_saldo_medio
    persuadible = reclamo_reciente_resuelto & uso_medio & saldo_medio

    uso_bajo = d["logins_3m"].to_numpy(dtype=float) <= par.logins_3m_bajo
    sensibilidad_alta = (d["lat_sensibilidad_comision"].to_numpy(dtype=float)
                        >= (1.0 - par.frac_sensibilidad_comision_alta))
    sleeping_dog = uso_bajo & sensibilidad_alta & (~persuadible)

    segmento = np.where(persuadible, "persuadible", np.where(sleeping_dog, "sleeping_dog", "neutral"))
    lat_cate = np.where(persuadible, -par.p_retencion_persuadible * riesgo,
                        np.where(sleeping_dog, par.p_despierta_sleeping_dog * (1.0 - riesgo), 0.0))

    out = df[["afiliado_id", "periodo_corte"]].copy()
    out["segmento"] = segmento
    out["lat_cate"] = lat_cate
    out["lat_riesgo_heuristico"] = riesgo
    return out


# --------------------------------------------------------------------- alteracion del target
def aplicar_efecto_causal(y0: np.ndarray, T: np.ndarray, segmento: np.ndarray, seed: int,
                          par: ParametrosCampana | None = None) -> np.ndarray:
    """Aplica el efecto causal al target de control `y0` (identico al de 02) en las filas tratadas.

    Filas de control (`T=0`): `y = y0` siempre (coherencia exacta con `traspasos`, nunca se toca).
    Filas tratadas (`T=1`): un persuadible que se iba (`y0=1`) se retiene con prob.
    `p_retencion_persuadible`; un sleeping dog que se quedaba (`y0=0`) se traspasa con prob.
    `p_despierta_sleeping_dog`; el resto no cambia. La randomizacion es propia (semilla `seed`),
    independiente de la que genero `y0`.
    """
    par = par or ParametrosCampana()
    rng = np.random.default_rng(seed + 3131)
    y0 = np.asarray(y0, dtype=np.int8)
    T = np.asarray(T, dtype=bool)
    segmento = np.asarray(segmento)
    y = y0.copy()

    persuadible_tratado = T & (segmento == "persuadible") & (y0 == 1)
    u1 = rng.random(len(y0))
    y[persuadible_tratado & (u1 < par.p_retencion_persuadible)] = 0

    sleeping_tratado = T & (segmento == "sleeping_dog") & (y0 == 0)
    u2 = rng.random(len(y0))
    y[sleeping_tratado & (u2 < par.p_despierta_sleeping_dog)] = 1
    return y
