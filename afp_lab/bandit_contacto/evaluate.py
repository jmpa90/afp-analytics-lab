"""Metricas de la simulacion online y evaluacion off-policy (IPS, SNIPS, doubly robust).

`resumen_online` agrega la curva mensual (`model.simular_todas`) al ultimo mes por politica x
semilla y calcula media +/- intervalo entre semillas (t de Student, `n-1` grados de libertad).

`evaluar_off_policy` es el corazon del brief: sobre el dataset LOGUEADO por la politica historica
(heuristica + epsilon=0,1, guardando su propension), estima el valor de cada politica candidata con
IPS, SNIPS y DR, y lo compara contra el valor VERDADERO (calculado exactamente con la tabla latente
`lat_p_churn_*`, que las politicas nunca ven). El "valor verdadero" no es una simulacion adicional:
es la esperanza analitica bajo la distribucion de accion de la politica evaluada, por eso no tiene
ruido de muestreo y sirve de referencia limpia para el error relativo de los tres estimadores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import BRAZOS, N_BRAZOS
from .model import COSTO_VEC, LGBM_PARAMS, PoliticaSinContacto, PoliticaTopKVisita


# --------------------------------------------------------------------- resumen online
def resumen_online(curvas_df: pd.DataFrame, mes_final: int | None = None) -> pd.DataFrame:
    """Media +/- intervalo (95%, t de Student) entre semillas de las metricas acumuladas al mes final."""
    mes_final = mes_final or curvas_df["mes"].max()
    ultimo = curvas_df[curvas_df["mes"] == mes_final].copy()
    ultimo["reward_neto"] = ultimo["reward"]
    ultimo["tasa_traspaso"] = ultimo["churn"] / ultimo["n"].clip(lower=1)
    cols = ["reward_neto", "regret", "costo", "tasa_traspaso", "sleeping_dog", "persuadible", "n"]
    filas = []
    for pol, g in ultimo.groupby("politica"):
        fila = {"politica": pol, "n_semillas": len(g)}
        for c in cols:
            m, sd, k = g[c].mean(), g[c].std(ddof=1), len(g)
            if k > 1 and sd == sd:  # sd no-NaN
                hw = stats.t.ppf(0.975, k - 1) * sd / np.sqrt(k)
            else:
                hw = 0.0
            fila[f"{c}_media"] = m
            fila[f"{c}_ic95"] = hw
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("reward_neto_media", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------- valor verdadero
def valor_verdadero(entorno: dict, pos: np.ndarray, propension: np.ndarray) -> float:
    """Valor esperado EXACTO (sin ruido de muestreo) de una politica con matriz de accion
    `propension` (n, 5) sobre las filas `pos`, usando la tabla latente `lat_p_churn_*`."""
    lat_p = entorno["lat_p"][pos]
    comision = entorno["comision"][pos]
    valor_arm = (1 - lat_p) * comision[:, None] - COSTO_VEC[None, :]
    return float(np.sum(propension * valor_arm, axis=1).mean())


# --------------------------------------------------------------------- direct method (Q-hat)
def entrenar_q_hat(entorno: dict, logueado: pd.DataFrame):
    import lightgbm as lgb

    raw = entorno["raw"][logueado["pos"].to_numpy()]
    Xa = np.hstack([raw, logueado["arm_idx"].to_numpy().reshape(-1, 1)])
    modelo = lgb.LGBMRegressor(**LGBM_PARAMS)
    modelo.fit(Xa, logueado["reward"].to_numpy())
    return modelo


def q_hat_matriz(modelo, entorno: dict, pos: np.ndarray) -> np.ndarray:
    """Q(x,a) del direct method para las 5 armas, filas `pos`."""
    raw = entorno["raw"][pos]
    m = len(pos)
    rep = np.repeat(raw, N_BRAZOS, axis=0)
    brazo_idx = np.tile(np.arange(N_BRAZOS), m).reshape(-1, 1)
    Xa = np.hstack([rep, brazo_idx])
    return modelo.predict(Xa).reshape(m, N_BRAZOS)


# --------------------------------------------------------------------- estimadores OPE
def estimador_ips(logueado: pd.DataFrame, pi_e_logged: np.ndarray) -> float:
    w = pi_e_logged / logueado["propension_logueada"].to_numpy()
    return float(np.mean(w * logueado["reward"].to_numpy()))


def estimador_snips(logueado: pd.DataFrame, pi_e_logged: np.ndarray) -> float:
    w = pi_e_logged / logueado["propension_logueada"].to_numpy()
    r = logueado["reward"].to_numpy()
    return float(np.sum(w * r) / np.sum(w))


def estimador_dr(logueado: pd.DataFrame, pi_e_logged: np.ndarray, pi_e_matriz: np.ndarray,
                 q_hat_logged: np.ndarray, q_hat_matriz_: np.ndarray) -> float:
    w = pi_e_logged / logueado["propension_logueada"].to_numpy()
    r = logueado["reward"].to_numpy()
    directo = np.sum(pi_e_matriz * q_hat_matriz_, axis=1)
    return float(np.mean(directo + w * (r - q_hat_logged)))


def evaluar_off_policy(entorno: dict, logueado: pd.DataFrame, politicas_congeladas: dict
                       ) -> pd.DataFrame:
    """Tabla de valor verdadero vs IPS/SNIPS/DR (y su error relativo) para cada politica candidata,
    mas los dos baselines y una fila de "sanity check" con la propia politica de logging (pi_e=pi_0,
    donde IPS/SNIPS deben coincidir exactamente con el valor verdadero: valida la implementacion)."""
    pos = logueado["pos"].to_numpy()
    arm_idx = logueado["arm_idx"].to_numpy()
    modelo_q = entrenar_q_hat(entorno, logueado)
    q_hat_full = q_hat_matriz(modelo_q, entorno, pos)          # (n,5)
    q_hat_logged = q_hat_full[np.arange(len(pos)), arm_idx]

    candidatas = dict(politicas_congeladas)
    candidatas["SIN_CONTACTO"] = PoliticaSinContacto()
    candidatas["TOPK_RIESGO_VISITA"] = PoliticaTopKVisita()
    candidatas["HEURISTICA_LOGGING (sanity)"] = None  # caso especial: pi_e = pi_0

    filas = []
    for nombre, politica in candidatas.items():
        if politica is None:
            pi_e_matriz = _propension_real_logueada(logueado, len(pos))
        else:
            pi_e_matriz = politica.propension_matriz(entorno, pos)
        pi_e_logged = pi_e_matriz[np.arange(len(pos)), arm_idx]

        v_true = valor_verdadero(entorno, pos, pi_e_matriz)
        ips = estimador_ips(logueado, pi_e_logged)
        snips = estimador_snips(logueado, pi_e_logged)
        dr = estimador_dr(logueado, pi_e_logged, pi_e_matriz, q_hat_logged, q_hat_full)
        filas.append({
            "politica": nombre, "valor_verdadero": v_true,
            "ips": ips, "snips": snips, "dr": dr,
            "error_rel_ips": abs(ips - v_true) / abs(v_true),
            "error_rel_snips": abs(snips - v_true) / abs(v_true),
            "error_rel_dr": abs(dr - v_true) / abs(v_true),
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------- interpretabilidad
def importancia_lineal(politica, columnas: list[str] | None = None) -> pd.DataFrame:
    """Peso de cada feature en la media posterior (theta) de cada brazo, para LinUCB/Thompson
    lineal: el analogo lineal de "importancia" para estos dos algoritmos."""
    from .features import CONTEXTO_NUMERICO
    cols = list(columnas or CONTEXTO_NUMERICO) + ["bias"]
    filas = []
    for a, brazo in enumerate(BRAZOS):
        mu, _ = politica._mu_cov(a)
        for col, peso in zip(cols, mu):
            filas.append({"brazo": brazo, "feature": col, "peso": float(peso)})
    df = pd.DataFrame(filas)
    df["abs_peso"] = df["peso"].abs()
    return (df.sort_values(["brazo", "abs_peso"], ascending=[True, False])
           .groupby("brazo").head(4).drop(columns="abs_peso").reset_index(drop=True))


def importancia_lgbm(politica_bootstrap) -> pd.DataFrame:
    """Importancia (gain, promedio del ensamble bootstrap) del ultimo refit de Thompson-LGBM."""
    from .features import CONTEXTO_NUMERICO
    cols = list(CONTEXTO_NUMERICO) + ["brazo_idx"]
    if politica_bootstrap.modelos is None:
        return pd.DataFrame(columns=["feature", "gain_medio"])
    gains = np.mean([m.booster_.feature_importance(importance_type="gain")
                     for m in politica_bootstrap.modelos], axis=0)
    return pd.DataFrame({"feature": cols, "gain_medio": gains}).sort_values(
        "gain_medio", ascending=False).reset_index(drop=True)


def ejemplos_individuales(entorno: dict, logueado: pd.DataFrame, n: int = 5, seed: int = 0
                          ) -> pd.DataFrame:
    """`n` decisiones logueadas al azar con su contexto de negocio y el resultado, para inspeccion
    cualitativa (CONVENCIONES §4). `tipo_respuesta_latente` se muestra SOLO como diagnostico: nunca
    la vio la politica que eligio el brazo."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(logueado), size=min(n, len(logueado)), replace=False)
    ctx = entorno["contexto"]
    filas = []
    for _, fila in logueado.iloc[idx].iterrows():
        pos = int(fila["pos"])
        filas.append({
            "afiliado_id": int(fila["afiliado_id"]), "periodo_corte": int(fila["periodo_corte"]),
            "score_riesgo": round(float(ctx.iloc[pos]["score_riesgo"]), 2),
            "rim_ref_clp": round(float(ctx.iloc[pos]["rim_ref"]), 0),
            "brecha_comision_rel": round(float(ctx.iloc[pos]["brecha_comision_rel"]), 3),
            "brazo_logueado": fila["brazo"], "propension_logueada": round(float(fila["propension_logueada"]), 3),
            "reward_realizado_clp": round(float(fila["reward"]), 0), "traspaso_realizado": bool(fila["traspaso"]),
            "tipo_respuesta_latente": entorno["tipo_resp"][pos],
        })
    return pd.DataFrame(filas)


def _propension_real_logueada(logueado: pd.DataFrame, n: int) -> np.ndarray:
    """Matriz de propension EXACTA de la politica de logging (mezcla `(1-eps)*1{a=tier} + eps/5`),
    reconstruida con la columna `tier_idx` guardada en el log (el brazo por terciles de riesgo,
    antes de la exploracion epsilon). Se usa solo para el sanity check: con esta matriz, IPS y SNIPS
    deben reproducir el valor verdadero casi exactamente (pi_e = pi_0)."""
    from . import EPSILON_LOGGING
    eps = EPSILON_LOGGING
    p = np.full((n, N_BRAZOS), eps / N_BRAZOS)
    fila = np.arange(n)
    p[fila, logueado["tier_idx"].to_numpy()] += 1 - eps
    return p
