"""Modelos del proyecto 11: respuesta, NPS (score + detractor), drivers (T-learner) y valor del punto.

* **Modelo de respuesta** (`entrenar_propension_respuesta`): LightGBM binario que predice, entre los
  INVITADOS, quien responde. Su inversa (recortada) pondera el modelo de NPS para corregir el sesgo
  de no respuesta (brief).
* **Modelo de NPS** (`entrenar_score` + `entrenar_detractor`): LightGBM entrenado SOLO con
  respondientes -- sin ponderar y ponderado por el inverso de la propension de responder -- para
  comparar (evaluate.py) contra el NPS ingenuo y el verdadero.
* **Drivers** (`estimar_drivers`): T-learner simple (regresion Ridge por grupo tratado/control,
  como alternativa "simple" a LightGBM -- brief: "T-learner o regresion ajustada por propension") por
  evento, comparado contra `synthetic.EFECTOS_VERDADEROS_EVENTOS`.
* **Valor del punto** (`valor_punto_traspaso` / `valor_punto_apv`): regresion logistica / lineal del
  target de 02/03 sobre el NPS inferido y covariables, con la advertencia de que es asociacion
  ajustada, no un experimento.

n_jobs=2 en todos los boosters (maquina de 4 vCPU compartida).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from . import N_JOBS_LGBM
from .features import COLUMNAS_CATEGORICAS, columnas_features
from .synthetic import DETRACTOR_HASTA, PROMOTOR_DESDE

PARAMS_CLASIFICADOR = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 24,
    "min_data_in_leaf": 40, "feature_fraction": 0.75, "bagging_fraction": 0.80, "bagging_freq": 1,
    "lambda_l2": 3.0, "max_cat_threshold": 24, "cat_smooth": 15.0,
    "num_threads": N_JOBS_LGBM, "verbosity": -1, "force_row_wise": True,
}
PARAMS_REGRESOR = {**PARAMS_CLASIFICADOR, "objective": "regression", "metric": "l2"}
N_ARBOLES_MAX = 300
EARLY_STOPPING = 30
TOPE_PESO_IPW = 15.0     # recorte del peso inverso a la propension de responder


def _categoricas(cols: list[str]) -> list[str]:
    return [c for c in COLUMNAS_CATEGORICAS if c in cols]


def _entrenar_booster(params: dict, X_train, y_train, X_valid, y_valid, cat, seed: int,
                      w_train=None, w_valid=None) -> lgb.Booster:
    p = dict(params)
    p["seed"] = seed
    dtr = lgb.Dataset(X_train, label=y_train, categorical_feature=cat, weight=w_train, free_raw_data=False)
    dva = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat, weight=w_valid,
                      reference=dtr, free_raw_data=False)
    return lgb.train(p, dtr, num_boost_round=N_ARBOLES_MAX, valid_sets=[dva], valid_names=["valid"],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])


def _predecir(booster: lgb.Booster, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    it = booster.best_iteration or booster.current_iteration()
    return np.asarray(booster.predict(df[cols], num_iteration=it), dtype=float)


# --------------------------------------------------------------------- modelo de respuesta
def entrenar_propension_respuesta(df_train: pd.DataFrame, df_valid: pd.DataFrame, seed: int = 42) -> dict:
    """LightGBM que predice `respondio` entre los INVITADOS de train/valid."""
    inv_tr = df_train[df_train["invitado"] == 1]
    inv_va = df_valid[df_valid["invitado"] == 1]
    cols = columnas_features(df_train)
    cat = _categoricas(cols)
    booster = _entrenar_booster(PARAMS_CLASIFICADOR, inv_tr[cols], inv_tr["respondio"].to_numpy(),
                               inv_va[cols], inv_va["respondio"].to_numpy(), cat, seed)
    return {"booster": booster, "columnas": cols, "n_train": int(len(inv_tr)), "n_valid": int(len(inv_va))}


def predecir_propension_respuesta(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    p = _predecir(modelo["booster"], df, modelo["columnas"])
    return np.clip(p, 1e-3, 1 - 1e-3)


def pesos_ipw(prob_responde: np.ndarray, tope: float = TOPE_PESO_IPW) -> np.ndarray:
    """Peso inverso a la propension de responder, recortado en `[1, tope]` (estabiliza la cola)."""
    return np.clip(1.0 / np.asarray(prob_responde, dtype=float), 1.0, tope)


# --------------------------------------------------------------------- modelo de NPS (score 0-10)
def entrenar_score(df_train_resp: pd.DataFrame, df_valid_resp: pd.DataFrame, seed: int = 42,
                   pesos_train: np.ndarray | None = None, pesos_valid: np.ndarray | None = None) -> dict:
    """Regresion LightGBM de la respuesta (0-10) sobre respondientes; `pesos_*` = IPW o `None`."""
    cols = columnas_features(df_train_resp)
    cat = _categoricas(cols)
    booster = _entrenar_booster(PARAMS_REGRESOR, df_train_resp[cols], df_train_resp["respuesta"].to_numpy(),
                               df_valid_resp[cols], df_valid_resp["respuesta"].to_numpy(), cat, seed,
                               w_train=pesos_train, w_valid=pesos_valid)
    return {"booster": booster, "columnas": cols}


def predecir_score(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    return np.clip(_predecir(modelo["booster"], df, modelo["columnas"]), 0.0, 10.0)


def entrenar_detractor(df_train_resp: pd.DataFrame, df_valid_resp: pd.DataFrame, seed: int = 42,
                       pesos_train: np.ndarray | None = None, pesos_valid: np.ndarray | None = None) -> dict:
    """Clasificador `es_detractor` (para el AUC/calibracion pedidos por el brief)."""
    cols = columnas_features(df_train_resp)
    cat = _categoricas(cols)
    booster = _entrenar_booster(PARAMS_CLASIFICADOR, df_train_resp[cols], df_train_resp["es_detractor"].to_numpy(),
                               df_valid_resp[cols], df_valid_resp["es_detractor"].to_numpy(), cat, seed + 1,
                               w_train=pesos_train, w_valid=pesos_valid)
    return {"booster": booster, "columnas": cols}


def predecir_detractor(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    return np.clip(_predecir(modelo["booster"], df, modelo["columnas"]), 1e-6, 1 - 1e-6)


def entrenar_promotor(df_train_resp: pd.DataFrame, df_valid_resp: pd.DataFrame, seed: int = 42,
                      pesos_train: np.ndarray | None = None, pesos_valid: np.ndarray | None = None) -> dict:
    """Clasificador `es_promotor` (simetrico a `entrenar_detractor`; se usa junto con el de detractor
    para agregar el NPS inferido como `100 x (P(promotor) - P(detractor))` promediado en la
    poblacion -- ver nota en `nps_de_probs`)."""
    cols = columnas_features(df_train_resp)
    cat = _categoricas(cols)
    booster = _entrenar_booster(PARAMS_CLASIFICADOR, df_train_resp[cols], df_train_resp["es_promotor"].to_numpy(),
                               df_valid_resp[cols], df_valid_resp["es_promotor"].to_numpy(), cat, seed + 2,
                               w_train=pesos_train, w_valid=pesos_valid)
    return {"booster": booster, "columnas": cols}


def predecir_promotor(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    return np.clip(_predecir(modelo["booster"], df, modelo["columnas"]), 1e-6, 1 - 1e-6)


def nps_de_probs(p_promotor: np.ndarray, p_detractor: np.ndarray) -> float:
    """NPS agregado a partir de PROBABILIDADES (no de un score puntual bucketizado).

    Bucketizar un score de regresion (redondear y clasificar en promotor/pasivo/detractor) hereda el
    "encogimiento" hacia la media tipico de un regresor entrenado con pocos respondientes: la
    dispersion de las PREDICCIONES queda mucho mas angosta que la dispersion real de la satisfaccion,
    asi que casi nadie cruza los umbrales 6/9 y el NPS bucketizado queda sesgado hacia 0 aunque el
    nivel medio este bien. Promediar P(promotor|x) - P(detractor|x) directamente de dos clasificadores
    no tiene ese problema (es una expectativa, no un umbral sobre un punto), y es la version que se
    usa para el NPS inferido (`run.py`); el score de regresion se conserva para drivers/valor del punto.
    """
    return float(100 * (np.mean(p_promotor) - np.mean(p_detractor)))


def categoria_de_score(score: np.ndarray) -> np.ndarray:
    """Bucketiza un score 0-10 con la MISMA regla del generador (promotor 9-10, detractor 0-6)."""
    s = np.clip(np.round(np.asarray(score, dtype=float)), 0, 10)
    return np.where(s >= PROMOTOR_DESDE, "PROMOTOR", np.where(s <= DETRACTOR_HASTA, "DETRACTOR", "PASIVO"))


def nps_de_score(score: np.ndarray) -> float:
    cat = categoria_de_score(score)
    n = len(cat)
    if n == 0:
        return float("nan")
    return float((cat == "PROMOTOR").sum() / n * 100 - (cat == "DETRACTOR").sum() / n * 100)


# --------------------------------------------------------------------- baselines
def baseline_score_trivial(df_train_resp: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Trivial: la respuesta media de los respondientes de train para todos (persistencia de nivel)."""
    return np.full(len(df), float(df_train_resp["respuesta"].mean()))


def baseline_detractor_trivial(df_train_resp: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), float(df_train_resp["es_detractor"].mean()))


UMBRAL_COMISION_REGLA = 0.30
# CORRECCION ronda 5 (bloqueante #1 de la revision): la version anterior de este baseline copiaba
# 5 de 6 coeficientes de evento Y el nivel base, a la decima, de `synthetic.ParametrosSatisfaccion`
# (el generador oculto) pese a que el docstring decia "sin ver el generador" -- eso invalidaba la
# comparacion contra el modelo (CONVENCIONES SS4), porque el "baseline de regla" en realidad tenia
# acceso indebido a la formula que genera `lat_satisfaccion`. Estos coeficientes son ahora
# DELIBERADAMENTE independientes de `ParametrosSatisfaccion`: numeros redondos ("medio punto" /
# "un punto" de NPS por senal, en la misma direccion que el brief pero SIN copiar sus magnitudes)
# que un analista de negocio anotaria a mano mirando solo los NOMBRES de los eventos operativos que
# la AFP ya registra -- nunca el codigo de `synthetic.py`. Ninguno de los valores de abajo coincide
# con ningun `ef_*`/`base_nivel` de `ParametrosSatisfaccion` (verificado en
# `test_regla_negocio_independiente_del_generador`).
COEF_REGLA = {
    # base = 7.5 (no 7.0 "de libro" ni 7.6 del generador): en el perfil `cuprum` la brecha de
    # comision (dato de MERCADO publico -- comision de Cuprum vs. la mas barata -- no un parametro
    # oculto) supera el umbral en ~100 % de las filas (verificado abajo), asi que con base=7.0 CASI
    # TODA la cartera quedaria con score = 7.0-0.5 = 6.5 -> redondea a 6 -> "detractor" solo por el
    # descuento de comision, un artefacto degenerado de la regla, no una lectura de negocio. Un
    # analista que mire el mercado (sin ver `synthetic.py`) subiria el nivel base un poco para que
    # ese descuento cronico no arrastre a casi todos a detractor; 7.5 es la eleccion redonda mas
    # simple que lo evita (7.5-0.5=7.0, banda pasiva) y de todos modos NO coincide con ningun
    # parametro de `ParametrosSatisfaccion` (ver `test_regla_negocio_independiente_del_generador`).
    "base": 7.5, "reclamos_6m": -1.0, "rezago_mensual_actual": -1.0, "licencia_larga_3m": -1.0,
    "dnp_empleador_actual": -0.5, "contactos_ejecutivo_propio_3m": 0.4, "brecha_comision": -0.5,
}


def baseline_score_regla(df: pd.DataFrame) -> np.ndarray:
    """Regla de negocio simple: parte de 7,5 (nivel "razonablemente satisfecho" en una escala 0-10,
    ya compensado por la brecha de comision cronica del mercado -- ver nota de `COEF_REGLA`) y resta/
    suma un punto (medio punto para las senales de contexto/mercado) por cada senal operativa
    observable presente, SIN ver `synthetic.ParametrosSatisfaccion` -- coeficientes redondeados a
    mano, deliberadamente distintos de los del generador."""
    s = (COEF_REGLA["base"]
        + COEF_REGLA["reclamos_6m"] * (df["reclamos_6m"].to_numpy() > 0)
        + COEF_REGLA["rezago_mensual_actual"] * (df["rezago_mensual_actual"].to_numpy() > 0)
        + COEF_REGLA["licencia_larga_3m"] * (df["licencia_larga_3m"].to_numpy() > 0)
        + COEF_REGLA["dnp_empleador_actual"] * (df["dnp_empleador_actual"].to_numpy() > 0)
        + COEF_REGLA["contactos_ejecutivo_propio_3m"] * (df["contactos_ejecutivo_propio_3m"].to_numpy() > 0)
        + COEF_REGLA["brecha_comision"] * (df["brecha_comision_pp"].to_numpy() > UMBRAL_COMISION_REGLA))
    return np.clip(s, 0.0, 10.0)


def baseline_detractor_regla(df: pd.DataFrame) -> np.ndarray:
    """Mismo riesgo que `baseline_score_regla`, invertido y escalado a [0, 1] (pseudo-probabilidad;
    no es una probabilidad calibrada, solo un score de riesgo para el ranking/AUC)."""
    return np.clip((10.0 - baseline_score_regla(df)) / 10.0, 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------- interpretabilidad
def importancias(modelo: dict, top: int = 20) -> pd.DataFrame:
    b = modelo["booster"]
    imp = pd.DataFrame({"feature": b.feature_name(), "gain": b.feature_importance(importance_type="gain")})
    imp["gain_pct"] = 100 * imp["gain"] / max(imp["gain"].sum(), 1e-9)
    return imp.sort_values("gain", ascending=False).head(top).reset_index(drop=True)


def explicar_prediccion_individual(modelo: dict, df: pd.DataFrame, fila: int, top: int = 5) -> pd.DataFrame:
    """Contribucion de cada feature a la prediccion de UNA fila puntual (LightGBM `pred_contrib`,
    equivalente a valores SHAP para arboles: suman exactamente el margen de la prediccion final,
    `base_value` incluido). CONVENCIONES §4 pide 3-5 ejemplos individuales explicados, ademas de la
    importancia global (`importancias`); esta funcion cubre eso para un clasificador/regresor ya
    entrenado, sin volver a entrenar nada (barato)."""
    b, cols = modelo["booster"], modelo["columnas"]
    x = df.iloc[[fila]][cols]
    it = b.best_iteration or b.current_iteration()
    contrib = np.asarray(b.predict(x, num_iteration=it, pred_contrib=True), dtype=float)[0]
    feats = pd.DataFrame({"feature": cols, "valor_feature": list(x.iloc[0].to_numpy()),
                          "contribucion": contrib[:-1]})
    feats["abs_contribucion"] = feats["contribucion"].abs()
    top_feats = feats.sort_values("abs_contribucion", ascending=False).head(top).drop(columns="abs_contribucion")
    base = pd.DataFrame([{"feature": "base_value", "valor_feature": np.nan, "contribucion": float(contrib[-1])}])
    return pd.concat([base, top_feats], ignore_index=True)


# --------------------------------------------------------------------- drivers: T-learner simple
EVENTOS_DRIVERS = {
    "reclamo_sin_resolver": lambda df: (df["reclamos_abiertos_12m"].to_numpy() > 0)
                                       | (df["reclamos_desfavorables_12m"].to_numpy() > 0),
    "rezago_mensual": lambda df: df["rezago_mensual_actual"].to_numpy() > 0,
    "licencia_mal_pagada": lambda df: df["licencia_larga_3m"].to_numpy() > 0,
    "dnp_empleador": lambda df: df["dnp_empleador_actual"].to_numpy() > 0,
    "contacto_ejecutivo_resuelto": lambda df: df["contactos_ejecutivo_propio_3m"].to_numpy() > 0,
    "rentabilidad_relativa_positiva": lambda df: df["brecha_rentabilidad_pp"].to_numpy() > 0,
    "brecha_comision_desfavorable": lambda df: df["brecha_comision_pp"].to_numpy() > UMBRAL_COMISION_REGLA,
}
CONFUSORES_NUMERICOS = ["edad", "meses_desde_afiliacion", "log_saldo", "rim_l1", "densidad_12",
                        "antiguedad_afp_meses", "meses_con_empleador", "mes"]


def _matriz_confusores(df: pd.DataFrame, excluir: str | None = None) -> np.ndarray:
    x = df[CONFUSORES_NUMERICOS].fillna(0.0).to_numpy(dtype=float)
    mu, sigma = x.mean(axis=0), x.std(axis=0)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    xs = (x - mu) / sigma
    dummies = pd.get_dummies(df["tipo_afiliado"].astype(str), prefix="seg").to_numpy(dtype=float)
    otros = [f(df).astype(float) for nombre, f in EVENTOS_DRIVERS.items() if nombre != excluir]
    return np.column_stack([xs, dummies] + otros)


def estimar_drivers(df_resp: pd.DataFrame, pesos: np.ndarray | None = None,
                    alpha: float = 8.0) -> pd.DataFrame:
    """T-learner (Ridge) del efecto de cada evento sobre `respuesta`, ponderado por `pesos` (IPW).

    Devuelve una fila por evento: efecto estimado (puntos de NPS), efecto verdadero del generador,
    coincidencia de signo y n de tratados/control. Estimador SIMPLE (brief): dos regresiones Ridge
    (tratado / control) sobre los mismos confusores observables, en vez de un LightGBM completo por
    evento (mas rapido y mas estable con el tamano de muestra de los respondientes).
    """
    from .synthetic import EFECTOS_VERDADEROS_EVENTOS, ParametrosSatisfaccion
    par = ParametrosSatisfaccion()
    y = df_resp["respuesta"].to_numpy(dtype=float)
    w = np.ones(len(df_resp)) if pesos is None else np.asarray(pesos, dtype=float)
    filas = []
    for nombre, flag_fn in EVENTOS_DRIVERS.items():
        T = flag_fn(df_resp).astype(int)
        if T.sum() < 15 or (len(T) - T.sum()) < 15:
            filas.append({"evento": nombre, "n_tratados": int(T.sum()), "n_control": int((1 - T).sum()),
                         "efecto_estimado": float("nan"), "efecto_verdadero": getattr(par, EFECTOS_VERDADEROS_EVENTOS[nombre][0]),
                         "signo_correcto": None})
            continue
        Xc = _matriz_confusores(df_resp, excluir=nombre)
        m0 = Ridge(alpha=alpha).fit(Xc[T == 0], y[T == 0], sample_weight=w[T == 0])
        m1 = Ridge(alpha=alpha).fit(Xc[T == 1], y[T == 1], sample_weight=w[T == 1])
        tau = m1.predict(Xc) - m0.predict(Xc)
        efecto = float(np.average(tau, weights=w))
        verdadero = getattr(par, EFECTOS_VERDADEROS_EVENTOS[nombre][0])
        filas.append({"evento": nombre, "n_tratados": int(T.sum()), "n_control": int((1 - T).sum()),
                     "efecto_estimado": efecto, "efecto_verdadero": verdadero,
                     "signo_correcto": bool(np.sign(efecto) == np.sign(verdadero))})
    out = pd.DataFrame(filas)
    out["orden_magnitud_ok"] = (out["efecto_estimado"].abs() > 0.3 * out["efecto_verdadero"].abs()) & \
        (out["efecto_estimado"].abs() < 3.0 * out["efecto_verdadero"].abs())
    return out.sort_values("efecto_estimado").reset_index(drop=True)


def tabla_que_arreglar_primero(drivers: pd.DataFrame, df_resp: pd.DataFrame) -> pd.DataFrame:
    """Prioriza eventos por impacto agregado = |efecto estimado| x incidencia (fraccion de filas)."""
    inc = {n: float(f(df_resp).mean()) for n, f in EVENTOS_DRIVERS.items()}
    out = drivers.copy()
    out["incidencia"] = out["evento"].map(inc)
    out["impacto_agregado_puntos"] = out["efecto_estimado"].abs() * out["incidencia"]
    return out.sort_values("impacto_agregado_puntos", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------- valor del punto de NPS
COLUMNAS_VALOR_PUNTO = ["edad", "meses_desde_afiliacion", "log_saldo", "rim_l1", "densidad_12",
                       "antiguedad_afp_meses", "mes"]


def _diseno_valor_punto(df: pd.DataFrame, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[COLUMNAS_VALOR_PUNTO].fillna(0.0).to_numpy(dtype=float)
    mu, sigma = x.mean(axis=0), x.std(axis=0)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    xs = (x - mu) / sigma
    dummies = pd.get_dummies(df["tipo_afiliado"].astype(str), prefix="seg").to_numpy(dtype=float)
    s = (np.asarray(score, dtype=float) - 5.0) / 2.0   # score centrado/escalado (± ~1 por 2 puntos de NPS)
    X = np.column_stack([s.reshape(-1, 1), xs, dummies])
    return X, mu, sigma


def valor_punto_traspaso(df: pd.DataFrame, score: np.ndarray, y_traspaso: np.ndarray) -> dict:
    """Asociacion (ajustada) entre el NPS inferido y el traspaso a 6 meses: probabilidad marginal
    de traspaso por +1 punto de NPS, y traspasos/saldo evitados (advertencia: asociacion, no
    experimento -- 02 no conoce la satisfaccion)."""
    X, *_ = _diseno_valor_punto(df, score)
    y = np.asarray(y_traspaso, dtype=int)
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return {"n": len(y), "tasa_base": float(y.mean()) if len(y) else float("nan"),
               "delta_prob_por_punto": float("nan"), "traspasos_evitados_por_punto": float("nan"),
               "saldo_evitado_por_punto_mm": float("nan")}
    modelo = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    X_mas = X.copy()
    X_mas[:, 0] += 0.5   # +1 punto de NPS en la escala centrada del score (s = (score-5)/2)
    p0 = modelo.predict_proba(X)[:, 1]
    p1 = modelo.predict_proba(X_mas)[:, 1]
    delta = float(np.mean(p1 - p0))
    saldo = df["saldo_obligatorio"].to_numpy(dtype=float) if "saldo_obligatorio" in df.columns else np.zeros(len(df))
    return {
        "n": int(len(y)), "tasa_base": float(y.mean()),
        "delta_prob_por_punto": delta,
        "traspasos_evitados_por_punto": float(-delta * len(y)),
        "saldo_evitado_por_punto_mm": float(-delta * np.mean(saldo) * len(y) / 1e6),
    }


def valor_punto_apv(df: pd.DataFrame, score: np.ndarray, monto_apv: np.ndarray) -> dict:
    """Asociacion entre el NPS inferido y el monto de aporte APV a 6 meses (CLP extra por punto)."""
    X, *_ = _diseno_valor_punto(df, score)
    y = np.asarray(monto_apv, dtype=float)
    modelo = LinearRegression().fit(X, y)
    coef_punto = float(modelo.coef_[0]) * 0.5   # por +1 punto de NPS (ver `_diseno_valor_punto`)
    return {
        "n": int(len(y)), "monto_medio_actual": float(y.mean()),
        "monto_extra_por_punto_clp": coef_punto,
        "monto_extra_por_punto_cartera_mm": float(coef_punto * len(y) / 1e6),
    }
