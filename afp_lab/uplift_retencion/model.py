"""Propension e(x), meta-learners de uplift (T-, X-, DR-learner) y baseline de riesgo.

Todo implementado a mano con LightGBM + pandas/numpy (econml/causalml no estan instalados, brief
07). Los meta-learners se entrenan SOLO con la poblacion confundida (no holdout) de train/valid: el
holdout aleatorio se reserva integro para evaluar Qini/AUUC sin sesgo (`evaluate.py`). El baseline
de riesgo (ranking de 02 usado como si fuera uplift) se entrena con toda la poblacion, tratamiento
incluido, porque ignora `T` por diseno.

n_jobs=2 en todos los boosters (ronda 4: 4 vCPU compartidas con otros dos workstreams).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from afp_lab.churn_traspaso.features import COLUMNAS_CATEGORICAS

from . import N_JOBS_LGBM
from .features import columnas_features

PARAMS_CLASIFICADOR = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_cat_threshold": 24,
    "cat_smooth": 15.0,
    "num_threads": N_JOBS_LGBM,
    "verbosity": -1,
    "force_row_wise": True,
}
PARAMS_REGRESOR = {**PARAMS_CLASIFICADOR, "objective": "regression", "metric": "l2"}
N_ARBOLES_MAX = 300
EARLY_STOPPING = 30
FRAC_VALID_PROPENSION = 0.20   # split interno (aleatorio) solo para el early stopping de e(x)


def _categoricas(cols: list[str]) -> list[str]:
    return [c for c in COLUMNAS_CATEGORICAS if c in cols]


def _entrenar_booster(params: dict, X_train, y_train, X_valid, y_valid, cat: list[str], seed: int) -> lgb.Booster:
    p = dict(params)
    p["seed"] = seed
    dtr = lgb.Dataset(X_train, label=y_train, categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat, reference=dtr, free_raw_data=False)
    return lgb.train(p, dtr, num_boost_round=N_ARBOLES_MAX, valid_sets=[dva], valid_names=["valid"],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])


def _predecir(booster: lgb.Booster, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    it = booster.best_iteration or booster.current_iteration()
    return np.asarray(booster.predict(df[cols], num_iteration=it), dtype=float)


# --------------------------------------------------------------------- propension e(x)
def estimar_propension(df_train_confundido: pd.DataFrame, seed: int) -> dict:
    """LightGBM que predice `T` a partir de `x`, SOLO con la poblacion confundida (fuera del holdout).

    Split interno aleatorio 80/20 para el early stopping: e(x) es un modelo transversal (no predice
    el futuro), asi que no necesita el split temporal de 02.
    """
    cols = columnas_features(df_train_confundido)
    cat = _categoricas(cols)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df_train_confundido))
    n_valid = max(200, int(FRAC_VALID_PROPENSION * len(idx)))
    va_idx, tr_idx = idx[:n_valid], idx[n_valid:]
    tr, va = df_train_confundido.iloc[tr_idx], df_train_confundido.iloc[va_idx]
    booster = _entrenar_booster(PARAMS_CLASIFICADOR, tr[cols], tr["T"].to_numpy(),
                               va[cols], va["T"].to_numpy(), cat, seed)
    return {"booster": booster, "columnas": cols}


def predecir_propension(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    p = _predecir(modelo["booster"], df, modelo["columnas"])
    return np.clip(p, 1e-3, 1 - 1e-3)


# --------------------------------------------------------------------- T-learner
def entrenar_t_learner(df_train: pd.DataFrame, df_valid: pd.DataFrame, seed: int) -> dict:
    """mu0(x)=E[y|T=0,x], mu1(x)=E[y|T=1,x] (LightGBM), CATE = mu1(x) - mu0(x)."""
    cols = columnas_features(df_train)
    cat = _categoricas(cols)
    tr0, tr1 = df_train[df_train["T"] == 0], df_train[df_train["T"] == 1]
    va0, va1 = df_valid[df_valid["T"] == 0], df_valid[df_valid["T"] == 1]
    mu0 = _entrenar_booster(PARAMS_CLASIFICADOR, tr0[cols], tr0["y"].to_numpy(),
                            va0[cols], va0["y"].to_numpy(), cat, seed)
    mu1 = _entrenar_booster(PARAMS_CLASIFICADOR, tr1[cols], tr1["y"].to_numpy(),
                            va1[cols], va1["y"].to_numpy(), cat, seed + 1)
    return {"mu0": mu0, "mu1": mu1, "columnas": cols}


def predecir_t_learner(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    cols = modelo["columnas"]
    p0 = _predecir(modelo["mu0"], df, cols)
    p1 = _predecir(modelo["mu1"], df, cols)
    return p1 - p0


# --------------------------------------------------------------------- X-learner
def entrenar_x_learner(df_train: pd.DataFrame, df_valid: pd.DataFrame, modelo_propension: dict,
                       seed: int) -> dict:
    """Efecto imputado (D1 en tratados, D0 en control) con el T-learner, y regresion de cada uno."""
    t_learner = entrenar_t_learner(df_train, df_valid, seed)
    cols, cat = t_learner["columnas"], _categoricas(t_learner["columnas"])
    tr0, tr1 = df_train[df_train["T"] == 0], df_train[df_train["T"] == 1]
    va0, va1 = df_valid[df_valid["T"] == 0], df_valid[df_valid["T"] == 1]

    d1_tr = tr1["y"].to_numpy() - _predecir(t_learner["mu0"], tr1, cols)
    d0_tr = _predecir(t_learner["mu1"], tr0, cols) - tr0["y"].to_numpy()
    d1_va = va1["y"].to_numpy() - _predecir(t_learner["mu0"], va1, cols)
    d0_va = _predecir(t_learner["mu1"], va0, cols) - va0["y"].to_numpy()

    tau1 = _entrenar_booster(PARAMS_REGRESOR, tr1[cols], d1_tr, va1[cols], d1_va, cat, seed + 2)
    tau0 = _entrenar_booster(PARAMS_REGRESOR, tr0[cols], d0_tr, va0[cols], d0_va, cat, seed + 3)
    return {"tau1": tau1, "tau0": tau0, "columnas": cols, "propension": modelo_propension}


def predecir_x_learner(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    cols = modelo["columnas"]
    tau1 = _predecir(modelo["tau1"], df, cols)
    tau0 = _predecir(modelo["tau0"], df, cols)
    e = predecir_propension(modelo["propension"], df)
    return e * tau0 + (1.0 - e) * tau1


# --------------------------------------------------------------------- DR-learner (doubly robust)
def entrenar_dr_learner(df_train: pd.DataFrame, df_valid: pd.DataFrame, modelo_propension: dict,
                        seed: int) -> dict:
    """Pseudo-outcome doblemente robusto (mu1-mu0 + correccion IPW) y regresion sobre `x`."""
    t_learner = entrenar_t_learner(df_train, df_valid, seed + 10)
    cols, cat = t_learner["columnas"], _categoricas(t_learner["columnas"])

    def _pseudo(df: pd.DataFrame) -> np.ndarray:
        mu0 = _predecir(t_learner["mu0"], df, cols)
        mu1 = _predecir(t_learner["mu1"], df, cols)
        e = predecir_propension(modelo_propension, df)
        T, y = df["T"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
        return (mu1 - mu0) + T * (y - mu1) / e - (1.0 - T) * (y - mu0) / (1.0 - e)

    psi_tr, psi_va = _pseudo(df_train), _pseudo(df_valid)
    reg = _entrenar_booster(PARAMS_REGRESOR, df_train[cols], psi_tr, df_valid[cols], psi_va, cat, seed + 20)
    return {"modelo": reg, "columnas": cols}


def predecir_dr_learner(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    return _predecir(modelo["modelo"], df, modelo["columnas"])


# --------------------------------------------------------------------- baseline: riesgo como si fuera uplift
def entrenar_baseline_riesgo(df_train: pd.DataFrame, df_valid: pd.DataFrame, seed: int) -> dict:
    """LightGBM de riesgo puro (ignora `T`, igual que el modelo de 02) usado COMO SI fuera un score
    de uplift: es el baseline obligatorio del brief ("ranking por riesgo usado como si fuera uplift")."""
    cols = columnas_features(df_train)
    cat = _categoricas(cols)
    booster = _entrenar_booster(PARAMS_CLASIFICADOR, df_train[cols], df_train["y"].to_numpy(),
                               df_valid[cols], df_valid["y"].to_numpy(), cat, seed + 99)
    return {"booster": booster, "columnas": cols}


def predecir_baseline_riesgo(modelo: dict, df: pd.DataFrame) -> np.ndarray:
    return _predecir(modelo["booster"], df, modelo["columnas"])
