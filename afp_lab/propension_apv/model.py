"""Modelo *hurdle* de propension a ahorro voluntario y baselines obligatorios (CONVENCIONES §4).

Dos etapas, ambas LightGBM (pandas + LightGBM + scikit-learn; la escala no exige Spark):

1. **Clasificador binario**: `p = P(al menos un aporte en corte+1..corte+6)`.
   Se calibra con regresion isotonica ajustada en el set de **validacion** (nunca en test),
   porque la probabilidad se agrega para proyectar el flujo de ahorro voluntario.
2. **Regresor** sobre `log(monto)` en las filas con aporte -> `monto_condicional`.
   Como la perdida L2 sobre logs estima la mediana-geometrica, se aplica un factor de
   *smearing* `s = sum(monto real) / sum(exp(pred))` calculado en entrenamiento.

Salida por afiliado: `prob_aporte`, `monto_condicional` y
`monto_esperado = prob_aporte x monto_condicional`, que es el score de priorizacion.

Baselines obligatorios:

* `tasa_base`: probabilidad constante = tasa de aporte del entrenamiento; monto esperado
  constante = tasa x monto medio de los aportantes.
* `regla_negocio`: la regla del brief -- "renta > 2 M **y** empleador con APVC **o** aporte en
  24 m" -- con las tasas empiricas de entrenamiento de cada grupo como probabilidad.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from .features import columnas_modelo

UMBRAL_RENTA_REGLA = 2_000_000.0     # "renta > 2 M" del brief


@dataclass
class ParametrosModelo:
    """Hiperparametros. `n_jobs=2` porque la maquina de 4 nucleos es compartida."""

    n_jobs: int = 2
    seed: int = 42
    # clasificador
    clf_learning_rate: float = 0.05
    clf_num_leaves: int = 48
    clf_min_child_samples: int = 120
    clf_n_estimators: int = 700
    clf_feature_fraction: float = 0.75
    clf_bagging_fraction: float = 0.80
    # regresor de log(monto)
    reg_learning_rate: float = 0.05
    reg_num_leaves: int = 24
    reg_min_child_samples: int = 60
    reg_n_estimators: int = 500
    reg_feature_fraction: float = 0.75
    reg_bagging_fraction: float = 0.80
    early_stopping: int = 60
    version: str = "apv-hurdle-lgbm-v1"


@dataclass
class ModeloAPV:
    """Modelo entrenado + calibracion + metadatos de trazabilidad."""

    clf: lgb.LGBMClassifier
    reg: lgb.LGBMRegressor
    calibrador: IsotonicRegression
    usa_calibracion: bool
    brier_valid_crudo: float
    brier_valid_calibrado: float
    columnas: list[str]
    smearing: float
    tasa_base_train: float
    monto_medio_aportantes_train: float
    params: ParametrosModelo = field(default_factory=ParametrosModelo)
    cortes_train: list[int] = field(default_factory=list)
    cortes_valid: list[int] = field(default_factory=list)

    @property
    def version(self) -> str:
        return self.params.version


# --------------------------------------------------------------------------- entrenamiento
def entrenar(train: pd.DataFrame, valid: pd.DataFrame,
             params: ParametrosModelo | None = None) -> ModeloAPV:
    """Entrena el hurdle con early stopping y calibracion sobre `valid`."""
    params = params or ParametrosModelo()
    cols = columnas_modelo(train)

    clf = lgb.LGBMClassifier(
        objective="binary", learning_rate=params.clf_learning_rate,
        num_leaves=params.clf_num_leaves, min_child_samples=params.clf_min_child_samples,
        n_estimators=params.clf_n_estimators, colsample_bytree=params.clf_feature_fraction,
        subsample=params.clf_bagging_fraction, subsample_freq=1,
        n_jobs=params.n_jobs, random_state=params.seed, verbose=-1)
    clf.fit(train[cols], train["y"], eval_X=valid[cols], eval_y=valid["y"], eval_metric="auc",
            callbacks=[lgb.early_stopping(params.early_stopping, verbose=False)])

    # --- calibracion isotonica sobre validacion (monotona: no altera el orden ni el AUC)
    # Se decide CON VALIDACION (nunca con test) si conviene calibrar: la isotonica se evalua
    # fuera de muestra con 3 folds dentro de validacion y solo se aplica si baja el Brier.
    p_valid = clf.predict_proba(valid[cols])[:, 1]
    y_valid = valid["y"].to_numpy()
    calibrador = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrador.fit(p_valid, y_valid)
    p_cv = np.empty_like(p_valid)
    for i_tr, i_te in StratifiedKFold(3, shuffle=True, random_state=params.seed).split(p_valid, y_valid):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p_valid[i_tr], y_valid[i_tr])
        p_cv[i_te] = iso.predict(p_valid[i_te])
    brier_crudo = float(np.mean((p_valid - y_valid) ** 2))
    brier_cal = float(np.mean((p_cv - y_valid) ** 2))
    usa_calibracion = brier_cal <= brier_crudo

    # --- regresor de log(monto) en aportantes
    tr_pos, va_pos = train[train["y"] == 1], valid[valid["y"] == 1]
    reg = lgb.LGBMRegressor(
        objective="regression", learning_rate=params.reg_learning_rate,
        num_leaves=params.reg_num_leaves, min_child_samples=params.reg_min_child_samples,
        n_estimators=params.reg_n_estimators, colsample_bytree=params.reg_feature_fraction,
        subsample=params.reg_bagging_fraction, subsample_freq=1,
        n_jobs=params.n_jobs, random_state=params.seed, verbose=-1)
    reg.fit(tr_pos[cols], np.log(tr_pos["monto"]),
            eval_X=va_pos[cols], eval_y=np.log(va_pos["monto"]), eval_metric="l2",
            callbacks=[lgb.early_stopping(params.early_stopping, verbose=False)])

    pred_tr = np.exp(reg.predict(tr_pos[cols]))
    smearing = float(tr_pos["monto"].sum() / max(pred_tr.sum(), 1e-9))

    return ModeloAPV(
        clf=clf, reg=reg, calibrador=calibrador, usa_calibracion=usa_calibracion,
        brier_valid_crudo=brier_crudo, brier_valid_calibrado=brier_cal,
        columnas=cols, smearing=smearing,
        tasa_base_train=float(train["y"].mean()),
        monto_medio_aportantes_train=float(train.loc[train["y"] == 1, "monto"].mean()),
        params=params,
        cortes_train=sorted(train["periodo_corte"].unique().tolist()),
        cortes_valid=sorted(valid["periodo_corte"].unique().tolist()))


def predecir(modelo: ModeloAPV, df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve `prob_aporte` (calibrada), `monto_condicional` y `monto_esperado`."""
    x = df[modelo.columnas]
    p_cruda = modelo.clf.predict_proba(x)[:, 1]
    p = modelo.calibrador.predict(p_cruda) if modelo.usa_calibracion else p_cruda
    p = np.clip(p, 1e-6, 1 - 1e-6)
    monto_cond = np.exp(modelo.reg.predict(x)) * modelo.smearing
    return pd.DataFrame({
        "afiliado_id": df["afiliado_id"].to_numpy(),
        "periodo_corte": df["periodo_corte"].to_numpy(),
        "prob_cruda": p_cruda,
        "prob_aporte": p,
        "monto_condicional": monto_cond,
        "monto_esperado": p * monto_cond,
    }, index=df.index)


# --------------------------------------------------------------------------- baselines
def regla_negocio(df: pd.DataFrame) -> np.ndarray:
    """Regla del brief: `(renta > 2 M y empleador con APVC) o aporte en 24 m`.

    Se toma la precedencia literal ("y" antes que "o"); la variante alternativa se discute
    en el README. Devuelve un vector booleano.
    """
    return (((df["rim_ref"] > UMBRAL_RENTA_REGLA) & (df["ofrece_apvc"] > 0))
            | (df["n_aportes_24m"] > 0)).to_numpy()


def baseline_tasa_base(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Baseline trivial: la tasa base de entrenamiento para todos."""
    p = float(train["y"].mean())
    monto = float(train.loc[train["y"] == 1, "monto"].mean())
    n = len(df)
    return pd.DataFrame({"prob_aporte": np.full(n, p), "monto_condicional": np.full(n, monto),
                         "monto_esperado": np.full(n, p * monto)}, index=df.index)


def baseline_regla(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Baseline de reglas de negocio, con las tasas y montos empiricos de entrenamiento."""
    flag_tr, flag = regla_negocio(train), regla_negocio(df)
    tasas, montos = {}, {}
    for v in (True, False):
        sub = train[flag_tr == v]
        tasas[v] = float(sub["y"].mean()) if len(sub) else float(train["y"].mean())
        pos = sub[sub["y"] == 1]
        montos[v] = float(pos["monto"].mean()) if len(pos) else float(
            train.loc[train["y"] == 1, "monto"].mean())
    p = np.where(flag, tasas[True], tasas[False])
    monto = np.where(flag, montos[True], montos[False])
    return pd.DataFrame({"prob_aporte": p, "monto_condicional": monto, "monto_esperado": p * monto},
                        index=df.index)


def baseline_renta(df: pd.DataFrame) -> pd.DataFrame:
    """Ordenamiento comercial habitual: contactar por renta (solo sirve como ranking)."""
    return pd.DataFrame({"score_renta": df["rim_ref"].to_numpy()}, index=df.index)


# --------------------------------------------------------------------------- interpretabilidad
def importancias(modelo: ModeloAPV) -> pd.DataFrame:
    """Importancia por ganancia de ambas etapas, normalizada a 100."""
    def tabla(booster, etapa):
        g = booster.booster_.feature_importance(importance_type="gain")
        return pd.DataFrame({"feature": modelo.columnas, "etapa": etapa,
                             "gain_pct": 100.0 * g / max(g.sum(), 1e-9)})
    return (pd.concat([tabla(modelo.clf, "clasificacion"), tabla(modelo.reg, "monto")])
            .sort_values(["etapa", "gain_pct"], ascending=[True, False]).reset_index(drop=True))


def explicar(modelo: ModeloAPV, df: pd.DataFrame, n_ejemplos: int = 5,
             n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    """Explica `n_ejemplos` casos con las contribuciones SHAP del clasificador.

    Se toman los `n_ejemplos//2 + 1` de mayor monto esperado y el resto al azar, para mostrar
    tanto el extremo que se contactaria como un caso tipico.
    """
    pred = predecir(modelo, df)
    orden = pred["monto_esperado"].sort_values(ascending=False).index
    n_top = n_ejemplos // 2 + 1
    rng = np.random.default_rng(seed)
    resto = rng.choice(df.index.difference(orden[:n_top]), size=max(n_ejemplos - n_top, 0),
                       replace=False)
    idx = list(orden[:n_top]) + list(resto)
    sub = df.loc[idx]
    contrib = modelo.clf.predict(sub[modelo.columnas], pred_contrib=True)
    filas = []
    for k, i in enumerate(idx):
        c = contrib[k][:-1]
        top = np.argsort(-np.abs(c))[:n_features]
        for rank, f in enumerate(top, start=1):
            filas.append({
                "caso": k + 1, "afiliado_id": int(sub.loc[i, "afiliado_id"]),
                "periodo_corte": int(sub.loc[i, "periodo_corte"]),
                "prob_aporte": float(pred.loc[i, "prob_aporte"]),
                "monto_esperado": float(pred.loc[i, "monto_esperado"]),
                "y_real": int(sub.loc[i, "y"]), "monto_real": float(sub.loc[i, "monto"]),
                "rank": rank, "feature": modelo.columnas[f],
                "valor": str(sub.loc[i, modelo.columnas[f]]),
                "contribucion_logit": float(c[f]),
            })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- persistencia
def guardar(modelo: ModeloAPV, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    with open(ruta / "modelo.pkl", "wb") as fh:
        pickle.dump(modelo, fh)
    (ruta / "calibracion.json").write_text(json.dumps({
        "version": modelo.version, "smearing": modelo.smearing,
        "usa_calibracion_isotonica": bool(modelo.usa_calibracion),
        "brier_valid_crudo": modelo.brier_valid_crudo,
        "brier_valid_calibrado_cv": modelo.brier_valid_calibrado,
        "tasa_base_train": modelo.tasa_base_train,
        "monto_medio_aportantes_train": modelo.monto_medio_aportantes_train,
        "cortes_train": modelo.cortes_train, "cortes_valid": modelo.cortes_valid,
        "n_arboles_clf": int(modelo.clf.best_iteration_ or modelo.clf.n_estimators),
        "n_arboles_reg": int(modelo.reg.best_iteration_ or modelo.reg.n_estimators),
        "columnas": modelo.columnas,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def cargar(ruta: str | Path) -> ModeloAPV:
    with open(Path(ruta) / "modelo.pkl", "rb") as fh:
        return pickle.load(fh)
