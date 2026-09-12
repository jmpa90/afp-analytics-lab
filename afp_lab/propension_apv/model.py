"""Modelo *hurdle* de propension a ahorro voluntario y baselines obligatorios (CONVENCIONES §4).

Dos etapas, ambas LightGBM (pandas + LightGBM + scikit-learn; la escala no exige Spark):

1. **Clasificador binario**: `p = P(al menos un aporte en corte+1..corte+6)`.
2. **Regresor** sobre `log(monto / rim_ref)` en las filas con aporte. Se modela el **ratio** y no
   el nivel para que el modelo aprenda "cuantos sueldos ahorra" y no un monto nominal: el
   entrenamiento esta 3-4 anos antes del test. `monto_condicional = rim_ref x exp(y) x s`, con
   *smearing* `s = sum(monto) / sum(rim_ref x exp(y))` **calculado en validacion** (con gap), no
   en entrenamiento, porque el regresor esta ajustado sobre train.

Salida por afiliado: `prob_aporte`, `monto_condicional` y
`monto_esperado = prob_aporte x monto_condicional`, que es el score de priorizacion.

**Protocolo (ronda 2)**: todos los hiperparametros que se eligen (numero de arboles por early
stopping, si calibrar o no, el factor de smearing) se deciden **solo con el set de validacion**,
que esta separado de entrenamiento por el mismo gap de `h` meses que lo separa de test
(CONVENCIONES §3). El numero de arboles no esta fijado a mano: se deja un tope alto y corta el
early stopping.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .features import columnas_modelo

UMBRAL_RENTA_REGLA = 2_000_000.0     # "renta > 2 M" del brief
# Criterio para aplicar la isotonica (revision ronda 2, m-8). Dos condiciones, ambas medidas
# dentro de validacion con 3 folds fuera de muestra:
#   (a) significancia: la mejora media del Brier entre folds debe superar `N_SIGMAS_CALIBRACION`
#       errores estandar de la dispersion ENTRE folds (si la mejora es del orden del ruido de
#       fold a fold, no hay nada que corregir);
#   (b) materialidad: ademas debe ser al menos `MEJORA_MINIMA_CALIBRACION` del Brier crudo, para
#       no deformar la probabilidad por una ganancia irrelevante para el negocio.
MEJORA_MINIMA_CALIBRACION = 0.01
N_SIGMAS_CALIBRACION = 2.0
LIMITES_LOG_RATIO = (-4.0, 4.0)      # winsorizacion del target del regresor


@dataclass
class ParametrosModelo:
    """Hiperparametros. `n_jobs=2` porque la maquina de 4 nucleos es compartida.

    `*_n_estimators` son TOPES: el numero efectivo de arboles lo elige el early stopping contra
    el set de validacion con gap. La regularizacion (hojas chicas, hojas con muchas
    observaciones, `lambda_l2`) esta puesta para frenar la memorizacion por afiliado.
    """

    n_jobs: int = 2
    seed: int = 42
    # clasificador
    clf_learning_rate: float = 0.05
    clf_num_leaves: int = 16
    clf_min_child_samples: int = 400
    clf_n_estimators: int = 3_000
    clf_feature_fraction: float = 0.70
    clf_bagging_fraction: float = 0.70
    clf_lambda_l2: float = 10.0
    # regresor de log(monto / rim_ref)
    reg_learning_rate: float = 0.05
    reg_num_leaves: int = 12
    reg_min_child_samples: int = 150
    reg_n_estimators: int = 2_000
    reg_feature_fraction: float = 0.70
    reg_bagging_fraction: float = 0.70
    reg_lambda_l2: float = 10.0
    early_stopping: int = 100
    version: str = "apv-hurdle-lgbm-v2"


@dataclass
class ModeloAPV:
    """Modelo entrenado + calibracion + metadatos de trazabilidad."""

    clf: lgb.LGBMClassifier
    reg: lgb.LGBMRegressor
    calibrador: IsotonicRegression
    usa_calibracion: bool
    brier_valid_crudo: float
    brier_valid_calibrado: float
    mejora_brier_media: float
    mejora_brier_sem: float
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

    @property
    def n_arboles(self) -> tuple[int, int]:
        return (int(self.clf.best_iteration_ or self.clf.n_estimators),
                int(self.reg.best_iteration_ or self.reg.n_estimators))


# --------------------------------------------------------------------------- entrenamiento
def _clasificador(params: ParametrosModelo) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", learning_rate=params.clf_learning_rate,
        num_leaves=params.clf_num_leaves, min_child_samples=params.clf_min_child_samples,
        n_estimators=params.clf_n_estimators, colsample_bytree=params.clf_feature_fraction,
        subsample=params.clf_bagging_fraction, subsample_freq=1, reg_lambda=params.clf_lambda_l2,
        n_jobs=params.n_jobs, random_state=params.seed, verbose=-1)


def _objetivo_regresion(df: pd.DataFrame) -> np.ndarray:
    """`log(monto / rim_ref)` winsorizado: el regresor aprende ratios, no niveles nominales."""
    ratio = df["monto"].to_numpy() / np.maximum(df["rim_ref_clp"].to_numpy(), 1.0)
    return np.clip(np.log(np.maximum(ratio, 1e-6)), *LIMITES_LOG_RATIO)


def entrenar(train: pd.DataFrame, valid: pd.DataFrame,
             params: ParametrosModelo | None = None) -> ModeloAPV:
    """Entrena el hurdle. Early stopping, smearing y calibracion se deciden solo con `valid`."""
    params = params or ParametrosModelo()
    cols = columnas_modelo(train)

    clf = _clasificador(params)
    clf.fit(train[cols], train["y"], eval_X=valid[cols], eval_y=valid["y"], eval_metric="auc",
            callbacks=[lgb.early_stopping(params.early_stopping, verbose=False)])

    # --- calibracion: se decide en validacion, con 3 folds fuera de muestra ---------------
    p_valid = clf.predict_proba(valid[cols])[:, 1]
    y_valid = valid["y"].to_numpy()
    calibrador = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrador.fit(p_valid, y_valid)
    p_cv = np.empty_like(p_valid)
    mejoras = []
    for i_tr, i_te in StratifiedKFold(3, shuffle=True, random_state=params.seed).split(p_valid, y_valid):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p_valid[i_tr], y_valid[i_tr])
        p_cv[i_te] = iso.predict(p_valid[i_te])
        mejoras.append(float(np.mean((p_valid[i_te] - y_valid[i_te]) ** 2)
                             - np.mean((p_cv[i_te] - y_valid[i_te]) ** 2)))
    brier_crudo = float(np.mean((p_valid - y_valid) ** 2))
    brier_cal = float(np.mean((p_cv - y_valid) ** 2))
    mejora_media = float(np.mean(mejoras))
    mejora_sem = float(np.std(mejoras, ddof=1) / np.sqrt(len(mejoras))) if len(mejoras) > 1 else 0.0
    # (a) significativa frente a la dispersion entre folds y (b) material frente al Brier crudo
    usa_calibracion = bool(mejora_media > N_SIGMAS_CALIBRACION * mejora_sem
                           and mejora_media >= MEJORA_MINIMA_CALIBRACION * brier_crudo)

    # --- regresor de log(monto / rim_ref) en aportantes -----------------------------------
    tr_pos, va_pos = train[train["y"] == 1], valid[valid["y"] == 1]
    reg = lgb.LGBMRegressor(
        objective="regression", learning_rate=params.reg_learning_rate,
        num_leaves=params.reg_num_leaves, min_child_samples=params.reg_min_child_samples,
        n_estimators=params.reg_n_estimators, colsample_bytree=params.reg_feature_fraction,
        subsample=params.reg_bagging_fraction, subsample_freq=1, reg_lambda=params.reg_lambda_l2,
        n_jobs=params.n_jobs, random_state=params.seed, verbose=-1)
    reg.fit(tr_pos[cols], _objetivo_regresion(tr_pos),
            eval_X=va_pos[cols], eval_y=_objetivo_regresion(va_pos), eval_metric="l2",
            callbacks=[lgb.early_stopping(params.early_stopping, verbose=False)])

    # smearing en VALIDACION (con gap), no en train: el regresor esta ajustado sobre train
    pred_va = va_pos["rim_ref_clp"].to_numpy() * np.exp(reg.predict(va_pos[cols]))
    smearing = float(va_pos["monto"].sum() / max(pred_va.sum(), 1e-9))

    return ModeloAPV(
        clf=clf, reg=reg, calibrador=calibrador, usa_calibracion=usa_calibracion,
        brier_valid_crudo=brier_crudo, brier_valid_calibrado=brier_cal,
        mejora_brier_media=mejora_media, mejora_brier_sem=mejora_sem,
        columnas=cols, smearing=smearing,
        tasa_base_train=float(train["y"].mean()),
        monto_medio_aportantes_train=float(train.loc[train["y"] == 1, "monto"].mean()),
        params=params,
        cortes_train=sorted(train["periodo_corte"].unique().tolist()),
        cortes_valid=sorted(valid["periodo_corte"].unique().tolist()))


def predecir(modelo: ModeloAPV, df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve `prob_aporte`, `monto_condicional` y `monto_esperado`."""
    x = df[modelo.columnas]
    p_cruda = modelo.clf.predict_proba(x)[:, 1]
    p = modelo.calibrador.predict(p_cruda) if modelo.usa_calibracion else p_cruda
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ratio = np.exp(np.clip(modelo.reg.predict(x), *LIMITES_LOG_RATIO))
    monto_cond = np.maximum(df["rim_ref_clp"].to_numpy(), 1.0) * ratio * modelo.smearing
    return pd.DataFrame({
        "afiliado_id": df["afiliado_id"].to_numpy(),
        "periodo_corte": df["periodo_corte"].to_numpy(),
        "prob_cruda": p_cruda,
        # prob_isotonica: lo que daria ESTE origen si se calibrara (referencia, no se usa salvo
        # que `usa_calibracion` sea True). Se calcula por origen para no aplicar el calibrador
        # de un origen a un bloque de test que no le corresponde (revision ronda 2, m-7).
        "prob_isotonica": np.clip(modelo.calibrador.predict(p_cruda), 1e-6, 1 - 1e-6),
        "prob_aporte": p,
        "monto_condicional": monto_cond,
        "monto_esperado": p * monto_cond,
    }, index=df.index)


def columnas_modelo_de(modelo: ModeloAPV) -> list[str]:
    """Columnas que usa el modelo (atajo para los reportes)."""
    return list(modelo.columnas)


# --------------------------------------------------------------------------- diagnostico
def diagnostico_memorizacion(train: pd.DataFrame, valid: pd.DataFrame, modelo: ModeloAPV,
                             n_folds: int = 3) -> pd.DataFrame:
    """Cuanto del AUC viene de memorizar afiliados.

    Compara: (a) AUC dentro de entrenamiento, (b) AUC en validacion temporal con gap y
    (c) AUC de una validacion cruzada **agrupada por afiliado** dentro de entrenamiento
    (`GroupKFold`: ningun afiliado aparece a la vez en el fold de ajuste y en el de evaluacion).
    Si (a) >> (c), el modelo esta memorizando afiliados y no aprendiendo el mecanismo.
    """
    cols = modelo.columnas
    n_arboles = modelo.n_arboles[0]
    p = ParametrosModelo(**{**modelo.params.__dict__, "clf_n_estimators": n_arboles})

    auc_train = float(roc_auc_score(train["y"], modelo.clf.predict_proba(train[cols])[:, 1]))
    auc_valid = float(roc_auc_score(valid["y"], modelo.clf.predict_proba(valid[cols])[:, 1]))

    oof = np.zeros(len(train))
    grupos = train["afiliado_id"].to_numpy()
    for i_tr, i_te in GroupKFold(n_folds).split(train, train["y"], groups=grupos):
        m = _clasificador(p)
        m.fit(train.iloc[i_tr][cols], train.iloc[i_tr]["y"])
        oof[i_te] = m.predict_proba(train.iloc[i_te][cols])[:, 1]
    auc_grupo = float(roc_auc_score(train["y"], oof))

    return pd.DataFrame([
        {"evaluacion": "train (en muestra)", "auc": auc_train},
        {"evaluacion": f"CV agrupada por afiliado ({n_folds} folds, dentro de train)", "auc": auc_grupo},
        {"evaluacion": "validacion temporal con gap", "auc": auc_valid},
    ])


# --------------------------------------------------------------------------- baselines
def regla_negocio(df: pd.DataFrame) -> np.ndarray:
    """Regla del brief: `(renta > 2 M y empleador con APVC) o aporte en 24 m`.

    Se toma la precedencia literal ("y" antes que "o"); la variante alternativa se discute
    en el README. Devuelve un vector booleano.
    """
    return (((df["rim_ref_clp"] > UMBRAL_RENTA_REGLA) & (df["ofrece_apvc"] > 0))
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
    return pd.DataFrame({"score_renta": df["rim_ref_clp"].to_numpy()}, index=df.index)


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
    # un afiliado aparece en varios cortes: se toma solo su mejor fila para no repetir el caso
    vistos, top = set(), []
    for i in orden:
        a = int(df.loc[i, "afiliado_id"])
        if a in vistos:
            continue
        vistos.add(a)
        top.append(i)
        if len(top) == n_top:
            break
    rng = np.random.default_rng(seed)
    resto = rng.choice(df.index.difference(top), size=max(n_ejemplos - n_top, 0), replace=False)
    idx = list(top) + list(resto)
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
    n_clf, n_reg = modelo.n_arboles
    (ruta / "calibracion.json").write_text(json.dumps({
        "version": modelo.version, "smearing_valid": modelo.smearing,
        "tasa_base_train": modelo.tasa_base_train,
        "monto_medio_aportantes_train": modelo.monto_medio_aportantes_train,
        "usa_calibracion_isotonica": bool(modelo.usa_calibracion),
        "brier_valid_crudo": modelo.brier_valid_crudo,
        "brier_valid_calibrado_cv": modelo.brier_valid_calibrado,
        "mejora_brier_media_entre_folds": modelo.mejora_brier_media,
        "mejora_brier_error_estandar": modelo.mejora_brier_sem,
        "n_sigmas_exigidas": N_SIGMAS_CALIBRACION,
        "mejora_relativa_minima": MEJORA_MINIMA_CALIBRACION,
        "cortes_train": modelo.cortes_train, "cortes_valid": modelo.cortes_valid,
        "n_arboles_clf": n_clf, "tope_arboles_clf": modelo.params.clf_n_estimators,
        "n_arboles_reg": n_reg, "tope_arboles_reg": modelo.params.reg_n_estimators,
        "columnas": modelo.columnas,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def cargar(ruta: str | Path) -> ModeloAPV:
    with open(Path(ruta) / "modelo.pkl", "rb") as fh:
        return pickle.load(fh)
