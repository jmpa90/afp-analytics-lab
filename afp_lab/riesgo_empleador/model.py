"""Modelo de riesgo de empleador y baselines obligatorios (CONVENCIONES §4).

Un `LGBMClassifier` por target (`y_dnp` principal, `y_tardio` y `y_dnp_3m` como variantes), con el
mismo protocolo que el proyecto 03 en su ronda 2:

* el numero de arboles lo elige el **early stopping** contra el `valid` del origen (con gap), no
  una constante;
* la **calibracion** isotonica se aplica solo si la mejora del Brier fuera de muestra dentro de
  validacion es significativa frente a la dispersion entre folds **y** material (>= 1 % del Brier);
* nada de test entra en ninguna decision.

Salida de negocio: `masa_en_riesgo_esperada = prob_dnp x masa_ref`, donde `masa_ref` es la masa
imponible de la **ultima planilla conocida** del empleador (la masa del periodo `corte` todavia no
existe cuando se arma la lista de cobranza). Es el score con que se ordena la cartera y el que se
agrega para proyectar la masa que no entrara.

Baselines obligatorios:

* `persistencia`: el estado de la ultima planilla conocida (deuda -> alto riesgo). Es el "modelo"
  que cobranza aplica hoy sin analitica.
* `regla_negocio`: la regla del brief, "DNP en 3 meses o en el boletin de infractores".
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

MEJORA_MINIMA_CALIBRACION = 0.01
N_SIGMAS_CALIBRACION = 2.0
TARGETS = ("y_dnp", "y_tardio", "y_dnp_3m")
TARGET_PRINCIPAL = "y_dnp"


@dataclass
class ParametrosModelo:
    """Hiperparametros. `n_jobs=2` porque la maquina de 4 nucleos es compartida.

    El dataset de empleadores es chico (unos cientos de empleadores por corte), asi que la
    regularizacion es mas fuerte que en el proyecto 03: hojas pequenas y muchas observaciones por
    hoja para no memorizar empleadores.
    """

    n_jobs: int = 2
    seed: int = 42
    learning_rate: float = 0.05
    num_leaves: int = 12
    min_child_samples: int = 150
    n_estimators: int = 3_000
    feature_fraction: float = 0.70
    bagging_fraction: float = 0.70
    lambda_l2: float = 10.0
    early_stopping: int = 100
    version: str = "riesgo-empleador-lgbm-v1"


@dataclass
class ModeloEmpleador:
    """Un clasificador por target + calibracion + metadatos de trazabilidad."""

    clfs: dict[str, lgb.LGBMClassifier]
    calibradores: dict[str, IsotonicRegression]
    usa_calibracion: dict[str, bool]
    brier_valid_crudo: dict[str, float]
    mejora_media: dict[str, float]
    mejora_sem: dict[str, float]
    columnas: list[str]
    tasas_base_train: dict[str, float]
    params: ParametrosModelo = field(default_factory=ParametrosModelo)
    cortes_train: list[int] = field(default_factory=list)
    cortes_valid: list[int] = field(default_factory=list)

    @property
    def version(self) -> str:
        return self.params.version

    def n_arboles(self, target: str = TARGET_PRINCIPAL) -> int:
        c = self.clfs[target]
        return int(c.best_iteration_ or c.n_estimators)


def _clasificador(params: ParametrosModelo) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", learning_rate=params.learning_rate, num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples, n_estimators=params.n_estimators,
        colsample_bytree=params.feature_fraction, subsample=params.bagging_fraction,
        subsample_freq=1, reg_lambda=params.lambda_l2, n_jobs=params.n_jobs,
        random_state=params.seed, verbose=-1)


def _decidir_calibracion(p_valid: np.ndarray, y_valid: np.ndarray, seed: int
                         ) -> tuple[IsotonicRegression, bool, float, float, float]:
    """Isotonica + decision (significativa entre folds y material) tomada solo en validacion."""
    calibrador = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrador.fit(p_valid, y_valid)
    brier_crudo = float(np.mean((p_valid - y_valid) ** 2))
    if y_valid.sum() < 30 or y_valid.sum() == len(y_valid):
        return calibrador, False, brier_crudo, 0.0, 0.0
    mejoras = []
    for i_tr, i_te in StratifiedKFold(3, shuffle=True, random_state=seed).split(p_valid, y_valid):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p_valid[i_tr], y_valid[i_tr])
        pc = iso.predict(p_valid[i_te])
        mejoras.append(float(np.mean((p_valid[i_te] - y_valid[i_te]) ** 2)
                             - np.mean((pc - y_valid[i_te]) ** 2)))
    media = float(np.mean(mejoras))
    sem = float(np.std(mejoras, ddof=1) / np.sqrt(len(mejoras)))
    usa = bool(media > N_SIGMAS_CALIBRACION * sem
               and media >= MEJORA_MINIMA_CALIBRACION * brier_crudo)
    return calibrador, usa, brier_crudo, media, sem


def entrenar(train: pd.DataFrame, valid: pd.DataFrame,
             params: ParametrosModelo | None = None,
             targets: tuple[str, ...] = TARGETS) -> ModeloEmpleador:
    """Entrena un clasificador por target. Todo lo que se elige se elige con `valid`."""
    params = params or ParametrosModelo()
    cols = columnas_modelo(train)
    clfs, cals, usa, briers, medias, sems, tasas = {}, {}, {}, {}, {}, {}, {}
    for t in targets:
        clf = _clasificador(params)
        clf.fit(train[cols], train[t], eval_X=valid[cols], eval_y=valid[t], eval_metric="auc",
                callbacks=[lgb.early_stopping(params.early_stopping, verbose=False)])
        p_valid = clf.predict_proba(valid[cols])[:, 1]
        cal, u, b, m, s = _decidir_calibracion(p_valid, valid[t].to_numpy(), params.seed)
        clfs[t], cals[t], usa[t], briers[t], medias[t], sems[t] = clf, cal, u, b, m, s
        tasas[t] = float(train[t].mean())
    return ModeloEmpleador(
        clfs=clfs, calibradores=cals, usa_calibracion=usa, brier_valid_crudo=briers,
        mejora_media=medias, mejora_sem=sems, columnas=cols, tasas_base_train=tasas, params=params,
        cortes_train=sorted(train["periodo_corte"].unique().tolist()),
        cortes_valid=sorted(valid["periodo_corte"].unique().tolist()))


def predecir(modelo: ModeloEmpleador, df: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad por target + `masa_en_riesgo_esperada` (score de cobranza)."""
    x = df[modelo.columnas]
    out = {"empleador_id": df["empleador_id"].to_numpy(),
           "periodo_corte": df["periodo_corte"].to_numpy()}
    for t, clf in modelo.clfs.items():
        cruda = clf.predict_proba(x)[:, 1]
        p = modelo.calibradores[t].predict(cruda) if modelo.usa_calibracion[t] else cruda
        out[f"prob_{t}"] = np.clip(p, 1e-6, 1 - 1e-6)
        out[f"prob_cruda_{t}"] = cruda
        out[f"prob_isotonica_{t}"] = np.clip(modelo.calibradores[t].predict(cruda), 1e-6, 1 - 1e-6)
    masa_ref = df["masa_ref_clp"].to_numpy()
    out["masa_ref"] = masa_ref
    out["masa_en_riesgo_esperada"] = out[f"prob_{TARGET_PRINCIPAL}"] * masa_ref
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- diagnostico
def diagnostico_memorizacion(train: pd.DataFrame, valid: pd.DataFrame, modelo: ModeloEmpleador,
                             target: str = TARGET_PRINCIPAL, n_folds: int = 3) -> pd.DataFrame:
    """Cuanto del AUC viene de memorizar empleadores (CV agrupada por `empleador_id`)."""
    cols = modelo.columnas
    p = ParametrosModelo(**{**modelo.params.__dict__, "n_estimators": modelo.n_arboles(target)})
    clf = modelo.clfs[target]
    auc_train = float(roc_auc_score(train[target], clf.predict_proba(train[cols])[:, 1]))
    auc_valid = float(roc_auc_score(valid[target], clf.predict_proba(valid[cols])[:, 1]))
    oof = np.zeros(len(train))
    grupos = train["empleador_id"].to_numpy()
    for i_tr, i_te in GroupKFold(n_folds).split(train, train[target], groups=grupos):
        m = _clasificador(p)
        m.fit(train.iloc[i_tr][cols], train.iloc[i_tr][target])
        oof[i_te] = m.predict_proba(train.iloc[i_te][cols])[:, 1]
    auc_grupo = float(roc_auc_score(train[target], oof))
    return pd.DataFrame([
        {"evaluacion": "train (en muestra)", "auc": auc_train},
        {"evaluacion": f"CV agrupada por empleador ({n_folds} folds, dentro de train)", "auc": auc_grupo},
        {"evaluacion": "validacion temporal con gap", "auc": auc_valid},
    ])


# --------------------------------------------------------------------------- baselines
def baseline_persistencia(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Estado de la ultima planilla conocida: si el empleador viene con deuda, alto riesgo.

    Se convierte en probabilidad con las tasas empiricas de entrenamiento de cada grupo (deuda en
    el ultimo mes conocido / sin deuda), de modo que el Brier sea comparable con el del modelo.
    """
    def flag(d):
        return (d["meses_desde_ult_dnp"] <= 0).to_numpy() | (d["n_faltantes_12"] > 0).to_numpy()

    return _baseline_por_grupo(train, df, flag(train), flag(df))


def regla_negocio(df: pd.DataFrame) -> np.ndarray:
    """Regla del brief: DNP en los ultimos 3 meses **o** presencia en el boletin de infractores."""
    return ((df["n_dnp_3"] > 0) | (df["en_boletin_infractores"] > 0)).to_numpy()


def baseline_regla(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return _baseline_por_grupo(train, df, regla_negocio(train), regla_negocio(df))


def _baseline_por_grupo(train: pd.DataFrame, df: pd.DataFrame, flag_tr: np.ndarray,
                        flag: np.ndarray) -> pd.DataFrame:
    out = {}
    for t in TARGETS:
        tasas = {}
        for v in (True, False):
            sub = train[flag_tr == v]
            tasas[v] = float(sub[t].mean()) if len(sub) else float(train[t].mean())
        out[f"prob_{t}"] = np.where(flag, tasas[True], tasas[False])
    masa_ref = df["masa_ref_clp"].to_numpy()
    out["masa_ref"] = masa_ref
    out["masa_en_riesgo_esperada"] = out[f"prob_{TARGET_PRINCIPAL}"] * masa_ref
    return pd.DataFrame(out, index=df.index)


def baseline_tasa_base(train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Baseline trivial: la tasa base del entrenamiento para todos (una sola constante)."""
    out = {}
    for t in TARGETS:
        out[f"prob_{t}"] = np.full(len(df), float(train[t].mean()))
    masa_ref = df["masa_ref_clp"].to_numpy()
    out["masa_ref"] = masa_ref
    out["masa_en_riesgo_esperada"] = out[f"prob_{TARGET_PRINCIPAL}"] * masa_ref
    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------- interpretabilidad
def importancias(modelo: ModeloEmpleador) -> pd.DataFrame:
    filas = []
    for t, clf in modelo.clfs.items():
        g = clf.booster_.feature_importance(importance_type="gain")
        filas.append(pd.DataFrame({"feature": modelo.columnas, "target": t,
                                   "gain_pct": 100.0 * g / max(g.sum(), 1e-9)}))
    return (pd.concat(filas).sort_values(["target", "gain_pct"], ascending=[True, False])
            .reset_index(drop=True))


def explicar(modelo: ModeloEmpleador, df: pd.DataFrame, n_ejemplos: int = 5,
             n_features: int = 5, seed: int = 42,
             target: str = TARGET_PRINCIPAL) -> pd.DataFrame:
    """Explica casos con las contribuciones SHAP del clasificador principal.

    Se toman los de mayor masa en riesgo esperada (sin repetir empleador) y el resto al azar.
    """
    pred = predecir(modelo, df)
    orden = pred["masa_en_riesgo_esperada"].sort_values(ascending=False).index
    n_top = n_ejemplos // 2 + 1
    vistos, top = set(), []
    for i in orden:
        e = int(df.loc[i, "empleador_id"])
        if e in vistos:
            continue
        vistos.add(e)
        top.append(i)
        if len(top) == n_top:
            break
    rng = np.random.default_rng(seed)
    resto = rng.choice(df.index.difference(top), size=max(n_ejemplos - n_top, 0), replace=False)
    idx = list(top) + list(resto)
    sub = df.loc[idx]
    contrib = modelo.clfs[target].predict(sub[modelo.columnas], pred_contrib=True)
    filas = []
    for k, i in enumerate(idx):
        c = contrib[k][:-1]
        for rank, f in enumerate(np.argsort(-np.abs(c))[:n_features], start=1):
            filas.append({
                "caso": k + 1, "empleador_id": int(sub.loc[i, "empleador_id"]),
                "periodo_corte": int(sub.loc[i, "periodo_corte"]),
                "rubro": str(sub.loc[i, "rubro"]), "tamano": str(sub.loc[i, "tamano_empleador"]),
                "prob_dnp": float(pred.loc[i, f"prob_{target}"]),
                "masa_en_riesgo_esperada": float(pred.loc[i, "masa_en_riesgo_esperada"]),
                "y_real": int(sub.loc[i, target]),
                "masa_en_riesgo_real": float(sub.loc[i, "masa_en_riesgo"]),
                "rank": rank, "feature": modelo.columnas[f],
                "valor": str(sub.loc[i, modelo.columnas[f]]),
                "contribucion_logit": float(c[f]),
            })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- persistencia
def guardar(modelo: ModeloEmpleador, ruta: str | Path) -> Path:
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    with open(ruta / "modelo.pkl", "wb") as fh:
        pickle.dump(modelo, fh)
    (ruta / "calibracion.json").write_text(json.dumps({
        "version": modelo.version,
        "targets": list(modelo.clfs),
        "n_arboles": {t: modelo.n_arboles(t) for t in modelo.clfs},
        "tope_arboles": modelo.params.n_estimators,
        "usa_calibracion_isotonica": modelo.usa_calibracion,
        "brier_valid_crudo": modelo.brier_valid_crudo,
        "mejora_brier_media_entre_folds": modelo.mejora_media,
        "mejora_brier_error_estandar": modelo.mejora_sem,
        "n_sigmas_exigidas": N_SIGMAS_CALIBRACION,
        "mejora_relativa_minima": MEJORA_MINIMA_CALIBRACION,
        "tasas_base_train": modelo.tasas_base_train,
        "cortes_train": modelo.cortes_train, "cortes_valid": modelo.cortes_valid,
        "columnas": modelo.columnas,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta


def cargar(ruta: str | Path) -> ModeloEmpleador:
    with open(Path(ruta) / "modelo.pkl", "rb") as fh:
        return pickle.load(fh)
