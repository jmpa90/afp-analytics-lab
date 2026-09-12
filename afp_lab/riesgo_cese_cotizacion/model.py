"""Modelo de clasificacion binaria (LightGBM) para cada uno de los tres targets del
proyecto (`y3`, `y6`, `y_reingreso`), con calibracion isotonica sobre un conjunto de
validacion separado del de ajuste, y los baselines obligatorios de CONVENCIONES.md #4.

Un modelo por target: aunque las tres tareas comparten casi el mismo set de features, las
poblaciones (cotizantes vs. no-cotizantes) y las tasas base son muy distintas, y el brief
pide reportar cada horizonte por separado; un unico modelo multi-tarea no simplifica el
pipeline y complica la calibracion por horizonte, asi que se opto por 3 modelos
independientes (mismo tipo, mismos hiperparametros salvo lo que LightGBM ajusta solo).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .features import FEATURES_CATEGORICAS, FEATURES_NUMERICAS

COLUMNAS = list(FEATURES_NUMERICAS) + list(FEATURES_CATEGORICAS)

# n_jobs=2: maquina de 4 nucleos compartida con otros dos data scientists.
PARAMS_LGBM = dict(
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=2,
    random_state=42,
    verbose=-1,
)


def _preparar_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[COLUMNAS].copy()
    for c in FEATURES_CATEGORICAS:
        X[c] = X[c].astype("category")
    return X


@dataclass
class ModeloCese:
    """Clasificador binario + calibracion isotonica para un target (`y3`, `y6` o
    `y_reingreso`)."""
    target: str
    clasificador: lgb.LGBMClassifier
    calibrador: IsotonicRegression
    columnas: list[str] = field(default_factory=lambda: list(COLUMNAS))

    @classmethod
    def fit(cls, train: pd.DataFrame, val: pd.DataFrame, target: str,
           params: dict | None = None) -> "ModeloCese":
        p = dict(PARAMS_LGBM)
        if params:
            p.update(params)
        clf = lgb.LGBMClassifier(**p)
        clf.fit(_preparar_X(train), train[target].to_numpy(), categorical_feature=FEATURES_CATEGORICAS)

        raw_val = clf.predict_proba(_preparar_X(val))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_val, val[target].to_numpy())
        return cls(target=target, clasificador=clf, calibrador=iso)

    def predict_proba_cruda(self, df: pd.DataFrame) -> np.ndarray:
        return self.clasificador.predict_proba(_preparar_X(df))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidad calibrada (isotonica sobre validacion): la que se agrega."""
        return self.calibrador.predict(self.predict_proba_cruda(df))

    def importancias(self, top: int = 20) -> pd.DataFrame:
        gain = self.clasificador.booster_.feature_importance(importance_type="gain")
        df = pd.DataFrame({"feature": self.columnas, "ganancia": gain})
        df["ganancia_pct"] = 100 * df["ganancia"] / df["ganancia"].sum()
        return df.sort_values("ganancia", ascending=False).head(top).reset_index(drop=True)

    def save(self, ruta: str) -> None:
        os.makedirs(ruta, exist_ok=True)
        joblib.dump(self, f"{ruta}/modelo_{self.target}.joblib")

    @classmethod
    def load(cls, ruta: str, target: str) -> "ModeloCese":
        return joblib.load(f"{ruta}/modelo_{target}.joblib")


# --------------------------------------------------------------- baselines ----
def baseline_tasa_base(train: pd.DataFrame, target: str,
                       cols: list[str] = ("rubro", "mes_corte")) -> tuple[pd.Series, float]:
    """Tasa base historica (train) por `cols` (por defecto rubro x mes_corte); respaldo =
    tasa global de train. `cols` es parametrizable (ronda 3, mayor A) para poder proyectar
    tambien a independientes/voluntarios, que no tienen rubro (siempre "SIN_EMPLEADOR"): ahi
    se agrupa por `tipo_afiliado x mes_corte` en su lugar (ver `run._negocio_agregado_cartera_completa`)."""
    cols = list(cols)
    tasas = train.groupby(cols)[target].mean()
    return tasas, float(train[target].mean())


def predecir_tasa_base(tasas: pd.Series, tasa_global: float, df: pd.DataFrame,
                       cols: list[str] = ("rubro", "mes_corte")) -> np.ndarray:
    cols = list(cols)
    idx = pd.MultiIndex.from_arrays([df[c] for c in cols])
    return tasas.reindex(idx).fillna(tasa_global).to_numpy()


def predecir_regla_negocio_cese(df: pd.DataFrame) -> np.ndarray:
    """Regla de negocio simple (brief): "termino AFC reciente o contrato a plazo con menos de
    6 meses". Devuelve un puntaje en {0, 0.5, 1} (1 si hay termino AFC muy reciente, 0.5 si
    solo aplica la condicion de contrato a plazo, para poder ordenar dentro de la regla)."""
    termino_reciente = (df["afc_termino_l1"] > 0) | (df["afc_termino_l2"] > 0)
    plazo_corto = (df["tipo_contrato"] == "PLAZO_FIJO") & (df["meses_con_empleador"] < 6)
    return np.where(termino_reciente, 1.0, np.where(plazo_corto, 0.5, 0.0))


def predecir_regla_negocio_reingreso(df: pd.DataFrame) -> np.ndarray:
    """Regla de negocio para reingreso: mientras mas reciente el ultimo aviso de inicio AFC
    tras el cese (o mientras mas corta la racha de ceros), mas probable el reingreso pronto;
    se usa 1/(1+meses_racha_cero) como puntaje simple y monotono."""
    return 1.0 / (1.0 + df["racha_cero"].clip(lower=0))
