"""Modelo de propension a traspaso: LightGBM binario + calibracion isotonica, y baselines.

El score se usa para dos decisiones distintas y por eso el modelo guarda DOS calibradores, ambos
ajustados solo con el set de validacion (nunca con test):

* `uso="ranking"` -> Platt sobre el log-odds. Es estrictamente monotona, no genera empates y por lo
  tanto conserva intactos el orden, el AUC, el lift y la captura. Es la salida que ordena la cartera
  para elegir el 10 % que contacta el equipo comercial.
* `uso="agregado"` -> isotonica por bloques. Ajusta mucho mejor el NIVEL de la probabilidad (lo que
  importa al sumar `prob x saldo` para estimar el saldo en riesgo del mes), a costa de aplanar el
  score en bloques: empeora el average precision y por eso no se usa para ordenar.

Baselines obligatorios:
    `baseline_tasa_base`      trivial: la tasa base del set de entrenamiento para todos.
    `baseline_regla_negocio`  reglas simples del BRIEF: reclamo o contacto de la competencia en 3 m,
                              o brecha de rentabilidad 12 m menor que -1 pp.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .features import COLUMNAS_CATEGORICAS, columnas_features

PARAMS_LGBM = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "learning_rate": 0.05,
    "num_leaves": 48,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_cat_threshold": 24,
    "cat_smooth": 20.0,
    "num_threads": 2,          # maquina de 4 nucleos compartida
    "verbosity": -1,
    "force_row_wise": True,
}
N_ARBOLES_MAX = 900
EARLY_STOPPING = 60
N_BINS_CALIBRACION = 40    # bloques de validacion sobre los que se ajusta la isotonica
UMBRAL_RENTABILIDAD_REGLA = -1.0   # pp de brecha de rentabilidad 12m que dispara la regla de negocio


# --------------------------------------------------------------------- calibracion
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class CalibradorPlatt:
    """Regresion logistica de 1 variable sobre el log-odds del modelo (Platt scaling).

    Es estrictamente monotona y continua: corrige nivel y pendiente de la probabilidad sin
    introducir empates, asi que el ranking (y por lo tanto AUC, lift y captura) queda intacto.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a, self.b = float(a), float(b)

    def fit(self, prob: np.ndarray, y: np.ndarray) -> "CalibradorPlatt":
        x = _logit(prob).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(x, np.asarray(y, dtype=int))
        self.a, self.b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        return self

    def predict(self, prob: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(prob) + self.b)))


def calibrador_isotonico(prob: np.ndarray, y: np.ndarray,
                         n_bins: int = N_BINS_CALIBRACION) -> IsotonicRegression:
    """Isotonica ajustada sobre bloques de validacion (no sobre observaciones sueltas).

    Ajustar la isotonica fila a fila con una tasa base de ~1,5 % genera bloques minusculos y
    probabilidades saturadas en 0 y 1 en las colas. Agrupando la validacion en `n_bins` bloques de
    igual tamano por score, cada punto de la curva descansa en cientos de afiliados y la
    probabilidad maxima queda acotada por la tasa observada del bloque mas riesgoso.
    """
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    grupos = np.array_split(np.argsort(prob), max(2, min(n_bins, len(prob) // 20)))
    x = np.array([prob[g].mean() for g in grupos])
    t = np.array([y[g].mean() for g in grupos])
    w = np.array([len(g) for g in grupos], dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, t, sample_weight=w)
    return iso


# --------------------------------------------------------------------- entrenamiento
def entrenar(df_train: pd.DataFrame, df_valid: pd.DataFrame, seed: int = 42,
             params: dict | None = None, n_arboles: int = N_ARBOLES_MAX) -> dict:
    """Entrena LightGBM binario con early stopping en validacion y ajusta los dos calibradores.

    Devuelve un dict con el booster, las columnas usadas, el calibrador de ranking (Platt) y el
    calibrador de agregados (isotonica por bloques).
    """
    cols = columnas_features(df_train)
    p = dict(PARAMS_LGBM)
    if params:
        p.update(params)
    p["seed"] = seed
    p["bagging_seed"] = seed + 1
    p["feature_fraction_seed"] = seed + 2

    cat = [c for c in COLUMNAS_CATEGORICAS if c in cols]
    dtrain = lgb.Dataset(df_train[cols], label=df_train["y"].to_numpy(),
                         categorical_feature=cat, free_raw_data=False)
    dvalid = lgb.Dataset(df_valid[cols], label=df_valid["y"].to_numpy(),
                         categorical_feature=cat, reference=dtrain, free_raw_data=False)
    evals: dict = {}
    booster = lgb.train(
        p, dtrain, num_boost_round=n_arboles, valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.record_evaluation(evals)],
    )
    p_valid = booster.predict(df_valid[cols], num_iteration=booster.best_iteration)
    y_valid = df_valid["y"].to_numpy()
    cal_ranking = CalibradorPlatt().fit(p_valid, y_valid)
    cal_agregado = calibrador_isotonico(p_valid, y_valid)
    return {
        "booster": booster,
        "columnas": cols,
        "categoricas": cat,
        "calibrador": cal_ranking,
        "calibrador_agregado": cal_agregado,
        "mejor_iteracion": int(booster.best_iteration or n_arboles),
        "auc_valid": float(evals["valid"]["auc"][booster.best_iteration - 1]) if evals else float("nan"),
        "tasa_base_train": float(df_train["y"].mean()),
    }


def predecir(modelo: dict, df: pd.DataFrame, uso: str = "ranking") -> np.ndarray:
    """Probabilidad de traspaso en los proximos 6 meses.

    `uso`: "ranking" (Platt, para ordenar la cartera), "agregado" (isotonica, para sumar
    prob x saldo) o "bruto" (salida cruda del booster).
    """
    bruto = np.asarray(modelo["booster"].predict(df[modelo["columnas"]],
                                                 num_iteration=modelo["mejor_iteracion"]), dtype=float)
    if uso == "bruto":
        return bruto
    if uso == "ranking":
        return np.asarray(modelo["calibrador"].predict(bruto), dtype=float)
    if uso == "agregado":
        return np.asarray(modelo["calibrador_agregado"].predict(bruto), dtype=float)
    raise ValueError(f"uso desconocido: {uso}")


def guardar_modelo(modelo: dict, directorio: str | Path) -> Path:
    """Persiste el booster (texto), el calibrador (puntos) y los metadatos (json)."""
    d = Path(directorio)
    d.mkdir(parents=True, exist_ok=True)
    modelo["booster"].save_model(str(d / "lightgbm.txt"), num_iteration=modelo["mejor_iteracion"])
    cal, iso = modelo["calibrador"], modelo["calibrador_agregado"]
    pd.DataFrame([{"a": cal.a, "b": cal.b}]).to_csv(d / "calibrador_platt.csv", index=False)
    pd.DataFrame({"x": iso.X_thresholds_, "y": iso.y_thresholds_}).to_csv(d / "calibrador_isotonico.csv",
                                                                         index=False)
    meta = {k: modelo[k] for k in ("columnas", "categoricas", "mejor_iteracion", "auc_valid",
                                   "tasa_base_train")}
    (d / "modelo.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def cargar_modelo(directorio: str | Path) -> dict:
    d = Path(directorio)
    meta = json.loads((d / "modelo.json").read_text(encoding="utf-8"))
    platt = pd.read_csv(d / "calibrador_platt.csv")
    cal = CalibradorPlatt(float(platt["a"][0]), float(platt["b"][0]))
    tab = pd.read_csv(d / "calibrador_isotonico.csv")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(tab["x"].to_numpy(), tab["y"].to_numpy())
    return {"booster": lgb.Booster(model_file=str(d / "lightgbm.txt")), "calibrador": cal,
            "calibrador_agregado": iso, **meta}


# --------------------------------------------------------------------- baselines
def baseline_tasa_base(df_train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Baseline trivial: la tasa base observada en entrenamiento para todos los afiliados."""
    return np.full(len(df), float(df_train["y"].mean()), dtype=float)


def _condiciones_regla(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reclamo = (df["reclamos_6m"].to_numpy() > 0) & (df["meses_desde_reclamo"].to_numpy() <= 3)
    competencia = df["contactos_vendedor_competencia_3m"].to_numpy() > 0
    rentabilidad = df["brecha_rentabilidad_pp"].to_numpy() < UMBRAL_RENTABILIDAD_REGLA
    return reclamo, competencia, rentabilidad


def baseline_regla_negocio(df: pd.DataFrame) -> np.ndarray:
    """Regla del BRIEF: reclamo o contacto de la competencia en 3 m, o brecha de rentabilidad < -1 pp."""
    reclamo, competencia, rentabilidad = _condiciones_regla(df)
    return (reclamo | competencia | rentabilidad).astype(float)


def baseline_regla_graduada(df: pd.DataFrame) -> np.ndarray:
    """Variante mas exigente del baseline de reglas: cuenta cuantas condiciones se cumplen (0-3).

    Rompe los empates de la regla binaria y por lo tanto ordena mejor la cartera; se reporta para no
    comparar el modelo contra un baseline artificialmente debil.
    """
    reclamo, competencia, rentabilidad = _condiciones_regla(df)
    return (reclamo.astype(float) + competencia.astype(float) + rentabilidad.astype(float)) / 3.0


# --------------------------------------------------------------------- interpretabilidad
def importancias(modelo: dict, top: int = 25) -> pd.DataFrame:
    """Importancia por ganancia (gain) de las features del modelo."""
    b = modelo["booster"]
    imp = pd.DataFrame({
        "feature": b.feature_name(),
        "gain": b.feature_importance(importance_type="gain"),
        "split": b.feature_importance(importance_type="split"),
    })
    imp["gain_pct"] = 100 * imp["gain"] / max(imp["gain"].sum(), 1e-9)
    return imp.sort_values("gain", ascending=False).head(top).reset_index(drop=True)


def seleccionar_casos(df: pd.DataFrame, prob: np.ndarray) -> list[int]:
    """Elige casos representativos para explicar (posiciones en `df`).

    No sirve de nada explicar los 5 de mayor score: son el mismo perfil repetido. Se toman los dos
    de mayor score, el falso positivo mas caro, el verdadero positivo con mas saldo y el falso
    negativo de menor score, que es donde se ve que le falta al modelo.
    """
    y = df["y"].to_numpy()
    saldo = df["saldo_obligatorio"].to_numpy(dtype=float)
    orden = np.argsort(-prob)
    sel: list[int] = [int(i) for i in orden[:2]]

    def agregar(cands: np.ndarray) -> None:
        for i in cands:
            if int(i) not in sel:
                sel.append(int(i))
                return

    k = max(1, len(df) // 10)
    top = orden[:k]
    agregar(top[y[top] == 0])                                     # falso positivo de mayor score
    agregar(top[np.argsort(-np.where(y[top] == 1, saldo[top], -1))])  # positivo con mas saldo
    agregar(orden[::-1][y[orden[::-1]] == 1])                     # falso negativo (score mas bajo)
    return sel[:5]


def explicar_casos(modelo: dict, df: pd.DataFrame, prob: np.ndarray, n_casos: int = 5,
                   n_features: int = 6) -> pd.DataFrame:
    """Explica casos representativos con las contribuciones SHAP del booster (`pred_contrib`)."""
    sel = seleccionar_casos(df, np.asarray(prob, dtype=float))[:n_casos]
    etiquetas = ["top_1", "top_2", "falso_positivo", "positivo_mayor_saldo", "falso_negativo"]
    sub = df.iloc[sel]
    cols = modelo["columnas"]
    contrib = modelo["booster"].predict(sub[cols], num_iteration=modelo["mejor_iteracion"],
                                        pred_contrib=True)
    filas = []
    for k, (_, fila) in enumerate(sub.iterrows()):
        c = contrib[k][:-1]
        top = np.argsort(-np.abs(c))[:n_features]
        for rank, j in enumerate(top, start=1):
            filas.append({
                "caso": etiquetas[k] if k < len(etiquetas) else f"caso_{k + 1}",
                "afiliado_id": int(fila["afiliado_id"]),
                "periodo_corte": int(fila["periodo_corte"]),
                "prob": float(prob[sel[k]]),
                "y_real": int(fila["y"]),
                "saldo_obligatorio": float(fila["saldo_obligatorio"]),
                "rank": rank,
                "feature": cols[j],
                "valor": str(fila[cols[j]]),
                "contribucion_logit": float(c[j]),
            })
    return pd.DataFrame(filas)
