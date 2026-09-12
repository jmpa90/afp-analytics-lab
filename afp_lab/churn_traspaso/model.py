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

from afp_lab.rim_nowcasting.calendario import periodo_to_idx

from .features import COLUMNAS_CATEGORICAS, columnas_features

# Hiperparametros elegidos por AUC de VALIDACION entre 5 configuraciones (nunca con test):
# el test solo se mira una vez, al final.
PARAMS_LGBM = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "learning_rate": 0.03,
    "num_leaves": 48,
    "min_data_in_leaf": 60,
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


def ajustar_prior(prob: np.ndarray, tasa_objetivo: float, tol: float = 1e-9,
                  peso: np.ndarray | None = None) -> np.ndarray:
    """Desplaza el log-odds para que la media de `prob` sea `tasa_objetivo` (prior shift).

    Sirve para **anclar el nivel** de la probabilidad a un pronostico agregado del flujo mensual de
    traspasos, que es la unica forma de corregir el error de nivel cuando la tasa base del mes que
    se esta puntuando difiere de la del ultimo mes con horizonte cerrado (ver README §6 y §8: entre
    semillas la tasa base del test se mueve hasta -54 % respecto de la ventana de calibracion).
    `peso` permite anclar una media **ponderada** en vez del conteo: con `peso=saldo` el ancla deja
    `sum(p x saldo) / sum(saldo)` en `tasa_objetivo`, que es exactamente el estimando del saldo en
    riesgo. Anclar el conteo no basta: el desplazamiento uniforme del logit sube mas a los afiliados
    de probabilidad alta, que son los de saldo alto, asi que el sesgo ponderado queda peor.
    No cambia el orden del score, asi que no afecta AUC, lift ni captura.
    """
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1 - 1e-9)
    if not 0 < tasa_objetivo < 1:
        raise ValueError("tasa_objetivo debe estar en (0, 1)")
    w = np.ones_like(p) if peso is None else np.asarray(peso, dtype=float)
    w = w / max(w.sum(), 1e-12)
    z = np.log(p / (1 - p))
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        m = float(np.sum(w / (1.0 + np.exp(-(z + mid)))))
        if abs(m - tasa_objetivo) < tol:
            break
        if m < tasa_objetivo:
            lo = mid
        else:
            hi = mid
    return 1.0 / (1.0 + np.exp(-(z + mid)))


# --------------------------------------------------------------------- entrenamiento
def entrenar(df_train: pd.DataFrame, df_valid: pd.DataFrame, seed: int = 42,
             params: dict | None = None, n_arboles: int = N_ARBOLES_MAX,
             columnas: list[str] | None = None) -> dict:
    """Entrena LightGBM binario con early stopping en validacion y ajusta los dos calibradores.

    `columnas` permite pasar una lista explicita (solo la usan los diagnosticos, que necesitan una
    columna latente que `columnas_features` excluye de forma dura). Devuelve un dict con el booster,
    las columnas usadas, el calibrador de ranking (Platt) y el de agregados (isotonica por bloques).
    """
    cols = list(columnas) if columnas is not None else columnas_features(df_train)
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


# --------------------------------------------------------------------- pronostico agregado del flujo
COLUMNAS_PRONOSTICO = ("brecha_comision_pp", "brecha_rentabilidad_pp", "antiguedad_afp_meses",
                       "es_marzo", "es_agosto", "t_rel")


def serie_mensual_flujo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el dataset a una serie mensual por corte: tasa observada y regresores del mes.

    Todos los regresores son conocidos AL CORTE: brecha de comision contra la AFP mas barata, brecha
    de rentabilidad 12 m del fondo del afiliado (promedio de la cartera), antiguedad media de la
    cartera, estacionalidad de marzo y agosto, y una tendencia lineal.
    """
    g = df.groupby("periodo_corte", observed=True)
    peso = df["saldo_obligatorio"].to_numpy(dtype=float)
    aux = df[["periodo_corte", "y"]].assign(_w=peso, _wy=peso * df["y"].to_numpy(dtype=float))
    ga = aux.groupby("periodo_corte", observed=True)
    d = pd.DataFrame({
        "periodo_corte": g["y"].mean().index.astype(int),
        "n": g["y"].size().to_numpy(),
        "tasa_observada": g["y"].mean().to_numpy(),
        # tasa ponderada por saldo: la que hay que pronosticar si el objetivo es el saldo en riesgo
        "tasa_saldo_observada": (ga["_wy"].sum() / ga["_w"].sum()).to_numpy(),
        "brecha_comision_pp": g["brecha_comision_pp"].mean().to_numpy(),
        "brecha_rentabilidad_pp": g["brecha_rentabilidad_pp"].mean().to_numpy(),
        "antiguedad_afp_meses": g["antiguedad_afp_meses"].mean().to_numpy(),
    }).reset_index(drop=True)
    mes = d["periodo_corte"] % 100
    d["es_marzo"] = (mes == 3).astype(float)
    d["es_agosto"] = (mes == 8).astype(float)
    d["t_rel"] = d["periodo_corte"].map(periodo_to_idx).astype(float)
    return d


class PronosticoFlujo:
    """Pronostico agregado del flujo mensual de traspasos (backlog de la ronda 1, encargo C).

    Regresion lineal del **logit de la tasa de traspaso a 6 meses** sobre variables conocidas al
    corte. Sirve para una sola cosa: **anclar el nivel** de la probabilidad calibrada
    (`ajustar_prior`) cuando se quiere sumar `prob x saldo`. No toca el orden del score.

    Se ajusta solo con cortes de **horizonte cerrado** (train y validacion); el test nunca entra.
    """

    def __init__(self, columnas: tuple[str, ...] = COLUMNAS_PRONOSTICO,
                 objetivo: str = "saldo") -> None:
        if objetivo not in ("conteo", "saldo"):
            raise ValueError("objetivo debe ser 'conteo' o 'saldo'")
        self.objetivo = objetivo
        self.col_objetivo = "tasa_observada" if objetivo == "conteo" else "tasa_saldo_observada"
        self.columnas = tuple(columnas)
        self.coef_: np.ndarray | None = None
        self.media_: np.ndarray | None = None
        self.escala_: np.ndarray | None = None
        self.r2_: float = float("nan")
        self.mae_logit_: float = float("nan")

    def _diseno(self, d: pd.DataFrame) -> np.ndarray:
        if not self.columnas:
            return np.ones((len(d), 1))
        x = d[list(self.columnas)].to_numpy(dtype=float)
        x = (x - self.media_) / self.escala_
        return np.column_stack([np.ones(len(x)), x])

    def ajustar(self, df_cerrado: pd.DataFrame) -> "PronosticoFlujo":
        return self._ajustar_serie(serie_mensual_flujo(df_cerrado))

    def _ajustar_serie(self, d: pd.DataFrame) -> "PronosticoFlujo":
        if len(d) < len(self.columnas) + 3:
            raise ValueError(f"se necesitan al menos {len(self.columnas) + 3} cortes cerrados; hay {len(d)}")
        x0 = d[list(self.columnas)].to_numpy(dtype=float) if self.columnas else np.zeros((len(d), 0))
        self.media_ = x0.mean(axis=0) if self.columnas else np.zeros(0)
        self.escala_ = (np.where(x0.std(axis=0) > 1e-12, x0.std(axis=0), 1.0) if self.columnas
                        else np.ones(0))
        x = self._diseno(d)
        y = _logit(d[self.col_objetivo].to_numpy())
        self.coef_, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ self.coef_
        self.r2_ = float(1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        self.mae_logit_ = float(np.mean(np.abs(y - pred)))
        return self

    def predecir(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tasa pronosticada por corte (una fila por corte)."""
        if self.coef_ is None:
            raise ValueError("el pronostico no esta ajustado")
        d = serie_mensual_flujo(df)
        z = self._diseno(d) @ self.coef_
        d["tasa_pronosticada"] = 1.0 / (1.0 + np.exp(-z))
        d["tasa_objetivo_observada"] = d[self.col_objetivo]
        d["error_relativo"] = d["tasa_pronosticada"] / d[self.col_objetivo] - 1.0
        return d[["periodo_corte", "n", "tasa_observada", "tasa_saldo_observada",
                  "tasa_objetivo_observada", "tasa_pronosticada", "error_relativo"]]

    def coeficientes(self) -> pd.DataFrame:
        return pd.DataFrame({"termino": ("intercepto",) + tuple(self.columnas), "coef": self.coef_})


ESPECIFICACIONES_PRONOSTICO = {
    "estructural": ("brecha_comision_pp", "brecha_rentabilidad_pp", "antiguedad_afp_meses",
                    "es_marzo", "es_agosto"),
    "estructural+tendencia": COLUMNAS_PRONOSTICO,
    "precio+antiguedad": ("brecha_comision_pp", "antiguedad_afp_meses"),
    "constante": (),
}


def seleccionar_pronostico(df_train: pd.DataFrame, df_valid: pd.DataFrame, objetivo: str = "saldo",
                           min_cortes: int = 10) -> tuple["PronosticoFlujo", pd.DataFrame]:
    """Elige la especificacion del pronostico con un **backtest de ventana expansiva**.

    Sobre la serie mensual de los cortes con horizonte cerrado (train + validacion) se recorre el
    tiempo: para cada corte `t >= min_cortes` se ajusta con los cortes anteriores y se predice `t`.
    Eso da del orden de 20 evaluaciones fuera de muestra en vez de las 3 de la validacion, y ningun
    numero de test participa. Gana la especificacion con menor error absoluto medio relativo; la
    ganadora se reajusta con toda la historia cerrada, que es lo que se usaria en produccion.

    La especificacion "constante" (la tasa media de la historia) es el baseline honesto, y la
    tendencia lineal compite en igualdad de condiciones en vez de venir por defecto: extrapolar una
    tendencia es justamente lo que rompe el nivel.
    """
    cerrado = pd.concat([df_train, df_valid])
    serie = serie_mensual_flujo(cerrado).sort_values("periodo_corte").reset_index(drop=True)
    filas = []
    for nombre, cols in ESPECIFICACIONES_PRONOSTICO.items():
        errores = []
        for t in range(max(min_cortes, len(cols) + 4), len(serie)):
            pr = PronosticoFlujo(columnas=cols, objetivo=objetivo)
            try:
                pr._ajustar_serie(serie.iloc[:t])
            except ValueError:
                break
            z = pr._diseno(serie.iloc[[t]]) @ pr.coef_
            pred = float(1.0 / (1.0 + np.exp(-z[0])))
            real = float(serie.iloc[t][pr.col_objetivo])
            errores.append(pred / max(real, 1e-9) - 1.0)
        if not errores:
            continue
        e = np.asarray(errores)
        pr_full = PronosticoFlujo(columnas=cols, objetivo=objetivo).ajustar(cerrado)
        filas.append({
            "especificacion": nombre,
            "n_variables": len(cols),
            "n_evaluaciones_backtest": len(e),
            "r2_dentro_de_muestra": pr_full.r2_,
            "mae_relativo_backtest": float(np.mean(np.abs(e))),
            "sesgo_relativo_backtest": float(np.mean(e)),
            "p90_abs_backtest": float(np.quantile(np.abs(e), 0.90)),
        })
    if not filas:
        raise ValueError(f"no hay suficientes cortes cerrados ({len(serie)}) para el backtest del "
                         f"pronostico con min_cortes={min_cortes}")
    tabla = pd.DataFrame(filas).sort_values("mae_relativo_backtest").reset_index(drop=True)
    ganadora = tabla.loc[0, "especificacion"]
    tabla["elegida"] = tabla["especificacion"] == ganadora
    final = PronosticoFlujo(columnas=ESPECIFICACIONES_PRONOSTICO[ganadora],
                            objetivo=objetivo).ajustar(cerrado)
    return final, tabla


def anclar_con_pronostico(df: pd.DataFrame, prob: np.ndarray, pronostico: PronosticoFlujo,
                          col_peso: str = "saldo_obligatorio") -> np.ndarray:
    """Aplica `ajustar_prior` corte por corte con la tasa pronosticada de ese corte.

    Si el pronostico es de tasa ponderada por saldo (`objetivo="saldo"`), el ancla se aplica con
    `peso=saldo`, de modo que lo que queda anclado es el saldo en riesgo y no el numero de traspasos.
    """
    out = np.asarray(prob, dtype=float).copy()
    tasas = pronostico.predecir(df).set_index("periodo_corte")["tasa_pronosticada"]
    cortes = df["periodo_corte"].to_numpy()
    peso_total = df[col_peso].to_numpy(dtype=float) if pronostico.objetivo == "saldo" else None
    for corte, tasa in tasas.items():
        m = cortes == corte
        if m.any():
            w = None if peso_total is None else peso_total[m]
            out[m] = ajustar_prior(out[m], float(np.clip(tasa, 1e-6, 0.5)), peso=w)
    return out


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
