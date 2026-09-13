"""Evaluacion del uplift: Qini/AUUC en el holdout aleatorio, correlacion vs `lat_cate`, overlap y
la metrica de negocio (traspasos evitados contactando al 10 %/20 % de la cartera).

Qini y AUUC solo tienen sentido con asignacion de tratamiento NO confundida (si no, un score que
solo replica el riesgo ya "gana" porque los tratados y los control no son comparables): por eso
todas las funciones de esta seccion se llaman SIEMPRE sobre el holdout aleatorio, nunca sobre la
poblacion confundida.

Convencion de signo: la formula clasica de Qini (Radcliffe/Surry) supone un resultado DESEABLE
(compra, conversion) que el tratamiento AUMENTA, y un ranking donde "mayor score = mas se beneficia".
Aqui `y` es traspaso (resultado INDESEABLE) y lo que se quiere es identificar a quien el contacto le
REDUCE `y`; ordenar por un score de "prioridad de contacto" (mayor = mas conviene contactar, ver
`run.puntuar`/`SIGNO_PRIORIDAD_CONTACTO`) y aplicar la formula cruda sobre `y` da un numero NEGATIVO
para un buen modelo (el top-k tratado tiene MENOS traspasos que el control, no mas). Para que
"Qini > 0" siga significando "mejor que el azar" (criterio del brief) las funciones de esta seccion
devuelven la formula aplicada a la RETENCION (`1-y`), que es exactamente el negativo de aplicarla a
`y`: `gain_retencion(k) = -[Y1(k) - Y0(k)*N1(k)/N0(k)]` (la resta de conteos se cancela porque
`N1(k) = ratio*N0(k)` por construccion). El signo de entrada sigue siendo el de `y` (traspaso);
el negativo se aplica INTERNAMENTE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

N_BOOT_REFERENCIA = 150   # remuestreos del bootstrap de Qini (ronda 4: velocidad, igual orden que 02)


# --------------------------------------------------------------------- Qini / AUUC
def curva_qini(y: np.ndarray, T: np.ndarray, score: np.ndarray, n_puntos: int = 20) -> pd.DataFrame:
    """Curva de ganancia Qini (en traspasos EVITADOS, positivo = bueno; ver convencion de signo arriba).

    Ordena por `score` descendente ("prioridad de contacto"): `Y1(k)`/`Y0(k)` son los traspasos
    acumulados entre tratados/control en el top `k`, `N1(k)`/`N0(k)` su conteo.
    """
    y, T, score = np.asarray(y, float), np.asarray(T, float), np.asarray(score, float)
    orden = np.argsort(-score)
    y, T = y[orden], T[orden]
    n = len(y)
    n1 = np.cumsum(T)
    n0 = np.cumsum(1.0 - T)
    y1 = np.cumsum(np.where(T == 1, y, 0.0))
    y0 = np.cumsum(np.where(T == 0, y, 0.0))
    ratio = np.divide(n1, n0, out=np.zeros(n), where=n0 > 0)
    g = -(y1 - y0 * ratio)                    # traspasos EVITADOS (positivo = bueno; ver docstring del modulo)
    idx = np.unique(np.clip(np.round(np.linspace(1, n, min(n_puntos, n))).astype(int) - 1, 0, n - 1))
    return pd.DataFrame({"frac_contactada": (idx + 1) / n, "n_acumulado": idx + 1, "ganancia_qini": g[idx]})


def qini_auuc(y: np.ndarray, T: np.ndarray, score: np.ndarray) -> dict:
    """Coeficiente de Qini (area del modelo menos area de la referencia aleatoria) y AUUC.

    Ambos normalizados por `n` (promedio de la ganancia por fila) para que sean comparables entre
    cortes de distinto tamano. `qini_coeficiente > 0` significa que el score ordena mejor que un
    ranking aleatorio; es el criterio de aceptacion del brief.
    """
    y, T, score = np.asarray(y, float), np.asarray(T, float), np.asarray(score, float)
    n = len(y)
    if n == 0 or T.sum() == 0 or (n - T.sum()) == 0:
        return {"auuc": float("nan"), "qini_coeficiente": float("nan"), "ganancia_total": float("nan"), "n": n}
    orden = np.argsort(-score)
    y, T = y[orden], T[orden]
    n1 = np.cumsum(T)
    n0 = np.cumsum(1.0 - T)
    y1 = np.cumsum(np.where(T == 1, y, 0.0))
    y0 = np.cumsum(np.where(T == 0, y, 0.0))
    ratio = np.divide(n1, n0, out=np.zeros(n), where=n0 > 0)
    g = -(y1 - y0 * ratio)                    # traspasos EVITADOS (positivo = bueno; ver docstring del modulo)
    x = np.arange(1, n + 1) / n
    _trapecio = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy >=2.0 renombro trapz
    auuc = float(_trapecio(g, x) / n)
    g_random = x * g[-1]                      # referencia: ganancia proporcional a la fraccion contactada
    auuc_random = float(_trapecio(g_random, x) / n)
    return {"auuc": auuc, "qini_coeficiente": auuc - auuc_random, "ganancia_total": float(g[-1]), "n": n}


def bootstrap_qini(df_holdout: pd.DataFrame, score_col: str, n_boot: int = N_BOOT_REFERENCIA,
                   seed: int = 42, alpha: float = 0.05) -> dict:
    """Intervalo bootstrap percentil (remuestreo con reemplazo) del coeficiente de Qini."""
    y = df_holdout["y"].to_numpy()
    T = df_holdout["T"].to_numpy()
    score = df_holdout[score_col].to_numpy()
    punto = qini_auuc(y, T, score)
    n = len(df_holdout)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = qini_auuc(y[idx], T[idx], score[idx])["qini_coeficiente"]
    validos = boot[~np.isnan(boot)]
    if len(validos) >= 2:
        lo, hi = (float(x) for x in np.percentile(validos, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    else:
        lo = hi = float("nan")
    return {**punto, "ic_low": lo, "ic_high": hi, "n_boot_validos": int(len(validos))}


def qini_por_corte(df_holdout: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Qini/AUUC del holdout aleatorio, por corte de test (criterio de aceptacion: > 0 en TODOS)."""
    filas = []
    for corte, g in df_holdout.groupby("periodo_corte", observed=True):
        filas.append({"periodo_corte": int(corte), "n": len(g),
                      **qini_auuc(g["y"].to_numpy(), g["T"].to_numpy(), g[score_col].to_numpy())})
    return pd.DataFrame(filas).sort_values("periodo_corte").reset_index(drop=True)


# --------------------------------------------------------------------- overlap
def overlap_propension(e_hat: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> float:
    """Fraccion de filas con propension estimada dentro de `[lo, hi]` (criterio de aceptacion >= 80 %)."""
    e = np.asarray(e_hat, dtype=float)
    return float(((e >= lo) & (e <= hi)).mean())


# --------------------------------------------------------------------- correlacion vs verdad sintetica
def correlacion_cate(lat_cate: np.ndarray, tau_hat: np.ndarray) -> dict:
    """Spearman y MAE entre el CATE estimado y el verdadero (`lat_cate`, tabla latente)."""
    lat_cate, tau_hat = np.asarray(lat_cate, dtype=float), np.asarray(tau_hat, dtype=float)
    rho, p = spearmanr(lat_cate, tau_hat)
    mae = float(np.mean(np.abs(tau_hat - lat_cate)))
    return {"spearman": float(rho), "p_valor": float(p), "mae": mae}


# --------------------------------------------------------------------- metrica de negocio
def traspasos_evitados(latente: pd.DataFrame, score: np.ndarray,
                       fracciones: tuple[float, ...] = (0.10, 0.20)) -> pd.DataFrame:
    """Traspasos evitados contactando al `frac` de mayor `score`, usando la verdad `lat_cate`.

    `traspasos_evitados = -sum(lat_cate)` entre los contactados: `lat_cate` es el efecto verdadero
    del contacto sobre `P(traspaso)` para esa fila, asi que negativo (persuadible) resta y positivo
    (sleeping dog) resta traspasos evitados (los aumenta). Solo valido porque `lat_cate` es la verdad
    del simulador: en produccion esto se estimaria con un piloto aleatorio real.
    """
    lat_cate = latente["lat_cate"].to_numpy()
    segmento = latente["segmento"].to_numpy()
    n = len(latente)
    orden = np.argsort(-np.asarray(score, dtype=float))
    filas = []
    for frac in fracciones:
        k = max(1, int(round(frac * n)))
        sel = orden[:k]
        filas.append({
            "frac_contactada": frac,
            "n_contactados": k,
            "traspasos_evitados": float(-lat_cate[sel].sum()),
            "persuadibles_contactados": int((segmento[sel] == "persuadible").sum()),
            "sleeping_dogs_contactados": int((segmento[sel] == "sleeping_dog").sum()),
            "neutros_contactados": int((segmento[sel] == "neutral").sum()),
        })
    return pd.DataFrame(filas)


def tabla_segmento_accion(latente: pd.DataFrame, tau_hat: np.ndarray) -> pd.DataFrame:
    """Tabla "segmento -> accion" para el README: tamano, CATE verdadero y estimado, y la recomendacion."""
    d = latente.copy()
    d["tau_hat"] = np.asarray(tau_hat, dtype=float)
    out = (d.groupby("segmento", observed=True)
          .agg(n=("lat_cate", "size"), frac=("lat_cate", lambda s: len(s) / len(d)),
               lat_cate_medio=("lat_cate", "mean"), tau_hat_medio=("tau_hat", "mean"))
          .reset_index())
    acciones = {
        "persuadible": "Contactar: el contacto reduce la probabilidad de traspaso.",
        "sleeping_dog": "NO contactar: el contacto la aumenta (efecto 'sleeping dog').",
        "neutral": "No priorizar: el contacto no cambia la decision (contacto desperdiciado).",
    }
    out["accion_recomendada"] = out["segmento"].map(acciones)
    return out.sort_values("segmento").reset_index(drop=True)
