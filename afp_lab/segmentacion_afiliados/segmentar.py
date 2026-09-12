"""Seleccion de k y ajuste de la segmentacion (k-means / GMM), con seleccion por silueta,
BIC (solo GMM) y estabilidad (ARI de re-muestreo y ARI entre cortes separados 6 meses),
como pide `BRIEF.md` §2.

Estabilidad de re-muestreo: se ajustan varios modelos independientes sobre submuestras
aleatorias del 80 % de las filas (bootstrap sin reemplazo) y se promedia el ARI entre cada
par de ajustes, restringido a las filas que ambos vieron (Ben-Hur, Elisseeff y Guyon, 2002:
"A Stability Based Method for Discovering Structure in Clustered Data"). Estabilidad
temporal: el mismo procedimiento de ajuste (mismo algoritmo y k), aplicado INDEPENDIENTEMENTE
a las features de `periodo_corte` y a las de `periodo_corte - 6`, comparado por ARI sobre los
afiliados presentes en ambos cortes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from .features import FEATURES_CLUSTERING

K_MIN, K_MAX = 5, 9
SEG_MIN_FRAC = 0.03      # ningun segmento < 3 % (brief)
SILUETA_MIN = 0.20
ARI_RESAMPLING_MIN = 0.60
N_BOOT_ESTABILIDAD = 5
FRAC_BOOT = 0.8
MAX_FILAS_SILUETA = 8_000    # muestreo para no encarecer el runtime a 20 000+ filas
MIN_FILAS_ARI = 30           # bajo esto, un ARI no es confiable (se descarta el par/corte)


def escalar(df: pd.DataFrame, scaler: RobustScaler | None = None) -> tuple[np.ndarray, RobustScaler]:
    """Escalado robusto (mediana / IQR) de `FEATURES_CLUSTERING`: robusto a los outliers de
    cola larga (renta, saldo) que un StandardScaler dejaria dominar la distancia euclidiana."""
    X = df[FEATURES_CLUSTERING].to_numpy(dtype=float)
    if scaler is None:
        scaler = RobustScaler().fit(X)
    return scaler.transform(X), scaler


def _ajustar_modelo(algoritmo: str, k: int, seed: int = 0):
    if algoritmo == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed)
    if algoritmo == "gmm":
        return GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, n_init=1)
    raise ValueError(f"algoritmo desconocido: {algoritmo}")


def _silueta_muestreada(X: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    n = len(X)
    if n > MAX_FILAS_SILUETA:
        idx = np.random.default_rng(seed).choice(n, MAX_FILAS_SILUETA, replace=False)
        return float(silhouette_score(X[idx], labels[idx]))
    return float(silhouette_score(X, labels))


def _tamanos_ok(labels: np.ndarray, k: int) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < k:
        return False
    return bool((counts / counts.sum() >= SEG_MIN_FRAC).all())


def _ari_resampling(X: np.ndarray, algoritmo: str, k: int, n_boot: int = N_BOOT_ESTABILIDAD,
                    frac: float = FRAC_BOOT, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(X)
    m = int(n * frac)
    particiones = []
    for b in range(n_boot):
        idx = rng.choice(n, m, replace=False)
        modelo = _ajustar_modelo(algoritmo, k, seed=1000 * (seed + 1) + b)
        labels = modelo.fit_predict(X[idx])
        particiones.append((idx, labels))
    aris = []
    for i in range(len(particiones)):
        for j in range(i + 1, len(particiones)):
            idx_i, lab_i = particiones[i]
            idx_j, lab_j = particiones[j]
            comunes, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(comunes) < max(MIN_FILAS_ARI, k * 5):
                continue
            aris.append(adjusted_rand_score(lab_i[pos_i], lab_j[pos_j]))
    return float(np.mean(aris)) if aris else float("nan")


@dataclass
class ResultadoK:
    algoritmo: str
    k: int
    silueta: float
    bic: float
    ari_resampling: float
    ari_temporal: float
    tamanos_ok: bool
    cumple: bool


def evaluar_k(X: np.ndarray, af_ids: np.ndarray, algoritmo: str, k: int,
             X_t6: np.ndarray | None = None, af_ids_t6: np.ndarray | None = None,
             seed: int = 0) -> ResultadoK:
    """Ajusta `algoritmo` con `k` segmentos sobre `X` y calcula sus 4 metricas de seleccion."""
    modelo = _ajustar_modelo(algoritmo, k, seed=seed)
    labels = modelo.fit_predict(X)
    silueta = _silueta_muestreada(X, labels, seed=seed)
    bic = float(modelo.bic(X)) if algoritmo == "gmm" else float("nan")
    ari_resamp = _ari_resampling(X, algoritmo, k, seed=seed)

    ari_temp = float("nan")
    if X_t6 is not None and af_ids_t6 is not None:
        modelo_t6 = _ajustar_modelo(algoritmo, k, seed=seed + 500)
        labels_t6 = modelo_t6.fit_predict(X_t6)
        comunes, pos0, pos6 = np.intersect1d(af_ids, af_ids_t6, return_indices=True)
        if len(comunes) >= max(MIN_FILAS_ARI, k * 5):
            ari_temp = float(adjusted_rand_score(labels[pos0], labels_t6[pos6]))

    tam_ok = _tamanos_ok(labels, k)
    ari_ok = ari_resamp >= ARI_RESAMPLING_MIN if np.isfinite(ari_resamp) else False
    cumple = (silueta >= SILUETA_MIN) and ari_ok and tam_ok
    return ResultadoK(algoritmo=algoritmo, k=k, silueta=silueta, bic=bic, ari_resampling=ari_resamp,
                      ari_temporal=ari_temp, tamanos_ok=tam_ok, cumple=cumple)


def seleccionar_k(df_corte: pd.DataFrame, df_corte_t6: pd.DataFrame | None = None,
                  k_min: int = K_MIN, k_max: int = K_MAX, seed: int = 0
                  ) -> tuple[pd.DataFrame, ResultadoK, RobustScaler]:
    """Evalua k-means y GMM para k en [`k_min`, `k_max`]. Devuelve la tabla comparativa
    completa, el resultado ganador (el de mayor silueta entre los que cumplen las 3
    condiciones de aceptacion; si ninguno cumple, el de mayor silueta igual, marcado
    `cumple=False` para que `run.py`/el README lo reporten como tal) y el `scaler` ajustado
    sobre `df_corte` (se reutiliza para escalar `df_corte_t6` y cortes futuros)."""
    X, scaler = escalar(df_corte)
    af_ids = df_corte["afiliado_id"].to_numpy()
    X_t6 = af_ids_t6 = None
    if df_corte_t6 is not None and len(df_corte_t6):
        X_t6, _ = escalar(df_corte_t6, scaler)
        af_ids_t6 = df_corte_t6["afiliado_id"].to_numpy()

    resultados = [evaluar_k(X, af_ids, algo, k, X_t6, af_ids_t6, seed=seed)
                 for algo in ("kmeans", "gmm") for k in range(k_min, k_max + 1)]
    tabla = pd.DataFrame([asdict(r) for r in resultados])

    candidatos = tabla[tabla["cumple"]]
    fila = (candidatos if len(candidatos) else tabla).sort_values("silueta", ascending=False).iloc[0]
    ganador = next(r for r in resultados if r.algoritmo == fila["algoritmo"] and r.k == fila["k"])
    return tabla, ganador, scaler


def ajustar(df_corte: pd.DataFrame, algoritmo: str, k: int, seed: int = 0,
           scaler: RobustScaler | None = None) -> tuple[np.ndarray, object, RobustScaler]:
    """Ajuste final de produccion: mismo procedimiento que `evaluar_k`, pero devolviendo el
    modelo entrenado (para poder `asignar` cortes futuros con `.predict`, ver matriz de
    transicion en `evaluate.py`)."""
    X, scaler = escalar(df_corte, scaler)
    modelo = _ajustar_modelo(algoritmo, k, seed=seed)
    labels = modelo.fit_predict(X)
    return labels, modelo, scaler


def asignar(modelo, scaler: RobustScaler, df: pd.DataFrame) -> np.ndarray:
    """Asigna (sin reajustar) las filas de `df` a los segmentos de un `modelo` ya entrenado
    (out-of-sample: `KMeans.predict`/`GaussianMixture.predict` funcionan igual)."""
    X, _ = escalar(df, scaler)
    return modelo.predict(X)


def fusionar_segmentos_chicos(X: np.ndarray, labels: np.ndarray,
                              min_frac: float = SEG_MIN_FRAC) -> tuple[np.ndarray, dict[int, int]]:
    """Fusiona iterativamente el segmento mas chico (bajo `min_frac` del total) con el
    segmento vecino mas cercano (distancia entre centroides en el espacio ya escalado),
    hasta que todos cumplan el piso de tamano del brief (`ningun segmento < 3 %`) o quede
    un unico segmento. `k` puede no coincidir exactamente con el ganador de `seleccionar_k`
    en la practica: a veces ninguno de los k evaluados en [5,9] deja los 5-9 segmentos
    *todos* sobre el piso (un sub-grupo genuino pero chico, p. ej. afiliados sin RIM
    conocida y sin empleador, se cuela bajo el 3 % en cualquier k probado); fusionarlo con
    su vecino mas parecido es la practica estandar en segmentacion aplicada, mas razonable
    que forzar una particion peor solo para cumplir el piso. Devuelve las etiquetas
    fusionadas (renumeradas 0..k'-1) y el mapa {etiqueta_original: etiqueta_fusionada} para
    aplicar la MISMA fusion a asignaciones futuras (ver `asignar_fusionado`)."""
    labels = np.asarray(labels).copy()
    mapa = {int(c): int(c) for c in np.unique(labels)}
    n = len(labels)
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) <= 1 or (counts / n).min() >= min_frac:
            break
        chico = int(uniq[np.argmin(counts)])
        centroides = {int(c): X[labels == c].mean(axis=0) for c in uniq}
        cc = centroides.pop(chico)
        destino = min(centroides, key=lambda c: float(np.linalg.norm(centroides[c] - cc)))
        labels = np.where(labels == chico, destino, labels)
        mapa = {orig: (destino if dest == chico else dest) for orig, dest in mapa.items()}

    restantes = sorted(np.unique(labels).tolist())
    renum = {old: new for new, old in enumerate(restantes)}
    labels_final = np.array([renum[int(l)] for l in labels])
    mapa_final = {orig: renum[dest] for orig, dest in mapa.items()}
    return labels_final, mapa_final


def asignar_fusionado(modelo, scaler: RobustScaler, df: pd.DataFrame, mapa_fusion: dict[int, int]) -> np.ndarray:
    """`asignar` seguido del mapa de `fusionar_segmentos_chicos`, para que un corte futuro
    (transicion a 6/12 meses) quede en el MISMO espacio de segmentos ya fusionado."""
    labels = asignar(modelo, scaler, df)
    return np.array([mapa_fusion[int(l)] for l in labels])
