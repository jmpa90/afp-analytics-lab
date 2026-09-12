"""Evaluacion del modelo de traspasos: metricas comunes, segmentos, calibracion y negocio.

Todas las metricas de clasificacion salen de `afp_lab.common.metrics` para que los resultados sean
comparables con los otros proyectos del laboratorio. Se agregan dos cosas propias del problema:

* **captura**: contactando al X % de mayor score, que porcentaje de los traspasos y del *saldo en
  riesgo* se alcanza. Es la metrica con la que el equipo comercial decide.
* **saldo en riesgo esperado**: sum(prob calibrada x saldo), que se compara con el saldo realmente
  perdido para verificar que la calibracion sirve para agregar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from afp_lab.common.metrics import metricas_clasificacion, tabla_metricas_por_segmento

SEGMENTOS = ["tramo_saldo", "tramo_edad", "tipo_afiliado", "fondo", "tramo_renta", "estado_laboral"]
FRACCIONES_CONTACTO = (0.05, 0.10, 0.20)
N_BOOT_REFERENCIA = 200          # ronda 3: minimo pedido por el brief para el reporte
SEGMENTOS_BOOTSTRAP = ["tramo_edad", "tramo_saldo"]      # brief: "al menos" estos dos


# --------------------------------------------------------------------- segmentos
def agregar_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los tramos de negocio usados para cortar las metricas."""
    out = df.copy()
    saldo = out["saldo_obligatorio"].to_numpy(dtype=float)
    out["tramo_saldo"] = pd.cut(saldo, [-1, 5e6, 15e6, 40e6, np.inf],
                                labels=["0-5M", "5-15M", "15-40M", "40M+"]).astype(str)
    edad = out["edad"].to_numpy()
    out["tramo_edad"] = pd.cut(edad, [0, 29, 39, 49, 59, 200],
                               labels=["<30", "30-39", "40-49", "50-59", "60+"]).astype(str)
    renta = out["rim_ref"].to_numpy(dtype=float)
    out["tramo_renta"] = pd.cut(np.nan_to_num(renta, nan=-1.0), [-2, 0, 6e5, 1.2e6, 2.5e6, np.inf],
                                labels=["sin_renta", "0-600k", "600k-1,2M", "1,2-2,5M", "2,5M+"]).astype(str)
    # a texto: `common.metrics.tabla_metricas_por_segmento` ya agrupa con observed=True (arreglado
    # por el gerente en la ronda 1), pero dejar los segmentos como texto mantiene estable el join de
    # `comparar_segmentos` y hace legibles los csv
    for c in ("tipo_afiliado", "fondo", "estado_laboral"):
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


# --------------------------------------------------------------------- tablas principales
def tabla_comparativa(df: pd.DataFrame, predicciones: dict[str, np.ndarray]) -> pd.DataFrame:
    """Metricas de `common.metrics.metricas_clasificacion` para el modelo y cada baseline."""
    y = df["y"].to_numpy()
    filas = []
    for nombre, p in predicciones.items():
        filas.append({"predictor": nombre, **metricas_clasificacion(y, np.asarray(p, dtype=float))})
    return pd.DataFrame(filas)


def metricas_por_segmento(df: pd.DataFrame, predicciones: dict[str, np.ndarray],
                          segmentos: list[str] | None = None, min_frac: float = 0.05) -> pd.DataFrame:
    """Metricas por segmento para cada predictor; marca los segmentos grandes (>= `min_frac`)."""
    segmentos = segmentos or SEGMENTOS
    filas = []
    for nombre, p in predicciones.items():
        d = df.assign(_p=np.asarray(p, dtype=float))
        for seg in segmentos:
            if seg not in d.columns:
                continue
            t = tabla_metricas_por_segmento(d, seg, metricas_clasificacion, "y", "_p")
            t = t.rename(columns={seg: "valor"})
            t.insert(0, "segmento", seg)
            t.insert(0, "predictor", nombre)
            filas.append(t)
    out = pd.concat(filas, ignore_index=True)
    out["frac_filas"] = out["n"] / len(df)
    out["segmento_grande"] = out["frac_filas"] >= min_frac
    return out


def comparar_segmentos(tabla_seg: pd.DataFrame, modelo: str, baseline: str,
                       metrica: str = "auc") -> pd.DataFrame:
    """Diferencia modelo - baseline en cada segmento (para el checklist: no perder en segmentos grandes)."""
    m = tabla_seg[tabla_seg["predictor"] == modelo].set_index(["segmento", "valor"])
    b = tabla_seg[tabla_seg["predictor"] == baseline].set_index(["segmento", "valor"])
    out = m[[metrica, "n", "tasa_base", "frac_filas", "segmento_grande"]].join(
        b[[metrica]], rsuffix="_baseline", how="inner").reset_index()
    out["dif"] = out[metrica] - out[f"{metrica}_baseline"]
    # un segmento sin positivos (o sin negativos) devuelve AUC NaN: no puede darse por ganado
    out["comparable"] = out[metrica].notna() & out[f"{metrica}_baseline"].notna()
    out["gana_modelo"] = out["comparable"] & (out["dif"] > 0)
    return out.sort_values(["segmento", "valor"]).reset_index(drop=True)


def segmentos_no_superados(tabla_comp: pd.DataFrame) -> pd.DataFrame:
    """Segmentos grandes donde el modelo NO supera al baseline, incluidos los no comparables."""
    g = tabla_comp[tabla_comp["segmento_grande"]]
    return g[~g["gana_modelo"]].reset_index(drop=True)


# --------------------------------------------------------------------- ronda 3: bootstrap por segmento
def _auc_o_nan(y: np.ndarray, p: np.ndarray) -> float:
    """AUC, o NaN si el remuestreo dejo un segmento sin positivos o sin negativos."""
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def _captura_frac(y: np.ndarray, p: np.ndarray, saldo: np.ndarray | None, frac: float,
                  objetivo: str) -> float:
    """Captura contactando el `frac` de mayor score DENTRO del segmento (sin desempate: pensado para
    correrse cientos de veces en el bootstrap, no como sustituto de `captura_top` en el reporte principal)."""
    n = len(y)
    k = max(1, int(round(frac * n)))
    sel = np.argsort(-p)[:k]
    if objetivo == "saldo":
        total = float(saldo[y == 1].sum())
        return float(saldo[sel][y[sel] == 1].sum() / total) if total > 0 else float("nan")
    total = float(y.sum())
    return float(y[sel].sum() / total) if total > 0 else float("nan")


def bootstrap_ic_segmento(df: pd.DataFrame, prob: np.ndarray, segmento: str, metrica: str = "auc",
                          n_boot: int = N_BOOT_REFERENCIA, seed: int = 42, alpha: float = 0.05,
                          frac_captura: float = 0.10, objetivo_captura: str = "traspasos",
                          umbral_aceptacion: float | None = None,
                          col_saldo: str = "saldo_obligatorio") -> pd.DataFrame:
    """Intervalo bootstrap percentil de `metrica` ('auc' o 'captura') por valor de `segmento`.

    Remuestrea CON REEMPLAZO `n_boot` veces (semilla fija, reproducible), DENTRO de cada valor del
    segmento (el tamano de cada segmento es el que es: no tiene sentido mezclar el error estandar del
    segmento grande con el del chico). `metrica="captura"` mide, dentro del segmento, la fraccion de
    `objetivo_captura` ('traspasos' o 'saldo') capturada contactando el `frac_captura` de mayor score.

    Backlog de la ronda 2 (revision, punto 2 de la seccion 12): "reportar el AUC por segmento con
    intervalo (bootstrap) y no dar por perdido un segmento con menos de ~100 eventos". `cruza_0_5`
    (solo para AUC) y `cruza_umbral_aceptacion` (si se pasa `umbral_aceptacion`) marcan exactamente
    eso: un intervalo que cruza el umbral no permite concluir que el segmento es distinto de el.
    """
    if metrica not in ("auc", "captura"):
        raise ValueError("metrica debe ser 'auc' o 'captura'")
    if n_boot < 2:
        raise ValueError("n_boot debe ser >= 2")
    rng = np.random.default_rng(seed)
    y_all = df["y"].to_numpy()
    p_all = np.asarray(prob, dtype=float)
    saldo_all = df[col_saldo].to_numpy(dtype=float) if metrica == "captura" else None

    def _calc(y: np.ndarray, p: np.ndarray, saldo: np.ndarray | None) -> float:
        if metrica == "auc":
            return _auc_o_nan(y, p)
        return _captura_frac(y, p, saldo, frac_captura, objetivo_captura)

    filas = []
    # `.indices`: posiciones ENTERAS (0-based) de cada valor del segmento en `df`, alineadas con
    # `y_all`/`p_all` (que salen de `.to_numpy()` en el mismo orden que las filas de `df`).
    for valor, pos in df.groupby(segmento, observed=True).indices.items():
        pos = np.asarray(pos)
        y, p = y_all[pos], p_all[pos]
        saldo = saldo_all[pos] if saldo_all is not None else None
        n = len(y)
        punto = _calc(y, p, saldo)
        boot = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            sel = rng.integers(0, n, n)
            boot[b] = _calc(y[sel], p[sel], saldo[sel] if saldo is not None else None)
        validos = boot[~np.isnan(boot)]
        if len(validos) >= 2:
            lo, hi = (float(x) for x in np.percentile(validos, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        else:
            lo = hi = float("nan")
        fila = {segmento: valor, "n": int(n), "n_eventos": int(y.sum()), "metrica": metrica,
               "punto": punto, "ic_low": lo, "ic_high": hi, "n_boot_validos": int(len(validos))}
        if metrica == "auc":
            fila["cruza_0_5"] = bool(np.isnan(lo) or (lo <= 0.5 <= hi))
        if umbral_aceptacion is not None:
            fila["cruza_umbral_aceptacion"] = bool(np.isnan(lo) or (lo <= umbral_aceptacion <= hi))
        filas.append(fila)
    out = pd.DataFrame(filas)
    return out.sort_values(segmento).reset_index(drop=True)


def bootstrap_auc_segmentos(df: pd.DataFrame, prob: np.ndarray,
                            segmentos: list[str] | None = None,
                            n_boot: int = N_BOOT_REFERENCIA, seed: int = 42,
                            umbral_aceptacion: float = 0.70) -> pd.DataFrame:
    """`bootstrap_ic_segmento` (AUC) apilado para varias familias de segmento a la vez.

    Por defecto `tramo_edad` (incluye 60+) y `tramo_saldo`, que es lo que pide el brief; `df` debe
    venir de `agregar_segmentos`. `umbral_aceptacion=0,70` es el criterio de aceptacion del proyecto
    (CONVENCIONES / gerente, ronda 2): un intervalo que lo cruza no permite decir que ESE segmento en
    particular cumple o no cumple el criterio global.
    """
    segmentos = segmentos or SEGMENTOS_BOOTSTRAP
    filas = [bootstrap_ic_segmento(df, prob, seg, metrica="auc", n_boot=n_boot, seed=seed,
                                   umbral_aceptacion=umbral_aceptacion).rename(columns={seg: "valor"})
             .assign(segmento=seg)
             for seg in segmentos]
    cols = ["segmento", "valor", "n", "n_eventos", "punto", "ic_low", "ic_high", "n_boot_validos",
           "cruza_0_5", "cruza_umbral_aceptacion"]
    return pd.concat(filas, ignore_index=True)[cols]


# --------------------------------------------------------------------- metrica de negocio
def captura_top(df: pd.DataFrame, predicciones: dict[str, np.ndarray],
                fracciones: tuple[float, ...] = FRACCIONES_CONTACTO,
                col_saldo: str = "saldo_obligatorio", seed: int = 7) -> pd.DataFrame:
    """% de traspasos y % de saldo en riesgo capturados contactando al X % de mayor score.

    Los empates (la regla de negocio binaria genera muchos) se rompen con un desempate aleatorio
    reproducible: es la forma honesta de evaluar un score con pocos valores distintos.
    """
    y = df["y"].to_numpy()
    saldo = df[col_saldo].to_numpy(dtype=float)
    saldo_riesgo_total = float(saldo[y == 1].sum())
    rng = np.random.default_rng(seed)
    desempate = rng.random(len(df))
    filas = []
    for nombre, p in predicciones.items():
        orden = np.lexsort((desempate, -np.asarray(p, dtype=float)))
        for frac in fracciones:
            k = max(1, int(round(frac * len(df))))
            sel = orden[:k]
            filas.append({
                "predictor": nombre,
                "frac_contactada": frac,
                "n_contactados": k,
                "traspasos_capturados": int(y[sel].sum()),
                "pct_traspasos_capturados": float(y[sel].sum() / max(y.sum(), 1)),
                "pct_saldo_riesgo_capturado": float(saldo[sel][y[sel] == 1].sum() / max(saldo_riesgo_total, 1.0)),
                "lift": float((y[sel].mean() / max(y.mean(), 1e-9))),
                "saldo_riesgo_capturado_mm": float(saldo[sel][y[sel] == 1].sum() / 1e6),
            })
    return pd.DataFrame(filas)


def saldo_en_riesgo(df: pd.DataFrame, prob: np.ndarray, col_saldo: str = "saldo_obligatorio") -> pd.DataFrame:
    """Saldo en riesgo esperado (sum p x saldo) contra el saldo efectivamente traspasado."""
    saldo = df[col_saldo].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    p = np.asarray(prob, dtype=float)
    esperado = float((p * saldo).sum())
    real = float(saldo[y == 1].sum())
    return pd.DataFrame([{
        "saldo_riesgo_esperado_mm": esperado / 1e6,
        "saldo_riesgo_real_mm": real / 1e6,
        "sesgo_relativo": (esperado - real) / max(real, 1.0),
        # descomposicion: cuanto del sesgo es error de NIVEL (numero de traspasos) y cuanto es la
        # correlacion entre la probabilidad y el saldo
        "sesgo_conteo": float(p.sum() - y.sum()) / max(float(y.sum()), 1.0),
        "saldo_total_cartera_mm": float(saldo.sum()) / 1e6,
    }])


# --------------------------------------------------------------------- calibracion
def saldo_en_riesgo_por_corte(df: pd.DataFrame, prob: np.ndarray,
                              col_saldo: str = "saldo_obligatorio") -> pd.DataFrame:
    """Saldo en riesgo esperado vs real en cada corte (el sesgo agregado de un solo numero esconde
    que la calibracion se mueve mes a mes)."""
    d = df.assign(_p=np.asarray(prob, dtype=float))
    filas = []
    for corte, g in d.groupby("periodo_corte", observed=True):
        r = saldo_en_riesgo(g, g["_p"].to_numpy(), col_saldo).iloc[0].to_dict()
        filas.append({"periodo_corte": int(corte), **r})
    return pd.DataFrame(filas)


def curva_calibracion(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de calibracion por decil de score (probabilidad media vs tasa observada)."""
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    orden = np.argsort(prob)
    grupos = np.array_split(orden, n_bins)
    filas = []
    for i, g in enumerate(grupos, start=1):
        filas.append({
            "decil": i,
            "n": len(g),
            "prob_media": float(prob[g].mean()),
            "tasa_observada": float(y[g].mean()),
            "dif": float(prob[g].mean() - y[g].mean()),
        })
    return pd.DataFrame(filas)


def resumen_por_corte(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """AUC y tasa base por corte de test (estabilidad temporal)."""
    d = df.assign(_p=np.asarray(prob, dtype=float))
    return tabla_metricas_por_segmento(d, "periodo_corte", metricas_clasificacion, "y", "_p")
