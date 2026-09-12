"""Features "as-of" de `segmentacion_afiliados` (pandas; sin Spark).

Convencion de tiempos (CONVENCIONES.md §3, misma que el resto del laboratorio):
`periodo_corte` es el mes en que se arma el segmento. El panel laboral y la RIM (universo
`cuprum` propio, `synthetic.generar_datos`) se conocen hasta `corte-2` (`corte_rim` abajo);
las tablas "operacionales propias de la AFP" (saldo, fondo, cambios de fondo, uso digital,
contactos, reclamos, aportes voluntarios — todas importadas de los proyectos 02/03, ver
`synthetic.generar_tablas_externas`) se conocen hasta `corte` mismo, sin desfase adicional.

Dos familias de features:

  * **Nucleo** (`_features_nucleo`): 100 % del universo `cuprum` propio — renta, densidad,
    estabilidad laboral, edad, saldo obligatorio aproximado. Sirven de base para el
    clustering y son internamente consistentes entre si (mismo afiliado, mismo universo).
  * **Externas** (`_features_externas`): de los proyectos 02/03 (saldo/fondo, cambios de
    fondo, uso digital, contactos, reclamos, ahorro voluntario, antiguedad en la AFP). Sujetas
    a la limitacion de perfil documentada en `synthetic.py` (la simulacion `sistema` de 02/03
    no es la misma persona demografica que el `afiliado_id` del universo `cuprum`, solo el
    mismo indice) — se incluyen porque el brief pide explicitamente esas familias y "no
    duplicar mecanismos", documentando la limitacion en vez de ignorarla.

`saldo_aprox` (obligatorio) es una formula CERRADA (valor futuro de una anualidad con la tasa
de cotizacion obligatoria y la renta y densidad historicas conocidas del propio afiliado, ver
`synthetic.TASA_MENSUAL_CAPITALIZACION`), no una simulacion mes a mes: es una aproximacion
razonable para segmentar (orden de magnitud del saldo relativo entre afiliados), documentada
como tal — no es el saldo exacto que calcularia un proyecto de capitalizacion dedicado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import TASA_COTIZACION_OBLIGATORIA, TASA_MENSUAL_CAPITALIZACION, add_months, periodo_to_idx

VENTANA_DENSIDAD = 24   # meses de historia para densidad/racha/n_empleadores (igual orden que proyecto 04)
VENTANA_EXTERNA_CORTA = 3    # uso digital: meses recientes
VENTANA_EXTERNA_MEDIA = 6    # ahorro voluntario (flujo): meses recientes
VENTANA_EXTERNA_LARGA = 12   # contactos, reclamos, cambios de fondo: meses recientes

FONDO_ORDEN = {"E": 0, "D": 1, "C": 2, "B": 3, "A": 4}   # E = mas conservador, A = mas riesgo/renta variable

# Features numericas que entran al ESPACIO DE CLUSTERING (ver segmentar.py): todas
# reescaladas con RobustScaler. El resto (categoricas, ids, fechas) es solo para perfilar
# segmentos y para el clasificador explicativo (que si puede usar categoricas).
FEATURES_CLUSTERING = [
    "log_renta", "densidad_12", "densidad_24", "racha_cotiza_m", "n_empleadores_12",
    "edad", "meses_hasta_pension", "log_saldo_aprox", "saldo_renta_ratio",
    "log_apv_saldo", "log_apv_flujo_6m", "fondo_ord", "n_cambios_fondo_12",
    "log_uso_digital_3m", "contactos_12", "reclamos_12", "log_antiguedad_afp",
    "es_independiente", "es_voluntario", "es_pensionado_activo",
]
# Columnas adicionales de perfilado (medianas/modas por segmento en evaluate.py), no entran
# al clustering pero ayudan a nombrar y describir cada segmento.
FEATURES_PERFIL = ["tipo_afiliado", "sexo", "region", "nivel_educacional", "rubro", "fondo",
                   "renta", "saldo_aprox", "apv_saldo_voluntario", "antiguedad_afp_meses"]


def _historia_nucleo(panel: pd.DataFrame, af_ids: np.ndarray, corte: int, periodo_inicio: int) -> pd.DataFrame:
    """Renta, densidad, racha y estabilidad laboral a partir del panel `cuprum` propio,
    ancladas en `corte_rim = corte-2` (igual convencion que `rim_nowcasting`/proyecto 04)."""
    corte_rim = add_months(corte, -2)
    periodo_min = max(add_months(corte_rim, -(VENTANA_DENSIDAD - 1)), periodo_inicio)
    p = panel[(panel["periodo"] >= periodo_min) & (panel["periodo"] <= corte_rim)].copy()
    p["rim"] = p["rim_emp1"].fillna(0.0) + p["rim_emp2"].fillna(0.0)
    p["cotiza"] = (p["rim"] > 0).astype(int)
    p = p.sort_values(["afiliado_id", "periodo"])
    g = p.groupby("afiliado_id", sort=False)

    def _ultimo(col):
        return g[col].last()

    out = pd.DataFrame(index=pd.Index(af_ids, name="afiliado_id"))
    out["renta"] = _ultimo("rim").reindex(out.index).fillna(0.0)
    out["empleador_actual"] = _ultimo("empleador_id").reindex(out.index)

    idx_corte = periodo_to_idx(corte_rim)
    for w, nombre in ((12, "densidad_12"), (24, "densidad_24")):
        ini = max(add_months(corte_rim, -(w - 1)), periodo_min)
        sub = p[p["periodo"] >= ini]
        dens = sub.groupby("afiliado_id")["cotiza"].mean()
        out[nombre] = dens.reindex(out.index).fillna(0.0)

    # racha de cotizacion (meses consecutivos con rim > 0, terminando en corte_rim)
    p_desc = p.sort_values(["afiliado_id", "periodo"], ascending=[True, False])
    gd = p_desc.groupby("afiliado_id", sort=False)
    racha = gd["cotiza"].cumprod().groupby(p_desc["afiliado_id"]).sum()
    out["racha_cotiza_m"] = racha.reindex(out.index).fillna(0.0)

    n_emp = g["empleador_id"].nunique(dropna=True)
    out["n_empleadores_12"] = n_emp.reindex(out.index).fillna(0.0)

    return out.reset_index()


def _saldo_aprox(panel: pd.DataFrame, af_ids: np.ndarray, corte: int, periodo_inicio: int) -> pd.DataFrame:
    """Saldo obligatorio aproximado: valor futuro de una anualidad de `TASA_COTIZACION_
    OBLIGATORIA x renta media historica`, capitalizada `TASA_MENSUAL_CAPITALIZACION` por los
    meses efectivamente cotizados desde `periodo_inicio` (o la afiliacion) hasta `corte-2`.
    No simula mes a mes (eso es lo que hacen los proyectos 02/03 con SU propio universo):
    es una formula cerrada, documentada como aproximacion, sobre el universo `cuprum` propio
    para que el saldo sea consistente con la renta/densidad de las features nucleo."""
    corte_rim = add_months(corte, -2)
    p = panel[(panel["periodo"] >= periodo_inicio) & (panel["periodo"] <= corte_rim)].copy()
    p["rim"] = p["rim_emp1"].fillna(0.0) + p["rim_emp2"].fillna(0.0)
    cot = p[p["rim"] > 0]
    agg = cot.groupby("afiliado_id")["rim"].agg(["mean", "count"]).reindex(af_ids).fillna(0.0)
    i_m = TASA_MENSUAL_CAPITALIZACION
    factor = (np.power(1 + i_m, agg["count"].to_numpy()) - 1) / i_m
    saldo = TASA_COTIZACION_OBLIGATORIA * agg["mean"].to_numpy() * factor
    return pd.DataFrame({"afiliado_id": af_ids, "saldo_aprox": np.round(saldo, 0)})


def _ultimo_valor_hasta(df: pd.DataFrame, col: str, corte: int, af_ids: np.ndarray, relleno=0.0) -> pd.Series:
    """Ultimo valor conocido de `col` hasta (e incluyendo) `corte`, por afiliado (tablas
    "operacionales propias de la AFP": desfase 0, CONVENCIONES §3)."""
    if df is None or df.empty:
        return pd.Series(relleno, index=af_ids)
    sub = df[df["periodo"] <= corte].sort_values(["afiliado_id", "periodo"])
    ultimo = sub.groupby("afiliado_id")[col].last()
    return ultimo.reindex(af_ids).fillna(relleno)


def _conteo_ventana(df: pd.DataFrame, corte: int, ventana: int, af_ids: np.ndarray,
                    col_periodo: str = "periodo", col_valor: str | None = None) -> pd.Series:
    """Conteo (o suma de `col_valor`) de filas en `(corte-ventana+1, corte]`, por afiliado."""
    if df is None or df.empty:
        return pd.Series(0.0, index=af_ids)
    ini = add_months(corte, -(ventana - 1))
    sub = df[(df[col_periodo] >= ini) & (df[col_periodo] <= corte)]
    if col_valor is None:
        agg = sub.groupby("afiliado_id").size()
    else:
        agg = sub.groupby("afiliado_id")[col_valor].sum()
    return agg.reindex(af_ids).fillna(0.0)


def _features_externas(tablas_externas: dict[str, pd.DataFrame], corte: int, af_ids: np.ndarray) -> pd.DataFrame:
    """Features de los proyectos 02/03 (ver limitacion de perfil documentada en
    `synthetic.py`): saldo/fondo/antiguedad AFP, cambios de fondo, uso digital, contactos,
    reclamos, ahorro voluntario (stock y flujo). Desfase 0 (conocidas hasta `corte`)."""
    out = pd.DataFrame(index=pd.Index(af_ids, name="afiliado_id"))
    churn_saldos = tablas_externas.get("churn_saldos")
    out["saldo_obligatorio_churn"] = _ultimo_valor_hasta(churn_saldos, "saldo_obligatorio", corte, af_ids)
    out["fondo"] = _ultimo_valor_hasta(churn_saldos, "fondo", corte, af_ids, relleno="C")
    out["antiguedad_afp_meses"] = _ultimo_valor_hasta(churn_saldos, "antiguedad_afp_meses", corte, af_ids)

    out["n_cambios_fondo_12"] = _conteo_ventana(tablas_externas.get("churn_cambios_fondo"), corte,
                                                VENTANA_EXTERNA_LARGA, af_ids)
    out["contactos_12"] = _conteo_ventana(tablas_externas.get("churn_contactos"), corte,
                                          VENTANA_EXTERNA_LARGA, af_ids)
    out["reclamos_12"] = _conteo_ventana(tablas_externas.get("churn_reclamos"), corte,
                                         VENTANA_EXTERNA_LARGA, af_ids)

    uso = tablas_externas.get("churn_uso_digital")
    if uso is not None and len(uso):
        uso = uso.assign(_uso_total=uso["logins_app"] + uso["logins_web"] + uso["consultas_saldo"])
        out["uso_digital_3m"] = _conteo_ventana(uso, corte, VENTANA_EXTERNA_CORTA, af_ids, col_valor="_uso_total")
    else:
        out["uso_digital_3m"] = 0.0

    apv_saldos = tablas_externas.get("apv_saldos")
    out["apv_saldo_voluntario"] = _ultimo_valor_hasta(apv_saldos, "saldo_voluntario", corte, af_ids)
    out["apv_flujo_6m"] = _conteo_ventana(tablas_externas.get("apv_aportes"), corte,
                                          VENTANA_EXTERNA_MEDIA, af_ids, col_valor="monto")
    return out.reset_index()


def construir_features(tablas: dict[str, pd.DataFrame], tablas_externas: dict[str, pd.DataFrame],
                       periodo_corte: int) -> pd.DataFrame:
    """Dataset de segmentacion "as-of" para `periodo_corte`: una fila por afiliado ya
    afiliado a esa fecha (todos los `tipo_afiliado`: la pregunta de negocio es sobre TODA la
    cartera, no solo dependientes)."""
    afiliados = tablas["afiliados"]
    panel = tablas["panel_mensual"]
    empleadores = tablas["empleadores"]
    macro = tablas["macro"]
    corte = periodo_corte
    periodo_inicio = int(macro["periodo"].min())
    activos = afiliados[afiliados["periodo_afiliacion"] <= corte].copy()
    af_ids = activos["afiliado_id"].to_numpy()

    nucleo = _historia_nucleo(panel, af_ids, corte, periodo_inicio)
    saldo = _saldo_aprox(panel, af_ids, corte, periodo_inicio)
    externas = _features_externas(tablas_externas, corte, af_ids)
    dim_emp = empleadores[["empleador_id", "rubro", "tamano_empleador"]].rename(
        columns={"empleador_id": "empleador_actual"})

    df = (activos.merge(nucleo, on="afiliado_id", how="left")
          .merge(saldo, on="afiliado_id", how="left")
          .merge(externas, on="afiliado_id", how="left")
          .merge(dim_emp, on="empleador_actual", how="left"))
    df["rubro"] = df["rubro"].fillna("SIN_EMPLEADOR")
    df["tamano_empleador"] = df["tamano_empleador"].fillna("SIN_EMPLEADOR")

    df["edad"] = df["edad_inicio"] + (periodo_to_idx(corte) - periodo_to_idx(periodo_inicio)) // 12
    df["meses_hasta_pension"] = np.maximum(
        (np.where(df["sexo"] == "F", 60, 65) - df["edad"]) * 12, 0)
    df["meses_desde_afiliacion"] = periodo_to_idx(corte) - periodo_to_idx(df["periodo_afiliacion"])

    df["log_renta"] = np.log1p(df["renta"].clip(lower=0))
    df["log_saldo_aprox"] = np.log1p(df["saldo_aprox"].clip(lower=0))
    df["saldo_renta_ratio"] = df["saldo_aprox"] / (df["renta"] * 12.0 + 1.0)
    df["saldo_renta_ratio"] = df["saldo_renta_ratio"].clip(upper=50.0)
    df["log_apv_saldo"] = np.log1p(df["apv_saldo_voluntario"].clip(lower=0))
    df["log_apv_flujo_6m"] = np.log1p(df["apv_flujo_6m"].clip(lower=0))
    df["log_uso_digital_3m"] = np.log1p(df["uso_digital_3m"].clip(lower=0))
    df["log_antiguedad_afp"] = np.log1p(df["antiguedad_afp_meses"].clip(lower=0))
    df["fondo_ord"] = df["fondo"].map(FONDO_ORDEN).fillna(2).astype(float)
    df["es_independiente"] = (df["tipo_afiliado"] == "INDEPENDIENTE").astype(float)
    df["es_voluntario"] = (df["tipo_afiliado"] == "VOLUNTARIO").astype(float)
    df["es_pensionado_activo"] = (df["tipo_afiliado"] == "PENSIONADO_ACTIVO").astype(float)

    rellenos_0 = ["renta", "densidad_12", "densidad_24", "racha_cotiza_m", "n_empleadores_12",
                 "saldo_aprox", "saldo_obligatorio_churn", "antiguedad_afp_meses", "n_cambios_fondo_12",
                 "contactos_12", "reclamos_12", "uso_digital_3m", "apv_saldo_voluntario", "apv_flujo_6m"]
    df[rellenos_0] = df[rellenos_0].fillna(0.0)
    df["fondo"] = df["fondo"].fillna("C")

    for c in FEATURES_CLUSTERING:
        df[c] = df[c].astype(float)

    cols = (["afiliado_id", "periodo_corte"] + FEATURES_PERFIL + FEATURES_CLUSTERING)
    df["periodo_corte"] = corte
    cols = list(dict.fromkeys(cols))   # sin duplicados (renta/saldo_aprox/etc. estan en ambas listas)
    return df[cols].reset_index(drop=True)
