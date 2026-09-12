"""Feature engineering "as-of" para `riesgo_cese_cotizacion` (pandas; sin Spark).

Convencion de tiempos de este proyecto (ver README, seccion 3 y 8):

  * `periodo_corte` (llamado `corte` abajo) es el "hoy" de la fila: el mes en que la AFP
    corre el modelo. Es el mismo rol que `periodo_snapshot` (T) en rim_nowcasting.
  * RIM: la AFP conoce las cotizaciones RECIBIDAS hasta `corte-1`; como el pago normal se
    declara y recibe en `mes+1`, el ultimo mes de remuneracion conocido es `corte-2`
    (`corte_rim` abajo). Este es el ancla de todas las features de historia de RIM
    (`rim_l1` = valor en `corte-2`, `rim_l2` = `corte-3`, ...).
  * Eventos AFC / licencias: se informan mas rapido (en `periodo+1`) y se usan hasta
    `corte` (no `corte-1`): el ultimo evento conocido es de `corte-1` (`afc_termino_l1`,
    `dias_licencia_l1`, ...). Esta es una decision explicita documentada en el README (el
    brief da dos reglas de desfase distintas para RIM vs. eventos; se reproduce tal cual,
    con un corte de recepcion mas laxo para los eventos, que llegan por una via mas rapida
    que la nomina).
  * Prestaciones de cesantia: "conocidas al mes" (sin rezago adicional de recepcion),
    tratadas con el mismo corte que los eventos (`corte`).

Todas las funciones de esta seccion filtran por `periodo_recepcion` antes de agregar: un
registro nunca se usa antes de estar "recibido" segun estas reglas. `construir_features`
NO conoce el panel de verdad ni ninguna tabla mas alla de `corte`; el target se agrega
aparte con `agregar_targets`, que si usa el panel completo (por definicion, un target habla
del futuro del corte).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import add_months, periodo_to_idx

N_LAGS = 12
VENTANA_HISTORIA = 30  # meses de historial (>= 24 de densidad + margen para lags/cambios)

FEATURES_NUMERICAS = (
    [f"rim_l{k}" for k in range(1, N_LAGS + 1)]
    + ["media_3", "media_6", "media_12", "media_24", "std_6", "max_12", "min_12",
       "n_cotiza_3", "n_cotiza_6", "n_cotiza_12", "n_cotiza_24", "meses_historia",
       "racha_cotiza", "racha_cero", "meses_desde_ult_cotiza", "meses_desde_cambio",
       "n_cambios_12", "cv_6", "ratio_l1_l2", "ratio_l1_media12", "tendencia_3_12", "sueldo_fijo",
       "meses_con_empleador", "n_empleadores_12",
       "emp_n_trabajadores", "emp_crecimiento_12", "emp_tasa_salida_12",
       "afc_termino_l1", "afc_termino_l2", "afc_inicio_l1", "afc_inicio_l2",
       "meses_desde_afc_termino", "meses_desde_afc_inicio",
       "dias_licencia_l1", "dias_licencia_l2", "licencia_en_curso",
       "cesantia_giro_actual", "cesantia_n_giros_recibidos", "cesantia_en_curso",
       "edad", "meses_hasta_pension", "meses_desde_afiliacion",
       "mes_corte", "es_enero", "es_marzo", "es_abril", "es_julio", "es_sept", "es_dic",
       "desempleo_corte", "estacionalidad_rubro"]
)
FEATURES_CATEGORICAS = ["tipo_afiliado", "sexo", "nivel_educacional", "region", "rubro", "tamano_empleador",
                        "tipo_contrato"]


# --------------------------------------------------------------- RIM as-of ----
def _rim_conocida_asof(cot: pd.DataFrame, hasta_recepcion: int, periodo_min: int) -> pd.DataFrame:
    """RIM conocida por (afiliado, periodo), sumando pagadores, con el empleador principal
    (el de mayor RIM ese mes) como referencia. Solo movimientos con
    `periodo_min <= periodo <= hasta_recepcion` y `periodo_recepcion <= hasta_recepcion`."""
    vis = cot[(cot["periodo_recepcion"] <= hasta_recepcion)
              & (cot["periodo"] >= periodo_min) & (cot["periodo"] <= hasta_recepcion)]
    if vis.empty:
        return pd.DataFrame(columns=["afiliado_id", "periodo", "rim", "empleador_principal", "n_pagadores"])
    rim = vis.groupby(["afiliado_id", "periodo"])["rim"].sum().rename("rim")
    n_pag = vis.groupby(["afiliado_id", "periodo"]).size().rename("n_pagadores")
    idx_max = vis.groupby(["afiliado_id", "periodo"])["rim"].idxmax()
    principal = (vis.loc[idx_max, ["afiliado_id", "periodo", "empleador_id"]]
                 .set_index(["afiliado_id", "periodo"])["empleador_id"].rename("empleador_principal"))
    return pd.concat([rim, n_pag, principal], axis=1).reset_index()


def _grid_ventana(afiliado_ids: np.ndarray, periodo_afiliacion: pd.Series, corte_rim: int,
                  ventana: int) -> pd.DataFrame:
    """Grilla densa afiliado x mes para los ultimos `ventana` meses hasta `corte_rim`,
    recortada por la fecha de afiliacion de cada uno (sin cross join con meses previos a
    su afiliacion)."""
    periodos = add_months(corte_rim, -np.arange(ventana - 1, -1, -1))
    idx = pd.MultiIndex.from_product([afiliado_ids, periodos], names=["afiliado_id", "periodo"])
    grid = pd.DataFrame(index=idx).reset_index()
    grid = grid.merge(periodo_afiliacion.rename("periodo_afiliacion"), on="afiliado_id", how="left")
    return grid[grid["periodo"] >= grid["periodo_afiliacion"]].drop(columns="periodo_afiliacion")


def _features_historia_rim(cot: pd.DataFrame, afiliados: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Features de historia propia de RIM, evaluadas en `corte-2` (una fila por afiliado)."""
    corte_rim = add_months(corte, -2)
    hasta = add_months(corte, -1)
    periodo_min = add_months(corte_rim, -(VENTANA_HISTORIA - 1))
    conocida = _rim_conocida_asof(cot, hasta, periodo_min)

    af_ids = afiliados["afiliado_id"].to_numpy()
    grid = _grid_ventana(af_ids, afiliados.set_index("afiliado_id")["periodo_afiliacion"], corte_rim,
                        VENTANA_HISTORIA)
    df = (grid.merge(conocida, on=["afiliado_id", "periodo"], how="left")
          .fillna({"rim": 0.0, "n_pagadores": 0}))
    df = df.sort_values(["afiliado_id", "periodo"]).reset_index(drop=True)
    df["cotiza"] = (df["rim"] > 0).astype(int)
    g = df.groupby("afiliado_id", sort=False)

    # cambio de nivel (persistencia rota): sin dato previo, cambio de 0<->positivo, o salto > 2%
    rim_prev = g["rim"].shift(1)
    cambio = ((rim_prev.isna())
              | ((rim_prev == 0) != (df["rim"] == 0))
              | ((rim_prev > 0) & ((df["rim"] - rim_prev).abs() > 0.02 * rim_prev))).astype(int)
    df["_cambio"] = cambio
    df["meses_historia"] = g.cumcount() + 1

    # Solo se necesita la ULTIMA fila (corte_rim) de cada afiliado: en vez de calcular una
    # ventana movil (groupby().rolling(), cara: recalcula limites de ventana en cada fila)
    # se toman las ultimas `w` filas por afiliado (groupby().tail, vectorizado) y se agregan
    # una sola vez. `_pos` = 0 es la fila mas reciente (corte_rim), 1 la anterior, etc.
    t = g.tail(max(N_LAGS, 24)).copy()
    t["_pos"] = t.groupby("afiliado_id", sort=False).cumcount(ascending=False)
    stats = {}
    for w in (3, 6, 12, 24):
        sub = t[t["_pos"] < w]
        gs = sub.groupby("afiliado_id", sort=False)
        stats[f"media_{w}"] = gs["rim"].mean()
        stats[f"n_cotiza_{w}"] = gs["cotiza"].sum()
        if w == 6:
            stats["std_6"] = gs["rim"].std()
        if w == 12:
            stats["max_12"] = gs["rim"].max()
            stats["min_12"] = gs["rim"].min()
            stats["n_empleadores_12"] = gs["empleador_principal"].nunique()
            stats["n_cambios_12"] = gs["_cambio"].sum()
    stats_df = pd.DataFrame(stats)

    lag_sub = t[t["_pos"] < N_LAGS]
    lag_piv = lag_sub.pivot(index="afiliado_id", columns="_pos", values="rim")
    lag_piv.columns = [f"rim_l{c + 1}" for c in lag_piv.columns]

    # racha actual de cotizacion / de ceros (terminando en la ultima fila de cada afiliado):
    # en orden descendente por periodo, el cumprod de la bandera se mantiene en 1 mientras
    # todas las filas mas recientes cumplan la condicion y cae a 0 en la primera que no.
    df_desc = df.sort_values(["afiliado_id", "periodo"], ascending=[True, False])
    df_desc["_no_cotiza"] = 1 - df_desc["cotiza"]
    gd = df_desc.groupby("afiliado_id", sort=False)
    df_desc["_streak_uno"] = gd["cotiza"].cumprod()
    df_desc["_streak_cero"] = gd["_no_cotiza"].cumprod()
    racha_uno = gd["_streak_uno"].sum()
    racha_cero = gd["_streak_cero"].sum()

    # empleador principal actual y su racha (meses seguidos con el mismo empleador)
    emp_prev = g["empleador_principal"].shift(1)
    misma_racha = (df["empleador_principal"].notna()
                  & (df["empleador_principal"] == emp_prev)).fillna(False).astype(int)
    # marca de "inicio de racha con este empleador" (cambia si difiere del anterior o es el primero)
    inicio_racha = 1 - misma_racha
    df["_racha_emp_id"] = inicio_racha.groupby(df["afiliado_id"]).cumsum()
    meses_con_emp = df.groupby(["afiliado_id", "_racha_emp_id"]).cumcount() + 1
    meses_con_emp = np.where(df["empleador_principal"].notna(), meses_con_emp, 0)
    df["meses_con_empleador"] = meses_con_emp

    corte_idx = periodo_to_idx(corte_rim)
    ultima = df.groupby("afiliado_id", sort=False).tail(1).set_index("afiliado_id")
    ultima_cotiza_idx = df.loc[df["cotiza"] == 1].groupby("afiliado_id")["periodo"].max().apply(
        lambda p: periodo_to_idx(p))
    ultimo_cambio_idx = df.loc[df["_cambio"] == 1].groupby("afiliado_id")["periodo"].max().apply(
        lambda p: periodo_to_idx(p))

    out = pd.DataFrame(index=ultima.index)
    out["empleador_principal"] = ultima["empleador_principal"]
    out = out.join(lag_piv).join(stats_df)
    out["meses_historia"] = ultima["meses_historia"]
    out["meses_con_empleador"] = ultima["meses_con_empleador"]
    out["racha_cotiza"] = racha_uno
    out["racha_cero"] = racha_cero
    out["meses_desde_ult_cotiza"] = (corte_idx - ultima_cotiza_idx).reindex(out.index)
    out["meses_desde_cambio"] = (corte_idx - ultimo_cambio_idx).reindex(out.index)
    out["cv_6"] = out["std_6"] / (out["media_6"] + 1.0)
    out["ratio_l1_l2"] = out["rim_l1"] / (out["rim_l2"] + 1.0)
    out["ratio_l1_media12"] = out["rim_l1"] / (out["media_12"] + 1.0)
    out["tendencia_3_12"] = out["media_3"] / (out["media_12"] + 1.0)
    out["sueldo_fijo"] = ((out["rim_l1"] > 0)
                          & ((out["rim_l1"] - out["rim_l2"]).abs() <= 0.005 * out["rim_l1"])
                          & ((out["rim_l2"] - out["rim_l3"]).abs() <= 0.005 * out["rim_l1"].clip(lower=1))
                          ).astype(int)
    return out.reset_index()


def _features_empleador(cot: pd.DataFrame, empleadores: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Tamano y dinamica reciente (12 m) del empleador de referencia, usando SOLO la RIM
    conocida (misma logica as-of que la historia individual)."""
    corte_rim = add_months(corte, -2)
    hasta = add_months(corte, -1)
    periodos_13 = add_months(corte_rim, -np.arange(12, -1, -1))  # corte_rim-12 .. corte_rim
    vis = cot[(cot["periodo_recepcion"] <= hasta) & cot["periodo"].isin(periodos_13) & (cot["rim"] > 0)]
    if vis.empty or vis["empleador_id"].notna().sum() == 0:
        return pd.DataFrame(columns=["empleador_principal", "emp_n_trabajadores", "emp_crecimiento_12",
                                     "emp_tasa_salida_12"])
    hc = (vis.dropna(subset=["empleador_id"])
          .groupby(["empleador_id", "periodo"])["afiliado_id"].nunique().unstack("periodo")
          .reindex(columns=periodos_13, fill_value=0))
    hc_prev = hc.shift(1, axis=1)
    sep = (hc_prev - hc).clip(lower=0)
    tasa_mensual = sep / (hc_prev + 1.0)
    emp = pd.DataFrame({
        "emp_n_trabajadores": hc[corte_rim],
        "emp_crecimiento_12": hc[corte_rim] / (hc[periodos_13[0]] + 1.0),
        "emp_tasa_salida_12": tasa_mensual.iloc[:, 1:].mean(axis=1),
    }).reset_index().rename(columns={"empleador_id": "empleador_principal"})
    return emp


def _features_eventos(afc: pd.DataFrame, licencias: pd.DataFrame, giros: pd.DataFrame,
                      afiliados: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Eventos AFC, licencias y giros de cesantia conocidos hasta `corte` (ver docstring
    del modulo: llegan mas rapido que la RIM, por eso el ultimo mes conocido es `corte-1`
    y no `corte-2`)."""
    af_ids = afiliados["afiliado_id"].to_numpy()
    out = pd.DataFrame({"afiliado_id": af_ids}).set_index("afiliado_id")
    corte_idx = periodo_to_idx(corte)

    # doble resguardo (defensa en profundidad): ademas del filtro por periodo_recepcion, se
    # exige explicitamente que el periodo del evento sea <= corte-1 (para RIM, corte-2). Con
    # los generadores de synthetic.py esto es redundante (periodo_recepcion ya lo implica),
    # pero evita que un dato con recepcion mal etiquetada filtre informacion del futuro.
    corte_menos_1 = add_months(corte, -1)
    afc_k = afc[(afc["periodo_recepcion"] <= corte) & (afc["periodo"] <= corte_menos_1)]
    for tipo, nombre in (("TERMINO", "afc_termino"), ("INICIO", "afc_inicio")):
        sub = afc_k[afc_k["tipo_evento"] == tipo]
        d = corte_idx - periodo_to_idx(sub["periodo"].to_numpy())
        marca = pd.DataFrame({"afiliado_id": sub["afiliado_id"].to_numpy(), "_d": d})
        out[f"{nombre}_l1"] = marca.loc[marca["_d"] == 1].groupby("afiliado_id").size().reindex(out.index, fill_value=0)
        out[f"{nombre}_l2"] = marca.loc[marca["_d"] == 2].groupby("afiliado_id").size().reindex(out.index, fill_value=0)
        ult = marca.groupby("afiliado_id")["_d"].min()
        out[f"meses_desde_{nombre}"] = ult.reindex(out.index)

    # tipo de contrato de la relacion vigente: lo informa el empleador a la AFC al iniciarla
    # (evento INICIO mas reciente conocido), igual que en rim_nowcasting.
    inicio = afc_k[afc_k["tipo_evento"] == "INICIO"].copy()
    inicio["_d"] = corte_idx - periodo_to_idx(inicio["periodo"].to_numpy())
    ult_inicio = inicio.sort_values("_d").groupby("afiliado_id").first()
    out["tipo_contrato_afc"] = ult_inicio["tipo_contrato"].reindex(out.index)

    lic_k = licencias[(licencias["periodo_recepcion"] <= corte)
                     & (licencias["periodo"] <= corte_menos_1)].copy()
    lic_k["_d"] = corte_idx - periodo_to_idx(lic_k["periodo"].to_numpy())
    for d, col in ((1, "dias_licencia_l1"), (2, "dias_licencia_l2")):
        out[col] = (lic_k.loc[lic_k["_d"] == d].groupby("afiliado_id")["dias_licencia"].max()
                    .reindex(out.index, fill_value=0))
    out["licencia_en_curso"] = (out["dias_licencia_l1"] >= 30).astype(int)

    giros_k = giros[(giros["periodo_recepcion"] <= corte) & (giros["periodo"] <= corte)].copy()
    giros_k["_d"] = corte_idx - periodo_to_idx(giros_k["periodo"].to_numpy())
    ultimo_giro = giros_k.loc[giros_k["_d"] >= 0].sort_values("_d").groupby("afiliado_id").first()
    out["cesantia_giro_actual"] = ultimo_giro["monto"].where(ultimo_giro["_d"] <= 1).reindex(out.index).fillna(0.0)
    out["cesantia_en_curso"] = (out["cesantia_giro_actual"] > 0).astype(int)
    n_recibidos = giros_k.groupby("afiliado_id")["n_giro"].max()
    out["cesantia_n_giros_recibidos"] = n_recibidos.reindex(out.index).fillna(0)

    return out.reset_index()


def _estacionalidad_rubro(afc: pd.DataFrame, empleadores: pd.DataFrame, corte: int, mes_corte: int) -> float:
    """Indice de estacionalidad: fraccion de los terminos historicos de cada rubro (conocidos
    hasta `corte`) que ocurren en el mes calendario `mes_corte`. 1/12 si no hay historia."""
    afc_k = afc[(afc["periodo_recepcion"] <= corte) & (afc["periodo"] <= add_months(corte, -1))
               & (afc["tipo_evento"] == "TERMINO")]
    if afc_k.empty:
        return {}
    conteo = afc_k.merge(empleadores[["empleador_id", "rubro"]], on="empleador_id", how="left")
    conteo["mes"] = conteo["periodo"] % 100
    n_rubro_mes = conteo.groupby(["rubro", "mes"]).size()
    n_rubro = conteo.groupby("rubro").size()
    idx = (n_rubro_mes / n_rubro).unstack("mes")
    return idx.get(mes_corte, pd.Series(dtype=float)).reindex(empleadores["rubro"].unique()).fillna(1 / 12).to_dict()


def construir_features(tablas: dict[str, pd.DataFrame], periodo_corte: int) -> pd.DataFrame:
    """Dataset de features "as-of" para `periodo_corte`: una fila por afiliado ya afiliado
    a esa fecha. No usa el panel de verdad ni informacion posterior a `periodo_corte`."""
    afiliados = tablas["afiliados"]
    empleadores = tablas["empleadores"]
    macro = tablas["macro"]
    corte = periodo_corte
    activos = afiliados[afiliados["periodo_afiliacion"] <= corte].copy()

    f_rim = _features_historia_rim(tablas["cotizaciones_conocidas"], activos, corte)
    f_emp = _features_empleador(tablas["cotizaciones_conocidas"], empleadores, corte)
    f_ev = _features_eventos(tablas["afc_eventos"], tablas["licencias"], tablas["prestaciones_cesantia"],
                             activos, corte)

    dim_emp = empleadores[["empleador_id", "rubro", "tamano_empleador", "region_empleador"]] \
        .rename(columns={"empleador_id": "empleador_principal"})

    df = (activos.merge(f_rim, on="afiliado_id", how="left")
          .merge(f_ev, on="afiliado_id", how="left")
          .merge(f_emp, on="empleador_principal", how="left")
          .merge(dim_emp, on="empleador_principal", how="left"))

    mes_corte = corte % 100
    m_corte = macro.set_index("periodo").loc[corte]
    df["mes_corte"] = mes_corte
    for mes, nombre in ((1, "es_enero"), (3, "es_marzo"), (4, "es_abril"), (7, "es_julio"),
                       (9, "es_sept"), (12, "es_dic")):
        df[nombre] = int(mes_corte == mes)
    df["desempleo_corte"] = float(m_corte["desempleo"])
    est = _estacionalidad_rubro(tablas["afc_eventos"], empleadores, corte, mes_corte)
    df["estacionalidad_rubro"] = df["rubro"].map(est).fillna(1 / 12)

    # edad_inicio es la edad al primer periodo simulado (synth_base.generar_afiliados_pandas,
    # igual que en rim_nowcasting.features): se proyecta al corte contando meses desde ahi.
    periodo_inicio = int(macro["periodo"].min())
    df["edad"] = df["edad_inicio"] + (periodo_to_idx(corte) - periodo_to_idx(periodo_inicio)) // 12
    df["meses_hasta_pension"] = (np.where(df["sexo"] == "F", 60, 65) - df["edad"]) * 12
    df["meses_desde_afiliacion"] = periodo_to_idx(corte) - periodo_to_idx(df["periodo_afiliacion"])

    # tipo de contrato de la relacion vigente: si no hay empleador de referencia, "SIN_EMPLEADOR";
    # si lo hay pero la relacion empezo antes del inicio de la historia disponible (sin aviso
    # AFC de inicio conocido), "SIN_DATO_AFC" (mismo criterio que rim_nowcasting).
    df["tipo_contrato"] = np.where(df["empleador_principal"].isna(), "SIN_EMPLEADOR",
                                   df["tipo_contrato_afc"].fillna("SIN_DATO_AFC"))
    df["rubro"] = df["rubro"].fillna("SIN_EMPLEADOR")
    df["tamano_empleador"] = df["tamano_empleador"].fillna("SIN_EMPLEADOR")

    rellenos_0 = ([f"rim_l{k}" for k in range(1, N_LAGS + 1)]
                  + ["media_3", "media_6", "media_12", "media_24", "std_6", "max_12", "min_12",
                     "n_cotiza_3", "n_cotiza_6", "n_cotiza_12", "n_cotiza_24", "meses_historia",
                     "racha_cotiza", "racha_cero", "n_cambios_12", "meses_con_empleador",
                     "n_empleadores_12", "sueldo_fijo", "cv_6", "ratio_l1_l2", "ratio_l1_media12",
                     "tendencia_3_12", "emp_n_trabajadores", "emp_crecimiento_12", "emp_tasa_salida_12",
                     "afc_termino_l1", "afc_termino_l2", "afc_inicio_l1", "afc_inicio_l2",
                     "dias_licencia_l1", "dias_licencia_l2", "licencia_en_curso",
                     "cesantia_giro_actual", "cesantia_n_giros_recibidos", "cesantia_en_curso"])
    df[rellenos_0] = df[rellenos_0].fillna(0.0)
    df["meses_desde_ult_cotiza"] = df["meses_desde_ult_cotiza"].fillna(99).clip(upper=99)
    df["meses_desde_cambio"] = df["meses_desde_cambio"].fillna(99).clip(upper=99)
    df["meses_desde_afc_termino"] = df["meses_desde_afc_termino"].fillna(99).clip(upper=99)
    df["meses_desde_afc_inicio"] = df["meses_desde_afc_inicio"].fillna(99).clip(upper=99)

    df["periodo_corte"] = corte
    df["corte_rim"] = add_months(corte, -2)

    for c in FEATURES_NUMERICAS:
        df[c] = df[c].astype(float)

    # -------- poblaciones de los dos modelos (definicion 100% as-of, sin mirar el futuro) --
    df["pop_cese"] = (df["rim_l1"] > 0).astype(int)
    df["pop_reingreso"] = ((df["tipo_afiliado"] == "DEPENDIENTE") & (df["rim_l1"] == 0)
                           & (df["n_cotiza_24"] > 0)).astype(int)

    cols = (["afiliado_id", "periodo_corte", "corte_rim", "pop_cese", "pop_reingreso",
             "tipo_afiliado", "sexo", "nivel_educacional", "region", "rubro", "tamano_empleador",
             "tipo_contrato"]
            + FEATURES_NUMERICAS)
    return df[cols].reset_index(drop=True)


def agregar_targets(features: pd.DataFrame, panel: pd.DataFrame, periodo_fin: int) -> pd.DataFrame:
    """Agrega y3 / y6 / y_reingreso usando el panel COMPLETO (verdad), nunca al construir
    features: el target es, por definicion, informacion del futuro del corte."""
    rim_true = (panel.assign(rim=lambda d: d["rim_emp1"].fillna(0) + d["rim_emp2"].fillna(0))
                [["afiliado_id", "periodo", "rim"]])
    rim_wide = rim_true.pivot_table(index="afiliado_id", columns="periodo", values="rim", fill_value=0.0)

    out = features.copy()
    corte = out["periodo_corte"].to_numpy()
    af = out["afiliado_id"].to_numpy()

    # Se agrupa por valor unico de corte (son pocos) en vez de reindexar fila a fila.
    y3 = np.full(len(out), np.nan)
    y6 = np.full(len(out), np.nan)
    yre = np.full(len(out), np.nan)
    horiz3_ok = np.zeros(len(out), dtype=bool)
    horiz6_ok = np.zeros(len(out), dtype=bool)
    for c in np.unique(corte):
        m = corte == c
        af_c = af[m]
        periodos_h3 = [add_months(int(c), h) for h in (1, 2, 3)]
        periodos_h6 = [add_months(int(c), h) for h in (1, 2, 3, 4, 5, 6)]
        ok3 = periodos_h3[-1] <= periodo_fin
        ok6 = periodos_h6[-1] <= periodo_fin
        horiz3_ok[m] = ok3
        horiz6_ok[m] = ok6
        cols3 = [p for p in periodos_h3 if p in rim_wide.columns]
        cols6 = [p for p in periodos_h6 if p in rim_wide.columns]
        sub3 = rim_wide.reindex(index=af_c, columns=cols3).fillna(0.0)
        sub6 = rim_wide.reindex(index=af_c, columns=cols6).fillna(0.0)
        if ok3:
            y3[m] = (sub3.to_numpy() <= 0).all(axis=1).astype(float)
        if ok6:
            y6[m] = (sub6.to_numpy() <= 0).all(axis=1).astype(float)
            yre[m] = (sub6.to_numpy() > 0).any(axis=1).astype(float)

    out["y3"] = y3
    out["y6"] = y6
    out["y_reingreso"] = yre
    out["horizonte3_completo"] = horiz3_ok
    out["horizonte6_completo"] = horiz6_ok
    return out
