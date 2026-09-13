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
  * Cobertura del seguro de cesantia: se reconstruye de forma puramente analitica a partir
    del TERMINO AFC conocido (<= corte-1), el tipo de contrato de esa relacion y las tasas
    legales (`synthetic.TASAS_GIRO`): `giros_restantes = max_giros - meses_desde_termino`.
    **No** se lee la tabla `prestaciones_cesantia` para features (ver correccion de ronda 1
    mas abajo): esa tabla decide si un giro se paga mirando si el afiliado ya reingreso
    (RIM real, no conocida), y usarla directamente filtraba informacion del futuro del
    corte a traves de `cesantia_giro_actual`/`cesantia_n_giros_recibidos` (bloqueante #1 de
    la revision de ronda 1, ver README §8). La cobertura analitica de este modulo asume
    "sigue cesante mientras no se demuestre lo contrario con datos ya conocidos" (RIM
    conocida, eventos AFC), exactamente el mismo principio as-of que el resto del proyecto:
    `construir_features` apaga `cesantia_en_curso`/`cesantia_giros_restantes`/
    `cesantia_tasa_actual` si hay un INICIO AFC conocido posterior al termino, o RIM
    conocida > 0 en algun mes estrictamente posterior al termino (menor G, ronda 2 de
    revision: antes esto quedaba solo como intencion en este parrafo, sin implementarse).

Todas las funciones de esta seccion filtran por `periodo_recepcion` antes de agregar: un
registro nunca se usa antes de estar "recibido" segun estas reglas. `construir_features`
NO conoce el panel de verdad ni ninguna tabla mas alla de `corte`; el target se agrega
aparte con `agregar_targets`, que si usa el panel completo (por definicion, un target habla
del futuro del corte). Tampoco se le pasa `prestaciones_cesantia` (ver parrafo anterior).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic import (P_TARDIO, TARDIO_EXTRA_MAX, TARDIO_EXTRA_MIN, TASAS_GIRO, add_months,
                        periodo_to_idx)

N_LAGS = 12
# Ventana de historial: 24 meses de densidad (el maximo que pide el brief) + 2 de margen
# para poder calcular `_cambio` (necesita el mes anterior al primero de la ventana) y para
# que `racha_cotiza`/`meses_desde_cambio` no se corten justo en el borde de `n_cotiza_24`.
# Se redujo de 30 a 26 meses (menor #7, ronda 1: con 30 meses + el desfase de 2 de la RIM,
# los cortes mas cercanos al inicio de la simulacion pedian meses anteriores a
# `periodo_inicio`, que no existen: la ventana se rellenaba con ceros "artificiales"
# indistinguibles de cesantia real, sesgando distinto cada corte). Ahora
# `construir_features` recorta explicitamente la ventana a `periodo_inicio` (ver
# `_features_historia_rim`) y `run.py` solo usa cortes donde la ventana completa cabe
# dentro del rango simulado, asi que el recorte no debiera activarse en una corrida normal;
# queda como resguardo.
VENTANA_HISTORIA = 26

FEATURES_NUMERICAS = (
    [f"rim_l{k}" for k in range(1, N_LAGS + 1)]
    + ["media_3", "media_6", "media_12", "media_24", "std_6", "max_12", "min_12",
       "n_cotiza_3", "n_cotiza_6", "n_cotiza_12", "n_cotiza_24", "meses_historia",
       "racha_cotiza", "racha_cero", "meses_desde_ult_cotiza", "meses_desde_cambio",
       "n_cambios_12", "cv_6", "ratio_l1_l2", "ratio_l1_media12", "tendencia_3_12", "sueldo_fijo",
       "meses_con_empleador", "n_empleadores_12",
       "emp_n_trabajadores", "emp_crecimiento_12", "emp_tasa_salida_12", "emp_sin_planilla_conocida",
       "afc_termino_l1", "afc_termino_l2", "afc_inicio_l1", "afc_inicio_l2",
       "meses_desde_afc_termino", "meses_desde_afc_inicio",
       "dias_licencia_l1", "dias_licencia_l2", "licencia_en_curso",
       "cesantia_en_curso", "cesantia_giros_restantes", "cesantia_tasa_actual",
       "edad", "meses_hasta_pension", "meses_desde_afiliacion",
       "mes_corte", "es_enero", "es_marzo", "es_abril", "es_julio", "es_sept", "es_dic",
       "desempleo_corte", "estacionalidad_rubro", "estacionalidad_rubro_ultimo_empleo"]
)
FEATURES_CATEGORICAS = ["tipo_afiliado", "sexo", "nivel_educacional", "region", "rubro", "tamano_empleador",
                        "tipo_contrato", "rubro_ultimo_empleo", "tamano_ultimo_empleo"]
# Corr. ronda 4 (correccion_04.md, ver `_features_historia_rim`): rubro/tamano/estacionalidad
# de la ULTIMA relacion laboral conocida (empleada o no HOY), a diferencia de
# `rubro`/`tamano_empleador`/`estacionalidad_rubro` que describen el empleador VIGENTE y
# caen siempre a "SIN_EMPLEADOR"/1-12 constante para toda la poblacion `pop_reingreso` (nadie
# cotiza hoy, por definicion de esa poblacion). Estas dos son iguales a las de arriba cuando
# el afiliado SI cotiza hoy (mismo empleador), y solo aportan senal nueva cuando esta
# cesante.

# ronda 2 (BRIEF_RONDA2.md #2): codigos de movimiento de personal de la planilla Previred
# sintetica (afp_lab/common/planilla.py), con el MISMO desfase que la RIM (l1 = corte-2,
# l2 = corte-3: la planilla es la propia declaracion de la RIM, no llega mas rapido que ella
# como los eventos AFC, que se conocen hasta corte-1). Se mantienen SEPARADAS de
# FEATURES_NUMERICAS (no entran al modelo por defecto) para poder medir su aporte marginal
# por ablation en `run.py` sin cambiar el modelo de la ronda 1; ver README "Ronda 2" para el
# resultado y la decision tomada sobre si se incorporan o no al modelo de produccion.
CODIGOS_PLANILLA_FEATURES = {2: "retiro", 3: "subsidio", 4: "permiso", 7: "contrato_nuevo",
                             8: "cambio_indefinido"}
FEATURES_PLANILLA = [f"mov_{nombre}_l{k}" for nombre in CODIGOS_PLANILLA_FEATURES.values() for k in (1, 2)]


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


def _features_historia_rim(cot: pd.DataFrame, afiliados: pd.DataFrame, corte: int,
                           periodo_inicio: int) -> pd.DataFrame:
    """Features de historia propia de RIM, evaluadas en `corte-2` (una fila por afiliado)."""
    corte_rim = add_months(corte, -2)
    hasta = add_months(corte, -1)
    # la ventana nunca mira antes de periodo_inicio (menor #7, ronda 1): antes de ese mes no
    # hay universo simulado, y rellenar con "sin cotizacion" ahi seria un cero artificial.
    periodo_min = max(add_months(corte_rim, -(VENTANA_HISTORIA - 1)), periodo_inicio)
    conocida = _rim_conocida_asof(cot, hasta, periodo_min)
    n_meses_grid = periodo_to_idx(corte_rim) - periodo_to_idx(periodo_min) + 1

    af_ids = afiliados["afiliado_id"].to_numpy()
    grid = _grid_ventana(af_ids, afiliados.set_index("afiliado_id")["periodo_afiliacion"], corte_rim,
                        n_meses_grid)
    df = (grid.merge(conocida, on=["afiliado_id", "periodo"], how="left")
          .fillna({"rim": 0.0, "n_pagadores": 0}))
    df = df.sort_values(["afiliado_id", "periodo"]).reset_index(drop=True)
    df["cotiza"] = (df["rim"] > 0).astype(int)
    # Corr. ronda 4 (correccion_04.md): `empleador_principal` solo existe en meses con RIM
    # conocida > 0 (por construccion de `_rim_conocida_asof`), asi que para alguien cesante
    # EN `corte_rim` (rim_l1 == 0: toda la poblacion `pop_reingreso`) queda NaN y las
    # features de rubro/tamano/estacionalidad derivadas de el (mas abajo, en
    # `construir_features`) caen siempre a "SIN_EMPLEADOR"/1-12 constante: el modelo de
    # reingreso queda ciego al rubro de la ULTIMA relacion laboral, justo la variable que en
    # el generador (rim_nowcasting.synthetic._simular_particion, SEASON_HIRE[rubro_pref,
    # mes]) determina la estacionalidad de recontratacion de cada persona. `empleador_ref_hist`
    # es el ultimo empleador conocido (con RIM > 0) hasta cada mes, propagado hacia adelante
    # (ffill) dentro de la ventana: para quien sigue cotizando coincide con
    # `empleador_principal`; para quien esta cesante, es el empleador de la relacion que
    # perdio (as-of: solo usa RIM ya conocida a `corte-1`, igual que el resto de esta
    # ventana).
    df["empleador_ref_hist"] = df.groupby("afiliado_id", sort=False)["empleador_principal"].ffill()
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
    out["empleador_ref_hist"] = ultima["empleador_ref_hist"]
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


def _maturity_recepcion(elapsed: np.ndarray) -> np.ndarray:
    """Fraccion esperada de declaraciones ya recibidas a los `elapsed` meses de emitida la
    remuneracion, con los mismos parametros de desfase de `synthetic.py` (95 % oportuno a 1
    mes, 5 % tardio uniforme entre `TARDIO_EXTRA_MIN` y `TARDIO_EXTRA_MAX` meses
    adicionales). Corrige el sesgo de censura de recepcion en los agregados por empleador
    (menor #9, ronda 1): sin esto, el mes mas reciente de la ventana (elapsed=1, `corte-2`)
    solo alcanza a ver el pago oportuno (~95 % de las planillas), mientras que un mes con
    13 meses de maduracion ya vio practicamente el 100 %; comparar ambos sin corregir hace
    parecer que TODOS los empleadores encogen ~5 % cada 12 meses (mediana de
    `emp_crecimiento_12` ~0,90 observada en la ronda 1), un artefacto de recepcion, no una
    señal real de dotacion."""
    e = np.asarray(elapsed, dtype=float)
    p_oportuno = 1.0 - P_TARDIO
    n_extra = TARDIO_EXTRA_MAX - TARDIO_EXTRA_MIN + 1
    frac_tardio_conocido = np.clip(e - TARDIO_EXTRA_MIN, 0, n_extra) / n_extra
    return np.where(e >= 1, p_oportuno, 0.0) + P_TARDIO * frac_tardio_conocido


def _features_empleador(cot: pd.DataFrame, empleadores: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Tamano y dinamica reciente (12 m) del empleador de referencia, usando SOLO la RIM
    conocida (misma logica as-of que la historia individual), corregida por madurez de
    recepcion (`_maturity_recepcion`)."""
    corte_rim = add_months(corte, -2)
    hasta = add_months(corte, -1)
    periodos_13 = list(add_months(corte_rim, -np.arange(12, -1, -1)))  # corte_rim-12 .. corte_rim
    cols_vacias = ["empleador_principal", "emp_n_trabajadores", "emp_crecimiento_12",
                  "emp_tasa_salida_12", "emp_sin_planilla_conocida"]
    vis = cot[(cot["periodo_recepcion"] <= hasta) & cot["periodo"].isin(periodos_13) & (cot["rim"] > 0)]
    if vis.empty or vis["empleador_id"].notna().sum() == 0:
        return pd.DataFrame(columns=cols_vacias)

    # bug de ronda 1: `reindex(..., fill_value=0)` solo rellena columnas/filas AUSENTES del
    # index, no los NaN que `unstack` ya deja cuando un empleador conocido no tiene planilla
    # conocida en un mes puntual de la ventana (quedaban como NaN y arrastraban NaN a
    # emp_crecimiento_12/emp_tasa_salida_12 para esos empleadores). Se rellena
    # explicitamente con `.fillna(0)`.
    hc_raw = (vis.dropna(subset=["empleador_id"])
              .groupby(["empleador_id", "periodo"])["afiliado_id"].nunique()
              .unstack("periodo").reindex(columns=periodos_13))
    # bug de ronda 2: `hc_raw[corte_rim]` NUNCA es NaN para un empleador que llega hasta
    # aca, porque `empleador_principal` (la clave de join) se define, mas arriba, como el
    # empleador con RIM > 0 EN corte_rim para el propio afiliado — asi que ese empleador
    # SIEMPRE tiene planilla conocida justo en corte_rim (es circular). La bandera util es
    # si falta planilla conocida en ALGUN OTRO mes de la ventana de 13 meses (indicio de
    # que la cobertura de la RIM conocida de ese empleador es incompleta, no solo hoy).
    sin_planilla_alguna = hc_raw.isna().any(axis=1)
    hc = hc_raw.fillna(0.0)

    elapsed = periodo_to_idx(hasta) - periodo_to_idx(np.array(periodos_13))
    maturity = pd.Series(_maturity_recepcion(elapsed), index=periodos_13)
    hc_corr = hc.div(maturity, axis=1)

    hc_prev = hc_corr.shift(1, axis=1)
    sep = (hc_prev - hc_corr).clip(lower=0)
    tasa_mensual = sep / (hc_prev + 1.0)
    emp = pd.DataFrame({
        "emp_n_trabajadores": hc_corr[corte_rim],
        "emp_crecimiento_12": hc_corr[corte_rim] / (hc_corr[periodos_13[0]] + 1.0),
        "emp_tasa_salida_12": tasa_mensual.iloc[:, 1:].mean(axis=1),
        "emp_sin_planilla_conocida": sin_planilla_alguna.astype(int),
    }).reset_index().rename(columns={"empleador_id": "empleador_principal"})
    return emp


def _features_eventos(afc: pd.DataFrame, licencias: pd.DataFrame, afiliados: pd.DataFrame,
                      corte: int) -> pd.DataFrame:
    """Eventos AFC y licencias conocidos hasta `corte` (ver docstring del modulo: llegan
    mas rapido que la RIM, por eso el ultimo mes conocido es `corte-1` y no `corte-2`), mas
    la cobertura de cesantia derivada analiticamente del ultimo TERMINO conocido (sin leer
    `prestaciones_cesantia`: ver correccion de leakage en el docstring del modulo)."""
    af_ids = afiliados["afiliado_id"].to_numpy()
    out = pd.DataFrame({"afiliado_id": af_ids}).set_index("afiliado_id")
    corte_idx = periodo_to_idx(corte)

    # doble resguardo (defensa en profundidad): ademas del filtro por periodo_recepcion, se
    # exige explicitamente que el periodo del evento sea <= corte-1 (para RIM, corte-2). Con
    # los generadores de synthetic.py esto es redundante (periodo_recepcion ya lo implica),
    # pero evita que un dato con recepcion mal etiquetada filtre informacion del futuro.
    corte_menos_1 = add_months(corte, -1)
    afc_k = afc[(afc["periodo_recepcion"] <= corte) & (afc["periodo"] <= corte_menos_1)]
    marcas_termino = None
    for tipo, nombre in (("TERMINO", "afc_termino"), ("INICIO", "afc_inicio")):
        sub = afc_k[afc_k["tipo_evento"] == tipo]
        d = corte_idx - periodo_to_idx(sub["periodo"].to_numpy())
        marca = pd.DataFrame({"afiliado_id": sub["afiliado_id"].to_numpy(), "_d": d,
                              "tipo_contrato": sub["tipo_contrato"].to_numpy()})
        out[f"{nombre}_l1"] = marca.loc[marca["_d"] == 1].groupby("afiliado_id").size().reindex(out.index, fill_value=0)
        out[f"{nombre}_l2"] = marca.loc[marca["_d"] == 2].groupby("afiliado_id").size().reindex(out.index, fill_value=0)
        ult = marca.groupby("afiliado_id")["_d"].min()
        out[f"meses_desde_{nombre}"] = ult.reindex(out.index)
        if tipo == "TERMINO":
            marcas_termino = marca

    # Cobertura del seguro de cesantia, 100% analitica: max_giros segun el tipo de contrato
    # DEL TERMINO mas reciente conocido (no del INICIO: la relacion terminada puede ser
    # distinta de la vigente) y `meses_desde_termino` (ya calculado arriba). No se lee RIM
    # futura ni la tabla `prestaciones_cesantia` (ver docstring del modulo). Asume "sigue
    # cesante" salvo evidencia YA CONOCIDA de reingreso (un INICIO posterior al termino, o
    # RIM conocida > 0 en un mes posterior al termino): esa correccion se aplica mas abajo,
    # en `construir_features`, porque necesita `rim_l1..rim_l3` (calculadas en
    # `_features_historia_rim`, no disponibles en esta funcion).
    ult_termino = marcas_termino.sort_values("_d").groupby("afiliado_id").first() if marcas_termino is not None else None
    d_termino = (ult_termino["_d"].reindex(out.index) if ult_termino is not None
                else pd.Series(np.nan, index=out.index))
    tipo_termino = (ult_termino["tipo_contrato"].reindex(out.index) if ult_termino is not None
                   else pd.Series(np.nan, index=out.index, dtype=object))
    max_giros_map = {t: len(tasas) for t, tasas in TASAS_GIRO.items()}
    max_giros = tipo_termino.map(max_giros_map)
    en_curso = d_termino.notna() & (d_termino >= 1) & (d_termino <= max_giros)
    out["cesantia_en_curso"] = en_curso.fillna(False).astype(int)
    out["cesantia_giros_restantes"] = np.where(en_curso, max_giros - d_termino, 0.0)
    tasa_actual = np.zeros(len(out))
    for tipo_contrato, tasas in TASAS_GIRO.items():
        tasas_arr = np.asarray(tasas)
        m = (en_curso & (tipo_termino == tipo_contrato)).to_numpy()
        if m.any():
            pos = (d_termino.to_numpy()[m] - 1).astype(int)
            tasa_actual[m] = tasas_arr[pos]
    out["cesantia_tasa_actual"] = tasa_actual

    lic_k = licencias[(licencias["periodo_recepcion"] <= corte)
                     & (licencias["periodo"] <= corte_menos_1)].copy()
    lic_k["_d"] = corte_idx - periodo_to_idx(lic_k["periodo"].to_numpy())
    for d, col in ((1, "dias_licencia_l1"), (2, "dias_licencia_l2")):
        out[col] = (lic_k.loc[lic_k["_d"] == d].groupby("afiliado_id")["dias_licencia"].max()
                    .reindex(out.index, fill_value=0))
    out["licencia_en_curso"] = (out["dias_licencia_l1"] >= 30).astype(int)

    return out.reset_index()


def _features_planilla(planilla: pd.DataFrame | None, af_ids: np.ndarray, corte: int) -> pd.DataFrame:
    """Codigos de movimiento de personal de la planilla (`CODIGOS_PLANILLA_FEATURES`), con el
    MISMO desfase que la RIM (ancla en `corte_rim = corte-2`, conocidos hasta `corte-1`; ver
    docstring del modulo y de `FEATURES_PLANILLA`). Devuelve 0 en todas las columnas si no hay
    planilla disponible (integracion opcional, BRIEF_RONDA2.md #2: "si ya existe")."""
    out = pd.DataFrame({"afiliado_id": af_ids}).set_index("afiliado_id")
    for col in FEATURES_PLANILLA:
        out[col] = 0.0
    if planilla is None or planilla.empty:
        return out.reset_index()

    corte_rim = add_months(corte, -2)
    hasta = add_months(corte, -1)
    periodos_ok = [add_months(corte_rim, -1), corte_rim]   # corte-3 (l2), corte-2 (l1)
    decl = planilla[(planilla["tipo_movimiento"] == "DECLARACION")
                    & (planilla["periodo_recepcion"] <= hasta)
                    & (planilla["periodo"].isin(periodos_ok))
                    & (planilla["codigo_movimiento_personal"].isin(CODIGOS_PLANILLA_FEATURES))]
    if decl.empty:
        return out.reset_index()

    _d = periodo_to_idx(corte_rim) - periodo_to_idx(decl["periodo"].to_numpy())   # 0->l1, 1->l2
    marca = pd.DataFrame({"afiliado_id": decl["rut_trabajador"].to_numpy(),
                          "codigo": decl["codigo_movimiento_personal"].to_numpy(), "_d": _d})
    for codigo, nombre in CODIGOS_PLANILLA_FEATURES.items():
        for dd, suf in ((0, "l1"), (1, "l2")):
            sub = marca[(marca["codigo"] == codigo) & (marca["_d"] == dd)]
            presente = sub.groupby("afiliado_id").size().clip(upper=1).reindex(out.index, fill_value=0)
            out[f"mov_{nombre}_{suf}"] = presente.astype(float)
    return out.reset_index()


def _tipo_contrato_por_relacion(afc: pd.DataFrame, corte: int) -> pd.DataFrame:
    """Tipo de contrato del aviso INICIO mas reciente conocido, por cada (afiliado,
    empleador). Se une en `construir_features` contra el empleador de referencia VIGENTE
    (el de la RIM conocida), no contra cualquier inicio pasado aunque esa relacion ya haya
    terminado (menor #8, ronda 1: antes se tomaba el INICIO mas reciente del afiliado sin
    importar el empleador, y podia describir una relacion ya cerrada)."""
    corte_idx = periodo_to_idx(corte)
    corte_menos_1 = add_months(corte, -1)
    inicio = afc[(afc["tipo_evento"] == "INICIO") & (afc["periodo_recepcion"] <= corte)
                & (afc["periodo"] <= corte_menos_1)].copy()
    if inicio.empty:
        return pd.DataFrame(columns=["afiliado_id", "empleador_id", "tipo_contrato_vigente"])
    inicio["_d"] = corte_idx - periodo_to_idx(inicio["periodo"].to_numpy())
    ult = (inicio.sort_values("_d").groupby(["afiliado_id", "empleador_id"]).first()
           .rename(columns={"tipo_contrato": "tipo_contrato_vigente"}))
    return ult[["tipo_contrato_vigente"]].reset_index()


def _estacionalidad_rubro(afc: pd.DataFrame, empleadores: pd.DataFrame, corte: int) -> dict[str, float]:
    """Indice de estacionalidad: fraccion de los terminos historicos de cada rubro
    (conocidos hasta `corte`) que ocurren, en promedio, en los meses calendario de la
    VENTANA DE TARGET `corte+1..corte+6` (no en el mes del corte mismo: menor #13, ronda 1
    — el riesgo estacional relevante es el de los meses que se estan prediciendo, y una
    misma fila alimenta tanto el horizonte de 3 como el de 6 meses). 1/12 (uniforme) si no
    hay historia para un rubro."""
    afc_k = afc[(afc["periodo_recepcion"] <= corte) & (afc["periodo"] <= add_months(corte, -1))
               & (afc["tipo_evento"] == "TERMINO")]
    meses_objetivo = [add_months(corte, h) % 100 for h in range(1, 7)]
    if afc_k.empty:
        return {r: 1 / 12 for r in empleadores["rubro"].unique()}
    conteo = afc_k.merge(empleadores[["empleador_id", "rubro"]], on="empleador_id", how="left")
    conteo["mes"] = conteo["periodo"] % 100
    n_rubro_mes = conteo.groupby(["rubro", "mes"]).size()
    n_rubro = conteo.groupby("rubro").size()
    idx = (n_rubro_mes / n_rubro).unstack("mes").reindex(columns=range(1, 13))
    promedio = idx[meses_objetivo].mean(axis=1)
    return promedio.reindex(empleadores["rubro"].unique()).fillna(1 / 12).to_dict()


# Tipos de afiliado que se comportan como una relacion de dependencia (RIM mensual movida
# por empleo/desempleo real, sujeta a AFC): ambos entran al Modelo A (cese). INDEPENDIENTE
# y VOLUNTARIO quedan fuera de los dos modelos (ver `_poblaciones` y README §8, bloqueante
# #2 de la revision de ronda 1): en este universo su RIM es esencialmente aleatoria mes a
# mes (voluntario) o constante dentro del anio calendario y se redibuja cada enero
# (independiente, `synthetic._simular_particion`), asi que "cesar"/"reingresar" es, para
# ellos, un fenomeno distinto (un sorteo anual o mensual, no una perdida de empleo) que un
# baseline ingenuo por rubro x mes replica mejor que cualquier feature de historia laboral.
TIPOS_DEPENDENCIA = ["DEPENDIENTE", "PENSIONADO_ACTIVO"]
# Los que quedan fuera de ambos modelos (ver comentario de arriba). Se exponen aca (y no
# solo como el complemento implicito) porque `run._negocio_agregado_cartera_completa`
# necesita proyectarlos por separado con un baseline propio (mayor A, ronda 2 de revision:
# el agregado de negocio del modelo solo, sin ellos, cubre ~88% de los ceses reales de la
# cartera completa).
TIPOS_EXCLUIDOS = ["INDEPENDIENTE", "VOLUNTARIO"]


def construir_features(tablas: dict[str, pd.DataFrame], periodo_corte: int) -> pd.DataFrame:
    """Dataset de features "as-of" para `periodo_corte`: una fila por afiliado ya afiliado
    a esa fecha. No usa el panel de verdad ni informacion posterior a `periodo_corte`."""
    afiliados = tablas["afiliados"]
    empleadores = tablas["empleadores"]
    macro = tablas["macro"]
    corte = periodo_corte
    activos = afiliados[afiliados["periodo_afiliacion"] <= corte].copy()
    # edad_inicio es la edad al primer periodo simulado (synth_base.generar_afiliados_pandas,
    # igual que en rim_nowcasting.features): se proyecta al corte contando meses desde ahi.
    periodo_inicio = int(macro["periodo"].min())

    f_rim = _features_historia_rim(tablas["cotizaciones_conocidas"], activos, corte, periodo_inicio)
    f_emp = _features_empleador(tablas["cotizaciones_conocidas"], empleadores, corte)
    f_ev = _features_eventos(tablas["afc_eventos"], tablas["licencias"], activos, corte)
    f_tc = _tipo_contrato_por_relacion(tablas["afc_eventos"], corte)
    f_mov = _features_planilla(tablas.get("planilla"), activos["afiliado_id"].to_numpy(), corte)

    dim_emp = empleadores[["empleador_id", "rubro", "tamano_empleador", "region_empleador"]] \
        .rename(columns={"empleador_id": "empleador_principal"})
    # Corr. ronda 4 (ver `_features_historia_rim`): rubro/tamano de la ULTIMA relacion
    # laboral conocida (`empleador_ref_hist`, ffill de `empleador_principal`), no solo la
    # vigente. Para quien SI cotiza hoy es el mismo empleador (mismo valor que `rubro`); para
    # `pop_reingreso` (nadie cotiza hoy) es la unica fuente de rubro/tamano disponible.
    dim_emp_hist = empleadores[["empleador_id", "rubro", "tamano_empleador"]].rename(
        columns={"empleador_id": "empleador_ref_hist", "rubro": "rubro_ultimo_empleo",
                 "tamano_empleador": "tamano_ultimo_empleo"})

    df = (activos.merge(f_rim, on="afiliado_id", how="left")
          .merge(f_ev, on="afiliado_id", how="left")
          .merge(f_mov, on="afiliado_id", how="left")
          .merge(f_emp, on="empleador_principal", how="left")
          .merge(dim_emp, on="empleador_principal", how="left")
          .merge(dim_emp_hist, on="empleador_ref_hist", how="left")
          # tipo de contrato atado a la relacion VIGENTE (afiliado + su empleador de
          # referencia), no a cualquier inicio pasado del afiliado (menor #8, ronda 1)
          .merge(f_tc.rename(columns={"empleador_id": "empleador_principal"}),
                 on=["afiliado_id", "empleador_principal"], how="left"))

    # Menor G (ronda 2 de revision): la cobertura de cesantia (punto 1, "sin leer RIM
    # futura") asume que el afiliado sigue cesante mientras no haya evidencia YA CONOCIDA
    # de lo contrario. Esa evidencia son dos: (a) un aviso AFC de INICIO conocido con
    # periodo posterior al del ultimo TERMINO (reingreso a cualquier empleador), o (b) RIM
    # conocida > 0 en algun mes ESTRICTAMENTE POSTERIOR al termino y anterior o igual a
    # `corte-2` (el propio `rim_l1` solo cae en ese rango si el termino fue hace >= 3
    # meses: para un termino en `corte-1` o `corte-2`, `rim_l1` describe el sueldo ANTES
    # de terminar, no una senal de reingreso, y no debe apagar nada). Ambas usan solo
    # columnas ya conocidas a `corte-1` (eventos) o `corte-2` (RIM): no leen el futuro.
    d_term = df["meses_desde_afc_termino"]     # NaN si nunca hubo termino conocido
    d_inicio = df["meses_desde_afc_inicio"]
    hay_inicio_posterior = d_inicio.notna() & d_term.notna() & (d_inicio < d_term)
    rim_tras_termino = pd.Series(False, index=df.index)
    for k in (1, 2, 3):                        # rim_lk cae despues del termino solo si k <= d_term-2
        aplica = d_term.notna() & ((d_term - 2) >= k)
        rim_tras_termino |= aplica & (df[f"rim_l{k}"] > 0)
    ya_reingreso_conocido = hay_inicio_posterior.fillna(False) | rim_tras_termino
    df.loc[ya_reingreso_conocido, ["cesantia_en_curso", "cesantia_giros_restantes", "cesantia_tasa_actual"]] = 0.0

    mes_corte = corte % 100
    m_corte = macro.set_index("periodo").loc[corte]
    df["mes_corte"] = mes_corte
    for mes, nombre in ((1, "es_enero"), (3, "es_marzo"), (4, "es_abril"), (7, "es_julio"),
                       (9, "es_sept"), (12, "es_dic")):
        df[nombre] = int(mes_corte == mes)
    df["desempleo_corte"] = float(m_corte["desempleo"])
    est = _estacionalidad_rubro(tablas["afc_eventos"], empleadores, corte)
    df["estacionalidad_rubro"] = df["rubro"].map(est).fillna(1 / 12)
    # Corr. ronda 4: mismo indice de estacionalidad, pero del rubro de la ULTIMA relacion
    # laboral (`rubro_ultimo_empleo`), no del empleador vigente (siempre "SIN_EMPLEADOR" en
    # `pop_reingreso`, ver comentario en la construccion de `dim_emp_hist` arriba).
    df["rubro_ultimo_empleo"] = df["rubro_ultimo_empleo"].fillna("SIN_EMPLEADOR")
    df["tamano_ultimo_empleo"] = df["tamano_ultimo_empleo"].fillna("SIN_EMPLEADOR")
    df["estacionalidad_rubro_ultimo_empleo"] = df["rubro_ultimo_empleo"].map(est).fillna(1 / 12)

    df["edad"] = df["edad_inicio"] + (periodo_to_idx(corte) - periodo_to_idx(periodo_inicio)) // 12
    df["meses_hasta_pension"] = (np.where(df["sexo"] == "F", 60, 65) - df["edad"]) * 12
    df["meses_desde_afiliacion"] = periodo_to_idx(corte) - periodo_to_idx(df["periodo_afiliacion"])

    # tipo de contrato de la relacion vigente: si no hay empleador de referencia, "SIN_EMPLEADOR";
    # si lo hay pero no hay un aviso AFC de INICIO conocido para ese empleador especifico
    # (la relacion empezo antes del inicio de la historia disponible), "SIN_DATO_AFC".
    df["tipo_contrato"] = np.where(df["empleador_principal"].isna(), "SIN_EMPLEADOR",
                                   df["tipo_contrato_vigente"].fillna("SIN_DATO_AFC"))
    df["rubro"] = df["rubro"].fillna("SIN_EMPLEADOR")
    df["tamano_empleador"] = df["tamano_empleador"].fillna("SIN_EMPLEADOR")

    rellenos_0 = ([f"rim_l{k}" for k in range(1, N_LAGS + 1)]
                  + ["media_3", "media_6", "media_12", "media_24", "std_6", "max_12", "min_12",
                     "n_cotiza_3", "n_cotiza_6", "n_cotiza_12", "n_cotiza_24", "meses_historia",
                     "racha_cotiza", "racha_cero", "n_cambios_12", "meses_con_empleador",
                     "n_empleadores_12", "sueldo_fijo", "cv_6", "ratio_l1_l2", "ratio_l1_media12",
                     "tendencia_3_12", "emp_n_trabajadores", "emp_crecimiento_12", "emp_tasa_salida_12",
                     "emp_sin_planilla_conocida",
                     "afc_termino_l1", "afc_termino_l2", "afc_inicio_l1", "afc_inicio_l2",
                     "dias_licencia_l1", "dias_licencia_l2", "licencia_en_curso",
                     "cesantia_en_curso", "cesantia_giros_restantes", "cesantia_tasa_actual"]
                  + FEATURES_PLANILLA)
    df[rellenos_0] = df[rellenos_0].fillna(0.0)
    df["meses_desde_ult_cotiza"] = df["meses_desde_ult_cotiza"].fillna(99).clip(upper=99)
    df["meses_desde_cambio"] = df["meses_desde_cambio"].fillna(99).clip(upper=99)
    df["meses_desde_afc_termino"] = df["meses_desde_afc_termino"].fillna(99).clip(upper=99)
    df["meses_desde_afc_inicio"] = df["meses_desde_afc_inicio"].fillna(99).clip(upper=99)

    df["periodo_corte"] = corte
    df["corte_rim"] = add_months(corte, -2)

    for c in FEATURES_NUMERICAS + FEATURES_PLANILLA:
        df[c] = df[c].astype(float)

    # -------- poblaciones de los dos modelos (definicion 100% as-of, sin mirar el futuro) --
    # Modelo A (cese): cotizando (RIM conocida > 0) en corte-2, solo dependientes/pensionados
    # activos (bloqueante #2, ronda 1: independientes y voluntarios excluidos, ver README §8).
    df["pop_cese"] = ((df["rim_l1"] > 0) & df["tipo_afiliado"].isin(TIPOS_DEPENDENCIA)).astype(int)
    # Modelo B (reingreso): dependientes sin RIM conocida en corte-2 NI en corte-3 (mayor #4,
    # ronda 1: exigir solo corte-2 == 0 dejaba adentro ~16,6% de cotizantes reales cuya
    # planilla de corte-2 aun no habia llegado -planillas pendientes con target casi
    # determinista- porque una recepcion tardia de un solo mes es comun; que corte-2 Y
    # corte-3 esten en 0 a la vez es mucho mas raro para alguien que sigue activo, y exige
    # ademas al menos una cotizacion en los 24 meses previos).
    df["pop_reingreso"] = ((df["tipo_afiliado"] == "DEPENDIENTE") & (df["rim_l1"] == 0)
                           & (df["rim_l2"] == 0) & (df["n_cotiza_24"] > 0)).astype(int)

    cols = (["afiliado_id", "periodo_corte", "corte_rim", "pop_cese", "pop_reingreso",
             "tipo_afiliado", "sexo", "nivel_educacional", "region", "rubro", "tamano_empleador",
             "tipo_contrato", "rubro_ultimo_empleo", "tamano_ultimo_empleo"]
            + FEATURES_NUMERICAS + FEATURES_PLANILLA)
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
