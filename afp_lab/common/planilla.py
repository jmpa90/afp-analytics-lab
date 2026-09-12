"""Planilla Previred sintética (`afp_lab/common/planilla.py`).

Modelo de datos regulatorio, ADITIVO: se construye a partir de `panel_mensual` (verdad laboral
simulada) y `empleadores_latente` (sólo para el comportamiento de pago: probabilidades de pago
anticipado/oportuno/tardío/DNP/rectificación por tamaño de empleador, igual que en
rim_nowcasting/proyecto 01) y **no modifica** `synth_base.generar_universo` ni sus salidas.

Objetivo (ver `afp_lab/common/BRIEF_PLANILLA.md` y `docs/normativa_tablas_afp.md`): que el
laboratorio tenga una tabla con la forma de la planilla real (Previred / Compendio SP), con los
mismos códigos y campos que verían los proyectos en producción, para no tener que inventar
columnas ad-hoc cada vez.

    generar_planilla(universo, seed) -> pd.DataFrame   # grano: afiliado x empleador x periodo x tipo_pago
    construir_rezagos(planilla, seed) -> pd.DataFrame  # cotizaciones recibidas sin imputar al RUT

Alcance y simplificaciones (documentadas también en el README de este módulo... este módulo vive
en `common/`, así que las decisiones quedan en el docstring y en `docs/normativa_tablas_afp.md`):

* Sólo declaran planilla los afiliados **dependientes o pensionados activos con empleador
  vigente** (`empleador_id` no nulo y RIM > 0 ese mes): los independientes pagan vía SII
  (Operación Renta) y los voluntarios de forma directa, ninguno de los dos pasa por Previred.
* **Coherencia obligatoria con `panel_mensual`**: para cada (afiliado, periodo), la suma de
  `renta_imponible_afp` de todas las líneas `DECLARACION` de tipo `01` remuneraciones + `02`
  gratificación + `05` subsidio, **sumando todos los empleadores del mes**, es exactamente
  `rim_emp1 + rim_emp2` (0 si ese vínculo no existe) del panel. La igualdad se cumple a nivel
  (afiliado, periodo); a nivel (afiliado, empleador, periodo) puede haber un caso borde: cuando
  `empleador2_id == empleador_id` (mismo empleador en ambos vínculos, posible por azar cuando el
  pool de empleadores es chico), las líneas de ambos vínculos comparten `empleador_id` y su suma da
  `rim_emp1 + rim_emp2` combinado — correcto (es lo que ese empleador realmente declaró para ese
  RUT ese mes), pero no coincide con `rim_emp1` ni con `rim_emp2` por separado. Los tests validan la
  igualdad al nivel (afiliado, periodo), que es el que siempre se cumple. La línea `06` (aguinaldo)
  se reporta con su monto real pero **no** se
  descuenta de las demás para no romper esa igualdad ni el monto total de cotización obligatoria
  que ya generaron los proyectos 01/04 sobre el mismo panel: en el Chile real muchos aguinaldos
  de empresa sí son imponibles, así que mantenerlo dentro del total es razonable y evita un
  "descuadre" de RIM entre proyectos que hoy consumen `cotizaciones_conocidas`/`panel_mensual`
  directamente. Se documenta como decisión, no como norma.
* Códigos de movimiento de personal (Tabla N° 7) que este generador SÍ produce: 0, 1, 2, 3, 4, 7,
  8, 11, 12 (ver `tablas_equivalencia.CODIGOS_NO_GENERADOS` para los que no: 5, 6, 13 no tienen
  regla propia en el brief).
* Permiso sin goce (4) y ausentismo (11) son una etiqueta + `dias_trabajados` reducido; la RIM del
  mes **no** se reduce por esto (se mantiene igual al panel), para no violar la coherencia de RIM
  exigida arriba — es la AFP real la que vería una RIM menor; aquí es una simplificación explícita.
* El cambio de contrato (8) se aplica sólo a la copia "planilla" de `tipo_contrato`
  (`tipo_contrato_planilla`): no se toca `panel_mensual.tipo_contrato` (aditivo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tablas_equivalencia import TASA_COTIZACION_OBLIGATORIA, TASA_SIS, tasa_aporte_empleador

# --------------------------------------------------------------- aritmetica de periodos (yyyymm)
def _periodo_to_idx(p):
    p = np.asarray(p)
    return (p // 100 - 2000) * 12 + (p % 100 - 1)


def _idx_to_periodo(i):
    i = np.asarray(i)
    return (2000 + i // 12) * 100 + (i % 12 + 1)


def _add_months(p, k):
    return _idx_to_periodo(_periodo_to_idx(p) + k)


def _ultimo_dia_mes(periodo: np.ndarray) -> np.ndarray:
    anio = periodo // 100
    mes = periodo % 100
    bisiesto = (anio % 4 == 0) & ((anio % 100 != 0) | (anio % 400 == 0))
    dias = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])[mes - 1]
    return np.where((mes == 2) & bisiesto, 29, dias)


# ------------------------------------------------------------- parametros (documentados aqui)
P_PERMISO_SIN_GOCE = 0.003    # probabilidad mensual, entre empleados sin otro movimiento ese mes
P_AUSENTISMO = 0.004          # idem, codigo 11
P_ANUAL_CAMBIO_CONTRATO = 0.20   # prob. anual de que un plazo fijo con >=12 m de antiguedad pase a indefinido
P_REZAGO_LINEA = 0.007        # 0,7% de las lineas (Libro II: RUT mal informado, sin imputar de inicio)
REZAGO_MESES_MIN, REZAGO_MESES_MAX = 2, 6

# DNP (declarada y no pagada, Libro II): persistencia y estacionalidad, configurables (pedido
# de DS-2/proyecto 05, ronda 2). La DNP es, en la practica, un problema de CAJA DEL EMPLEADOR
# que afecta a toda su planilla ese mes, no a un trabajador a la vez, y tiende a repetirse mes
# a mes en un empleador con problemas (persistencia) y a concentrarse en ciertos meses
# (estacionalidad). Se modela como un latente AR(1) por empleador (`_estado_dnp_empleador`):
# `u_t = rho*u_{t-1} + sqrt(1-rho^2)*eps_t`, `DNP_t = u_t < Phi^-1(p_efectivo_t)` (probit-AR1),
# que preserva la probabilidad marginal `p_dnp` del empleador (`empleadores_latente`) pero
# correlaciona el estado mes a mes. `rho=0` reproduce el comportamiento anterior (independiente
# mes a mes, aunque ya no linea a linea: ver bug corregido en `_aplicar_recepcion`).
DNP_RHO_PERSISTENCIA = 0.35
# Enero/febrero (caja mas apretada tras el periodo de vacaciones/aguinaldo de fin de ano) y
# diciembre (aguinaldos, cierre de ejercicio): mas DNP que el resto del ano (supuesto, no cifra
# oficial; documentado para que se pueda recalibrar sin tocar la logica).
DNP_ESTACIONALIDAD_MES = {1: 1.30, 2: 1.15, 12: 1.20}


def _tipo_trabajador(tipo_afiliado: np.ndarray, edad: np.ndarray, sexo: np.ndarray) -> np.ndarray:
    mayor = np.where(sexo == "F", edad >= 60, edad >= 65)
    return np.select(
        [tipo_afiliado == "PENSIONADO_ACTIVO", mayor],
        ["PENSIONADO_COTIZA", "ACTIVO_MAYOR"],
        "ACTIVO",
    )


def _movimiento_y_dias(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Codigo de movimiento de personal, dias trabajados y tipo de contrato "de planilla"
    (puede diferir de `panel_mensual.tipo_contrato` tras un cambio de contrato, codigo 8), para el
    empleador PRINCIPAL de cada (afiliado, periodo) con relacion laboral vigente."""
    df = panel.sort_values(["afiliado_id", "periodo"]).reset_index(drop=True).copy()
    rng = np.random.default_rng(seed + 901)
    n = len(df)

    codigo = np.zeros(n, dtype=np.int64)
    codigo = np.where(df["evento_afc"] == "INICIO",
                     np.where(df["tipo_contrato"] == "INDEFINIDO", 1, 7), codigo)
    codigo = np.where(df["evento_afc"] == "TERMINO", 2, codigo)
    codigo = np.where((codigo == 0) & (df["dias_licencia"] > 0), 3, codigo)

    libre = codigo == 0
    u = rng.random(n)
    codigo = np.where(libre & (u < P_PERMISO_SIN_GOCE), 4, codigo)
    libre = codigo == 0
    u2 = rng.random(n)
    codigo = np.where(libre & (u2 < P_AUSENTISMO), 11, codigo)

    # cambio de contrato plazo fijo -> indefinido: un sorteo por (afiliado, empleador, inicio de
    # la racha de antiguedad), aplicado el primer mes con antiguedad >= 12 en que aun no se habia
    # sorteado; una vez ocurre, se propaga tipo_contrato_planilla = INDEFINIDO hacia adelante
    # dentro de la misma racha con el mismo empleador.
    tipo_contrato_planilla = df["tipo_contrato"].to_numpy(dtype=object).copy()
    plazo_fijo_elegible = (df["tipo_contrato"] == "PLAZO_FIJO") & (df["meses_antiguedad"] >= 12) & (codigo == 0)
    u3 = rng.random(n)
    hazard_mensual = 1 - (1 - P_ANUAL_CAMBIO_CONTRATO) ** (1 / 12)
    cambia = plazo_fijo_elegible & (u3 < hazard_mensual)
    codigo = np.where(cambia, 8, codigo)

    df["codigo_movimiento_personal"] = codigo
    df["tipo_contrato_planilla"] = tipo_contrato_planilla
    # el mes del cambio y todos los posteriores de la misma racha (mismo empleador, antiguedad
    # creciente sin cortes) quedan como INDEFINIDO: como el df esta ordenado por periodo dentro de
    # cada afiliado, basta propagar hacia adelante un flag por (afiliado, empleador) desde el mes
    # del cambio mientras la racha de antiguedad siga (meses_antiguedad no vuelve a 0 sin pasar por
    # un nuevo INICIO, que ya crea su propia fila con su propio tipo_contrato).
    df["_cambio8"] = cambia.astype(int)
    df["_racha_id"] = (df["evento_afc"] == "INICIO").astype(int)
    df["_racha_id"] = df.groupby("afiliado_id")["_racha_id"].cumsum()
    ya_convertido = (df.groupby(["afiliado_id", "_racha_id"])["_cambio8"].cummax() > 0)
    df["tipo_contrato_planilla"] = np.where(ya_convertido, "INDEFINIDO", df["tipo_contrato_planilla"])
    df = df.drop(columns=["_cambio8", "_racha_id"])

    dias = np.full(n, 30, dtype=np.int64)
    dias = np.where(np.isin(codigo, (1, 7)), rng.integers(8, 29, n), dias)
    dias = np.where(codigo == 2, rng.integers(3, 26, n), dias)
    dias = np.where(codigo == 3, np.clip(30 - df["dias_licencia"].to_numpy(), 0, 30), dias)
    dias = np.where(codigo == 4, 30 - rng.integers(3, 15, n), dias)
    dias = np.where(codigo == 11, 30 - rng.integers(1, 6, n), dias)
    df["dias_trabajados"] = dias
    return df


def _lineas_pago(df_mov: pd.DataFrame) -> pd.DataFrame:
    """Explota cada fila (afiliado, empleador principal, periodo) en 1 a 3 lineas de tipo_pago
    (01 remuneraciones, 02 gratificacion, 05 subsidio) que suman exactamente `rim_emp1`, mas la
    linea 06 (aguinaldo) informativa (ver docstring del modulo: no se descuenta del total)."""
    rim = df_mov["rim_emp1"].to_numpy(dtype=float)
    grat = np.minimum(df_mov["gratificacion"].to_numpy(dtype=float), rim)
    aguinaldo = df_mov["aguinaldo"].to_numpy(dtype=float)
    dias_lic = df_mov["dias_licencia"].to_numpy(dtype=float)
    # subsidio reconstruido: la porcion de la RIM atribuible a dias de licencia (subsidio de
    # incapacidad laboral, tambien imponible para AFP). Se resta de la linea 01 para no duplicar.
    subsidio = np.where(dias_lic > 0, np.round(rim * 0.10 * np.minimum(dias_lic, 30) / 30, 0), 0.0)
    subsidio = np.minimum(subsidio, np.maximum(rim - grat, 0))
    base = np.maximum(rim - grat - subsidio, 0.0)

    comunes = ["afiliado_id", "empleador_id", "periodo", "codigo_movimiento_personal",
              "fecha_desde", "fecha_hasta", "dias_trabajados", "tipo_trabajador",
              "tipo_contrato_planilla"]
    filas = []
    l01 = df_mov[comunes].copy()
    l01["tipo_pago"] = "01"
    l01["renta_imponible_afp"] = base
    filas.append(l01)

    m_grat = grat > 0
    if m_grat.any():
        l02 = df_mov.loc[m_grat, comunes].copy()
        l02["tipo_pago"] = "02"
        l02["renta_imponible_afp"] = grat[m_grat]
        filas.append(l02)

    m_sub = subsidio > 0
    if m_sub.any():
        l05 = df_mov.loc[m_sub, comunes].copy()
        l05["tipo_pago"] = "05"
        l05["renta_imponible_afp"] = subsidio[m_sub]
        filas.append(l05)

    m_ag = aguinaldo > 0
    if m_ag.any():
        l06 = df_mov.loc[m_ag, comunes].copy()
        l06["tipo_pago"] = "06"
        l06["renta_imponible_afp"] = aguinaldo[m_ag]
        filas.append(l06)

    return pd.concat(filas, ignore_index=True)


def _lineas_segundo_empleador(panel: pd.DataFrame) -> pd.DataFrame:
    """Lineas 01 del empleador SECUNDARIO (multiempleo, ~5-7% de los afiliados): sin desglose de
    conceptos ni movimiento de personal propio (simplificacion; el vinculo principal ya lo
    reporta)."""
    m = panel["empleador2_id"].notna() & (panel["rim_emp2"] > 0)
    sub = panel.loc[m, ["afiliado_id", "periodo", "rim_emp2"]].copy()
    sub = sub.rename(columns={"rim_emp2": "renta_imponible_afp"})
    sub["empleador_id"] = panel.loc[m, "empleador2_id"].astype("Int64")
    sub["codigo_movimiento_personal"] = 0
    sub["dias_trabajados"] = 30
    sub["tipo_pago"] = "01"
    sub["tipo_contrato_planilla"] = None
    return sub


def _estado_dnp_empleador(periodos: np.ndarray, empleadores_latente: pd.DataFrame, seed: int,
                          rho: float = DNP_RHO_PERSISTENCIA,
                          estacionalidad: dict[int, float] | None = None) -> pd.DataFrame:
    """Estado DNP por (empleador, periodo) — UNA vez por empleador y mes, para TODOS sus
    trabajadores (ver constantes `DNP_RHO_PERSISTENCIA`/`DNP_ESTACIONALIDAD_MES` arriba).
    Devuelve una fila por (empleador_id, periodo) de `periodos`, para TODOS los empleadores de
    `empleadores_latente` (no solo los que declararon planilla ese mes: simplifica el merge
    posterior con un `left join`)."""
    estacionalidad = DNP_ESTACIONALIDAD_MES if estacionalidad is None else estacionalidad
    rng = np.random.default_rng(seed + 904)
    periodos_u = np.sort(np.unique(periodos))
    emp_ids = empleadores_latente["empleador_id"].to_numpy()
    p_base = empleadores_latente.set_index("empleador_id")["p_dnp"].reindex(emp_ids).fillna(0.0).to_numpy()
    n_emp = len(emp_ids)

    u_prev = np.zeros(n_emp)
    filas = []
    for periodo in periodos_u:
        mes = int(periodo % 100)
        mult = estacionalidad.get(mes, 1.0)
        p_ef = np.clip(p_base * mult, 0.0, 0.95)
        # inversa de la normal estandar sin depender de scipy (aproximacion racional de
        # Acklam, error < 1.15e-9): evita agregar una dependencia nueva al laboratorio.
        thr = _norm_ppf(np.clip(p_ef, 1e-6, 1 - 1e-6))
        eps = rng.normal(0.0, 1.0, n_emp)
        u = rho * u_prev + np.sqrt(max(1 - rho ** 2, 0.0)) * eps
        filas.append(pd.DataFrame({"empleador_id": emp_ids, "periodo": periodo, "_dnp": u < thr}))
        u_prev = u
    return pd.concat(filas, ignore_index=True)


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inversa de la CDF normal estandar (aproximacion racional de Acklam), vectorizada, sin
    dependencia de scipy. Precision suficiente (~1e-9) para un umbral de probabilidad."""
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    out = np.empty_like(p)

    lo = p < p_low
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    mid = (~lo) & (p <= p_high)
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    hi = p > p_high
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    return out


def _aplicar_recepcion(lineas: pd.DataFrame, empleadores_latente: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Recepcion/DNP/rectificacion con las MISMAS reglas del proyecto 01: cada DECLARACION
    (afiliado x empleador x periodo, con todas sus lineas/conceptos) hereda el comportamiento
    de pago de su empleador (anticipado/oportuno/tardio/rezago/rectificacion,
    `empleadores_latente.p_pago_*`).

    Bug corregido (ronda 2, reportado por DS-2 desde el proyecto 05): `periodo_recepcion` y
    `estado_pago`/rectificacion se sorteaban con `rng.random(n)` con `n` = numero de LINEAS,
    asi que dos lineas de una misma declaracion (p. ej. `01` y `02` del mismo afiliado, mismo
    empleador, mismo periodo) podian terminar con un `periodo_recepcion` distinto — al reves
    de lo que decia (y sigue diciendo) este docstring. Ahora el sorteo de recepcion y de
    rectificacion se hace UNA vez por declaracion (`rut_trabajador x empleador_id x periodo`)
    y se propaga a todas sus lineas; el estado DNP se eleva un nivel mas, a
    `empleador_id x periodo` (`_estado_dnp_empleador`, con persistencia y estacionalidad
    configurables), porque en la practica una DNP es un problema de caja del empleador que
    afecta a toda su planilla del mes, no a un trabajador a la vez."""
    rng = np.random.default_rng(seed + 902)
    cols_pago = ["p_pago_anticipado", "p_pago_oportuno", "p_pago_tardio", "p_rezago", "p_rectificacion"]
    pago = empleadores_latente.set_index("empleador_id")[cols_pago]
    df = lineas.merge(pago, left_on="empleador_id", right_index=True, how="left")
    for c in cols_pago:
        df[c] = df[c].fillna(0.0)

    llave = ["rut_trabajador", "empleador_id", "periodo"]
    claves = (df[llave + cols_pago].drop_duplicates(subset=llave)
             .sort_values(llave).reset_index(drop=True))
    m = len(claves)
    r = rng.random(m)
    p_a = claves["p_pago_anticipado"].to_numpy()
    p_o = claves["p_pago_oportuno"].to_numpy()
    p_t = claves["p_pago_tardio"].to_numpy()
    lag = np.where(r < p_a, 0,
          np.where(r < p_a + p_o, 1,
          np.where(r < p_a + p_o + p_t, 2, 3 + rng.integers(0, 6, m))))
    claves["_periodo_recepcion"] = _add_months(claves["periodo"].to_numpy(), lag)
    claves["_rectifica"] = rng.random(m) < claves["p_rectificacion"].to_numpy()
    claves["_factor_rect"] = rng.uniform(0.85, 1.20, m)
    claves["_extra_rect"] = 1 + rng.integers(0, 3, m)
    claves["_extra_dnp"] = 2 + rng.integers(0, 6, m)

    dnp_emp = _estado_dnp_empleador(claves["periodo"].to_numpy(), empleadores_latente, seed)
    claves = claves.merge(dnp_emp, on=["empleador_id", "periodo"], how="left")
    claves["_dnp"] = claves["_dnp"].fillna(False)

    cols_nuevas = ["_periodo_recepcion", "_rectifica", "_factor_rect", "_extra_rect", "_extra_dnp", "_dnp"]
    df = df.merge(claves[llave + cols_nuevas], on=llave, how="left")
    df["periodo_recepcion"] = df["_periodo_recepcion"]
    df["estado_pago"] = np.where(df["_dnp"], "DNP", "PAGADA")
    df["tipo_movimiento"] = "DECLARACION"

    rect_mask = df["_rectifica"].to_numpy()
    rect = df.loc[rect_mask].copy()
    rect["renta_imponible_afp"] = np.round(rect["renta_imponible_afp"] * rect["_factor_rect"], 0)
    rect["periodo_recepcion"] = _add_months(rect["periodo_recepcion"].to_numpy(), rect["_extra_rect"].to_numpy())
    rect["tipo_movimiento"] = "RECTIFICACION"
    rect["codigo_movimiento_personal"] = 12   # reliquidacion

    dnp_mask = (df["estado_pago"] == "DNP").to_numpy()
    pago_dnp = df.loc[dnp_mask].copy()
    pago_dnp["periodo_recepcion"] = _add_months(pago_dnp["periodo_recepcion"].to_numpy(),
                                                pago_dnp["_extra_dnp"].to_numpy())
    pago_dnp["estado_pago"] = "PAGADA"
    pago_dnp["tipo_movimiento"] = "PAGO_DNP"

    cols_drop = list(pago.columns) + cols_nuevas
    out = pd.concat([df.drop(columns=cols_drop),
                     rect.drop(columns=cols_drop),
                     pago_dnp.drop(columns=cols_drop)], ignore_index=True)
    return out


def construir_agregado_empleador(planilla: pd.DataFrame) -> pd.DataFrame:
    """Agregado empleador x periodo (pedido de DS-2 para el proyecto 05): dotacion (n de
    trabajadores con una linea `01` DECLARACION ese periodo), masa imponible (suma de
    `renta_imponible_afp` de esas lineas), y dos resumenes de recepcion/estado sobre TODAS
    las lineas DECLARACION del periodo (no solo la `01`): `frac_dnp` (fraccion de lineas en
    estado DNP en su declaracion original) y `lag_recepcion_medio` (meses entre `periodo` y
    `periodo_recepcion`, promedio). No es una tabla nueva de la simulacion (no agrega
    aleatoriedad): es una vista agregada de `planilla`, pensada para proyectos que necesiten
    el tamano/comportamiento de un empleador sin reconstruirlo linea a linea."""
    decl = planilla[planilla["tipo_movimiento"] == "DECLARACION"]
    l01 = decl[decl["tipo_pago"] == "01"]
    dotacion = l01.groupby(["empleador_id", "periodo"])["rut_trabajador"].nunique().rename("dotacion")
    masa = decl.groupby(["empleador_id", "periodo"])["renta_imponible_afp"].sum().rename("masa_imponible")

    lag = (decl["periodo_recepcion"].map(_periodo_to_idx) - decl["periodo"].map(_periodo_to_idx))
    resumen = decl.assign(_lag=lag, _dnp=(decl["estado_pago"] == "DNP").astype(float)).groupby(
        ["empleador_id", "periodo"]).agg(lag_recepcion_medio=("_lag", "mean"), frac_dnp=("_dnp", "mean"),
                                         n_lineas=("_dnp", "size"))

    out = pd.concat([dotacion, masa, resumen], axis=1).reset_index()
    out["dotacion"] = out["dotacion"].fillna(0).astype(int)
    out["masa_imponible"] = out["masa_imponible"].fillna(0.0)
    return out.sort_values(["empleador_id", "periodo"]).reset_index(drop=True)


def generar_planilla(universo: dict[str, pd.DataFrame], seed: int) -> pd.DataFrame:
    """Planilla Previred sintetica, grano afiliado x empleador x periodo x tipo_pago.

    `universo` es el dict que devuelve `synth_base.generar_universo` (no se modifica). Sólo se
    usa `empleadores_latente` para el comportamiento de pago (probabilidades `p_pago_*`),
    permitido por CONVENCIONES §2 ("sí pueden usarse para generar la dinámica"); nunca se expone
    como feature de un modelo.
    """
    panel = universo["panel_mensual"]
    afiliados = universo["afiliados"]
    empleadores_lat = universo["empleadores_latente"]

    elegibles = panel["empleador_id"].notna() & (panel["rim_emp1"] > 0)
    principal = panel.loc[elegibles].merge(
        afiliados[["afiliado_id", "tipo_afiliado", "sexo"]], on="afiliado_id", how="left")

    df_mov = _movimiento_y_dias(principal, seed)
    periodos = df_mov["periodo"].to_numpy()
    df_mov["fecha_desde"] = pd.to_datetime(periodos.astype(str), format="%Y%m")
    df_mov["fecha_hasta"] = pd.to_datetime(periodos.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df_mov["tipo_trabajador"] = _tipo_trabajador(df_mov["tipo_afiliado"].to_numpy(),
                                                 df_mov["edad"].to_numpy(), df_mov["sexo"].to_numpy())

    lineas = _lineas_pago(df_mov)
    lineas2 = _lineas_segundo_empleador(panel)
    if len(lineas2):
        lineas2 = lineas2.merge(afiliados[["afiliado_id", "tipo_afiliado", "sexo"]], on="afiliado_id", how="left")
        edad_map = panel.set_index(["afiliado_id", "periodo"])["edad"]
        lineas2["edad"] = edad_map.reindex(pd.MultiIndex.from_frame(lineas2[["afiliado_id", "periodo"]])).to_numpy()
        lineas2["tipo_trabajador"] = _tipo_trabajador(lineas2["tipo_afiliado"].to_numpy(),
                                                      lineas2["edad"].to_numpy(), lineas2["sexo"].to_numpy())
        p2 = lineas2["periodo"].to_numpy()
        lineas2["fecha_desde"] = pd.to_datetime(p2.astype(str), format="%Y%m")
        lineas2["fecha_hasta"] = pd.to_datetime(p2.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
        lineas2 = lineas2.drop(columns=["tipo_afiliado", "sexo", "edad"])
        todas = pd.concat([lineas, lineas2], ignore_index=True)
    else:
        todas = lineas

    todas["rut_trabajador"] = todas["afiliado_id"]
    todas = _aplicar_recepcion(todas, empleadores_lat, seed)

    # renta imponible AFC: igual regla que rim_nowcasting/proyecto 04 (dependientes de empresas
    # privadas; excluye admin. publica y pensionados activos, no afectos a la Ley 19.728)
    publicos = set(empleadores_lat.loc[empleadores_lat["rubro"] == "O_ADMIN_PUBLICA", "empleador_id"])
    afecto_afc = (~todas["empleador_id"].isin(publicos)) & (todas["tipo_trabajador"] != "PENSIONADO_COTIZA")
    todas["renta_imponible_afc"] = np.where(afecto_afc, todas["renta_imponible_afp"], 0.0)

    todas["cotizacion_obligatoria"] = np.round(todas["renta_imponible_afp"] * TASA_COTIZACION_OBLIGATORIA, 0)
    todas["sis"] = np.round(todas["renta_imponible_afp"] * TASA_SIS, 0)
    todas["aporte_empleador"] = todas["periodo"].map(lambda p: round(tasa_aporte_empleador(int(p)), 4)) * \
        todas["renta_imponible_afp"]
    todas["aporte_empleador"] = np.round(todas["aporte_empleador"], 0)

    cols = ["rut_trabajador", "empleador_id", "periodo", "tipo_pago", "dias_trabajados",
           "codigo_movimiento_personal", "fecha_desde", "fecha_hasta", "tipo_trabajador",
           "tipo_contrato_planilla", "renta_imponible_afp", "renta_imponible_afc",
           "cotizacion_obligatoria", "sis", "aporte_empleador", "periodo_recepcion",
           "estado_pago", "tipo_movimiento"]
    todas["empleador_id"] = todas["empleador_id"].astype("Int64")
    todas["periodo"] = todas["periodo"].astype(int)
    return todas[cols].sort_values(["rut_trabajador", "periodo", "empleador_id", "tipo_pago"]).reset_index(drop=True)


def construir_rezagos(planilla: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cotizaciones recibidas por la AFP que no se pudieron imputar de inmediato al RUT del
    trabajador (Libro II del Compendio: RUT mal informado en la planilla) y se regularizan
    `REZAGO_MESES_MIN`-`REZAGO_MESES_MAX` meses después. `P_REZAGO_LINEA` de las líneas
    `DECLARACION` con `estado_pago="PAGADA"` quedan marcadas; mientras no se imputan, esa parte de
    la RIM no es atribuible al afiliado (simplificación: no se resta de `planilla`, se documenta
    como información adicional para quien quiera modelar el efecto)."""
    rng = np.random.default_rng(seed + 903)
    candidatas = planilla[(planilla["tipo_movimiento"] == "DECLARACION") & (planilla["estado_pago"] == "PAGADA")]
    marca = rng.random(len(candidatas)) < P_REZAGO_LINEA
    rez = candidatas.loc[marca, ["rut_trabajador", "empleador_id", "periodo", "tipo_pago",
                                "renta_imponible_afp", "periodo_recepcion"]].copy()
    rez = rez.rename(columns={"periodo_recepcion": "periodo_recepcion_original"})
    extra = rng.integers(REZAGO_MESES_MIN, REZAGO_MESES_MAX + 1, len(rez))
    rez["periodo_imputacion"] = _add_months(rez["periodo_recepcion_original"].to_numpy(), extra)
    return rez.reset_index(drop=True)
