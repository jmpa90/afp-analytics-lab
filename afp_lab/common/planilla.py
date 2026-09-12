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


def _aplicar_recepcion(lineas: pd.DataFrame, empleadores_latente: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Recepcion/DNP/rectificacion con las MISMAS reglas del proyecto 01: cada linea hereda el
    comportamiento de pago de su empleador (anticipado/oportuno/tardio/rezago/DNP/rectificacion,
    `empleadores_latente.p_pago_*`), consistente entre lineas de la misma (afiliado, empleador,
    periodo) porque la AFP recibe la planilla completa de una vez."""
    rng = np.random.default_rng(seed + 902)
    pago = empleadores_latente.set_index("empleador_id")[
        ["p_pago_anticipado", "p_pago_oportuno", "p_pago_tardio", "p_rezago", "p_dnp", "p_rectificacion"]]
    df = lineas.merge(pago, left_on="empleador_id", right_index=True, how="left")
    for c in ["p_pago_anticipado", "p_pago_oportuno", "p_pago_tardio", "p_rezago", "p_dnp", "p_rectificacion"]:
        df[c] = df[c].fillna(0.0)

    n = len(df)
    r = rng.random(n)
    p_a, p_o, p_t = df["p_pago_anticipado"].to_numpy(), df["p_pago_oportuno"].to_numpy(), df["p_pago_tardio"].to_numpy()
    lag = np.where(r < p_a, 0,
          np.where(r < p_a + p_o, 1,
          np.where(r < p_a + p_o + p_t, 2, 3 + rng.integers(0, 6, n))))
    df["periodo_recepcion"] = _add_months(df["periodo"].to_numpy(), lag)
    df["estado_pago"] = np.where(rng.random(n) < df["p_dnp"].to_numpy(), "DNP", "PAGADA")
    df["tipo_movimiento"] = "DECLARACION"

    rect_mask = rng.random(n) < df["p_rectificacion"].to_numpy()
    rect = df.loc[rect_mask].copy()
    rect["renta_imponible_afp"] = np.round(rect["renta_imponible_afp"] * rng.uniform(0.85, 1.20, len(rect)), 0)
    rect["periodo_recepcion"] = _add_months(rect["periodo_recepcion"].to_numpy(), 1 + rng.integers(0, 3, len(rect)))
    rect["tipo_movimiento"] = "RECTIFICACION"
    rect["codigo_movimiento_personal"] = 12   # reliquidacion

    dnp_mask = df["estado_pago"] == "DNP"
    pago_dnp = df.loc[dnp_mask].copy()
    pago_dnp["periodo_recepcion"] = _add_months(pago_dnp["periodo_recepcion"].to_numpy(), 2 + rng.integers(0, 6, len(pago_dnp)))
    pago_dnp["estado_pago"] = "PAGADA"
    pago_dnp["tipo_movimiento"] = "PAGO_DNP"

    out = pd.concat([df.drop(columns=list(pago.columns)),
                     rect.drop(columns=list(pago.columns)),
                     pago_dnp.drop(columns=list(pago.columns))], ignore_index=True)
    return out


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
