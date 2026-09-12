"""Tablas sinteticas del proyecto 02 (traspaso a otra AFP).

Parte del universo comun (`afp_lab.common.synth_base.generar_universo`: afiliados, empleadores,
panel laboral mensual y macro) y le agrega las tablas propias del problema de retencion:

    mercado        mes x AFP       comision, rentabilidad 12m por fondo, adjudicataria de licitacion
    saldos         afiliado x mes  saldo de la cuenta obligatoria y fondo elegido (solo meses vigente)
    cambios_fondo  evento          cambio de fondo A-E
    traspasos      evento          solicitud / efecto de traspaso de SALIDA y de ENTRADA  (TARGET)
    contactos      evento          ejecutivo propio, vendedor de la competencia, call center, digital
    reclamos       evento          rentabilidad / atencion / comision / pension
    uso_digital    afiliado x mes  logins app y web, consultas de saldo, descargas de certificado

La dinamica se simula mes a mes (loop sobre periodos, numpy vectorizado sobre afiliados) para que
las senales sean *anteriores* al traspaso y no contemporaneas: el afiliado primero recibe contactos
y reclamos, luego entra en un estado latente de "intencion de traspaso" durante el cual aumenta su
uso digital y pide el certificado de saldo, y solo despues firma la solicitud.

Todos los parametros numericos estan en `ParametrosChurn` y se documentan en el README.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from afp_lab.common.synth_base import generar_universo
from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx, rango_periodos

# --------------------------------------------------------------------- catalogos
AFP_PROPIA = "AFP_ANDES"
AFP_COMPETIDORAS = ("AFP_COSTA", "AFP_SUR", "AFP_AUSTRAL", "AFP_CUMBRE", "AFP_PACIFICO")
AFP_TODAS = (AFP_PROPIA,) + AFP_COMPETIDORAS
FONDOS = ("A", "B", "C", "D", "E")
CANALES = ("EJECUTIVO_PROPIO", "VENDEDOR_COMPETENCIA", "CALL_CENTER", "CAMPANA_DIGITAL")
RESULTADOS_CONTACTO = ("NO_CONTACTADO", "CONTACTADO", "INTERESADO", "RECHAZADO")
TIPOS_RECLAMO = ("RENTABILIDAD", "ATENCION", "COMISION", "PENSION")
ESTADOS_RECLAMO = ("ABIERTO", "RESUELTO_FAVORABLE", "RESUELTO_DESFAVORABLE")

# Comision sobre la renta imponible (% ), tramos vigentes desde el periodo indicado.
# Rango realista del sistema chileno (~0,58 % a ~1,45 %). La propia es cara y recorta dos veces.
COMISIONES = {
    "AFP_ANDES": [(202001, 1.27), (202401, 1.19), (202507, 1.14)],
    "AFP_COSTA": [(202001, 0.69), (202208, 0.58)],
    "AFP_SUR": [(202001, 1.16), (202408, 0.99)],
    "AFP_AUSTRAL": [(202001, 1.45)],
    "AFP_CUMBRE": [(202001, 1.27)],
    "AFP_PACIFICO": [(202001, 1.44), (202601, 1.39)],
}
# Licitacion de nuevos afiliados: la adjudicataria recibe a TODOS los que se afilian al sistema
# durante la vigencia y ellos deben permanecer 24 meses (regla real chilena).
LICITACIONES = [(202001, "AFP_COSTA"), (202208, AFP_PROPIA), (202408, "AFP_SUR")]
MESES_PERMANENCIA_LICITACION = 24

# Rentabilidad real anual esperada y volatilidad por multifondo (supuestos de largo plazo).
FONDO_MU_ANUAL = {"A": 0.062, "B": 0.052, "C": 0.042, "D": 0.032, "E": 0.022}
FONDO_SIGMA_ANUAL = {"A": 0.160, "B": 0.120, "C": 0.085, "D": 0.055, "E": 0.035}
# Fondo por defecto segun edad (asignacion legal por tramo etario).
FONDO_DEFECTO_JOVEN, FONDO_DEFECTO_MEDIO, FONDO_DEFECTO_MAYOR = "B", "C", "D"

TASA_COTIZACION_OBLIGATORIA = 0.10  # 10 % de la renta imponible a la cuenta de capitalizacion


@dataclass(frozen=True)
class ParametrosChurn:
    """Parametros de la simulacion. La tabla del README se genera desde aqui (`tabla_parametros`)."""

    # --- composicion de la cartera propia
    p_antiguo_desde_afiliacion: float = 0.78   # siempre estuvo en la AFP propia
    p_antiguo_traspaso_previo: float = 0.12    # entro antes de la ventana simulada
    # el resto (10 %) entra durante la ventana via traspaso de ENTRADA

    # --- saldo inicial
    tasa_mensual_saldo_inicial: float = 0.0035  # rentabilidad real mensual para capitalizar la historia
    densidad_cotizacion_historica: float = 0.68
    sigma_saldo_inicial: float = 0.35

    # --- mercado: rentabilidad relativa entre AFP
    rho_ar_rentabilidad: float = 0.92        # persistencia mensual de la desviacion de cada AFP
    sigma_ar_rentabilidad: float = 0.00025   # innovacion mensual; da ~0,5 pp de sd en la ventana de 12 m
    castigo_rentabilidad_propia: float = -0.010   # -1,0 pp anual en la ventana de castigo
    castigo_desde: int = 202401
    castigo_hasta: int = 202512

    # --- contactos (probabilidad mensual base por afiliado)
    p_vendedor_competencia: float = 0.016
    p_ejecutivo_propio: float = 0.012
    p_call_center: float = 0.030
    p_campana_digital: float = 0.075
    factor_saldo_alto_competencia: float = 2.6   # multiplicador en el decil superior de saldo
    factor_retencion_ejecutivo: float = 3.0      # el ejecutivo reacciona a reclamos / visitas de la competencia
    factor_campana_marzo_agosto: float = 1.45
    p_campana_empleador: float = 0.020       # prob. mensual de que la competencia haga una charla en una empresa
    meses_campana_empleador: int = 3         # duracion de esa campana
    factor_campana_empleador: float = 8.0    # multiplicador de la visita de la competencia durante la campana

    # --- reclamos
    p_reclamo_base: float = 0.0040
    factor_reclamo_rentabilidad_mala: float = 2.2
    factor_reclamo_comision_alta: float = 1.6

    # --- uso digital
    lambda_login_base: float = 0.55
    factor_login_joven: float = 1.8
    p_intencion_visible: float = 0.35        # fraccion de afiliados cuya intencion se ve en los canales
    factor_login_intencion: float = 1.7      # solo para esa fraccion
    p_certificado_base: float = 0.010        # motivo administrativo corriente
    p_certificado_hipotecario: float = 0.006 # credito / tramite bancario (30-50 anos con renta media-alta)
    p_certificado_tramite_pension: float = 0.030  # a menos de 24 meses de la edad legal
    p_certificado_cambio_empleo: float = 0.020    # finiquito o cambio de empleador reciente
    factor_certificado_operacion_renta: float = 2.5   # abril-mayo
    p_certificado_intencion: float = 0.060   # incremento por intencion, solo en la fraccion visible

    # --- cambios de fondo
    p_cambio_fondo_base: float = 0.0055
    factor_cambio_fondo_activo: float = 3.0

    # --- hazard mensual de traspaso (escala logit)
    b0: float = -5.90
    b_brecha_comision: float = 1.05        # por punto porcentual de comision sobre la mas barata
    b_brecha_rentabilidad: float = 0.55    # por punto porcentual de rentabilidad 12m bajo el promedio
    b_vendedor_competencia_3m: float = 1.45
    b_vendedor_interesado_3m: float = 0.55
    b_reclamo_6m: float = 0.95
    b_reclamo_desfavorable_6m: float = 0.40
    b_ejecutivo_propio_3m: float = -0.85
    b_joven: float = 0.036                 # por ano bajo los 40
    b_renta: float = 0.55                  # por log(rim_ref / IMM)
    b_antiguedad: float = -0.75            # por log1p(anos en la AFP)
    b_cambios_fondo_12m: float = 0.33
    b_cesante: float = -1.15
    b_retirado: float = -1.60
    b_marzo: float = 0.35
    b_agosto: float = 0.30
    b_u_switcher: float = 0.55             # el que ya se cambio de AFP una vez se vuelve a cambiar
    # El "switcher" es LATENTE y la brecha de antiguedad solo es un proxy imperfecto: parte de los
    # que llegaron de otra AFP se movieron una sola vez por asignacion (licitacion previa, convenio
    # del empleador) y no son switchers activos, y parte de los que no registran brecha si lo son.
    p_switcher_dado_traspaso: float = 0.60
    p_switcher_sin_traspaso: float = 0.22
    sigma_afiliado: float = 0.40           # efecto aleatorio NO observable (fija el techo de AUC)
    sigma_mes: float = 0.15                # ruido idiosincratico mensual
    p_retardo_intencion: tuple = (0.10, 0.22, 0.28, 0.25, 0.15)  # meses entre la intencion y la solicitud

    def __post_init__(self) -> None:
        if abs(sum(self.p_retardo_intencion) - 1.0) > 1e-9:
            raise ValueError("p_retardo_intencion debe sumar 1")


DESCRIPCION_PARAMETROS = {
    "p_antiguo_desde_afiliacion": "Afiliados antiguos que siempre estuvieron en la AFP propia",
    "p_antiguo_traspaso_previo": "Afiliados antiguos que entraron antes de la ventana simulada",
    "tasa_mensual_saldo_inicial": "Rentabilidad real mensual con la que se capitaliza la historia previa",
    "densidad_cotizacion_historica": "Densidad de cotizacion supuesta para el saldo inicial",
    "sigma_saldo_inicial": "Dispersion lognormal del saldo inicial",
    "rho_ar_rentabilidad": "Persistencia AR(1) de la desviacion de rentabilidad de cada AFP",
    "sigma_ar_rentabilidad": "Innovacion mensual del AR(1) (~0,5 pp de sd en la ventana de 12 meses)",
    "castigo_rentabilidad_propia": "Castigo anual de rentabilidad de la AFP propia en su ventana",
    "castigo_desde": "Primer mes del castigo de rentabilidad",
    "castigo_hasta": "Ultimo mes del castigo de rentabilidad",
    "p_vendedor_competencia": "Prob. mensual de visita de un vendedor de la competencia",
    "p_ejecutivo_propio": "Prob. mensual de contacto del ejecutivo propio",
    "p_call_center": "Prob. mensual de contacto del call center propio",
    "p_campana_digital": "Prob. mensual de impacto de campana digital",
    "factor_saldo_alto_competencia": "Multiplicador de la visita de la competencia en el decil de saldo alto",
    "factor_retencion_ejecutivo": "Multiplicador del contacto propio tras reclamo o visita de la competencia",
    "factor_campana_marzo_agosto": "Estacionalidad de campanas en marzo y agosto",
    "p_campana_empleador": "Prob. mensual de que la competencia haga una charla de venta en una empresa",
    "meses_campana_empleador": "Duracion en meses de esa campana por empleador",
    "factor_campana_empleador": "Multiplicador de la visita de la competencia durante la campana",
    "p_reclamo_base": "Prob. mensual base de reclamo",
    "factor_reclamo_rentabilidad_mala": "Multiplicador del reclamo si la rentabilidad propia va mal",
    "factor_reclamo_comision_alta": "Multiplicador del reclamo si la comision propia es cara",
    "lambda_login_base": "Logins mensuales medios (Poisson) de un afiliado tipico",
    "factor_login_joven": "Multiplicador de logins para jovenes y universitarios",
    "p_intencion_visible": "Fraccion de afiliados cuya intencion de traspaso se nota en los canales propios",
    "factor_login_intencion": "Multiplicador de logins durante la intencion (solo fraccion visible)",
    "p_certificado_base": "Prob. mensual de pedir certificado por motivo administrativo corriente",
    "p_certificado_hipotecario": "Prob. adicional por credito o tramite bancario (30-50 anos, renta media-alta)",
    "p_certificado_tramite_pension": "Prob. adicional a menos de 24 meses de la edad legal de pension",
    "p_certificado_cambio_empleo": "Prob. adicional tras finiquito o cambio de empleador",
    "factor_certificado_operacion_renta": "Estacionalidad del certificado en abril-mayo (Operacion Renta)",
    "p_certificado_intencion": "Prob. adicional por intencion de traspaso (solo fraccion visible)",
    "p_cambio_fondo_base": "Prob. mensual base de cambio de fondo",
    "factor_cambio_fondo_activo": "Multiplicador del cambio de fondo para el afiliado activo",
    "b0": "Intercepto del hazard mensual (escala logit)",
    "b_brecha_comision": "Efecto de 1 pp de comision sobre la AFP mas barata",
    "b_brecha_rentabilidad": "Efecto de 1 pp de rentabilidad 12m bajo el promedio del sistema",
    "b_vendedor_competencia_3m": "Efecto de una visita de la competencia en los ultimos 3 meses",
    "b_vendedor_interesado_3m": "Efecto adicional si esa visita termino en INTERESADO",
    "b_reclamo_6m": "Efecto de un reclamo en los ultimos 6 meses",
    "b_reclamo_desfavorable_6m": "Efecto adicional si el reclamo se resolvio en contra",
    "b_ejecutivo_propio_3m": "Efecto (retencion) del contacto del ejecutivo propio en 3 meses",
    "b_joven": "Efecto por cada ano bajo los 40",
    "b_renta": "Efecto por log(RIM de referencia / IMM)",
    "b_antiguedad": "Efecto por log1p(anos en la AFP)",
    "b_cambios_fondo_12m": "Efecto por cambio de fondo en 12 meses (tope 3)",
    "b_cesante": "Efecto de estar cesante",
    "b_retirado": "Efecto de estar retirado",
    "b_marzo": "Estacionalidad de marzo",
    "b_agosto": "Estacionalidad de agosto",
    "b_u_switcher": "Desplazamiento de la propension para quien ya se cambio de AFP antes",
    "p_switcher_dado_traspaso": "P(switcher activo | llego de otra AFP): el resto se movio por asignacion",
    "p_switcher_sin_traspaso": "P(switcher activo | sin brecha de antiguedad): switcher que aun no se movio",
    "sigma_afiliado": "Desviacion del efecto aleatorio NO observable por afiliado",
    "sigma_mes": "Desviacion del ruido idiosincratico mensual",
    "p_retardo_intencion": "Distribucion del retardo (meses) entre la intencion y la firma",
}


def tabla_parametros(par: "ParametrosChurn | None" = None) -> pd.DataFrame:
    """Tabla parametro / valor / descripcion generada desde `ParametrosChurn` (fuente unica)."""
    par = par or ParametrosChurn()
    filas = []
    for campo in fields(par):
        v = getattr(par, campo.name)
        filas.append({
            "parametro": campo.name,
            "valor": ", ".join(str(x) for x in v) if isinstance(v, tuple) else v,
            "descripcion": DESCRIPCION_PARAMETROS.get(campo.name, ""),
        })
    return pd.DataFrame(filas)


def tabla_parametros_markdown(par: "ParametrosChurn | None" = None) -> str:
    """La misma tabla en markdown, para pegarla (y verificarla con un test) en el README."""
    df = tabla_parametros(par)
    lineas = ["| Parametro | Valor | Que representa |", "|---|---|---|"]
    for _, f in df.iterrows():
        lineas.append(f"| `{f['parametro']}` | {f['valor']} | {f['descripcion']} |")
    return "\n".join(lineas)


# --------------------------------------------------------------------- utilidades
def _valor_vigente(tramos: list[tuple[int, float]], periodo: int) -> float:
    """Ultimo valor de una lista [(periodo_desde, valor)] vigente en `periodo`."""
    val = tramos[0][1]
    for desde, v in tramos:
        if desde <= periodo:
            val = v
    return val


def adjudicataria_licitacion(periodo: int) -> str:
    """AFP que tiene adjudicada la licitacion de nuevos afiliados en `periodo`."""
    afp = LICITACIONES[0][1]
    for desde, nombre in LICITACIONES:
        if desde <= periodo:
            afp = nombre
    return afp


# --------------------------------------------------------------------- mercado
def generar_mercado(periodo_inicio: int, periodo_fin: int, seed: int,
                    par: ParametrosChurn | None = None) -> pd.DataFrame:
    """Tabla mes x AFP: comision, rentabilidad 12m de cada fondo y licitacion vigente.

    La rentabilidad del sistema por fondo es un paseo aleatorio mensual con la media y volatilidad
    de `FONDO_MU_ANUAL` / `FONDO_SIGMA_ANUAL`; cada AFP le suma una desviacion persistente (AR(1))
    de pocos decimos de punto, que es lo que se observa en la industria. La AFP propia arrastra un
    castigo de rentabilidad entre `castigo_desde` y `castigo_hasta` (escenario que motiva el churn).
    """
    par = par or ParametrosChurn()
    rng = np.random.default_rng(seed + 101)
    # se parte 12 meses antes para poder calcular la rentabilidad movil de 12 meses del primer mes
    periodos = rango_periodos(add_months(periodo_inicio, -12), periodo_fin)
    n_t = len(periodos)

    filas: list[dict] = []
    ret_sistema = {}
    for f in FONDOS:
        mu_m = FONDO_MU_ANUAL[f] / 12.0
        sd_m = FONDO_SIGMA_ANUAL[f] / np.sqrt(12.0)
        ret_sistema[f] = rng.normal(mu_m, sd_m, n_t)

    # desviacion persistente de cada AFP en cada fondo (AR(1) con media 0)
    desv = {}
    for a in AFP_TODAS:
        for f in FONDOS:
            e = rng.normal(0, par.sigma_ar_rentabilidad, n_t)
            x = np.zeros(n_t)
            for t in range(1, n_t):
                x[t] = par.rho_ar_rentabilidad * x[t - 1] + e[t]
            desv[(a, f)] = x

    ret = {}
    for a in AFP_TODAS:
        for f in FONDOS:
            serie = ret_sistema[f] + desv[(a, f)]
            if a == AFP_PROPIA:
                castigo = np.array([par.castigo_rentabilidad_propia / 12.0
                                    if par.castigo_desde <= p <= par.castigo_hasta else 0.0
                                    for p in periodos])
                serie = serie + castigo
            ret[(a, f)] = serie

    for t, p in enumerate(periodos):
        if p < periodo_inicio:
            continue
        for a in AFP_TODAS:
            fila = {
                "periodo": p,
                "afp": a,
                "es_propia": int(a == AFP_PROPIA),
                "comision": _valor_vigente(COMISIONES[a], p),
                "gano_licitacion": int(adjudicataria_licitacion(p) == a),
            }
            for f in FONDOS:
                v = np.prod(1.0 + ret[(a, f)][t - 11: t + 1]) - 1.0
                fila[f"rent_12m_{f}"] = round(float(v) * 100, 3)
            filas.append(fila)
    return pd.DataFrame(filas)


# --------------------------------------------------------------------- simulacion
def _matriz_panel(panel: pd.DataFrame, afil_pos: dict[int, int], per_pos: dict[int, int],
                  columna: str, dtype, relleno) -> np.ndarray:
    """Panel largo -> matriz (n_afiliados x n_periodos)."""
    m = np.full((len(afil_pos), len(per_pos)), relleno, dtype=dtype)
    i = panel["afiliado_id"].map(afil_pos).to_numpy()
    j = panel["periodo"].map(per_pos).to_numpy()
    m[i, j] = panel[columna].to_numpy()
    return m


def _fondo_defecto(edad: np.ndarray, sexo: np.ndarray) -> np.ndarray:
    """Asignacion por defecto segun tramo etario (mujeres hasta 50, hombres hasta 55)."""
    limite = np.where(sexo == "F", 50, 55)
    idx = np.where(edad <= 35, FONDOS.index(FONDO_DEFECTO_JOVEN),
                   np.where(edad <= limite, FONDOS.index(FONDO_DEFECTO_MEDIO),
                            FONDOS.index(FONDO_DEFECTO_MAYOR)))
    return idx.astype(np.int8)


def generar_tablas(n_afiliados: int = 20_000, periodo_inicio: int = 202101,
                   periodo_fin: int = 202609, seed: int = 42,
                   par: ParametrosChurn | None = None) -> dict[str, pd.DataFrame]:
    """Universo comun + tablas propias del proyecto de traspasos.

    Devuelve un dict con: afiliados, empleadores, panel_mensual, macro (del universo comun) y
    mercado, saldos, cambios_fondo, traspasos, contactos, reclamos, uso_digital (propias).
    """
    par = par or ParametrosChurn()
    rng = np.random.default_rng(seed + 202)

    universo = generar_universo(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                periodo_fin=periodo_fin, seed=seed)
    afiliados = universo["afiliados"]
    empleadores_id_max = int(universo["empleadores"]["empleador_id"].max())
    panel = universo["panel_mensual"]
    macro = universo["macro"]
    mercado = generar_mercado(periodo_inicio, periodo_fin, seed, par)

    periodos = rango_periodos(periodo_inicio, periodo_fin)
    n_t, n = len(periodos), len(afiliados)
    per_pos = {p: t for t, p in enumerate(periodos)}
    afil_pos = {a: i for i, a in enumerate(afiliados["afiliado_id"].to_numpy())}
    af_ids = afiliados["afiliado_id"].to_numpy()

    # ---- matrices del panel laboral (verdad conocida por la AFP con desfase, ver features.py)
    panel = panel[panel["periodo"].isin(per_pos)]
    rim_mat = (_matriz_panel(panel, afil_pos, per_pos, "rim_emp1", np.float64, 0.0)
               + _matriz_panel(panel, afil_pos, per_pos, "rim_emp2", np.float64, 0.0))
    estado_mat = _matriz_panel(panel, afil_pos, per_pos, "estado", object, "SIN_PANEL")
    edad_mat = _matriz_panel(panel, afil_pos, per_pos, "edad", np.int16, 0)
    edad0 = afiliados["edad_inicio"].to_numpy().astype(np.int16)
    for t in range(n_t):  # afiliados aun no afiliados: se completa la edad para usarla en el hazard
        edad_mat[:, t] = np.where(edad_mat[:, t] > 0, edad_mat[:, t], edad0 + t // 12)

    imm = macro.set_index("periodo")["imm"].reindex(periodos).to_numpy(dtype=float)
    tope = macro.set_index("periodo")["tope_clp"].reindex(periodos).to_numpy(dtype=float)
    rim_mat = np.minimum(rim_mat, tope[None, :])
    es_cesante_mat = np.isin(estado_mat, ["CESANTE", "SIN_PANEL"])
    es_retirado_mat = estado_mat == "RETIRADO"

    # ---- mercado en matrices (periodo x fondo) para la AFP propia y el sistema
    merc_idx = mercado.set_index(["periodo", "afp"])
    com_propia = np.array([merc_idx.loc[(p, AFP_PROPIA), "comision"] for p in periodos])
    com_min = np.array([min(merc_idx.loc[(p, a), "comision"] for a in AFP_TODAS) for p in periodos])
    rent_propia = np.array([[merc_idx.loc[(p, AFP_PROPIA), f"rent_12m_{f}"] for f in FONDOS]
                            for p in periodos])                                    # (n_t, 5)
    rent_prom = np.array([[np.mean([merc_idx.loc[(p, a), f"rent_12m_{f}"] for a in AFP_TODAS])
                           for f in FONDOS] for p in periodos])                    # (n_t, 5)
    licit = np.array([adjudicataria_licitacion(p) for p in periodos])
    com_comp = np.array([[merc_idx.loc[(p, a), "comision"] for a in AFP_COMPETIDORAS] for p in periodos])
    rent_comp_mat = np.array([[[merc_idx.loc[(p, a), f"rent_12m_{f}"] for f in FONDOS]
                               for a in AFP_COMPETIDORAS] for p in periodos])      # (n_t, 5 afp, 5 fondos)
    gana_comp = np.array([[1.0 if adjudicataria_licitacion(p) == a else 0.0
                           for a in AFP_COMPETIDORAS] for p in periodos])

    # ---- composicion de la cartera: quien esta en la AFP propia y desde cuando
    periodo_afil = afiliados["periodo_afiliacion"].to_numpy()
    idx_afil = np.array([periodo_to_idx(int(p)) for p in periodo_afil]) - periodo_to_idx(periodo_inicio)
    nuevo_sistema = idx_afil >= 0

    entrada = np.full(n, -1, dtype=np.int32)          # indice de mes de entrada a la AFP propia
    antig_ini = np.zeros(n, dtype=np.int32)           # meses de antiguedad al momento de entrada
    licitado = np.zeros(n, dtype=bool)
    via_traspaso = np.zeros(n, dtype=bool)

    # afiliados nuevos al sistema: entran a la adjudicataria de la licitacion vigente
    for i in np.where(nuevo_sistema)[0]:
        t = int(idx_afil[i])
        if t < n_t and licit[t] == AFP_PROPIA:
            entrada[i], licitado[i] = t, True

    antiguos = np.where(~nuevo_sistema)[0]
    u = rng.random(len(antiguos))
    meses_previos = -idx_afil[antiguos]                # meses entre afiliacion y periodo_inicio
    caso = np.where(u < par.p_antiguo_desde_afiliacion, 0,
                    np.where(u < par.p_antiguo_desde_afiliacion + par.p_antiguo_traspaso_previo, 1, 2))
    entrada[antiguos] = np.where(caso <= 1, 0, -1)
    antig_ini[antiguos] = np.where(
        caso == 0, meses_previos,
        np.where(caso == 1, np.minimum(meses_previos, 1 + np.floor(rng.random(len(antiguos)) * 120)), 0))
    idx_entrada_ventana = antiguos[caso == 2]
    t_ent = 1 + np.floor(rng.random(len(idx_entrada_ventana)) * max(n_t - 13, 1)).astype(np.int32)
    entrada[idx_entrada_ventana] = t_ent
    via_traspaso[idx_entrada_ventana] = True

    # ---- estado inicial de cuenta
    edad_ini = edad_mat[:, 0]
    sexo = afiliados["sexo"].to_numpy()
    fondo = _fondo_defecto(edad_ini, sexo)
    activo_fondo = (rng.random(n) < 0.35)
    joven = edad_ini < 40
    fondo = np.where(activo_fondo & joven, 0, np.where(activo_fondo & (edad_ini > 55), 4, fondo)).astype(np.int8)

    rim_ref = np.zeros(n)
    for i in range(n):
        fila = rim_mat[i]
        pos = np.nonzero(fila > 0)[0]
        rim_ref[i] = fila[pos[0]] if len(pos) else imm[0]
    meses_acum = np.maximum(antig_ini, np.where(nuevo_sistema, 0, -idx_afil))
    i_m = par.tasa_mensual_saldo_inicial
    factor = (np.power(1 + i_m, meses_acum) - 1) / i_m
    saldo = (TASA_COTIZACION_OBLIGATORIA * rim_ref * factor * par.densidad_cotizacion_historica
             * np.exp(rng.normal(0, par.sigma_saldo_inicial, n) - par.sigma_saldo_inicial ** 2 / 2))
    saldo[entrada < 0] = 0.0

    # rentabilidad mensual efectiva del fondo en la AFP propia (para capitalizar el saldo)
    rent_mensual = np.zeros((n_t, len(FONDOS)))
    for k, f in enumerate(FONDOS):
        serie = rent_propia[:, k] / 100.0
        rent_mensual[:, k] = np.power(1.0 + np.clip(serie, -0.5, 2.0), 1 / 12.0) - 1.0

    # ---- efectos aleatorios: parte NO observable + el desplazamiento de los "switchers"
    # (quien ya se cambio de AFP antes: la AFP lo ve porque su antiguedad en la AFP es menor que su
    # antiguedad en el sistema, asi que es un driver OBSERVABLE aunque persistente)
    # brecha de antiguedad OBSERVABLE (la AFP ve que su antiguedad es menor que la del sistema)
    con_brecha = via_traspaso | (~nuevo_sistema & (antig_ini < np.maximum(-idx_afil, 0)))
    # ...y el switcher ACTIVO, que es latente: el proxy observable falla en ambos sentidos
    p_sw = np.where(con_brecha, par.p_switcher_dado_traspaso, par.p_switcher_sin_traspaso)
    es_switcher = rng.random(n) < p_sw
    u_afil = rng.normal(0, par.sigma_afiliado, n) + par.b_u_switcher * es_switcher

    # ---- campanas de venta de la competencia por empleador (charlas en la empresa): los contactos
    # se agrupan por empleador y mes, que es como trabaja la fuerza de venta en Chile
    emp_mat = _matriz_panel(panel.assign(empleador_id=panel["empleador_id"].fillna(0).astype("int64")),
                            afil_pos, per_pos, "empleador_id", np.int64, 0)
    emp_mat = np.clip(emp_mat, 0, None)
    n_emp_max = int(max(emp_mat.max(), empleadores_id_max)) + 1
    campana_emp = np.zeros(n_emp_max, dtype=np.int32)

    # ---- estado dinamico
    en_afp = np.zeros(n, dtype=bool)
    antig = antig_ini.astype(np.int32).copy()
    intencion = np.full(n, -1, dtype=np.int32)     # meses que faltan para firmar la solicitud
    bloqueo = np.where(licitado, entrada + MESES_PERMANENCIA_LICITACION, -1)
    saldo_act = saldo.copy()

    mat_vc = np.zeros((n, n_t), dtype=bool)        # contacto de vendedor de la competencia
    mat_vc_int = np.zeros((n, n_t), dtype=bool)    # ... con resultado INTERESADO
    mat_ep = np.zeros((n, n_t), dtype=bool)        # contacto de ejecutivo propio
    mat_rec = np.zeros((n, n_t), dtype=bool)       # reclamo
    mat_rec_desf = np.zeros((n, n_t), dtype=bool)
    mat_cf = np.zeros((n, n_t), dtype=bool)        # cambio de fondo
    mat_vig = np.zeros((n, n_t), dtype=bool)       # vigente en la AFP propia
    mat_fondo = np.zeros((n, n_t), dtype=np.int8)
    mat_saldo = np.zeros((n, n_t), dtype=np.float64)
    mat_antig = np.zeros((n, n_t), dtype=np.int32)

    ev_contactos: list[dict] = []
    ev_reclamos: list[dict] = []
    ev_cambios: list[dict] = []
    ev_traspasos: list[dict] = []
    dig_logins_app = np.zeros((n, n_t), dtype=np.int16)
    dig_logins_web = np.zeros((n, n_t), dtype=np.int16)
    dig_consultas = np.zeros((n, n_t), dtype=np.int16)
    dig_certif = np.zeros((n, n_t), dtype=np.int16)

    joven_login = (rng.random(n) < 0.5) | (afiliados["nivel_educacional"].to_numpy() == "UNIVERSITARIA")
    # solo una fraccion de los afiliados manifiesta la intencion en los canales propios: el resto
    # decide por fuera (la AFP de destino tramita el traspaso y no necesita nada de la de origen)
    intencion_visible = rng.random(n) < par.p_intencion_visible
    sexo_arr = afiliados["sexo"].to_numpy()
    edad_legal = np.where(sexo_arr == "F", 60, 65)
    # cambio de estado laboral respecto del mes anterior (finiquito, nuevo empleo): motivo
    # administrativo tipico para pedir un certificado de saldo
    cambio_estado = np.zeros((n, n_t), dtype=bool)
    cambio_estado[:, 1:] = estado_mat[:, 1:] != estado_mat[:, :-1]
    retardos = np.arange(len(par.p_retardo_intencion))

    for t, periodo in enumerate(periodos):
        mes = periodo % 100
        # --- entradas del mes
        entra = (entrada == t) & (~en_afp)
        if entra.any():
            en_afp |= entra
            antig[entra] = np.where(licitado[entra], 0, antig_ini[entra])
            idx = np.where(entra & via_traspaso)[0]
            if len(idx):
                origen = rng.choice(np.array(AFP_COMPETIDORAS), size=len(idx))
                ev_traspasos.append({"afiliado_id": af_ids[idx],
                                     "periodo_solicitud": np.full(len(idx), add_months(periodo, -1)),
                                     "periodo_efectivo": np.full(len(idx), periodo),
                                     "afp_origen": origen,
                                     "afp_destino": np.full(len(idx), AFP_PROPIA),
                                     "tipo": np.full(len(idx), "ENTRADA")})
            idx_l = np.where(entra & licitado)[0]
            if len(idx_l):
                ev_traspasos.append({"afiliado_id": af_ids[idx_l],
                                     "periodo_solicitud": np.full(len(idx_l), periodo),
                                     "periodo_efectivo": np.full(len(idx_l), periodo),
                                     "afp_origen": np.full(len(idx_l), "NUEVO_AFILIADO"),
                                     "afp_destino": np.full(len(idx_l), AFP_PROPIA),
                                     "tipo": np.full(len(idx_l), "ENTRADA")})

        act = np.where(en_afp)[0]
        if len(act) == 0:
            continue
        mat_vig[act, t] = True

        rim_t = rim_mat[:, t]
        pos_rim = rim_t > 0
        rim_ref = np.where(pos_rim, rim_t, rim_ref)
        renta_rel = np.clip(np.log(np.maximum(rim_ref, 1.0) / imm[t]), -1.0, 2.5)

        # --- 1. contactos comerciales
        saldo_pct = np.zeros(n)
        if len(act) > 1:
            orden = np.argsort(np.argsort(saldo_act[act]))
            saldo_pct[act] = orden / max(len(act) - 1, 1)
        estac = par.factor_campana_marzo_agosto if mes in (3, 8) else 1.0

        campana_emp[campana_emp > 0] -= 1
        nuevas = rng.random(n_emp_max) < par.p_campana_empleador
        nuevas[0] = False                                   # 0 = sin empleador
        campana_emp[nuevas] = par.meses_campana_empleador
        en_campana = campana_emp[emp_mat[:, t]] > 0
        en_campana[emp_mat[:, t] == 0] = False

        p_vc = np.zeros(n)
        p_vc[act] = (par.p_vendedor_competencia * estac
                     * (1.0 + (par.factor_saldo_alto_competencia - 1.0) * saldo_pct[act] ** 3)
                     * (1.0 + 0.5 * (renta_rel[act] > 1.0))
                     * np.where(en_campana[act], par.factor_campana_empleador, 1.0))
        vc = (rng.random(n) < p_vc) & en_afp

        reactivo = (mat_vc[:, max(0, t - 2):t].any(axis=1) | mat_rec[:, max(0, t - 3):t].any(axis=1))
        p_ep = np.zeros(n)
        p_ep[act] = (par.p_ejecutivo_propio
                     * (1.0 + (par.factor_retencion_ejecutivo - 1.0) * reactivo[act])
                     * (1.0 + 1.2 * saldo_pct[act] ** 2))
        ep = (rng.random(n) < p_ep) & en_afp
        cc = (rng.random(n) < par.p_call_center) & en_afp
        cd = (rng.random(n) < par.p_campana_digital * estac) & en_afp

        for canal, flag in (("VENDEDOR_COMPETENCIA", vc), ("EJECUTIVO_PROPIO", ep),
                            ("CALL_CENTER", cc), ("CAMPANA_DIGITAL", cd)):
            idx = np.where(flag)[0]
            if not len(idx):
                continue
            if canal == "VENDEDOR_COMPETENCIA":
                p_res = [0.30, 0.38, 0.22, 0.10]
            elif canal == "EJECUTIVO_PROPIO":
                p_res = [0.25, 0.55, 0.14, 0.06]
            else:
                p_res = [0.55, 0.33, 0.06, 0.06]
            res = rng.choice(np.array(RESULTADOS_CONTACTO), size=len(idx), p=p_res)
            ev_contactos.append({"afiliado_id": af_ids[idx], "periodo": np.full(len(idx), periodo),
                                 "canal": np.full(len(idx), canal), "resultado": res})
            if canal == "VENDEDOR_COMPETENCIA":
                mat_vc[idx, t] = True
                mat_vc_int[idx[res == "INTERESADO"], t] = True
            elif canal == "EJECUTIVO_PROPIO":
                mat_ep[idx, t] = True

        # --- 2. reclamos
        brecha_rent_t = rent_propia[t, fondo] - rent_prom[t, fondo]
        brecha_com_t = com_propia[t] - com_min[t]
        p_rec = np.zeros(n)
        p_rec[act] = (par.p_reclamo_base
                      * np.where(brecha_rent_t[act] < -0.5, par.factor_reclamo_rentabilidad_mala, 1.0)
                      * (par.factor_reclamo_comision_alta if brecha_com_t > 0.5 else 1.0)
                      * (1.0 + 0.8 * (edad_mat[:, t][act] > 55)))
        rec = (rng.random(n) < p_rec) & en_afp
        idx = np.where(rec)[0]
        if len(idx):
            w = np.array([0.20 + 0.35 * (brecha_rent_t[i] < -0.5) for i in idx])
            r = rng.random(len(idx))
            tipo = np.where(r < w, "RENTABILIDAD",
                            np.where(r < w + 0.35, "ATENCION",
                                     np.where(r < w + 0.55, "COMISION", "PENSION")))
            estado_rec = rng.choice(np.array(ESTADOS_RECLAMO), size=len(idx), p=[0.20, 0.50, 0.30])
            ev_reclamos.append({"afiliado_id": af_ids[idx], "periodo": np.full(len(idx), periodo),
                                "tipo": tipo, "estado": estado_rec})
            mat_rec[idx, t] = True
            mat_rec_desf[idx[estado_rec == "RESUELTO_DESFAVORABLE"], t] = True

        # --- 3. hazard de traspaso y arranque de la intencion
        vc3 = mat_vc[:, max(0, t - 2):t + 1].any(axis=1)
        vci3 = mat_vc_int[:, max(0, t - 2):t + 1].any(axis=1)
        ep3 = mat_ep[:, max(0, t - 2):t + 1].any(axis=1)
        rec6 = mat_rec[:, max(0, t - 5):t + 1].any(axis=1)
        recd6 = mat_rec_desf[:, max(0, t - 5):t + 1].any(axis=1)
        cf12 = mat_cf[:, max(0, t - 11):t + 1].sum(axis=1)

        z = (par.b0
             + par.b_brecha_comision * brecha_com_t
             + par.b_brecha_rentabilidad * np.maximum(0.0, -brecha_rent_t)
             + par.b_vendedor_competencia_3m * vc3
             + par.b_vendedor_interesado_3m * vci3
             + par.b_reclamo_6m * rec6
             + par.b_reclamo_desfavorable_6m * recd6
             + par.b_ejecutivo_propio_3m * ep3
             + par.b_joven * np.maximum(0, 40 - edad_mat[:, t])
             + par.b_renta * renta_rel
             + par.b_antiguedad * np.log1p(antig / 12.0)
             + par.b_cambios_fondo_12m * np.minimum(cf12, 3)
             + par.b_cesante * es_cesante_mat[:, t]
             + par.b_retirado * es_retirado_mat[:, t]
             + (par.b_marzo if mes == 3 else 0.0)
             + (par.b_agosto if mes == 8 else 0.0)
             + u_afil
             + rng.normal(0, par.sigma_mes, n))
        h = 1.0 / (1.0 + np.exp(-z))
        h = np.where(en_afp & (intencion < 0), h, 0.0)
        h = np.where((bloqueo >= 0) & (t < bloqueo), 0.0, h)   # permanencia obligatoria por licitacion
        arranca = rng.random(n) < h
        idx = np.where(arranca)[0]
        if len(idx):
            intencion[idx] = rng.choice(retardos, size=len(idx), p=list(par.p_retardo_intencion))

        # --- 4. uso digital: la intencion se nota ANTES de la solicitud, pero solo en la fraccion
        # `p_intencion_visible` de los afiliados y con multiplicadores moderados. La mayor parte de
        # las descargas de certificado tiene causas exogenas (credito, tramite de pension, cambio de
        # empleo, Operacion Renta), no el traspaso.
        con_int = intencion >= 0
        int_vis = con_int & intencion_visible
        lam = np.zeros(n)
        lam[act] = (par.lambda_login_base
                    * np.where(joven_login[act], par.factor_login_joven, 1.0)
                    * (1.0 + 0.6 * saldo_pct[act])
                    * np.where(int_vis[act], par.factor_login_intencion, 1.0))
        la = rng.poisson(lam)
        lw = rng.poisson(lam * 0.55)
        cs = rng.poisson(lam * 0.8)
        edad_t = edad_mat[:, t]
        p_cert = (par.p_certificado_base
                  * (par.factor_certificado_operacion_renta if mes in (4, 5) else 1.0)
                  + par.p_certificado_hipotecario * ((edad_t >= 30) & (edad_t <= 50) & (renta_rel > 0.3))
                  + par.p_certificado_tramite_pension * ((edad_legal - edad_t) * 12 < 24)
                  + par.p_certificado_cambio_empleo * cambio_estado[:, t]
                  + par.p_certificado_intencion * int_vis)
        ce = (rng.random(n) < p_cert) & en_afp
        dig_logins_app[act, t] = la[act]
        dig_logins_web[act, t] = lw[act]
        dig_consultas[act, t] = cs[act]
        dig_certif[act, t] = ce[act]

        # --- 5. solicitudes de traspaso (salida)
        firma = con_int & (intencion == 0) & en_afp
        idx = np.where(firma)[0]
        if len(idx):
            # la AFP destino se elige por comision baja, buena rentabilidad 12m del fondo del
            # afiliado y un pequeno empuje de la adjudicataria de la licitacion (mas publicidad)
            score = (-1.6 * com_comp[t][None, :]
                     + 0.22 * rent_comp_mat[t][:, fondo[idx]].T
                     + 0.45 * gana_comp[t][None, :])
            w = np.exp(score - score.max(axis=1, keepdims=True))
            w = w / w.sum(axis=1, keepdims=True)
            sorteo = rng.random(len(idx))[:, None]
            destino = np.array(AFP_COMPETIDORAS)[(w.cumsum(axis=1) < sorteo).sum(axis=1).clip(0, 4)]
            ev_traspasos.append({"afiliado_id": af_ids[idx],
                                 "periodo_solicitud": np.full(len(idx), periodo),
                                 "periodo_efectivo": np.full(len(idx), add_months(periodo, 1)),
                                 "afp_origen": np.full(len(idx), AFP_PROPIA),
                                 "afp_destino": destino,
                                 "tipo": np.full(len(idx), "SALIDA")})

        # --- 6. cambios de fondo
        p_cf = np.zeros(n)
        activo = joven_login & (renta_rel > 0.4)
        mala_rent = brecha_rent_t < -0.5
        p_cf[act] = (par.p_cambio_fondo_base
                     * np.where(activo[act], par.factor_cambio_fondo_activo, 1.0)
                     * np.where(mala_rent[act], 1.8, 1.0))
        cf = (rng.random(n) < p_cf) & en_afp & (~firma)
        idx = np.where(cf)[0]
        if len(idx):
            paso = rng.choice(np.array([-1, 1]), size=len(idx), p=[0.45, 0.55])
            destino_f = np.clip(fondo[idx] + paso, 0, len(FONDOS) - 1)
            mueve = destino_f != fondo[idx]
            idx, destino_f = idx[mueve], destino_f[mueve]
            if len(idx):
                ev_cambios.append({"afiliado_id": af_ids[idx], "periodo": np.full(len(idx), periodo),
                                   "fondo_origen": np.array([FONDOS[k] for k in fondo[idx]]),
                                   "fondo_destino": np.array([FONDOS[k] for k in destino_f])})
                mat_cf[idx, t] = True
                fondo[idx] = destino_f

        # --- 7. cierre del mes: saldo, antiguedad, salidas efectivas
        aporte = np.zeros(n)
        if t >= 2:  # la cotizacion del mes m se recauda y acredita en m+2
            aporte = TASA_COTIZACION_OBLIGATORIA * rim_mat[:, t - 2]
        saldo_act = np.where(en_afp, saldo_act * (1.0 + rent_mensual[t, fondo]) + aporte, saldo_act)
        saldo_act = np.maximum(saldo_act, 0.0)
        mat_saldo[act, t] = saldo_act[act]
        mat_fondo[act, t] = fondo[act]
        mat_antig[act, t] = antig[act]
        antig[act] += 1
        intencion[con_int] -= 1
        en_afp = np.where(firma, False, en_afp)   # el traspaso es efectivo al mes siguiente
        intencion[firma] = -1

    # ---------------------------------------------------------------- tablas finales
    def _concat(evs: list[dict]) -> pd.DataFrame:
        if not evs:
            return pd.DataFrame()
        return pd.concat([pd.DataFrame(e) for e in evs], ignore_index=True)

    contactos = _concat(ev_contactos)
    reclamos = _concat(ev_reclamos)
    cambios_fondo = _concat(ev_cambios)
    traspasos = _concat(ev_traspasos)

    ii, jj = np.nonzero(mat_vig)
    per_arr = np.array(periodos)
    saldos = pd.DataFrame({
        "afiliado_id": af_ids[ii],
        "periodo": per_arr[jj],
        "saldo_obligatorio": np.round(mat_saldo[ii, jj], 0),
        "fondo": np.array(FONDOS)[mat_fondo[ii, jj]],
        "antiguedad_afp_meses": mat_antig[ii, jj],
    })
    uso_digital = pd.DataFrame({
        "afiliado_id": af_ids[ii],
        "periodo": per_arr[jj],
        "logins_app": dig_logins_app[ii, jj],
        "logins_web": dig_logins_web[ii, jj],
        "consultas_saldo": dig_consultas[ii, jj],
        "descargas_certificado": dig_certif[ii, jj],
    })

    for df in (contactos, reclamos, cambios_fondo, traspasos):
        if len(df):
            df.sort_values(["afiliado_id", "periodo" if "periodo" in df.columns else "periodo_solicitud"],
                           inplace=True, ignore_index=True)

    latente = pd.DataFrame({"afiliado_id": af_ids, "lat_u_traspaso": u_afil,
                            "lat_intencion_visible": intencion_visible.astype(int),
                            "lat_switcher": es_switcher.astype(int),
                            "lat_con_brecha_antiguedad": con_brecha.astype(int)})
    return {
        "afiliados": afiliados,
        "afiliados_latente_churn": latente,
        "empleadores": universo["empleadores"],
        "panel_mensual": panel.reset_index(drop=True),
        "macro": macro,
        "mercado": mercado,
        "saldos": saldos,
        "cambios_fondo": cambios_fondo,
        "traspasos": traspasos,
        "contactos": contactos,
        "reclamos": reclamos,
        "uso_digital": uso_digital,
    }


def resumen_simulacion(tablas: dict[str, pd.DataFrame], dataset: pd.DataFrame | None = None,
                       prob: np.ndarray | None = None) -> pd.DataFrame:
    """Chequeos de plausibilidad y de **determinismo** de la simulacion.

    Los de plausibilidad salen de las tablas (tasa de traspaso y su maximo/minimo movil de 12 meses,
    dispersion de rentabilidad entre AFP, frecuencia de contactos y reclamos). Los de determinismo
    necesitan el dataset ya armado (y opcionalmente el score del modelo) y responden a la pregunta
    "¿hay alguna feature que sea una observacion disfrazada del target?":

      * `p_target_dado_certificado` y su lift: si pedir el certificado casi garantizara el traspaso,
        la simulacion estaria regalando el target;
      * `auc_mejor_feature_suelta`: ninguna feature individual deberia ordenar casi tan bien como el
        modelo completo;
      * `frac_score_mayor_0_9`: una cola de probabilidades saturadas indica un mundo determinista.
    """
    tr = tablas["traspasos"]
    salidas = tr[tr["tipo"] == "SALIDA"] if len(tr) else tr
    saldos = tablas["saldos"]
    cartera_mes = saldos.groupby("periodo")["afiliado_id"].count()
    salidas_mes = salidas.groupby("periodo_solicitud")["afiliado_id"].count().reindex(cartera_mes.index).fillna(0)
    tasa_12m = (salidas_mes.rolling(12).sum() / cartera_mes.rolling(12).mean()).dropna()
    merc = tablas["mercado"]
    rangos = []
    for f in FONDOS:
        piv = merc.pivot(index="periodo", columns="afp", values=f"rent_12m_{f}")
        rangos.append(piv.max(axis=1) - piv.min(axis=1))
    rango_rent = pd.concat(rangos)
    n_anios = max(len(cartera_mes) / 12, 1e-9)
    # cotizantes = afiliados de la cartera con RIM > 0 ese mes (el brief fija la tasa objetivo sobre
    # la cartera COTIZANTE, no sobre la cartera total)
    pan = tablas["panel_mensual"]
    rim = pan["rim_emp1"].to_numpy() + pan["rim_emp2"].to_numpy()
    cot = pan.loc[rim > 0, ["afiliado_id", "periodo"]].merge(
        saldos[["afiliado_id", "periodo"]], on=["afiliado_id", "periodo"])
    cot_mes = cot.groupby("periodo")["afiliado_id"].count().reindex(cartera_mes.index)
    tasa_cot = float((salidas_mes / cot_mes).mean() * 12)
    filas = [
        {"indicador": "afiliados_en_cartera_promedio", "valor": float(cartera_mes.mean()), "detalle": ""},
        {"indicador": "tasa_traspaso_anualizada", "valor": float((salidas_mes / cartera_mes).mean() * 12), "detalle": "sobre la cartera total"},
        {"indicador": "tasa_traspaso_anualizada_cotizantes", "valor": tasa_cot, "detalle": "brief: 4-7 %"},
        {"indicador": "tasa_traspaso_12m_min", "valor": float(tasa_12m.min()), "detalle": "minimo movil de 12 meses"},
        {"indicador": "tasa_traspaso_12m_max", "valor": float(tasa_12m.max()), "detalle": "maximo movil de 12 meses"},
        {"indicador": "rango_rentabilidad_12m_entre_afp_mediana_pp", "valor": float(rango_rent.median()), "detalle": "real en Chile: 1-2 pp"},
        {"indicador": "rango_rentabilidad_12m_entre_afp_p95_pp", "valor": float(rango_rent.quantile(0.95)), "detalle": ""},
        {"indicador": "traspasos_salida_total", "valor": float(len(salidas)), "detalle": ""},
        {"indicador": "traspasos_entrada_total", "valor": float((tr["tipo"] == "ENTRADA").sum()) if len(tr) else 0.0, "detalle": ""},
        {"indicador": "saldo_mediano_clp", "valor": float(saldos["saldo_obligatorio"].median()), "detalle": ""},
        {"indicador": "contactos_por_afiliado_ano", "valor": float(len(tablas["contactos"]) / max(cartera_mes.mean(), 1) / n_anios), "detalle": ""},
        {"indicador": "reclamos_por_afiliado_ano", "valor": float(len(tablas["reclamos"]) / max(cartera_mes.mean(), 1) / n_anios), "detalle": ""},
        {"indicador": "p_descarga_certificado_mes", "valor": float(tablas["uso_digital"]["descargas_certificado"].mean()), "detalle": ""},
    ]
    if dataset is not None and len(dataset):
        filas.extend(chequeos_determinismo(dataset, prob))
        filas.extend(_chequeos_proxy_latente(tablas, dataset))
    return pd.DataFrame(filas)


def _chequeos_proxy_latente(tablas: dict[str, pd.DataFrame], dataset: pd.DataFrame) -> list[dict]:
    """Concordancia entre una feature observable y el latente que la inspira.

    Una feature que reproduce un latente con concordancia ~1 es el latente disfrazado: el gerente
    rechazo `llego_por_traspaso` en la ronda 2 justamente por eso.
    """
    lat = tablas.get("afiliados_latente_churn")
    if lat is None or "llego_por_traspaso" not in dataset.columns:
        return []
    m = dataset[["afiliado_id", "llego_por_traspaso"]].merge(
        lat[["afiliado_id", "lat_switcher"]], on="afiliado_id", how="left")
    a = m["llego_por_traspaso"].to_numpy()
    b = m["lat_switcher"].to_numpy()
    return [
        {"indicador": "frac_llego_por_traspaso", "valor": float(a.mean()), "detalle": "proxy observable"},
        {"indicador": "frac_switcher_latente", "valor": float(b.mean()), "detalle": "latente real"},
        {"indicador": "concordancia_proxy_switcher", "valor": float((a == b).mean()),
         "detalle": "1,0 = la feature es el latente disfrazado"},
    ]


def chequeos_determinismo(dataset: pd.DataFrame, prob: np.ndarray | None = None,
                          max_filas_auc: int = 60_000) -> list[dict]:
    """Chequeos de que ninguna feature sea una observacion directa del target futuro."""
    from sklearn.metrics import roc_auc_score

    from .features import columnas_features

    y = dataset["y"].to_numpy()
    base = float(y.mean())
    out = [{"indicador": "tasa_base_dataset", "valor": base, "detalle": ""}]
    if "certificados_1m" in dataset.columns:
        con_cert = dataset["certificados_1m"].to_numpy() > 0
        p_cert = float(y[con_cert].mean()) if con_cert.any() else float("nan")
        out += [
            {"indicador": "frac_filas_con_certificado_1m", "valor": float(con_cert.mean()), "detalle": ""},
            {"indicador": "p_target_dado_certificado", "valor": p_cert, "detalle": "P(y=1 | certificados_1m>0)"},
            {"indicador": "lift_certificado", "valor": p_cert / max(base, 1e-9), "detalle": "vs tasa base"},
        ]
    # AUC de la mejor feature numerica suelta (sobre una muestra, para no encarecer el run)
    d = dataset if len(dataset) <= max_filas_auc else dataset.sample(max_filas_auc, random_state=0)
    yy = d["y"].to_numpy()
    mejor, mejor_auc = "", 0.5
    if 0 < yy.mean() < 1:
        for c in columnas_features(d):
            v = d[c]
            if str(v.dtype) == "category" or v.dtype == object:
                continue
            x = v.to_numpy(dtype=float)
            if not np.isfinite(x).any() or np.nanstd(x) == 0:
                continue
            a = roc_auc_score(yy, np.nan_to_num(x, nan=float(np.nanmedian(x))))
            a = max(a, 1 - a)
            if a > mejor_auc:
                mejor, mejor_auc = c, a
    out.append({"indicador": "auc_mejor_feature_suelta", "valor": float(mejor_auc), "detalle": mejor})
    if prob is not None:
        pr = np.asarray(prob, dtype=float)
        out += [
            {"indicador": "frac_score_mayor_0_9", "valor": float((pr > 0.9).mean()), "detalle": ""},
            {"indicador": "frac_score_mayor_0_5", "valor": float((pr > 0.5).mean()), "detalle": ""},
            {"indicador": "score_maximo", "valor": float(pr.max()), "detalle": ""},
        ]
    return out
