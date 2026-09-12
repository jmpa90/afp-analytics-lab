"""Tablas sinteticas del proyecto 05 (riesgo de empleador: DNP y pago tardio).

Capas:

1. `common.synth_base.generar_universo(perfil="cuprum")`: afiliados, empleadores, panel laboral.
2. `common.planilla.generar_planilla` (modulo de DS-3, copiado a este worktree; ver §"Version del
   modulo de planilla" mas abajo): planilla Previred **linea a linea** (afiliado x empleador x
   periodo x tipo_pago) con los codigos de la Tabla N 7, dias trabajados, renta imponible AFP/AFC,
   cotizacion obligatoria, SIS y aporte del empleador. De ahi se toma el **contenido** de la
   planilla, agregado a empleador x periodo.
3. Este modulo: la **capa de recepcion y pago a nivel de empleador**, que es la que el proyecto 05
   necesita y que `common.planilla` no modela: persistencia de la DNP, estacionalidad, episodios de
   estres financiero con regularizacion o liquidacion, y los boletines publicos.

| Tabla | Grano | Contenido |
|---|---|---|
| `planillas` | empleador x periodo | n_trabajadores, masa imponible, cotizacion, `periodo_recepcion`, `estado` (PAGADA / DNP / SIN_DECLARAR), `periodo_regularizacion`, `n_rectificaciones`, altas/bajas/licencias |
| `boletin_infractores` | empleador x mes | inclusion en el Boletin de Infractores Laborales y Previsionales (DT) tras `MESES_DEUDA_BOLETIN` meses de deuda, con un mes de rezago de publicacion |
| `boletin_concursal` | evento | REORGANIZACION / LIQUIDACION (Superir) con `periodo_publicacion` |
| `shocks_empleador` | latente | episodios de estres financiero (inicio, fin, desenlace). **Nunca es feature** |

Por que la capa de recepcion es propia y no la de `common.planilla`: ese modulo sortea el rezago y
la DNP **linea por linea** con las probabilidades fijas por tamano de `empleadores_latente`, sin
persistencia, sin estacionalidad y sin episodios. Con eso el target del proyecto 05 seria casi
impredecible mas alla del tamano del empleador y el brief exige explicitamente persistencia,
estacionalidad y shocks. La coherencia se mantiene porque aqui se decide **una sola vez por
(empleador, periodo)** y esa decision es la que se reporta en `planillas`.

Todos los parametros estan explicitos como constantes de modulo. `empleadores_latente.p_*` y la
tabla `shocks_empleador` se usan **solo para generar**; el README lo documenta y hay test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afp_lab.common.planilla import generar_planilla
from afp_lab.common.synth_base import generar_universo
from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx, rango_periodos

# --------------------------------------------------------------------------- calendario
PERIODO_INICIO = 201901          # 12 meses de burn-in para el historial de pago del empleador
PERIODO_FIN = 202609             # "hoy" del laboratorio
PERFIL = "cuprum"                # somos AFP Cuprum (docs/contexto_afp_cuprum.md)
HORIZONTE = 1                    # target principal: la planilla del periodo `corte`
HORIZONTE_LARGO = 3              # variante: alguna DNP en corte .. corte+2

# Version del modulo de planilla de DS-3 usado (copiado a este worktree el 2026-09-12):
# afp_lab/common/planilla.py md5 1fc28ef5cf5dea3d5ca189f54e872f59
# afp_lab/common/tablas_equivalencia.py md5 ea60c8bc886f0d127fba5864fe0c41e1
# La revision anterior (18d3e167) difiere solo en el docstring: se verifico que la planilla
# generada es identica bit a bit, asi que los numeros del README valen para las dos.
VERSION_PLANILLA = "planilla.py@1fc28ef5 (DS-3, 2026-09-12)"

# --------------------------------------------------------------------------- estados
ESTADOS_PLANILLA = ("PAGADA", "DNP", "SIN_DECLARAR")
COD_PAGADA, COD_DNP, COD_SIN_DECLARAR, COD_NO_APLICA = 1, 2, 3, 0
LAG_INFINITO = 999               # planilla que nunca llega

# --------------------------------------------------------------------------- comportamiento de pago
# Puntos de partida: las mismas probabilidades por tamano del proyecto 01 (TAMANO_PAGO en
# rim_nowcasting.synthetic), que viajan en `empleadores_latente.p_*`. Aqui se les agrega la
# dinamica que el brief pide. El sector publico no cae nunca en DNP (paga por ley).
RUBRO_PUBLICO = "O_ADMIN_PUBLICA"
# Multiplicador de la probabilidad de DNP por rubro (rubros con mas mora previsional en Chile:
# construccion, agricultura de temporada, turismo y servicios intensivos en mano de obra).
MULT_DNP_RUBRO = {"F_CONSTRUCCION": 1.8, "A_AGRICULTURA": 1.7, "I_TURISMO": 1.6,
                  "N_SERVICIOS": 1.3, "G_COMERCIO": 1.2, "T_CASA_PARTICULAR": 1.2,
                  "B_MINERIA": 0.5, "K_FINANZAS": 0.5, "Q_SALUD": 0.8, "P_EDUCACION": 0.9}
# Estacionalidad: diciembre-marzo concentran gratificacion, aguinaldo, vacaciones y marzo
MULT_DNP_MES = {12: 1.5, 1: 1.6, 2: 1.4, 3: 1.3}
# Heterogeneidad estable del empleador ("fragilidad"): en Chile la mora previsional esta muy
# concentrada, hay empleadores cronicamente atrasados. Es un factor log-normal por empleador que
# multiplica la probabilidad de DNP y de recepcion tardia y **no cambia en el tiempo**, asi que el
# historial de pago de 12 meses lo revela: es la fuente principal de predictibilidad del modelo.
SIGMA_FRAGILIDAD = 0.90
# Persistencia: un empleador que declaro y no pago el mes anterior tiende a repetir
MULT_DNP_REPITE = 3.5
MULT_DNP_REPITE_2M = 1.6         # extra si viene de dos meses seguidos
# Episodios de estres financiero
P_ESTRES_BASE = 0.006            # hazard mensual de entrar en estres (empleador tipico)
MULT_ESTRES_TAMANO = {"MICRO": 2.2, "PEQUENA": 1.5, "MEDIANA": 0.8, "GRANDE": 0.35}
ELAST_ESTRES_DESEMPLEO = 12.0    # por punto de desempleo sobre 8 %
P_SALIDA_ESTRES = 0.12           # duracion media ~8 meses
P_LIQUIDACION_TRAS_ESTRES = 0.15 # el resto regulariza
# Tras la liquidacion el empleador deja de declarar pero sus trabajadores siguen en la nomina unos
# meses (hasta el finiquito): esos meses son planillas NO DECLARADAS con deuda real, y despues el
# empleador sale del universo.
MESES_SIN_DECLARAR_TRAS_LIQUIDACION = 3
MULT_DNP_ESTRES = 10.0
MULT_TARDIO_ESTRES = 3.0
P_REORGANIZACION_EN_ESTRES = 0.10   # publica reorganizacion (Superir) durante el episodio
# Regularizacion de la DNP (pago de la deuda declarada)
REGULARIZACION_MESES = (2, 9)    # meses tras la declaracion, uniforme
P_NUNCA_REGULARIZA = 0.12        # deuda que queda impaga (tipica de empleadores liquidados)
# Boletin de Infractores Laborales y Previsionales (DT): se publica tras k meses de deuda,
# con un mes de rezago, y se mantiene mientras haya deuda y dos meses mas.
MESES_DEUDA_BOLETIN = 2
REZAGO_PUBLICACION_BOLETIN = 1
MESES_PERMANENCIA_BOLETIN = 2
# Rectificaciones: se heredan de `p_rectificacion` del empleador
MULT_RECTIFICACION_ESTRES = 1.5


@dataclass(frozen=True)
class ParametrosEmpleador:
    """Escalares que calibran las tasas objetivo del brief (3-6 % DNP, 5-10 % recepcion tardia)."""

    # Calibrados para las tasas objetivo del brief con 20.000 afiliados: 5,6 % de las planillas
    # del mes con deuda (DNP 4,3 % + no declarada 1,3 %) y 8,6 % con recepcion tardia (lag >= 2).
    factor_dnp: float = 0.45        # escala global de la probabilidad de DNP
    factor_tardio: float = 0.65     # escala global de la probabilidad de recepcion tardia
    tope_p_dnp: float = 0.85
    tope_p_tardio: float = 0.90


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# =========================================================================== contenido de planilla
def _contenido_planilla(universo: dict[str, pd.DataFrame], seed: int,
                        empleadores: pd.DataFrame, periodos: list[int]) -> pd.DataFrame:
    """Agrega la planilla Previred (linea a linea, DS-3) a grano empleador x periodo.

    Solo se usan las lineas `DECLARACION` (las de RECTIFICACION y PAGO_DNP repiten la misma RIM y
    duplicarian la masa). De los codigos de movimiento de personal salen altas, bajas y licencias.
    """
    pl = generar_planilla(universo, seed=seed)
    dec = pl[pl["tipo_movimiento"] == "DECLARACION"]
    g = dec.groupby(["empleador_id", "periodo"], observed=True).agg(
        n_trabajadores=("rut_trabajador", "nunique"),
        masa_imponible=("renta_imponible_afp", "sum"),
        cotizacion_obligatoria=("cotizacion_obligatoria", "sum"),
        aporte_empleador=("aporte_empleador", "sum"),
        dias_trabajados_medio=("dias_trabajados", "mean"),
    ).reset_index()
    # movimientos de personal (Tabla N 7): 1/7 contratacion, 2 retiro, 3 subsidio (licencia)
    cod = dec.drop_duplicates(["rut_trabajador", "empleador_id", "periodo"])
    mov = cod.assign(
        alta=cod["codigo_movimiento_personal"].isin([1, 7]).astype(int),
        baja=(cod["codigo_movimiento_personal"] == 2).astype(int),
        licencia=(cod["codigo_movimiento_personal"] == 3).astype(int),
        plazo_fijo=(cod["tipo_contrato_planilla"] == "PLAZO_FIJO").astype(int),
    ).groupby(["empleador_id", "periodo"], observed=True)[
        ["alta", "baja", "licencia", "plazo_fijo"]].sum().reset_index()
    g = g.merge(mov, on=["empleador_id", "periodo"], how="left").fillna(0.0)
    g["frac_plazo_fijo"] = g["plazo_fijo"] / np.maximum(g["n_trabajadores"], 1)
    g = g.rename(columns={"alta": "n_altas", "baja": "n_bajas", "licencia": "n_licencias"})
    g["empleador_id"] = g["empleador_id"].astype("int64")
    g["periodo"] = g["periodo"].astype("int32")
    return g.drop(columns=["plazo_fijo"])


# =========================================================================== capa de recepcion
def _simular_recepcion(contenido: pd.DataFrame, empleadores_lat: pd.DataFrame, macro: pd.DataFrame,
                       periodos: list[int], rng: np.random.Generator,
                       params: ParametrosEmpleador) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decide, por empleador y periodo, si la planilla se paga, se declara sin pagar o no se declara.

    Devuelve (`planillas`, `shocks_empleador`, `boletin_concursal`).
    """
    emp = empleadores_lat.sort_values("empleador_id").reset_index(drop=True)
    n_emp, n_per = len(emp), len(periodos)
    pos = {p: i for i, p in enumerate(periodos)}
    fila_emp = pd.Series(np.arange(n_emp), index=emp["empleador_id"].to_numpy())

    # matriz densa de presencia y contenido
    ie = contenido["empleador_id"].map(fila_emp).to_numpy()
    ip = contenido["periodo"].map(pos).to_numpy()
    presente = np.zeros((n_emp, n_per), dtype=bool)
    presente[ie, ip] = True

    rubro = emp["rubro"].to_numpy()
    publico = rubro == RUBRO_PUBLICO
    mult_rubro = np.array([MULT_DNP_RUBRO.get(r, 1.0) for r in rubro])
    mult_estres_tam = np.array([MULT_ESTRES_TAMANO[t] for t in emp["tamano_empleador"]])
    # fragilidad estable por empleador (latente; se usa solo para generar)
    fragilidad = rng.lognormal(-0.5 * SIGMA_FRAGILIDAD ** 2, SIGMA_FRAGILIDAD, n_emp)
    p_dnp_base = emp["p_dnp"].to_numpy() * params.factor_dnp * fragilidad
    p_tardio_base = ((emp["p_pago_tardio"].to_numpy() + emp["p_rezago"].to_numpy())
                     * params.factor_tardio * np.sqrt(fragilidad))
    p_antic = emp["p_pago_anticipado"].to_numpy()
    p_rect = emp["p_rectificacion"].to_numpy()

    mac = macro.set_index("periodo").loc[periodos]
    desempleo = mac["desempleo"].to_numpy() / 100.0
    meses = np.array([p % 100 for p in periodos])

    # estado latente
    estres = np.zeros(n_emp, dtype=bool)
    liquidado = np.zeros(n_emp, dtype=bool)
    mes_liquidacion = np.full(n_emp, 10**6, dtype=np.int32)
    inicio_estres = np.full(n_emp, -1, dtype=np.int32)
    dnp_prev1 = np.zeros(n_emp, dtype=bool)
    dnp_prev2 = np.zeros(n_emp, dtype=bool)
    meses_deuda = np.zeros(n_emp, dtype=np.int32)
    reorganizado = np.zeros(n_emp, dtype=bool)

    estado = np.zeros((n_emp, n_per), dtype=np.int8)
    lag = np.full((n_emp, n_per), LAG_INFINITO, dtype=np.int16)
    regulariza = np.full((n_emp, n_per), LAG_INFINITO, dtype=np.int16)
    n_rect = np.zeros((n_emp, n_per), dtype=np.int16)
    deuda_viva = np.zeros((n_emp, n_per), dtype=bool)
    estres_mat = np.zeros((n_emp, n_per), dtype=bool)

    filas_shock, filas_concursal = [], []

    for j in range(n_per):
        mes, des = int(meses[j]), desempleo[j]
        # tras la liquidacion el empleador no declara pero la deuda existe unos meses mas
        en_cola = liquidado & (j - mes_liquidacion <= MESES_SIN_DECLARAR_TRAS_LIQUIDACION)
        vivo = presente[:, j] & ~liquidado
        cola = presente[:, j] & en_cola

        # --- episodios de estres ------------------------------------------------------
        p_entra = np.clip(P_ESTRES_BASE * mult_estres_tam * mult_rubro
                          * (1.0 + ELAST_ESTRES_DESEMPLEO * (des - 0.08)), 0.0, 0.2)
        entra = vivo & ~estres & ~publico & (rng.random(n_emp) < p_entra)
        estres |= entra
        inicio_estres = np.where(entra, j, inicio_estres)
        if entra.any():
            nuevos = np.where(entra)[0]
            reorg = rng.random(len(nuevos)) < P_REORGANIZACION_EN_ESTRES
            for k, i in enumerate(nuevos):
                if reorg[k] and not reorganizado[i]:
                    reorganizado[i] = True
                    filas_concursal.append({
                        "empleador_id": int(emp["empleador_id"].iloc[i]),
                        "tipo": "REORGANIZACION",
                        "periodo_publicacion": periodos[min(j + 1, n_per - 1)]})

        sale = estres & (rng.random(n_emp) < P_SALIDA_ESTRES)
        if sale.any():
            liquida = sale & (rng.random(n_emp) < P_LIQUIDACION_TRAS_ESTRES)
            for i in np.where(sale)[0]:
                filas_shock.append({
                    "empleador_id": int(emp["empleador_id"].iloc[i]),
                    "periodo_inicio": periodos[int(inicio_estres[i])],
                    "periodo_fin": periodos[j],
                    "desenlace": "LIQUIDACION" if liquida[i] else "REGULARIZACION"})
                if liquida[i]:
                    filas_concursal.append({
                        "empleador_id": int(emp["empleador_id"].iloc[i]),
                        "tipo": "LIQUIDACION",
                        "periodo_publicacion": periodos[min(j + 1, n_per - 1)]})
            estres &= ~sale
            mes_liquidacion = np.where(liquida & ~liquidado, j, mes_liquidacion)
            liquidado |= liquida
        estres_mat[:, j] = estres

        # --- probabilidad de DNP del mes ----------------------------------------------
        p_dnp = p_dnp_base * mult_rubro * MULT_DNP_MES.get(mes, 1.0)
        p_dnp = p_dnp * np.where(estres, MULT_DNP_ESTRES, 1.0)
        p_dnp = p_dnp * np.where(dnp_prev1, MULT_DNP_REPITE, 1.0)
        p_dnp = p_dnp * np.where(dnp_prev1 & dnp_prev2, MULT_DNP_REPITE_2M, 1.0)
        p_dnp = np.where(publico, 0.0, np.clip(p_dnp, 0.0, params.tope_p_dnp))

        # --- probabilidad de recepcion tardia / no declaracion ------------------------
        p_tardio = p_tardio_base * np.where(estres, MULT_TARDIO_ESTRES, 1.0)
        p_tardio = np.where(publico, p_tardio_base * 0.3, np.clip(p_tardio, 0.0, params.tope_p_tardio))

        u = rng.random(n_emp)
        es_dnp = vivo & (u < p_dnp)
        u2 = rng.random(n_emp)
        # lag de recepcion: 0 anticipado, 1 oportuno, 2 tardio, 3..8 rezago; la DNP se declara
        # (llega la planilla) con el mismo perfil de rezago pero sin pago
        tarde = vivo & (u2 < p_tardio)
        lag_j = np.where(rng.random(n_emp) < p_antic, 0, 1)
        lag_j = np.where(tarde, 2 + rng.integers(0, 6, n_emp), lag_j)
        # el empleador liquidado o en estres profundo puede simplemente no declarar
        no_declara = vivo & es_dnp & (rng.random(n_emp) < 0.25)

        estado[vivo, j] = COD_PAGADA
        estado[es_dnp, j] = COD_DNP
        estado[no_declara, j] = COD_SIN_DECLARAR
        lag[vivo, j] = lag_j[vivo]
        lag[no_declara, j] = LAG_INFINITO
        # meses posteriores a la liquidacion: planilla no declarada, deuda viva
        estado[cola, j] = COD_SIN_DECLARAR
        lag[cola, j] = LAG_INFINITO

        # regularizacion de la deuda declarada
        idx_dnp = np.where(es_dnp & ~no_declara)[0]
        if len(idx_dnp):
            nunca = rng.random(len(idx_dnp)) < P_NUNCA_REGULARIZA
            extra = rng.integers(*REGULARIZACION_MESES, len(idx_dnp))
            regulariza[idx_dnp, j] = np.where(nunca, LAG_INFINITO, lag[idx_dnp, j] + extra)

        # rectificaciones
        p_r = p_rect * np.where(estres, MULT_RECTIFICACION_ESTRES, 1.0)
        n_rect[vivo, j] = (rng.random(n_emp)[vivo] < p_r[vivo]).astype(np.int16)

        # deuda viva y boletin
        con_deuda = es_dnp | no_declara | cola
        meses_deuda = np.where(con_deuda, meses_deuda + 1, 0)
        deuda_viva[:, j] = con_deuda
        dnp_prev2, dnp_prev1 = dnp_prev1, con_deuda

    # --- tabla planillas -------------------------------------------------------------
    # Un empleador liquidado deja de declarar: sus meses posteriores no generan planilla (el panel
    # laboral, que se genero antes, sigue mostrando a sus trabajadores; es una limitacion conocida
    # de superponer el shock sobre un panel ya simulado y esta documentada en el README).
    ee, pp = np.nonzero(presente & (estado > 0))
    per_arr = np.asarray(periodos)
    lag_v = lag[ee, pp]
    reg_v = regulariza[ee, pp]
    planillas = pd.DataFrame({
        "empleador_id": emp["empleador_id"].to_numpy()[ee],
        "periodo": per_arr[pp],
        "estado": np.array(["", "PAGADA", "DNP", "SIN_DECLARAR"])[estado[ee, pp]],
        "lag_recepcion": lag_v,
        "periodo_recepcion": np.where(lag_v >= LAG_INFINITO, -1,
                                      [add_months(int(per_arr[p]), int(l)) if l < LAG_INFINITO else -1
                                       for p, l in zip(pp, lag_v)]),
        "periodo_regularizacion": np.where(reg_v >= LAG_INFINITO, -1,
                                           [add_months(int(per_arr[p]), int(l)) if l < LAG_INFINITO else -1
                                            for p, l in zip(pp, reg_v)]),
        "n_rectificaciones": n_rect[ee, pp],
    })
    planillas = planillas.merge(contenido, on=["empleador_id", "periodo"], how="left")
    planillas = planillas.sort_values(["periodo", "empleador_id"]).reset_index(drop=True)

    shocks = (pd.DataFrame(filas_shock) if filas_shock else
              pd.DataFrame(columns=["empleador_id", "periodo_inicio", "periodo_fin", "desenlace"]))
    concursal = (pd.DataFrame(filas_concursal) if filas_concursal else
                 pd.DataFrame(columns=["empleador_id", "tipo", "periodo_publicacion"]))
    return planillas, shocks, concursal


def _construir_boletin_infractores(planillas: pd.DataFrame, periodos: list[int]) -> pd.DataFrame:
    """Boletin de Infractores Laborales y Previsionales (DT), empleador x mes.

    Un empleador entra al boletin cuando acumula `MESES_DEUDA_BOLETIN` meses consecutivos con
    deuda previsional (DNP o planilla no declarada); la publicacion tiene `REZAGO_PUBLICACION_BOLETIN`
    mes de rezago y se mantiene `MESES_PERMANENCIA_BOLETIN` meses despues de que la deuda se paga.
    """
    pos = {p: i for i, p in enumerate(periodos)}
    emps = np.sort(planillas["empleador_id"].unique())
    fila = pd.Series(np.arange(len(emps)), index=emps)
    deuda = np.zeros((len(emps), len(periodos)), dtype=bool)
    con_deuda = planillas["estado"].isin(["DNP", "SIN_DECLARAR"])
    sub = planillas.loc[con_deuda]
    deuda[sub["empleador_id"].map(fila).to_numpy(), sub["periodo"].map(pos).to_numpy()] = True

    # meses consecutivos de deuda
    consec = np.zeros_like(deuda, dtype=np.int16)
    for j in range(deuda.shape[1]):
        consec[:, j] = np.where(deuda[:, j], (consec[:, j - 1] if j else 0) + 1, 0)
    en_boletin = np.zeros_like(deuda)
    for j in range(deuda.shape[1]):
        nuevo = consec[:, j] >= MESES_DEUDA_BOLETIN
        desde = min(j + REZAGO_PUBLICACION_BOLETIN, deuda.shape[1] - 1)
        hasta = min(desde + MESES_PERMANENCIA_BOLETIN + 1, deuda.shape[1])
        en_boletin[nuevo, desde:hasta] = True
    ee, pp = np.nonzero(en_boletin)
    return pd.DataFrame({"empleador_id": emps[ee], "periodo": np.asarray(periodos)[pp],
                         "en_boletin": np.int8(1)}).sort_values(
        ["periodo", "empleador_id"]).reset_index(drop=True)


# =========================================================================== generador principal
def generar_tablas(n_afiliados: int = 5_000, periodo_inicio: int = PERIODO_INICIO,
                   periodo_fin: int = PERIODO_FIN, seed: int = 42, perfil: str = PERFIL,
                   params: ParametrosEmpleador | None = None) -> dict[str, pd.DataFrame]:
    """Universo comun + planilla Previred + capa de recepcion/pago del empleador. Determinista."""
    params = params or ParametrosEmpleador()
    rng = np.random.default_rng(seed + 505)
    universo = generar_universo(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                               periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    periodos = rango_periodos(periodo_inicio, periodo_fin)
    empleadores = universo["empleadores"]
    contenido = _contenido_planilla(universo, seed, empleadores, periodos)
    planillas, shocks, concursal = _simular_recepcion(
        contenido, universo["empleadores_latente"], universo["macro"], periodos, rng, params)
    boletin = _construir_boletin_infractores(planillas, periodos)
    return {
        "afiliados": universo["afiliados"],
        "empleadores": empleadores,
        "panel_mensual": universo["panel_mensual"],
        "macro": universo["macro"],
        "planillas": planillas,
        "boletin_infractores": boletin,
        "boletin_concursal": concursal,
        "shocks_empleador": shocks,
    }


def truncar_tablas(tablas: dict[str, pd.DataFrame], periodo_corte: int) -> dict[str, pd.DataFrame]:
    """Copia de `tablas` sin nada que no este disponible en el mes de decision `periodo_corte`.

    * `planillas`: solo las **recibidas** (`0 <= periodo_recepcion <= corte`), y de ellas se borra
      la fecha de regularizacion posterior al corte y el contenido de las no recibidas.
    * `boletin_infractores` / `boletin_concursal`: publicaciones hasta el corte.
    * `panel_mensual`: hasta `corte-2` (desfase de la cotizacion), como en el proyecto 03.
    * `shocks_empleador` se elimina por completo: es latente y no puede alimentar features.
    """
    idx_corte = periodo_to_idx(periodo_corte)
    fuera: dict[str, pd.DataFrame] = {}
    for nombre, df in tablas.items():
        if nombre == "shocks_empleador":
            fuera[nombre] = df.iloc[0:0].copy()
        elif nombre == "planillas":
            g = df[(df["periodo_recepcion"] > 0)
                   & (df["periodo_recepcion"].map(periodo_to_idx) <= idx_corte)].copy()
            reg = g["periodo_regularizacion"]
            g["periodo_regularizacion"] = np.where(
                (reg > 0) & (reg.map(lambda p: periodo_to_idx(p) if p > 0 else 10**9) <= idx_corte),
                reg, -1)
            fuera[nombre] = g.reset_index(drop=True)
        elif nombre == "boletin_concursal":
            fuera[nombre] = df[df["periodo_publicacion"].map(periodo_to_idx) <= idx_corte
                               ].reset_index(drop=True) if len(df) else df.copy()
        elif nombre == "panel_mensual":
            fuera[nombre] = df[df["periodo"].map(periodo_to_idx) <= idx_corte - 2].reset_index(drop=True)
        elif nombre == "macro":
            fuera[nombre] = df.copy()
        elif "periodo" in df.columns:
            fuera[nombre] = df[df["periodo"].map(periodo_to_idx) <= idx_corte].reset_index(drop=True)
        else:
            fuera[nombre] = df.copy()
    return fuera
