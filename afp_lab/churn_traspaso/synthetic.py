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

from dataclasses import dataclass

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
    """Parametros de la simulacion (ver README, seccion 2)."""

    # --- composicion de la cartera propia
    p_antiguo_desde_afiliacion: float = 0.78   # siempre estuvo en la AFP propia
    p_antiguo_traspaso_previo: float = 0.12    # entro antes de la ventana simulada
    # el resto (10 %) entra durante la ventana via traspaso de ENTRADA
    p_entrada_no_licitado: float = 0.0         # un afiliado nuevo al sistema solo puede entrar por licitacion

    # --- saldo inicial
    tasa_mensual_saldo_inicial: float = 0.0035  # rentabilidad real mensual usada para capitalizar la historia
    densidad_cotizacion_historica: float = 0.68
    sigma_saldo_inicial: float = 0.35

    # --- contactos (probabilidad mensual base por afiliado)
    p_vendedor_competencia: float = 0.016
    p_ejecutivo_propio: float = 0.012
    p_call_center: float = 0.030
    p_campana_digital: float = 0.075
    factor_saldo_alto_competencia: float = 2.6   # multiplicador en el decil superior de saldo
    factor_retencion_ejecutivo: float = 3.0      # el ejecutivo reacciona a reclamos / visitas de la competencia
    factor_campana_marzo_agosto: float = 1.45

    # --- reclamos
    p_reclamo_base: float = 0.0040
    factor_reclamo_rentabilidad_mala: float = 2.2
    factor_reclamo_comision_alta: float = 1.6

    # --- uso digital
    lambda_login_base: float = 0.55
    factor_login_joven: float = 1.8
    factor_login_intencion: float = 3.5
    p_certificado_base: float = 0.012
    p_certificado_intencion: float = 0.40

    # --- cambios de fondo
    p_cambio_fondo_base: float = 0.0055
    factor_cambio_fondo_activo: float = 3.0

    # --- hazard mensual de traspaso (escala logit)
    b0: float = -6.20
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
    sigma_afiliado: float = 0.40           # efecto aleatorio NO observable (fija el techo de AUC)
    sigma_mes: float = 0.15                # ruido idiosincratico mensual
    p_retardo_intencion: tuple = (0.10, 0.22, 0.28, 0.25, 0.15)  # meses entre la intencion y la solicitud

    # --- escenario de rentabilidad de la AFP propia
    castigo_rentabilidad_propia: float = -0.010   # -1,0 pp anual
    castigo_desde: int = 202401
    castigo_hasta: int = 202512

    def __post_init__(self) -> None:
        if abs(sum(self.p_retardo_intencion) - 1.0) > 1e-9:
            raise ValueError("p_retardo_intencion debe sumar 1")


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
            e = rng.normal(0, 0.0009, n_t)
            x = np.zeros(n_t)
            for t in range(1, n_t):
                x[t] = 0.92 * x[t - 1] + e[t]
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

    # ---- efectos aleatorios no observables
    u_afil = rng.normal(0, par.sigma_afiliado, n)

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

        p_vc = np.zeros(n)
        p_vc[act] = (par.p_vendedor_competencia * estac
                     * (1.0 + (par.factor_saldo_alto_competencia - 1.0) * saldo_pct[act] ** 3)
                     * (1.0 + 0.5 * (renta_rel[act] > 1.0)))
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

        # --- 4. uso digital (la intencion se manifiesta antes de la solicitud)
        con_int = intencion >= 0
        lam = np.zeros(n)
        lam[act] = (par.lambda_login_base
                    * np.where(joven_login[act], par.factor_login_joven, 1.0)
                    * (1.0 + 0.6 * saldo_pct[act])
                    * np.where(con_int[act], par.factor_login_intencion, 1.0))
        la = rng.poisson(lam)
        lw = rng.poisson(lam * 0.55)
        cs = rng.poisson(lam * 0.8)
        p_cert = np.where(con_int, par.p_certificado_intencion, par.p_certificado_base)
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

    return {
        "afiliados": afiliados,
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


def resumen_simulacion(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Chequeos rapidos de plausibilidad de la simulacion (se imprime y guarda en el run)."""
    tr = tablas["traspasos"]
    salidas = tr[tr["tipo"] == "SALIDA"] if len(tr) else tr
    saldos = tablas["saldos"]
    cartera_mes = saldos.groupby("periodo")["afiliado_id"].count()
    salidas_mes = salidas.groupby("periodo_solicitud")["afiliado_id"].count().reindex(cartera_mes.index).fillna(0)
    tasa_anual = (salidas_mes / cartera_mes).mean() * 12
    filas = [
        {"indicador": "afiliados_en_cartera_promedio", "valor": float(cartera_mes.mean())},
        {"indicador": "tasa_traspaso_anualizada", "valor": float(tasa_anual)},
        {"indicador": "traspasos_salida_total", "valor": float(len(salidas))},
        {"indicador": "traspasos_entrada_total", "valor": float((tr["tipo"] == "ENTRADA").sum()) if len(tr) else 0.0},
        {"indicador": "saldo_mediano_clp", "valor": float(saldos["saldo_obligatorio"].median())},
        {"indicador": "contactos_por_afiliado_ano", "valor": float(len(tablas["contactos"]) / max(cartera_mes.mean(), 1) / (len(cartera_mes) / 12))},
        {"indicador": "reclamos_por_afiliado_ano", "valor": float(len(tablas["reclamos"]) / max(cartera_mes.mean(), 1) / (len(cartera_mes) / 12))},
        {"indicador": "p_descarga_certificado_mes", "valor": float(tablas["uso_digital"]["descargas_certificado"].mean())},
    ]
    return pd.DataFrame(filas)
