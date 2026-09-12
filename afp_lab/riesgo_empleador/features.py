"""Features "as-of" del empleador (CONVENCIONES §3).

**Convencion de fechas**: `periodo_corte` es el **mes de la decision** (el mes en que cobranza arma
su lista). Lo conocido en ese mes es:

| Fuente | Regla |
|---|---|
| `planillas` | solo las **recibidas**: `0 < periodo_recepcion <= corte`. La planilla del periodo `corte-1` llega en `corte` si es puntual; una que todavia no llego es informacion en si misma ("faltante") |
| `periodo_regularizacion` de una DNP | solo si es `<= corte` (si no, la deuda sigue viva) |
| `boletin_infractores` / `boletin_concursal` | publicaciones con periodo `<= corte` |
| `panel_mensual` | hasta `corte-2` (desfase de la cotizacion), igual que el proyecto 03 |
| `shocks_empleador` | **nunca**: es latente |

El target es la planilla del periodo `corte`, que se declara en `corte+1`: en el momento de la
decision no existe todavia ninguna informacion sobre ella.

Los montos en CLP se expresan en **unidades de IMM** del mes de corte (sufijo `_imm`): el
entrenamiento queda anos antes del test y la masa imponible nominal crece.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx, rango_periodos

from .synthetic import (COD_DNP, COD_PAGADA, COD_SIN_DECLARAR, HORIZONTE, HORIZONTE_LARGO,
                        LAG_INFINITO)

COLUMNAS_ID = ("empleador_id", "periodo_corte")
COLUMNAS_TARGET = ("y_dnp", "y_tardio", "y_dnp_3m", "masa_en_riesgo", "masa_periodo")
COLUMNAS_ESCALA = ("masa_ref_clp", "imm_corte")

CATEGORICAS = ("rubro", "tamano_empleador", "region_empleador", "politica_gratificacion",
               "freq_reajuste")

TOPE_MESES_DNP = 24
TOPE_MESES_BOLETIN = 24
VENTANA = 12                     # ventana de historia de pago

CORTES_TRAMO_DOTACION = (0, 5, 20, 100, 10**9)
ETIQUETAS_TRAMO_DOTACION = ("1_hasta_4", "2_5_19", "3_20_99", "4_100_mas")


@dataclass
class PanelEmpleador:
    """Matrices densas `empleador x periodo` con toda la historia simulada.

    Construirlas completas no genera leakage: `construir_features` solo lee lo que la mascara
    `recibida` (recepcion <= corte) permite, y el target se calcula aparte.
    """

    periodos: list[int]
    pos: dict[int, int]
    empleador_ids: np.ndarray
    empleadores: pd.DataFrame
    existe: np.ndarray            # la planilla del periodo existe (el empleador declaro o debia)
    estado: np.ndarray            # COD_PAGADA / COD_DNP / COD_SIN_DECLARAR
    recep_idx: np.ndarray         # indice del mes de recepcion (LAG_INFINITO si nunca llega)
    regul_idx: np.ndarray         # indice del mes de regularizacion de la DNP
    lag: np.ndarray
    masa: np.ndarray
    n_trab: np.ndarray
    cotiz: np.ndarray
    n_altas: np.ndarray
    n_bajas: np.ndarray
    n_licencias: np.ndarray
    frac_pf: np.ndarray
    dias_medio: np.ndarray
    n_rect: np.ndarray
    boletin: np.ndarray           # en el boletin de infractores ese mes
    concursal_liq: np.ndarray     # liquidacion publicada en o antes de ese mes (acumulado)
    concursal_reorg: np.ndarray
    imm: np.ndarray
    desempleo: np.ndarray
    ipc: np.ndarray
    imm_var: np.ndarray


def preparar_panel(tablas: dict[str, pd.DataFrame]) -> PanelEmpleador:
    """Convierte las tablas largas en matrices densas empleador x periodo (una sola vez)."""
    empleadores = tablas["empleadores"].sort_values("empleador_id").reset_index(drop=True)
    macro = tablas["macro"]
    periodos = rango_periodos(int(macro["periodo"].min()), int(macro["periodo"].max()))
    pos = {p: i for i, p in enumerate(periodos)}
    ids = empleadores["empleador_id"].to_numpy()
    n_e, n_p = len(ids), len(periodos)
    fila = pd.Series(np.arange(n_e), index=ids)

    pl = tablas["planillas"]
    ie = pl["empleador_id"].map(fila).to_numpy()
    ip = pl["periodo"].map(pos).to_numpy()

    def mat(col, dtype=np.float32, default=0):
        m = np.full((n_e, n_p), default, dtype=dtype)
        m[ie, ip] = pl[col].to_numpy()
        return m

    existe = np.zeros((n_e, n_p), dtype=bool)
    existe[ie, ip] = True
    estado_cod = pl["estado"].map({"PAGADA": COD_PAGADA, "DNP": COD_DNP,
                                   "SIN_DECLARAR": COD_SIN_DECLARAR}).to_numpy()
    estado = np.zeros((n_e, n_p), dtype=np.int8)
    estado[ie, ip] = estado_cod

    lag = np.full((n_e, n_p), LAG_INFINITO, dtype=np.int16)
    lag[ie, ip] = pl["lag_recepcion"].to_numpy()
    recep_idx = np.full((n_e, n_p), LAG_INFINITO, dtype=np.int32)
    fin = np.array([pos[p] for p in pl["periodo"]], dtype=np.int32)
    l = pl["lag_recepcion"].to_numpy()
    recep_idx[ie, ip] = np.where(l >= LAG_INFINITO, LAG_INFINITO, fin + l)
    regul_idx = np.full((n_e, n_p), LAG_INFINITO, dtype=np.int32)
    reg = pl["periodo_regularizacion"].to_numpy()
    tiene_reg = reg > 0
    # una regularizacion posterior al fin de la ventana simulada equivale a "no ocurrio"
    regul_idx[ie[tiene_reg], ip[tiene_reg]] = [pos.get(int(p), LAG_INFINITO)
                                              for p in reg[tiene_reg]]

    bol = tablas["boletin_infractores"]
    boletin = np.zeros((n_e, n_p), dtype=bool)
    if len(bol):
        boletin[bol["empleador_id"].map(fila).to_numpy(), bol["periodo"].map(pos).to_numpy()] = True

    conc = tablas["boletin_concursal"]
    concursal_liq = np.zeros((n_e, n_p), dtype=bool)
    concursal_reorg = np.zeros((n_e, n_p), dtype=bool)
    for _, r in conc.iterrows():
        e = int(fila[r["empleador_id"]])
        j0 = pos[int(r["periodo_publicacion"])]
        (concursal_liq if r["tipo"] == "LIQUIDACION" else concursal_reorg)[e, j0:] = True

    mac = macro.set_index("periodo").loc[periodos]
    return PanelEmpleador(
        periodos=periodos, pos=pos, empleador_ids=ids, empleadores=empleadores,
        existe=existe, estado=estado, recep_idx=recep_idx, regul_idx=regul_idx, lag=lag,
        masa=mat("masa_imponible"), n_trab=mat("n_trabajadores"),
        cotiz=mat("cotizacion_obligatoria"), n_altas=mat("n_altas"), n_bajas=mat("n_bajas"),
        n_licencias=mat("n_licencias"), frac_pf=mat("frac_plazo_fijo"),
        dias_medio=mat("dias_trabajados_medio"), n_rect=mat("n_rectificaciones"),
        boletin=boletin, concursal_liq=concursal_liq, concursal_reorg=concursal_reorg,
        imm=mac["imm"].to_numpy(float), desempleo=mac["desempleo"].to_numpy(float),
        ipc=mac["ipc_12m"].to_numpy(float), imm_var=mac["imm_var_12m"].to_numpy(float))


# ------------------------------------------------------------------ helpers
def _ultimo_con(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(indice de la ultima columna True de cada fila, hubo alguna)."""
    hubo = mask.any(axis=1)
    idx = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    return idx, hubo


def _valor_en(m: np.ndarray, col: np.ndarray, hubo: np.ndarray) -> np.ndarray:
    out = m[np.arange(m.shape[0]), np.clip(col, 0, m.shape[1] - 1)].astype(float)
    return np.where(hubo, out, 0.0)


# ------------------------------------------------------------------ features
def construir_features(tablas: dict[str, pd.DataFrame], periodo_corte: int,
                       denso: PanelEmpleador | None = None,
                       solo_elegibles: bool = True) -> pd.DataFrame:
    """Features as-of del mes de decision (una fila por empleador elegible).

    Poblacion: empleadores con al menos una planilla **recibida** en los 12 meses previos al corte
    y con al menos un trabajador afiliado en esa ultima planilla conocida.
    """
    d = denso or preparar_panel(tablas)
    if periodo_corte not in d.pos:
        raise ValueError(f"periodo_corte {periodo_corte} fuera de la ventana simulada")
    j = d.pos[periodo_corte]
    if j < VENTANA:
        raise ValueError(f"el corte {periodo_corte} no tiene {VENTANA} meses de historia previa")
    imm = d.imm[j]
    n_e = len(d.empleador_ids)
    ini = j - VENTANA                       # ventana [corte-12, corte-1]

    # mascara de lo conocido: planilla existente cuya recepcion ya ocurrio
    recibida = d.existe & (d.recep_idx <= j)
    vent = slice(ini, j)                    # periodos corte-12 .. corte-1
    rec_v = recibida[:, vent]
    est_v = np.where(rec_v, d.estado[:, vent], 0)
    lag_v = np.where(rec_v, d.lag[:, vent], LAG_INFINITO)

    n_recibidas_12 = rec_v.sum(axis=1).astype(float)
    n_faltantes_12 = float(VENTANA) - n_recibidas_12
    puntual_v = rec_v & (d.lag[:, vent] <= 1)
    tasa_punt_12 = puntual_v.sum(axis=1) / np.maximum(n_recibidas_12, 1.0)
    tasa_punt_6 = (puntual_v[:, -6:].sum(axis=1)
                   / np.maximum(rec_v[:, -6:].sum(axis=1), 1.0))
    tasa_punt_3 = (puntual_v[:, -3:].sum(axis=1)
                   / np.maximum(rec_v[:, -3:].sum(axis=1), 1.0))

    deuda_v = rec_v & np.isin(d.estado[:, vent], [COD_DNP, COD_SIN_DECLARAR])
    n_dnp_12 = deuda_v.sum(axis=1).astype(float)
    n_dnp_6 = deuda_v[:, -6:].sum(axis=1).astype(float)
    n_dnp_3 = deuda_v[:, -3:].sum(axis=1).astype(float)
    idx_dnp, hubo_dnp = _ultimo_con(deuda_v)
    meses_desde_dnp = np.where(hubo_dnp, (VENTANA - 1 - idx_dnp).astype(float),
                               float(TOPE_MESES_DNP))
    # racha de puntualidad al final de la ventana
    racha = np.zeros(n_e)
    activa = np.ones(n_e, dtype=bool)
    for k in range(VENTANA - 1, -1, -1):
        ok = puntual_v[:, k] & activa
        racha += ok
        activa &= ok

    # lag medio observado (solo planillas recibidas; 12 = tope). Sin planillas recibidas en la
    # ventana el lag no esta definido: se usa el tope, y `n_recibidas_12` dice que no habia dato.
    lag_cap = np.minimum(d.lag[:, vent], 12)
    lag_medio_12 = ((lag_cap * rec_v).sum(axis=1) / np.maximum(rec_v.sum(axis=1), 1.0))
    lag_medio_12 = np.where(rec_v.any(axis=1), lag_medio_12, 12.0)
    lag_medio_6 = ((lag_cap[:, -6:] * rec_v[:, -6:]).sum(axis=1)
                   / np.maximum(rec_v[:, -6:].sum(axis=1), 1.0))
    lag_medio_6 = np.where(rec_v[:, -6:].any(axis=1), lag_medio_6, 12.0)

    # senales de deterioro reciente (alerta temprana en empleadores grandes, que casi no tienen
    # historia de DNP: lo primero que se mueve es el lag de recepcion y la rotacion)
    lag_max_6 = np.where(rec_v[:, -6:].any(axis=1),
                         np.max(np.where(rec_v[:, -6:], lag_cap[:, -6:], 0), axis=1), 12.0)
    lag_medio_3 = ((lag_cap[:, -3:] * rec_v[:, -3:]).sum(axis=1)
                   / np.maximum(rec_v[:, -3:].sum(axis=1), 1.0))
    lag_medio_3 = np.where(rec_v[:, -3:].any(axis=1), lag_medio_3, 12.0)
    deterioro_lag = lag_medio_3 - lag_medio_12

    # ultima planilla conocida
    idx_ult, hubo_ult = _ultimo_con(rec_v)
    col_ult = ini + idx_ult
    masa_ref = _valor_en(d.masa, col_ult, hubo_ult)
    lag_ult = _valor_en(np.minimum(d.lag, 12).astype(np.float32), col_ult, hubo_ult)
    meses_desde_ult_planilla = np.where(hubo_ult, (VENTANA - 1 - idx_ult).astype(float), float(VENTANA))
    n_trab_ult = _valor_en(d.n_trab, col_ult, hubo_ult)
    frac_pf = _valor_en(d.frac_pf, col_ult, hubo_ult)
    dias_medio = _valor_en(d.dias_medio, col_ult, hubo_ult)

    def ref_hace(k: int) -> tuple[np.ndarray, np.ndarray]:
        """masa y dotacion de la ultima planilla conocida hasta k meses antes del corte."""
        hasta = max(ini, j - k)
        m = recibida[:, ini:hasta] if hasta > ini else np.zeros((n_e, 1), dtype=bool)
        idx, hubo = _ultimo_con(m)
        col = ini + idx
        return _valor_en(d.masa, col, hubo), _valor_en(d.n_trab, col, hubo)

    masa_3, trab_3 = ref_hace(3)
    masa_12, trab_12 = ref_hace(11)

    # deuda viva: DNP recibidas cuya regularizacion no ha ocurrido al corte
    pendiente = (recibida[:, :j] & (d.estado[:, :j] == COD_DNP) & (d.regul_idx[:, :j] > j))
    sin_declarar_hist = (d.existe[:, :j] & (d.estado[:, :j] == COD_SIN_DECLARAR)
                         & (d.recep_idx[:, :j] > j))
    n_dnp_pendientes = pendiente.sum(axis=1).astype(float)
    masa_dnp_pendiente = (d.masa[:, :j] * pendiente).sum(axis=1)
    # meses consecutivos con deuda al cierre de la ventana
    deuda_cierre = deuda_v | (~rec_v)
    meses_deuda_consec = np.zeros(n_e)
    activa = np.ones(n_e, dtype=bool)
    for k in range(VENTANA - 1, -1, -1):
        ok = deuda_cierre[:, k] & activa
        meses_deuda_consec += ok
        activa &= ok

    # boletines (publicados <= corte)
    en_boletin = d.boletin[:, j - 1] if j > 0 else np.zeros(n_e, dtype=bool)
    meses_boletin_12 = d.boletin[:, vent].sum(axis=1).astype(float)
    liq = d.concursal_liq[:, j]
    reorg = d.concursal_reorg[:, j]

    # antiguedad como empleador con trabajadores nuestros
    primera = np.argmax(recibida[:, :j + 1], axis=1)
    tuvo = recibida[:, :j + 1].any(axis=1)
    meses_como_empleador = np.where(tuvo, (j - primera).astype(float), 0.0)

    mes = periodo_corte % 100
    emp = d.empleadores
    out = pd.DataFrame({
        "empleador_id": d.empleador_ids,
        "periodo_corte": np.int32(periodo_corte),
        # escala (no son features)
        "masa_ref_clp": masa_ref,
        "imm_corte": float(imm),
        # --- historia de pago
        "tasa_puntualidad_3": tasa_punt_3,
        "tasa_puntualidad_6": tasa_punt_6,
        "tasa_puntualidad_12": tasa_punt_12,
        "n_dnp_3": n_dnp_3, "n_dnp_6": n_dnp_6, "n_dnp_12": n_dnp_12,
        "meses_desde_ult_dnp": meses_desde_dnp,
        "nunca_dnp_12m": (~hubo_dnp).astype(float),
        "racha_puntualidad": racha,
        "lag_medio_6": lag_medio_6, "lag_medio_12": lag_medio_12,
        "lag_medio_3": lag_medio_3, "lag_max_6": lag_max_6, "deterioro_lag": deterioro_lag,
        "lag_ultima_planilla": lag_ult,
        "meses_desde_ult_planilla": meses_desde_ult_planilla,
        "n_recibidas_12": n_recibidas_12,
        "n_faltantes_12": n_faltantes_12,
        "n_rectificaciones_12": (d.n_rect[:, vent] * rec_v).sum(axis=1),
        # --- deuda viva
        "n_dnp_pendientes": n_dnp_pendientes,
        "masa_dnp_pendiente_imm": masa_dnp_pendiente / imm,
        "meses_deuda_consecutivos": meses_deuda_consec,
        # --- dotacion y masa
        "n_trabajadores": n_trab_ult,
        "var_n_trab_3": n_trab_ult / np.maximum(trab_3, 1.0),
        "var_n_trab_12": n_trab_ult / np.maximum(trab_12, 1.0),
        "masa_ref_imm": masa_ref / imm,
        "masa_por_trabajador_imm": masa_ref / np.maximum(n_trab_ult, 1.0) / imm,
        "var_masa_3": masa_ref / np.maximum(masa_3, 1.0),
        "var_masa_12": masa_ref / np.maximum(masa_12, 1.0),
        "n_altas_3": (d.n_altas[:, j - 3:j] * rec_v[:, -3:]).sum(axis=1),
        "n_bajas_3": (d.n_bajas[:, j - 3:j] * rec_v[:, -3:]).sum(axis=1),
        "n_licencias_3": (d.n_licencias[:, j - 3:j] * rec_v[:, -3:]).sum(axis=1),
        "frac_bajas_3": (d.n_bajas[:, j - 3:j] * rec_v[:, -3:]).sum(axis=1)
                        / np.maximum(n_trab_ult, 1.0),
        "frac_altas_3": (d.n_altas[:, j - 3:j] * rec_v[:, -3:]).sum(axis=1)
                        / np.maximum(n_trab_ult, 1.0),
        "frac_plazo_fijo": frac_pf,
        "dias_trabajados_medio": dias_medio,
        "meses_como_empleador": meses_como_empleador,
        # --- boletines publicos
        "en_boletin_infractores": en_boletin.astype(float),
        "meses_en_boletin_12": meses_boletin_12,
        "concursal_liquidacion": liq.astype(float),
        "concursal_reorganizacion": reorg.astype(float),
        # --- calendario y macro
        "mes_corte": float(mes),
        "es_dic_a_marzo": float(mes in (12, 1, 2, 3)),
        "desempleo": d.desempleo[j],
        "ipc_12m": d.ipc[j],
        "imm_var_12m": d.imm_var[j],
        # --- categoricas del empleador
        "rubro": emp["rubro"].to_numpy(),
        "tamano_empleador": emp["tamano_empleador"].to_numpy(),
        "region_empleador": emp["region_empleador"].astype(str).to_numpy(),
        "politica_gratificacion": emp["politica_gratificacion"].to_numpy(),
        "freq_reajuste": emp["freq_reajuste"].to_numpy(),
    })
    out["elegible"] = (n_recibidas_12 >= 1) & (n_trab_ult >= 1)
    if solo_elegibles:
        out = out[out["elegible"]].drop(columns=["elegible"]).reset_index(drop=True)
    return out


# ------------------------------------------------------------------ target
def construir_target(tablas: dict[str, pd.DataFrame], periodo_corte: int,
                     denso: PanelEmpleador | None = None) -> pd.DataFrame:
    """Target del periodo `corte` (la planilla que se declara en `corte+1`).

    La poblacion del target son los empleadores que **tenian que declarar** ese periodo (existe
    planilla: tenian trabajadores afiliados). Un empleador liquidado sigue en esa poblacion los
    `MESES_SIN_DECLARAR_TRAS_LIQUIDACION` meses en que sus trabajadores siguen en la nomina, con
    planilla `SIN_DECLARAR`; despues sale del universo. Un mes sin planilla es un mes sin
    trabajadores, no una DNP.

    * `y_dnp`   = 1 si la planilla del periodo `corte` es DNP o no se declara.
    * `y_tardio`= 1 si llega con lag >= 2 o no llega.
    * `y_dnp_3m`= 1 si hay alguna DNP / no declaracion en `corte … corte+2`.
    * `masa_periodo` = masa imponible del periodo; `masa_en_riesgo = y_dnp x masa_periodo`.
    """
    d = denso or preparar_panel(tablas)
    j = d.pos[periodo_corte]
    existe = d.existe[:, j]
    estado = d.estado[:, j]
    lag = d.lag[:, j]

    con_deuda = np.isin(estado, [COD_DNP, COD_SIN_DECLARAR])
    fin = min(j + HORIZONTE_LARGO, d.existe.shape[1])
    vent = slice(j, fin)
    deuda_fut = d.existe[:, vent] & np.isin(d.estado[:, vent], [COD_DNP, COD_SIN_DECLARAR])
    return pd.DataFrame({
        "empleador_id": d.empleador_ids,
        "y_dnp": (existe & con_deuda).astype(np.int8),
        "y_tardio": (existe & (lag >= 2)).astype(np.int8),
        "y_dnp_3m": (existe & deuda_fut.any(axis=1)).astype(np.int8),
        "masa_periodo": np.where(existe, d.masa[:, j], 0.0),
        "aplica": existe,
    })


def cortes_disponibles(tablas: dict[str, pd.DataFrame], primer_corte: int,
                       horizonte: int = HORIZONTE_LARGO) -> list[int]:
    """Cortes con el target completo (se exige la ventana larga de 3 meses)."""
    periodos = rango_periodos(int(tablas["macro"]["periodo"].min()),
                              int(tablas["macro"]["periodo"].max()))
    ultimo = add_months(periodos[-1], -(horizonte - 1))
    return [p for p in periodos if primer_corte <= p <= ultimo]


def construir_dataset(tablas: dict[str, pd.DataFrame], cortes: list[int],
                      denso: PanelEmpleador | None = None) -> pd.DataFrame:
    """Apila features + target de varios cortes (una fila por empleador elegible x corte)."""
    d = denso or preparar_panel(tablas)
    partes = []
    for corte in cortes:
        f = construir_features(tablas, corte, denso=d)
        t = construir_target(tablas, corte, denso=d)
        f = f.merge(t, on="empleador_id", how="left")
        f = f[f["aplica"].fillna(False)].drop(columns=["aplica"]).reset_index(drop=True)
        f["masa_en_riesgo"] = f["y_dnp"] * f["masa_periodo"]
        partes.append(f)
    df = pd.concat(partes, ignore_index=True)
    return agregar_segmentos(tipar(df))


def tipar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    excluir = (set(CATEGORICAS) | set(COLUMNAS_ID) | set(COLUMNAS_TARGET)
               | {"tramo_dotacion", "elegible"})
    num = [c for c in df.columns if c not in excluir]
    df[num] = df[num].astype(np.float32)
    return df


def columnas_modelo(df: pd.DataFrame) -> list[str]:
    excluir = (set(COLUMNAS_ID) | set(COLUMNAS_TARGET) | set(COLUMNAS_ESCALA)
               | {"tramo_dotacion", "elegible", "aplica"})
    cols = [c for c in df.columns if c not in excluir]
    prohibidas = [c for c in cols if c.startswith("p_") or c.startswith("lat_")
                  or "estres" in c or "shock" in c]
    if prohibidas:
        raise ValueError(f"columnas latentes usadas como feature: {prohibidas}")
    return cols


def agregar_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tramo_dotacion"] = pd.cut(df["n_trabajadores"], bins=list(CORTES_TRAMO_DOTACION),
                                  labels=list(ETIQUETAS_TRAMO_DOTACION), right=False).astype(str)
    return df


def origenes_moviles(cortes: list[int], n_test: int = 12, n_valid: int = 12,
                     gap: int = HORIZONTE_LARGO, horizonte: int = HORIZONTE_LARGO,
                     cada: int = 3) -> list[dict]:
    """Origenes moviles con gap entre cada par de particiones (CONVENCIONES §3).

    El gap se dimensiona con la ventana de target **mas larga** que se reporta (3 meses), no con
    la del target principal (1 mes), para que las tres variantes queden limpias con un solo split.
    """
    salto = gap + horizonte
    test = cortes[-n_test:]
    bloques = [test[i:i + cada] for i in range(0, len(test), cada)]
    origenes = []
    for bloque in bloques:
        corte_max_valid = add_months(bloque[0], -salto)
        candidatos = [c for c in cortes if c <= corte_max_valid]
        if len(candidatos) <= n_valid:
            raise ValueError(f"no hay cortes suficientes para validar el origen {bloque[0]}")
        valid = candidatos[-n_valid:]
        train = [c for c in cortes if c <= add_months(valid[0], -salto)]
        if not train:
            raise ValueError(f"no hay cortes suficientes para entrenar el origen {bloque[0]}")
        origenes.append({"origen": bloque[0], "train": train, "valid": valid, "test": bloque})
    return origenes


def solape_valid_pool_test(origenes: list[dict], horizonte: int = HORIZONTE_LARGO) -> pd.DataFrame:
    """Meses de target que el `valid` de cada origen comparte con el pool de test (ver README)."""
    def meses(corte):
        return [add_months(corte, k) for k in range(0, horizonte)]

    pool = {m for o in origenes for c in o["test"] for m in meses(c)}
    filas = []
    for o in origenes:
        mv = {m for c in o["valid"] for m in meses(c)}
        propio = {m for c in o["test"] for m in meses(c)}
        comun = sorted(mv & pool)
        filas.append({"origen": o["origen"], "meses_valid": f"{min(mv)}-{max(mv)}",
                      "solape_con_su_propio_test": len(mv & propio),
                      "solape_con_el_pool_de_test": len(comun),
                      "meses_solapados": ",".join(str(m) for m in comun) if comun else ""})
    return pd.DataFrame(filas)
