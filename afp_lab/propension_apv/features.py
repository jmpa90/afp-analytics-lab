"""Features "as-of" para la propension a ahorro voluntario (CONVENCIONES §3).

Regla unica: **una fila se construye en `periodo_corte` con informacion cuyo periodo es
<= `periodo_corte`**. El `periodo_corte` ya incorpora el desfase de 2 meses de la AFP
(la cotizacion del mes m se conoce en m+2), de modo que en produccion
`periodo_corte = hoy - 2`. El target ocurre en `corte+1 .. corte+6` y nunca se mira.

Para que el calculo sea barato con decenas de cortes, las tablas largas se convierten una
sola vez en matrices densas `afiliado x periodo` (`preparar_panel`); cada corte es entonces
un corte de columnas de esas matrices. `construir_features(tablas, periodo_corte)` funciona
igual sin pasar el panel precalculado (lo construye internamente).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx, rango_periodos

from .synthetic import (ESTADOS, ESTADOS_INV, HORIZONTE, MESES_DESFASE, indexar_periodos,
                        matriz_densa, tramo_igc)

# columnas que nunca entran al modelo
COLUMNAS_ID = ("afiliado_id", "periodo_corte")
COLUMNAS_TARGET = ("y", "monto")

CATEGORICAS = ("tipo_afiliado", "sexo", "nivel_educacional", "rubro", "tamano_empleador",
               "estado_corte", "region")

# tramos usados para reportar por segmento (sueldos brutos mensuales en CLP)
CORTES_TRAMO_RENTA = (0, 600_000, 1_000_000, 1_800_000, 3_000_000, np.inf)
ETIQUETAS_TRAMO_RENTA = ("1_hasta_600k", "2_600k_1M", "3_1M_1.8M", "4_1.8M_3M", "5_sobre_3M")
CORTES_TRAMO_EDAD = (0, 30, 40, 50, 60, 200)
ETIQUETAS_TRAMO_EDAD = ("1_menos_30", "2_30_39", "3_40_49", "4_50_59", "5_60_mas")


@dataclass
class PanelDenso:
    """Matrices `afiliado x periodo` con TODA la historia simulada.

    No hay leakage por construir estas matrices completas: `construir_features` solo lee
    columnas con indice <= `periodo_corte`.
    """

    periodos: list[int]
    pos: dict[int, int]
    afiliado_ids: np.ndarray
    afiliados: pd.DataFrame
    empleadores: pd.DataFrame
    tributario: pd.DataFrame
    rim: np.ndarray
    cotiza: np.ndarray
    estado: np.ndarray
    antig: np.ndarray
    edad: np.ndarray
    dias_lic: np.ndarray
    emp_idx: np.ndarray
    aporte_monto: np.ndarray
    aporte_flag: np.ndarray
    aporte_mensual: np.ndarray
    camp_total: np.ndarray
    camp_canal: dict[str, np.ndarray]
    camp_abrio: np.ndarray
    saldo_obl: np.ndarray
    saldo_vol: np.ndarray
    apvc_ofrece: np.ndarray
    apvc_match: np.ndarray
    imm: np.ndarray
    uf: np.ndarray


def preparar_panel(tablas: dict[str, pd.DataFrame]) -> PanelDenso:
    """Convierte las tablas largas en matrices densas afiliado x periodo (una sola vez)."""
    afiliados = tablas["afiliados"].sort_values("afiliado_id").reset_index(drop=True)
    empleadores = tablas["empleadores"].sort_values("empleador_id").reset_index(drop=True)
    panel = tablas["panel_mensual"]
    macro = tablas["macro"]

    periodos = rango_periodos(int(macro["periodo"].min()), int(macro["periodo"].max()))
    pos = indexar_periodos(periodos)
    n_af, n_per, n_emp = len(afiliados), len(periodos), len(empleadores)
    ids = afiliados["afiliado_id"].to_numpy()
    fila_af = pd.Series(np.arange(n_af), index=ids)

    def idx(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (df["afiliado_id"].map(fila_af).to_numpy(),
                df["periodo"].map(pos).to_numpy())

    ia, ip = idx(panel)
    rim = matriz_densa(panel.assign(_r=panel["rim_emp1"] + panel["rim_emp2"]), "_r", n_af, n_per, ia, ip)
    estado = matriz_densa(panel.assign(_e=panel["estado"].map(ESTADOS)), "_e", n_af, n_per, ia, ip,
                          dtype=np.int8)
    antig = matriz_densa(panel, "meses_antiguedad", n_af, n_per, ia, ip, dtype=np.int16)
    edad = matriz_densa(panel, "edad", n_af, n_per, ia, ip, dtype=np.int16)
    dias_lic = matriz_densa(panel, "dias_licencia", n_af, n_per, ia, ip, dtype=np.int16)
    emp_idx = matriz_densa(panel.assign(_e=panel["empleador_id"].fillna(0).astype("int64")), "_e",
                           n_af, n_per, ia, ip, dtype=np.int64, default=0) - 1

    ap = tablas["aportes_voluntarios"]
    aporte_monto = np.zeros((n_af, n_per), dtype=np.float32)
    aporte_flag = np.zeros((n_af, n_per), dtype=np.float32)
    aporte_mensual = np.zeros((n_af, n_per), dtype=np.float32)
    if len(ap):
        ia, ip = idx(ap)
        np.add.at(aporte_monto, (ia, ip), ap["monto"].to_numpy(np.float32))
        aporte_flag[ia, ip] = 1.0
        aporte_mensual[ia[ap["periodicidad"].to_numpy() == "MENSUAL"],
                       ip[ap["periodicidad"].to_numpy() == "MENSUAL"]] = 1.0

    camp = tablas["campanas"]
    camp_total = np.zeros((n_af, n_per), dtype=np.float32)
    camp_abrio = np.zeros((n_af, n_per), dtype=np.float32)
    camp_canal = {c: np.zeros((n_af, n_per), dtype=np.float32)
                  for c in ("EMAIL", "APP", "EJECUTIVO", "ASESOR")}
    if len(camp):
        ia, ip = idx(camp)
        np.add.at(camp_total, (ia, ip), 1.0)
        abrio = (camp["respuesta"].to_numpy() == "ABRIO").astype(np.float32)
        np.add.at(camp_abrio, (ia, ip), abrio)
        canal = camp["canal"].to_numpy()
        for c, m in camp_canal.items():
            sel = canal == c
            np.add.at(m, (ia[sel], ip[sel]), 1.0)

    sal = tablas["saldos"]
    ia, ip = idx(sal)
    saldo_obl = matriz_densa(sal, "saldo_obligatorio", n_af, n_per, ia, ip)
    saldo_vol = matriz_densa(sal, "saldo_voluntario", n_af, n_per, ia, ip)

    apvc = tablas["apvc_empleador"]
    fila_emp = pd.Series(np.arange(n_emp), index=empleadores["empleador_id"].to_numpy())
    ie = apvc["empleador_id"].map(fila_emp).to_numpy()
    ipe = apvc["periodo"].map(pos).to_numpy()
    apvc_ofrece = np.zeros((n_emp, n_per), dtype=np.float32)
    apvc_match = np.zeros((n_emp, n_per), dtype=np.float32)
    apvc_ofrece[ie, ipe] = apvc["ofrece_apvc"].to_numpy(np.float32)
    apvc_match[ie, ipe] = apvc["pct_match"].to_numpy(np.float32)

    mac = macro.set_index("periodo").loc[periodos]
    return PanelDenso(
        periodos=periodos, pos=pos, afiliado_ids=ids, afiliados=afiliados, empleadores=empleadores,
        tributario=tablas["tributario"], rim=rim, cotiza=(rim > 0).astype(np.float32), estado=estado,
        antig=antig, edad=edad, dias_lic=dias_lic, emp_idx=emp_idx, aporte_monto=aporte_monto,
        aporte_flag=aporte_flag, aporte_mensual=aporte_mensual, camp_total=camp_total,
        camp_canal=camp_canal, camp_abrio=camp_abrio, saldo_obl=saldo_obl, saldo_vol=saldo_vol,
        apvc_ofrece=apvc_ofrece, apvc_match=apvc_match,
        imm=mac["imm"].to_numpy(float), uf=mac["uf"].to_numpy(float))


# ------------------------------------------------------------------ helpers de ventana
def _ventana(m: np.ndarray, j: int, k: int) -> np.ndarray:
    """Ultimas `k` columnas terminando en `j` (inclusive)."""
    return m[:, max(0, j - k + 1): j + 1]


def _suma(m: np.ndarray, j: int, k: int) -> np.ndarray:
    """Suma de las ultimas `k` columnas terminando en `j` (inclusive)."""
    return _ventana(m, j, k).sum(axis=1, dtype=np.float64)


def _meses_desde(m: np.ndarray, j: int, ventana: int, sin_dato: float = 99.0) -> np.ndarray:
    """Meses transcurridos desde la ultima columna con valor > 0 dentro de `ventana`."""
    ini = max(0, j - ventana + 1)
    v = m[:, ini: j + 1] > 0
    largo = v.shape[1]
    ultimo = largo - 1 - np.argmax(v[:, ::-1], axis=1)
    return np.where(v.any(axis=1), (largo - 1 - ultimo).astype(float), sin_dato)


def _meses_a_marzo(mes_corte: int) -> int:
    """Meses desde el corte hasta el proximo marzo (0 si el corte es marzo)."""
    return (3 - mes_corte) % 12


# ------------------------------------------------------------------ features
def construir_features(tablas: dict[str, pd.DataFrame], periodo_corte: int,
                       denso: PanelDenso | None = None, solo_elegibles: bool = True) -> pd.DataFrame:
    """Features as-of de un `periodo_corte` (una fila por afiliado elegible).

    Poblacion (definida solo con informacion <= corte):

    * afiliado vigente (con registro en el panel del mes de corte y no RETIRADO),
    * que cotizo al menos una vez en los 12 meses previos al corte,
    * **sin aporte voluntario en los 6 meses previos** (problema de adquisicion/reactivacion;
      el mantenimiento de los aportantes mensuales es otro problema).
    """
    d = denso or preparar_panel(tablas)
    if periodo_corte not in d.pos:
        raise ValueError(f"periodo_corte {periodo_corte} fuera de la ventana simulada")
    j = d.pos[periodo_corte]
    anio, mes = periodo_corte // 100, periodo_corte % 100
    imm, uf = d.imm[j], d.uf[j]
    n_af = len(d.afiliado_ids)

    # --- renta y estabilidad (as-of, con el desfase de 2 meses ya aplicado) -----------
    rim_corte = d.rim[:, j].astype(float)
    n_cot_12 = _suma(d.cotiza, j, 12)
    n_cot_6 = _suma(d.cotiza, j, 6)
    n_cot_3 = _suma(d.cotiza, j, 3)
    suma_12 = _suma(d.rim, j, 12)
    suma_3 = _suma(d.rim, j, 3)
    media_12 = suma_12 / np.maximum(n_cot_12, 1.0)
    media_3 = suma_3 / np.maximum(n_cot_3, 1.0)
    sumsq_12 = (_ventana(d.rim, j, 12).astype(np.float64) ** 2).sum(axis=1)
    var_12 = np.maximum(sumsq_12 / np.maximum(n_cot_12, 1.0) - media_12 ** 2, 0.0)
    rim_ref = np.where(rim_corte > 0, rim_corte, media_12)

    emp = d.emp_idx[:, j]
    tiene_emp = emp >= 0
    emp_seguro = np.clip(emp, 0, None)
    ventana_emp = d.emp_idx[:, max(0, j - 11): j + 1]
    cambios = (ventana_emp[:, 1:] != ventana_emp[:, :-1]) & (ventana_emp[:, 1:] >= 0)
    n_empleadores_12 = cambios.sum(axis=1) + (ventana_emp[:, 0] >= 0)

    # --- tributario ------------------------------------------------------------------
    tramo_12m = tramo_igc(suma_12, anio)
    # la Operacion Renta del anio Y llega en abril de Y+1: al corte solo se conoce Y-1 (si mes>=4) o Y-2
    anio_disp = anio - 1 if mes >= 4 else anio - 2
    trib = d.tributario[d.tributario["anio"] == anio_disp]
    mapa_tramo = pd.Series(trib["tramo_igc"].to_numpy(), index=trib["afiliado_id"].to_numpy())
    tramo_declarado = pd.Series(d.afiliado_ids).map(mapa_tramo).fillna(-1).to_numpy(float)

    # --- empleador -------------------------------------------------------------------
    ofrece_apvc = np.where(tiene_emp, d.apvc_ofrece[emp_seguro, j], 0.0)
    pct_match = np.where(tiene_emp, d.apvc_match[emp_seguro, j], 0.0)
    meses_con_apvc = np.zeros(n_af)
    if j > 0:
        hist = d.apvc_ofrece[:, max(0, j - 35): j + 1]
        acum = hist[:, ::-1].cumprod(axis=1).sum(axis=1)   # meses consecutivos ofreciendo, hasta 36
        meses_con_apvc = np.where(tiene_emp, acum[emp_seguro], 0.0)

    aporto_12m = (_suma(d.aporte_flag, j, 12) > 0).astype(float)
    df_emp = pd.DataFrame({"emp": np.where(tiene_emp, emp, -1), "rim": rim_corte, "apv": aporto_12m})
    agg = df_emp[df_emp["emp"] >= 0].groupby("emp").agg(
        emp_n_trabajadores=("rim", "size"), emp_rim_mediana=("rim", "median"),
        frac_apv_empleador=("apv", "mean"))
    emp_n = np.where(tiene_emp, pd.Series(np.where(tiene_emp, emp, -1)).map(
        agg["emp_n_trabajadores"]).fillna(0).to_numpy(), 0.0)
    emp_mediana = np.where(tiene_emp, pd.Series(np.where(tiene_emp, emp, -1)).map(
        agg["emp_rim_mediana"]).fillna(0).to_numpy(), 0.0)
    frac_apv_emp = np.where(tiene_emp, pd.Series(np.where(tiene_emp, emp, -1)).map(
        agg["frac_apv_empleador"]).fillna(0).to_numpy(), 0.0)

    emp_attr = d.empleadores
    rubro = np.where(tiene_emp, emp_attr["rubro"].to_numpy()[emp_seguro], "SIN_EMPLEADOR")
    tamano = np.where(tiene_emp, emp_attr["tamano_empleador"].to_numpy()[emp_seguro], "SIN_EMPLEADOR")

    # --- cuenta y aportes ------------------------------------------------------------
    n_ap_6 = _suma(d.aporte_flag, j, 6)
    n_ap_12 = _suma(d.aporte_flag, j, 12)
    n_ap_24 = _suma(d.aporte_flag, j, 24)
    monto_ap_24 = _suma(d.aporte_monto, j, 24)
    mensual_24 = _suma(d.aporte_mensual, j, 24)
    meses_desde_aporte = _meses_desde(d.aporte_flag, j, 36)

    # --- campanas --------------------------------------------------------------------
    camp_1 = _suma(d.camp_total, j, 1)
    camp_3 = _suma(d.camp_total, j, 3)
    camp_6 = _suma(d.camp_total, j, 6)
    abrio_6 = _suma(d.camp_abrio, j, 6)
    canal_6 = {c: _suma(m, j, 6) for c, m in d.camp_canal.items()}
    meses_desde_contacto = _meses_desde(d.camp_total, j, 24)

    # --- perfil ----------------------------------------------------------------------
    af = d.afiliados
    edad = d.edad[:, j].astype(float)
    sexo = af["sexo"].to_numpy()
    edad_pension = np.where(sexo == "F", 60, 65)
    meses_desde_afiliacion = np.array(
        [max(0, periodo_to_idx(periodo_corte) - periodo_to_idx(int(p))) for p in af["periodo_afiliacion"]],
        dtype=float)

    estado_corte = pd.Series(d.estado[:, j]).map(ESTADOS_INV).to_numpy()

    out = pd.DataFrame({
        "afiliado_id": d.afiliado_ids,
        "periodo_corte": np.int32(periodo_corte),
        # renta y estabilidad
        "rim_corte": rim_corte,
        "rim_ref": rim_ref,
        "rim_media_12": media_12,
        "rim_media_3": media_3,
        "rim_cv_12": np.sqrt(var_12) / np.maximum(media_12, 1.0),
        "rim_tendencia_3_12": media_3 / np.maximum(media_12, 1.0),
        "rim_sobre_imm": rim_ref / imm,
        "cerca_minimo": (rim_ref < 1.15 * imm).astype(float),
        "densidad_12": n_cot_12 / 12.0,
        "densidad_6": n_cot_6 / 6.0,
        "meses_desde_ult_cotizacion": _meses_desde(d.cotiza, j, 24),
        "meses_con_empleador": d.antig[:, j].astype(float),
        "n_empleadores_12": n_empleadores_12.astype(float),
        "meses_cesante_12": (_ventana(d.estado, j, 12) == ESTADOS["CESANTE"]).sum(axis=1).astype(float),
        "dias_licencia_12": _ventana(d.dias_lic, j, 12).sum(axis=1, dtype=np.float64),
        "licencia_larga_reciente": (_ventana(d.dias_lic, j, 3) >= 20).any(axis=1).astype(float),
        # tributario
        "tramo_igc_12m": tramo_12m.astype(float),
        "tramo_igc_declarado": tramo_declarado,
        "renta_anual_12m": suma_12,
        # calendario
        "mes_corte": float(mes),
        "es_marzo": float(mes == 3),
        "meses_a_marzo": float(_meses_a_marzo(mes)),
        "marzo_en_ventana": float(1 <= _meses_a_marzo(mes) <= HORIZONTE),
        "diciembre_en_ventana": float(1 <= ((12 - mes) % 12) <= HORIZONTE),
        # cuenta
        "saldo_obligatorio": d.saldo_obl[:, j].astype(float),
        "saldo_voluntario": d.saldo_vol[:, j].astype(float),
        "saldo_vol_sobre_rim": d.saldo_vol[:, j] / np.maximum(rim_ref, 1.0),
        "saldo_obl_sobre_rim": d.saldo_obl[:, j] / np.maximum(rim_ref, 1.0),
        "meses_desde_ult_aporte": meses_desde_aporte,
        "n_aportes_12m": n_ap_12,
        "n_aportes_24m": n_ap_24,
        "monto_aportes_24m": monto_ap_24,
        "tuvo_aporte_mensual_24m": (mensual_24 > 0).astype(float),
        # campanas
        "contactos_1m": camp_1,
        "contactos_3m": camp_3,
        "contactos_6m": camp_6,
        "contactos_6m_email": canal_6["EMAIL"],
        "contactos_6m_app": canal_6["APP"],
        "contactos_6m_ejecutivo": canal_6["EJECUTIVO"],
        "contactos_6m_asesor": canal_6["ASESOR"],
        "aperturas_6m": abrio_6,
        "tasa_apertura_6m": abrio_6 / np.maximum(camp_6, 1.0),
        "meses_desde_ult_contacto": meses_desde_contacto,
        # empleador
        "ofrece_apvc": ofrece_apvc.astype(float),
        "pct_match_apvc": pct_match.astype(float),
        "meses_con_apvc": meses_con_apvc,
        "frac_apv_empleador": frac_apv_emp,
        "emp_n_trabajadores": emp_n,
        "emp_rim_mediana": emp_mediana,
        "pos_rel_empleador": rim_corte / np.maximum(emp_mediana, 1.0),
        # perfil
        "edad": edad,
        "meses_hasta_pension": np.maximum(edad_pension - edad, 0) * 12.0,
        "meses_desde_afiliacion": meses_desde_afiliacion,
        # categoricas
        "tipo_afiliado": af["tipo_afiliado"].to_numpy(),
        "sexo": sexo,
        "nivel_educacional": af["nivel_educacional"].to_numpy(),
        "region": af["region"].astype(str).to_numpy(),
        "rubro": rubro,
        "tamano_empleador": tamano,
        "estado_corte": estado_corte,
    })

    vigente = (d.estado[:, j] > 0) & (d.estado[:, j] != ESTADOS["RETIRADO"])
    out["elegible"] = (vigente & (n_cot_12 >= 1) & (n_ap_6 == 0)).astype(bool)
    if solo_elegibles:
        out = out[out["elegible"]].drop(columns=["elegible"]).reset_index(drop=True)
    return out


# ------------------------------------------------------------------ target
def construir_target(tablas: dict[str, pd.DataFrame], periodo_corte: int,
                     horizonte: int = HORIZONTE) -> pd.DataFrame:
    """`y` = 1 si hay al menos un aporte voluntario en `corte+1 .. corte+horizonte`.

    `monto` = suma de esos aportes. Solo se llama para cortes cuyos `horizonte` meses estan
    completos en los datos.
    """
    meses = [add_months(periodo_corte, k) for k in range(1, horizonte + 1)]
    ap = tablas["aportes_voluntarios"]
    vent = ap[ap["periodo"].isin(meses)]
    agg = vent.groupby("afiliado_id", as_index=False)["monto"].sum()
    agg["y"] = 1
    return agg.rename(columns={"monto": "monto"})[["afiliado_id", "y", "monto"]]


def cortes_disponibles(tablas: dict[str, pd.DataFrame], primer_corte: int,
                       horizonte: int = HORIZONTE) -> list[int]:
    """Cortes con los `horizonte` meses de target completos dentro de la ventana simulada."""
    periodos = rango_periodos(int(tablas["macro"]["periodo"].min()), int(tablas["macro"]["periodo"].max()))
    ultimo = add_months(periodos[-1], -horizonte)
    return [p for p in periodos if primer_corte <= p <= ultimo]


def construir_dataset(tablas: dict[str, pd.DataFrame], cortes: list[int],
                      horizonte: int = HORIZONTE, denso: PanelDenso | None = None) -> pd.DataFrame:
    """Apila features + target de varios cortes (una fila por afiliado elegible x corte)."""
    d = denso or preparar_panel(tablas)
    partes = []
    for corte in cortes:
        f = construir_features(tablas, corte, denso=d)
        t = construir_target(tablas, corte, horizonte)
        f = f.merge(t, on="afiliado_id", how="left")
        f["y"] = f["y"].fillna(0).astype(np.int8)
        f["monto"] = f["monto"].fillna(0.0).astype(float)
        partes.append(f)
    df = pd.concat(partes, ignore_index=True)
    return agregar_segmentos(tipar(df))


def tipar(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricas como `category` con el mismo dominio en train/valid/test; el resto float32."""
    df = df.copy()
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    num = [c for c in df.columns if c not in CATEGORICAS + COLUMNAS_ID + COLUMNAS_TARGET
           and c not in ("tramo_renta", "tramo_edad")]
    df[num] = df[num].astype(np.float32)
    return df


def columnas_modelo(df: pd.DataFrame) -> list[str]:
    excluir = set(COLUMNAS_ID) | set(COLUMNAS_TARGET) | {"tramo_renta", "tramo_edad", "elegible"}
    cols = [c for c in df.columns if c not in excluir]
    prohibidas = [c for c in cols if c.startswith("lat_") or c.startswith("p_")]
    if prohibidas:
        raise ValueError(f"columnas latentes usadas como feature: {prohibidas}")
    return cols


def agregar_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Segmentos de reporte (§4 de CONVENCIONES): tramo de renta y tramo de edad."""
    df = df.copy()
    df["tramo_renta"] = pd.cut(df["rim_ref"], bins=list(CORTES_TRAMO_RENTA),
                               labels=list(ETIQUETAS_TRAMO_RENTA), right=False).astype(str)
    df["tramo_edad"] = pd.cut(df["edad"], bins=list(CORTES_TRAMO_EDAD),
                              labels=list(ETIQUETAS_TRAMO_EDAD), right=False).astype(str)
    return df


def split_temporal(cortes: list[int], n_test: int = 3, n_valid: int = 3,
                   gap: int = HORIZONTE, horizonte: int = HORIZONTE) -> dict[str, list[int]]:
    """Split temporal con gap (CONVENCIONES §3).

    `test` son los `n_test` cortes mas recientes. El **ultimo target de entrenamiento**
    (`corte + horizonte`) queda al menos `gap` meses antes del primer corte de test, de modo
    que ningun mes usado para etiquetar el entrenamiento se solapa con la ventana de test.
    """
    test = cortes[-n_test:]
    corte_max_train = add_months(test[0], -(gap + horizonte))
    previos = [c for c in cortes if c <= corte_max_train]
    if len(previos) <= n_valid:
        raise ValueError("no hay cortes suficientes para entrenar y validar con este gap")
    return {"train": previos[:-n_valid], "valid": previos[-n_valid:], "test": test,
            "descartados": [c for c in cortes if c > corte_max_train and c not in test]}
