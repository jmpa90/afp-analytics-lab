"""Tablas sinteticas propias de `riesgo_cese_cotizacion`, construidas sobre el universo
comun (`afp_lab.common.synth_base.generar_universo`): NO se redefine la dinamica laboral,
solo se derivan del panel mensual las 4 tablas que pide el brief, con el desfase de
informacion real de una AFP.

    cotizaciones_conocidas   afiliado x periodo x empleador: RIM con el desfase de pago
                             (pago normal declarado/recibido en m+1, 5% con rezago adicional
                             de 1 a 6 meses). Misma logica de recepcion que rim_nowcasting
                             (proyecto 01), sin modelar DNP ni rectificaciones (fuera del
                             alcance de este proyecto: aqui solo interesa CUANDO se conoce
                             el monto, no las correcciones posteriores).
    afc_eventos              inicio / termino de la relacion laboral (Seguro de Cesantia),
                             informados por el empleador con ~10 dias de anticipacion
                             (aqui: conocidos en periodo+1). Excluye funcionarios publicos
                             y pensionados activos (no afectos a la Ley 19.728), igual que
                             en rim_nowcasting.
    licencias                dias de licencia medica, conocidos en periodo+1 (aviso COMPIN
                             mas rapido que el pago del subsidio).
    prestaciones_cesantia    giros del seguro de cesantia tras un termino: hasta 5 giros
                             decrecientes (contrato indefinido) o 2 (plazo fijo / por obra),
                             sobre la remuneracion imponible media de los ultimos 12 meses
                             de la relacion terminada; se detienen si el afiliado reingresa
                             antes de agotarlos. Conocidos al mes (sin rezago adicional).

Los parametros (proporciones, tasas) son supuestos razonables para Chile, documentados en
el README del proyecto junto con la justificacion de cada uno.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.common.synth_base import generar_universo

# --------------------------------------------------------------- periodos ----
# Aritmetica de periodos yyyymm vectorizada con numpy (evita llamar, fila a fila,
# a las utilidades escalares de afp_lab.rim_nowcasting.calendario sobre cientos de
# miles de filas). Formula identica a la de ese modulo.
def periodo_to_idx(p):
    p = np.asarray(p)
    return (p // 100 - 2000) * 12 + (p % 100 - 1)


def idx_to_periodo(i):
    i = np.asarray(i)
    return (2000 + i // 12) * 100 + (i % 12 + 1)


def add_months(p, k):
    return idx_to_periodo(periodo_to_idx(p) + k)


# ------------------------------------------------------------- parametros ----
# Desfase de recepcion de las cotizaciones: pago normal declarado y recibido en m+1;
# una fraccion llega con rezago adicional de 1 a 6 meses (multiempleo, atrasos del
# empleador). Mismos ordenes de magnitud que rim_nowcasting (proyecto 01).
P_TARDIO = 0.05
TARDIO_EXTRA_MIN, TARDIO_EXTRA_MAX = 1, 6

# Giros del Seguro de Cesantia (Ley 19.728): tasas legales simplificadas sobre la
# remuneracion imponible promedio de los ultimos 12 meses de la relacion terminada.
# Contrato indefinido: hasta 5 giros decrecientes (aprox. 70/55/45/40/35%); contrato a
# plazo fijo o por obra: hasta 2 giros (aprox. 70/55%, financiados por el Fondo de
# Cesantia Solidario). No son los porcentajes legales exactos (que distinguen aportes
# del propio afiliado vs. del empleador y el Fondo Solidario) sino una aproximacion
# razonable para dar una senal de "cuanto le queda de cobertura".
TASAS_GIRO = {
    "INDEFINIDO": [0.70, 0.55, 0.45, 0.40, 0.35],
    "PLAZO_FIJO": [0.70, 0.55],
}
MAX_GIROS = max(len(v) for v in TASAS_GIRO.values())


def construir_cotizaciones_conocidas(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Movimientos de cotizacion (afiliado x periodo x empleador) con el desfase real
    de recepcion. Grano: una fila por (afiliado, periodo, empleador pagador)."""
    rng = np.random.default_rng(seed + 501)
    p = panel
    m1 = p["rim_emp1"] > 0
    principal = pd.DataFrame({
        "afiliado_id": p.loc[m1, "afiliado_id"].to_numpy(),
        "periodo": p.loc[m1, "periodo"].to_numpy(),
        "empleador_id": p.loc[m1, "empleador_id"].to_numpy(),
        "rim": p.loc[m1, "rim_emp1"].to_numpy(dtype=float),
    })
    m2 = p["empleador2_id"].notna() & (p["rim_emp2"] > 0)
    secundario = pd.DataFrame({
        "afiliado_id": p.loc[m2, "afiliado_id"].to_numpy(),
        "periodo": p.loc[m2, "periodo"].to_numpy(),
        "empleador_id": p.loc[m2, "empleador2_id"].to_numpy(),
        "rim": p.loc[m2, "rim_emp2"].to_numpy(dtype=float),
    })
    mov = pd.concat([principal, secundario], ignore_index=True)
    n = len(mov)
    tardio = rng.random(n) < P_TARDIO
    extra = np.where(tardio, rng.integers(TARDIO_EXTRA_MIN, TARDIO_EXTRA_MAX + 1, n), 0)
    mov["periodo_recepcion"] = add_months(mov["periodo"].to_numpy(), 1 + extra)
    mov["empleador_id"] = mov["empleador_id"].astype("Int64")
    return mov[["afiliado_id", "periodo", "empleador_id", "rim", "periodo_recepcion"]]


def construir_afc_eventos(panel: pd.DataFrame, empleadores: pd.DataFrame, afiliados: pd.DataFrame) -> pd.DataFrame:
    """Eventos de inicio / termino de la relacion laboral, derivados de `evento_afc`."""
    ev = panel.loc[panel["evento_afc"].notna(),
                   ["afiliado_id", "periodo", "empleador_id", "evento_afc", "tipo_contrato"]].copy()
    publicos = set(empleadores.loc[empleadores["rubro"] == "O_ADMIN_PUBLICA", "empleador_id"])
    pensionados = set(afiliados.loc[afiliados["tipo_afiliado"] == "PENSIONADO_ACTIVO", "afiliado_id"])
    ev = ev[~ev["empleador_id"].isin(publicos) & ~ev["afiliado_id"].isin(pensionados)]
    ev = ev.rename(columns={"evento_afc": "tipo_evento"})
    ev["periodo_recepcion"] = add_months(ev["periodo"].to_numpy(), 1)
    return ev.reset_index(drop=True)[
        ["afiliado_id", "periodo", "empleador_id", "tipo_evento", "tipo_contrato", "periodo_recepcion"]]


def construir_licencias(panel: pd.DataFrame) -> pd.DataFrame:
    """Dias de licencia medica por afiliado x mes."""
    lic = panel.loc[panel["dias_licencia"] > 0, ["afiliado_id", "periodo", "dias_licencia"]].copy()
    lic["periodo_recepcion"] = add_months(lic["periodo"].to_numpy(), 1)
    return lic.reset_index(drop=True)


def construir_prestaciones_cesantia(panel: pd.DataFrame, afc_eventos: pd.DataFrame) -> pd.DataFrame:
    """Giros del seguro de cesantia tras cada termino elegible (ver docstring del modulo)."""
    tmp = pd.DataFrame({
        "afiliado_id": panel["afiliado_id"].to_numpy(),
        "periodo": panel["periodo"].to_numpy(),
        "rim_total": panel["rim_emp1"].to_numpy(dtype=float) + panel["rim_emp2"].fillna(0).to_numpy(dtype=float),
    }).sort_values(["afiliado_id", "periodo"]).reset_index(drop=True)
    tmp["base12"] = tmp.groupby("afiliado_id")["rim_total"].transform(
        lambda s: s.rolling(12, min_periods=1).mean())
    rim_lookup = pd.Series(tmp["rim_total"].to_numpy(),
                            index=pd.MultiIndex.from_arrays([tmp["afiliado_id"], tmp["periodo"]]))

    termino = afc_eventos.loc[afc_eventos["tipo_evento"] == "TERMINO",
                              ["afiliado_id", "periodo", "tipo_contrato"]].copy()
    termino = termino[termino["tipo_contrato"].isin(TASAS_GIRO)].reset_index(drop=True)
    termino = termino.merge(tmp[["afiliado_id", "periodo", "base12"]], on=["afiliado_id", "periodo"], how="left")
    n = len(termino)
    if n == 0:
        return pd.DataFrame(columns=["afiliado_id", "periodo", "n_giro", "monto", "periodo_recepcion"])

    af_ids = termino["afiliado_id"].to_numpy()
    periodos0 = termino["periodo"].to_numpy()
    base = termino["base12"].fillna(0.0).to_numpy()
    es_indef = (termino["tipo_contrato"] == "INDEFINIDO").to_numpy()
    max_giros = np.where(es_indef, len(TASAS_GIRO["INDEFINIDO"]), len(TASAS_GIRO["PLAZO_FIJO"]))
    tasas_indef = np.array(TASAS_GIRO["INDEFINIDO"] + [0.0] * (MAX_GIROS - len(TASAS_GIRO["INDEFINIDO"])))
    tasas_plazo = np.array(TASAS_GIRO["PLAZO_FIJO"] + [0.0] * (MAX_GIROS - len(TASAS_GIRO["PLAZO_FIJO"])))

    filas = []
    activo = np.ones(n, dtype=bool)   # aun sin reingresar y dentro del maximo de giros
    for k in range(1, MAX_GIROS + 1):
        periodo_k = add_months(periodos0, k)
        rim_fut = rim_lookup.reindex(pd.MultiIndex.from_arrays([af_ids, periodo_k])).to_numpy()
        rim_fut = np.nan_to_num(rim_fut, nan=0.0)  # sin dato futuro (fuera del rango simulado): se asume aun cesante
        reingreso = rim_fut > 0
        activo &= ~reingreso
        elegible = activo & (k <= max_giros)
        if elegible.any():
            tasa = np.where(es_indef, tasas_indef[k - 1], tasas_plazo[k - 1])
            monto = np.round(base * tasa, 0)
            filas.append(pd.DataFrame({
                "afiliado_id": af_ids[elegible],
                "periodo": periodo_k[elegible],
                "n_giro": k,
                "monto": monto[elegible],
                "periodo_recepcion": periodo_k[elegible],   # conocido al mes: sin rezago adicional
            }))
        activo &= elegible
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(
        columns=["afiliado_id", "periodo", "n_giro", "monto", "periodo_recepcion"])


def generar_datos(n_afiliados: int = 5000, periodo_inicio: int = 202201, periodo_fin: int = 202609,
                  seed: int = 42) -> dict[str, pd.DataFrame]:
    """Universo comun + las 4 tablas propias de este proyecto."""
    universo = generar_universo(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                periodo_fin=periodo_fin, seed=seed)
    panel = universo["panel_mensual"]
    afc = construir_afc_eventos(panel, universo["empleadores"], universo["afiliados"])
    return {
        "afiliados": universo["afiliados"],
        "empleadores": universo["empleadores"],
        "macro": universo["macro"],
        "panel_mensual": panel,
        "cotizaciones_conocidas": construir_cotizaciones_conocidas(panel, seed),
        "afc_eventos": afc,
        "licencias": construir_licencias(panel),
        "prestaciones_cesantia": construir_prestaciones_cesantia(panel, afc),
    }
