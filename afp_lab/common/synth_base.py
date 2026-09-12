"""Universo sintetico COMUN a todos los proyectos del laboratorio (pandas/numpy, sin Spark).

Reutiliza la dinamica mensual del proyecto de nowcasting de RIM (`rim_nowcasting.synthetic`):
afiliados, empleadores y un panel mensual afiliado x periodo con estado laboral, empleador,
sueldo, RIM, licencias y eventos AFC. Cada proyecto parte de este universo y le agrega las
variables propias de su problema (traspasos, APV, etc.) de forma consistente con la historia
laboral simulada.

    from afp_lab.common.synth_base import generar_universo
    u = generar_universo(n_afiliados=5000, periodo_inicio=202201, periodo_fin=202609, seed=42)
    u["afiliados"], u["empleadores"], u["panel_mensual"], u["macro"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.rim_nowcasting.calendario import idx_to_periodo, macro_pandas, periodo_to_idx, rango_periodos
from afp_lab.rim_nowcasting.config import Config
from afp_lab.rim_nowcasting.synthetic import N_RUBROS, _simular_particion, generar_empleadores_pandas

COLUMNAS_LATENTES = ["lat_productividad", "lat_rubro_pref", "lat_p_sep_factor", "lat_p_lic_factor",
                     "lat_multi_empleador", "lat_jornada", "lat_masa_imm"]

# Perfiles de cartera. "sistema" = promedio del sistema de AFP; "cuprum" = AFP Cuprum (ver
# docs/contexto_afp_cuprum.md §4): mayores ingresos, mas educacion universitaria, mas Region
# Metropolitana, rubros de finanzas/mineria/salud/servicios profesionales, mayor edad, menos
# jornada parcial y casi nadie en el ingreso minimo. Los pesos de rubro siguen el orden de RUBROS
# en rim_nowcasting.synthetic (A_AGRICULTURA, B_MINERIA, C_MANUFACTURA, F_CONSTRUCCION, G_COMERCIO,
# H_TRANSPORTE, I_TURISMO, K_FINANZAS, N_SERVICIOS, O_ADMIN_PUBLICA, P_EDUCACION, Q_SALUD, T_CASA_PARTICULAR).
PERFILES = {
    "sistema": {
        "desplaz_productividad": 0.0,
        "p_educacion": [0.15, 0.45, 0.20, 0.20],          # BASICA, MEDIA, TECNICA, UNIVERSITARIA
        "p_region_rm": 0.42, "p_region_v": 0.10, "p_region_viii": 0.10,
        "pesos_rubro": None,                               # uniforme
        "edad_min": 18, "edad_rango": 47, "edad_sesgo": 0.0,
        "p_independiente": 0.08, "p_voluntario": 0.02,
        "p_multi_empleador": 0.05, "p_jornada_parcial": 0.12, "p_masa_imm": 0.15,
        "p_activo_inicial": 0.78, "factor_contratacion": 1.0,
    },
    # Calibrado con la Ficha Estadistica Previsional N 165 (SP, ago 2026): ingreso imponible promedio
    # de cotizantes Cuprum $2.331.030 vs sistema $1.346.859 (1,73x); densidad Cuprum ~67 % (memoria
    # 2025) vs 53,5 % del sistema; 4,5 % de afiliados y 6,3 % de cotizantes del sistema.
    "cuprum": {
        "desplaz_productividad": 0.40,
        "p_educacion": [0.03, 0.27, 0.25, 0.45],
        "p_region_rm": 0.62, "p_region_v": 0.08, "p_region_viii": 0.05,
        "pesos_rubro": [0.02, 0.10, 0.10, 0.06, 0.08, 0.05, 0.02, 0.15, 0.12, 0.10, 0.07, 0.12, 0.01],
        "edad_min": 21, "edad_rango": 43, "edad_sesgo": 0.0,
        "p_independiente": 0.10, "p_voluntario": 0.02,
        "p_multi_empleador": 0.07, "p_jornada_parcial": 0.06, "p_masa_imm": 0.03,
        "p_activo_inicial": 0.68, "factor_contratacion": 0.60,
    },
}


def generar_afiliados_pandas(n: int, periodo_inicio: int, periodo_fin: int, seed: int,
                             perfil: str = "sistema") -> pd.DataFrame:
    """Misma logica que la version Spark de rim_nowcasting.synthetic.generar_afiliados, con perfil de cartera."""
    pf = PERFILES[perfil]
    rng = np.random.default_rng(seed + 7)
    idx_ini, idx_fin = periodo_to_idx(periodo_inicio), periodo_to_idx(periodo_fin)
    u = lambda: rng.random(n)  # noqa: E731
    edad = (pf["edad_min"] + np.floor(np.clip((u() + u()) / 2 + pf["edad_sesgo"], 0, 0.999) * pf["edad_rango"])).astype(int)
    r_reg = u()
    p_rm, p_v, p_viii = pf["p_region_rm"], pf["p_region_v"], pf["p_region_viii"]
    region = np.where(r_reg < p_rm, 13, np.where(r_reg < p_rm + p_v, 5,
                      np.where(r_reg < p_rm + p_v + p_viii, 8, rng.integers(1, 17, n))))
    educ = rng.choice(["BASICA", "MEDIA", "TECNICA", "UNIVERSITARIA"], size=n, p=pf["p_educacion"])
    r_tipo = u()
    p_ind, p_vol = pf["p_independiente"], pf["p_voluntario"]
    tipo = np.select([(edad >= 60) & (r_tipo < 0.5), r_tipo < p_ind, r_tipo < p_ind + p_vol],
                     ["PENSIONADO_ACTIVO", "INDEPENDIENTE", "VOLUNTARIO"], "DEPENDIENTE")
    nuevo = u() < 0.10
    idx_afil = np.where(nuevo, idx_ini + 1 + np.floor(u() * (idx_fin - idx_ini - 1)), idx_ini - np.floor(u() * 240)).astype(int)
    bonus_edu = np.select([educ == "UNIVERSITARIA", educ == "TECNICA", educ == "BASICA"], [0.55, 0.25, -0.20], 0.0)
    if pf["pesos_rubro"] is None:
        rubro_pref = rng.integers(0, N_RUBROS, n)
    else:
        w = np.asarray(pf["pesos_rubro"], float)
        rubro_pref = rng.choice(N_RUBROS, size=n, p=w / w.sum())
    return pd.DataFrame({
        "afiliado_id": np.arange(1, n + 1, dtype=np.int64),
        "sexo": np.where(u() < 0.47, "F", "M"),
        "edad_inicio": edad,
        "region": region.astype(int),
        "nivel_educacional": educ,
        "tipo_afiliado": tipo,
        "periodo_afiliacion": [idx_to_periodo(int(i)) for i in idx_afil],
        "lat_productividad": np.exp(rng.normal(0, 0.45, n) + bonus_edu + pf["desplaz_productividad"]),
        "lat_rubro_pref": rubro_pref,
        "lat_p_sep_factor": np.exp(rng.normal(0, 0.4, n)),
        "lat_p_lic_factor": np.exp(rng.normal(0, 0.5, n)),
        "lat_multi_empleador": (u() < pf["p_multi_empleador"]).astype(int),
        "lat_jornada": np.where(u() < pf["p_jornada_parcial"], 0.5, 1.0),
        "lat_masa_imm": (u() < pf["p_masa_imm"]).astype(int),
    })


def generar_universo(n_afiliados: int = 5000, periodo_inicio: int = 202201, periodo_fin: int = 202609,
                     seed: int = 42, lote: int = 2000, perfil: str = "sistema",
                     afiliados_por_empleador: int = 40) -> dict[str, pd.DataFrame]:
    """Universo sintetico comun. `perfil`: "sistema" (promedio de las AFP) o "cuprum" (nuestra cartera).
    `afiliados_por_empleador` fija el numero de empleadores (n_afiliados / afiliados_por_empleador)."""
    cfg = Config(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio, periodo_snapshot=periodo_fin, seed=seed,
                 afiliados_por_empleador=afiliados_por_empleador)
    empleadores = generar_empleadores_pandas(cfg)
    afiliados = generar_afiliados_pandas(n_afiliados, periodo_inicio, periodo_fin, seed, perfil=perfil)
    macro = macro_pandas(periodo_inicio, periodo_fin)
    periodos = rango_periodos(periodo_inicio, periodo_fin)
    pf = PERFILES[perfil]
    simular = _simular_particion(empleadores, macro, periodos, seed,
                                 p_activo_inicial=pf["p_activo_inicial"],
                                 factor_contratacion=pf["factor_contratacion"])
    lotes = [afiliados.iloc[i:i + lote] for i in range(0, len(afiliados), lote)]
    panel = pd.concat(list(simular(iter(lotes))), ignore_index=True)
    # los parametros latentes de pago del empleador no son observables: se separan
    cols_pago = [c for c in empleadores.columns if c.startswith("p_") or c == "peso_contratacion"]
    return {
        "afiliados": afiliados.drop(columns=COLUMNAS_LATENTES),
        "afiliados_latente": afiliados,
        "empleadores": empleadores.drop(columns=cols_pago),
        "empleadores_latente": empleadores,
        "panel_mensual": panel,
        "macro": macro,
    }
