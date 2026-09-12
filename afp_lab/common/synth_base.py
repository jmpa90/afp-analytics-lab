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


def generar_afiliados_pandas(n: int, periodo_inicio: int, periodo_fin: int, seed: int) -> pd.DataFrame:
    """Misma logica que la version Spark de rim_nowcasting.synthetic.generar_afiliados."""
    rng = np.random.default_rng(seed + 7)
    idx_ini, idx_fin = periodo_to_idx(periodo_inicio), periodo_to_idx(periodo_fin)
    u = lambda: rng.random(n)  # noqa: E731
    edad = (18 + np.floor((u() + u()) / 2 * 47)).astype(int)
    r_reg = u()
    region = np.where(r_reg < 0.42, 13, np.where(r_reg < 0.52, 5, np.where(r_reg < 0.62, 8, rng.integers(1, 17, n))))
    r_edu = u()
    educ = np.select([r_edu < 0.15, r_edu < 0.60, r_edu < 0.80], ["BASICA", "MEDIA", "TECNICA"], "UNIVERSITARIA")
    r_tipo = u()
    tipo = np.select([(edad >= 60) & (r_tipo < 0.5), r_tipo < 0.08, r_tipo < 0.10],
                     ["PENSIONADO_ACTIVO", "INDEPENDIENTE", "VOLUNTARIO"], "DEPENDIENTE")
    nuevo = u() < 0.10
    idx_afil = np.where(nuevo, idx_ini + 1 + np.floor(u() * (idx_fin - idx_ini - 1)), idx_ini - np.floor(u() * 240)).astype(int)
    bonus_edu = np.select([educ == "UNIVERSITARIA", educ == "TECNICA", educ == "BASICA"], [0.55, 0.25, -0.20], 0.0)
    return pd.DataFrame({
        "afiliado_id": np.arange(1, n + 1, dtype=np.int64),
        "sexo": np.where(u() < 0.47, "F", "M"),
        "edad_inicio": edad,
        "region": region.astype(int),
        "nivel_educacional": educ,
        "tipo_afiliado": tipo,
        "periodo_afiliacion": [idx_to_periodo(int(i)) for i in idx_afil],
        "lat_productividad": np.exp(rng.normal(0, 0.45, n) + bonus_edu),
        "lat_rubro_pref": rng.integers(0, N_RUBROS, n),
        "lat_p_sep_factor": np.exp(rng.normal(0, 0.4, n)),
        "lat_p_lic_factor": np.exp(rng.normal(0, 0.5, n)),
        "lat_multi_empleador": (u() < 0.05).astype(int),
        "lat_jornada": np.where(u() < 0.12, 0.5, 1.0),
        "lat_masa_imm": (u() < 0.15).astype(int),
    })


def generar_universo(n_afiliados: int = 5000, periodo_inicio: int = 202201, periodo_fin: int = 202609,
                     seed: int = 42, lote: int = 2000) -> dict[str, pd.DataFrame]:
    cfg = Config(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio, periodo_snapshot=periodo_fin, seed=seed)
    empleadores = generar_empleadores_pandas(cfg)
    afiliados = generar_afiliados_pandas(n_afiliados, periodo_inicio, periodo_fin, seed)
    macro = macro_pandas(periodo_inicio, periodo_fin)
    periodos = rango_periodos(periodo_inicio, periodo_fin)
    simular = _simular_particion(empleadores, macro, periodos, seed)
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
