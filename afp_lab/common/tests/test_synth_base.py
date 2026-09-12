"""Ronda 3 (brief `empleador`, objetivo 1): la densidad de cotizacion (fraccion de afiliados con
RIM > 0 en un mes tipico) del universo comun debe quedar calibrada contra la Ficha Estadistica
Previsional N 165 (SP, ago 2026; `docs/contexto_afp_cuprum.md` S0): 5.775.070 cotizantes sobre
12.201.632 afiliados = 47,3 % para el sistema, 66 % para Cuprum (66,94 % en la memoria 2025). Antes
de la ronda 3 el perfil `sistema` dejaba ~79 % (optimista) y `cuprum` ~71-73 % (tambien fuera de
banda); ver `PERFILES` en `synth_base.py`."""
from __future__ import annotations

import pandas as pd
import pytest

from afp_lab.common.synth_base import generar_universo

N_AFILIADOS = 6000
PERIODO_INICIO = 202201
PERIODO_FIN = 202609
SEED = 42


def _densidad_por_periodo(perfil: str, n_afiliados: int = N_AFILIADOS, seed: int = SEED) -> pd.Series:
    """Fraccion de afiliados con RIM > 0 (rim_emp1 + rim_emp2), por periodo, sobre los afiliados
    ya incorporados ese mes (denominador = filas de `panel_mensual` del periodo, que es exactamente
    la poblacion "afiliados que existen ese mes"). Se descartan los primeros 12 meses (burn-in del
    estado inicial) y los ultimos 3 (ventana de generacion mas corta) para medir un "mes tipico"."""
    universo = generar_universo(n_afiliados=n_afiliados, periodo_inicio=PERIODO_INICIO,
                                periodo_fin=PERIODO_FIN, seed=seed, perfil=perfil)
    panel = universo["panel_mensual"]
    rim_total = panel["rim_emp1"].fillna(0) + panel["rim_emp2"].fillna(0)
    cotiza = (rim_total > 0)
    por_periodo = pd.DataFrame({"periodo": panel["periodo"], "cotiza": cotiza}).groupby("periodo").agg(
        n=("cotiza", "size"), n_cotiza=("cotiza", "sum"))
    periodos = sorted(por_periodo.index)
    assert len(periodos) > 20, "ventana de periodos demasiado corta para descartar burn-in"
    centrales = periodos[12:-3]
    return (por_periodo.loc[centrales, "n_cotiza"] / por_periodo.loc[centrales, "n"])


@pytest.mark.parametrize("perfil,objetivo", [("sistema", 0.473), ("cuprum", 0.66)])
def test_densidad_de_cotizacion_calibrada_con_ficha_165(perfil, objetivo):
    """Ficha 165: 47,3 % de densidad para el sistema, 66 % para Cuprum. Tolerancia +/- 3 pp
    (brief ronda 3 `empleador`, objetivo 1)."""
    frac = _densidad_por_periodo(perfil)
    media = float(frac.mean())
    assert objetivo - 0.03 <= media <= objetivo + 0.03, (
        f"perfil={perfil}: densidad media {media:.4f}, objetivo {objetivo} +/- 0.03")


def test_densidad_cuprum_mayor_que_sistema():
    """Cuprum debe seguir teniendo mayor densidad que el promedio del sistema (66 % vs 47,3 %,
    Ficha 165), con margen frente al ruido de una sola semilla."""
    frac_sistema = _densidad_por_periodo("sistema").mean()
    frac_cuprum = _densidad_por_periodo("cuprum").mean()
    assert frac_cuprum > frac_sistema + 0.10
