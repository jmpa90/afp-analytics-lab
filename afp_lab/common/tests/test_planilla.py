"""Tests de `afp_lab/common/planilla.py` y `tablas_equivalencia.py` (pandas, sin Spark):
coherencia con `panel_mensual`, forma del dataset, y que el modulo sea aditivo (no cambia
`generar_universo` ni sus salidas). CONVENCIONES.md #1 pide, como minimo, forma del dataset y
coherencia; el brief (`BRIEF_PLANILLA.md`, punto 4) pide explicitamente la coherencia de RIM con
el panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.common.planilla import (_estado_dnp_empleador, construir_agregado_empleador,
                                     construir_rezagos, generar_planilla)
from afp_lab.common.synth_base import generar_universo
from afp_lab.common.tablas_equivalencia import (CODIGOS_NO_GENERADOS, MOVIMIENTO_PERSONAL,
                                                 TIPO_PAGO, tasa_aporte_empleador)

N_AFILIADOS = 900
PERIODO_INICIO = 202201
PERIODO_FIN = 202312
SEED = 7


@pytest.fixture(scope="module")
def universo():
    return generar_universo(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO,
                            periodo_fin=PERIODO_FIN, seed=SEED, lote=N_AFILIADOS)


@pytest.fixture(scope="module")
def planilla(universo):
    return generar_planilla(universo, seed=SEED)


# --------------------------------------------------------------------- tablas_equivalencia
def test_tasa_aporte_empleador():
    """Ley 21.735: 0% antes del escalon, 3,5% desde agosto 2025 (unico escalon dentro del
    horizonte simulado; ver docstring de APORTE_EMPLEADOR_CRONOGRAMA)."""
    assert tasa_aporte_empleador(202507) == 0.0
    assert tasa_aporte_empleador(202508) == pytest.approx(0.035)
    assert tasa_aporte_empleador(202609) == pytest.approx(0.035)


def test_codigos_movimiento_personal_documentados():
    assert set(CODIGOS_NO_GENERADOS) == {5, 6, 13}
    for c in (0, 1, 2, 3, 4, 7, 8, 11, 12):
        assert c in MOVIMIENTO_PERSONAL
    assert set(TIPO_PAGO) >= {"01", "02", "05", "06"}


# --------------------------------------------------------------------------------- aditivo
def test_generar_universo_no_se_modifica(universo):
    """El brief es explicito: planilla.py es ADITIVO, no cambia generar_universo ni sus
    salidas. Se re-genera con la misma semilla y se compara columna a columna y valor a valor
    contra la version usada para construir la planilla (fixture `universo`, ya consumida por
    `generar_planilla`) para descartar mutacion in-place."""
    universo2 = generar_universo(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO,
                                 periodo_fin=PERIODO_FIN, seed=SEED, lote=N_AFILIADOS)
    assert set(universo) == set(universo2)
    for k in universo:
        assert list(universo[k].columns) == list(universo2[k].columns), f"columnas cambiaron: {k}"
        pd.testing.assert_frame_equal(universo[k].reset_index(drop=True),
                                      universo2[k].reset_index(drop=True), check_like=False)


# ------------------------------------------------------------------------- forma del dataset
def test_forma_y_tipos(planilla):
    cols = ["rut_trabajador", "empleador_id", "periodo", "tipo_pago", "dias_trabajados",
           "codigo_movimiento_personal", "fecha_desde", "fecha_hasta", "tipo_trabajador",
           "tipo_contrato_planilla", "renta_imponible_afp", "renta_imponible_afc",
           "cotizacion_obligatoria", "sis", "aporte_empleador", "periodo_recepcion",
           "estado_pago", "tipo_movimiento"]
    assert list(planilla.columns) == cols
    assert len(planilla) > 0
    # sin NaN salvo tipo_contrato_planilla (simplificacion documentada para el segundo empleador)
    nulos = planilla.isna().sum()
    assert nulos.drop("tipo_contrato_planilla").eq(0).all(), nulos[nulos > 0].to_dict()
    assert (planilla["renta_imponible_afp"] >= 0).all()
    assert (planilla["renta_imponible_afc"] >= 0).all()
    assert (planilla["renta_imponible_afc"] <= planilla["renta_imponible_afp"] + 1e-6).all()
    assert set(planilla["tipo_pago"]) <= set(TIPO_PAGO)
    assert set(planilla["codigo_movimiento_personal"]) <= set(MOVIMIENTO_PERSONAL)
    assert set(planilla["estado_pago"]) <= {"PAGADA", "DNP"}
    assert set(planilla["tipo_movimiento"]) <= {"DECLARACION", "RECTIFICACION", "PAGO_DNP"}
    assert set(planilla["codigo_movimiento_personal"]).isdisjoint(CODIGOS_NO_GENERADOS)


def test_periodo_recepcion_no_anterior_al_periodo(planilla):
    """La AFP nunca recibe una linea antes del periodo que declara (CONVENCIONES: la recepcion
    puede adelantarse al mes calendario del periodo -pago anticipado- pero no al periodo mismo)."""
    assert (planilla["periodo_recepcion"] >= planilla["periodo"]).all()


def test_dias_trabajados_en_rango(planilla):
    assert planilla["dias_trabajados"].between(0, 31).all()


# -------------------------------------------------------------- coherencia con panel_mensual
def test_coherencia_rim_con_panel(universo, planilla):
    """Brief punto 4: 'coherencia obligatoria con panel_mensual (misma RIM total, mismos
    empleadores)'. Se valida a nivel (afiliado, periodo) sumando TODOS los empleadores del mes
    (ver docstring de `generar_planilla` sobre el caso borde empleador_id == empleador2_id)."""
    panel = universo["panel_mensual"]
    decl = planilla[(planilla["tipo_movimiento"] == "DECLARACION")
                    & (planilla["tipo_pago"].isin(["01", "02", "05"]))]
    total = (decl.groupby(["rut_trabajador", "periodo"])["renta_imponible_afp"]
            .sum().reset_index().rename(columns={"rut_trabajador": "afiliado_id"}))

    esperado = panel[["afiliado_id", "periodo", "empleador_id", "rim_emp1",
                      "empleador2_id", "rim_emp2"]].copy()
    esperado["rim_emp1"] = np.where(esperado["empleador_id"].notna(), esperado["rim_emp1"], 0.0)
    esperado["rim_emp2"] = np.where(esperado["empleador2_id"].notna(), esperado["rim_emp2"], 0.0)
    esperado["rim_total"] = esperado["rim_emp1"].fillna(0) + esperado["rim_emp2"].fillna(0)
    esperado = esperado.loc[esperado["rim_total"] > 0, ["afiliado_id", "periodo", "rim_total"]]

    m = esperado.merge(total, on=["afiliado_id", "periodo"], how="left")
    m["renta_imponible_afp"] = m["renta_imponible_afp"].fillna(0.0)
    assert len(m) > 0
    assert np.allclose(m["rim_total"], m["renta_imponible_afp"], atol=1.0)


def test_mismos_empleadores_que_panel(universo, planilla):
    """El brief pide 'mismos empleadores': todo empleador_id que aparece en la planilla debe
    existir en panel_mensual/empleadores (no se inventan empleadores nuevos)."""
    empleadores_panel = set(universo["empleadores"]["empleador_id"])
    empleadores_planilla = set(planilla["empleador_id"].dropna().astype(int))
    assert empleadores_planilla <= empleadores_panel


def test_solo_dependientes_y_pensionados_con_empleador(universo, planilla):
    """Independientes (SII) y voluntarios (aporte directo) no pasan por Previred: todo RUT en la
    planilla debe tener, en el panel, un empleador vigente (principal o secundario) en algun
    periodo declarado."""
    panel = universo["panel_mensual"]
    con_empleador = set(panel.loc[panel["empleador_id"].notna(), "afiliado_id"]) | \
        set(panel.loc[panel["empleador2_id"].notna(), "afiliado_id"])
    ruts_planilla = set(planilla["rut_trabajador"])
    assert ruts_planilla <= con_empleador

    afiliados = universo["afiliados"].set_index("afiliado_id")
    tipos = afiliados.loc[list(ruts_planilla), "tipo_afiliado"]
    # los tipos DEPENDIENTE y PENSIONADO_ACTIVO son los unicos afectos a Previred
    assert set(tipos.unique()) <= {"DEPENDIENTE", "PENSIONADO_ACTIVO"}


def test_codigo_2_marca_retiro_y_1_7_marca_inicio(universo, planilla):
    """Los codigos de movimiento vienen de evento_afc del panel (INICIO/TERMINO), con el mismo
    desfase que la RIM (CONVENCIONES §3: panel/RIM conocidos en corte-2), asi que deben calzar
    exactamente con los eventos AFC ya generados por el proyecto 04/01."""
    panel = universo["panel_mensual"]
    linea01 = planilla[(planilla["tipo_pago"] == "01") & (planilla["tipo_movimiento"] == "DECLARACION")]
    # todo TERMINO en el panel (con empleador principal) debe tener su codigo 2 en la planilla
    terminos = panel.loc[(panel["evento_afc"] == "TERMINO") & panel["empleador_id"].notna(),
                         ["afiliado_id", "empleador_id", "periodo"]]
    marcados = linea01.loc[linea01["codigo_movimiento_personal"] == 2,
                           ["rut_trabajador", "empleador_id", "periodo"]].rename(
        columns={"rut_trabajador": "afiliado_id"})
    m = terminos.merge(marcados.assign(_ok=1), on=["afiliado_id", "empleador_id", "periodo"], how="left")
    assert m["_ok"].notna().all()


def test_afc_excluye_admin_publica_y_pensionados(universo, planilla):
    empleadores_lat = universo["empleadores_latente"]
    publicos = set(empleadores_lat.loc[empleadores_lat["rubro"] == "O_ADMIN_PUBLICA", "empleador_id"])
    afc_publico = planilla[planilla["empleador_id"].isin(publicos)]
    assert (afc_publico["renta_imponible_afc"] == 0).all()
    afc_pensionado = planilla[planilla["tipo_trabajador"] == "PENSIONADO_COTIZA"]
    assert (afc_pensionado["renta_imponible_afc"] == 0).all()


def test_cotizaciones_calculadas_con_las_tasas_declaradas(planilla):
    esperado_cot = np.round(planilla["renta_imponible_afp"] * 0.10, 0)
    assert np.allclose(planilla["cotizacion_obligatoria"], esperado_cot, atol=1.0)
    esperado_aporte = planilla["periodo"].map(lambda p: tasa_aporte_empleador(int(p))) * \
        planilla["renta_imponible_afp"]
    assert np.allclose(planilla["aporte_empleador"], np.round(esperado_aporte, 0), atol=1.0)
    assert (planilla.loc[planilla["periodo"] < 202508, "aporte_empleador"] == 0).all()


# ---------------------------------------------------------------------------- rezagos
def test_construir_rezagos(planilla):
    rezagos = construir_rezagos(planilla, seed=SEED)
    cols = ["rut_trabajador", "empleador_id", "periodo", "tipo_pago", "renta_imponible_afp",
           "periodo_recepcion_original", "periodo_imputacion"]
    assert list(rezagos.columns) == cols
    assert len(rezagos) > 0
    assert (rezagos["periodo_imputacion"] > rezagos["periodo_recepcion_original"]).all()
    # toda linea rezagada debe existir en la planilla como DECLARACION/PAGADA
    decl_pagada = planilla[(planilla["tipo_movimiento"] == "DECLARACION")
                          & (planilla["estado_pago"] == "PAGADA")]
    llaves = set(map(tuple, decl_pagada[["rut_trabajador", "empleador_id", "periodo", "tipo_pago"]]
                    .itertuples(index=False, name=None)))
    llaves_rez = set(map(tuple, rezagos[["rut_trabajador", "empleador_id", "periodo", "tipo_pago"]]
                        .itertuples(index=False, name=None)))
    assert llaves_rez <= llaves


# ---------------------------------------------------------------------------- perfil cuprum
def test_funciona_con_perfil_cuprum():
    """El perfil ("sistema"/"cuprum") de `generar_universo` no rompe planilla.py: se usa igual
    en el proyecto 04 (BRIEF_RONDA2, --perfil) y en el 06 (segmentacion, perfil='cuprum')."""
    universo_c = generar_universo(n_afiliados=300, periodo_inicio=PERIODO_INICIO,
                                  periodo_fin=PERIODO_FIN, seed=SEED, lote=300, perfil="cuprum")
    planilla_c = generar_planilla(universo_c, seed=SEED)
    assert len(planilla_c) > 0
    assert planilla_c.isna().drop(columns="tipo_contrato_planilla").sum().eq(0).all()


# --------------------------------------------------------- ronda 2: bug de recepcion (DS-2)
def test_recepcion_y_estado_consistentes_por_declaracion(planilla):
    """Bug reportado por DS-2 desde el proyecto 05: antes, `periodo_recepcion` y
    `estado_pago` se sorteaban linea a linea (`rng.random(n)` con n = numero de LINEAS), asi
    que dos lineas de una misma declaracion (mismo rut, empleador y periodo — p. ej. la 01 y
    la 02 de un afiliado con gratificacion) podian recibir un `periodo_recepcion` o un
    `estado_pago` distinto. Ahora deben ser IDENTICOS dentro de cada declaracion."""
    decl = planilla[planilla["tipo_movimiento"] == "DECLARACION"]
    g = decl.groupby(["rut_trabajador", "empleador_id", "periodo"])
    assert (g["periodo_recepcion"].nunique() == 1).all()
    assert (g["estado_pago"].nunique() == 1).all()


def test_dnp_es_por_empleador_y_periodo_no_por_linea(planilla):
    """La DNP (problema de caja del empleador) debe ser identica para TODOS los
    trabajadores del mismo empleador en el mismo periodo, no un sorteo independiente por
    afiliado."""
    decl = planilla[planilla["tipo_movimiento"] == "DECLARACION"]
    frac_dnp = (decl.groupby(["empleador_id", "periodo"])["estado_pago"]
               .apply(lambda s: (s == "DNP").mean()))
    assert set(frac_dnp.unique()) <= {0.0, 1.0}


def test_dnp_persistencia_configurable():
    """`_estado_dnp_empleador` con `rho` mas alto debe dar rachas de DNP mas largas (misma
    probabilidad marginal, mas persistencia mes a mes), sobre muchos empleadores para que el
    promedio sea estable con una sola semilla."""
    n_emp = 300
    # periodos yyyymm consecutivos, sin depender de utilidades internas del modulo
    periodos = []
    p = 202201
    for _ in range(48):
        periodos.append(p)
        p = p + 1 if p % 100 < 12 else (p // 100 + 1) * 100 + 1
    periodos = np.array(periodos)
    emp_lat = pd.DataFrame({"empleador_id": np.arange(n_emp), "p_dnp": np.full(n_emp, 0.25)})

    def racha_media(rho, seed):
        out = _estado_dnp_empleador(periodos, emp_lat, seed=seed, rho=rho)
        out = out.sort_values(["empleador_id", "periodo"])
        racha_total, n_rachas = 0, 0
        for _, serie in out.groupby("empleador_id")["_dnp"]:
            x = serie.to_numpy().astype(int)
            cambios = np.diff(x) != 0
            n_rachas += cambios.sum() + 1
            racha_total += len(x)
        return racha_total / n_rachas

    r_baja = racha_media(0.0, seed=1)
    r_alta = racha_media(0.7, seed=1)
    assert r_alta > r_baja * 1.15


def test_construir_agregado_empleador(universo, planilla):
    agg = construir_agregado_empleador(planilla)
    cols = ["empleador_id", "periodo", "dotacion", "masa_imponible", "lag_recepcion_medio",
           "frac_dnp", "n_lineas"]
    assert list(agg.columns) == cols
    assert not agg.isna().any().any()
    assert (agg["dotacion"] >= 0).all()
    assert (agg["masa_imponible"] >= 0).all()
    assert set(agg["frac_dnp"].unique()) <= {0.0, 1.0}
    assert (agg["lag_recepcion_medio"] >= 0).all()
    # dotacion de un (empleador, periodo) coincide con los rut distintos de sus lineas 01
    decl01 = planilla[(planilla["tipo_movimiento"] == "DECLARACION") & (planilla["tipo_pago"] == "01")]
    esperado = decl01.groupby(["empleador_id", "periodo"])["rut_trabajador"].nunique()
    m = agg.set_index(["empleador_id", "periodo"])["dotacion"]
    comparado = esperado.reindex(m.index).fillna(0).astype(int)
    assert (comparado == m).all()
