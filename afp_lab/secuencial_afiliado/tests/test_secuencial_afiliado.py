"""Tests de `secuencial_afiliado`: universo identico a 02/04, forma de las secuencias, sin
leakage (CONVENCIONES.md #1 y #3) y una corrida end-to-end pequena (modelo diminuto, 1 epoca).
Pensado para correr en < 60 s: escalas chicas (300-500 afiliados) y red diminuta."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.churn_traspaso.synthetic import generar_tablas as generar_tablas_churn
from afp_lab.riesgo_cese_cotizacion.synthetic import generar_datos as generar_datos_cese
from afp_lab.secuencial_afiliado.features import (VENTANA_SECUENCIA, CANALES_CAT, CANALES_NUM,
                                                   construir_matrices_panel, construir_secuencias)
from afp_lab.secuencial_afiliado.run import (_preparar_cabeza_cese, _preparar_cabeza_traspaso, main)
from afp_lab.secuencial_afiliado.synthetic import generar_tablas

N_AFILIADOS = 500
SEED = 123
PERIODO_INICIO = 202201
PERIODO_FIN = 202609
CORTE = 202406


@pytest.fixture(scope="module")
def tablas():
    return generar_tablas(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO,
                          periodo_fin=PERIODO_FIN, seed=SEED, perfil="cuprum")


@pytest.fixture(scope="module")
def mats(tablas):
    return construir_matrices_panel(tablas, N_AFILIADOS)


# ------------------------------------------------------------- universo comun (02 y 04)
def test_universo_identico_02_04():
    """Mismo seed/n/perfil/periodo -> mismo universo comun (afiliados, panel, macro): asi
    pueden combinarse las tablas de 02 y 04 sin re-generar ni duplicar nada (synthetic.py)."""
    tc = generar_tablas_churn(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO,
                              periodo_fin=PERIODO_FIN, seed=SEED, perfil="cuprum")
    tk = generar_datos_cese(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO,
                            periodo_fin=PERIODO_FIN, seed=SEED, perfil="cuprum")
    assert tc["afiliados"].equals(tk["afiliados"])
    assert tc["panel_mensual"].equals(tk["panel_mensual"])
    assert tc["macro"].equals(tk["macro"])


def test_tablas_combinadas_traen_todo_lo_necesario(tablas):
    for k in ("saldos", "contactos", "reclamos", "cambios_fondo", "uso_digital", "traspasos",
             "cotizaciones_conocidas", "afc_eventos", "licencias", "panel_mensual", "afiliados"):
        assert k in tablas and tablas[k] is not None


# ------------------------------------------------------------- forma de las secuencias
def test_forma_secuencias(tablas, mats):
    pares = pd.DataFrame({"afiliado_id": [1, 2, 3, 10], "periodo_corte": [CORTE, CORTE, CORTE, CORTE]})
    seq = construir_secuencias(mats, tablas["afiliados"], pares)
    assert seq.num.shape == (4, VENTANA_SECUENCIA, len(CANALES_NUM))
    assert seq.cat.shape == (4, VENTANA_SECUENCIA, len(CANALES_CAT))
    assert seq.mask.shape == (4, VENTANA_SECUENCIA)
    assert seq.static_num.shape[0] == 4 and seq.static_cat.shape[0] == 4
    assert np.isin(seq.mask, [0.0, 1.0]).all()
    # el ultimo paso (corte-2) esta observado para un afiliado afiliado desde antes del inicio
    assert seq.mask[0, -1] == 1.0


def test_padding_afiliado_nuevo(tablas, mats):
    """Un afiliado cuya `periodo_afiliacion` es posterior al inicio de la ventana debe tener los
    meses anteriores enmascarados (relleno, sin fuga: simplemente no hay informacion)."""
    afiliados = tablas["afiliados"]
    recientes = afiliados[afiliados["periodo_afiliacion"] > 202401]
    if recientes.empty:
        pytest.skip("no hay afiliados nuevos en esta semilla/escala")
    af_id = int(recientes.iloc[0]["afiliado_id"])
    pares = pd.DataFrame({"afiliado_id": [af_id], "periodo_corte": [202406]})
    seq = construir_secuencias(mats, afiliados, pares)
    assert seq.mask[0].sum() < VENTANA_SECUENCIA   # no todos los meses estan observados
    assert seq.mask[0, 0] == 0.0                   # el mes mas antiguo de la ventana, sin historia


# ------------------------------------------------------------- sin leakage (CONVENCIONES #3)
def test_secuencia_sin_leakage(tablas, mats):
    """Alterar el panel DESPUES de `corte-2` no puede cambiar la secuencia construida para
    `corte` (CONVENCIONES #3: "alterar o borrar cualquier registro posterior al corte no
    cambia las features")."""
    from afp_lab.rim_nowcasting.calendario import add_months

    pares = pd.DataFrame({"afiliado_id": [1, 2, 3], "periodo_corte": [CORTE, CORTE, CORTE]})
    seq_original = construir_secuencias(mats, tablas["afiliados"], pares)

    corte_rim = add_months(CORTE, -2)
    panel_alterado = tablas["panel_mensual"].copy()
    futuro = panel_alterado["periodo"] > corte_rim
    assert futuro.sum() > 0
    panel_alterado.loc[futuro, "rim_emp1"] = 999_999_999.0
    panel_alterado.loc[futuro, "estado"] = "CESANTE"
    panel_alterado = panel_alterado[~(futuro & (panel_alterado["afiliado_id"] == 1))]  # borrar filas de 1 tambien

    tablas_alteradas = dict(tablas)
    tablas_alteradas["panel_mensual"] = panel_alterado
    mats_alteradas = construir_matrices_panel(tablas_alteradas, N_AFILIADOS)
    seq_alterada = construir_secuencias(mats_alteradas, tablas["afiliados"], pares)

    np.testing.assert_array_equal(seq_original.num, seq_alterada.num)
    np.testing.assert_array_equal(seq_original.cat, seq_alterada.cat)
    np.testing.assert_array_equal(seq_original.mask, seq_alterada.mask)


# ------------------------------------------------------------- mismos cortes/split que 02 y 04
def test_mismos_cortes_test_que_02_y_04(tablas):
    """BRIEF.md: "los mismos cortes evaluables y el mismo split_temporal ... que 02". Los
    cortes de test dependen solo del rango de periodos (no de `n_afiliados`), y en el rango de
    referencia coinciden entre 02 y 04 (mismo horizonte de 6 meses, mismo `periodo_fin`)."""
    cab_tr = _preparar_cabeza_traspaso(tablas, max_cortes_train=3, n_test=3, n_valid=3)
    cab_ce = _preparar_cabeza_cese(tablas, PERIODO_INICIO, PERIODO_FIN, max_cortes_train=3)
    assert sorted(cab_tr["split"]["test"]) == sorted(cab_ce["split"]["cortes"]["test"])
    assert len(cab_tr["train"]) > 0 and len(cab_ce["train"]) > 0
    assert set(cab_tr["train"]["afiliado_id"]).issubset(set(tablas["afiliados"]["afiliado_id"]))


# ------------------------------------------------------------- corrida end-to-end pequena
def test_run_end_to_end_pequeno():
    """Corrida completa con un modelo diminuto (hidden=4, 1 epoca) a escala chica: verifica que
    todo el pipeline (tablas -> datasets -> baselines -> red -> metricas) corre sin errores y
    devuelve las tablas esperadas."""
    res = main(n_afiliados=400, seed=7, salida="/tmp/secuencial_afiliado_test_run",
              max_cortes_train_traspaso=3, max_cortes_train_cese=3, epocas=1, paciencia=1,
              hidden=4, dim_embedding=8, batch_size=64, n_boot=10)
    for tabla in (res["tabla_traspaso"], res["tabla_cese"]):
        assert {"modelo", "auc", "pr_auc", "lift_decil_1"}.issubset(tabla.columns)
        assert "red_multitarea" in tabla["modelo"].to_numpy()
        assert "lightgbm_produccion" in tabla["modelo"].to_numpy()
        assert "lightgbm_tabular_lags" in tabla["modelo"].to_numpy()
        assert tabla["auc"].between(0, 1).all()
    assert res["conclusion"]["veredicto"] in ("reemplaza", "complementa", "no_aporta")
