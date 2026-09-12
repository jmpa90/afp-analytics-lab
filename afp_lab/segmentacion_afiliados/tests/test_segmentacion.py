"""Tests de `segmentacion_afiliados` (pandas + sklearn, sin Spark): leakage, forma del
dataset, seleccion de k/estabilidad, validacion de negocio y clasificador. CONVENCIONES.md
#1 pide, como minimo, leakage, forma del dataset y una corrida chica end-to-end.

Universo chico (600 afiliados, 24 meses) para que el suite corra en segundos: la validacion
de los UMBRALES de aceptacion del brief (silueta >= 0,20, ARI >= 0,60, etc.) se hace en
`run.py` a escala de produccion (20 000 afiliados) y se reporta en el README, no aqui."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.segmentacion_afiliados.evaluate import (construir_targets_negocio, entrenar_clasificador,
                                                     matriz_transicion, perfil_segmentos,
                                                     separacion_negocio, tasas_por_segmento)
from afp_lab.segmentacion_afiliados.features import (FEATURES_CLUSTERING, FEATURES_PERFIL,
                                                      construir_features)
from afp_lab.segmentacion_afiliados.segmentar import (ajustar, asignar, escalar, evaluar_k,
                                                       seleccionar_k)
from afp_lab.segmentacion_afiliados.synthetic import add_months, generar_datos, generar_tablas_externas

N_AFILIADOS = 600
PERIODO_INICIO = 202201
PERIODO_FIN = 202312
SEED = 11
CORTE = 202305   # corte-2=202303 (>= periodo_inicio) y corte+6=202311 (<= periodo_fin)


@pytest.fixture(scope="module")
def tablas():
    return generar_datos(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO,
                         periodo_fin=PERIODO_FIN, seed=SEED, perfil="cuprum")


@pytest.fixture(scope="module")
def externas():
    return generar_tablas_externas(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO,
                                   periodo_fin=PERIODO_FIN, seed=SEED)


@pytest.fixture(scope="module")
def feats(tablas, externas):
    return construir_features(tablas, externas, CORTE)


# --------------------------------------------------------------------------------- forma
def test_features_una_fila_por_afiliado(tablas, feats):
    activos = tablas["afiliados"][tablas["afiliados"]["periodo_afiliacion"] <= CORTE]
    assert len(feats) == len(activos)
    assert feats["afiliado_id"].is_unique


def test_features_sin_nulos_ni_infinitos(feats):
    num = feats[FEATURES_CLUSTERING]
    assert not num.isna().any().any(), num.isna().sum()[lambda s: s > 0]
    assert np.isfinite(num.to_numpy()).all()
    for c in FEATURES_PERFIL:
        assert feats[c].isna().sum() == 0, c


def test_todos_los_tipos_de_afiliado_incluidos(feats):
    """A diferencia de un modelo de riesgo, la segmentacion cubre TODA la cartera (la
    pregunta de negocio es sobre toda la cartera de Cuprum, no solo dependientes)."""
    assert set(feats["tipo_afiliado"].unique()) <= {"DEPENDIENTE", "INDEPENDIENTE", "VOLUNTARIO",
                                                     "PENSIONADO_ACTIVO"}
    assert len(feats["tipo_afiliado"].unique()) >= 2


# ------------------------------------------------------------------------------- leakage
def test_leakage_panel_futuro_no_cambia_features(tablas, externas):
    """Alterar el panel DESPUES de `corte-2` (el ancla de las features nucleo) no debe
    cambiar ni una feature: CONVENCIONES #3, test obligatorio."""
    original = construir_features(tablas, externas, CORTE)

    corte_rim = add_months(CORTE, -2)
    panel_alterado = tablas["panel_mensual"].copy()
    fut = panel_alterado["periodo"] > corte_rim
    assert fut.any()
    rng = np.random.default_rng(0)
    panel_alterado.loc[fut, "rim_emp1"] = rng.uniform(1, 5_000_000, fut.sum())
    panel_alterado.loc[fut, "empleador_id"] = 999_999
    tablas_alt = {**tablas, "panel_mensual": panel_alterado}

    alterado = construir_features(tablas_alt, externas, CORTE)
    pd.testing.assert_frame_equal(
        original[["afiliado_id"] + FEATURES_CLUSTERING].sort_values("afiliado_id").reset_index(drop=True),
        alterado[["afiliado_id"] + FEATURES_CLUSTERING].sort_values("afiliado_id").reset_index(drop=True),
    )


def test_leakage_externas_futuras_no_cambian_features(tablas, externas):
    """Las tablas 'operacionales propias' (churn/apv) se conocen hasta `corte` mismo (desfase
    0): alterar lo posterior a `corte` tampoco debe cambiar las features."""
    original = construir_features(tablas, externas, CORTE)

    ext_alt = dict(externas)
    aportes = externas["apv_aportes"].copy()
    fut = aportes["periodo"] > CORTE
    assert fut.any()
    aportes.loc[fut, "monto"] = 999_999_999.0
    ext_alt["apv_aportes"] = aportes

    contactos = externas["churn_contactos"].copy()
    fut_c = contactos["periodo"] > CORTE
    if fut_c.any():
        contactos = pd.concat([contactos, contactos[fut_c].assign(canal="EJECUTIVO_PROPIO")], ignore_index=True)
    ext_alt["churn_contactos"] = contactos

    alterado = construir_features(tablas, ext_alt, CORTE)
    pd.testing.assert_frame_equal(
        original[["afiliado_id"] + FEATURES_CLUSTERING].sort_values("afiliado_id").reset_index(drop=True),
        alterado[["afiliado_id"] + FEATURES_CLUSTERING].sort_values("afiliado_id").reset_index(drop=True),
    )


# --------------------------------------------------------------------- clustering (chico)
def test_escalar_y_evaluar_k(feats):
    X, scaler = escalar(feats)
    assert X.shape == (len(feats), len(FEATURES_CLUSTERING))
    af_ids = feats["afiliado_id"].to_numpy()
    r = evaluar_k(X, af_ids, "kmeans", k=4, seed=0)
    assert -1.0 <= r.silueta <= 1.0
    assert np.isnan(r.ari_resampling) or (0.0 <= r.ari_resampling <= 1.0)
    assert r.cumple == (r.silueta >= 0.20 and (np.isfinite(r.ari_resampling) and r.ari_resampling >= 0.60)
                       and r.tamanos_ok)


def test_seleccionar_k_rango_acotado(feats):
    """Se acota k a {5,6} (en vez del rango completo 5-9) solo para que el test corra
    rapido; `seleccionar_k` es la MISMA funcion que usa `run.py` a escala de produccion."""
    tabla, ganador, scaler = seleccionar_k(feats, None, k_min=5, k_max=6, seed=0)
    assert set(tabla["algoritmo"]) == {"kmeans", "gmm"}
    assert set(tabla["k"]) == {5, 6}
    assert len(tabla) == 4
    assert ganador.k in (5, 6)
    assert np.isnan(ganador.ari_temporal)   # sin df_corte_t6, no se calcula


def test_ajustar_y_asignar_son_consistentes(feats):
    labels, modelo, scaler = ajustar(feats, "kmeans", k=5, seed=0)
    assert len(labels) == len(feats)
    assert set(labels) <= set(range(5))
    labels2 = asignar(modelo, scaler, feats)
    assert (labels == labels2).all()   # asignar sobre los mismos datos reproduce el ajuste


# ------------------------------------------------------------------- negocio y clasificador
def test_targets_negocio_en_rango(tablas, externas, feats):
    af_ids = feats["afiliado_id"].to_numpy()
    targets = construir_targets_negocio(tablas, externas, af_ids, CORTE)
    assert set(targets["afiliado_id"]) == set(af_ids)
    for col in ("cese_3m", "traspaso_6m", "apv_6m"):
        vals = targets[col].dropna()
        assert vals.isin([0.0, 1.0]).all()
    # apv_6m se evalua para TODOS (no hay poblacion "no aplica")
    assert targets["apv_6m"].isna().sum() == 0


def test_tasas_y_separacion_negocio(tablas, externas, feats):
    labels, _, _ = ajustar(feats, "kmeans", k=5, seed=0)
    df = feats.copy()
    df["segmento"] = labels
    targets = construir_targets_negocio(tablas, externas, df["afiliado_id"].to_numpy(), CORTE)
    tasas = tasas_por_segmento(df, targets)
    assert len(tasas) == df["segmento"].nunique()
    for col in ("tasa_cese_3m", "tasa_traspaso_6m", "tasa_apv_6m"):
        vals = tasas[col].dropna()
        assert ((vals >= 0) & (vals <= 1)).all()
    sep = separacion_negocio(tasas)
    assert set(sep["razones"]) == {"tasa_cese_3m", "tasa_traspaso_6m", "tasa_apv_6m"}
    assert isinstance(sep["cumple"], bool)


def test_perfil_segmentos(feats):
    labels, _, _ = ajustar(feats, "kmeans", k=5, seed=0)
    df = feats.copy()
    df["segmento"] = labels
    perfil = perfil_segmentos(df)
    assert len(perfil) == df["segmento"].nunique()
    assert np.isclose(perfil["pct"].sum(), 100.0)
    assert (perfil["n"] > 0).all()


def test_clasificador_explicativo(feats):
    labels, _, _ = ajustar(feats, "kmeans", k=5, seed=0)
    df = feats.copy()
    df["segmento"] = labels
    resultado = entrenar_clasificador(df, seed=0, test_size=0.3)
    assert 0.0 <= resultado["accuracy"] <= 1.0
    assert resultado["n_train"] + resultado["n_test"] == len(df)
    assert set(resultado["importancias"]["feature"]) == set(FEATURES_CLUSTERING) | {
        "tipo_afiliado", "sexo", "region", "nivel_educacional", "rubro", "fondo"}


def test_matriz_transicion():
    af_a = np.array([1, 2, 3, 4])
    lab_a = np.array([0, 0, 1, 1])
    af_b = np.array([2, 3, 4, 5])
    lab_b = np.array([0, 1, 1, 0])
    tab, n_comun = matriz_transicion(af_a, lab_a, af_b, lab_b)
    assert n_comun == 3   # afiliados 2, 3, 4
    filas_no_nulas = tab.sum(axis=1).dropna()
    assert np.allclose(filas_no_nulas.to_numpy(), 1.0)
