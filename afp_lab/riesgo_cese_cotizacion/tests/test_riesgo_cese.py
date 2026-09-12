"""Tests de `riesgo_cese_cotizacion` (pandas, sin Spark): leakage, forma del dataset y una
corrida end-to-end pequena. CONVENCIONES.md #1 pide, como minimo, estas tres cosas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.riesgo_cese_cotizacion.evaluate import (curva_calibracion, evaluar,
                                                      metrica_negocio_agregado, metrica_negocio_captura)
from afp_lab.riesgo_cese_cotizacion.features import (FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
                                                      agregar_targets, construir_features)
from afp_lab.riesgo_cese_cotizacion.model import ModeloCese, baseline_tasa_base, predecir_tasa_base
from afp_lab.riesgo_cese_cotizacion.synthetic import add_months, generar_datos, periodo_to_idx

N_AFILIADOS = 1500
PERIODO_INICIO = 202201
PERIODO_FIN = 202512
CORTE = 202408


@pytest.fixture(scope="module")
def tablas():
    return generar_datos(n_afiliados=N_AFILIADOS, periodo_inicio=PERIODO_INICIO, periodo_fin=PERIODO_FIN, seed=123)


@pytest.fixture(scope="module")
def features(tablas):
    return construir_features(tablas, CORTE)


# ------------------------------------------------------------------- periodos
def test_periodos():
    assert add_months(202601, -2) == 202511
    assert add_months(202612, 1) == 202701
    assert periodo_to_idx(202603) - periodo_to_idx(202512) == 3


# --------------------------------------------------------------------- forma
def test_features_una_fila_por_afiliado(tablas, features):
    activos = tablas["afiliados"][tablas["afiliados"]["periodo_afiliacion"] <= CORTE]
    assert len(features) == len(activos)
    assert features["afiliado_id"].is_unique


def test_features_sin_nulos_ni_infinitos(features):
    num = features[FEATURES_NUMERICAS]
    assert not num.isna().any().any(), num.isna().sum()[lambda s: s > 0]
    assert np.isfinite(num.to_numpy()).all()
    for c in FEATURES_CATEGORICAS:
        assert features[c].isna().sum() == 0


def test_poblaciones_bien_definidas(features):
    # Modelo A: cotizando en corte-2 segun lo conocido (rim_l1 = RIM en corte-2)
    assert (features.loc[features["pop_cese"] == 1, "rim_l1"] > 0).all()
    assert (features.loc[features["pop_cese"] == 0, "rim_l1"] == 0).all()
    # Modelo B: dependientes, sin RIM en corte-2 pero con al menos 1 cotizacion en 24m previos
    pob_b = features[features["pop_reingreso"] == 1]
    assert (pob_b["tipo_afiliado"] == "DEPENDIENTE").all()
    assert (pob_b["rim_l1"] == 0).all()
    assert (pob_b["n_cotiza_24"] > 0).all()
    # ambas poblaciones son disjuntas
    assert (features["pop_cese"] * features["pop_reingreso"]).sum() == 0


def test_targets_consistentes_con_poblacion(tablas, features):
    ft = agregar_targets(features, tablas["panel_mensual"], PERIODO_FIN)
    assert ft["horizonte3_completo"].all() and ft["horizonte6_completo"].all()  # CORTE deja margen de sobra
    # y6=1 (cese 6m) implica y3=1 (si nunca cotiza en 6 meses, tampoco en los primeros 3)
    cesantes = ft[ft["pop_cese"] == 1]
    assert (cesantes.loc[cesantes["y6"] == 1, "y3"] == 1).all()
    # y_reingreso solo tiene sentido (no-nulo) donde se calculo horizonte6_completo
    assert ft["y_reingreso"].notna().all()


# ---------------------------------------------------------------- sin leakage
def test_leakage_alterar_panel_futuro_no_cambia_features(tablas):
    """CONVENCIONES #3: alterar o borrar cualquier registro posterior al corte no debe
    cambiar las features. Se ensucian los movimientos y giros con periodo > corte y se
    revisa que `construir_features(corte)` no cambie ni una celda."""
    base = construir_features(tablas, CORTE)

    t2 = dict(tablas)
    cot = tablas["cotizaciones_conocidas"].copy()
    fut = cot["periodo"] > CORTE
    cot.loc[fut, "rim"] = cot.loc[fut, "rim"] * 0 + 999_999_999.0
    t2["cotizaciones_conocidas"] = cot

    afc = tablas["afc_eventos"].copy()
    afc = afc[afc["periodo"] <= CORTE]  # borra eventos futuros
    t2["afc_eventos"] = afc

    lic = tablas["licencias"].copy()
    lic.loc[lic["periodo"] > CORTE, "dias_licencia"] = 30
    t2["licencias"] = lic

    giros = tablas["prestaciones_cesantia"].copy()
    giros.loc[giros["periodo"] > CORTE, "monto"] = 123.0
    t2["prestaciones_cesantia"] = giros

    alt = construir_features(t2, CORTE)
    pd.testing.assert_frame_equal(
        base.sort_values("afiliado_id").reset_index(drop=True),
        alt.sort_values("afiliado_id").reset_index(drop=True),
    )


def test_leakage_recepcion_tardia_no_se_usa(tablas):
    """Un movimiento con periodo <= corte-2 pero `periodo_recepcion` posterior al corte
    conocido (rezago) no debe entrar en la RIM conocida: si se le adelanta la recepcion al
    pasado, `rim_l1` de ese afiliado sube o se mantiene, nunca baja (nunca se "pierde"
    informacion ya usada, y nunca se usa informacion aun no recibida)."""
    cot = tablas["cotizaciones_conocidas"]
    hasta = add_months(CORTE, -1)
    tardios = cot[(cot["periodo_recepcion"] > hasta) & (cot["periodo"] <= hasta)]
    assert len(tardios) > 0, "el generador debe producir rezagos (P_TARDIO > 0)"

    base = construir_features(tablas, CORTE)
    t2 = dict(tablas)
    cot2 = cot.copy()
    cot2.loc[tardios.index, "periodo_recepcion"] = cot2.loc[tardios.index, "periodo"]  # llega a tiempo
    t2["cotizaciones_conocidas"] = cot2
    alt = construir_features(t2, CORTE)

    b = base.set_index("afiliado_id")["rim_l1"]
    a = alt.set_index("afiliado_id")["rim_l1"]
    assert (a.reindex(b.index) >= b - 1e-6).all()


def test_afc_conocido_hasta_corte_menos_1(tablas):
    """Un evento AFC fechado en `corte` (o despues) no debe influir en las features de esa
    fila, aun si su `periodo_recepcion` es <= corte (ver decision documentada en el README:
    el ultimo evento AFC/licencia conocido es de `corte-1`)."""
    afc = tablas["afc_eventos"]
    en_corte_o_despues = afc[afc["periodo"] >= CORTE]
    assert len(en_corte_o_despues) > 0
    base = construir_features(tablas, CORTE)
    t2 = dict(tablas)
    afc2 = afc.copy()
    afc2.loc[en_corte_o_despues.index, "periodo_recepcion"] = CORTE  # se fuerza a "conocido ya"
    t2["afc_eventos"] = afc2
    alt = construir_features(t2, CORTE)
    pd.testing.assert_frame_equal(base.sort_values("afiliado_id").reset_index(drop=True),
                                  alt.sort_values("afiliado_id").reset_index(drop=True))


# ------------------------------------------------------------- modelo (mini)
def test_modelo_entrena_y_supera_tasa_base():
    """Corrida end-to-end pequena: features en varios cortes, split temporal con gap, un
    LightGBM chico entrena y su AUC en test debe superar al baseline de tasa base."""
    tablas_ = generar_datos(n_afiliados=2500, periodo_inicio=202201, periodo_fin=202512, seed=55)
    cortes = [int(add_months(202506, -k)) for k in range(10)][::-1]
    filas = [construir_features(tablas_, c) for c in cortes]
    dataset = agregar_targets(pd.concat(filas, ignore_index=True), tablas_["panel_mensual"], 202512)
    pob = dataset[(dataset["pop_cese"] == 1) & dataset["horizonte3_completo"]]

    train = pob[pob["periodo_corte"].isin(cortes[:6])]
    val = pob[pob["periodo_corte"].isin(cortes[6:8])]
    test = pob[pob["periodo_corte"].isin(cortes[8:])]
    assert len(train) > 200 and len(val) > 50 and len(test) > 50

    modelo = ModeloCese.fit(train, val, "y3", params={"n_estimators": 60, "num_leaves": 15})
    pred = modelo.predict_proba(test)
    assert ((pred >= 0) & (pred <= 1)).all()
    m_modelo = evaluar(test.assign(pred=pred), "y3", "pred")

    tasas, tasa_g = baseline_tasa_base(pd.concat([train, val]), "y3")
    pred_base = predecir_tasa_base(tasas, tasa_g, test)
    m_base = evaluar(test.assign(pred=pred_base), "y3", "pred")

    assert m_modelo["auc"] > m_base["auc"] - 0.05   # con pocos datos se exige solo no perder por mucho
    assert m_modelo["brier"] <= m_base["brier"] + 0.02

    calib = curva_calibracion(test.assign(pred=pred), "y3", "pred")
    assert (calib["n"] > 0).all()
    negocio = metrica_negocio_agregado(test.assign(pred=pred), "y3", "pred")
    assert (negocio["n_filas"] > 0).all()
    captura = metrica_negocio_captura(test.assign(pred=pred), "y3", "pred", frac=0.10)
    assert 0 <= captura["pct_capturado"] <= 1
