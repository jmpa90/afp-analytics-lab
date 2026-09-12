"""Tests del proyecto 03 (propension a APV).

Cubren el minimo de CONVENCIONES §1: leakage (§3), forma del dataset y corrida end-to-end
pequena; ademas poblacion/target, baselines, determinismo y coherencia de la simulacion.
Se usan universos chicos para que la suite corra en menos de 2 minutos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.rim_nowcasting.calendario import add_months
from afp_lab.propension_apv import evaluate as ev
from afp_lab.propension_apv import features as F
from afp_lab.propension_apv import model as M
from afp_lab.propension_apv import run as R
from afp_lab.propension_apv.synthetic import (ESTADOS_CON_INGRESO, ESTADOS, HORIZONTE,
                                              generar_tablas, truncar_tablas)

N_TEST = 900
SEED = 7
CORTE_A = 202412
CORTE_B = 202506


@pytest.fixture(scope="module")
def tablas():
    return generar_tablas(n_afiliados=N_TEST, seed=SEED)


@pytest.fixture(scope="module")
def denso(tablas):
    return F.preparar_panel(tablas)


# ============================================================== 1. sin leakage (CONVENCIONES §3)
@pytest.mark.parametrize("corte", [CORTE_A, CORTE_B])
def test_features_no_dependen_de_informacion_posterior_al_corte(tablas, corte):
    """Borrar TODO lo posterior al corte no puede cambiar ni una celda de las features."""
    completas = F.construir_features(tablas, corte)
    truncadas = F.construir_features(truncar_tablas(tablas, corte), corte)
    pd.testing.assert_frame_equal(completas, truncadas)


@pytest.mark.parametrize("corte", [CORTE_A, CORTE_B])
def test_alterar_el_futuro_no_cambia_las_features(tablas, corte):
    """Alterar (no solo borrar) los registros posteriores al corte deja las features iguales."""
    base = F.construir_features(tablas, corte)
    alteradas = {k: v.copy() for k, v in tablas.items()}

    fut_ap = alteradas["aportes_voluntarios"]["periodo"] > corte
    alteradas["aportes_voluntarios"].loc[fut_ap, "monto"] *= 100.0
    fut_panel = alteradas["panel_mensual"]["periodo"] > corte
    alteradas["panel_mensual"].loc[fut_panel, ["rim_emp1", "rim_emp2"]] = 0.0
    alteradas["panel_mensual"].loc[fut_panel, "estado"] = "CESANTE"
    fut_camp = alteradas["campanas"]["periodo"] > corte
    alteradas["campanas"].loc[fut_camp, "respuesta"] = "ABRIO"
    fut_sal = alteradas["saldos"]["periodo"] > corte
    alteradas["saldos"].loc[fut_sal, "saldo_voluntario"] = 1e12
    fut_apvc = alteradas["apvc_empleador"]["periodo"] > corte
    alteradas["apvc_empleador"].loc[fut_apvc, "ofrece_apvc"] = 1

    pd.testing.assert_frame_equal(base, F.construir_features(alteradas, corte))


def test_tramo_tributario_respeta_la_operacion_renta(tablas, denso):
    """En un corte anterior a abril solo se conoce el anio Y-2 (la renta de Y-1 llega en abril)."""
    f_ene = F.construir_features(tablas, 202501, denso=denso)
    trib = tablas["tributario"]
    esperado = trib[trib["anio"] == 2023].set_index("afiliado_id")["tramo_igc"]
    obtenido = f_ene.set_index("afiliado_id")["tramo_igc_declarado"]
    comunes = esperado.index.intersection(obtenido.index)
    assert (esperado.loc[comunes].to_numpy() == obtenido.loc[comunes].to_numpy()).all()
    # y en un corte de abril o posterior, el anio Y-1
    f_may = F.construir_features(tablas, 202505, denso=denso)
    esperado = trib[trib["anio"] == 2024].set_index("afiliado_id")["tramo_igc"]
    obtenido = f_may.set_index("afiliado_id")["tramo_igc_declarado"]
    comunes = esperado.index.intersection(obtenido.index)
    assert (esperado.loc[comunes].to_numpy() == obtenido.loc[comunes].to_numpy()).all()


def test_ninguna_feature_es_latente(tablas, denso):
    f = F.construir_features(tablas, CORTE_A, denso=denso)
    cols = F.columnas_modelo(f.assign(y=0, monto=0.0))
    assert not [c for c in cols if c.startswith("lat_") or c.startswith("p_")]
    # las tablas publicadas tampoco traen columnas latentes
    for df in tablas.values():
        assert not [c for c in df.columns if c.startswith("lat_")]


def test_split_temporal_deja_gap_entre_target_de_train_y_test(tablas):
    cortes = F.cortes_disponibles(tablas, 202301, HORIZONTE)
    split = F.split_temporal(cortes, n_test=3, n_valid=3, gap=HORIZONTE, horizonte=HORIZONTE)
    ultimo_target_train = add_months(max(split["train"] + split["valid"]), HORIZONTE)
    assert add_months(ultimo_target_train, HORIZONTE) <= min(split["test"])
    assert set(split["train"]).isdisjoint(split["valid"])
    assert max(split["train"] + split["valid"]) < min(split["test"])


# ============================================================== 2. as-of verificado a mano
def test_as_of_calculado_a_mano_en_dos_casos(tablas, denso):
    """Recalcula a mano, desde las tablas largas, dos casos de dos cortes distintos (§7)."""
    for corte in (CORTE_A, CORTE_B):
        f = F.construir_features(tablas, corte, denso=denso).set_index("afiliado_id")
        meses_12 = [add_months(corte, -k) for k in range(12)]
        meses_24 = [add_months(corte, -k) for k in range(24)]
        panel, ap = tablas["panel_mensual"], tablas["aportes_voluntarios"]
        for afiliado in list(f.index[:2]):
            p = panel[(panel["afiliado_id"] == afiliado) & (panel["periodo"].isin(meses_12))]
            rim = p["rim_emp1"] + p["rim_emp2"]
            assert f.loc[afiliado, "densidad_12"] == pytest.approx((rim > 0).sum() / 12.0, abs=1e-6)
            assert f.loc[afiliado, "rim_media_12"] == pytest.approx(
                rim[rim > 0].sum() / max((rim > 0).sum(), 1), rel=1e-4)

            a24 = ap[(ap["afiliado_id"] == afiliado) & (ap["periodo"].isin(meses_24))]
            assert f.loc[afiliado, "n_aportes_24m"] == pytest.approx(len(a24))
            assert f.loc[afiliado, "monto_aportes_24m"] == pytest.approx(a24["monto"].sum(), rel=1e-3)

            sal = tablas["saldos"]
            s = sal[(sal["afiliado_id"] == afiliado) & (sal["periodo"] == corte)]
            assert f.loc[afiliado, "saldo_voluntario"] == pytest.approx(
                float(s["saldo_voluntario"].iloc[0]), rel=1e-3)


# ============================================================== 3. poblacion y target
def test_poblacion_elegible_cumple_las_reglas_del_brief(tablas, denso):
    corte = CORTE_A
    f = F.construir_features(tablas, corte, denso=denso)
    ap, panel = tablas["aportes_voluntarios"], tablas["panel_mensual"]

    meses_6 = [add_months(corte, -k) for k in range(6)]
    con_aporte = set(ap[ap["periodo"].isin(meses_6)]["afiliado_id"])
    assert not set(f["afiliado_id"]) & con_aporte, "hay elegibles con aporte en los 6 meses previos"

    meses_12 = [add_months(corte, -k) for k in range(12)]
    p = panel[panel["periodo"].isin(meses_12)]
    cotizantes = set(p[(p["rim_emp1"] + p["rim_emp2"]) > 0]["afiliado_id"])
    assert set(f["afiliado_id"]) <= cotizantes, "hay elegibles sin cotizacion en 12 meses"

    retirados = set(panel[(panel["periodo"] == corte) & (panel["estado"] == "RETIRADO")]["afiliado_id"])
    assert not set(f["afiliado_id"]) & retirados


def test_target_es_la_suma_de_los_6_meses_siguientes(tablas):
    corte = CORTE_A
    t = F.construir_target(tablas, corte, HORIZONTE)
    meses = [add_months(corte, k) for k in range(1, HORIZONTE + 1)]
    ap = tablas["aportes_voluntarios"]
    esperado = ap[ap["periodo"].isin(meses)].groupby("afiliado_id")["monto"].sum()
    obtenido = t.set_index("afiliado_id")["monto"]
    assert set(esperado.index) == set(obtenido.index)
    assert np.allclose(esperado.sort_index().to_numpy(), obtenido.sort_index().to_numpy())
    assert (t["y"] == 1).all()


def test_forma_del_dataset(tablas, denso):
    cortes = [CORTE_A, CORTE_B]
    d = F.construir_dataset(tablas, cortes, HORIZONTE, denso=denso)
    assert not d.duplicated(["afiliado_id", "periodo_corte"]).any()
    assert set(d["periodo_corte"].unique()) == set(cortes)
    assert d["y"].isin([0, 1]).all()
    assert (d["monto"] >= 0).all()
    assert ((d["monto"] > 0) == (d["y"] == 1)).all()
    assert not d[F.columnas_modelo(d)].isna().any().any()
    for c in F.CATEGORICAS:
        assert str(d[c].dtype) == "category"
    assert 0.01 < d["y"].mean() < 0.25


# ============================================================== 4. simulacion coherente
def test_determinismo(tablas):
    otras = generar_tablas(n_afiliados=N_TEST, seed=SEED)
    for k in tablas:
        pd.testing.assert_frame_equal(tablas[k], otras[k])


def test_un_cesante_no_hace_aportes(tablas):
    panel = tablas["panel_mensual"][["afiliado_id", "periodo", "estado"]]
    ap = tablas["aportes_voluntarios"].merge(panel, on=["afiliado_id", "periodo"], how="left")
    estados_validos = {k for k, v in ESTADOS.items() if v in ESTADOS_CON_INGRESO}
    assert set(ap["estado"].dropna().unique()) <= estados_validos


def test_descuento_por_planilla_solo_con_empleador(tablas):
    panel = tablas["panel_mensual"][["afiliado_id", "periodo", "empleador_id"]]
    ap = tablas["aportes_voluntarios"].merge(panel, on=["afiliado_id", "periodo"], how="left")
    planilla = ap[ap["canal"] == "DESCUENTO_PLANILLA"]
    assert planilla["empleador_id"].notna().all()


def test_tasas_objetivo_del_brief():
    """12-18 % de los cotizantes con algun aporte en el anio y 5-8 % con aporte mensual.

    Se usa un universo mas grande que el del resto de los tests porque con pocos empleadores
    el sorteo del APVC mueve mucho la tasa; las bandas son algo mas anchas que las del brief
    por el ruido de muestreo (la corrida de referencia de 20.000 afiliados queda dentro).
    """
    tablas = generar_tablas(n_afiliados=5_000, seed=SEED)
    panel = tablas["panel_mensual"]
    panel = panel.assign(rim=panel["rim_emp1"] + panel["rim_emp2"], anio=panel["periodo"] // 100)
    ap = tablas["aportes_voluntarios"].assign(anio=tablas["aportes_voluntarios"]["periodo"] // 100)
    for anio in (2023, 2024, 2025):
        cot = set(panel[(panel["anio"] == anio) & (panel["rim"] > 0)]["afiliado_id"])
        a = ap[ap["anio"] == anio]
        con = len(set(a["afiliado_id"]) & cot) / len(cot)
        men = len(set(a[a["periodicidad"] == "MENSUAL"]["afiliado_id"]) & cot) / len(cot)
        assert 0.11 <= con <= 0.20, f"{anio}: tasa anual de aporte {con:.3f}"
        assert 0.04 <= men <= 0.09, f"{anio}: tasa de aporte mensual {men:.3f}"


def test_saldo_voluntario_es_consistente_con_los_aportes(tablas):
    """El saldo voluntario solo puede subir en los meses en que hay aporte (o por rentabilidad)."""
    sal = tablas["saldos"].sort_values(["afiliado_id", "periodo"])
    ap = tablas["aportes_voluntarios"].groupby(["afiliado_id", "periodo"])["monto"].sum()
    sal = sal.join(ap.rename("aporte"), on=["afiliado_id", "periodo"]).fillna({"aporte": 0.0})
    sal["previo"] = sal.groupby("afiliado_id")["saldo_voluntario"].shift()
    d = sal.dropna(subset=["previo"])
    esperado = d["previo"] * 1.005 + d["aporte"]
    assert np.allclose(d["saldo_voluntario"], esperado, atol=2.0)


# ============================================================== 5. modelo y baselines
@pytest.fixture(scope="module")
def entrenado(tablas, denso):
    cortes = F.cortes_disponibles(tablas, 202301, HORIZONTE)
    split = F.split_temporal(cortes, n_test=2, n_valid=2, gap=HORIZONTE, horizonte=HORIZONTE)
    usados = split["train"][-6:] + split["valid"] + split["test"]
    datos = F.construir_dataset(tablas, usados, HORIZONTE, denso=denso)
    train = datos[datos["periodo_corte"].isin(split["train"][-6:])]
    valid = datos[datos["periodo_corte"].isin(split["valid"])]
    test = datos[datos["periodo_corte"].isin(split["test"])]
    params = M.ParametrosModelo(clf_n_estimators=120, reg_n_estimators=80, early_stopping=20)
    return M.entrenar(train, valid, params), train, valid, test


def test_prediccion_tiene_forma_y_rangos_validos(entrenado):
    modelo, _, _, test = entrenado
    p = M.predecir(modelo, test)
    assert len(p) == len(test)
    assert p["prob_aporte"].between(0, 1).all()
    assert (p["monto_condicional"] > 0).all()
    assert np.allclose(p["monto_esperado"], p["prob_aporte"] * p["monto_condicional"])


def test_baselines_y_metricas(entrenado):
    modelo, train, _, test = entrenado
    pred = {"modelo": M.predecir(modelo, test),
            "tasa_base": M.baseline_tasa_base(train, test),
            "regla_negocio": M.baseline_regla(train, test)}
    tabla = ev.tabla_clasificacion(test, pred).set_index("predictor")
    assert tabla.loc["tasa_base", "auc"] == pytest.approx(0.5)
    assert tabla.loc["modelo", "auc"] > tabla.loc["regla_negocio", "auc"]
    monto = ev.tabla_monto(test, pred)
    assert set(monto["vista"]) == {"condicional_en_aportantes", "esperado_en_todos"}
    neg = ev.metrica_negocio(test, {"modelo_monto_esperado": pred["modelo"]["monto_esperado"].to_numpy(),
                                    "renta": test["rim_ref"].to_numpy()})
    assert (neg["captura_monto"].between(0, 1)).all()
    assert len(M.importancias(modelo)) == 2 * len(modelo.columnas)
    assert len(M.explicar(modelo, test, n_ejemplos=3)["caso"].unique()) == 3


def test_persistencia(entrenado, tmp_path):
    modelo, _, _, test = entrenado
    M.guardar(modelo, tmp_path / "modelo")
    recargado = M.cargar(tmp_path / "modelo")
    assert (tmp_path / "modelo" / "calibracion.json").exists()
    pd.testing.assert_frame_equal(M.predecir(modelo, test), M.predecir(recargado, test))


# ============================================================== 6. end-to-end
def test_corrida_end_to_end_pequena(tmp_path):
    res = R.ejecutar(n_afiliados=1_200, salida=tmp_path, seed=11, n_test=2, n_valid=2)
    assert (tmp_path / "modelo" / "modelo.pkl").exists()
    assert (tmp_path / "predicciones_test.parquet").exists()
    for nombre in ("metricas_clasificacion", "metricas_monto", "metrica_negocio", "aceptacion",
                   "importancias", "ejemplos_explicados", "curva_calibracion"):
        assert (tmp_path / "reportes" / f"{nombre}.csv").exists()
    clas = res["clasificacion"].set_index("predictor")
    assert clas.loc["modelo", "auc"] > 0.65
    assert clas.loc["modelo", "auc"] > clas.loc["tasa_base", "auc"]
    assert res["tiempos"]["total"] < 120
