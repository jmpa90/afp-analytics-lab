"""Tests del proyecto 03 (propension a APV).

Cubren el minimo de CONVENCIONES §1: leakage (§3), forma del dataset y corrida end-to-end
pequena; ademas la **convencion de fechas** de la ronda 2 (cada fuente con su propio desfase),
los gaps entre TODAS las particiones, poblacion/target, baselines, determinismo y coherencia de
la simulacion. Se usan universos chicos para que la suite corra en menos de 2 minutos.
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
from afp_lab.propension_apv.synthetic import (ANCLAS_FICHA_165, CUENTAS_APV_POR_COTIZANTE_SISTEMA,
                                              ESTADOS, ESTADOS_CON_INGRESO, HORIZONTE,
                                              MESES_DESFASE, MESES_DESFASE_LICENCIA,
                                              anio_tributario_disponible, generar_tablas,
                                              truncar_tablas)

N_AFILIADOS = 900
SEED = 7
CORTE_A = 202412
CORTE_B = 202506


@pytest.fixture(scope="module")
def tablas():
    return generar_tablas(n_afiliados=N_AFILIADOS, seed=SEED)


@pytest.fixture(scope="module")
def denso(tablas):
    return F.preparar_panel(tablas)


# ============================================================== 1. sin leakage (CONVENCIONES §3)
@pytest.mark.parametrize("corte", [CORTE_A, CORTE_B])
def test_features_no_dependen_de_informacion_posterior_al_corte(tablas, corte):
    """Quitar todo lo que no esta disponible en el corte no cambia ni una celda.

    `truncar_tablas` aplica el desfase de cada fuente y ademas **neutraliza la RIM, el estado y
    el empleador del mes `corte-1`**: de ese mes solo se conocen las licencias.
    """
    completas = F.construir_features(tablas, corte)
    truncadas = F.construir_features(truncar_tablas(tablas, corte), corte)
    pd.testing.assert_frame_equal(completas, truncadas)


@pytest.mark.parametrize("corte", [CORTE_A, CORTE_B])
def test_alterar_el_futuro_no_cambia_las_features(tablas, corte):
    """Alterar (no solo borrar) lo que no esta disponible deja las features iguales."""
    base = F.construir_features(tablas, corte)
    alteradas = {k: v.copy() for k, v in tablas.items()}

    fut_ap = alteradas["aportes_voluntarios"]["periodo"] > corte
    alteradas["aportes_voluntarios"].loc[fut_ap, "monto"] *= 100.0
    # el panel se lee en corte-2: mutar corte-1 y corte no puede cambiar nada salvo licencias
    fut_panel = alteradas["panel_mensual"]["periodo"] > add_months(corte, -MESES_DESFASE)
    alteradas["panel_mensual"].loc[fut_panel, ["rim_emp1", "rim_emp2"]] = 0.0
    alteradas["panel_mensual"].loc[fut_panel, "estado"] = "CESANTE"
    alteradas["panel_mensual"].loc[fut_panel, "meses_antiguedad"] = 0
    fut_camp = alteradas["campanas"]["periodo"] > corte
    alteradas["campanas"].loc[fut_camp, "respuesta"] = "ABRIO"
    fut_sal = alteradas["saldos"]["periodo"] > corte
    alteradas["saldos"].loc[fut_sal, "saldo_voluntario"] = 1e12
    fut_apvc = alteradas["apvc_empleador"]["periodo"] > add_months(corte, -MESES_DESFASE)
    alteradas["apvc_empleador"].loc[fut_apvc, "ofrece_apvc"] = 1
    fut_trib = alteradas["tributario"]["anio"] > anio_tributario_disponible(corte)
    alteradas["tributario"].loc[fut_trib, "tramo_igc"] = 7

    pd.testing.assert_frame_equal(base, F.construir_features(alteradas, corte))


def test_licencias_se_leen_un_mes_antes_que_el_panel(tablas, corte=CORTE_A):
    """Las licencias SI se leen en `corte-1` (y solo ellas): mutar ese mes debe moverlas."""
    base = F.construir_features(tablas, corte)
    alteradas = {k: v.copy() for k, v in tablas.items()}
    mes_lic = add_months(corte, -MESES_DESFASE_LICENCIA)
    alteradas["panel_mensual"].loc[alteradas["panel_mensual"]["periodo"] == mes_lic,
                                   "dias_licencia"] = 30
    nuevas = F.construir_features(alteradas, corte)
    assert not np.allclose(base["dias_licencia_12"], nuevas["dias_licencia_12"])
    # y el resto de las familias no se mueve
    for col in ("rim_corte_imm", "densidad_12", "meses_con_empleador", "saldo_obligatorio_imm"):
        assert np.allclose(base[col], nuevas[col]), col


def test_tramo_tributario_respeta_la_operacion_renta(tablas, denso):
    """En un corte anterior a abril solo se conoce el ano Y-2; desde abril, Y-1."""
    for corte in (202501, 202505):
        f = F.construir_features(tablas, corte, denso=denso)
        trib = tablas["tributario"]
        esperado = trib[trib["anio"] == anio_tributario_disponible(corte)].set_index("afiliado_id")
        obtenido = f.set_index("afiliado_id")["tramo_igc_declarado"]
        comunes = esperado.index.intersection(obtenido.index)
        assert len(comunes) > 0
        assert (esperado.loc[comunes, "tramo_igc"].to_numpy()
                == obtenido.loc[comunes].to_numpy()).all()


def test_ninguna_feature_es_latente_ni_de_escala(tablas, denso):
    f = F.construir_features(tablas, CORTE_A, denso=denso)
    cols = F.columnas_modelo(f.assign(y=0, monto=0.0))
    assert not [c for c in cols if c.startswith("lat_") or c.startswith("p_")]
    # las columnas de escala en CLP no pueden entrar al modelo
    assert not set(F.COLUMNAS_ESCALA) & set(cols)
    for df in tablas.values():
        assert not [c for c in df.columns if c.startswith("lat_")]


def test_split_temporal_deja_gap_entre_todas_las_particiones(tablas):
    """CONVENCIONES §3 (ronda 2): el gap se aplica train->valid y valid->test."""
    cortes = F.cortes_disponibles(tablas, R.PRIMER_CORTE, HORIZONTE)
    split = F.split_temporal(cortes, n_test=12, n_valid=12, gap=HORIZONTE, horizonte=HORIZONTE)
    for antes, despues in (("train", "valid"), ("valid", "test")):
        ultimo_target = add_months(max(split[antes]), HORIZONTE)
        assert add_months(ultimo_target, HORIZONTE) <= min(split[despues]), (antes, despues)
    assert len(split["test"]) == 12 and len(split["valid"]) == 12
    assert set(split["train"]).isdisjoint(split["valid"])


def test_origenes_moviles_respetan_los_gaps(tablas):
    cortes = F.cortes_disponibles(tablas, R.PRIMER_CORTE, HORIZONTE)
    origenes = F.origenes_moviles(cortes, n_test=12, n_valid=12, gap=HORIZONTE,
                                  horizonte=HORIZONTE, cada=3)
    assert sum(len(o["test"]) for o in origenes) == 12
    for o in origenes:
        for antes, despues in ((o["train"], o["valid"]), (o["valid"], o["test"])):
            ultimo_target = add_months(max(antes), HORIZONTE)
            assert add_months(ultimo_target, HORIZONTE) <= min(despues)
        assert set(o["train"]).isdisjoint(o["valid"])
        assert set(o["valid"]).isdisjoint(o["test"])
    # ventana expansiva: cada origen entrena con al menos tanto como el anterior
    tam = [len(o["train"]) for o in origenes]
    assert tam == sorted(tam)


def test_solape_conocido_entre_valid_y_el_pool_de_test(tablas):
    """Limitacion declarada (revision ronda 2, m-2).

    Dentro de cada origen el gap es exacto: el `valid` de un origen no toca NINGUN mes de target
    de su propio bloque de test, que es la condicion que importa (cada prediccion la hace un
    modelo que no vio los meses de target de esa fila). Al agrupar los 12 cortes para reportar en
    conjunto, el `valid` del ultimo origen si toca 3 meses de target de un bloque anterior, que
    predijo OTRO modelo. El test fija ese hecho para que no crezca sin que nadie lo note.
    """
    cortes = F.cortes_disponibles(tablas, R.PRIMER_CORTE, HORIZONTE)
    origenes = F.origenes_moviles(cortes, n_test=12, n_valid=12, gap=HORIZONTE,
                                  horizonte=HORIZONTE, cada=3)
    solape = F.solape_valid_pool_test(origenes, HORIZONTE)
    # condicion dura: cero solape de cada valid con su propio bloque de test
    assert (solape["solape_con_su_propio_test"] == 0).all()
    # limitacion conocida y acotada: solo el ultimo origen, y a lo mas 3 meses
    con_solape = solape[solape["solape_con_el_pool_de_test"] > 0]
    assert list(con_solape["origen"]) == [origenes[-1]["origen"]]
    assert con_solape["solape_con_el_pool_de_test"].max() <= 3


# ============================================================== 2. as-of verificado a mano
def test_as_of_calculado_a_mano_en_dos_casos(tablas, denso):
    """Recalcula a mano, desde las tablas largas, dos casos de dos cortes distintos (§7).

    El punto fino de la ronda 2: la RIM se lee en `corte-2` y los aportes hasta `corte`.
    """
    for corte in (CORTE_A, CORTE_B):
        f = F.construir_features(tablas, corte, denso=denso).set_index("afiliado_id")
        corte_panel = add_months(corte, -MESES_DESFASE)
        meses_panel_12 = [add_months(corte_panel, -k) for k in range(12)]
        meses_ap_24 = [add_months(corte, -k) for k in range(24)]
        panel, ap, sal = tablas["panel_mensual"], tablas["aportes_voluntarios"], tablas["saldos"]
        imm_p = float(tablas["macro"].set_index("periodo").loc[corte_panel, "imm"])
        imm_d = float(tablas["macro"].set_index("periodo").loc[corte, "imm"])
        for afiliado in list(f.index[:2]):
            p = panel[(panel["afiliado_id"] == afiliado) & (panel["periodo"].isin(meses_panel_12))]
            rim = p["rim_emp1"] + p["rim_emp2"]
            assert f.loc[afiliado, "densidad_12"] == pytest.approx((rim > 0).sum() / 12.0, abs=1e-6)
            assert f.loc[afiliado, "rim_media_12_imm"] == pytest.approx(
                rim[rim > 0].sum() / max((rim > 0).sum(), 1) / imm_p, rel=1e-4)
            fila = p[p["periodo"] == corte_panel]
            esperado = float(fila["rim_emp1"].iloc[0] + fila["rim_emp2"].iloc[0]) if len(fila) else 0.0
            assert f.loc[afiliado, "rim_corte_imm"] == pytest.approx(esperado / imm_p, rel=1e-4)

            a24 = ap[(ap["afiliado_id"] == afiliado) & (ap["periodo"].isin(meses_ap_24))]
            assert f.loc[afiliado, "n_aportes_24m"] == pytest.approx(len(a24))
            assert f.loc[afiliado, "monto_aportes_24m_imm"] == pytest.approx(
                a24["monto"].sum() / imm_d, rel=1e-3)

            s = sal[(sal["afiliado_id"] == afiliado) & (sal["periodo"] == corte)]
            assert f.loc[afiliado, "saldo_voluntario_imm"] == pytest.approx(
                float(s["saldo_voluntario"].iloc[0]) / imm_d, rel=1e-3)


# ============================================================== 3. poblacion y target
def test_poblacion_elegible_cumple_las_reglas_del_brief(tablas, denso):
    corte = CORTE_A
    corte_panel = add_months(corte, -MESES_DESFASE)
    f = F.construir_features(tablas, corte, denso=denso)
    ap, panel = tablas["aportes_voluntarios"], tablas["panel_mensual"]

    # sin aporte en corte-5 .. corte (ventana operacional, sin desfase)
    meses_6 = [add_months(corte, -k) for k in range(6)]
    con_aporte = set(ap[ap["periodo"].isin(meses_6)]["afiliado_id"])
    assert not set(f["afiliado_id"]) & con_aporte, "hay elegibles con aporte en los 6 meses previos"

    # cotizo al menos una vez en los 12 meses de panel conocidos
    meses_12 = [add_months(corte_panel, -k) for k in range(12)]
    p = panel[panel["periodo"].isin(meses_12)]
    cotizantes = set(p[(p["rim_emp1"] + p["rim_emp2"]) > 0]["afiliado_id"])
    assert set(f["afiliado_id"]) <= cotizantes, "hay elegibles sin cotizacion en 12 meses"

    retirados = set(panel[(panel["periodo"] == corte_panel)
                          & (panel["estado"] == "RETIRADO")]["afiliado_id"])
    assert not set(f["afiliado_id"]) & retirados


def test_el_target_empieza_despues_del_corte(tablas):
    """Nada del target esta observado cuando se arma la lista (BLOQUEANTE 5 de la ronda 1)."""
    corte = CORTE_A
    t = F.construir_target(tablas, corte, HORIZONTE)
    meses = [add_months(corte, k) for k in range(1, HORIZONTE + 1)]
    assert min(meses) == add_months(corte, 1)
    ap = tablas["aportes_voluntarios"]
    esperado = ap[ap["periodo"].isin(meses)].groupby("afiliado_id")["monto"].sum()
    obtenido = t.set_index("afiliado_id")["monto"]
    assert set(esperado.index) == set(obtenido.index)
    assert np.allclose(esperado.sort_index().to_numpy(), obtenido.sort_index().to_numpy())
    assert (t["y"] == 1).all()
    # ningun aporte del target ocurre en un mes visible al construir las features
    assert not set(meses) & {add_months(corte, -k) for k in range(0, 24)}


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
    otras = generar_tablas(n_afiliados=N_AFILIADOS, seed=SEED)
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
    assert ap[ap["canal"] == "DESCUENTO_PLANILLA"]["empleador_id"].notna().all()


def test_tasas_objetivo_del_brief():
    """12-18 % de los cotizantes con algun aporte en el ano y 5-8 % con aporte mensual.

    Universo mas grande que el del resto de los tests porque con pocos empleadores el sorteo del
    APVC mueve mucho la tasa; las bandas son algo mas anchas que las del brief por el ruido de
    muestreo (la corrida de referencia de 20.000 afiliados queda dentro).
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
    """El saldo voluntario solo sube por rentabilidad y por los aportes del mes."""
    sal = tablas["saldos"].sort_values(["afiliado_id", "periodo"])
    ap = tablas["aportes_voluntarios"].groupby(["afiliado_id", "periodo"])["monto"].sum()
    sal = sal.join(ap.rename("aporte"), on=["afiliado_id", "periodo"]).fillna({"aporte": 0.0})
    sal["previo"] = sal.groupby("afiliado_id")["saldo_voluntario"].shift()
    d = sal.dropna(subset=["previo"])
    assert np.allclose(d["saldo_voluntario"], d["previo"] * 1.005 + d["aporte"], atol=2.0)


# ============================================================== 5. modelo y baselines
@pytest.fixture(scope="module")
def entrenado(tablas, denso):
    cortes = F.cortes_disponibles(tablas, R.PRIMER_CORTE, HORIZONTE)
    split = F.split_temporal(cortes, n_test=3, n_valid=4, gap=HORIZONTE, horizonte=HORIZONTE)
    usados = split["train"][-8:] + split["valid"] + split["test"]
    datos = F.construir_dataset(tablas, usados, HORIZONTE, denso=denso)
    train = datos[datos["periodo_corte"].isin(split["train"][-8:])]
    valid = datos[datos["periodo_corte"].isin(split["valid"])]
    test = datos[datos["periodo_corte"].isin(split["test"])]
    params = M.ParametrosModelo(clf_n_estimators=120, reg_n_estimators=80, early_stopping=20)
    return M.entrenar(train, valid, params), train, valid, test


def test_hiperparametros_se_eligen_con_validacion(entrenado):
    """El numero de arboles lo decide el early stopping contra validacion, no una constante."""
    modelo, _, _, _ = entrenado
    n_clf, n_reg = modelo.n_arboles
    assert 1 <= n_clf <= modelo.params.clf_n_estimators
    assert 1 <= n_reg <= modelo.params.reg_n_estimators
    # la decision de calibrar tambien sale de validacion, no de test, y exige que la mejora sea
    # significativa frente a la dispersion ENTRE folds y material frente al Brier crudo
    esperado = (modelo.mejora_brier_media > M.N_SIGMAS_CALIBRACION * modelo.mejora_brier_sem
                and modelo.mejora_brier_media
                >= M.MEJORA_MINIMA_CALIBRACION * modelo.brier_valid_crudo)
    assert modelo.usa_calibracion == esperado
    assert modelo.mejora_brier_sem >= 0.0


def test_prediccion_tiene_forma_y_rangos_validos(entrenado):
    modelo, _, _, test = entrenado
    p = M.predecir(modelo, test)
    assert len(p) == len(test)
    assert p["prob_aporte"].between(0, 1).all()
    assert (p["monto_condicional"] > 0).all()
    assert np.allclose(p["monto_esperado"], p["prob_aporte"] * p["monto_condicional"])
    # el regresor modela un ratio sobre rim_ref: escalar la renta escala el monto condicional
    doble = test.copy()
    doble["rim_ref_clp"] = doble["rim_ref_clp"] * 2.0
    assert np.allclose(M.predecir(modelo, doble)["monto_condicional"],
                       2.0 * p["monto_condicional"], rtol=1e-6)


def test_smearing_se_calcula_en_validacion(entrenado):
    modelo, train, valid, _ = entrenado
    va_pos = valid[valid["y"] == 1]
    pred = va_pos["rim_ref_clp"].to_numpy() * np.exp(modelo.reg.predict(va_pos[modelo.columnas]))
    assert modelo.smearing == pytest.approx(va_pos["monto"].sum() / pred.sum(), rel=1e-9)


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
                                    "renta": test["rim_ref_clp"].to_numpy()})
    assert (neg["captura_monto"].between(0, 1)).all()
    assert len(M.importancias(modelo)) == 2 * len(modelo.columnas)
    assert len(M.explicar(modelo, test, n_ejemplos=3)["caso"].unique()) == 3


def test_reportes_de_sesgo_por_decil_y_por_corte(entrenado):
    modelo, _, _, test = entrenado
    pred = M.predecir(modelo, test)
    dec = ev.sesgo_monto_por_decil(test, pred)
    assert len(dec) == 10
    assert dec["share_monto_real"].sum() == pytest.approx(1.0, abs=1e-6)
    assert dec["monto_predicho"].sum() == pytest.approx(pred["monto_esperado"].sum(), rel=1e-6)
    corte = ev.metricas_por_corte(test, pred)
    assert len(corte) == test["periodo_corte"].nunique()
    assert set(["auc", "tasa_base", "sesgo_monto_esperado"]) <= set(corte.columns)


def test_persistencia(entrenado, tmp_path):
    modelo, _, _, test = entrenado
    M.guardar(modelo, tmp_path / "modelo")
    recargado = M.cargar(tmp_path / "modelo")
    assert (tmp_path / "modelo" / "calibracion.json").exists()
    pd.testing.assert_frame_equal(M.predecir(modelo, test), M.predecir(recargado, test))


# ============================================================== 6. end-to-end
def test_corrida_end_to_end_pequena(tmp_path):
    res = R.ejecutar(n_afiliados=800, salida=tmp_path, seed=11, n_test=6, n_valid=6,
                     cada_origen=3, diagnostico=False)
    # se persiste un modelo por origen (el .pkl de uno solo no reproduce el agregado)
    dirs_modelo = sorted((tmp_path / "modelo").glob("origen_*"))
    assert len(dirs_modelo) == len(res["origenes"])
    for d in dirs_modelo:
        assert (d / "modelo.pkl").exists() and (d / "calibracion.json").exists()
    assert (tmp_path / "predicciones_test.parquet").exists()
    for nombre in ("metricas_clasificacion", "metricas_monto", "metrica_negocio", "aceptacion",
                   "importancias", "ejemplos_explicados", "curva_calibracion",
                   "metricas_por_corte", "sesgo_monto_por_decil", "contexto_negocio",
                   "modelos_por_origen", "solape_valid_pool_test",
                   "solape_monto_valid_pool_test", "deriva_features"):
        assert (tmp_path / "reportes" / f"{nombre}.csv").exists()
    clas = res["clasificacion"].set_index("predictor")
    assert clas.loc["modelo", "auc"] > 0.65
    assert clas.loc["modelo", "auc"] > clas.loc["regla_negocio", "auc"]
    # el baseline trivial es UNA constante en todo el pool -> AUC exactamente 0,5 (m-1)
    assert clas.loc["tasa_base", "auc"] == pytest.approx(0.5)
    # la aceptacion se reporta con el rango por corte al lado del agregado (m-3 / m-4)
    assert "rango_por_corte" in res["aceptacion"].columns
    assert len(res["por_corte"]) == 6
    assert res["tiempos"]["total"] < 120


# ============================================================== 7. perfiles de cartera (ronda 2)
def _tasas_apv(tablas, anios=(2023, 2024, 2025)):
    """(tasa anual de aportantes, tasa de aporte mensual) promedio de los anios pedidos."""
    panel = tablas["panel_mensual"]
    panel = panel.assign(rim=panel["rim_emp1"] + panel["rim_emp2"], anio=panel["periodo"] // 100)
    ap = tablas["aportes_voluntarios"]
    ap = ap.assign(anio=ap["periodo"] // 100)
    tasas, mensuales = [], []
    for a in anios:
        cot = set(panel[(panel["anio"] == a) & (panel["rim"] > 0)]["afiliado_id"])
        x = ap[ap["anio"] == a]
        tasas.append(len(set(x["afiliado_id"]) & cot) / len(cot))
        mensuales.append(len(set(x[x["periodicidad"] == "MENSUAL"]["afiliado_id"]) & cot) / len(cot))
    return float(np.mean(tasas)), float(np.mean(mensuales))


@pytest.fixture(scope="module")
def tablas_cuprum():
    return generar_tablas(n_afiliados=5_000, seed=SEED, perfil="cuprum")


@pytest.fixture(scope="module")
def tablas_sistema_5k():
    return generar_tablas(n_afiliados=5_000, seed=SEED, perfil="sistema")


def test_perfil_cuprum_tiene_mas_renta_y_mas_apv(tablas_sistema_5k, tablas_cuprum):
    """El perfil Cuprum debe reproducir la cartera de la Ficha N 165: mas renta y mas APV."""
    def rim_media(t):
        p = t["panel_mensual"]
        r = p["rim_emp1"] + p["rim_emp2"]
        return float(r[r > 0].mean())

    ratio_renta = rim_media(tablas_cuprum) / rim_media(tablas_sistema_5k)
    # Ficha N 165: ingreso imponible promedio Cuprum / sistema = 2.331.030 / 1.346.859 = 1,73
    objetivo = (ANCLAS_FICHA_165["ingreso_imponible_promedio_cuprum"]
                / ANCLAS_FICHA_165["ingreso_imponible_promedio_sistema"])
    assert 0.85 * objetivo <= ratio_renta <= 1.15 * objetivo, ratio_renta

    t_sis, m_sis = _tasas_apv(tablas_sistema_5k)
    t_cup, m_cup = _tasas_apv(tablas_cuprum)
    # liderazgo en APV: la tasa de aportantes de Cuprum casi duplica la del sistema
    assert t_cup > 1.4 * t_sis, (t_sis, t_cup)
    assert m_cup > 1.4 * m_sis, (m_sis, m_cup)
    # y sigue bajo el techo implicito de la Ficha (0,319 cuentas APV en AFP por cotizante,
    # con ~la mitad de las cuentas activas en un anio -> el doble del techo como cota holgada)
    assert t_cup <= 2.0 * CUENTAS_APV_POR_COTIZANTE_SISTEMA, t_cup


def test_mezcla_regimen_a_b_calibrada_con_la_ficha(tablas_sistema_5k, tablas_cuprum):
    """76,7 % de los depositos en regimen A (Ficha N 165, nivel sistema)."""
    def share_a(t):
        ap = t["aportes_voluntarios"]
        apv = ap[ap["tipo"].isin(["APV_REGIMEN_A", "APV_REGIMEN_B"])]
        return float((apv["tipo"] == "APV_REGIMEN_A").mean())

    objetivo = ANCLAS_FICHA_165["share_depositos_regimen_a"]
    assert abs(share_a(tablas_sistema_5k) - objetivo) <= 0.06, share_a(tablas_sistema_5k)
    # en Cuprum el regimen B pesa mas porque la cartera esta en tramos mas altos del IGC
    assert share_a(tablas_cuprum) < share_a(tablas_sistema_5k)


def test_tabla_de_calibracion_contra_la_ficha(tablas_cuprum):
    cal = R.calibrar_contra_ficha(tablas_cuprum, "cuprum")
    assert {"indicador", "simulado", "oficial_ficha_165", "nivel_del_dato_oficial",
            "comentario"} <= set(cal.columns)
    assert len(cal) >= 6
    ind = cal.set_index("indicador")["oficial_ficha_165"]
    assert ind["ingreso imponible promedio de cotizantes (CLP)"] == (
        ANCLAS_FICHA_165["ingreso_imponible_promedio_cuprum"])


def test_los_dos_perfiles_corren_end_to_end_y_se_comparan(tmp_path):
    res = {}
    for perfil in ("sistema", "cuprum"):
        res[perfil] = R.ejecutar(n_afiliados=800, salida=tmp_path / perfil, seed=11,
                                 n_test=6, n_valid=6, cada_origen=3, diagnostico=False,
                                 perfil=perfil)
        assert (tmp_path / perfil / "reportes" / "calibracion_ficha_165.csv").exists()
    comp = R.comparar_perfiles(res)
    assert list(comp["perfil"]) == ["sistema", "cuprum"]
    for col in ("auc", "lift_decil_1", "wape_condicional", "captura_10pct",
                "tasa_anual_aportantes", "share_regimen_a", "ece"):
        assert col in comp.columns and comp[col].notna().all()
    # la cartera Cuprum tiene mas aportantes, asi que la tasa base del test es mayor
    assert comp.set_index("perfil").loc["cuprum", "tasa_base_test"] > \
        comp.set_index("perfil").loc["sistema", "tasa_base_test"]
