"""Tests del proyecto 05 (riesgo de empleador: DNP y pago tardio).

Minimo de CONVENCIONES §1: leakage (§3), forma del dataset y corrida end-to-end pequena; ademas
la convencion de fechas por fuente, los gaps entre todas las particiones, la coherencia de la
simulacion con `common.planilla` y las reglas de poblacion y target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.rim_nowcasting.calendario import add_months
from afp_lab.riesgo_empleador import evaluate as ev
from afp_lab.riesgo_empleador import features as F
from afp_lab.riesgo_empleador import model as M
from afp_lab.riesgo_empleador import run as R
from afp_lab.riesgo_empleador.synthetic import (COD_DNP, COD_SIN_DECLARAR, HORIZONTE_LARGO,
                                                LAG_INFINITO, MESES_DEUDA_BOLETIN, RUBRO_PUBLICO,
                                                generar_tablas, truncar_tablas)

N_AFILIADOS = 4_000
SEED = 7
CORTE_A = 202406
CORTE_B = 202511


@pytest.fixture(scope="module")
def tablas():
    return generar_tablas(n_afiliados=N_AFILIADOS, seed=SEED)


@pytest.fixture(scope="module")
def denso(tablas):
    return F.preparar_panel(tablas)


# ============================================================== 1. sin leakage
@pytest.mark.parametrize("corte", [CORTE_A, CORTE_B])
def test_features_no_dependen_de_informacion_posterior_al_corte(tablas, corte):
    """Quitar lo que no esta disponible en el corte no cambia ni una celda."""
    completas = F.construir_features(tablas, corte)
    truncadas = F.construir_features(truncar_tablas(tablas, corte), corte)
    pd.testing.assert_frame_equal(completas, truncadas)


@pytest.mark.parametrize("corte", [CORTE_A, CORTE_B])
def test_alterar_el_futuro_no_cambia_las_features(tablas, corte):
    """Mutar planillas no recibidas, boletines futuros y shocks no mueve nada."""
    base = F.construir_features(tablas, corte)
    alt = {k: v.copy() for k, v in tablas.items()}
    pl = alt["planillas"]
    no_recibida = (pl["periodo_recepcion"] <= 0) | (pl["periodo_recepcion"] > corte)
    pl.loc[no_recibida, "masa_imponible"] *= 100.0
    pl.loc[no_recibida, "estado"] = "DNP"
    pl.loc[no_recibida, "n_trabajadores"] = 9999
    # regularizaciones futuras
    pl.loc[pl["periodo_regularizacion"] > corte, "periodo_regularizacion"] = -1
    bol = alt["boletin_infractores"]
    alt["boletin_infractores"] = bol[bol["periodo"] <= corte].reset_index(drop=True)
    conc = alt["boletin_concursal"]
    conc.loc[conc["periodo_publicacion"] > corte, "tipo"] = "LIQUIDACION"
    alt["shocks_empleador"] = alt["shocks_empleador"].iloc[0:0]
    pd.testing.assert_frame_equal(base, F.construir_features(alt, corte))


def test_la_planilla_del_periodo_de_corte_no_es_visible(tablas, denso):
    """El target es la planilla del periodo `corte`: no puede filtrarse a las features."""
    corte = CORTE_A
    base = F.construir_features(tablas, corte, denso=denso)
    alt = {k: v.copy() for k, v in tablas.items()}
    pl = alt["planillas"]
    del_corte = pl["periodo"] >= corte
    pl.loc[del_corte, ["masa_imponible", "n_trabajadores"]] = 0.0
    pl.loc[del_corte, "estado"] = "SIN_DECLARAR"
    pl.loc[del_corte, ["periodo_recepcion", "periodo_regularizacion"]] = -1
    pl.loc[del_corte, "lag_recepcion"] = LAG_INFINITO
    pd.testing.assert_frame_equal(base, F.construir_features(alt, corte))


def test_ninguna_feature_es_latente(tablas, denso):
    f = F.construir_features(tablas, CORTE_A, denso=denso)
    cols = F.columnas_modelo(f.assign(y_dnp=0, y_tardio=0, y_dnp_3m=0, masa_en_riesgo=0.0,
                                      masa_periodo=0.0))
    assert not [c for c in cols if c.startswith("p_") or c.startswith("lat_")]
    assert not [c for c in cols if "estres" in c or "shock" in c]
    assert not set(F.COLUMNAS_ESCALA) & set(cols)
    # las tablas publicadas no traen los parametros de pago del empleador
    assert not [c for c in tablas["empleadores"].columns if c.startswith("p_")]


def test_gaps_entre_todas_las_particiones(tablas):
    cortes = F.cortes_disponibles(tablas, R.PRIMER_CORTE, HORIZONTE_LARGO)
    orig = F.origenes_moviles(cortes, n_test=12, n_valid=12, gap=HORIZONTE_LARGO,
                              horizonte=HORIZONTE_LARGO, cada=3)
    assert sum(len(o["test"]) for o in orig) == 12
    for o in orig:
        for antes, despues in ((o["train"], o["valid"]), (o["valid"], o["test"])):
            ultimo_target = add_months(max(antes), HORIZONTE_LARGO - 1)
            assert add_months(ultimo_target, HORIZONTE_LARGO) <= min(despues)
        assert set(o["train"]).isdisjoint(o["valid"])
    tam = [len(o["train"]) for o in orig]
    assert tam == sorted(tam)
    solape = F.solape_valid_pool_test(orig, HORIZONTE_LARGO)
    assert (solape["solape_con_su_propio_test"] == 0).all()


# ============================================================== 2. as-of verificado a mano
def test_as_of_calculado_a_mano_en_dos_casos(tablas, denso):
    """Recalcula a mano, desde `planillas`, dos empleadores de dos cortes distintos (§7)."""
    for corte in (CORTE_A, CORTE_B):
        f = F.construir_features(tablas, corte, denso=denso).set_index("empleador_id")
        pl = tablas["planillas"]
        meses = [add_months(corte, -k) for k in range(1, F.VENTANA + 1)]
        recibidas = pl[(pl["periodo"].isin(meses)) & (pl["periodo_recepcion"] > 0)
                       & (pl["periodo_recepcion"] <= corte)]
        imm = float(tablas["macro"].set_index("periodo").loc[corte, "imm"])
        for emp in list(f.index[:2]):
            sub = recibidas[recibidas["empleador_id"] == emp]
            assert f.loc[emp, "n_recibidas_12"] == pytest.approx(len(sub))
            assert f.loc[emp, "n_dnp_12"] == pytest.approx(
                int(sub["estado"].isin(["DNP", "SIN_DECLARAR"]).sum()))
            assert f.loc[emp, "tasa_puntualidad_12"] == pytest.approx(
                (sub["lag_recepcion"] <= 1).sum() / max(len(sub), 1), rel=1e-5)
            if len(sub):
                ult = sub.sort_values("periodo").iloc[-1]
                assert f.loc[emp, "masa_ref_imm"] == pytest.approx(
                    ult["masa_imponible"] / imm, rel=1e-4)
                assert f.loc[emp, "n_trabajadores"] == pytest.approx(ult["n_trabajadores"])


# ============================================================== 3. poblacion y target
def test_poblacion_y_target(tablas, denso):
    corte = CORTE_A
    f = F.construir_features(tablas, corte, denso=denso)
    t = F.construir_target(tablas, corte, denso=denso)
    pl = tablas["planillas"]
    meses = [add_months(corte, -k) for k in range(1, F.VENTANA + 1)]
    con_planilla = set(pl[(pl["periodo"].isin(meses)) & (pl["periodo_recepcion"] > 0)
                          & (pl["periodo_recepcion"] <= corte)]["empleador_id"])
    assert set(f["empleador_id"]) <= con_planilla
    assert (f["n_trabajadores"] >= 1).all()

    # el target es exactamente el estado de la planilla del periodo de corte
    del_corte = pl[pl["periodo"] == corte].set_index("empleador_id")
    tt = t[t["aplica"]].set_index("empleador_id")
    comunes = tt.index.intersection(del_corte.index)
    assert len(comunes) == len(tt)
    esperado = del_corte.loc[comunes, "estado"].isin(["DNP", "SIN_DECLARAR"]).astype(int)
    assert (tt.loc[comunes, "y_dnp"].to_numpy() == esperado.to_numpy()).all()
    assert (tt.loc[comunes, "y_tardio"].to_numpy()
            == (del_corte.loc[comunes, "lag_recepcion"] >= 2).astype(int).to_numpy()).all()
    # masa en riesgo consistente
    assert (tt.loc[comunes, "masa_periodo"].to_numpy()
            == pytest.approx(del_corte.loc[comunes, "masa_imponible"].to_numpy()))


def test_forma_del_dataset(tablas, denso):
    cortes = [CORTE_A, CORTE_B]
    d = F.construir_dataset(tablas, cortes, denso=denso)
    assert not d.duplicated(["empleador_id", "periodo_corte"]).any()
    assert set(d["periodo_corte"].unique()) == set(cortes)
    for c in ("y_dnp", "y_tardio", "y_dnp_3m"):
        assert d[c].isin([0, 1]).all()
    assert (d["masa_periodo"] > 0).all()
    assert np.allclose(d["masa_en_riesgo"], d["y_dnp"] * d["masa_periodo"])
    assert not d[F.columnas_modelo(d)].isna().any().any()
    for c in F.CATEGORICAS:
        assert str(d[c].dtype) == "category"
    # el target a 3 meses contiene al de 1 mes
    assert (d["y_dnp_3m"] >= d["y_dnp"]).all()


# ============================================================== 4. simulacion coherente
def test_determinismo(tablas):
    otras = generar_tablas(n_afiliados=N_AFILIADOS, seed=SEED)
    for k in tablas:
        pd.testing.assert_frame_equal(tablas[k], otras[k])


def test_tasas_objetivo_del_brief(tablas):
    """3-6 % de las planillas del mes con deuda y 5-10 % con recepcion tardia."""
    pl = tablas["planillas"]
    v = pl["estado"].value_counts(normalize=True)
    deuda = float(v.get("DNP", 0.0) + v.get("SIN_DECLARAR", 0.0))
    tardia = float((pl["lag_recepcion"] >= 2).mean())
    assert 0.03 <= deuda <= 0.07, deuda
    assert 0.04 <= tardia <= 0.11, tardia


def test_el_sector_publico_nunca_cae_en_dnp(tablas):
    emp = tablas["empleadores"]
    publicos = set(emp.loc[emp["rubro"] == RUBRO_PUBLICO, "empleador_id"])
    pl = tablas["planillas"]
    sub = pl[pl["empleador_id"].isin(publicos)]
    assert len(sub) > 0
    assert not sub["estado"].isin(["DNP", "SIN_DECLARAR"]).any()


def test_persistencia_y_estacionalidad_de_la_dnp(tablas):
    """La DNP se repite (persistencia) y sube en diciembre-marzo (estacionalidad)."""
    pl = tablas["planillas"].sort_values(["empleador_id", "periodo"]).copy()
    pl["deuda"] = pl["estado"].isin(["DNP", "SIN_DECLARAR"]).astype(int)
    pl["deuda_prev"] = pl.groupby("empleador_id")["deuda"].shift()
    d = pl.dropna(subset=["deuda_prev"])
    p_repite = d.loc[d["deuda_prev"] == 1, "deuda"].mean()
    p_nuevo = d.loc[d["deuda_prev"] == 0, "deuda"].mean()
    assert p_repite > 3 * p_nuevo, (p_repite, p_nuevo)
    pl["mes"] = pl["periodo"] % 100
    alta = pl[pl["mes"].isin([12, 1, 2, 3])]["deuda"].mean()
    baja = pl[~pl["mes"].isin([12, 1, 2, 3])]["deuda"].mean()
    assert alta > baja, (alta, baja)


def test_boletin_infractores_solo_tras_meses_de_deuda(tablas):
    """Nadie entra al boletin sin haber acumulado `MESES_DEUDA_BOLETIN` meses de deuda."""
    bol = tablas["boletin_infractores"]
    assert len(bol) > 0
    pl = tablas["planillas"]
    deuda = pl[pl["estado"].isin(["DNP", "SIN_DECLARAR"])]
    con_deuda = set(deuda["empleador_id"])
    assert set(bol["empleador_id"]) <= con_deuda
    # y cada empleador del boletin tiene al menos dos meses de deuda consecutivos
    d = pl.sort_values(["empleador_id", "periodo"]).copy()
    d["deuda"] = d["estado"].isin(["DNP", "SIN_DECLARAR"]).astype(int)
    d["prev"] = d.groupby("empleador_id")["deuda"].shift().fillna(0)
    dos = set(d.loc[(d["deuda"] == 1) & (d["prev"] == 1), "empleador_id"])
    assert set(bol["empleador_id"]) <= dos


def test_coherencia_con_la_planilla_de_common(tablas):
    """La masa de `planillas` cuadra con la RIM del panel de los mismos trabajadores."""
    panel = tablas["panel_mensual"]
    rim = (panel[panel["empleador_id"].notna() & (panel["rim_emp1"] > 0)]
           .groupby(["empleador_id", "periodo"], observed=True)["rim_emp1"].sum())
    pl = tablas["planillas"].set_index(["empleador_id", "periodo"])["masa_imponible"]
    comunes = pl.index.intersection(rim.index)
    assert len(comunes) > 100
    # la planilla suma tambien la gratificacion y el aguinaldo del panel, asi que la masa es >=
    assert (pl.loc[comunes].to_numpy() >= rim.loc[comunes].to_numpy() - 1.0).all()


# ============================================================== 5. modelo y baselines
@pytest.fixture(scope="module")
def entrenado(tablas, denso):
    cortes = F.cortes_disponibles(tablas, R.PRIMER_CORTE, HORIZONTE_LARGO)
    orig = F.origenes_moviles(cortes, n_test=3, n_valid=6, gap=HORIZONTE_LARGO,
                              horizonte=HORIZONTE_LARGO, cada=3)
    o = orig[-1]
    usados = o["train"][-12:] + o["valid"] + o["test"]
    datos = F.construir_dataset(tablas, usados, denso=denso)
    tr = datos[datos["periodo_corte"].isin(o["train"][-12:])]
    va = datos[datos["periodo_corte"].isin(o["valid"])]
    te = datos[datos["periodo_corte"].isin(o["test"])]
    return M.entrenar(tr, va, M.ParametrosModelo(n_estimators=150, early_stopping=25)), tr, va, te


def test_hiperparametros_se_eligen_con_validacion(entrenado):
    modelo, _, _, _ = entrenado
    assert 1 <= modelo.n_arboles() <= modelo.params.n_estimators
    t = M.TARGET_PRINCIPAL
    esperado = (modelo.mejora_media[t] > M.N_SIGMAS_CALIBRACION * modelo.mejora_sem[t]
                and modelo.mejora_media[t] >= M.MEJORA_MINIMA_CALIBRACION * modelo.brier_valid_crudo[t])
    assert modelo.usa_calibracion[t] == esperado


def test_prediccion_y_score_de_negocio(entrenado):
    modelo, _, _, te = entrenado
    p = M.predecir(modelo, te)
    assert len(p) == len(te)
    for t in M.TARGETS:
        assert p[f"prob_{t}"].between(0, 1).all()
    assert np.allclose(p["masa_en_riesgo_esperada"],
                       p[f"prob_{M.TARGET_PRINCIPAL}"] * te["masa_ref_clp"].to_numpy())


def test_baselines_y_metricas(entrenado):
    modelo, tr, _, te = entrenado
    pred = {"modelo": M.predecir(modelo, te),
            "persistencia": M.baseline_persistencia(tr, te),
            "regla_negocio": M.baseline_regla(tr, te),
            "tasa_base": M.baseline_tasa_base(tr, te)}
    tabla = ev.tabla_clasificacion(te, pred).set_index("predictor")
    assert tabla.loc["tasa_base", "auc"] == pytest.approx(0.5)
    assert tabla.loc["modelo", "auc"] > tabla.loc["persistencia", "auc"]
    assert tabla.loc["modelo", "auc"] > tabla.loc["regla_negocio", "auc"]
    neg = ev.metrica_negocio(te.reset_index(drop=True),
                             {"modelo_masa_esperada": pred["modelo"]["masa_en_riesgo_esperada"].to_numpy(),
                              "regla_negocio": pred["regla_negocio"]["masa_en_riesgo_esperada"].to_numpy()})
    assert neg["captura_masa_en_riesgo"].between(0, 1).all()
    por_corte = ev.metricas_por_corte(te, pred["modelo"])
    assert len(por_corte) == te["periodo_corte"].nunique()
    agg = ev.error_agregado_mensual(por_corte)
    assert {"meses_dentro_de_tolerancia", "meses_dentro_de_1_sigma_intrinseca",
            "volatilidad_intrinseca_media_pct"} <= set(agg.columns)
    assert len(M.importancias(modelo)) == len(M.TARGETS) * len(modelo.columnas)
    assert len(M.explicar(modelo, te, n_ejemplos=3)["caso"].unique()) == 3


def test_persistencia_del_modelo(entrenado, tmp_path):
    modelo, _, _, te = entrenado
    M.guardar(modelo, tmp_path / "m")
    recargado = M.cargar(tmp_path / "m")
    assert (tmp_path / "m" / "calibracion.json").exists()
    pd.testing.assert_frame_equal(M.predecir(modelo, te), M.predecir(recargado, te))


# ============================================================== 6. end-to-end
def test_corrida_end_to_end_pequena(tmp_path):
    res = R.ejecutar(n_afiliados=3_000, salida=tmp_path, seed=11, n_test=6, n_valid=6,
                     cada_origen=3, diagnostico=False)
    dirs = sorted((tmp_path / "modelo").glob("origen_*"))
    assert len(dirs) == len(res["origenes"])
    assert (tmp_path / "predicciones_test.parquet").exists()
    for nombre in ("metricas_clasificacion", "metricas_por_target", "metricas_por_corte",
                   "error_agregado_mensual", "metrica_negocio", "aceptacion", "importancias",
                   "ejemplos_explicados", "curva_calibracion", "resumen_simulacion",
                   "modelos_por_origen", "solape_valid_pool_test"):
        assert (tmp_path / "reportes" / f"{nombre}.csv").exists()
    clas = res["clasificacion"].set_index("predictor")
    assert clas.loc["modelo", "auc"] > 0.70
    assert clas.loc["tasa_base", "auc"] == pytest.approx(0.5)
    assert len(res["por_corte"]) == 6
    assert set(res["targets"]["target"]) == set(M.TARGETS)
    assert res["tiempos"]["total"] < 120
