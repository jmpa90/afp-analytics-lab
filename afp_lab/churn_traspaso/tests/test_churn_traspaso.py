"""Tests del proyecto 02 (propension a traspaso).

Cubre el minimo exigido por CONVENCIONES §1 y §3:
  * leakage: alterar o borrar cualquier registro posterior al corte no cambia las features;
  * equivalencia entre la ruta de un corte y la ruta matricial de muchos cortes;
  * verificacion "a mano" de dos familias de features as-of (RIM con desfase 2 y contactos 3 m);
  * forma del dataset, definicion del target, poblacion y exclusiones;
  * split temporal con gap; baselines y metrica de negocio; corrida end-to-end pequena.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx
from afp_lab.churn_traspaso import HORIZONTE_MESES, MESES_DESFASE_RIM
from afp_lab.churn_traspaso import evaluate as E
from afp_lab.churn_traspaso import features as F
from afp_lab.churn_traspaso import model as M
from afp_lab.churn_traspaso.run import main as run_main
from afp_lab.churn_traspaso.synthetic import AFP_PROPIA, generar_tablas

N_TEST = 1_200
SEED = 42
CORTE = 202409          # corte de referencia para los tests as-of


@pytest.fixture(scope="module")
def tablas() -> dict[str, pd.DataFrame]:
    return generar_tablas(n_afiliados=N_TEST, seed=SEED)


@pytest.fixture(scope="module")
def dataset(tablas) -> pd.DataFrame:
    cortes = F.cortes_evaluables(tablas)
    return F.construir_dataset(tablas, cortes[-12:])


# --------------------------------------------------------------------- datos sinteticos
def test_generacion_determinista():
    a = generar_tablas(n_afiliados=300, seed=7)
    b = generar_tablas(n_afiliados=300, seed=7)
    for nombre in ("saldos", "traspasos", "contactos", "reclamos", "uso_digital", "mercado"):
        pd.testing.assert_frame_equal(a[nombre], b[nombre])


def test_tablas_coherentes_con_la_historia_laboral(tablas):
    tr = tablas["traspasos"]
    salidas = tr[tr["tipo"] == "SALIDA"]
    # el traspaso se solicita en un mes y es efectivo al mes siguiente
    assert (salidas["periodo_efectivo"].map(periodo_to_idx)
            - salidas["periodo_solicitud"].map(periodo_to_idx) == 1).all()
    assert set(salidas["afp_origen"]) == {AFP_PROPIA}
    assert AFP_PROPIA not in set(salidas["afp_destino"])
    # nadie tiene saldo despues de irse
    ult_saldo = tablas["saldos"].groupby("afiliado_id")["periodo"].max()
    sol = salidas.groupby("afiliado_id")["periodo_solicitud"].min()
    comun = ult_saldo.index.intersection(sol.index)
    assert (ult_saldo.loc[comun].map(periodo_to_idx) <= sol.loc[comun].map(periodo_to_idx)).all()
    # la tasa anualizada de traspaso queda en el rango pedido por el brief (4-7 % de la cartera)
    cartera = tablas["saldos"].groupby("periodo")["afiliado_id"].count()
    por_mes = salidas.groupby("periodo_solicitud")["afiliado_id"].count().reindex(cartera.index).fillna(0)
    assert 0.03 <= float((por_mes / cartera).mean() * 12) <= 0.08


# --------------------------------------------------------------------- leakage
def _mutar_futuro(tablas: dict[str, pd.DataFrame], corte: int, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Borra o altera TODO lo posterior al corte (y al corte-2 en el panel laboral)."""
    rng = np.random.default_rng(seed)
    out = dict(tablas)
    corte_lab = add_months(corte, -MESES_DESFASE_RIM)

    pan = tablas["panel_mensual"].copy()
    fut = pan["periodo"] > corte_lab
    pan.loc[fut, "rim_emp1"] = pan.loc[fut, "rim_emp1"] * 17.0 + 1_000_000
    pan.loc[fut, "estado"] = "RETIRADO"
    pan.loc[fut, "empleador_id"] = 0
    out["panel_mensual"] = pan.drop(index=pan.index[fut][::2])   # ademas borra la mitad

    for nombre in ("saldos", "uso_digital", "contactos", "reclamos", "cambios_fondo", "mercado"):
        d = tablas[nombre].copy()
        out[nombre] = d[d["periodo"] <= corte]

    dig = out["uso_digital"]
    extra = dig[dig["periodo"] == corte].copy()
    extra["periodo"] = add_months(corte, 1)
    extra["logins_app"] = 99
    extra["descargas_certificado"] = 1
    out["uso_digital"] = pd.concat([dig, extra], ignore_index=True)

    tr = tablas["traspasos"].copy()
    falsos = tr.head(50).copy()
    falsos["periodo_solicitud"] = add_months(corte, 1)
    falsos["periodo_efectivo"] = add_months(corte, 2)
    falsos["tipo"] = "SALIDA"
    falsos["afiliado_id"] = rng.choice(tablas["afiliados"]["afiliado_id"].to_numpy(), len(falsos))
    out["traspasos"] = pd.concat([tr, falsos], ignore_index=True)
    return out


def test_sin_leakage_alterar_el_futuro_no_cambia_las_features(tablas):
    base = F.construir_features(tablas, CORTE)
    mutado = F.construir_features(_mutar_futuro(tablas, CORTE), CORTE)
    assert len(base) == len(mutado) > 0
    cols = sorted(F.columnas_features(base))
    izq = base.sort_values("afiliado_id")[["afiliado_id"] + cols].reset_index(drop=True)
    der = mutado.sort_values("afiliado_id")[["afiliado_id"] + cols].reset_index(drop=True)
    for c in ["afiliado_id"] + cols:
        a, b = izq[c], der[c]
        if str(a.dtype) == "category" or a.dtype == object:
            assert (a.astype(str) == b.astype(str)).all(), f"feature con leakage: {c}"
        else:
            np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                                       rtol=1e-6, atol=1e-6, err_msg=f"feature con leakage: {c}")


def test_features_multi_igual_a_corte_unico(tablas):
    """La ruta matricial de muchos cortes debe dar exactamente lo mismo que recortar y calcular uno."""
    cortes = [202312, CORTE]
    multi = F.construir_features_multi(tablas, cortes)
    for corte in cortes:
        uno = F.construir_features(tablas, corte).sort_values("afiliado_id").reset_index(drop=True)
        sub = multi[multi["periodo_corte"] == corte].sort_values("afiliado_id").reset_index(drop=True)
        assert len(uno) == len(sub) > 0
        for c in F.columnas_features(uno):
            a, b = uno[c], sub[c]
            if str(a.dtype) == "category" or a.dtype == object:
                assert (a.astype(str) == b.astype(str)).all(), f"{c} difiere en {corte}"
            else:
                np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                                           rtol=1e-6, atol=1e-6, err_msg=f"{c} difiere en {corte}")


def test_features_asof_verificadas_a_mano(tablas):
    """Dos casos verificados a mano (checklist §7): RIM con desfase 2 y contactos de 3 meses."""
    df = F.construir_features(tablas, CORTE).set_index("afiliado_id")
    corte_lab = add_months(CORTE, -MESES_DESFASE_RIM)
    panel = tablas["panel_mensual"]
    con = tablas["contactos"]

    # caso 1: rim_l1 es la RIM del mes corte-2, no la del corte
    fila_panel = panel[(panel["periodo"] == corte_lab) & (panel["afiliado_id"].isin(df.index))]
    fila_panel = fila_panel[fila_panel["rim_emp1"] > 0].iloc[0]
    af = int(fila_panel["afiliado_id"])
    esperado = float(fila_panel["rim_emp1"] + fila_panel["rim_emp2"])
    assert df.loc[af, "rim_l1"] == pytest.approx(esperado, rel=1e-5)
    rim_corte = panel[(panel["afiliado_id"] == af) & (panel["periodo"] == CORTE)]
    if len(rim_corte) and float(rim_corte["rim_emp1"].iloc[0]) != esperado:
        assert df.loc[af, "rim_l1"] != pytest.approx(float(rim_corte["rim_emp1"].iloc[0]), rel=1e-5)

    # caso 2: contactos_vendedor_competencia_3m = contactos del canal en [corte-2, corte]
    ventana = [add_months(CORTE, -2), add_months(CORTE, -1), CORTE]
    esperado_con = (con[(con["canal"] == "VENDEDOR_COMPETENCIA") & (con["periodo"].isin(ventana))]
                    .groupby("afiliado_id").size())
    obs = df["contactos_vendedor_competencia_3m"]
    for af in esperado_con.index[:5]:
        if af in obs.index:
            assert obs.loc[af] == pytest.approx(float(esperado_con.loc[af]))
    assert obs.sum() == pytest.approx(float(esperado_con.reindex(obs.index).fillna(0).sum()))


# --------------------------------------------------------------------- dataset y target
def test_forma_del_dataset(dataset):
    assert len(dataset) > 0
    assert not dataset.duplicated(["afiliado_id", "periodo_corte"]).any()
    assert set(["y", "afiliado_id", "periodo_corte", "saldo_obligatorio", "rim_ref"]) <= set(dataset.columns)
    assert dataset["y"].isin([0, 1]).all()
    assert 0.001 < dataset["y"].mean() < 0.15
    cols = F.columnas_features(dataset)
    assert len(cols) >= 40
    assert not any(c.startswith("lat_") or c.startswith("p_") for c in cols), "feature latente en el dataset"
    assert "periodo_solicitud_target" not in cols and "y" not in cols


def test_target_es_solicitud_en_la_ventana(tablas, dataset):
    tr = tablas["traspasos"]
    sal = tr[tr["tipo"] == "SALIDA"].sort_values("periodo_solicitud").drop_duplicates("afiliado_id")
    sol = sal.set_index("afiliado_id")["periodo_solicitud"]
    d = dataset[dataset["afiliado_id"].isin(sol.index)].copy()
    d["idx_sol"] = d["afiliado_id"].map(sol).map(periodo_to_idx)
    d["idx_corte"] = d["periodo_corte"].map(periodo_to_idx)
    dentro = (d["idx_sol"] > d["idx_corte"]) & (d["idx_sol"] <= d["idx_corte"] + HORIZONTE_MESES)
    assert (d["y"].to_numpy() == dentro.to_numpy().astype(int)).all()
    # los que nunca se traspasan tienen y = 0
    sin = dataset[~dataset["afiliado_id"].isin(sol.index)]
    assert sin["y"].sum() == 0


def test_poblacion_excluye_retirados_y_traspasos_ya_solicitados(tablas, dataset):
    assert (dataset["estado_laboral"].astype(str) != "RETIRADO").all()
    tr = tablas["traspasos"]
    sal = tr[tr["tipo"] == "SALIDA"].groupby("afiliado_id")["periodo_solicitud"].min()
    d = dataset[dataset["afiliado_id"].isin(sal.index)]
    assert (d["periodo_corte"].map(periodo_to_idx) < d["afiliado_id"].map(sal).map(periodo_to_idx)).all()
    # solo afiliados vigentes: existe fila de saldo en ese mes
    vig = set(map(tuple, tablas["saldos"][["afiliado_id", "periodo"]].to_numpy()))
    muestra = dataset.sample(300, random_state=0)[["afiliado_id", "periodo_corte"]].to_numpy()
    assert all(tuple(x) in vig for x in muestra)


def test_split_temporal_respeta_el_gap(tablas):
    cortes = F.cortes_evaluables(tablas)
    sp = F.split_temporal(cortes, n_test=3, n_valid=3)
    assert len(sp["test"]) == 3 and len(sp["valid"]) == 3 and len(sp["train"]) > 5
    ult_target_train = add_months(max(sp["train"]), HORIZONTE_MESES)
    ult_target_valid = add_months(max(sp["valid"]), HORIZONTE_MESES)
    assert periodo_to_idx(min(sp["valid"])) - periodo_to_idx(ult_target_train) >= HORIZONTE_MESES
    assert periodo_to_idx(min(sp["test"])) - periodo_to_idx(ult_target_valid) >= HORIZONTE_MESES
    assert max(sp["train"]) < min(sp["valid"]) < min(sp["test"])


# --------------------------------------------------------------------- modelo y metricas
def test_modelo_supera_baselines_y_persiste(dataset, tmp_path):
    cortes = sorted(dataset["periodo_corte"].unique())
    tr = dataset[dataset["periodo_corte"].isin(cortes[:-4])]
    va = dataset[dataset["periodo_corte"] == cortes[-4]]
    te = dataset[dataset["periodo_corte"].isin(cortes[-3:])].copy()
    modelo = M.entrenar(tr, va, seed=SEED, n_arboles=120)
    prob = M.predecir(modelo, te)
    preds = {"modelo": prob,
             "tasa_base": M.baseline_tasa_base(tr, te),
             "regla": M.baseline_regla_negocio(te)}
    tabla = E.tabla_comparativa(te, preds).set_index("predictor")
    assert tabla.loc["modelo", "auc"] > tabla.loc["regla", "auc"] > 0.5
    assert tabla.loc["tasa_base", "auc"] == pytest.approx(0.5)
    cap = E.captura_top(te, preds, fracciones=(0.10,)).set_index("predictor")
    assert cap.loc["modelo", "pct_traspasos_capturados"] > cap.loc["regla", "pct_traspasos_capturados"]

    M.guardar_modelo(modelo, tmp_path / "modelo")
    recargado = M.cargar_modelo(tmp_path / "modelo")
    np.testing.assert_allclose(prob, M.predecir(recargado, te), rtol=1e-9, atol=1e-12)
    assert len(M.importancias(modelo, 10)) == 10
    assert len(M.explicar_casos(modelo, te, prob, n_casos=3, n_features=4)) == 12


def test_calibrador_platt_no_altera_el_orden():
    rng = np.random.default_rng(0)
    p = rng.random(5000) * 0.3
    y = (rng.random(5000) < p).astype(int)
    cal = M.CalibradorPlatt().fit(p, y)
    q = cal.predict(p)
    assert np.array_equal(np.argsort(p), np.argsort(q))
    assert 0.0 <= q.min() and q.max() <= 1.0


def test_metrica_de_negocio_captura():
    df = pd.DataFrame({"y": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       "saldo_obligatorio": [100.0, 50.0] + [10.0] * 8})
    perfecto = np.array([9, 8, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    cap = E.captura_top(df, {"perfecto": perfecto}, fracciones=(0.2,)).iloc[0]
    assert cap["pct_traspasos_capturados"] == pytest.approx(1.0)
    assert cap["pct_saldo_riesgo_capturado"] == pytest.approx(1.0)
    riesgo = E.saldo_en_riesgo(df, np.full(10, 0.2))
    assert riesgo["saldo_riesgo_real_mm"].iloc[0] == pytest.approx(150 / 1e6)


def test_segmentos_y_calibracion(dataset):
    d = E.agregar_segmentos(dataset.head(4000))
    for c in ("tramo_saldo", "tramo_edad", "tramo_renta"):
        assert d[c].nunique() >= 2
    curva = E.curva_calibracion(d["y"].to_numpy(), np.clip(d["y"] * 0.5 + 0.01, 0, 1).to_numpy(), n_bins=5)
    assert len(curva) == 5 and curva["n"].sum() == len(d)


# --------------------------------------------------------------------- end to end
def test_run_end_to_end(tmp_path):
    assert run_main(["--n-afiliados", "900", "--salida", str(tmp_path), "--sin-fuentes"]) == 0
    rep = tmp_path / "reportes"
    for f in ("metricas_test.csv", "captura_negocio.csv", "importancias.csv", "calibracion.csv",
              "saldo_en_riesgo.csv", "split.csv", "tiempos.csv", "casos_explicados.csv"):
        assert (rep / f).exists(), f
    assert (tmp_path / "modelo" / "lightgbm.txt").exists()
    met = pd.read_csv(rep / "metricas_test.csv").set_index("predictor")
    assert met.loc["modelo_lgbm", "auc"] > met.loc["baseline_regla_negocio", "auc"]
