"""Tests del proyecto 02 (propension a traspaso).

Cubre el minimo exigido por CONVENCIONES §1 y §3:
  * leakage: alterar o borrar cualquier registro posterior al corte no cambia las features;
  * equivalencia entre la ruta de un corte y la ruta matricial de muchos cortes;
  * verificacion "a mano" de dos familias de features as-of (RIM con desfase 2 y contactos 3 m);
  * forma del dataset, definicion del target, poblacion y exclusiones;
  * split temporal con gap; baselines y metrica de negocio; corrida end-to-end pequena.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx
from afp_lab.churn_traspaso import HORIZONTE_MESES, MESES_DESFASE_RIM
from afp_lab.churn_traspaso import evaluate as E
from afp_lab.churn_traspaso import features as F
from afp_lab.churn_traspaso import model as M
from afp_lab.churn_traspaso.run import main as run_main
from afp_lab.churn_traspaso.synthetic import (AFP_PROPIA, AFP_TODAS, FONDOS, TIPOS_RECLAMO,
                                             generar_tablas, resumen_simulacion,
                                             tabla_parametros, tabla_parametros_markdown)

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


def test_mercado_y_olas_de_traspaso_plausibles(tablas):
    """Ronda 2, bloqueante 2: la dispersion de rentabilidad entre AFP y la amplitud de las olas."""
    r = resumen_simulacion(tablas).set_index("indicador")["valor"]
    # en Chile la diferencia de rentabilidad 12 m entre AFP es de 1-2 pp, no de 6
    assert 0.5 <= r["rango_rentabilidad_12m_entre_afp_mediana_pp"] <= 3.0
    assert r["rango_rentabilidad_12m_entre_afp_p95_pp"] <= 5.0
    # la tasa sobre la cartera cotizante debe quedar en el rango del brief (4-7 %)
    assert 0.035 <= r["tasa_traspaso_anualizada_cotizantes"] <= 0.075, r["tasa_traspaso_anualizada_cotizantes"]
    # y la tasa movil de 12 meses no puede oscilar entre 1,6 % y 12,6 % como en la ronda 1
    assert 0.02 <= r["tasa_traspaso_12m_min"], r["tasa_traspaso_12m_min"]
    assert r["tasa_traspaso_12m_max"] <= 0.09, r["tasa_traspaso_12m_max"]
    assert r["tasa_traspaso_12m_max"] / max(r["tasa_traspaso_12m_min"], 1e-9) <= 2.5


def test_castigo_de_rentabilidad_domina_su_ventana(tablas):
    """El castigo documentado (-1 pp anual) debe verse por sobre el ruido del AR(1)."""
    m = tablas["mercado"]
    par = tabla_parametros().set_index("parametro")["valor"]
    brechas = []
    for f in FONDOS:
        piv = m.pivot(index="periodo", columns="afp", values=f"rent_12m_{f}")
        brechas.append(piv[AFP_PROPIA] - piv[list(AFP_TODAS)].mean(axis=1))
    br = pd.concat(brechas, axis=1).mean(axis=1)
    dentro = br[(br.index >= add_months(int(par["castigo_desde"]), 6))
                & (br.index <= int(par["castigo_hasta"]))]
    fuera = br[br.index < int(par["castigo_desde"])]
    assert dentro.mean() < fuera.mean() - 0.4, (dentro.mean(), fuera.mean())
    assert dentro.mean() < -0.3
    assert fuera.mean() > -0.2


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

    for nombre in ("uso_digital", "contactos", "reclamos", "cambios_fondo", "mercado"):
        d = tablas[nombre].copy()
        fut = d["periodo"] > corte
        d = d[~fut]                     # borra el futuro operacional...
        out[nombre] = d
    # ... y ademas inyecta registros futuros extremos, que tampoco pueden mover nada
    dig = out["uso_digital"]
    extra = dig[dig["periodo"] == corte].copy()
    extra["periodo"] = add_months(corte, 1)
    extra["logins_app"] = 99
    extra["descargas_certificado"] = 1
    out["uso_digital"] = pd.concat([dig, extra], ignore_index=True)
    sal = tablas["saldos"].copy()
    fut = sal["periodo"] > corte
    sal.loc[fut, "saldo_obligatorio"] = 0.0
    sal.loc[fut, "antiguedad_afp_meses"] = 0
    out["saldos"] = sal.drop(index=sal.index[fut][::3])

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


def test_sin_leakage_en_la_ruta_matricial(tablas):
    """Menor 15 de la revision: el test anterior recorta antes de calcular y es tautologico.

    Aqui se muta el futuro y se recalcula con `construir_features_multi`, que NO recorta nada y es
    la ruta que usa el `run`. El corte evaluado no es el ultimo, asi que hay futuro real que mutar.
    """
    cortes = [202312, CORTE]
    base = F.construir_features_multi(tablas, cortes)
    mutado = F.construir_features_multi(_mutar_futuro(tablas, CORTE), cortes)
    base = base[base["periodo_corte"] == CORTE].sort_values("afiliado_id").reset_index(drop=True)
    mutado = mutado[mutado["periodo_corte"] == CORTE].sort_values("afiliado_id").reset_index(drop=True)
    assert len(base) == len(mutado) > 0
    for c in F.columnas_features(base):
        a, b = base[c], mutado[c]
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


def test_split_temporal_purga_y_guardas(tablas):
    cortes = F.cortes_evaluables(tablas)
    sp = F.split_temporal(cortes, n_test=3, n_valid=3)
    assert len(sp["test"]) == 3 and len(sp["valid"]) == 3 and len(sp["train"]) > 5
    assert max(sp["train"]) < min(sp["valid"]) < min(sp["test"])
    # purga: ningun target de una particion alcanza el primer corte de la siguiente
    assert add_months(max(sp["train"]), HORIZONTE_MESES) < min(sp["valid"])
    assert add_months(max(sp["valid"]), HORIZONTE_MESES) < min(sp["test"])
    # CONVENCIONES §3: gap >= horizonte entre el ultimo target de train y el primer corte de test
    gap = periodo_to_idx(min(sp["test"])) - periodo_to_idx(add_months(max(sp["train"]), HORIZONTE_MESES))
    assert gap >= HORIZONTE_MESES
    # la validacion queda pegada al test (ronda 1 la dejaba 26 meses antes)
    assert periodo_to_idx(min(sp["test"])) - periodo_to_idx(max(sp["valid"])) <= HORIZONTE_MESES + 2
    # particiones disjuntas y descartados explicitos
    usados = sp["train"] + sp["valid"] + sp["test"]
    assert len(set(usados)) == len(usados)
    assert set(sp["descartados"]) == set(cortes) - set(usados)
    # guardas (menor 8): sin cortes suficientes falla con mensaje, no con IndexError ni en silencio
    with pytest.raises(ValueError):
        F.split_temporal(cortes[:8])


def test_cortes_sin_historia_fallan_explicitamente(tablas):
    """Menor 9: un corte sin historia laboral suficiente no se descarta en silencio."""
    primero = int(tablas["saldos"]["periodo"].min())
    with pytest.raises(ValueError, match="sin historia"):
        F.construir_features_multi(tablas, [primero])


def test_catalogo_fijo_de_tipos_de_reclamo(tablas):
    """Menor 10: las categorias no dependen del slice que toque a cada corte."""
    a = F.construir_features(tablas, 202312)
    b = F.construir_features(tablas, CORTE)
    esperado = ["SIN_RECLAMO"] + list(TIPOS_RECLAMO)
    assert list(a["tipo_ultimo_reclamo"].cat.categories) == esperado
    assert list(b["tipo_ultimo_reclamo"].cat.categories) == esperado
    assert list(a["fondo"].cat.categories) == ["SIN_FONDO"] + list(FONDOS)


def test_meses_hasta_pension_negativo_es_intencional(dataset):
    """Menor 14: el valor negativo esta documentado (cotizantes pasada la edad legal)."""
    neg = dataset[dataset["meses_hasta_pension"] < 0]
    if len(neg):
        assert (neg["edad"] >= 60).all()
    doc = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "meses_hasta_pension" in doc and "negativ" in doc.lower()


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
    # la regla del brief es debil y con pocos afiliados puede quedar bajo 0,5; lo exigible es que el
    # modelo le gane a ambos baselines en AUC y en la metrica de negocio
    assert tabla.loc["modelo", "auc"] > max(tabla.loc["regla", "auc"], 0.5)
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


def test_columnas_features_excluye_latentes_de_forma_dura(dataset):
    """Menor 7 de la ronda 2: aunque un diagnostico pegue una columna latente, el modelo no la ve."""
    d = dataset.copy()
    d["lat_u_traspaso"] = 1.0
    d["p_dnp"] = 0.5
    cols = F.columnas_features(d)
    assert "lat_u_traspaso" not in cols and "p_dnp" not in cols
    assert not any(c.startswith(("lat_", "p_")) for c in cols)
    assert len(cols) == len(F.columnas_features(dataset))


def test_llego_por_traspaso_es_proxy_imperfecto(tablas, dataset):
    """Decision 2 del gerente en la ronda 2: la feature no puede reproducir `lat_switcher`."""
    lat = tablas["afiliados_latente_churn"].set_index("afiliado_id")
    m = dataset[["afiliado_id", "llego_por_traspaso"]].copy()
    m["lat"] = m["afiliado_id"].map(lat["lat_switcher"])
    concordancia = float((m["llego_por_traspaso"] == m["lat"]).mean())
    assert 0.60 <= concordancia <= 0.85, concordancia
    # y falla en los dos sentidos (no es el latente con otro nombre)
    assert 0.05 < float(m.loc[m["llego_por_traspaso"] == 1, "lat"].mean()) < 0.95
    assert 0.05 < float(m.loc[m["llego_por_traspaso"] == 0, "lat"].mean()) < 0.95


def test_chequeos_de_determinismo(tablas, dataset):
    """Ronda 2, bloqueante 1: ninguna feature puede ser una observacion disfrazada del target."""
    from afp_lab.churn_traspaso.synthetic import chequeos_determinismo
    d = {x["indicador"]: x for x in chequeos_determinismo(dataset)}
    # pedir el certificado NO puede casi garantizar el traspaso
    assert d["lift_certificado"]["valor"] < 6.0, d["lift_certificado"]
    assert d["p_target_dado_certificado"]["valor"] < 0.25
    # ninguna feature suelta puede ordenar como el modelo completo
    assert d["auc_mejor_feature_suelta"]["valor"] < 0.72, d["auc_mejor_feature_suelta"]
    # y el resumen de simulacion debe traer los chequeos cuando se le pasa el dataset
    r = resumen_simulacion(tablas, dataset).set_index("indicador")
    assert "auc_mejor_feature_suelta" in r.index and "p_target_dado_certificado" in r.index
    assert 0.60 <= r.loc["concordancia_proxy_switcher", "valor"] <= 0.85


def test_segmento_sin_positivos_no_se_da_por_ganado():
    """Menor 12: un AUC NaN no puede contar como victoria del modelo."""
    seg = pd.DataFrame([
        {"predictor": "modelo", "segmento": "s", "valor": "a", "auc": float("nan"), "n": 100,
         "tasa_base": 0.0, "frac_filas": 0.5, "segmento_grande": True},
        {"predictor": "base", "segmento": "s", "valor": "a", "auc": float("nan"), "n": 100,
         "tasa_base": 0.0, "frac_filas": 0.5, "segmento_grande": True},
        {"predictor": "modelo", "segmento": "s", "valor": "b", "auc": 0.8, "n": 100,
         "tasa_base": 0.1, "frac_filas": 0.5, "segmento_grande": True},
        {"predictor": "base", "segmento": "s", "valor": "b", "auc": 0.6, "n": 100,
         "tasa_base": 0.1, "frac_filas": 0.5, "segmento_grande": True},
    ])
    comp = E.comparar_segmentos(seg, "modelo", "base", "auc")
    assert not comp.set_index("valor").loc["a", "gana_modelo"]
    assert comp.set_index("valor").loc["b", "gana_modelo"]
    assert list(E.segmentos_no_superados(comp)["valor"]) == ["a"]


def test_readme_documenta_los_parametros_vigentes():
    """Mayor 7: el README no puede desviarse del codigo; la tabla se genera desde ParametrosChurn."""
    doc = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    ini, fin = "<!-- PARAMS:INICIO -->", "<!-- PARAMS:FIN -->"
    assert ini in doc and fin in doc, "faltan los marcadores de la tabla de parametros"
    bloque = doc.split(ini)[1].split(fin)[0].strip()
    assert bloque == tabla_parametros_markdown().strip(), (
        "la tabla de parametros del README no coincide con ParametrosChurn; "
        "regenerar con synthetic.tabla_parametros_markdown()")


def test_segmentos_y_calibracion(dataset):
    d = E.agregar_segmentos(dataset.head(4000))
    for c in ("tramo_saldo", "tramo_edad", "tramo_renta"):
        assert d[c].nunique() >= 2
    curva = E.curva_calibracion(d["y"].to_numpy(), np.clip(d["y"] * 0.5 + 0.01, 0, 1).to_numpy(), n_bins=5)
    assert len(curva) == 5 and curva["n"].sum() == len(d)


# --------------------------------------------------------------------- end to end
def test_run_end_to_end(tmp_path):
    assert run_main(["--n-afiliados", "900", "--salida", str(tmp_path), "--sin-fuentes",
                     "--sin-techo", "--semillas-estabilidad", ""]) == 0
    rep = tmp_path / "reportes"
    for f in ("metricas_test.csv", "captura_negocio.csv", "importancias.csv", "calibracion.csv",
              "saldo_en_riesgo.csv", "split.csv", "tiempos.csv", "casos_explicados.csv"):
        assert (rep / f).exists(), f
    assert (tmp_path / "modelo" / "lightgbm.txt").exists()
    met = pd.read_csv(rep / "metricas_test.csv").set_index("predictor")
    # con 900 afiliados el modelo no tiene senal suficiente (AUC ~0,5): este test verifica que el
    # pipeline corre y escribe todo, no la calidad. La comparacion contra los baselines se exige en
    # `test_modelo_supera_baselines_y_persiste`, con una muestra donde el modelo si puede aprender.
    for pred in ("modelo_lgbm", "baseline_tasa_base", "baseline_regla_negocio", "baseline_regla_graduada"):
        assert pred in met.index
        assert 0.0 <= met.loc[pred, "auc"] <= 1.0
    assert met.loc["baseline_tasa_base", "lift_decil_1"] == pytest.approx(1.0)
    for f in ("parametros.csv", "resumen_simulacion.csv", "saldo_en_riesgo_por_corte.csv"):
        assert (rep / f).exists(), f


def test_run_no_escribe_tablas_latentes(tmp_path):
    """Menor 7 de la ronda 2: `afiliados_latente_churn` es latente y no se persiste con las fuentes."""
    assert run_main(["--n-afiliados", "700", "--salida", str(tmp_path), "--sin-techo",
                     "--semillas-estabilidad", ""]) == 0
    fuentes = {f.name for f in (tmp_path / "fuentes").glob("*.parquet")}
    assert "afiliados_latente_churn.parquet" not in fuentes
    assert "saldos.parquet" in fuentes and "traspasos.parquet" in fuentes


def test_run_con_techo_y_estabilidad(tmp_path):
    """El barrido de semillas y el diagnostico de techo corren y producen rangos."""
    assert run_main(["--n-afiliados", "700", "--salida", str(tmp_path), "--sin-fuentes",
                     "--semillas-estabilidad", "42,123"]) == 0
    est = pd.read_csv(tmp_path / "reportes" / "estabilidad_semillas.csv")
    assert len(est) == 2 and {"auc", "lift_decil_1", "sesgo_saldo_isotonica"} <= set(est.columns)
    rangos = pd.read_csv(tmp_path / "reportes" / "estabilidad_rangos.csv")
    assert {"metrica", "min", "mediana", "max"} == set(rangos.columns)
    techo = pd.read_csv(tmp_path / "reportes" / "techo_predictibilidad.csv")
    assert len(techo) == 2
