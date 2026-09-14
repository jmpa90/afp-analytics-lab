"""Tests del proyecto 11 (voz del afiliado). Minimo exigido por CONVENCIONES SS1/SS3:

  * leakage: alterar la encuesta o las tablas operacionales POSTERIORES al corte (incluida la
    respuesta del PROPIO corte) no cambia las features de ese corte;
  * forma del dataset y columnas de features (excluye ids/target/`lat_`/`p_`);
  * el generador respeta las tasas objetivo de invitacion/respuesta y produce una tabla latente
    coherente (nunca expuesta como feature);
  * NPS: la bucketizacion de categorias y la agregacion NPS(score)/NPS(probabilidades) verificadas
    a mano;
  * corrida end-to-end pequena (`run.main`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afp_lab.voz_afiliado import evaluate as E
from afp_lab.voz_afiliado import features as F
from afp_lab.voz_afiliado import model as M
from afp_lab.voz_afiliado.run import main as run_main
from afp_lab.voz_afiliado.synthetic import (DETRACTOR_HASTA, PROMOTOR_DESDE, generar_tablas,
                                            resumen_simulacion)

N_TEST = 700
SEED = 42
PERIODO_FIN_TEST = 202412   # ventana corta: tests rapidos (< 60 s todo el archivo)


@pytest.fixture(scope="module")
def tablas() -> dict[str, pd.DataFrame]:
    return generar_tablas(n_afiliados=N_TEST, seed=SEED, periodo_fin=PERIODO_FIN_TEST)


@pytest.fixture(scope="module")
def cortes(tablas) -> list[int]:
    return F.cortes_evaluables(tablas)


@pytest.fixture(scope="module")
def dataset(tablas, cortes) -> pd.DataFrame:
    return F.construir_dataset(tablas, cortes)


# --------------------------------------------------------------------- universo comun (02 == 03)
def test_universo_comun_identico_entre_02_y_03(tablas):
    """`churn_traspaso` y `propension_apv` parten del MISMO universo (mismo seed/perfil/ventana):
    afiliados y panel deben coincidir exactamente (si no, dejarian de ser reutilizables tal cual)."""
    from afp_lab.propension_apv.synthetic import generar_tablas as apv_generar

    apv = apv_generar(n_afiliados=N_TEST, seed=SEED, periodo_inicio=2022 * 100 + 1,
                      periodo_fin=PERIODO_FIN_TEST, perfil="cuprum")
    pd.testing.assert_frame_equal(
        tablas["afiliados"].reset_index(drop=True), apv["afiliados"].reset_index(drop=True))
    cols_comunes = ["afiliado_id", "periodo", "estado", "rim_emp1", "rim_emp2"]
    pd.testing.assert_frame_equal(
        tablas["panel_mensual"][cols_comunes].reset_index(drop=True),
        apv["panel_mensual"][cols_comunes].reset_index(drop=True))


# --------------------------------------------------------------------- forma del dataset
def test_forma_dataset_y_columnas(dataset):
    assert len(dataset) > 0
    assert dataset["invitado"].isin([0, 1]).all()
    assert dataset["respondio"].isin([0, 1]).all()
    assert (dataset.loc[dataset["respondio"] == 1, "respuesta"].between(0, 10)).all()
    assert dataset.loc[dataset["invitado"] == 0, "respondio"].eq(0).all()

    cols = F.columnas_features(dataset)
    for prohibida in ("afiliado_id", "periodo_corte", "invitado", "respondio", "respuesta", "categoria",
                      "es_detractor", "es_promotor"):
        assert prohibida not in cols
    assert not any(c.startswith("lat_") for c in cols)
    assert not any(c.startswith("p_") for c in cols)
    assert "peso_ipw" not in cols
    # tasas del brief, con margen (ventana corta / cartera chica)
    assert 0.04 <= float(dataset["invitado"].mean()) <= 0.14
    resp = dataset[dataset["invitado"] == 1]
    assert 0.12 <= float(resp["respondio"].mean()) <= 0.40


def test_satisfaccion_latente_nunca_es_feature(tablas):
    assert "satisfaccion_latente" in tablas
    assert {"afiliado_id", "periodo", "lat_satisfaccion", "lat_p_responde_si_invitado"} <= set(
        tablas["satisfaccion_latente"].columns)
    assert tablas["satisfaccion_latente"]["lat_satisfaccion"].between(0, 10).all()


def test_resumen_simulacion_tasas_objetivo(tablas):
    r = resumen_simulacion(tablas).set_index("indicador")["valor"]
    assert 0.04 <= r["frac_invitada_mensual"] <= 0.14
    assert 0.10 <= r["frac_respuesta_entre_invitados"] <= 0.45


# --------------------------------------------------------------------- leakage
def test_leakage_futuro_no_cambia_features(tablas, cortes):
    """Igual que 02/03: alterar/borrar todo lo posterior al corte (incluida la RESPUESTA de la
    encuesta del propio corte y de los meses futuros) no puede cambiar esa fila de features."""
    corte = cortes[len(cortes) // 2]
    df_base = F.construir_features_multi(tablas, cortes)

    tablas_mut = dict(tablas)
    for nombre in ("reclamos", "contactos", "uso_digital", "cambios_fondo", "mercado", "saldos",
                  "aportes_voluntarios", "campanas_apv", "apvc_empleador", "agregado_empleador",
                  "planilla", "rezagos"):
        d = tablas[nombre].copy()
        fut = d["periodo"] > corte
        assert fut.any(), f"la tabla {nombre} no tiene futuro que mutar en este corte de prueba"
        num_cols = [c for c in d.select_dtypes("number").columns
                   if "periodo" not in c and not c.endswith("_id")]
        d[num_cols] = d[num_cols].astype("float64")
        d.loc[fut, num_cols] = d.loc[fut, num_cols] * 0.0 + 999_999.0
        tablas_mut[nombre] = d

    # la encuesta: mutar el PROPIO corte (nunca debe entrar al historial) y todo lo posterior
    enc = tablas["encuesta"].copy()
    fut_o_actual = enc["periodo"] >= corte
    assert fut_o_actual.any()
    enc.loc[fut_o_actual, "respuesta"] = 0.0
    enc.loc[fut_o_actual, "respondio"] = 1
    enc.loc[fut_o_actual, "categoria"] = "DETRACTOR"
    tablas_mut["encuesta"] = enc

    pan = tablas["panel_mensual"].copy()
    corte_lab = corte  # MESES_DESFASE_RIM se aplica dentro de construir_features_multi
    fut = pan["periodo"] > corte
    pan.loc[fut, "rim_emp1"] = pan.loc[fut, "rim_emp1"] * 13.0 + 1_000_000
    pan.loc[fut, "dias_licencia"] = 999
    pan.loc[fut, "estado"] = "RETIRADO"
    tablas_mut["panel_mensual"] = pan
    tr = tablas["traspasos"].copy()
    if len(tr):
        tr = tr[tr["periodo_solicitud"] <= corte]
    tablas_mut["traspasos"] = tr

    df_mut = F.construir_features_multi(tablas_mut, cortes)
    a = df_base[df_base["periodo_corte"] == corte].sort_values("afiliado_id").reset_index(drop=True)
    b = df_mut[df_mut["periodo_corte"] == corte].sort_values("afiliado_id").reset_index(drop=True)
    assert len(a) == len(b) > 0
    for c in F.columnas_features(a):
        av, bv = a[c], b[c]
        if str(av.dtype) == "category" or av.dtype == object:
            assert (av.astype(str) == bv.astype(str)).all(), f"columna con leakage: {c}"
        else:
            np.testing.assert_allclose(av.to_numpy(dtype=float), bv.to_numpy(dtype=float),
                                       rtol=1e-6, atol=1e-6, err_msg=f"columna con leakage: {c}")


def test_historial_encuesta_excluye_el_propio_corte():
    """Verificacion a mano: la respuesta del corte t NUNCA debe verse en el historial de t."""
    n = 5
    afiliados = pd.DataFrame({"afiliado_id": [1], "sexo": ["F"], "edad_inicio": [30],
                             "region": [13], "nivel_educacional": ["MEDIA"],
                             "tipo_afiliado": ["DEPENDIENTE"], "periodo_afiliacion": [202101]})
    periodos = [202303, 202304, 202305, 202306, 202307]
    encuesta = pd.DataFrame({
        "afiliado_id": [1, 1], "periodo": [202304, 202306],
        "tipo_invitacion": ["RELACIONAL", "RELACIONAL"], "respondio": [1, 1],
        "respuesta": [9.0, 2.0], "categoria": ["PROMOTOR", "DETRACTOR"],
    })
    tablas = {"afiliados": afiliados, "panel_mensual": pd.DataFrame({
        "afiliado_id": [1] * n, "periodo": periodos, "empleador_id": [1] * n,
        "dias_licencia": [0] * n}),
        "encuesta": encuesta, "agregado_empleador": pd.DataFrame(
            columns=["empleador_id", "periodo", "frac_dnp", "lag_recepcion_medio"])}
    out = F._features_propias_multi(tablas, [202305, 202306, 202307])
    f305 = out[out["periodo_corte"] == 202305].iloc[0]
    assert f305["encuesta_respuestas_12m"] == 1        # solo ve 202304 (PROMOTOR)
    assert f305["ultima_respuesta_nps"] == 9.0
    f306 = out[out["periodo_corte"] == 202306].iloc[0]
    assert f306["encuesta_respuestas_12m"] == 1        # 202306 (su propio corte) NO cuenta aun
    assert f306["ultima_respuesta_nps"] == 9.0
    f307 = out[out["periodo_corte"] == 202307].iloc[0]
    assert f307["encuesta_respuestas_12m"] == 2        # ahora si ve 202304 y 202306
    assert f307["ultima_respuesta_nps"] == 2.0
    assert f307["ultima_categoria_nps"] == "DETRACTOR"


# --------------------------------------------------------------------- NPS: bucketizacion y agregacion
def test_categoria_de_score_limites():
    s = np.array([0, 6, 6.4, 6.6, 7, 8, 8.6, 9, 10])
    cat = M.categoria_de_score(s)
    assert list(cat) == ["DETRACTOR", "DETRACTOR", "DETRACTOR", "PASIVO", "PASIVO", "PASIVO",
                        "PROMOTOR", "PROMOTOR", "PROMOTOR"]
    assert DETRACTOR_HASTA == 6 and PROMOTOR_DESDE == 9


def test_nps_de_score_y_de_probs_casos_conocidos():
    todos_promotores = np.full(10, 9.5)
    assert M.nps_de_score(todos_promotores) == pytest.approx(100.0)
    todos_detractores = np.full(10, 2.0)
    assert M.nps_de_score(todos_detractores) == pytest.approx(-100.0)
    mitad_mitad = np.array([9.0] * 5 + [2.0] * 5)
    assert M.nps_de_score(mitad_mitad) == pytest.approx(0.0)
    assert M.nps_de_probs(np.array([0.5, 0.5]), np.array([0.2, 0.2])) == pytest.approx(30.0)


def test_nps_por_corte_verdadero_e_ingenuo(tablas, cortes):
    """`nps_por` con `by="periodo_corte"` debe reproducir a mano el NPS verdadero de un corte chico
    (misma poblacion que `construir_dataset`: se compara contra el merge, no contra `satisfaccion_latente`
    entera, porque esta ultima puede incluir un afiliado con solicitud de salida ya presentada que
    `construir_features_multi` ya excluye de la poblacion de ese corte)."""
    corte = cortes[0]
    df = F.construir_dataset(tablas, [corte])
    lat = tablas["satisfaccion_latente"].rename(columns={"periodo": "periodo_corte"})
    m = df[["afiliado_id", "periodo_corte"]].merge(lat, on=["afiliado_id", "periodo_corte"], how="left")
    esperado = M.nps_de_score(m["lat_satisfaccion"].to_numpy())
    prep = E.preparar_comparacion(tablas, df, scores={}, probs={})
    tabla = E.nps_por(prep, "periodo_corte")
    assert tabla.loc[tabla["periodo_corte"] == corte, "nps_verdadera"].iloc[0] == pytest.approx(esperado)


# --------------------------------------------------------------------- correccion ronda 5 (revision)
def test_columnas_features_excluye_derivadas_pipeline(dataset):
    """Bloqueante #2 de la revision: `peso_ipw`/`p_hat_responde` (agregadas por `run.py` como
    sample_weight/diagnostico del modelo de respuesta) NUNCA deben quedar como feature de entrada,
    aunque se agreguen al DataFrame antes de entrenar (como hace `run.py`)."""
    d = dataset.copy()
    d["p_hat_responde"] = 0.2
    d["peso_ipw"] = 3.0
    cols = F.columnas_features(d)
    assert "peso_ipw" not in cols
    assert "p_hat_responde" not in cols
    assert {"p_hat_responde", "peso_ipw"} <= set(F.COLUMNAS_DERIVADAS_PIPELINE)


def test_regla_negocio_independiente_del_generador():
    """Bloqueante #1 de la revision: el baseline de regla de negocio (`COEF_REGLA`) no debe copiar
    los coeficientes del generador oculto (`synthetic.ParametrosSatisfaccion`) -- ni el nivel base
    ni ningun efecto de evento."""
    from afp_lab.voz_afiliado.synthetic import ParametrosSatisfaccion

    par = ParametrosSatisfaccion()
    valores_generador = {
        par.base_nivel, par.ef_reclamo_sin_resolver, par.ef_rezago_mensual,
        par.ef_licencia_mal_pagada, par.ef_dnp_empleador, par.ef_contacto_ejecutivo_resuelto,
        par.ef_rentabilidad_relativa_positiva, par.ef_brecha_comision_desfavorable,
    }
    for nombre, valor in M.COEF_REGLA.items():
        assert valor not in valores_generador, (
            f"COEF_REGLA[{nombre!r}]={valor} coincide con un parametro del generador oculto")


# --------------------------------------------------------------------- drivers
def test_estimar_drivers_recupera_signo_en_caso_sintetico():
    """Caso sintetico simple: un evento con efecto verdadero conocido debe recuperar el signo."""
    rng = np.random.default_rng(0)
    n = 4000
    edad = rng.uniform(20, 60, n)
    df = pd.DataFrame({
        "edad": edad, "meses_desde_afiliacion": rng.uniform(0, 100, n),
        "log_saldo": rng.normal(14, 1, n), "rim_l1": rng.normal(1e6, 1e5, n),
        "densidad_12": rng.uniform(0, 1, n), "antiguedad_afp_meses": rng.uniform(0, 100, n),
        "meses_con_empleador": rng.uniform(0, 100, n), "mes": rng.integers(1, 13, n),
        "tipo_afiliado": rng.choice(["DEPENDIENTE", "INDEPENDIENTE"], n),
        "reclamos_abiertos_12m": np.zeros(n), "reclamos_desfavorables_12m": np.zeros(n),
        "rezago_mensual_actual": np.zeros(n), "licencia_larga_3m": np.zeros(n),
        "dnp_empleador_actual": np.zeros(n),
        "contactos_ejecutivo_propio_3m": (rng.random(n) < 0.3).astype(float),
        "brecha_rentabilidad_pp": rng.normal(0, 1, n), "brecha_comision_pp": rng.normal(0, 1, n),
    })
    T = df["contactos_ejecutivo_propio_3m"].to_numpy() > 0
    ruido = rng.normal(0, 0.5, n)
    df["respuesta"] = np.clip(7.0 + 2.0 * T + ruido, 0, 10)
    out = M.estimar_drivers(df, pesos=np.ones(n))
    fila = out.set_index("evento").loc["contacto_ejecutivo_resuelto"]
    assert fila["efecto_estimado"] > 1.0   # efecto verdadero simulado: +2.0


# --------------------------------------------------------------------- corrida end-to-end
def test_run_end_to_end(tmp_path: Path):
    codigo = run_main(["--n-afiliados", str(N_TEST), "--periodo-fin", "202506",
                      "--n-test", "3", "--n-valid", "3", "--salida", str(tmp_path / "salida"),
                      "--sin-fuentes"])
    assert codigo == 0
    rep = tmp_path / "salida" / "reportes"
    for archivo in ("split.csv", "nps_por_corte.csv", "nps_error_resumen.csv", "nps_por_segmento.csv",
                   "detractor_metricas.csv", "drivers.csv", "drivers_prioridad.csv",
                   "valor_punto_traspaso.csv", "valor_punto_apv.csv", "aceptacion.csv", "tiempos.csv",
                   "importancias_score.csv", "ejemplos_individuales.csv"):
        assert (rep / archivo).exists(), archivo
