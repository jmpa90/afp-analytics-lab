"""Tests del proyecto 07 (uplift de retencion). Minimo exigido por CONVENCIONES SS1/SS3:

  * leakage: alterar la campana o el panel POSTERIOR al corte no cambia las features de ese corte;
  * coherencia: en las filas de control (T=0) el target `y` es IDENTICO al de 02 (`y0`), siempre;
  * historial de campana (conteo 6/12 m, meses desde el ultimo) verificado a mano;
  * forma del dataset, columnas de features (excluye T/holdout/tramo_*/lat_*), segmentos;
  * sanity check de Qini/AUUC con un caso sintetico de signo conocido;
  * corrida end-to-end pequena (`run.main`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afp_lab.churn_traspaso import features as CT_F
from afp_lab.churn_traspaso.synthetic import generar_tablas
from afp_lab.rim_nowcasting.calendario import add_months

from afp_lab.uplift_retencion import evaluate as E
from afp_lab.uplift_retencion import features as F
from afp_lab.uplift_retencion import model as M
from afp_lab.uplift_retencion.run import main as run_main
from afp_lab.uplift_retencion.synthetic import ParametrosCampana

N_TEST = 900
SEED = 42
PERIODO_FIN_TEST = 202512   # ventana corta: tests rapidos (< 60 s todo el archivo)


@pytest.fixture(scope="module")
def tablas() -> dict[str, pd.DataFrame]:
    return generar_tablas(n_afiliados=N_TEST, seed=SEED, perfil="cuprum", periodo_fin=PERIODO_FIN_TEST)


@pytest.fixture(scope="module")
def cortes(tablas) -> list[int]:
    return CT_F.cortes_evaluables(tablas)


@pytest.fixture(scope="module")
def dataset(tablas, cortes) -> tuple[pd.DataFrame, pd.DataFrame]:
    return F.construir_dataset_causal(tablas, cortes, seed=SEED)


# --------------------------------------------------------------------- forma del dataset
def test_forma_dataset_y_columnas(dataset):
    df, latente = dataset
    assert len(df) == len(latente) > 0
    assert set(df["afiliado_id"]).issubset(set(latente["afiliado_id"]))
    assert df["T"].isin([0, 1]).all()
    assert df["y"].isin([0, 1]).all()
    assert df["holdout"].dtype == bool
    # las columnas latentes / de diseno NUNCA son features
    cols = F.columnas_features(df)
    for prohibida in ("T", "holdout", "afiliado_id", "periodo_corte", "y"):
        assert prohibida not in cols, prohibida
    assert not any(c.startswith("lat_") for c in cols)
    assert "tramo_saldo" not in df.columns   # no se filtra a las features de 02 (se calcula aparte)
    # la fraccion de holdout debe rondar el parametro (10 %), con margen por el tamano chico del test
    assert 0.03 <= float(df["holdout"].mean()) <= 0.20


def test_segmentos_y_cate_verdadero(dataset):
    _, latente = dataset
    assert set(latente["segmento"].unique()) <= {"persuadible", "sleeping_dog", "neutral"}
    # persuadibles: CATE <= 0 (protege); sleeping dogs: CATE >= 0 (empuja); neutros: CATE == 0
    assert (latente.loc[latente["segmento"] == "persuadible", "lat_cate"] <= 0).all()
    assert (latente.loc[latente["segmento"] == "sleeping_dog", "lat_cate"] >= 0).all()
    assert (latente.loc[latente["segmento"] == "neutral", "lat_cate"] == 0).all()
    # la mayoria del target no deberia ser el efecto (brief: "cero en la mayoria")
    assert float((latente["segmento"] == "neutral").mean()) > 0.85


def test_control_y_es_identico_a_y0_de_02(dataset):
    """Coherencia con `traspasos`: en las filas SIN tratamiento el target no se toca."""
    df, latente = dataset
    m = df[["afiliado_id", "periodo_corte", "T", "y"]].merge(
        latente[["afiliado_id", "periodo_corte", "y0_control_potencial"]], on=["afiliado_id", "periodo_corte"])
    control = m[m["T"] == 0]
    assert len(control) > 0
    assert (control["y"] == control["y0_control_potencial"]).all()


def test_holdout_es_aleatorio_no_confundido(dataset):
    """Dentro del holdout la asignacion no debe depender del riesgo: T no puede separar casi perfecto."""
    df, _ = dataset
    hold = df[df["holdout"]]
    assert len(hold) > 50
    tasa = float(hold["T"].mean())
    assert 0.30 <= tasa <= 0.70   # par.p_tratamiento_holdout = 0.50, con margen de muestreo


# --------------------------------------------------------------------- leakage
def test_leakage_futuro_no_cambia_features_ni_historial(tablas, cortes):
    """Igual que en 02: alterar/borrar TODO lo posterior al corte no puede cambiar esa fila."""
    corte = cortes[len(cortes) // 2]
    df_base, _ = F.construir_dataset_causal(tablas, cortes, seed=SEED)

    tablas_mut = dict(tablas)
    for nombre in ("uso_digital", "contactos", "reclamos", "cambios_fondo", "mercado", "saldos"):
        d = tablas[nombre].copy()
        fut = d["periodo"] > corte
        assert fut.any(), f"la tabla {nombre} no tiene futuro que mutar en este corte de prueba"
        num_cols = d.select_dtypes("number").columns
        d[num_cols] = d[num_cols].astype("float64")
        d.loc[fut, num_cols] = d.loc[fut, num_cols] * 0.0 + 999_999.0
        tablas_mut[nombre] = d
    pan = tablas["panel_mensual"].copy()
    corte_lab = add_months(corte, -2)
    fut = pan["periodo"] > corte_lab
    pan.loc[fut, "rim_emp1"] = pan.loc[fut, "rim_emp1"] * 13.0 + 1_000_000
    pan.loc[fut, "estado"] = "RETIRADO"
    tablas_mut["panel_mensual"] = pan
    tr = tablas["traspasos"].copy()
    if len(tr):
        fut = tr["periodo_solicitud"] > corte
        tr = tr[~fut]
    tablas_mut["traspasos"] = tr

    df_mut, _ = F.construir_dataset_causal(tablas_mut, cortes, seed=SEED)
    a = df_base[df_base["periodo_corte"] == corte].sort_values("afiliado_id").reset_index(drop=True)
    b = df_mut[df_mut["periodo_corte"] == corte].sort_values("afiliado_id").reset_index(drop=True)
    assert len(a) == len(b) > 0
    cols = ["T", "campana_contactos_6m", "campana_contactos_12m", "campana_meses_desde_ultimo"] + F.columnas_features(a)
    for c in cols:
        av, bv = a[c], b[c]
        if str(av.dtype) == "category" or av.dtype == object:
            assert (av.astype(str) == bv.astype(str)).all(), f"columna con leakage: {c}"
        else:
            np.testing.assert_allclose(av.to_numpy(dtype=float), bv.to_numpy(dtype=float),
                                       rtol=1e-6, atol=1e-6, err_msg=f"columna con leakage: {c}")


def test_historial_campana_excluye_el_corte_actual():
    """Verificacion a mano del historial de campana en un afiliado con 5 cortes consecutivos."""
    trat = pd.DataFrame({
        "afiliado_id": [1, 1, 1, 1, 1],
        "periodo_corte": [202401, 202402, 202403, 202404, 202405],
        "T": [1, 0, 1, 1, 0],
    })
    hist = F._historial_campana(trat, ventanas=(2, 4))
    # corte 1: sin historia previa -> 0 contactos, "nunca" (tope)
    f1 = hist.iloc[0]
    assert f1["campana_contactos_2m"] == 0 and f1["campana_meses_desde_ultimo"] == 99.0
    # corte 2: el corte 1 tuvo T=1, es historia -> meses_desde = 0 (el contacto fue el mes anterior)
    f2 = hist.iloc[1]
    assert f2["campana_contactos_2m"] == 1 and f2["campana_meses_desde_ultimo"] == 0.0
    # corte 3: su propio T=1 (de ESTE corte) NO debe contar en su propio historial
    f3 = hist.iloc[2]
    assert f3["campana_contactos_2m"] == 1  # solo el contacto del corte 1 (el de 202402 fue T=0)
    assert f3["campana_meses_desde_ultimo"] == 1.0
    # corte 4: ventana de 2 ve los contactos de los cortes 2 y 3 (T=0,1) -> 1; meses_desde=0 (corte 3 fue T=1)
    f4 = hist.iloc[3]
    assert f4["campana_contactos_2m"] == 1 and f4["campana_meses_desde_ultimo"] == 0.0
    # corte 5: ventana de 4 ve los contactos de los cortes 1-4 (1,0,1,1) -> 3
    f5 = hist.iloc[4]
    assert f5["campana_contactos_4m"] == 3 and f5["campana_meses_desde_ultimo"] == 0.0


def test_historial_campana_no_cruza_entre_afiliados():
    trat = pd.DataFrame({
        "afiliado_id": [1, 1, 2, 2],
        "periodo_corte": [202401, 202402, 202401, 202402],
        "T": [1, 1, 0, 0],
    })
    hist = F._historial_campana(trat, ventanas=(6,))
    fila_af2_corte1 = hist[(hist["afiliado_id"] == 2) & (hist["periodo_corte"] == 202401)].iloc[0]
    # el afiliado 2 no puede heredar el T=1 del afiliado 1 (ultimo del bloque anterior)
    assert fila_af2_corte1["campana_contactos_6m"] == 0
    assert fila_af2_corte1["campana_meses_desde_ultimo"] == 99.0


# --------------------------------------------------------------------- Qini / AUUC (sanity)
def test_qini_positivo_cuando_el_score_ordena_bien():
    """Caso sintetico de signo conocido: la mitad con mejor score es realmente mas persuadible."""
    rng = np.random.default_rng(0)
    n = 4000
    T = (rng.random(n) < 0.5).astype(int)
    bueno = np.arange(n) < n // 2                      # la mitad "buena": el tratamiento reduce y
    p = np.where(bueno, np.where(T == 1, 0.05, 0.20), 0.20)
    y = (rng.random(n) < p).astype(int)
    score_bien = np.where(bueno, 1.0, 0.0)             # prioriza exactamente a los "buenos"
    score_mal = 1.0 - score_bien
    r_bien = E.qini_auuc(y, T, score_bien)
    r_mal = E.qini_auuc(y, T, score_mal)
    assert r_bien["qini_coeficiente"] > 0.005          # ordenar bien: Qini positivo (mejor que el azar)
    assert r_mal["qini_coeficiente"] < 0                # ordenar al reves: Qini negativo
    assert r_mal["qini_coeficiente"] < r_bien["qini_coeficiente"]


def test_overlap_propension():
    e = np.array([0.5, 0.5, 0.5, 0.01, 0.99, 0.5, 0.5, 0.5])
    assert E.overlap_propension(e) == pytest.approx(6 / 8)


def test_correlacion_cate():
    lat = np.array([-0.2, -0.1, 0.0, 0.0, 0.1, 0.2])
    tau = np.array([-0.25, -0.05, 0.01, -0.01, 0.05, 0.25])
    r = E.correlacion_cate(lat, tau)
    assert r["spearman"] > 0.9
    assert r["mae"] < 0.1


# --------------------------------------------------------------------- modelos (chequeo minimo)
def test_columnas_features_no_incluye_tratamiento_ni_holdout(dataset):
    df, _ = dataset
    cols = M.columnas_features(df)
    assert "T" not in cols and "holdout" not in cols


def test_propension_entrena_y_predice_en_0_1(dataset):
    df, _ = dataset
    tr = df[df["periodo_corte"] < df["periodo_corte"].quantile(0.7)]
    tr_c = tr[~tr["holdout"]]
    modelo = M.estimar_propension(tr_c, seed=SEED)
    p = M.predecir_propension(modelo, tr_c.head(50))
    assert ((p > 0) & (p < 1)).all()


# --------------------------------------------------------------------- corrida end-to-end
def test_run_end_to_end(tmp_path: Path):
    codigo = run_main(["--n-afiliados", "700", "--periodo-fin", "202412",
                      "--salida", str(tmp_path / "salida")])
    assert codigo == 0
    rep = tmp_path / "salida" / "reportes"
    for archivo in ("split.csv", "qini_general.csv", "correlacion_cate.csv",
                   "traspasos_evitados.csv", "segmento_accion.csv", "overlap.csv"):
        assert (rep / archivo).exists(), archivo
