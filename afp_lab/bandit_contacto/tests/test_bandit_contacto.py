"""Tests del proyecto 09 (bandit contextual de contacto). Escala chica, < 60 s.

Cubre CONVENCIONES §1 y §3 aplicadas a este proyecto (no se re-testea 02, que ya prueba su propio
leakage/features): sin leakage al construir el contexto con nuestras tablas propias sumadas,
ninguna columna latente llega al contexto de las politicas, la respuesta latente es coherente
(persuadibles/indiferentes/sleeping dogs), las restricciones de negocio (presupuesto, recontacto,
opt-out) se respetan, las matrices de propension son distribuciones validas y una corrida chica
end-to-end (simulacion online + evaluacion off-policy) corre sin errores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afp_lab.rim_nowcasting.calendario import add_months
from afp_lab.churn_traspaso.features import columnas_features

from afp_lab.bandit_contacto import (BRAZOS, DELTA_LOGODDS, EPSILON_LOGGING, IDX_BRAZO,
                                     MESES_ENTRE_CONTACTOS, N_BRAZOS, TIPOS_RESPUESTA)
from afp_lab.bandit_contacto import evaluate as E
from afp_lab.bandit_contacto import features as F
from afp_lab.bandit_contacto import model as M
from afp_lab.bandit_contacto.synthetic import calcular_respuesta_latente, generar_tablas_bandit

N_TEST = 600
SEED = 42
PERIODO_FIN_TEST = 202512  # 48 meses desde 202201: alcanza para 8-12 cortes de simulacion chicos


@pytest.fixture(scope="module")
def tablas() -> dict[str, pd.DataFrame]:
    return generar_tablas_bandit(n_afiliados=N_TEST, periodo_fin=PERIODO_FIN_TEST, seed=SEED)


@pytest.fixture(scope="module")
def cortes_test(tablas) -> list[int]:
    cortes = F.cortes_evaluables(tablas)
    return cortes[-8:]


@pytest.fixture(scope="module")
def entorno(tablas, cortes_test) -> dict:
    return M.preparar_entorno(tablas, cortes_test, seed_mundo=SEED)


# --------------------------------------------------------------------- tablas propias
def test_generar_tablas_bandit_agrega_tablas_propias(tablas):
    assert "preferencias_contacto" in tablas and "respuesta_latente_tipo" in tablas
    assert set(tablas["preferencias_contacto"]["afiliado_id"]) <= set(tablas["afiliados"]["afiliado_id"])
    tipos = set(tablas["respuesta_latente_tipo"]["lat_tipo_respuesta"])
    assert tipos <= set(TIPOS_RESPUESTA)
    # una fila por afiliado, sin duplicados
    assert tablas["respuesta_latente_tipo"]["afiliado_id"].is_unique


def test_opt_out_es_minoritario_y_estable(tablas):
    frac = tablas["preferencias_contacto"]["solicito_no_contacto"].mean()
    assert 0.0 < frac < 0.10  # ~3% esperado (P_OPT_OUT)


# --------------------------------------------------------------------- sin leakage
def test_contexto_sin_leakage(tablas, cortes_test):
    """Alterar datos POSTERIORES al corte no cambia el contexto construido para ese corte (aplica
    CONVENCIONES §3 con la tabla `preferencias_contacto` y `respuesta_latente_tipo` sumadas)."""
    corte = cortes_test[len(cortes_test) // 2]
    ctx_a = F.construir_contexto(tablas, [corte])

    tablas_alteradas = dict(tablas)
    saldos = tablas["saldos"].copy()
    fut = saldos["periodo"] > corte
    saldos.loc[fut, "saldo_obligatorio"] = saldos.loc[fut, "saldo_obligatorio"] * 0 + 999_999_999
    tablas_alteradas["saldos"] = saldos
    panel = tablas["panel_mensual"].copy()
    fut_p = panel["periodo"] > add_months(corte, -2)
    panel.loc[fut_p, "rim_emp1"] = 777_777
    tablas_alteradas["panel_mensual"] = panel

    ctx_b = F.construir_contexto(tablas_alteradas, [corte])
    pd.testing.assert_frame_equal(ctx_a.sort_values("afiliado_id").reset_index(drop=True),
                                  ctx_b.sort_values("afiliado_id").reset_index(drop=True))


def test_ninguna_columna_latente_en_el_contexto(tablas, cortes_test):
    ctx = F.construir_contexto(tablas, cortes_test[-2:])
    prohibidas = [c for c in ctx.columns if c.startswith("lat_") or c.startswith("p_")]
    assert prohibidas == []
    # y las que SI usan las politicas contextuales (numericas) son un subconjunto de las permitidas
    permitidas = set(columnas_features(ctx)) | {"score_riesgo"}
    assert set(F.CONTEXTO_NUMERICO) <= permitidas


# --------------------------------------------------------------------- respuesta latente
def test_respuesta_latente_en_rango_y_coherente(tablas, cortes_test):
    ctx = F.construir_contexto(tablas, cortes_test[-2:])
    tipo_tabla = tablas["respuesta_latente_tipo"]
    lat = calcular_respuesta_latente(ctx, tipo_tabla, SEED)
    cols_p = [f"lat_p_churn_{b}" for b in BRAZOS]
    for c in cols_p:
        assert lat[c].between(0, 1).all()
    # persuadibles: el contacto reduce la prob. de traspaso vs sin contacto (en promedio)
    pers = lat[lat["lat_tipo_respuesta"] == "PERSUADIBLE"]
    for brazo in ("EMAIL", "LLAMADA", "VISITA", "APV"):
        assert pers[f"lat_p_churn_{brazo}"].mean() < pers["lat_p_churn_SIN_CONTACTO"].mean()
    # sleeping dogs: el contacto (los brazos mas intensos) AUMENTA la prob. de traspaso
    sd = lat[lat["lat_tipo_respuesta"] == "SLEEPING_DOG"]
    assert sd["lat_p_churn_VISITA"].mean() > sd["lat_p_churn_SIN_CONTACTO"].mean()
    assert sd["lat_p_churn_LLAMADA"].mean() > sd["lat_p_churn_SIN_CONTACTO"].mean()
    # techo de predictibilidad razonable (CONVENCIONES §2): ninguna prob. degenerada en {0,1}
    assert lat[cols_p].to_numpy().std() > 0.01


# --------------------------------------------------------------------- restricciones de negocio
def test_pool_respeta_presupuesto_recontacto_y_opt_out(entorno, cortes_test):
    ultimo_contacto = np.full(entorno["n_afiliados"] + 1, -10_000, dtype=np.int64)
    presupuesto = 50
    for corte in cortes_test:
        pool = M.seleccionar_pool(entorno, corte, ultimo_contacto, presupuesto=presupuesto)
        assert len(pool) <= presupuesto
        af_ids = entorno["af_ids"][pool]
        assert not entorno["opt_out"][af_ids].any()
        from afp_lab.rim_nowcasting.calendario import periodo_to_idx
        corte_idx = periodo_to_idx(corte)
        assert np.all(corte_idx - ultimo_contacto[af_ids] >= MESES_ENTRE_CONTACTOS)
        ultimo_contacto[af_ids] = corte_idx  # simula que todo el pool fue contactado (peor caso)


# --------------------------------------------------------------------- politicas: propensiones validas
@pytest.mark.parametrize("nombre", ["SIN_CONTACTO", "TOPK_RIESGO_VISITA", "EPSILON_GREEDY",
                                    "LINUCB", "THOMPSON_LINEAL", "THOMPSON_BOOTSTRAP_LGBM"])
def test_propension_matriz_es_distribucion_valida(entorno, cortes_test, nombre):
    politica = M.crear_politica(nombre, entorno, seed=SEED)
    pool = entorno["por_corte"][cortes_test[-1]][:30]
    p = politica.propension_matriz(entorno, pool)
    assert p.shape == (len(pool), N_BRAZOS)
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_heuristica_logging_propension_coherente_con_epsilon(entorno, cortes_test):
    politica = M.PoliticaHeuristicaLogging(eps=EPSILON_LOGGING)
    rng = np.random.default_rng(0)
    pool = entorno["por_corte"][cortes_test[-1]][:60]
    arms, propension, tier = politica.elegir(entorno, pool, rng)
    esperado = np.where(arms == tier, (1 - EPSILON_LOGGING) + EPSILON_LOGGING / N_BRAZOS,
                        EPSILON_LOGGING / N_BRAZOS)
    np.testing.assert_allclose(propension, esperado)


# --------------------------------------------------------------------- simulacion + OPE end-to-end
def test_simulacion_online_y_ope_end_to_end(entorno, cortes_test):
    seeds = (42, 7)
    curvas, logueado, congeladas = M.simular_todas(entorno, cortes_test, seeds)
    assert set(curvas["politica"].unique()) == set(M.TODAS_LAS_POLITICAS)
    assert logueado is not None and len(logueado) > 0
    assert {"pos", "arm_idx", "tier_idx", "propension_logueada", "reward"} <= set(logueado.columns)

    resumen = E.resumen_online(curvas)
    assert set(resumen["politica"]) == set(M.TODAS_LAS_POLITICAS)
    assert (resumen["n_semillas"] == len(seeds)).all()

    ope = E.evaluar_off_policy(entorno, logueado, congeladas)
    assert {"HEURISTICA_LOGGING (sanity)", "SIN_CONTACTO", "TOPK_RIESGO_VISITA"} <= set(ope["politica"])
    for col in ("error_rel_ips", "error_rel_snips", "error_rel_dr"):
        assert (ope[col] >= 0).all()
    # sanity check: cuando pi_e = pi_0 (la propia politica de logging), IPS/SNIPS deben reproducir
    # el valor verdadero con error chico (solo ruido de muestreo, sin sesgo de importance sampling)
    fila = ope.set_index("politica").loc["HEURISTICA_LOGGING (sanity)"]
    assert fila["error_rel_ips"] < 0.25
    assert fila["error_rel_snips"] < 0.25


def test_thompson_bootstrap_lgbm_refit_y_eleccion(entorno, cortes_test):
    """Fuerza un refit (refit_cada=1) para probar el camino de entrenamiento LightGBM, que en la
    corrida chica de `test_simulacion_online_y_ope_end_to_end` (8 meses, refit_cada=6, retraso de
    6 meses) nunca alcanza a dispararse."""
    politica = M.PoliticaThompsonBootstrapLGBM(entorno["d_raw"], B=2, refit_cada=1, buffer_max=500, seed=1)
    rng = np.random.default_rng(3)
    pool = entorno["por_corte"][cortes_test[0]][:80]
    arms0 = politica.elegir(entorno, pool, rng)  # sin modelos aun -> exploracion aleatoria
    assert politica.modelos is None
    assert arms0.shape == (len(pool),) and set(arms0) <= set(range(N_BRAZOS))
    rewards = rng.normal(0, 1000, len(pool))
    politica.actualizar(entorno, pool, arms0, rewards)
    politica.tick(0)
    assert politica.modelos is not None and len(politica.modelos) == 2
    arms1 = politica.elegir(entorno, pool, rng)
    assert arms1.shape == (len(pool),)
    p = politica.propension_matriz(entorno, pool)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-8)


def test_baselines_orden_de_costo(entorno, cortes_test):
    """SIN_CONTACTO no gasta nada; TOPK_RIESGO_VISITA gasta el brazo mas caro en todo el pool."""
    seeds = (42,)
    curvas, _, _ = M.simular_todas(entorno, cortes_test, seeds)
    ultimo = curvas[curvas["mes"] == curvas["mes"].max()]
    costo_sin = ultimo.loc[ultimo["politica"] == "SIN_CONTACTO", "costo"].iloc[0]
    costo_topk = ultimo.loc[ultimo["politica"] == "TOPK_RIESGO_VISITA", "costo"].iloc[0]
    assert costo_sin == 0.0
    assert costo_topk > costo_sin
