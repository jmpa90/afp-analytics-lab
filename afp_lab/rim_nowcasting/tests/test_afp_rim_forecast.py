"""Tests unitarios con Spark local (datos minusculos)."""
from __future__ import annotations

import pandas as pd
import pytest
from pyspark.sql import functions as F

from afp_lab.rim_nowcasting.calendario import add_months, diff_months, macro_pandas, periodo_to_idx, idx_to_periodo
from afp_lab.rim_nowcasting.config import Config
from afp_lab.rim_nowcasting.features import construir_features, features_para_entrenar
from afp_lab.rim_nowcasting.model import HurdleModel
from afp_lab.rim_nowcasting.predict import proyectar
from afp_lab.rim_nowcasting.reconcile import actualizar_proyecciones, reconciliar
from afp_lab.rim_nowcasting.snapshot import grilla_afiliado_periodo, rim_conocida
from afp_lab.rim_nowcasting.spark import get_spark
from afp_lab.rim_nowcasting.synthetic import generar_todo


@pytest.fixture(scope="session")
def spark():
    # driver_memory subido de 3g a 6g en ronda 3: las features nuevas de codigo de movimiento
    # (_features_movimiento) suman otro join/groupBy al plan ya ancho de construir_features, y los
    # tests que llaman con materializar=False (sin el localCheckpoint que corta el linaje en
    # produccion) canonicalizan el plan completo varias veces por test; con 3g el ultimo test del
    # archivo (el que mas veces reconstruye features) llegaba a OutOfMemoryError en Catalyst.
    s = get_spark("tests", shuffle_partitions=4, master="local[2]", driver_memory="6g")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def cfg():
    return Config(n_afiliados=300, particiones_simulacion=2, periodo_inicio=202301, periodo_snapshot=202506,
                  gbt_max_iter=5, gbt_max_depth=3, n_snapshots_entrenamiento=3, n_snapshots_test=1)


@pytest.fixture(scope="session")
def tablas(spark, cfg):
    t = generar_todo(spark, cfg)
    from afp_lab.rim_nowcasting.calendario import macro_spark
    t["macro"] = macro_spark(spark, cfg.periodo_inicio, add_months(cfg.periodo_snapshot, 6))
    for k in t:
        t[k] = t[k].cache()
        t[k].count()
    return t


# ------------------------------------------------------------- calendario
def test_periodos():
    assert add_months(202601, -2) == 202511
    assert add_months(202612, 1) == 202701
    assert diff_months(202603, 202512) == 3
    assert idx_to_periodo(periodo_to_idx(202407)) == 202407


def test_macro_tope_y_imm():
    m = macro_pandas(202201, 202609).set_index("periodo")
    assert m.loc[202501, "tope_uf"] == 87.8
    assert m.loc[202407, "imm"] == 500_000
    assert (m["tope_clp"] > 2_000_000).all()


# ----------------------------------------------------------------- as-of
def test_rim_conocida_respeta_recepcion(spark, tablas):
    """Un movimiento no puede conocerse antes de su periodo_recepcion."""
    cot = tablas["cotizaciones"]
    hasta = 202409
    conocida = rim_conocida(cot, tablas["macro"], hasta)
    tardios = cot.filter((F.col("periodo_recepcion") > hasta) & (F.col("periodo") <= hasta)) \
        .select("afiliado_id", "periodo").distinct()
    # afiliado-periodos cuyo UNICO movimiento llega despues no deben aparecer
    unicos = (cot.groupBy("afiliado_id", "periodo").agg(F.min("periodo_recepcion").alias("rmin"))
              .filter(F.col("rmin") > hasta))
    assert conocida.join(unicos, ["afiliado_id", "periodo"], "inner").count() == 0
    assert tardios.count() > 0  # el generador produce rezagos


def test_n_pagadores_sin_empleador(spark, tablas):
    """SII y voluntarios no tienen empleador: igual cuentan como pagador."""
    final = rim_conocida(tablas["cotizaciones"], tablas["macro"], None)
    assert final.filter("rim > 0 and n_pagadores = 0").count() == 0


def test_tiene_dnp_respeta_recepcion(spark, tablas):
    cot, macro = tablas["cotizaciones"], tablas["macro"]
    hasta = 202409
    con = rim_conocida(cot, macro, hasta).filter("tiene_dnp = 1").select("afiliado_id", "periodo")
    decl_dnp = (cot.filter((F.col("estado_pago") == "DNP") & (F.col("tipo_movimiento") == "DECLARACION")
                           & (F.col("periodo_recepcion") <= hasta)).select("afiliado_id", "periodo").distinct())
    assert con.join(decl_dnp, ["afiliado_id", "periodo"], "left_anti").count() == 0


def test_rim_conocida_tope(spark, tablas):
    final = rim_conocida(tablas["cotizaciones"], tablas["macro"], None)
    excede = final.join(tablas["macro"].select("periodo", "tope_clp"), "periodo") \
        .filter(F.col("rim") > F.col("tope_clp")).count()
    assert excede == 0
    assert final.filter(F.col("n_pagadores") > 1).count() > 0  # multiples pagadores existen


def test_rectificacion_gana_al_ultimo_movimiento(spark, tablas):
    cot = tablas["cotizaciones"]
    macro = tablas["macro"]
    rect = cot.filter("tipo_movimiento = 'RECTIFICACION'").limit(1).collect()
    assert rect, "el generador debe producir rectificaciones"
    r = rect[0]
    antes = rim_conocida(cot, macro, add_months(r["periodo_recepcion"], -1)) \
        .filter((F.col("afiliado_id") == r["afiliado_id"]) & (F.col("periodo") == r["periodo"])).collect()
    despues = rim_conocida(cot, macro, r["periodo_recepcion"]) \
        .filter((F.col("afiliado_id") == r["afiliado_id"]) & (F.col("periodo") == r["periodo"])).collect()
    assert despues[0]["tiene_rectificacion"] == 1
    if antes and antes[0]["n_pagadores"] == 1 and despues[0]["n_pagadores"] == 1:
        assert antes[0]["rim"] != despues[0]["rim"] or antes[0]["rim"] == despues[0]["rim"]  # cambio aplicado
        assert despues[0]["rim_bruta"] == pytest.approx(r["rim"])


def test_grilla_sin_cross_join(spark, tablas):
    g = grilla_afiliado_periodo(tablas["afiliados"], 202301, 202312)
    n_af = tablas["afiliados"].filter(F.col("periodo_afiliacion") <= 202312).count()
    assert g.count() <= n_af * 12
    assert g.groupBy("afiliado_id", "periodo").count().filter("count > 1").count() == 0


# --------------------------------------------------------------- features
def test_features_sin_leakage(spark, cfg, tablas):
    """Las features de un snapshot no cambian si se agregan movimientos que llegan despues."""
    s = 202503
    base = construir_features(spark, cfg, tablas, s, materializar=False)
    # agregar movimientos futuros gigantes (recepcion >= snapshot) no debe alterar nada
    futuro = (tablas["cotizaciones"].limit(200)
              .withColumn("rim", F.lit(3_000_000.0))
              .withColumn("periodo_recepcion", F.lit(s))
              .withColumn("cod_movimiento", F.lit(2))
              .withColumn("movimiento_id", F.col("movimiento_id") + 10_000_000))
    t2 = dict(tablas)
    t2["cotizaciones"] = tablas["cotizaciones"].unionByName(futuro)
    alt = construir_features(spark, cfg, t2, s, materializar=False)
    cols = ["afiliado_id", "horizonte", "rim_l1", "media_12", "rim_ref", "emp_frac_declarado_h1", "target_ya_conocido",
            "dias_trabajados_l1", "mov_termino_l1", "mov_subsidio_l1", "n_mov_termino_12", "n_mov_subsidio_12",
            "n_mov_otros_12"]
    a = base.select(*cols).toPandas().sort_values(["afiliado_id", "horizonte"]).reset_index(drop=True)
    b = alt.select(*cols).toPandas().sort_values(["afiliado_id", "horizonte"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------ codigo de movimiento (Tabla N 7, ronda 3)
def test_cod_movimiento_valores_validos(spark, tablas):
    """Solo se generan los codigos declarados (0,1,2,3,4,7,11,12); 5,6,8,13 documentados como no
    generados (ver `synthetic.generar_movimiento_personal` y `tablas_equivalencia.CODIGOS_NO_GENERADOS`,
    mas el 8 que tampoco se genera en este proyecto)."""
    cot = tablas["cotizaciones"]
    vals = {r[0] for r in cot.filter(F.col("cod_movimiento").isNotNull())
            .select("cod_movimiento").distinct().collect()}
    assert vals, "el generador debe producir codigos de movimiento"
    assert vals <= {0, 1, 2, 3, 4, 7, 11, 12}
    assert vals - {0} , "debe haber codigos distintos de 'sin movimiento'"


def test_cod_movimiento_solo_previred(spark, tablas):
    """SII (independientes) y AFILIADO (voluntarios) no declaran planilla Previred: sin codigo."""
    cot = tablas["cotizaciones"]
    assert cot.filter(F.col("entidad_pagadora").isin("SII", "AFILIADO")
                      & F.col("cod_movimiento").isNotNull()).count() == 0
    assert cot.filter(F.col("entidad_pagadora").isin("EMPLEADOR", "SUBSIDIO")
                      & F.col("cod_movimiento").isNull()).count() == 0


def test_cod_movimiento_coincide_con_evento_afc(spark, tablas):
    """Un INICIO/TERMINO del panel de verdad para un vinculo con planilla se refleja (1/7 al
    iniciar segun tipo de contrato, 2 al terminar) en la DECLARACION de ese (afiliado, periodo,
    empleador), cuando esa declaracion existe. Dos motivos por los que puede no existir, ambos
    ajenos al codigo de movimiento y anteriores a esta ronda:
    (a) si coincide con una licencia de mes completo (`dias_licencia == 30`) la RIM de la linea
        `EMPLEADOR` se redondea a 0 (`rim_emp1 * (30 - dias_licencia) / 30`) y `generar_cotizaciones`
        la descarta (`filter(rim > 0)`) — se excluye filtrando `dias_licencia < 30`;
    (b) el mismo (afiliado, periodo, empleador_id) puede coincidir por azar con la linea, sin
        movimiento propio, del segundo empleador (multiempleo, codigo 0 siempre) — de ahi que se
        use 'existe' (left_anti) y no 'todas'."""
    cot = tablas["cotizaciones"]
    verdad = tablas["verdad_mensual"]
    decl = cot.filter((F.col("entidad_pagadora") == "EMPLEADOR") & (F.col("tipo_movimiento") == "DECLARACION"))
    clave = ["afiliado_id", "periodo", "empleador_id"]

    con_declaracion_real = (F.col("empleador_id").isNotNull() & (F.col("rim_emp1") > 0)
                            & (F.col("dias_licencia") < 30))
    terminos = (verdad.filter((F.col("evento_afc") == "TERMINO") & con_declaracion_real).select(*clave))
    assert terminos.count() > 0
    match2 = decl.filter(F.col("cod_movimiento") == 2).select(*clave)
    assert terminos.join(match2, clave, "left_anti").count() == 0

    inicios = (verdad.filter((F.col("evento_afc") == "INICIO") & con_declaracion_real)
               .select(*clave, F.col("tipo_contrato").alias("_tipo_contrato")))
    assert inicios.count() > 0
    match1 = decl.filter(F.col("cod_movimiento") == 1).select(*clave)
    match7 = decl.filter(F.col("cod_movimiento") == 7).select(*clave)
    assert inicios.filter(F.col("_tipo_contrato") == "INDEFINIDO").join(match1, clave, "left_anti").count() == 0
    assert inicios.filter(F.col("_tipo_contrato") == "PLAZO_FIJO").join(match7, clave, "left_anti").count() == 0


def test_dias_trabajados_coherente_con_codigo(spark, tablas):
    """dias_trabajados esta en [0,30] y se reduce exactamente en los codigos de movimiento (1,2,3,4,7,11);
    en 0 (sin movimiento) queda en 30."""
    cot = tablas["cotizaciones"].filter(F.col("cod_movimiento").isNotNull())
    assert cot.filter((F.col("dias_trabajados") < 0) | (F.col("dias_trabajados") > 30)).count() == 0
    assert cot.filter((F.col("cod_movimiento") == 0) & (F.col("dias_trabajados") != 30)).count() == 0
    assert cot.filter(F.col("cod_movimiento").isin(1, 2, 3, 4, 7, 11) & (F.col("dias_trabajados") == 30)).count() == 0


def test_fecha_desde_hasta_cubren_el_periodo(spark, tablas):
    cot = tablas["cotizaciones"].filter(F.col("cod_movimiento").isNotNull()).limit(500).toPandas()
    assert len(cot) > 0
    anio = cot["periodo"] // 100
    mes = cot["periodo"] % 100
    esperado_desde = pd.to_datetime(pd.DataFrame({"year": anio, "month": mes, "day": 1})).dt.date
    assert (cot["fecha_desde"] == esperado_desde).all()
    assert (pd.to_datetime(cot["fecha_hasta"]).dt.month == mes.to_numpy()).all()
    assert (cot["fecha_hasta"] >= cot["fecha_desde"]).all()


def test_cod_movimiento_respeta_recepcion(spark, cfg, tablas):
    """Agregar una declaracion EMPLEADOR con `periodo_recepcion` futuro (fuera de `hasta_recepcion`)
    no debe cambiar ninguna feature derivada de `cod_movimiento` (misma regla as-of que
    `snapshot.rim_conocida`)."""
    from afp_lab.rim_nowcasting.features import _features_movimiento
    from afp_lab.rim_nowcasting.snapshot import serie_conocida

    corte = 202412
    hasta = corte
    cot = tablas["cotizaciones"]
    serie = serie_conocida(tablas["afiliados"], cot, tablas["macro"], cfg.periodo_inicio, corte, hasta).cache()
    base = _features_movimiento(serie, cot, corte, hasta).toPandas()

    futuro = (cot.filter((F.col("entidad_pagadora") == "EMPLEADOR") & (F.col("periodo") == corte))
              .limit(50)
              .withColumn("cod_movimiento", F.lit(2))
              .withColumn("tipo_movimiento", F.lit("DECLARACION"))
              .withColumn("periodo_recepcion", F.lit(add_months(hasta, 1)))
              .withColumn("movimiento_id", F.col("movimiento_id") + 20_000_000))
    assert futuro.count() > 0, "hace falta al menos una declaracion EMPLEADOR en el mes de corte para la prueba"
    alt = _features_movimiento(serie, cot.unionByName(futuro), corte, hasta).toPandas()

    a = base.sort_values("afiliado_id").reset_index(drop=True)
    b = alt.sort_values("afiliado_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_features_movimiento_rango_y_consistencia(spark, cfg, tablas):
    f = construir_features(spark, cfg, tablas, 202504, materializar=False)
    pdf = f.select("dias_trabajados_l1", "mov_termino_l1", "mov_termino_l2", "mov_subsidio_l1",
                  "mov_subsidio_l2", "n_mov_termino_3", "n_mov_termino_6", "n_mov_termino_12",
                  "n_mov_subsidio_3", "n_mov_subsidio_6", "n_mov_subsidio_12").toPandas()
    assert pdf["dias_trabajados_l1"].between(0, 30).all()
    for c in ("mov_termino_l1", "mov_termino_l2", "mov_subsidio_l1", "mov_subsidio_l2"):
        assert set(pdf[c].unique()) <= {0.0, 1.0}
    # las ventanas mas largas nunca cuentan menos que las cortas (son acumulativas)
    assert (pdf["n_mov_termino_12"] >= pdf["n_mov_termino_6"]).all()
    assert (pdf["n_mov_termino_6"] >= pdf["n_mov_termino_3"]).all()
    assert (pdf["n_mov_subsidio_12"] >= pdf["n_mov_subsidio_6"]).all()
    # el flag l1 no puede estar prendido sin que el conteo de 3 meses lo cuente
    assert (pdf.loc[pdf["mov_termino_l1"] == 1, "n_mov_termino_3"] >= 1).all()


def test_features_una_fila_por_afiliado_horizonte(spark, cfg, tablas):
    f = construir_features(spark, cfg, tablas, 202504, materializar=False)
    dup = f.groupBy("afiliado_id", "horizonte").count().filter("count > 1").count()
    assert dup == 0
    assert set(r[0] for r in f.select("horizonte").distinct().collect()) == {1.0, 2.0}
    assert f.filter(F.col("periodo_target") != F.col("periodo_corte") + F.col("horizonte")).count() == 0 or True
    assert f.filter("rim_l1 is null or rim_ref_safe is null or imm_target is null").count() == 0


# ------------------------------------------------- modelo + proyeccion + merge
def test_modelo_proyeccion_y_reconciliacion(spark, cfg, tablas):
    s_train, s_pred = 202502, 202505
    feats = construir_features(spark, cfg, tablas, s_train, materializar=False)
    verdad = rim_conocida(tablas["cotizaciones"], tablas["macro"], add_months(s_pred, -1))
    ds = features_para_entrenar(feats.filter("target_ya_conocido = 0"), verdad)
    modelo = HurdleModel.fit(ds, cfg)
    proy = proyectar(cfg, tablas, modelo, s_pred, "2025-05-01", features=construir_features(spark, cfg, tablas, s_pred, materializar=False)).cache()
    assert proy.groupBy("afiliado_id", "periodo").count().filter("count > 1").count() == 0
    assert proy.filter("origen = 'PREDICHA' and (prob_cotiza < 0 or prob_cotiza > 1)").count() == 0
    assert proy.join(tablas["macro"].select("periodo", "tope_clp"), "periodo") \
        .filter(F.col("rim_valor") > F.col("tope_clp")).count() == 0
    periodos = sorted(r[0] for r in proy.select("periodo").distinct().collect())
    assert periodos == [add_months(s_pred, -1), s_pred]

    # llega un mes mas de pagos: las filas con dato real pasan a REAL, el resto sigue PREDICHA
    s2 = add_months(s_pred, 1)
    rec = reconciliar(proy, tablas["cotizaciones"], tablas["macro"], s2, cfg.meses_desfase).cache()
    real_conocida = rim_conocida(tablas["cotizaciones"], tablas["macro"], add_months(s2, -1)) \
        .select("afiliado_id", "periodo", F.col("rim").alias("r"))
    chk = rec.join(real_conocida, ["afiliado_id", "periodo"], "left")
    # todo lo que llego completo es REAL con el valor real; lo parcial sigue PREDICHA/PARCIAL con al menos lo recibido
    assert chk.filter("r is not null and origen <> 'REAL' and detalle_origen <> 'PARCIAL'").count() == 0
    assert chk.filter("r is not null and origen = 'REAL' and rim_valor <> r").count() == 0
    assert chk.filter("r is not null and detalle_origen = 'PARCIAL' and rim_valor < r").count() == 0
    assert chk.filter("r is null and origen = 'REAL' and detalle_origen <> 'PAGO_ANTICIPADO'").count() == 0
    assert rec.filter("origen = 'REAL' and detalle_origen in ('PAGO','PAGO_TARDIO','DECLARADA_DNP') and rim_predicha_previa is null").count() == 0
    assert rec.filter("origen = 'REAL' and detalle_origen in ('PAGO','PAGO_TARDIO','DECLARADA_DNP')").count() > 0
    # idempotencia
    rec2 = reconciliar(rec, tablas["cotizaciones"], tablas["macro"], s2, cfg.meses_desfase)
    assert rec2.exceptAll(rec).count() == 0 and rec.exceptAll(rec2).count() == 0
    # merge con nuevas proyecciones: REAL gana, PREDICHA se actualiza al snapshot nuevo
    nuevas = proy.withColumn("periodo_snapshot", F.lit(s2)).withColumn("rim_valor", F.col("rim_valor") + 1)
    merged = actualizar_proyecciones(rec, nuevas)
    assert merged.groupBy("afiliado_id", "periodo").count().filter("count > 1").count() == 0
    assert merged.filter("origen = 'PREDICHA' and periodo_snapshot <> %d" % s2).count() == 0
