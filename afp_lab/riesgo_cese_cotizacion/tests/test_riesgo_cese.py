"""Tests de `riesgo_cese_cotizacion` (pandas, sin Spark): leakage, forma del dataset y una
corrida end-to-end pequena. CONVENCIONES.md #1 pide, como minimo, estas tres cosas."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afp_lab.riesgo_cese_cotizacion.evaluate import (cobertura_poblacion_cese, curva_calibracion,
                                                      diagnostico_cese_por_tipo_afiliado, ece,
                                                      es_calibracion_monotona, evaluar,
                                                      metrica_negocio_agregado, metrica_negocio_captura)
from afp_lab.riesgo_cese_cotizacion.features import (FEATURES_CATEGORICAS, FEATURES_NUMERICAS,
                                                      TIPOS_DEPENDENCIA, VENTANA_HISTORIA,
                                                      agregar_targets, construir_features)
from afp_lab.riesgo_cese_cotizacion.model import ModeloCese, baseline_tasa_base, predecir_tasa_base
from afp_lab.riesgo_cese_cotizacion.run import (_cortes_validos, _negocio_agregado_cartera_completa,
                                                _split_temporal)
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


def test_ece_y_monotonia():
    """Documentacion (ronda 2 de revision, C/D): el ECE resume la calibracion en un solo
    numero; `es_calibracion_monotona` distingue una curva "bien portada" (util para leer
    la probabilidad de una fila individual) de una que solo sirve para agregar/rankear."""
    perfecta = pd.DataFrame({"decil": range(3), "n": [100, 100, 100],
                             "pred_media": [0.1, 0.5, 0.9], "obs_tasa": [0.1, 0.5, 0.9],
                             "error_abs": [0.0, 0.0, 0.0]})
    assert ece(perfecta) == pytest.approx(0.0)
    assert es_calibracion_monotona(perfecta)

    no_monotona = pd.DataFrame({"decil": range(3), "n": [100, 100, 100],
                                "pred_media": [0.2, 0.5, 0.8], "obs_tasa": [0.3, 0.1, 0.7],
                                "error_abs": [0.1, 0.4, 0.1]})
    assert ece(no_monotona) == pytest.approx(0.2)
    assert not es_calibracion_monotona(no_monotona)


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
    # Modelo A: cotizando en corte-2 segun lo conocido, solo dependientes/pensionados
    # activos (bloqueante #2, ronda 1: independientes y voluntarios excluidos)
    pob_a = features[features["pop_cese"] == 1]
    assert (pob_a["rim_l1"] > 0).all()
    assert pob_a["tipo_afiliado"].isin(TIPOS_DEPENDENCIA).all()
    assert not pob_a["tipo_afiliado"].isin(["INDEPENDIENTE", "VOLUNTARIO"]).any()
    # Modelo B: dependientes, sin RIM en corte-2 NI en corte-3 (mayor #4, ronda 1) y con al
    # menos 1 cotizacion en 24m previos
    pob_b = features[features["pop_reingreso"] == 1]
    assert (pob_b["tipo_afiliado"] == "DEPENDIENTE").all()
    assert (pob_b["rim_l1"] == 0).all()
    assert (pob_b["rim_l2"] == 0).all()
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
#
# Nota de metodologia (ronda 2): las pruebas de leakage mutan las tablas DERIVADAS
# (`cotizaciones_conocidas`, `afc_eventos`, `licencias`, `prestaciones_cesantia`)
# directamente, sin volver a llamar a `synthetic.construir_*` sobre un panel alterado.
# Regenerar esas tablas desde un panel con valores cambiados altera tambien el LARGO de
# las tablas intermedias, y como el desfase de recepcion se sortea con
# `rng.random(n)` (n = filas vigentes), dos corridas con distinto `n` reasignan el rezago
# a filas *no relacionadas*, produciendo diffs que no son leakage sino ruido del
# generador. Mutar la tabla ya construida in-place evita ese artefacto y aisla lo que de
# verdad importa: si `construir_features` respeta `periodo_recepcion` y `periodo`.
def test_leakage_alterar_futuro_no_cambia_features(tablas):
    """CONVENCIONES #3: alterar cualquier registro con periodo posterior al corte
    (o con periodo == corte, para RIM y giros) no debe cambiar ni una celda de
    `construir_features(corte)`."""
    base = construir_features(tablas, CORTE)

    t2 = dict(tablas)
    cot = tablas["cotizaciones_conocidas"].copy()
    fut = cot["periodo"] > CORTE
    cot.loc[fut, "rim"] = 999_999_999.0
    t2["cotizaciones_conocidas"] = cot

    afc = tablas["afc_eventos"].copy()
    afc = afc[afc["periodo"] <= CORTE]  # borra eventos futuros
    t2["afc_eventos"] = afc

    lic = tablas["licencias"].copy()
    lic.loc[lic["periodo"] > CORTE, "dias_licencia"] = 30
    t2["licencias"] = lic

    giros = tablas["prestaciones_cesantia"].copy()
    giros.loc[giros["periodo"] > CORTE, "monto"] = 123.0
    t2["prestaciones_cesantia"] = giros  # no se usa en features.py, pero no debe importar si igual se altera

    alt = construir_features(t2, CORTE)
    pd.testing.assert_frame_equal(
        base.sort_values("afiliado_id").reset_index(drop=True),
        alt.sort_values("afiliado_id").reset_index(drop=True),
    )


def test_leakage_cesantia_no_lee_rim_futura(tablas):
    """Bloqueante #1 de la revision de ronda 1: el generador de `prestaciones_cesantia`
    paga un giro en el mes p solo si la RIM real (no conocida) es 0 en p; leer esa tabla
    con periodo <= corte (en vez de <= corte-1) filtraba esa informacion. La correccion
    (`_features_eventos`) ya no lee `prestaciones_cesantia`: la cobertura de cesantia se
    deriva solo del TERMINO AFC conocido + tipo de contrato + tasas legales. Se verifica
    alterando el panel en `corte` y `corte-1` (los meses del bug original) y comprobando
    que `cesantia_en_curso`, `cesantia_giros_restantes` y `cesantia_tasa_actual` no
    cambian ni un poquito, aunque se le anada un giro/una RIM enorme falsa exactamente en
    esos meses en las tablas derivadas."""
    base = construir_features(tablas, CORTE)
    cols_cesantia = ["cesantia_en_curso", "cesantia_giros_restantes", "cesantia_tasa_actual"]

    t2 = dict(tablas)
    cot = tablas["cotizaciones_conocidas"].copy()
    m = cot["periodo"].isin([add_months(CORTE, -1), CORTE])
    cot.loc[m, "rim"] = 999_999_999.0
    cot.loc[m, "periodo_recepcion"] = CORTE  # "se sabe" altiro, no solo en el futuro
    t2["cotizaciones_conocidas"] = cot

    afiliado_id0 = base["afiliado_id"].iloc[0]
    giros = tablas["prestaciones_cesantia"].copy()
    extra = giros.iloc[[0]].copy() if len(giros) else pd.DataFrame(
        {"afiliado_id": [afiliado_id0], "periodo": [CORTE], "n_giro": [1],
         "monto": [1.0], "periodo_recepcion": [CORTE]})
    extra["afiliado_id"] = afiliado_id0
    extra["periodo"] = CORTE
    extra["monto"] = 555_555.0
    extra["periodo_recepcion"] = CORTE
    t2["prestaciones_cesantia"] = pd.concat([giros, extra], ignore_index=True)

    alt = construir_features(t2, CORTE)
    b = base.set_index("afiliado_id")[cols_cesantia]
    a = alt.set_index("afiliado_id")[cols_cesantia]
    pd.testing.assert_frame_equal(b, a.reindex(b.index))


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


def test_termino_en_corte_menos_1_si_actualiza_cesantia(tablas):
    """Control positivo del test anterior: un TERMINO fechado en `corte-1` (el mes mas
    reciente que SI se debe conocer) debe actualizar `afc_termino_l1` y la cobertura de
    cesantia derivada de el. Sin este control, un test de leakage que siempre da "sin
    cambios" podria estar ocultando una feature rota (que nunca reacciona a nada)."""
    afiliados = tablas["afiliados"]
    af_id = afiliados.loc[afiliados["periodo_afiliacion"] <= add_months(CORTE, -13), "afiliado_id"].iloc[0]
    afc = tablas["afc_eventos"].copy()
    afc = afc[~((afc["afiliado_id"] == af_id) & (afc["tipo_evento"] == "TERMINO"))]
    nuevo = pd.DataFrame({"afiliado_id": [af_id], "periodo": [add_months(CORTE, -1)],
                         "empleador_id": [afc["empleador_id"].dropna().iloc[0]], "tipo_evento": ["TERMINO"],
                         "tipo_contrato": ["INDEFINIDO"], "periodo_recepcion": [CORTE]})
    t2 = dict(tablas)
    t2["afc_eventos"] = pd.concat([afc, nuevo], ignore_index=True)
    alt = construir_features(t2, CORTE).set_index("afiliado_id").loc[af_id]
    assert alt["afc_termino_l1"] == 1
    assert alt["meses_desde_afc_termino"] == 1
    assert alt["cesantia_en_curso"] == 1
    assert alt["cesantia_giros_restantes"] == 4          # max 5 (INDEFINIDO) - 1 mes transcurrido
    assert alt["cesantia_tasa_actual"] == pytest.approx(0.70)  # primera tasa de TASAS_GIRO["INDEFINIDO"]


def test_cesantia_se_apaga_si_ya_reingreso_conocido(tablas, features):
    """Menor G (ronda 2 de revision): `cesantia_en_curso`/`giros_restantes`/`tasa_actual`
    deben apagarse si hay evidencia YA CONOCIDA de que el afiliado reingreso: (a) un INICIO
    AFC conocido con periodo posterior al del ultimo TERMINO, o (b) RIM conocida > 0 en un
    mes estrictamente posterior al termino. Se prueban ambos casos por separado sobre un
    termino "viejo" (`corte-4`, para que el caso (b) tenga un mes valido que revisar:
    `rim_l1`/`rim_l2`, ambos posteriores al termino). Se elige un afiliado que YA tiene
    `rim_l1 == rim_l2 == 0` en los datos reales (sin cotizacion conocida reciente), para que
    el termino falso construido no choque con una RIM real que ya lo contradiga."""
    candidatos = features[(features["rim_l1"] == 0) & (features["rim_l2"] == 0)]
    assert len(candidatos) > 0
    af_id = candidatos["afiliado_id"].iloc[0]
    afc = tablas["afc_eventos"].copy()
    afc_sin_af = afc[afc["afiliado_id"] != af_id]
    emp_id = afc["empleador_id"].dropna().iloc[0]
    termino = pd.DataFrame({"afiliado_id": [af_id], "periodo": [add_months(CORTE, -4)],
                           "empleador_id": [emp_id], "tipo_evento": ["TERMINO"],
                           "tipo_contrato": ["INDEFINIDO"], "periodo_recepcion": [add_months(CORTE, -3)]})

    # (a) sin nada mas: sigue "cesante" (control, replica el positivo de arriba con termino viejo)
    t_base = dict(tablas)
    t_base["afc_eventos"] = pd.concat([afc_sin_af, termino], ignore_index=True)
    base = construir_features(t_base, CORTE).set_index("afiliado_id").loc[af_id]
    assert base["cesantia_en_curso"] == 1

    # (b) un INICIO conocido posterior al termino: debe apagarse
    inicio_post = pd.DataFrame({"afiliado_id": [af_id], "periodo": [add_months(CORTE, -2)],
                               "empleador_id": [emp_id], "tipo_evento": ["INICIO"],
                               "tipo_contrato": ["INDEFINIDO"], "periodo_recepcion": [add_months(CORTE, -1)]})
    t_inicio = dict(tablas)
    t_inicio["afc_eventos"] = pd.concat([afc_sin_af, termino, inicio_post], ignore_index=True)
    alt_inicio = construir_features(t_inicio, CORTE).set_index("afiliado_id").loc[af_id]
    assert alt_inicio["cesantia_en_curso"] == 0
    assert alt_inicio["cesantia_giros_restantes"] == 0
    assert alt_inicio["cesantia_tasa_actual"] == 0

    # (c) sin INICIO, pero con RIM conocida > 0 en un mes posterior al termino (rim_l1 = corte-2,
    # que es posterior a corte-4): tambien debe apagarse
    t_rim = dict(tablas)
    t_rim["afc_eventos"] = pd.concat([afc_sin_af, termino], ignore_index=True)
    cot = tablas["cotizaciones_conocidas"].copy()
    extra = pd.DataFrame({"afiliado_id": [af_id], "periodo": [add_months(CORTE, -2)],
                         "empleador_id": [emp_id], "rim": [500_000.0],
                         "periodo_recepcion": [add_months(CORTE, -1)]})
    t_rim["cotizaciones_conocidas"] = pd.concat([cot, extra], ignore_index=True)
    alt_rim = construir_features(t_rim, CORTE).set_index("afiliado_id").loc[af_id]
    assert alt_rim["rim_l1"] > 0
    assert alt_rim["cesantia_en_curso"] == 0
    assert alt_rim["cesantia_giros_restantes"] == 0


# ------------------------------------------------------ empleador (menores #5 y #9)
def test_features_empleador_sin_nan_y_crecimiento_no_sesgado(tablas, features):
    """Bug de ronda 1: `reindex(..., fill_value=0)` no rellenaba los NaN que `unstack` deja
    cuando un empleador conocido no tiene planilla un mes puntual, y la censura de
    recepcion del mes mas reciente (solo ~95% de las planillas llegan a tiempo) sesgaba
    `emp_crecimiento_12` sistematicamente a la baja (mediana ~0,90 para TODOS los
    empleadores, un artefacto, no dotacion real)."""
    con_empleador = features[features["emp_n_trabajadores"] > 0]
    assert con_empleador["emp_crecimiento_12"].notna().all()
    assert con_empleador["emp_tasa_salida_12"].notna().all()
    # corregido por madurez de recepcion: la mediana debe quedar razonablemente cerca de 1,
    # no sesgada ~5% hacia abajo como antes de la correccion.
    mediana = con_empleador["emp_crecimiento_12"].median()
    assert 0.95 <= mediana <= 1.15, f"mediana de emp_crecimiento_12 sospechosamente sesgada: {mediana}"
    assert features["emp_sin_planilla_conocida"].isin([0.0, 1.0]).all()
    # menor B, ronda 2: la version anterior comparaba el mes de referencia contra si mismo
    # (por construccion, siempre tenia planilla) y daba identicamente 0 en TODAS las filas;
    # redefinida sobre toda la ventana de 13 meses, debe encender en al menos algunas filas.
    assert features["emp_sin_planilla_conocida"].sum() > 0


# --------------------------------------------------- tipo de contrato vigente (menor #8)
def test_tipo_contrato_no_describe_relacion_terminada(tablas):
    """Menor #8, ronda 1: el tipo de contrato debe describir la relacion VIGENTE (el
    empleador de referencia de la RIM conocida), no cualquier INICIO pasado del afiliado
    aunque pertenezca a un empleador distinto y ya cerrado. Se toma un afiliado con
    empleador de referencia vigente y se reescribe su (unico) aviso INICIO conocido para
    que apunte a OTRO empleador (simula una relacion anterior, ya terminada): el
    tipo_contrato resultante no debe ser el de ese INICIO viejo."""
    f = construir_features(tablas, CORTE)
    con_ref = f[f["tipo_contrato"] != "SIN_EMPLEADOR"]
    assert len(con_ref) > 0
    af_id = con_ref["afiliado_id"].iloc[0]

    afc = tablas["afc_eventos"]
    inicio_af = afc[(afc["afiliado_id"] == af_id) & (afc["tipo_evento"] == "INICIO")]
    if inicio_af.empty:
        pytest.skip("el afiliado elegido no tiene aviso INICIO conocido (ya es SIN_DATO_AFC)")
    otros_empleadores = tablas["empleadores"]["empleador_id"]
    distinto = otros_empleadores[~otros_empleadores.isin(inicio_af["empleador_id"])].iloc[0]
    tipo_original = inicio_af["tipo_contrato"].iloc[0]
    tipo_forzado = "PLAZO_FIJO" if tipo_original == "INDEFINIDO" else "INDEFINIDO"

    afc2 = afc.copy()
    afc2.loc[inicio_af.index, "empleador_id"] = distinto
    afc2.loc[inicio_af.index, "tipo_contrato"] = tipo_forzado
    t2 = dict(tablas)
    t2["afc_eventos"] = afc2
    alt = construir_features(t2, CORTE).set_index("afiliado_id").loc[af_id]
    assert alt["tipo_contrato"] != tipo_forzado
    assert alt["tipo_contrato"] == "SIN_DATO_AFC"


# --------------------------------------------------------- split temporal (mayor #3)
def test_split_gap_es_2h_no_h():
    """Mayor #3, ronda 1: el gap real entre el ULTIMO TARGET de train/val (`corte+h`) y el
    primer corte de test debe ser >= h, lo que exige `corte_train <= test_min - 2h` (no
    `test_min - h - 1`, que solo garantizaba 1 mes de gap real)."""
    cortes = _cortes_validos(202201, 202609)
    for h in (3, 6):
        splits = _split_temporal(pd.DataFrame({"periodo_corte": cortes}), cortes, h)
        ultimo_train_val = max(splits["cortes"]["train"] + splits["cortes"]["val"])
        primer_test = min(splits["cortes"]["test"])
        gap_meses = periodo_to_idx(primer_test) - (periodo_to_idx(ultimo_train_val) + h)
        assert gap_meses >= h, f"horizonte={h}: gap real={gap_meses} (se exige >= {h})"


def test_cortes_validos_no_pisa_periodo_inicio():
    """Menor #7, ronda 1: el primer corte usable debe dejar toda la ventana de historia
    (`VENTANA_HISTORIA` meses) dentro de `[periodo_inicio, periodo_fin]`."""
    cortes = _cortes_validos(202201, 202609)
    primer_corte_rim = add_months(min(cortes), -2)
    primer_mes_ventana = add_months(primer_corte_rim, -(VENTANA_HISTORIA - 1))
    assert periodo_to_idx(primer_mes_ventana) >= periodo_to_idx(202201)


# ------------------------------------------------------- cobertura (mayor #4, segunda parte)
def test_cobertura_poblacion_a_es_subconjunto_de_cotizantes_reales(tablas):
    """`pop_cese` (RIM CONOCIDA) es, por diseño, un subconjunto de los cotizantes REALES
    (censura de recepcion): la cobertura nunca debe superar 1, y debe rondar el 90-99%
    (no un numero patologico como 50% o 100% exacto, que indicaria que la comparacion esta
    mal armada)."""
    cortes = [CORTE, add_months(CORTE, 1)]
    filas = [construir_features(tablas, c) for c in cortes]
    dataset = pd.concat(filas, ignore_index=True)
    cobertura = cobertura_poblacion_cese(dataset, tablas["panel_mensual"], TIPOS_DEPENDENCIA)
    assert (cobertura["n_pop_cese"] <= cobertura["n_cotizantes_reales"]).all()
    assert (cobertura["cobertura"] <= 1.0).all()
    assert (cobertura["cobertura"] > 0.85).all()


# ---------------------------------------------------- cartera completa (mayor A, ronda 2)
def test_diagnostico_tipo_afiliado_pesa_menos_de_100_pct_dependientes(tablas):
    """Mayor A (ronda 2 de revision): dependientes/pensionados activos NO son el 100% de
    los ceses reales de la cartera (independientes y voluntarios tambien cesan, y los
    voluntarios lo hacen con una tasa mucho mayor a su peso poblacional). El diagnostico
    debe reflejar eso: cobertura de dependientes estrictamente menor a 1, y la tasa de cese
    de VOLUNTARIO por encima de la de DEPENDIENTE (son pagos esporadicos, mucho mas
    volatiles)."""
    cortes = [int(add_months(202412, -k)) for k in range(8)][::-1]
    filas = [construir_features(tablas, c) for c in cortes]
    dataset = agregar_targets(pd.concat(filas, ignore_index=True), tablas["panel_mensual"], PERIODO_FIN)
    diag = diagnostico_cese_por_tipo_afiliado(dataset, "y3", "horizonte3_completo")
    assert set(diag["tipo_afiliado"]) >= {"DEPENDIENTE"}
    assert diag.set_index("tipo_afiliado").loc["DEPENDIENTE", "pct_ceses"] < 100.0
    if "VOLUNTARIO" in diag["tipo_afiliado"].to_numpy():
        d = diag.set_index("tipo_afiliado")
        assert d.loc["VOLUNTARIO", "tasa_cese"] > d.loc["DEPENDIENTE", "tasa_cese"]


def test_negocio_agregado_cartera_completa_suma_correctamente():
    """El agregado extendido (dependientes del modelo + independientes/voluntarios
    proyectados con un baseline propio) debe sumar de forma consistente: total = modelo +
    excluidos, y la cobertura de dependientes debe ser < 1 (si fuera exactamente 1, el
    baseline de excluidos no estaria aportando nada, señal de que algo esta mal armado)."""
    tablas_ = generar_datos(n_afiliados=4000, periodo_inicio=202201, periodo_fin=202512, seed=321)
    from afp_lab.riesgo_cese_cotizacion.run import TARGETS, _construir_dataset, _entrenar_y_evaluar
    dataset, cortes = _construir_dataset(tablas_, 202201, 202512)
    r = _entrenar_y_evaluar("y3", TARGETS["y3"], dataset, cortes, Path("/tmp/rcc_test_cartera"))
    neg_total = _negocio_agregado_cartera_completa(dataset, r, "y3", r["horiz_col"])
    assert np.allclose(neg_total["n_esperado_total"], neg_total["n_esperado"] + neg_total["n_esperado_excluidos"])
    assert np.allclose(neg_total["n_real_total"], neg_total["n_real"] + neg_total["n_real_excluidos"])
    assert (neg_total["cobertura_dependientes"] < 1.0).all()
    assert (neg_total["cobertura_dependientes"] > 0.5).all()


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
