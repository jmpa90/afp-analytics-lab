"""Proyecto 07 - Uplift de retencion (a quien conviene contactar).

Ronda 4: no basta con saber quien se va (proyecto 02, `churn_traspaso`); hace falta saber a quien
CONVIENE contactar, es decir el efecto CAUSAL del contacto sobre el traspaso. Modulos:

    synthetic.py  campana historica confundida + holdout aleatorio (10 %) + efecto causal
                  heterogeneo `lat_cate` (tabla latente), sobre el universo y el target de 02.
    features.py   features as-of de 02 (`construir_features_multi`) + historial de la campana
                  propia (conteo 6/12 m, meses desde el ultimo contacto), siempre con datos
                  estrictamente ANTERIORES al corte que se esta puntuando.
    model.py      propension e(x), T-learner, X-learner y DR-learner (LightGBM a mano), y el
                  baseline de riesgo (ranking de 02 usado como si fuera uplift).
    evaluate.py   Qini/AUUC en el holdout aleatorio, correlacion vs `lat_cate`, overlap, y la
                  metrica de negocio (traspasos evitados contactando al 10 %/20 % de la cartera).
    run.py        demo end-to-end: `python -m afp_lab.uplift_retencion.run`

Reutiliza SIN modificar `afp_lab.churn_traspaso`: `synthetic.generar_tablas` (universo + target),
`features.construir_features_multi` / `construir_target` / `cortes_evaluables` / `split_temporal`
(identico `HORIZONTE_MESES` / `MESES_DESFASE_RIM`, para que el target sea IDENTICO al de 02) y
`evaluate.agregar_segmentos` (tramos de negocio). La "verdad" causal (`lat_cate` y las variables
latentes que la generan) vive en `afiliados_latente_uplift` (tabla `*_latente`, nunca feature).
"""
from __future__ import annotations

from afp_lab.churn_traspaso import HORIZONTE_MESES, MESES_DESFASE_RIM  # noqa: F401  (misma definicion)

# Ronda 4 (restriccion dura del gerente, prioridad velocidad): escala de referencia 10.000 afiliados
# (el brief original pedia 20.000, como en 02) y n_jobs=2 en LightGBM (4 vCPU compartidas con otros
# dos workstreams de la ronda). Documentado en el README, seccion "Decisiones tomadas".
N_AFILIADOS_REFERENCIA = 10_000
N_JOBS_LGBM = 2
