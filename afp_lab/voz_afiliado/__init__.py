"""Proyecto 11 - Voz del afiliado: NPS inferido, drivers y valor del punto.

Pregunta de negocio: solo ~8 % de la cartera es invitada a la encuesta de NPS y de esos responde
~25 %, con sesgo de no respuesta (contestan mas los muy satisfechos y los muy insatisfechos). Este
proyecto:

  1. Infiere el NPS de TODA la cartera a partir de eventos operativos que la AFP ya registra,
     corrigiendo el sesgo de no respuesta por ponderacion inversa a la probabilidad de responder.
  2. Estima el efecto de cada evento sobre el NPS (en puntos) con un T-learner simple.
  3. Asocia el NPS inferido con el traspaso a 6 meses (target de `churn_traspaso`) y el aporte APV a
     6 meses (target de `propension_apv`), para expresar el valor de un punto de NPS.

Modulos:
    synthetic.py  satisfaccion latente + encuesta de NPS, sobre el universo y las tablas de
                  `churn_traspaso`, `propension_apv` y `common/planilla` (reutilizadas sin modificar).
    features.py   features as-of por (afiliado, periodo_corte): reusa `churn_traspaso.features` y
                  `propension_apv.features` y agrega DNP/rezago del empleador e historial de encuesta.
    model.py      modelo de respuesta (LightGBM), modelo de NPS (ponderado y sin ponderar),
                  T-learner de drivers y regresiones de valor del punto.
    evaluate.py   comparacion NPS verdadero/ingenuo/inferido, AUC de detractor, calibracion,
                  tabla de drivers y valor del punto.
    run.py        demo end-to-end (`python -m afp_lab.voz_afiliado.run`).
"""
from __future__ import annotations

HORIZONTE_MESES = 6          # horizonte del target de valor del punto (traspaso / aporte APV)
MESES_DESFASE_RIM = 2        # panel laboral / RIM: igual convencion que 02 y 03
PERIODO_INICIO = 202201
PERIODO_FIN = 202812
N_AFILIADOS_REFERENCIA = 10_000
N_JOBS_LGBM = 2               # maquina de 4 vCPU compartida (ronda 5)
