"""Proyecto 02 - Propension a traspaso a otra AFP (churn de afiliados).

Modulos:
    synthetic.py  tablas sinteticas propias (saldos, traspasos, contactos, reclamos, ...)
    features.py   features as-of por (afiliado, periodo_corte) y armado del dataset
    model.py      LightGBM binario, calibracion isotonica y baselines obligatorios
    evaluate.py   metricas de common/metrics, por segmento, calibracion y metrica de negocio
    run.py        demo end-to-end (python -m afp_lab.churn_traspaso.run)
"""
from __future__ import annotations

HORIZONTE_MESES = 6
MESES_DESFASE_RIM = 2
