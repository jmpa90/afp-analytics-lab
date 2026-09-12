"""Proyecto 05 - Riesgo de empleador: declaracion y no pago (DNP) y pago tardio.

Modelo de cobranza y de proyeccion de recaudacion. Para cada **empleador** y cada
`periodo_corte` (mes de decision) estima:

* `prob_dnp`   = P(la planilla del periodo `corte` sea DNP o no se declare en `corte+1`),
* `prob_tardio`= P(esa planilla llegue con lag >= 2 o no llegue),
* `prob_dnp_3m`= P(alguna DNP en `corte … corte+2`),
* `masa_en_riesgo_esperada` = `prob_dnp` x masa imponible de referencia del empleador, que es
  el score con que se ordena la cartera de cobranza y el que se agrega para proyectar la masa
  imponible que no entrara este mes.

Modulos:

``synthetic``  tablas propias (planillas, boletines, shocks) sobre `common.synth_base` +
               `common.planilla` (planilla Previred de DS-3) con perfil "cuprum".
``features``   features as-of del empleador (solo planillas con `periodo_recepcion <= corte`).
``model``      LightGBM binario calibrado, baselines y persistencia.
``evaluate``   metricas de `common.metrics` por segmento y por corte + metrica de negocio.
``run``        demo end-to-end (`python -m afp_lab.riesgo_empleador.run --n-afiliados 20000`).
"""
from __future__ import annotations

__all__ = ["synthetic", "features", "model", "evaluate"]
