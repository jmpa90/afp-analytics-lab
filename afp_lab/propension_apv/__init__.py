"""Proyecto 03 - Propension a ahorro previsional voluntario (APV y cotizacion voluntaria).

Modelo *hurdle* (LightGBM binario + LightGBM de regresion) que estima, para cada afiliado
y cada `periodo_corte`:

* `prob_aporte`  = P(al menos un aporte voluntario en `corte+1 .. corte+6`),
* `monto_condicional` = monto esperado dado que aporta,
* `monto_esperado` = `prob_aporte` x `monto_condicional`, que es lo que se usa para
  priorizar contactos de asesoria y para proyectar el flujo de ahorro voluntario.

Modulos:

``synthetic``  tablas sinteticas propias (aportes, APVC del empleador, campanas, saldos, tributario)
               construidas sobre `afp_lab.common.synth_base.generar_universo`.
``features``   features "as-of" (informacion conocida hasta `periodo_corte`, con el desfase de 2 meses).
``model``      entrenamiento, calibracion, prediccion, persistencia y baselines.
``evaluate``   metricas de `afp_lab.common.metrics` por horizonte, segmento y metrica de negocio.
``run``        demo end-to-end (`python -m afp_lab.propension_apv.run --n-afiliados 20000`).
"""
from __future__ import annotations

__all__ = ["synthetic", "features", "model", "evaluate"]
