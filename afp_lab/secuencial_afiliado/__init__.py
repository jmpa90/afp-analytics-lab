"""Proyecto 08 - secuencial_afiliado: red neuronal secuencial multi-tarea (PyTorch CPU).

Pregunta (BRIEF.md): una representacion secuencial (GRU sobre 24 meses de panel) del
afiliado, entrenada multi-tarea, ¿supera a los LightGBM tabulares de 02 (traspaso 6 m) y 04
(cese) en las MISMAS poblaciones, cortes y splits?

Alcance de esta entrega (ver README, "Decisiones tomadas" y "Pendientes"): por presupuesto de
tiempo/computo de la ronda se priorizan las dos cabezas que pide el brief como prioritarias,
traspaso (02) y cese (04); la cabeza de APV (03) y la de RIM (regresion) y la comparacion del
embedding contra la segmentacion de 06 quedan declaradas como pendientes, no implementadas.

Modulos:
    synthetic.py  combina las tablas de 02 (churn_traspaso) y 04 (riesgo_cese_cotizacion),
                  SIN modificarlas (mismo universo determinista: mismo seed/n/perfil/periodo).
    features.py   secuencias as-of (panel hasta corte-2) + features estaticas; construye los
                  datasets de cada cabeza reusando construir_target/agregar_targets/split de
                  origen.
    model.py      encoder GRU compartido + cabezas lineales (multi-tarea y single-task),
                  entrenamiento con torch.set_num_threads(2) y early stopping.
    evaluate.py   metricas de common.metrics (AUC, PR-AUC, lift decil 1) + bootstrap CI.
    run.py        demo end-to-end (python -m afp_lab.secuencial_afiliado.run).
"""
from __future__ import annotations

# Mismo horizonte y desfase que 02 (churn_traspaso.HORIZONTE_MESES/MESES_DESFASE_RIM): la
# secuencia usa panel hasta corte-2 (regla de la RIM, proyecto 01) y el target de traspaso
# vive en corte+1..corte+6.
HORIZONTE_TRASPASO = 6
MESES_DESFASE_RIM = 2
# Horizonte de la cabeza de cese: se usa y6 (cese a 6 meses) de 04, para comparar cabezas con
# el MISMO horizonte (6 meses) que traspaso; y3/reingreso quedan como PENDIENTE (ver README).
HORIZONTE_CESE = 6
VENTANA_SECUENCIA = 24  # meses de historia por secuencia (igual al maximo que usan 02/04)
