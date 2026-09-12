# Brief 09 — `afp_lab/bandit_contacto` (worktree `/home/user/wt/empleador`)

## Pregunta
Política de contacto para retención: cada mes, para cada afiliado en riesgo, decidir entre
{sin contacto, email, llamada, visita de ejecutivo, oferta de ahorro/APV}, maximizando la
retención neta de costo, aprendiendo online (bandit contextual) y validando offline antes de
"desplegar" (evaluación off-policy, OPE).

## Simulador (`synthetic.py`)
* Contexto: features as-of de 02 (`churn_traspaso.construir_features_multi`) y, si aporta, el
  segmento de 06; universo `cuprum`.
* Brazos y costos (parámetros documentados como supuestos): sin contacto 0; email 200; llamada
  2 000; visita 15 000; oferta APV 3 000 CLP más costo de la oferta.
* Respuesta verdadera por brazo (tabla latente, `lat_`): probabilidad de traspaso a 6 m bajo cada
  brazo, heterogénea por contexto, con persuadibles, indiferentes y sleeping dogs (reglas
  coherentes con las del brief 07, aunque el generador sea propio). Recompensa = 1{no traspasa
  en 6 m} × comisión anual esperada (1,44 % × RIM × 12, supuesto) − costo del brazo. Recompensa
  observada con retraso de 6 meses (el bandit aprende con rezago, como en la realidad).
* Política histórica de logging: heurística por riesgo con exploración ε = 0,1; guardar la
  propensión de cada acción registrada (necesaria para OPE).
* Restricciones: presupuesto mensual de contactos, máximo un contacto por afiliado cada 3
  meses, exclusión de quienes pidieron no ser contactados.

## Algoritmos (`model.py`)
* ε-greedy, LinUCB, Thompson sampling lineal y Thompson sampling con bootstrap de LightGBM.
* Baselines: "contactar top-k por riesgo con visita" y "sin contacto".
* Simulación online por cortes mensuales (≥ 24 meses) con actualización cuando llega la
  recompensa rezagada; ≥ 5 semillas.

## Evaluación off-policy (`evaluate.py`)
* Sobre los datos logged de la política histórica: IPS, SNIPS y doubly robust para estimar el
  valor de cada política candidata; comparar contra el valor verdadero del simulador (error
  relativo) y mostrar que DR tiene el menor error.

## Métricas y aceptación
* Recompensa neta acumulada, regret frente al oráculo (que conoce `lat_`), tasa de traspaso
  resultante, costo total y contactos a sleeping dogs; intervalos por semillas.
* Aceptación propuesta: Thompson sampling o LinUCB superan a ε-greedy y a top-k-por-riesgo en
  recompensa neta a 24 meses en ≥ 4 de 5 semillas; el estimador DR queda a < 15 % del valor
  verdadero en las políticas candidatas.

## Entregables
Proyecto completo, README con resultados, guardrails y "cómo se desplegaría" (OPE antes,
exploración acotada, monitoreo), tests rápidos, corrida de referencia
`python3 -m afp_lab.bandit_contacto.run` (< 5 min a 10 000 afiliados) e informe estructurado.
