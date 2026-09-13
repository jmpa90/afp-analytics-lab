# Revisión ronda 4 · 09 bandit_contacto (bandit contextual + OPE, ε-greedy/LinUCB/Thompson)

**Veredicto: APROBAR** (revisor adversarial Sonnet). Sin bloqueantes: sin leakage hacia las políticas
(las variables `lat_*` solo se usan dentro del simulador/oráculo, nunca en el contexto `X`/`raw` que
ven las políticas), propensión de logging bien guardada y reconstruible exactamente (validado con el
sanity check IPS=SNIPS=valor verdadero sobre la propia política de logging), y las cifras del README
reproducen casi exactamente con el comando exacto del README. El proyecto ya pasó por una corrección
del gerente en esta misma ronda (tipo de respuesta latente independiente del contexto → dependiente
del contexto, commit `f7f7311`/`515dfad`) y la corrección está bien hecha y verificada. Los resultados
principales del brief (Thompson/LinUCB ganan a ε-greedy y top-k en ≥4/5 semillas; DR es el estimador
de menor error) **no se cumplen** tras la corrección, y el README lo reporta con total honestidad en
vez de forzarlo — exactamente lo que pide `docs/briefs/ronda_4/README.md` ("se reportan aunque la
técnica pierda").

## Qué reproduje

- `PYTHONPATH=/home/user/wt/churn python3 -m pytest afp_lab/bandit_contacto/tests -q` →
  **19 passed en 2,38 s** (README: "19 casos, ~2,5 s" — coincide).
- `PYTHONPATH=/home/user/wt/churn python3 -m afp_lab.bandit_contacto.run --n-afiliados 10000 --seeds
  42,7,123,2024,99 --salida /tmp/rev_bandit` (comando exacto del README §9) → **~7,6-8,9 s** (README:
  "~8-9 s" — coincide). Cifras clave reproducidas:
  - Reward neto medio a 24 meses (mil M CLP): THOMPSON_LINEAL 2,1375; EPSILON_GREEDY 2,1302;
    HEURISTICA_LOGGING 2,1158; THOMPSON_BOOTSTRAP_LGBM 2,1086; LINUCB 2,0634; SIN_CONTACTO 2,0354;
    TOPK_RIESGO_VISITA 2,0118 — coincide con la tabla README §6.1 (2,138/2,130/2,116/2,109/2,063/
    2,035/2,012), mismo orden.
  - Regret vs oráculo: 95,2±11,0 / 91,6±15,4 / 116,4±0,3 / 94,5±4,9 / 92,7±1,4 / 89,2±0 / 212,9±0 M —
    coincide dígito a dígito con README §6.1.
  - Contactos a sleeping dogs: 2.230/2.266/2.927/1.680/230/0/2.992 — coincide con README §6.1 y §6.0.
  - OPE: error relativo DR para EPSILON_GREEDY **0,115 %** (README: 0,1 %, el mejor de los 21 pares);
    LINUCB DR 9,33 % (peor de sus tres); sanity `HEURISTICA_LOGGING (sanity)` con
    IPS=SNIPS=218.726, DR=218.667, error ≤1 % en los tres — coincide con README §6.3.
  - Confirmé también que el criterio de aceptación del brief **no se cumple**: THOMPSON_LINEAL gana a
    TOPK en 5/5 pero a EPSILON_GREEDY solo en 3/5 (no ≥4/5); LINUCB nunca gana a EPSILON_GREEDY (0/5).

No re-corrí el barrido `--barrido-fuerza-senal` (14,8 s adicionales, README §6.5) ni la ablación §6.0
completa por presupuesto de llamadas; ambos usan el mismo `run.py`/`model.py` ya ejercitado por la
corrida principal y su mecanismo (`fuerza_senal` multiplicando `TIPO_RESPUESTA_COEF`,
`synthetic.py:84-142`) es directo de leer.

## Sin leakage (verificado a mano, además de los tests)

- `features.py` delega enteramente en `churn_traspaso.construir_features_multi` (as-of ya validado)
  y agrega solo `score_riesgo` (reglas de negocio sobre columnas as-of, `features.py:45-60`) y un
  escalador z-score (`ajustar_escalador`/`transformar_contexto`, líneas 63-78): ninguna columna
  `lat_*`/`p_*` entra a `CONTEXTO_NUMERICO`.
- `synthetic.py`: confirmé con grep que `entorno["lat_p"]` y `entorno["tipo_resp"]` (las columnas
  latentes) solo se leen en `model.py` líneas 411/427 (cómputo de recompensa/regret dentro de
  `simular_politica_online`) y 511/513 (evaluación off-policy), **nunca** dentro de los métodos
  `elegir`/`actualizar`/`propension_matriz` de ninguna política (esos usan `entorno["X"]` o
  `entorno["raw"]`, ambos derivados solo de `CONTEXTO_NUMERICO`). Es decir, la verdad latente
  (`lat_tipo_respuesta`, `lat_p_churn_*`) nunca llega a lo que decide una política, solo al oráculo
  que calcula recompensa/regret — arquitectura correcta.
- Recompensa observada con rezago: `simular_politica_online` (`model.py:396-445`) sortea el traspaso
  en el momento de la decisión con la probabilidad verdadera, pero solo la entrega a
  `politica.actualizar` `HORIZONTE_MESES` (6) después vía la cola `pendientes[corte_idx +
  HORIZONTE_MESES]` — confirmado leyendo el loop, coherente con "recompensa rezagada" del brief.
- Propensión de logging: `PoliticaHeuristicaLogging.elegir` (`model.py:155-175`) devuelve
  `propension = (1-eps)+eps/N_BRAZOS` si el brazo coincide con el tercil elegido, `eps/N_BRAZOS` si
  no — fórmula correcta para una mezcla `(1-eps)·determinista + eps·uniforme(5)`, y coincide
  exactamente con `_propension_real_logueada` en `evaluate.py:187-197` (reconstrucción desde
  `tier_idx`), lo que explica que el sanity check IPS=SNIPS=valor verdadero pase con error ≤1 %.
- Los tres estimadores OPE (`evaluate.py:78-94`) usan las fórmulas estándar (IPS = media de
  `w·r`; SNIPS = normalizado por `Σw`; DR = direct method + corrección IPW), sin errores de signo o
  normalización.

## Hallazgos menores

1. **El escalador (`ajustar_escalador`) se ajusta una sola vez sobre los 24 meses completos de la
   simulación** (`model.py:59`, `preparar_contexto_base`), incluyendo meses futuros respecto de
   decisiones tempranas. No es fuga hacia el target (`y`/reward no interviene, solo media/desvío de
   covariables para estandarizar `X`), y está documentado como decisión en README/`features.py:11-15`,
   pero es un mirar-adelante técnico que en producción no existiría (el escalador de un sistema real
   solo vería datos hasta la fecha de la decisión); efecto esperado mínimo dado que es una
   transformación lineal fija, no debería explicar el resultado del brief.
2. **README §9 ("Cómo correr") todavía referencia el worktree `/home/user/wt/empleador`** (el DS
   original), que ya no es el path vigente tras la fusión a `main`/`ronda-4/revision`
   (`/home/user/wt/churn`, donde de hecho reproduje los comandos sin problema porque son rutas de
   módulo, no del worktree). Falta actualizar antes de cerrar la ronda.
3. **DR no fue uniformemente el estimador de menor error** (solo para EPSILON_GREEDY; peor que IPS y
   SNIPS para LinUCB y Thompson-LGBM) y **Thompson/LinUCB no superan a ε-greedy en ≥4/5 semillas**:
   ambos son resultados honestos y bien reportados (README §6.1/§6.3/§8), no maquillados, pero el
   gerente debe leerlos como "el bandit contextual, con esta calibración de señal y 24 meses, no
   justifica reemplazar la heurística todavía" (recomendación explícita del propio README, §7) — no
   es un hallazgo de revisión sino la lectura de negocio que hay que trasladar tal cual.

## Checklist §7 CONVENCIONES

- [x] Sin leakage: verificado a mano (arriba); `lat_*` nunca llega a las políticas.
- [x] Target/recompensa bien definida, con restricciones (presupuesto, frecuencia, opt-out) aplicadas
  antes de la elección de brazo.
- [x] Dinámicas sintéticas documentadas con parámetros explícitos y corregidas tras revisión previa.
- [x] Baselines presentes (sin contacto, top-k); los candidatos NO los superan de forma consistente
  con el criterio del brief, y el README lo declara sin maquillar.
- [x] Métricas comparables (reward neto, regret, OPE) + métrica de negocio.
- [x] Runtime y tests dentro de límites; reproducible con semillas fijas.
- [x] README completo según plantilla, incluida sección de guardrails/despliegue.
