# 09 · `bandit_contacto` — bandit contextual de contacto para retencion

Ronda 4. AFP Cuprum, datos 100% sinteticos. Sigue la plantilla de `CONVENCIONES.md` §6.

## 1. Pregunta de negocio y decision que apoya

Cada mes, para cada afiliado identificado como "en riesgo" de traspaso: decidir entre **{sin
contacto, email, llamada, visita de ejecutivo, oferta de ahorro/APV}** para maximizar la
**retencion neta de costo** (comision retenida menos el costo del contacto). Se aprende **online**
(bandit contextual, actualizando cuando llega la recompensa con 6 meses de rezago) y se audita
**offline** antes de tocar produccion (evaluacion off-policy: IPS, SNIPS, doubly robust).

Decide: si vale la pena reemplazar la regla actual ("visitar a todo el top-riesgo") por un bandit
que aprenda a quien, por que canal y con que oferta contactar -- y si se puede confiar en medir eso
con datos ya logueados antes de arriesgar presupuesto real.

## 2. Datos sinteticos

### 2.1 Universo y tablas reutilizadas (NO modificadas)

`synthetic.generar_tablas_bandit` llama a `afp_lab.churn_traspaso.synthetic.generar_tablas(...,
perfil="cuprum")` tal cual: mismo universo comun (`afp_lab.common.synth_base.generar_universo`),
mismas tablas operacionales (`saldos`, `contactos`, `reclamos`, `uso_digital`, `cambios_fondo`,
`mercado`, `traspasos`, `licitacion_stock`). El "contacto de retencion" que decide el bandit es un
mecanismo propio, **separado** de la tabla `contactos` de 02 (que sigue representando el historial
de contactos "de negocio como siempre" -- ejecutivo propio, vendedor de la competencia, call center,
campana digital -- y alimenta features as-of como `contactos_ejecutivo_1m`). Simplificacion
documentada: el contexto (features as-of) es **exogeno** a la simulacion del bandit -- no se
recalcula cada mes con las propias decisiones del bandit -- para no tener que rehacer
`construir_features_multi` en cada paso; ver §8.

Ventana de referencia: **2022-01 a 2026-12** (60 meses; mas corta que los 96 meses por defecto de
02 porque aqui solo se necesitan burn-in (14 m) + 24 m de simulacion + 6 m de resolucion de la
ultima recompensa, con margen).

### 2.2 Tablas propias (`synthetic.py`)

* **`preferencias_contacto`** (OBSERVABLE, no latente): afiliados que pidieron no ser contactados.
  `P_OPT_OUT = 0,03` (supuesto: 3% de la cartera). Persistente en el tiempo (simplificacion: en la
  realidad estas solicitudes llegan de a poco; aqui alcanza con un flag fijo por afiliado).
* **`respuesta_latente_tipo`** (LATENTE, columna `lat_tipo_respuesta`): tipo de respuesta al
  contacto, fijo por afiliado, mezcla `P_TIPO_RESPUESTA = (0,30 PERSUADIBLE; 0,45 INDIFERENTE;
  0,25 SLEEPING_DOG)` -- coherente con la heterogeneidad del brief de uplift de la ronda 4
  (generador propio, mismos tres arquetipos).
* **`calcular_respuesta_latente`** (funcion LATENTE, no una tabla persistida en `salidas/`: se
  recalcula desde `synthetic.py` para mantener el generador determinista y liviano): probabilidad
  VERDADERA de traspaso a 6 meses por brazo, `lat_p_churn_<BRAZO>`. Riesgo base logistico:

  ```
  logit0 = -2.6 + 1.10*brecha_comision_rel + 0.55*log1p(reclamos_6m)
                + 0.65*log1p(contactos_competencia_interesado_6m)
                - 0.35*densidad_12 - 0.10*log1p(antiguedad_afp_meses) + ruido~N(0, 0.55^2)
  ```

  mas un `DELTA_LOGODDS[tipo][brazo]` (log-odds) segun el tipo latente:

  | tipo | EMAIL | LLAMADA | VISITA | APV |
  |---|---|---|---|---|
  | PERSUADIBLE (retiene) | -0,10 | -0,35 | -0,55 | -0,65 |
  | INDIFERENTE (~sin efecto) | -0,02 | -0,03 | -0,03 | -0,05 |
  | SLEEPING_DOG (empeora) | +0,05 | +0,25 | +0,45 | +0,10 |

  El ruido gaussiano evita que el riesgo sea perfectamente predecible (CONVENCIONES §2). Probabilidad
  final recortada a `[0,01; 0,6]`.

### 2.3 Supuestos economicos (declarados, `__init__.py`)

| Parametro | Valor | Razon (Chile/AFP) |
|---|---|---|
| Costo EMAIL | 200 CLP | envio automatizado (brief) |
| Costo LLAMADA | 2.000 CLP | call center (brief) |
| Costo VISITA | 15.000 CLP | visita de ejecutivo, el brazo mas caro (brief) |
| Costo APV | 3.000 + `COSTO_OFERTA_APV_CLP` (7.000) = 10.000 CLP | "3.000 CLP mas costo de la oferta" (brief); el incentivo/beneficio de la oferta APV es un **SUPUESTO propio** (7.000 CLP) |
| `TASA_COMISION_ANUAL` | 1,44% | comision anual sobre la RIM (brief) |
| `PRESUPUESTO_MENSUAL_CONTACTOS` | 400 (4% de 10.000) | tope operativo mensual (supuesto propio, ver §8) |
| `MESES_ENTRE_CONTACTOS` | 3 | maximo un contacto pagado cada 3 meses (brief) |
| `EPSILON_LOGGING` | 0,10 | exploracion de la politica historica (brief) |
| `P_OPT_OUT` | 0,03 | fraccion que pidio no ser contactada (supuesto propio) |

Recompensa por decision: `1{no traspasa en 6m} * (0,0144 * RIM_ref * 12) - costo_brazo` (formula
exacta del brief). Se sortea con la probabilidad VERDADERA en el momento de la decision (el
simulador/oraculo lo sabe de inmediato) pero la politica solo la observa 6 meses despues
(`actualizar`), como en la realidad.

## 3. Poblacion, horizonte y exclusiones

* **Poblacion "en riesgo" de cada mes** = el top `PRESUPUESTO_MENSUAL_CONTACTOS` (400) por
  `score_riesgo` (heuristica de negocio, §4) entre los afiliados vigentes, **excluyendo** opt-out y
  a quien se contacto (brazo pagado) hace menos de `MESES_ENTRE_CONTACTOS`. Decision: **el
  presupuesto define directamente el tamano del pool** (en vez de un pool mas grande + un segundo
  paso de racionamiento) -- ver §8.
* **Horizonte**: 6 meses (`HORIZONTE_MESES`, igual al target de 02 por diseno del brief).
* **Ventana simulada**: 24 cortes mensuales consecutivos, `2024-07` a `2026-06` (dentro de los 60
  meses generados, con 14 m de burn-in antes y 6 m de margen para resolver la ultima recompensa).

## 4. Features (contexto as-of)

Contexto = `afp_lab.churn_traspaso.features.construir_features_multi` (reutilizado tal cual, mismas
reglas as-of de CONVENCIONES §3: panel laboral/RIM en `corte-2`, tablas operacionales en `corte`,
nunca informacion de `corte+1` en adelante). Dos derivados propios (NO son `lat_`/`p_`, son reglas
de negocio sobre columnas as-of):

| Familia | Ejemplos | Regla as-of |
|---|---|---|
| Laboral/RIM (02) | `rim_l1`, `rim_media_12`, `densidad_12`, `antiguedad_afp_meses` | `corte-2` |
| Cuenta/mercado (02) | `log_saldo`, `brecha_comision_rel`, `brecha_rentabilidad_pp` | `corte` (mercado se conoce al dia) |
| Relacion con la AFP (02) | `reclamos_6m`, `contactos_competencia_interesado_6m`, `logins_3m` | `corte` |
| `score_riesgo` (propio) | combinacion de negocio (brecha comision, reclamos, interes por competencia, densidad, antiguedad) | igual que sus insumos, `corte` |
| `CONTEXTO_NUMERICO` (propio) | 16 columnas numericas + bias, estandarizadas (z-score) | mismas columnas as-of; el escalador se ajusta una vez sobre todo el contexto de la simulacion (transformacion de covariables, no del target: no hay fuga) |

`score_riesgo` es **deliberadamente distinto** de `calcular_respuesta_latente` (pesos y variables
distintos, sin ruido): una AFP real tampoco conoce la propension verdadera, solo reglas razonables.

## 5. Modelo y baselines (`model.py`)

| Politica | Tipo | Nota |
|---|---|---|
| `SIN_CONTACTO` | baseline | nunca contacta |
| `TOPK_RIESGO_VISITA` | baseline de negocio | visita a todo el pool (top-K por riesgo) |
| `HEURISTICA_LOGGING` | **politica historica** | terciles de riesgo -> {VISITA, LLAMADA, EMAIL} + exploracion `eps=0,10` uniforme sobre los 5 brazos; guarda su propension exacta (`propension_logueada`) y el brazo por tercil (`tier_idx`), necesarios para IPS/SNIPS/DR |
| `EPSILON_GREEDY` | candidato | **no contextual** (promedio de recompensa por brazo, clasico), `eps=0,10` |
| `LINUCB` | candidato | ridge lineal por brazo + banda de confianza, `alpha=1,0` |
| `THOMPSON_LINEAL` | candidato | regresion lineal bayesiana por brazo, muestreo de `theta` por fila |
| `THOMPSON_BOOTSTRAP_LGBM` | candidato | ensamble bootstrap de `B=3` LightGBM (`n_jobs=2`, `n_estimators=60`, `max_depth=4`) con el brazo como feature extra (un solo modelo por miembro, no 5 x B); refit cada 6 meses sobre las ultimas 4.000 filas resueltas |

Simulacion online (`simular_politica_online`/`simular_todas`): cada politica corre su **propia**
trayectoria de 24 meses (su propio "quien fue contactado hace <3 meses"), sobre el **mismo** mundo
fijo (contexto + respuesta latente, generados una sola vez). Las 5 semillas varian solo el azar de
exploracion/observacion de cada politica, no el universo.

## 6. Resultados

*Corrida de referencia: 10.000 afiliados, perfil `cuprum`, 24 meses (2024-07 a 2026-06), 5 semillas
(42, 7, 123, 2024, 99), presupuesto 400 contactos/mes → 9.600 decisiones logueadas.*

### 6.1 Recompensa neta acumulada y regret a 24 meses (media ± IC95% entre 5 semillas)

| politica | reward neto (CLP) | regret vs oraculo | costo total | tasa traspaso | contactos a sleeping dogs |
|---|---|---|---|---|---|
| **THOMPSON_LINEAL** | **2.153 ± 0.018 mil M** | 91,0 ± 4,2 M | 43,2 M | 30,1% | 1.860 |
| EPSILON_GREEDY | 2.130 ± 0.015 mil M | 87,2 ± 9,0 M | 39,7 M | 30,5% | 1.860 |
| HEURISTICA_LOGGING (politica actual) | 2.126 ± 0.018 mil M | 110,1 ± 0,7 M | 55,2 M | 30,1% | 2.388 |
| THOMPSON_BOOTSTRAP_LGBM | 2.124 ± 0.016 mil M | 90,3 ± 4,7 M | 41,5 M | 30,6% | 1.422 |
| LINUCB | 2.063 ± 0.029 mil M | 91,8 ± 1,0 M | 1,8 M | 33,7% | 137 |
| SIN_CONTACTO | 2.035 ± 0.013 mil M | 88,7 ± 0 M | 0 | 33,6% | 0 |
| TOPK_RIESGO_VISITA | 2.018 ± 0.013 mil M | 197,3 ± 0 M | 144,0 M | 30,4% | 2.432 |

(tabla completa: `salidas/bandit_contacto/resumen_online.csv`; regret = esperanza bajo la
probabilidad latente, no la recompensa sorteada, para que sea comparable entre semillas sin ruido
extra de muestreo -- ver `model.simular_politica_online`)

**Aceptacion del brief** ("Thompson o LinUCB superan a eps-greedy y a top-k-por-riesgo en reward
neto a 24 meses en >= 4 de 5 semillas"): **se cumple con THOMPSON_LINEAL** (gana a eps-greedy en
4/5 semillas y a top-k en 5/5). LINUCB **no** la cumple (gana a top-k en 5/5 pero nunca a
eps-greedy: se vuelve demasiado conservador, ver §8). THOMPSON_BOOTSTRAP_LGBM tampoco (2/5 contra
eps-greedy).

### 6.2 Composicion del contacto por tipo de respuesta latente (diagnostico, seed 42; nunca visto por las politicas)

Poblacion base del pool: ~29,9% PERSUADIBLE, ~44,8% INDIFERENTE, ~25,3% SLEEPING_DOG.

| politica | % contactos a PERSUADIBLE | % a SLEEPING_DOG | contactos totales |
|---|---|---|---|
| TOPK_RIESGO_VISITA | 29,7% | 25,3% | 9.600 |
| HEURISTICA_LOGGING | 29,6% | 25,4% | 9.417 |
| EPSILON_GREEDY | 29,3% | 25,7% | 7.260 |
| THOMPSON_LINEAL | 29,8% | 25,1% | 7.013 |
| THOMPSON_BOOTSTRAP_LGBM | 29,6% | 24,7% | 6.050 |
| LINUCB | 28,1% | 25,4% | 523 |

**Ninguna politica desvia su mezcla de contactos lejos de la proporcion poblacional**: el tipo
latente es independiente del contexto por diseno (§8), asi que ningun algoritmo puede aprender a
"identificar" sleeping dogs via features -- solo puede aprender el valor **promedio** por brazo (lo
que si logran Thompson/eps-greedy) y **cuando** vale la pena contactar segun el valor en juego
(RIM/comision), no **a quien especificamente**. Ver limitaciones (§8).

### 6.3 Evaluacion off-policy: valor verdadero vs IPS/SNIPS/DR

| politica | valor verdadero (CLP/decision) | IPS | SNIPS | DR | err. IPS | err. SNIPS | err. DR |
|---|---|---|---|---|---|---|---|
| EPSILON_GREEDY | 213.279 | 228.014 | 221.360 | 224.448 | 6,9% | 3,8% | 5,2% |
| LINUCB | 223.938 | 231.792 | 237.098 | 244.240 | 3,5% | 5,9% | 9,1% |
| THOMPSON_LINEAL | 223.644 | 220.563 | 228.046 | 229.581 | 1,4% | 2,0% | 2,7% |
| THOMPSON_BOOTSTRAP_LGBM | 223.586 | 208.056 | 223.336 | 227.028 | 6,9% | 0,1% | 1,5% |
| SIN_CONTACTO | 224.189 | 226.801 | 237.828 | 243.042 | 1,2% | 6,1% | 8,4% |
| TOPK_RIESGO_VISITA | 212.768 | 228.353 | 220.894 | 224.121 | 7,3% | 3,8% | 5,3% |
| HEURISTICA_LOGGING (sanity, pi_e=pi_0) | 221.413 | 220.493 | 220.493 | 220.446 | 0,4% | 0,4% | 0,4% |

(`salidas/bandit_contacto/ope.csv`; sanity check: cuando se evalua la propia politica de logging
IPS=SNIPS por construccion y el error es solo ruido de muestreo -- 0,4%, confirma que el pipeline de
importance sampling esta bien implementado)

**Aceptacion del brief** ("DR < 15% del valor verdadero en las candidatas"): **se cumple** (2,7% a
9,1% en las 4 candidatas). ("DR tiene el menor error"): **NO se observa aqui** -- IPS o SNIPS son
mas precisos que DR en 3 de 4 candidatas. Con propensiones bien comportadas (pool acotado, sin pesos
extremos) y un `Q_hat` (LightGBM chico sobre 9.600 filas) que aporta su propio sesgo de
aproximacion, la reduccion de varianza de DR no alcanza a compensar ese sesgo en esta corrida:
resultado honesto, no el esperado por el brief, y se reporta igual.

### 6.4 Interpretabilidad

* **Thompson lineal** (pesos de la media posterior, top 4 |peso| por brazo): en los 5 brazos domina
  `bias` y `rim_media_12`/`rim_l1` (positivo: mas RIM, mas comision en juego, mayor reward), seguido
  de `densidad_12` (negativo: cotizante estable, menos que ganar reteniendolo) y
  `brecha_comision_rel` (negativo en TODOS los brazos, incluido `SIN_CONTACTO`: mayor brecha de
  comision predice menor reward independientemente del brazo, no es una senal que distinga QUE
  brazo conviene, sino que tan probable es el traspaso en general).
* **Thompson-LGBM** (gain del ultimo refit): `rim_l1` domina con ~20x el gain de la siguiente
  variable; `brazo_idx` aporta gain moderado (4,1e12, por debajo de `logins_3m` o `edad`) -- de
  nuevo, el modelo distingue mejor "cuanto hay en juego" que "que brazo conviene".
* **Ejemplos individuales** (`salidas/bandit_contacto/ejemplos_individuales.csv`, 5 filas al azar de
  la politica de logging): incluyen un caso VISITA a un SLEEPING_DOG con reward positivo (no
  traspaso esta vez, ruido a favor) y un caso LLAMADA con traspaso realizado (reward = -2.000,
  exactamente el costo del brazo, cero comision: confirma la formula de recompensa).

## 7. Lectura de negocio

* La politica **actual** (heuristica top-riesgo + visita/llamada/email por tercil) ya genera
  retencion neta positiva frente a no hacer nada (+90 M CLP sobre `SIN_CONTACTO` en 24 meses a esta
  escala), pero **gasta 55 M CLP** en el proceso -- 26% mas que Thompson lineal, que logra **mas**
  reward neto con **menos** gasto (43 M CLP) porque aprende a **evitar el brazo mas caro (VISITA)
  cuando el RIM en juego es bajo**, no porque identifique sleeping dogs (§6.2).
* **LinUCB "se apaga"**: converge a contactar casi a nadie (523 de 9.600 decisiones posibles). Es
  una politica *segura* (evita perder plata en promedio) pero deja sobre la mesa la retencion que
  si captura Thompson lineal -- una AFP que la usara sin mas exploracion forzada estaria
  subinvirtiendo en retencion.
* **Antes de desplegar cualquier candidata**: la variable que mas mueve el reward es el RIM/comision
  en juego, no el tipo de respuesta -- la recomendacion practica es **usar el RIM como filtro de
  "vale la pena" y una politica adaptativa (Thompson) para decidir el brazo**, no reemplazar
  completamente la heuristica actual de un dia para otro.

## 8. Decisiones tomadas, supuestos y limitaciones

**Decisiones de diseno** (para que el simulador fuera manejable en el tiempo de la ronda):

* El **contexto es exogeno**: no se recalcula con las propias decisiones del bandit (la tabla
  `contactos` de 02 sigue igual). Reduce mucho el costo computacional; el precio es que el contexto
  no refleja "cuantas veces YA lo contacto el bandit este año", solo el historial de negocio de 02.
* El **mundo (contexto + respuesta latente) es fijo entre las 5 semillas**; solo varia el azar de
  exploracion/observacion de cada politica. Es la practica estandar para medir estabilidad de un
  ALGORITMO de aprendizaje (no para remuestrear la poblacion), y ahorra generar el universo 5 veces.
* **Presupuesto = tamano del pool**: en vez de un pool "en riesgo" mas grande + un paso de
  racionamiento por presupuesto, el top-400 por riesgo ES el pool. Mas simple de razonar y de
  auditar (propensiones exactas), a costa de no modelar un pool de candidatos mas amplio del que
  solo una fraccion recibe presupuesto.
* **Propension de evaluacion (`eps_eval=0,05`)**: LinUCB y Thompson (lineal y LGBM) eligen brazo de
  forma determinista/aproximadamente determinista dado su estado; para evaluarlos con IPS/SNIPS/DR
  se "suaviza" su eleccion con un piso de exploracion ficticio del 5% (evita dividir por 0 y da una
  distribucion bien definida). Thompson lineal se congela con su **media** posterior (no una
  muestra) para esa distribucion -- aproximacion, no el muestreo real de Thompson.
* **`Q_hat` (direct method de DR)**: un unico LightGBM con el brazo como feature (no 5 modelos), 60
  arboles, entrenado sobre las 9.600 filas logueadas -- barato pero con su propio sesgo, lo que
  probablemente explica que DR no gane a IPS/SNIPS en esta corrida (§6.3).

**Supuestos economicos**: ver §2.3 (todos nombrados y con valor en el codigo, `__init__.py`).

**Limitaciones**:

* El **tipo de respuesta latente es independiente del contexto observable** (asignado al azar por
  afiliado). Es una simplificacion fuerte: ningun algoritmo, por bueno que sea, puede aprender a
  **discriminar** personas via contexto en este mundo -- solo puede aprender el efecto **promedio**
  por brazo y usar el contexto para decidir **cuando** vale la pena (RIM en juego), no **a quien**.
  Si se quiere una prueba mas exigente de bandits contextuales (que si puedan "targetear"
  persuadibles), habria que correlacionar el tipo con alguna variable observable (p. ej., mayor
  `brecha_comision_rel` -> mas probable persuadible) -- propuesta para una proxima ronda.
* **DR no fue el estimador de menor error** en esta corrida, al reves de lo que anticipaba el brief;
  se documenta como hallazgo, no se fuerza el resultado.
* **`LGBM_PARAMS` y el calendario de refit (cada 6 meses) no se afinaron por validacion** (no hay un
  split valid/test para un bandit online): son valores razonables elegidos a mano por velocidad.
* El **alpha de LinUCB (1,0) y la varianza de Thompson (v=1,0) tampoco se afinaron**: un experimento
  rapido variando alpha (0,1 a 2,0) no cambio el resultado cualitativo (LinUCB sigue siendo
  conservador), asi que no parece ser solo un problema de calibracion del hiperparametro sino del
  diseno (varianza del reward dominada por el RIM, ver interpretabilidad §6.4).
* Se usaron **3 semillas de universo comparten un unico `seed_mundo`** (fijo en 42): el universo y
  la respuesta latente NO varian entre las 5 semillas de la simulacion online (ver "decisiones").
* El dataset logueado para OPE proviene de la **primera semilla** (42) unicamente: no se promedia
  sobre semillas (asi funcionaria en produccion: se audita un lote ya guardado, no un promedio de
  corridas paralelas).

## 9. Como correr y tiempos

```bash
cd /home/user/wt/empleador
PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.bandit_contacto.run \
    --n-afiliados 10000 --seeds 42,7,123,2024,99 --salida salidas/bandit_contacto

PYTHONPATH=/home/user/wt/empleador python3 -m pytest afp_lab/bandit_contacto/tests -q
```

Tiempos medidos en esta maquina (4 vCPU compartidas, `n_jobs=2` en LightGBM):

| Etapa | Tiempo |
|---|---|
| Tablas sinteticas (10.000 afiliados, 60 meses) | ~2,4 s |
| Contexto as-of + respuesta latente (161.028 filas) | ~1,9 s |
| Simulacion online (5 semillas x 7 politicas x 24 meses) | ~2,5 s |
| Evaluacion off-policy (IPS/SNIPS/DR + interpretabilidad) | ~0,2 s |
| **Total `run.py`** | **~7-8 s** (limite: 4 min) |
| Tests (`pytest`, 16 casos) | ~2,1 s (limite: 60 s) |

Muy por debajo de los limites de la ronda: **se corrio con las 5 semillas pedidas**, sin necesidad
de bajar a 3.

## 10. Guardrails y como se desplegaria

1. **OPE antes de arriesgar presupuesto real**: nunca promover una politica candidata a produccion
   sin antes estimar su valor con IPS/SNIPS/DR sobre el logueado de la politica vigente (como aqui)
   Y sin verificar que el error relativo de los tres estimadores sea chico (< 15%, idealmente <
   10%) -- si no, no hay senal confiable y toca seguir logueando antes de decidir.
2. **Exploracion acotada** (`epsilon<=0,10`, nunca 100% greedy) durante todo el despliegue: el
   negocio necesita seguir generando datos con propension positiva en los 5 brazos para poder
   seguir auditando con OPE en el futuro (propension 0 = no se puede evaluar ese brazo nunca mas).
3. **Presupuesto y frecuencia como limites duros, no sugerencias**: `PRESUPUESTO_MENSUAL_CONTACTOS`
   y `MESES_ENTRE_CONTACTOS` se aplican ANTES de que la politica elija brazo (el pool ya viene
   acotado), para que ningun algoritmo pueda "gastar de mas" por diseno.
4. **Monitoreo continuo post-despliegue**: tasa de traspaso realizada vs la esperada por el modelo
   (alerta si diverge > 2 sigma), gasto mensual real vs presupuesto, y --el guardrail especifico de
   este problema-- **tasa de contactos a quienes despues traspasan de todas formas** (proxy
   observable de "sleeping dogs" ya que el tipo latente nunca se observa): si sube, congelar el
   brazo mas agresivo (VISITA) para ese segmento y re-auditar con OPE.
5. **Nunca lanzar LinUCB "tal cual"** con los hiperparametros de esta corrida: se volvio
   excesivamente conservador (§8); si se elige un algoritmo de banda de confianza, forzar un piso de
   exploracion minimo (p. ej. `eps` inyectado ademas de la UCB) para evitar que se apague.
6. **Reentrenar/revisar mensualmente** (como el refit de Thompson-LGBM), no solo al desplegar: el
   mercado (comisiones, licitacion) y la composicion de la cartera cambian.

## Propuestas para `common/` y proximos pasos

* Proponer a `common/` un helper generico de **evaluacion off-policy** (IPS/SNIPS/DR + suavizado de
  propensiones) parametrizado por `(logueado, propension_objetivo, recompensa)`: hoy vive en
  `evaluate.py` de este proyecto pero el patron es identico al que necesitaria cualquier bandit del
  laboratorio (p. ej. ofertas de fondos, campanas de digitalizacion).
* Proximos pasos: (a) correlacionar el tipo de respuesta latente con variables observables para una
  prueba mas exigente de targeting contextual (§8); (b) probar LinUCB/Thompson con un piso de
  exploracion explicito ademas de su mecanismo propio; (c) sumar el segmento de 06
  (`segmentacion_afiliados`) al contexto si aporta separacion mas alla de lo que ya capturan las
  features de 02 (no se hizo en esta ronda por tiempo, brief lo marcaba como opcional -- "si
  aporta").
