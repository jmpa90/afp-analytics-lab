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
* **`generar_tipo_respuesta(contexto, seed)`** (funcion LATENTE, columna `lat_tipo_respuesta`; NO es
  una tabla pre-calculada dentro de `generar_tablas_bandit` porque necesita el contexto as-of, que
  todavia no existe en ese punto -- se calcula en `model.preparar_entorno`): tipo de respuesta al
  contacto -- PERSUADIBLE, INDIFERENTE o SLEEPING_DOG -- **por FILA (afiliado_id, periodo_corte)**,
  como una funcion logit (softmax de 3 clases, INDIFERENTE de referencia) de variables as-of
  **que varian por afiliado** mas dos componentes de ruido (uno persistente por afiliado --
  "personalidad", fijo en el tiempo -- y uno transitorio por fila), calibrada para que la mezcla
  global quede cerca de `P_TIPO_RESPUESTA = (0,30 PERSUADIBLE; 0,45 INDIFERENTE; 0,25 SLEEPING_DOG)`.
  **Correccion de revision** (ver §8): la primera version de esta ronda sorteaba el tipo
  independiente del contexto (solo ruido) -- el gerente la observo como incoherente con el brief
  ("heterogénea por contexto") porque ningun bandit contextual puede aprender algo que no depende
  del contexto por construccion. `TIPO_RESPUESTA_COEF` (`__init__.py`):

  | tipo | sube con | baja con |
  |---|---|---|
  | PERSUADIBLE | uso digital (`logins_3m`), cambios de fondo recientes (`cambios_fondo_3m`), saldo (`log_saldo`) | edad |
  | SLEEPING_DOG | reclamos recientes (`reclamos_6m`), edad | rentabilidad del FONDO PROPIO vs el sistema (`brecha_rentabilidad_pp`) |

  Se descarto `brecha_comision_rel` como driver del tipo (aunque si es parte del riesgo base, ver
  abajo): es una comision de MERCADO, identica para todos los clientes de la AFP en un mes dado, asi
  que no aporta heterogeneidad ENTRE personas -- se reemplazo por `brecha_rentabilidad_pp`, que si
  varia por afiliado (depende de en que FONDO A-E esta cada uno) y sirve de proxy de "ya
  insatisfecho con el desempeno de mi propio fondo".
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

### 6.0 Ablacion: tipo aleatorio (version original) vs tipo dependiente del contexto (version corregida)

La primera entrega de esta ronda sorteaba `lat_tipo_respuesta` independiente del contexto (solo
ruido). El gerente la devolvio: eso contradice "heterogenea por contexto" del brief y hace que la
comparacion contextual-vs-no-contextual no tenga sentido (nada es aprendible). Se corrigio
`generar_tipo_respuesta` (§2.2) y se re-corrio la misma referencia (10.000 afiliados, 5 semillas). La
version aleatoria queda documentada como **ablacion sin senal contextual**:

| politica | reward neto -- ablacion (tipo aleatorio) | reward neto -- version corregida (tipo por contexto) | contactos a sleeping dogs -- ablacion | contactos a sleeping dogs -- corregida |
|---|---|---|---|---|
| THOMPSON_LINEAL | 2.153 mil M | 2.138 mil M | 1.860 | 2.230 |
| EPSILON_GREEDY | 2.130 mil M | 2.130 mil M | 1.860 | 2.266 |
| HEURISTICA_LOGGING | 2.126 mil M | 2.116 mil M | 2.388 | 2.927 |
| THOMPSON_BOOTSTRAP_LGBM | 2.124 mil M | 2.109 mil M | 1.422 | 1.680 |
| LINUCB | 2.063 mil M | 2.063 mil M | 137 | 230 |
| SIN_CONTACTO | 2.035 mil M | 2.035 mil M | 0 | 0 |
| TOPK_RIESGO_VISITA | 2.018 mil M | 2.012 mil M | 2.432 | 2.992 |

Notar que **suben los contactos a sleeping dogs en la version corregida para TODAS las politicas,
incluidas las que no discriminan** (heuristica, top-k): es real, no un error -- `reclamos_6m` ahora
empuja a la vez el `score_riesgo` (que arma el pool "en riesgo") Y la probabilidad de SLEEPING_DOG,
asi que el pool de mayor riesgo queda con MAS sleeping dogs que la poblacion general (31% del pool
vs 24,8% poblacional, seed 42) -- una AFP que persigue "riesgo" sin mas termina, con este generador,
sobre-representando justamente al segmento en el que el contacto puede ser contraproducente. Esa es
la heterogeneidad que un bandit contextual **si puede**, en principio, aprender a explotar (a
diferencia de la ablacion). Sigue abajo el detalle de la version corregida (la vigente).

### 6.1 Recompensa neta acumulada y regret a 24 meses (media ± IC95% entre 5 semillas)

| politica | reward neto (CLP) | regret vs oraculo | costo total | tasa traspaso | contactos a sleeping dogs |
|---|---|---|---|---|---|
| **THOMPSON_LINEAL** | **2.138 ± 0.029 mil M** | 95,2 ± 11,0 M | 42,3 M | 30,3% | 2.230 |
| EPSILON_GREEDY | 2.130 ± 0.026 mil M | 91,6 ± 15,4 M | 35,1 M | 30,8% | 2.266 |
| HEURISTICA_LOGGING (politica actual) | 2.116 ± 0.022 mil M | 116,4 ± 0,3 M | 55,2 M | 30,6% | 2.927 |
| THOMPSON_BOOTSTRAP_LGBM | 2.109 ± 0.030 mil M | 94,5 ± 4,9 M | 40,1 M | 31,0% | 1.680 |
| LINUCB | 2.063 ± 0.029 mil M | 92,7 ± 1,4 M | 1,8 M | 33,7% | 230 |
| SIN_CONTACTO | 2.035 ± 0.013 mil M | 89,2 ± 0 M | 0 | 33,6% | 0 |
| TOPK_RIESGO_VISITA | 2.012 ± 0.013 mil M | 212,9 ± 0 M | 144,0 M | 31,2% | 2.992 |

(tabla completa: `salidas/bandit_contacto/resumen_online.csv`; regret = esperanza bajo la
probabilidad latente, no la recompensa sorteada, para que sea comparable entre semillas sin ruido
extra de muestreo -- ver `model.simular_politica_online`)

**Aceptacion del brief** ("Thompson o LinUCB superan a eps-greedy y a top-k-por-riesgo en reward
neto a 24 meses en >= 4 de 5 semillas"): **NO se cumple con la version corregida** (a diferencia de
la ablacion, donde si se cumplia). THOMPSON_LINEAL gana a top-k en 5/5 semillas pero solo a
eps-greedy en **3/5**; LINUCB gana a top-k en 5/5 pero nunca a eps-greedy (0/5, sigue demasiado
conservador, ver §8); THOMPSON_BOOTSTRAP_LGBM gana a eps-greedy en 1/5. Con senal contextual real
pero debil (§6.2), 24 meses y recompensa rezagada 6 meses no alcanzan para que el aprendizaje
contextual saque una ventaja consistente sobre el mejor brazo global (que aprende eps-greedy) en la
mayoria de las semillas -- resultado honesto, no el buscado, y se reporta igual.

### 6.2 Composicion del contacto por tipo de respuesta latente (diagnostico, seed 42; nunca visto por las politicas)

Poblacion base (todas las filas de contexto): ~29,9% PERSUADIBLE, ~45,2% INDIFERENTE, ~24,8%
SLEEPING_DOG. Poblacion del POOL "en riesgo" (top-score, antes de elegir brazo): ~27,9%/41,0%/31,2%
-- el pool ya sobre-representa sleeping dogs frente a la poblacion general (§6.0).

| politica | % contactos a PERSUADIBLE | % a SLEEPING_DOG | contactos totales |
|---|---|---|---|
| TOPK_RIESGO_VISITA | 27,9% | 31,2% | 9.600 |
| HEURISTICA_LOGGING | 27,6% | 31,0% | 9.417 |
| EPSILON_GREEDY | 27,4% | 31,6% | 7.260 |
| THOMPSON_LINEAL | 27,3% | 31,3% | 7.219 |
| **THOMPSON_BOOTSTRAP_LGBM** | **28,8%** | **30,1%** | 5.921 |
| LINUCB | 24,1% | 42,8% | 523 |

**Thompson-LGBM es la unica que discrimina en la direccion correcta**: mas persuadibles (28,8% vs
27,6-27,9% de heuristica/top-k) y menos sleeping dogs (30,1% vs 31,0-31,2%) que las politicas que no
usan contexto para elegir brazo -- y, en contactos TOTALES, contacta 1.680 sleeping dogs contra
2.927-2.992 de heuristica/top-k (43% menos), aunque una parte de esa caida es simplemente que
contacta menos en total (5.921 vs 9.417-9.600). **LinUCB va en la direccion CONTRARIA** (42,8% de
sus pocos contactos son sleeping dogs, mas que el pool completo): con solo 523 contactos en 24
meses la composicion es ruidosa, no evidencia de que "identifique" bien a quien evitar. Eps-greedy y
Thompson lineal (no usan las variables one-by-one para discriminar tipo, ver §8) quedan muy cerca de
la composicion del pool.

### 6.3 Evaluacion off-policy: valor verdadero vs IPS/SNIPS/DR

| politica | valor verdadero (CLP/decision) | IPS | SNIPS | DR | err. IPS | err. SNIPS | err. DR |
|---|---|---|---|---|---|---|---|
| EPSILON_GREEDY | 224.189 | 204.950 | 227.251 | 224.447 | 8,6% | 1,4% | **0,1%** |
| LINUCB | 223.879 | 231.846 | 237.153 | 244.763 | 3,6% | 5,9% | 9,3% |
| THOMPSON_LINEAL | 223.988 | 234.905 | 251.808 | 234.414 | 4,9% | 12,4% | 4,7% |
| THOMPSON_BOOTSTRAP_LGBM | 224.988 | 230.333 | 232.157 | 237.447 | **2,4%** | 3,2% | 5,5% |
| SIN_CONTACTO | 224.158 | 226.776 | 237.802 | 243.512 | 1,2% | 6,1% | 8,6% |
| TOPK_RIESGO_VISITA | 211.306 | 225.458 | 218.093 | 221.583 | 6,7% | 3,2% | 4,9% |
| HEURISTICA_LOGGING (sanity, pi_e=pi_0) | 220.799 | 218.726 | 218.726 | 218.667 | 0,9% | 0,9% | 1,0% |

(`salidas/bandit_contacto/ope.csv`; sanity check: cuando se evalua la propia politica de logging
IPS=SNIPS por construccion y el error es solo ruido de muestreo -- 0,9%, confirma que el pipeline de
importance sampling esta bien implementado)

**Aceptacion del brief** ("DR < 15% del valor verdadero en las candidatas"): **se cumple** (4,7% a
9,3% en las 4 candidatas). ("DR tiene el menor error"): **parcial** -- es el mejor para
EPSILON_GREEDY (0,1%, el mas preciso de los 21 pares politica x estimador) pero NO para LinUCB (el
peor de sus tres) ni para Thompson-LGBM (el peor de sus tres); para Thompson lineal es mejor que
SNIPS pero peor que IPS. Con propensiones bien comportadas (pool acotado, sin pesos extremos) y un
`Q_hat` (LightGBM chico sobre 9.600 filas) que aporta su propio sesgo de aproximacion, la reduccion
de varianza de DR no alcanza a compensar ese sesgo de forma consistente en esta corrida: resultado
honesto, no el esperado por el brief, y se reporta igual.

### 6.4 Interpretabilidad

* **Thompson lineal** (pesos de la media posterior, top 4 |peso| por brazo): en los 5 brazos domina
  `bias` y `rim_media_12`/`rim_l1` (positivo: mas RIM, mas comision en juego, mayor reward), seguido
  de `densidad_12` (negativo: cotizante estable, menos que ganar reteniendolo). Ninguna de las 4
  variables mas pesadas por brazo es una que distinga TIPO (serian `logins_3m`, `reclamos_6m`,
  `cambios_fondo_3m`, `brecha_rentabilidad_pp`): el modelo lineal esta usando el contexto sobre todo
  para estimar "cuanto hay en juego", no "que tipo de persona es", lo que es coherente con que gane
  a eps-greedy solo en 3/5 semillas (§6.1).
* **Thompson-LGBM** (gain del ultimo refit, ya en la version corregida): `rim_l1` sigue dominando
  (2,7e14) pero ahora **`brecha_rentabilidad_pp` (1,8e13) y `logins_3m` (1,5e13) entran entre las 8
  variables de mayor gain** -- dos de los tres drivers nuevos del tipo latente -- consistente con que
  sea la UNICA politica que discrimina en la direccion correcta en §6.2.
* **Ejemplos individuales** (`salidas/bandit_contacto/ejemplos_individuales.csv`, 5 filas al azar de
  la politica de logging): incluyen un caso APV a un SLEEPING_DOG con reward positivo (no traspaso
  esta vez, ruido a favor -- exactamente el tipo de contacto que, en promedio, empeora las cosas) y
  un caso LLAMADA con traspaso realizado (reward = -2.000, exactamente el costo del brazo, cero
  comision: confirma la formula de recompensa).

## 7. Lectura de negocio

* La politica **actual** (heuristica top-riesgo + visita/llamada/email por tercil) sigue generando
  retencion neta positiva frente a no hacer nada (+80 M CLP sobre `SIN_CONTACTO` en 24 meses a esta
  escala), pero **gasta 55 M CLP** y es la que MAS sleeping dogs contacta en absoluto (2.927): al no
  usar contexto para elegir brazo, hereda tal cual la composicion del pool "en riesgo", que ya viene
  sesgada hacia el segmento donde el contacto puede ser contraproducente (§6.0/§6.2).
* **Thompson-LGBM es, de las 4 candidatas, la unica que aprendio a leer la heterogeneidad**: contacta
  proporcionalmente mas persuadibles y menos sleeping dogs que la heuristica o top-k (§6.2) -- pero
  esa mejora en composicion **no le alcanza** para superar a eps-greedy en reward neto (solo en 1/5
  semillas): con 24 meses, presupuesto acotado y recompensa rezagada 6 meses, la senal (débil por
  diseno, con harto ruido) no da para que la ganancia de precision compense un aprendizaje mas lento
  que el de un promedio simple por brazo.
* **LinUCB sigue "apagandose"**: contacta a casi nadie (523 de 9.600 decisiones posibles) y, cuando
  contacta, lo hace peor que al azar en composicion (42,8% sleeping dogs). Una AFP que la desplegara
  sin forzar mas exploracion estaria subinvirtiendo en retencion Y sin garantia de estar evitando a
  quien corresponde.
* **Recomendacion practica**: no reemplazar la heuristica actual por un solo algoritmo todavia. El
  candidato mas prometedor para una prueba piloto acotada es **Thompson-LGBM** (unico que muestra
  discriminacion real, aunque su reward neto no le gane aun a la heuristica ni a eps-greedy) --
  correrlo en paralelo (sin desplazar la politica vigente) por mas de 24 meses, con el pool completo
  de afiliados en riesgo (no solo 400/mes) para acumular mas señal, y volver a auditar con OPE antes
  de subir su participacion del presupuesto.

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

* **La senal contextual del tipo de respuesta es real pero debil** (§6.0/§6.2/§6.4): los pesos de
  `TIPO_RESPUESTA_COEF` se eligieron a mano, sin calibrar cuanta heterogeneidad "deberia" ser
  aprendible en 24 meses -- solo se verifico que la asociacion es estadisticamente significativa
  (test `test_tipo_respuesta_correlaciona_con_el_contexto`) y que las proporciones globales quedan
  parecidas a las declaradas. Que solo Thompson-LGBM (no lineal, no LinUCB) la aproveche sugiere que
  la relacion contexto -> tipo no es lineal en las variables usadas, o que el ruido (persistente +
  transitorio) es demasiado grande relativo a las 6-9 observaciones por afiliado disponibles en la
  ventana simulada.
* **`brecha_comision_rel` se descarto como driver del TIPO** (§2.2) por ser una variable de mercado
  identica para todos los clientes de la AFP en un mes dado (no aporta heterogeneidad entre
  personas); se sigue usando para el riesgo BASE (`calcular_respuesta_latente`), donde si tiene
  sentido (afecta a todos por igual en un momento dado, pero legitimamente mueve la propension
  agregada de traspaso en el tiempo).
* **DR no fue uniformemente el estimador de menor error** en esta corrida (solo lo es para
  EPSILON_GREEDY), al reves de lo que anticipaba el brief; se documenta como hallazgo, no se fuerza
  el resultado.
* **`LGBM_PARAMS` y el calendario de refit (cada 6 meses) no se afinaron por validacion** (no hay un
  split valid/test para un bandit online): son valores razonables elegidos a mano por velocidad.
* El **alpha de LinUCB (1,0) y la varianza de Thompson (v=1,0) tampoco se afinaron**: un experimento
  rapido variando alpha (0,1 a 2,0) no cambio el resultado cualitativo (LinUCB sigue siendo
  conservador), asi que no parece ser solo un problema de calibracion del hiperparametro sino del
  diseno (varianza del reward dominada por el RIM, ver interpretabilidad §6.4).
* **El universo (contexto + respuesta latente) comparte un unico `seed_mundo`** (fijo en 42): NO
  varia entre las 5 semillas de la simulacion online, que solo aportan azar de exploracion/observacion
  (ver "decisiones de diseno").
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
| Tablas sinteticas (10.000 afiliados, 60 meses) | ~2,5 s |
| Contexto as-of + respuesta latente (161.028 filas) | ~2,1 s |
| Simulacion online (5 semillas x 7 politicas x 24 meses) | ~3,1 s |
| Evaluacion off-policy (IPS/SNIPS/DR + interpretabilidad) | ~0,3 s |
| **Total `run.py`** | **~8-9 s** (limite: 4 min) |
| Tests (`pytest`, 18 casos) | ~2,5 s (limite: 60 s) |

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
* Proximos pasos: (a) que solo Thompson-LGBM (no lineal) explote la heterogeneidad del tipo (§6.2,
  §6.4) sugiere revisar si la relacion contexto->tipo necesita interacciones/no-linealidades mas
  fuertes, o si simplemente hacen falta mas de 24 meses / mas presupuesto mensual para que el
  aprendizaje contextual lineal alcance a distinguirse del promedio global (eps-greedy); (b) probar
  LinUCB/Thompson con un piso de exploracion explicito ademas de su mecanismo propio, para que dejen
  de "apagarse"; (c) sumar el segmento de 06 (`segmentacion_afiliados`) al contexto si aporta
  separacion mas alla de lo que ya capturan las features de 02 (no se hizo en esta ronda por tiempo,
  brief lo marcaba como opcional -- "si aporta").
