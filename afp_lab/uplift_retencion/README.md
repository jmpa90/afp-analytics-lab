# 07 — `uplift_retencion`: a quien conviene contactar

Ronda 4, objetivo 07 (`docs/briefs/ronda_4/uplift.md`). Persona: AFP Cuprum, datos 100 % sinteticos.

## 1. Pregunta de negocio y decision que apoya

El proyecto 02 (`churn_traspaso`) predice **quien se va**. Para la licitacion de stock 2027 (Ley
21.735) y para el dia a dia del equipo comercial, la pregunta relevante es otra: **a quien conviene
CONTACTAR**, es decir en quien el contacto de un ejecutivo cambia de verdad la decision. Hay cuatro
tipos de afiliado frente a un contacto: se queda igual (contacto desperdiciado), se va igual
(contacto desperdiciado), **persuadible** (el contacto lo retiene) y **sleeping dog** (el contacto lo
despierta y lo hace irse). Rankear por riesgo (02) mezcla los cuatro tipos; este proyecto separa el
efecto causal del contacto (uplift) del riesgo de traspaso, con T-, X- y DR-learner sobre LightGBM.

Decision que apoya: **a quien asigna el ejecutivo propio su cupo de contactos** (capacidad limitada),
dado un presupuesto de contacto del 10 % o 20 % de la cartera.

## 2. Datos sinteticos

Reutiliza **sin modificar** `afp_lab.churn_traspaso.synthetic.generar_tablas(..., perfil="cuprum")`
(mismo universo, mismo mercado, misma tabla `traspasos`) y agrega, por encima, una capa propia de 07
(`synthetic.py`): la campana historica de contacto y el efecto causal verdadero.

### 2.1 Campana historica (confounding real) + holdout aleatorio

Cada fila (afiliado x `periodo_corte`) recibe un tratamiento `T` (contacto de un ejecutivo, ver
`asignar_tratamiento`):

* **Holdout experimental (10 % de la cartera, fijo por afiliado en el tiempo)**: asignacion
  **aleatoria** (`p_tratamiento_holdout = 0,50`), sin relacion con `x`. Sirve para medir Qini/AUUC sin
  sesgo (nadie en el mundo real tiene esto sin correrlo).
* **90 % restante: politica historica confundida**. La propension `e_hist(x)` depende del percentil
  de saldo dentro del corte, de si el afiliado es mayor (>= 50 anios, cartera premium) y de un
  **score heuristico de riesgo** propio (`riesgo_latente_heuristico`, no es el hazard real de 02):
  combina reclamo en 6 m, visita de la competencia en 3 m, brecha de rentabilidad mala, comision alta
  y poca antiguedad. Es el confounding real que un uplift model tiene que corregir: el ejecutivo
  contacta mas a quien tiene MAS riesgo y MAS saldo, exactamente lo contrario de "aleatorio".

### 2.2 Efecto causal heterogeneo `lat_cate` (tabla latente, nunca feature)

Tres segmentos, definidos con features as-of (mas una sensibilidad a comision **latente** por
afiliado, `lat_sensibilidad_comision ~ Uniform(0,1)`, analoga a `lat_switcher` de 02):

| Segmento | Condicion (as-of) | Efecto del contacto |
|---|---|---|
| **Persuadible** | reclamo en 6 m **resuelto** (no abierto, no desfavorable) + saldo medio (tramo `5-15M`) + uso digital medio (`logins_3m` en `[2, 10)`) | Reduce `P(traspaso)`: retiene con prob. `p_retencion_persuadible = 0,40` si el afiliado se iba a ir |
| **Sleeping dog** | uso digital bajo (`logins_3m <= 1`) + sensibilidad a comision alta (top 30 % de `lat_sensibilidad_comision`) | Aumenta `P(traspaso)`: lo traspasa con prob. `p_despierta_sleeping_dog = 0,12` si se iba a quedar |
| **Neutral** (mayoria) | resto | Efecto cero |

La magnitud escala con el score de riesgo (`lat_cate = -0,40 x riesgo` en persuadibles,
`+0,12 x (1-riesgo)` en sleeping dogs): a mas riesgo, mas hay que ganar reteniendo a un persuadible; a
menos riesgo, mas sorpresa si se despierta un sleeping dog.

### 2.3 Target `y`: identico a 02, alterado solo en las filas tratadas

`y0` (target de CONTROL) sale de `churn_traspaso.construir_target`, **byte a byte igual** al de 02
(mismo horizonte, misma tabla `traspasos`). El efecto se aplica con randomizacion propia
(`aplicar_efecto_causal`), **nunca tocando `traspasos` ni ninguna tabla de 02**:

* `T = 0` (control): `y = y0` siempre. Coherencia garantizada y verificada por test
  (`test_control_y_es_identico_a_y0_de_02`).
* `T = 1`, persuadible, `y0 = 1`: se retiene (`y = 0`) con prob. `p_retencion_persuadible`.
* `T = 1`, sleeping dog, `y0 = 0`: se traspasa (`y = 1`) con prob. `p_despierta_sleeping_dog`.
* Resto de las filas tratadas: `y = y0` (efecto cero).

### 2.4 Parametros (`ParametrosCampana`, con valores por defecto)

| Parametro | Valor | Que representa |
|---|---|---|
| `p_holdout` | 0,10 | Fraccion de la cartera en el holdout aleatorio |
| `p_tratamiento_holdout` | 0,50 | Prob. de tratamiento DENTRO del holdout |
| `b0_propension`, `b_saldo_percentil`, `b_tramo_edad_mayor`, `b_score_riesgo` | -1,6 / 1,3 / 0,35 / 2,0 | Logit de la propension historica confundida |
| `edad_umbral_mayor` | 50 anios | Umbral de "afiliado mayor" en la propension |
| `intercepto_riesgo`, `w_reclamo_6m`, `w_contacto_competencia_3m`, `w_brecha_rentabilidad_mala`, `w_brecha_comision_alta`, `w_antiguedad_baja` | -1,1 / 1,1 / 1,3 / 0,9 / 0,5 / 0,6 | Score heuristico de riesgo (logit) |
| `p_retencion_persuadible` | 0,40 | Prob. de retener a un persuadible tratado |
| `p_despierta_sleeping_dog` | 0,12 | Prob. de despertar a un sleeping dog tratado |
| `frac_sensibilidad_comision_alta` | 0,30 | Top de `lat_sensibilidad_comision` que califica como sleeping dog |
| `logins_3m_bajo` / `logins_3m_medio_lo` / `logins_3m_medio_hi` | 1,0 / 2,0 / 10,0 | Umbrales de uso digital "bajo" / "medio" |
| `tramo_saldo_medio` | `5-15M` | Tramo de saldo considerado "medio" (mismo bin que `churn_traspaso.evaluate`) |

Son **supuestos de trabajo** (no hay una campana real de contacto+causalidad publicada para calibrar
magnitudes de "sleeping dog" en AFP chilenas); las MAGNITUDES son razonables (10-40 % de efecto sobre
una tasa base de 2-4 %) pero arbitrarias, y se declaran como tales.

## 3. Definicion del target, poblacion, horizonte, exclusiones

**Identica a 02**: `y = 1` si el afiliado solicita un traspaso de SALIDA en `corte+1 .. corte+6`
(`HORIZONTE_MESES = 6`, importado de `churn_traspaso`, nunca redefinido). Poblacion: vigente en la AFP
propia al corte, sin solicitud de salida ya presentada, no retirado (misma poblacion de 02, mismos
cortes evaluables via `cortes_evaluables`, que ya excluye los cortes contaminados por la licitacion de
stock de la Ley 21.735). Split temporal con purga de 6 meses entre train/valid/test, reutilizando
**sin modificar** `churn_traspaso.features.split_temporal`.

## 4. Features (por familia, regla as-of)

| Familia | Ejemplos | Regla as-of |
|---|---|---|
| Panel laboral, cuenta, relacion con la AFP, mercado | `rim_media_12`, `saldo_obligatorio`, `reclamos_6m`, `contactos_vendedor_competencia_3m`, `brecha_comision_pp`, ... | Identicas a 02 (`construir_features_multi`), sin cambios: panel laboral en `corte-2`, resto en `corte` |
| Historial de campana propia (07) | `campana_contactos_6m`, `campana_contactos_12m`, `campana_meses_desde_ultimo` | Ventana movil de contactos **estrictamente anteriores** al corte (se excluye el `T` del propio corte con un `shift` antes de la ventana; verificado a mano en los tests) |
| Tratamiento y diseno (NO son features) | `T`, `holdout` | Se excluyen de `columnas_features` (variables de decision/diseno, no covariables) |

`columnas_features` (`features.py`) reutiliza `CT_F.columnas_features` (excluye ids, target y
prefijos `lat_`/`p_`) y ademas excluye `T`, `holdout` y los tramos de negocio (`tramo_saldo`, etc., que
solo existen en la tabla interna usada para construir el efecto causal, nunca llegan al dataset final).

## 5. Modelo y baselines

Todo implementado a mano con LightGBM + pandas/numpy (`n_jobs=2`/`num_threads=2`, ronda 4: 4 vCPU
compartidas con otros dos workstreams); econml/causalml no estan instalados.

* **Propension `e(x)`**: LightGBM binario, entrenado **solo con la poblacion confundida** (fuera del
  holdout) de train/valid, split interno 80/20 aleatorio para early stopping.
* **T-learner**: `mu0(x)`, `mu1(x)` (LightGBM binario) sobre cada brazo; `CATE = mu1(x) - mu0(x)`.
* **X-learner**: efecto imputado (`D1` en tratados via `mu0`, `D0` en control via `mu1`), regresion de
  cada uno (`tau1`, `tau0`), combinados con la propension: `tau_x = e(x)*tau0(x) + (1-e(x))*tau1(x)`.
* **DR-learner (doblemente robusto)**: pseudo-outcome `mu1-mu0 + T(y-mu1)/e - (1-T)(y-mu0)/(1-e)`,
  regresion de ese pseudo-outcome sobre `x`.
* **Baseline obligatorio (riesgo como uplift)**: LightGBM de riesgo puro (ignora `T`, igual que el
  modelo de 02) usado COMO SI fuera un score de uplift — la practica ingenua que este proyecto busca
  superar.
* **Baseline trivial**: ranking aleatorio.

Meta-learners y propension entrenados **solo con la poblacion confundida**; el holdout aleatorio se
reserva integro para evaluar Qini/AUUC sin sesgo. El baseline de riesgo se entrena con toda la
poblacion (ignora `T` por diseno).

**Convencion de signo** (importante, documentada en `evaluate.py`): `y` es traspaso (malo), asi que el
CATE de un persuadible es negativo. Para *rankear* a quien contactar se usa `-tau_hat` en los tres
meta-learners (mayor prioridad = CATE mas negativo); el riesgo se usa tal cual (mayor riesgo = contactar
primero). Qini/AUUC se calculan sobre la escala de "traspasos evitados" (positivo = bueno), que es
exactamente el negativo de la formula clasica aplicada a `y`.

## 6. Resultados (10 000 afiliados, semilla 42, `--perfil cuprum`, ventana 2021-01/2028-12)

Split: train 31 cortes (202311-202605, 184 005 filas), valid 3 cortes (202612-202702, 18 212 filas),
test 3 cortes (202709-202711, 18 247 filas). `frac_tratados` ~ 51-52 %, `frac_holdout` ~ 10,0 % en las
tres particiones.

### 6.1 Overlap de la propension estimada

| Metrica | Valor | Criterio de aceptacion |
|---|---|---|
| `overlap` (fraccion con `e(x)` en `[0,05; 0,95]`, test) | **1,00** | >= 0,80 -> **cumple** |

### 6.2 Correlacion (Spearman) y MAE del CATE estimado vs `lat_cate` (verdad del simulador, test completo)

| Metodo | Spearman | p-valor | MAE |
|---|---|---|---|
| DR-learner | **0,195** | < 1e-150 | **0,0078** |
| X-learner | 0,186 | < 1e-140 | 0,0080 |
| T-learner | 0,152 | < 1e-90 | 0,0133 |
| Riesgo (baseline) | 0,054 | 2e-13 | 0,0264 |
| Aleatorio | 0,012 (ns) | 0,10 | 0,50\* |

\* El MAE del aleatorio no es comparable: su score (`Uniform(0,1)`) no esta en la escala de un CATE
(diferencia de probabilidad en `[-1,1]`), asi que el MAE queda inflado por el desajuste de escala, no
por falta de habilidad causal. Lo mismo aplica parcialmente al riesgo (probabilidad de traspaso, no un
efecto): su Spearman SI es interpretable (ordena, aunque mal) y su MAE es orientativo, no estricto.

**Los tres meta-learners capturan senal causal real y estadisticamente significativa** (p << 0,001):
el DR-learner tiene la mejor correlacion y el menor error contra la verdad del simulador, seguido de
cerca por el X-learner. El baseline de riesgo tiene una correlacion positiva pero mucho mas debil
(0,054): el riesgo SI esta parcialmente correlacionado con ser persuadible (comparten drivers como el
reclamo reciente), pero no distingue sleeping dogs de persuadibles, que es justamente el punto.

### 6.3 Qini / AUUC en el holdout aleatorio de test (con IC bootstrap, 150 remuestreos)

| Metodo | Qini (agregado, 3 cortes) | IC 95 % | Qini > 0 en los 3 cortes de test |
|---|---|---|---|
| **X-learner** | **+0,00086** | [-0,0021; 0,0041] | **Si** (202709: +0,0013; 202710: +0,0014; 202711: +0,00015) |
| T-learner | -0,00044 | [-0,0031; 0,0023] | No (202711 negativo) |
| DR-learner | -0,00133 | [-0,0038; 0,0014] | No (202709 negativo) |
| Riesgo | -0,00127 | [-0,0037; 0,0010] | No (negativo en los 3 cortes) |
| Aleatorio | -0,00096 | [-0,0030; 0,0011] | No (2 de 3 negativos, como se espera del azar) |

**El X-learner es el UNICO metodo que cumple, LITERALMENTE, el criterio de aceptacion** ("Qini > 0 en
todos los cortes de test"), pero con un margen minusculo (el peor corte, 202711, da +0,00015) y un
intervalo bootstrap del agregado que **cruza cero**: no se puede afirmar con confianza estadistica que
supera al azar en este holdout. El baseline de riesgo, en cambio, es NEGATIVO en los tres cortes: usar
el ranking de 02 como si fuera uplift efectivamente ordena peor que el azar para este objetivo (mezcla
persuadibles y sleeping dogs sin distinguirlos). Ver LIMITACIONES: el holdout de test tiene ~600 filas
por corte y el efecto ocurre en ~2,7 % de la poblacion, asi que Qini esta subpotenciado a esta escala.

### 6.4 Metrica de negocio: traspasos evitados (verdad del simulador, `lat_cate`, test completo)

| Ranking | 10 % (1 825 contactados) | 20 % (3 649 contactados) | Persuadibles / sleeping dogs contactados (10 % / 20 %) |
|---|---|---|---|
| **X-learner** | **+14,6** | **+20,8** | 53 / 16 -- 75 / 21 |
| DR-learner | +10,8 | +17,1 | 40 / 17 -- 63 / 18 |
| T-learner | +5,3 | +13,9 | 23 / 23 -- 53 / 28 |
| Riesgo (baseline) | +3,2 | +7,6 | 22 / 61 -- 47 / 110 |
| Aleatorio | +0,2 | +0,4 | 10 / 38 -- 20 / 77 |

**Razon X-learner / riesgo: 4,54x al 10 % y 2,75x al 20 %** (criterio de aceptacion del brief: >= 1,3x
-> **cumple con holgura en ambos presupuestos**). Ademas, el X-learner contacta MENOS sleeping dogs que
su tasa base en la poblacion (0,9 % al 10 %, 0,6 % al 20 %, vs 2,0 % en la poblacion), mientras que el
ranking de riesgo los SOBRE-representa (3,3 % al 10 %, 3,0 % al 20 %): el riesgo alto correlaciona
parcialmente con "sleeping dog", asi que contactar por riesgo activamente empeora el problema que se
quiere evitar.

### 6.5 Segmento -> accion (verdad del simulador + CATE estimado por el X-learner)

| Segmento | n (test) | % cartera | CATE verdadero medio | CATE estimado medio (X-learner) | Accion recomendada |
|---|---|---|---|---|---|
| Persuadible | 123 | 0,67 % | -0,286 | -0,0045 | **Contactar**: el contacto reduce la probabilidad de traspaso |
| Sleeping dog | 358 | 1,96 % | +0,066 | +0,0124 | **NO contactar**: el contacto la aumenta |
| Neutral | 17 766 | 97,36 % | 0,000 | +0,0025 | No priorizar: el contacto no cambia la decision (desperdiciado) |

El X-learner subestima la MAGNITUD del efecto en persuadibles (-0,0045 estimado vs -0,286 verdadero:
el segmento es solo 0,67 % de la cartera, muy pocos ejemplos tratados para calibrar bien el nivel) pero
**ordena correctamente** (persuadible < neutral < sleeping dog en promedio), que es lo que importa para
rankear a quien contactar.

### 6.6 Runtime

| Paso | Segundos |
|---|---|
| `generar_tablas` + `construir_dataset_causal` | 9,5 |
| Entrenar (propension + T/X/DR-learner + riesgo) | 16,2 |
| Qini/AUUC | 0,4 |
| **Total** | **26,2 s** |

Muy por debajo del limite de 3 minutos de la ronda (10 000 afiliados). Un diagnostico adicional a
30 000 afiliados (no forma parte de la corrida de referencia) corrio en 70 s y mostro la MISMA
conclusion: mejor correlacion/MAE con mas datos (Spearman X-learner 0,246, DR-learner 0,250) y mejor
razon de negocio (X-learner 3,2x-6,6x vs riesgo), pero el Qini por corte del holdout SIGUE sin ser
positivo en los 3 cortes para ningun metodo de forma consistente: el problema es el TAMANO del holdout
aleatorio (10 % de un test ya chico), no la escala de la simulacion.

## 7. Lectura de negocio

**Recomendacion: usar el X-learner para priorizar el cupo de contacto del ejecutivo propio**, no el
ranking de riesgo de 02. Con un presupuesto del 20 % de la cartera se evitan ~2,75x mas traspasos que
contactando por riesgo, y se contactan muchos menos sleeping dogs (0,6 % vs 3,0 % de los contactados).
El DR-learner es una alternativa muy cercana (mejor correlacion con la verdad, algo peor en la metrica
de negocio agregada) y **como los dos meta-learners coinciden en el ranking cualitativo, valdria la
pena promediar sus scores en produccion** en vez de elegir uno solo (proximo paso, no implementado por
tiempo).

La tabla "segmento -> accion" (S6.5) es la version accionable para el equipo comercial: dejar de
contactar automaticamente a quien tiene bajo uso digital y (si se pudiera observar) alta sensibilidad a
comision -- son sleeping dogs, un contacto los empuja a irse -- y priorizar a quien tuvo un reclamo
reciente YA resuelto, saldo medio y uso digital medio.

**Advertencia honesta**: el Qini medido en el holdout aleatorio (la validacion sin sesgo real) NO
alcanza significancia estadistica a esta escala (S6.3): la recomendacion se apoya en la correlacion
contra la verdad del simulador (S6.2, solo posible en este sandbox) y en la metrica de negocio
oracular (S6.4, tambien solo posible aqui). Antes de apostar presupuesto real a esto, la AFP
necesitaria un piloto aleatorio real de tamano bastante mayor al 10 % x 3 cortes de aqui.

## 8. Decisiones tomadas, supuestos y limitaciones

**Decisiones / desviaciones del brief (declaradas):**

* **Escala de referencia: 10 000 afiliados** (no 20 000 como en 02/el brief original de 07), y
  `n_jobs=2` en todo LightGBM -- restriccion dura de esta ronda (4 vCPU compartidas con otros dos
  workstreams, prioridad velocidad). Con 10 000 afiliados la corrida completa toma 26 s, muy por
  debajo del limite de 3 minutos; se valido adicionalmente a 30 000 (70 s, S6.6) para confirmar que la
  conclusion no cambia con mas escala.
* El efecto causal se aplica con un **flip probabilistico del target ya construido** (`y0` de 02),
  con randomizacion propia e independiente, en vez de reconstruir el hazard mes a mes dentro del loop
  de `churn_traspaso.synthetic` (que esta prohibido modificar y no expone la probabilidad interna por
  fila). Es matematicamente equivalente a desplazar la probabilidad del evento en la magnitud del
  efecto (`p_retencion_persuadible`, `p_despierta_sleeping_dog`) condicional en `y0`, y garantiza
  coherencia EXACTA con `traspasos` para las filas de control (verificado por test).
* El CATE verdadero (`lat_cate`) es **piecewise-continuo**: constante por segmento salvo por el factor
  de riesgo (`riesgo_latente_heuristico`) que lo escala. No es una funcion suave arbitraria de `x`,
  pero es exactamente lo que el brief describe ("heterogeneo por segmento").
* No se integro con `churn_traspaso.licitacion.ranking_combinado` (opcional en el brief): sin tiempo
  en esta ronda.
* No se genero una tabla de importancia de features (gain) ni ejemplos individuales explicados en
  detalle (CONVENCIONES SS4 los pide para el modelo principal): con 5 modelos LightGBM distintos
  (propension, mu0, mu1, tau0/tau1 o el regresor DR) elegir cual explicar y como comparaba mal contra
  el resto del alcance en el tiempo disponible. **Pendiente.**

**Limitaciones:**

* **Qini/AUUC en el holdout aleatorio no alcanza significancia estadistica** a la escala de referencia
  (S6.3): el holdout de test tiene ~600 filas por corte y el efecto ocurre en ~2,7 % de la poblacion
  con magnitudes de 10-40 pp sobre una tasa base de 2-4 % -- hacen falta muchas mas filas en el
  holdout (un piloto real bastante mas grande, o una ventana de test mas larga) para que la
  diferencia entre metodos sea distinguible del ruido de muestreo con este diseno.
* Las magnitudes de `p_retencion_persuadible` y `p_despierta_sleeping_dog`, y los umbrales que definen
  "sleeping dog" (sensibilidad a comision latente, uso digital bajo), son **supuestos de trabajo sin
  fuente publica** (no existe, que se haya encontrado, una campana de retencion de AFP con evaluacion
  causal publicada): se declaran como tales, no como calibracion.
* El score heuristico de riesgo que genera la propension confundida y escala `lat_cate` es una formula
  simple y transparente PROPIA de este proyecto, no el hazard real de `churn_traspaso` (que es mucho
  mas rico): puede subestimar el confounding real de una campana verdadera.
* Los meta-learners se entrenaron con una sola semilla (42): no se corrio un barrido de estabilidad de
  semillas como en 02 (`--semillas-estabilidad`) por tiempo; el diagnostico a 30 000 afiliados (S6.6)
  es la unica evidencia de robustez fuera de la semilla de referencia.
* `run.py` no persiste los modelos entrenados a disco (a diferencia de `churn_traspaso.model`, que
  tiene `guardar_modelo`/`cargar_modelo`): no hizo falta para esta demo, pero seria necesario para
  produccion.

## 9. Como correr y tiempos

```bash
# tests del proyecto (13 tests, ~5 s)
PYTHONPATH=/home/user/wt/churn python3 -m pytest afp_lab/uplift_retencion/tests -q

# corrida de referencia (10 000 afiliados, ~26 s)
PYTHONPATH=/home/user/wt/churn python3 -m afp_lab.uplift_retencion.run \
    --n-afiliados 10000 --salida salidas/uplift_retencion
```

Salidas en `salidas/uplift_retencion/reportes/*.csv` (split, segmentos, overlap, Qini agregado y por
corte, correlacion vs `lat_cate`, traspasos evitados, razon uplift/riesgo, tabla segmento -> accion,
tiempos).

## 10. Propuestas para `common/` y proximos pasos

* **Propuesta para `common/`**: una funcion `qini_auuc`/`bootstrap_qini` generica (hoy vive en
  `uplift_retencion/evaluate.py`) le serviria tambien al proyecto 09 (`bandit_contacto`, evaluacion
  off-policy) si su diseno tambien usa un holdout aleatorio; no se propuso el movimiento en esta
  ronda por tiempo, queda para que el gerente lo evalue.
* Promediar (o apilar) los scores de X-learner y DR-learner en vez de elegir uno solo (S7).
* Aumentar el tamano del holdout aleatorio (o el numero de cortes de test) para que Qini alcance
  significancia estadistica; hoy es el cuello de botella de la validacion, no la calidad del modelo.
* Integrar el CATE con `churn_traspaso.licitacion.ranking_combinado` (opcional del brief, no hecho).
* Barrido de semillas (como en 02) para reportar rangos, no solo un punto.
* Guardar/cargar modelos a disco si esto pasa de demo a piloto real.
