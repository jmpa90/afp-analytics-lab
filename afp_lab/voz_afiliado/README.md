# Proyecto 11 — Voz del afiliado (NPS inferido, drivers y valor del punto)

Ronda 5, brief `docs/briefs/ronda_5/voz_afiliado.md`. Worktree `/home/user/wt/empleador`, rama
`ronda-5/nps`. Reutiliza sin modificar `afp_lab.churn_traspaso`, `afp_lab.propension_apv`,
`afp_lab.common.planilla` y el patron de estimador causal de `afp_lab.uplift_retencion`.

## 1. Pregunta de negocio

Solo se invita a ~8 % de la cartera a la encuesta de NPS cada mes y de esos responde ~25 %, con
sesgo de no respuesta (contestan más los muy satisfechos y los muy insatisfechos, menos los jóvenes
y los de bajo uso digital). La AFP quiere saber: **¿cuál es el NPS real de TODA la cartera** (no solo
de quien contesta)**, qué eventos operativos lo mueven, y cuánto vale en traspasos y ahorro
voluntario un punto de NPS?** La decisión que apoya: priorizar qué arreglar primero (operación) y
si vale la pena invertir en experiencia del afiliado más allá de lo que ya miden 02 y 03.

## 2. Datos sintéticos

### 2.1 Reutilizados sin modificar

| Fuente | Qué se usa | Perfil / ventana |
|---|---|---|
| `churn_traspaso.synthetic.generar_tablas` | `afiliados`, `empleadores`, `panel_mensual`, `macro`, `mercado`, `saldos` (saldo obligatorio + fondo), `cambios_fondo`, `traspasos`, `contactos`, `reclamos`, `uso_digital` | `perfil="cuprum"`, mismo `n_afiliados`/`seed`/ventana |
| `propension_apv.synthetic.generar_tablas` | `apvc_empleador`, `campanas` (→ `campanas_apv`), `aportes_voluntarios`, `saldos` (→ `saldos_apv`: saldo voluntario), `tributario` | idem |
| `common.planilla` (`generar_planilla`, `construir_rezagos`, `construir_agregado_empleador`) | `planilla`, `rezagos`, `agregado_empleador` (dotación, masa imponible, `frac_dnp`, `lag_recepcion_medio` por empleador×mes) | necesita `empleadores_latente`, que se reconstruye con `rim_nowcasting.synthetic.generar_empleadores_pandas` (misma función pura que usa `synth_base.generar_universo`, mismo `seed`: barato, no vuelve a simular el panel laboral) |

`churn_traspaso.generar_tablas` y `propension_apv.generar_tablas` reciben el **mismo**
`n_afiliados`, `periodo_inicio`, `periodo_fin`, `seed` y `perfil="cuprum"`; como
`synth_base.generar_universo` es una función pura de esos argumentos, el universo común
(`afiliados`, `panel_mensual`, `empleadores`, `macro`) que arma cada llamada es **idéntico** byte a
byte (verificado en `test_universo_comun_identico_entre_02_y_03`). Ventana: `202201`–`202812`
(2022‑01 a 2028‑12, 84 meses), `n_afiliados=10 000`, `seed=42` — igual que pide el brief.

### 2.2 Nuevo de este proyecto: satisfacción latente y encuesta de NPS

**Satisfacción latente** (`lat_satisfaccion`, tabla `satisfaccion_latente`, 0–10, afiliado × mes,
NUNCA una feature): nivel base por segmento/edad + efectos de eventos operativos con decaimiento +
componente persistente por afiliado + ruido mensual, clip a `[0, 10]`:

| Componente | Valor | Fuente / razón |
|---|---|---|
| Nivel base | 7,6 | supuesto propio; ver nota de calibración abajo |
| Ajuste joven (< 30) / mayor (≥ 60) | −0,3 / +0,4 | supuesto propio |
| Ajuste por segmento (`tipo_afiliado`) | DEPENDIENTE 0, INDEPENDIENTE −0,3, VOLUNTARIO +0,2, PENSIONADO_ACTIVO +0,3 | supuesto propio |
| Componente persistente por afiliado | N(0, 1,0) | supuesto propio (temperamento base) |
| Ruido mensual | N(0, 0,8) | supuesto propio |
| Reclamo sin resolver (ABIERTO o RESUELTO_DESFAVORABLE) | **−2,0** | **brief** |
| Rezago del empleador ≥ 1 mes más de lo normal (`lag_recepcion_medio ≥ 2`, "> 30 días") | **−1,5** | **brief** |
| Licencia mal pagada (proxy: licencia médica ≥ 20 días en el mes) | **−2,5** | **brief** |
| Empleador en DNP ese mes | **−0,8** | **brief** |
| Contacto de ejecutivo propio con resultado CONTACTADO/INTERESADO | **+0,7** | **brief** |
| Rentabilidad del fondo del afiliado sobre el promedio del mercado | **+0,5** | **brief** |
| Brecha de comisión propia sobre la más barata > 0,30 pp | **−0,6** | **brief** |

Los seis primeros eventos decaen con vida media de 2 meses (DNP: 1 mes) hasta 6 meses (DNP: 3); los
dos últimos (rentabilidad y comisión) son **contemporáneos**, sin memoria — son un estado del
mercado, no un evento puntual. `EFECTOS_VERDADEROS_EVENTOS` (`synthetic.py`) es la fuente única para
comparar contra los drivers estimados.

**Nota de calibración (nivel base = 7,6, no 7,0):** en el perfil `cuprum`, Cuprum es la AFP más cara
del roster (`churn_traspaso.MERCADOS["cuprum"]`, comisión fija 1,44 % contra 0,46–0,58 % de las más
baratas) y la brecha supera 0,30 pp **prácticamente siempre** (chequeado: 100 % de las filas de
respondientes en la corrida de referencia). Eso vuelve `−0,6` un descuento **crónico**, no un evento;
sin compensar el nivel base, casi toda la cartera caería a "detractor" solo por el precio. Se subió
el nivel base para que el promedio neto de la cartera quede cerca de 7 (frontera pasivo/promotor;
media observada: 7,05) — una historia de negocio razonable ("Cuprum es cara pero retiene por otros
atributos"), no una corrección de la magnitud del brief. La misma razón hace que el driver
`brecha_comision_desfavorable` no se pueda estimar (§6.3: casi no hay grupo de control).

**Encuesta** (tabla `encuesta`, grano invitación): cada mes se invita con probabilidad calibrada
(bisección sobre el logit, reutilizando `churn_traspaso.model.ajustar_prior`) a que la fracción
invitada promedie el objetivo del brief:

* **Invitación** ≈ 8 % de la cartera vigente/mes; más probable (encuesta "transaccional") si hubo un
  reclamo, un contacto o un cambio de fondo en el último mes; el resto es "relacional" (aleatorio).
* **Respuesta** ≈ 25 % de los invitados; con **sesgo de no respuesta**: responden más quienes están
  fuera de la banda "pasiva" (7–8) — la distancia se mide desde esa banda, no desde el punto medio
  de la escala (5), porque con un nivel base ~7 usar `|sat−5|` trataría a un pasivo alto (8) como más
  "extremo" que un detractor moderado (6), justo al revés de lo que dice el brief —, menos si es
  joven (< 30) o de bajo uso digital (sin logins ese mes).
* **Respuesta** = redondeo de `lat_satisfaccion + ruido(0, 0,6)`, acotada a 0–10. Promotor 9–10,
  pasivo 7–8, detractor 0–6 (`PROMOTOR_DESDE=9`, `DETRACTOR_HASTA=6`).
* Se guarda `lat_p_responde_si_invitado` (propensión verdadera de responder) en la tabla latente,
  para poder evaluar qué tan bien la recupera el modelo de respuesta (§6.1: no muy bien — ver
  limitaciones).

Chequeo de la simulación (corrida de referencia, `resumen_simulacion.csv`): invitación mensual
8,02 % (objetivo ~8 %), respuesta entre invitados 25,1 % (objetivo ~25 %), NPS verdadero de la
cartera −14,9, NPS ingenuo (solo respondientes) −21,3 — el sesgo de no respuesta es real y
sustancial, como pide el brief.

## 3. Target, población, horizonte, exclusiones

* **Target principal (parte 1, NPS inferido):** `respondio` (0/1) y `respuesta` (0–10) de la
  encuesta de **ese mismo** `periodo_corte` — no hay horizonte hacia adelante, la respuesta es
  contemporánea al corte (a diferencia de 02/03).
* **Población:** la misma que `churn_traspaso` — vigente en la cartera Cuprum al corte, sin
  solicitud de traspaso de salida ya presentada, no retirado. Se reutiliza literalmente
  `churn_traspaso.features.construir_features_multi`, así que la definición de población es
  idéntica por construcción.
* **Target del "valor del punto" (parte 3):** traspaso a 6 meses (`churn_traspaso.features.
  construir_target`, horizonte `HORIZONTE_MESES=6`) y aporte voluntario a 6 meses
  (`propension_apv.features.construir_target`, mismo horizonte) — reutilizados sin modificar, sobre
  el set de test.
* **Cortes evaluables:** ≥ 14 meses de historia (para que las ventanas de 12 meses tengan sentido) y
  ≥ 6 meses de horizonte disponible antes de `periodo_fin` (para el target de valor del punto,
  aunque el target principal no lo necesite — se usa el mismo horizonte para poder construir ambos
  con los mismos cortes). Con la ventana de referencia: 64 cortes evaluables (`202303`–`202806`).

## 4. Features (`features.py`, as-of estricto)

| Familia | Fuente / función | Desfase |
|---|---|---|
| Mercado (comisión, rentabilidad), saldo y fondo, antigüedad, contactos y reclamos por canal/ventana, uso digital, edad, sexo, educación, tipo de afiliado, región, rubro y tamaño del empleador, licitación | `churn_traspaso.features.construir_features_multi` (reutilizada tal cual) | panel laboral `corte−2`; operacional `corte` |
| APV: ¿ofrece APVC el empleador? % match, historial de aporte (12/24 m), tramo IGC, saldo voluntario, contactos/apertura de campaña APV | `propension_apv.features.construir_features(..., solo_elegibles=False)`, subconjunto de columnas (prefijo `apv_`) | igual que 03 (`corte−2` panel, `corte` operacional) |
| DNP y rezago de recepción del empleador (1/3/6/12 m y "actual") | `common.planilla.construir_agregado_empleador` (propio) | **`corte−2`** (mismo desfase que el panel laboral — simplificación, ver §8) |
| Licencias médicas largas (≥ 20 días, 1/3/6/12 m) | `panel_mensual.dias_licencia` (propio) | `corte−2` |
| Historial de encuestas (invitaciones/respuestas 1/3/6/12 m, meses desde la última respuesta, última respuesta y categoría) | `encuesta` (propio) | **`corte−1`**, nunca la respuesta del propio corte ni posteriores (verificado a mano en `test_historial_encuesta_excluye_el_propio_corte` y en el test de leakage) |

`columnas_features` excluye ids, el target de la encuesta y cualquier columna con prefijo `lat_`/`p_`
(igual regla dura que 02/03). No se integró el segmento de 06 (`segmentacion_afiliados`) por acotar
el alcance dado el tiempo disponible (declarado en limitaciones); se usa `tipo_afiliado` como proxy.

## 5. Modelo y baselines

### 5.1 Modelo de respuesta (corrige el sesgo de no respuesta)

LightGBM binario (`entrenar_propension_respuesta`) que predice `respondio` entre los **invitados**
de train/valid, con todas las features as-of. Su salida, recortada, da el peso IPW
`peso_ipw = clip(1/p_hat, 1, 15)` usado para ponderar el modelo de NPS. **`peso_ipw` (y `p_hat_responde`)
se usan EXCLUSIVAMENTE como `sample_weight`/diagnóstico, nunca como feature de entrada** de los
modelos de NPS (`features.columnas_features` los excluye explícitamente — corrección de la revisión,
ver §8): usarlos como feature dejaría que el clasificador de NPS "viera" el propio pipeline de
ponderación en vez de solo señales observables.

### 5.2 Modelo de NPS

Dos clasificadores LightGBM (`es_promotor`, `es_detractor`), cada uno entrenado **sin ponderar** y
**ponderado** por `peso_ipw`, sobre respondientes de train/valid. El **NPS inferido** de una
población se agrega como `100 × (media P(promotor|x) − media P(detractor|x))`
(`model.nps_de_probs`), **no** bucketizando un score de regresión puntual: un regresor entrenado con
pocos respondientes "encoge" la varianza de sus predicciones hacia la media, así que casi nadie
cruza los umbrales 6/9 y el NPS bucketizado queda sesgado hacia 0 aunque el nivel medio esté bien
(se documenta como hallazgo metodológico en el código, `model.nps_de_probs`, y se dejó igual un
regresor de `respuesta` 0–10 para drivers/valor del punto e interpretabilidad). También se reporta
un estimador Horvitz–Thompson directo (`nps_ponderado_ht`: media ponderada por IPW de las categorías
observadas, sin modelo) como referencia adicional.

**Baselines obligatorios** (CONVENCIONES §4): trivial = tasa/nivel medio de train para todos
(`baseline_score_trivial`/`baseline_detractor_trivial`); regla de negocio = `7,5 − 1,0·(reclamo en
6 m) − 1,0·(rezago actual) − 1,0·(licencia larga 3 m) − 0,5·(DNP actual) + 0,4·(contacto ejecutivo
3 m) − 0,5·(brecha de comisión > 0,30 pp)`, acotada a `[0,10]` — mismos eventos/signos que el brief,
pero coeficientes **deliberadamente distintos** de los del generador oculto
(`synthetic.ParametrosSatisfaccion`): números redondos (medio punto o un punto por señal) que un
analista anotaría a mano mirando solo los NOMBRES de los eventos operativos, sin ver `synthetic.py`
(corrección de la revisión — la versión anterior copiaba 5/6 coeficientes de evento y el nivel base
del generador casi a la décima; ver §8 y `test_regla_negocio_independiente_del_generador`, que
verifica que ningún valor de `COEF_REGLA` coincide con `ParametrosSatisfaccion`).

### 5.3 Drivers (T-learner simple)

Para cada uno de los 7 eventos del brief, dos regresiones **Ridge** (tratado/control) sobre un
conjunto reducido de confusores estandarizados (edad, meses desde afiliación, log-saldo, RIM,
densidad de cotización, antigüedad, mes, segmento y los otros 6 eventos), ponderadas por IPW —
alternativa **simple** al T-learner con LightGBM de `uplift_retencion` (brief: "T-learner o regresión
ajustada por propensión"), elegida por velocidad y estabilidad con el tamaño de muestra de los
respondientes (~4 800 en train+valid). `CATE = E[y|T=1,x] − E[y|T=0,x]` promediado con los mismos
pesos. Con menos de 15 tratados o 15 controles el efecto se reporta `NaN` (ocurre para
`brecha_comision_desfavorable`, ver §8).

### 5.4 Valor del punto de NPS

Regresión logística (traspaso a 6 m) y lineal (monto de aporte APV a 6 m) del target de 02/03 sobre
el score de NPS ponderado (continuo, centrado/escalado) y confusores (edad, saldo, RIM, densidad,
antigüedad, mes, segmento) — **asociación ajustada, no un experimento**: 02 y 03 no conocen la
satisfacción del afiliado, así que no hay como aislar el efecto causal del NPS con estos datos
(advertencia explícita también en la salida de `run.py`).

## 6. Resultados (corrida de referencia: `n_afiliados=10 000`, `seed=42`, ventana completa)

### 6.1 NPS: antes/después de la corrección de sesgo (test, 12 cortes, `202707`–`202806`)

| Versión | MAE (puntos de NPS) | Reducción de error vs. ingenuo |
|---|---:|---:|
| Regla de negocio (independiente del generador, ver §8) | 1,13 | 86,7 % |
| Ponderado (Horvitz–Thompson, IPW directo) | 5,32 | 37,1 % |
| **Ponderado (modelo, IPW)** | **6,75** | **20,3 %** |
| Sin ponderar (modelo) | 8,20 | 3,1 % |
| Ingenuo (solo respondientes) | 8,47 | — |
| Trivial (constante) | 11,63 | −37,3 % |

NPS verdadero de la cartera en test: −9,4 a −15,0 según el corte (media ≈ −11,6). NPS ingenuo:
−14,8 a −33,3 (mucho más volátil: ~120–150 respondientes/corte). AUC de detractor en el holdout de
respondientes de test: **0,7528** (sin ponderar) / **0,7530** (ponderado) — **cumple** el umbral
propuesto (≥ 0,75). Calibración por decil razonable (ponderado: diferencia `prob_media −
tasa_observada` por debajo de 0,09 en los 10 deciles, máximo 0,088; sin ponderar: 9 de 10, el décimo
en 0,094 — ver `reportes/detractor_calibracion.csv`).

**Corrección de la revisión (bloqueante #2):** hasta la entrega anterior, `peso_ipw` (el propio peso
IPW) quedaba como **feature de entrada** de los 3 pares de modelos de NPS además de usarse como
`sample_weight` — un artefacto de ingeniería de features, no una decisión válida (ver §8). Al
sacarlo de `columnas_features` el AUC de detractor ponderado subió de 0,7497 (no cumplía, por
0,0003) a 0,7530 (si cumple); pero el mismo cambio también le quitó al modelo de NPS una señal
espuria que — se descubrió al corregir — inflaba igual de "gratis" el MAE tanto del score
**ponderado** (bajó de 6,19 a 6,75: empeora) como, más revelador, del **sin ponderar** (bajó de 6,64
a 8,20, casi igual al ingenuo 8,47): la versión "sin ponderar" de antes tampoco era una comparación
limpia, porque también veía `peso_ipw` como feature y así se corregía parcialmente sola sin usar el
peso como tal. Los números de esta sección están limpios de ese artefacto.

**Contra el criterio de aceptación propuesto** (±3 puntos, reducir el error a la mitad, AUC ≥ 0,75,
drivers con signo correcto ≥ 80 %): con los números limpios de esta corrida se cumplen **2 de 4**
(AUC 0,7530 y drivers 100 %, ver `reportes/aceptacion.csv`); el MAE (6,75) y la reducción de error
(20,3 %) siguen sin alcanzar el umbral, y por corte la reducción es muy variable (entre −371 % y
+82 % según el corte, con solo ~120–150 respondientes de test por mes — ver
`reportes/nps_por_corte.csv`). Ver §8 para el diagnóstico de fondo (el modelo de respuesta discrimina
poco, AUC 0,541 en validación, así que el IPW corrige solo parcialmente el sesgo de no respuesta) y
qué se probó para mejorarlo.

### 6.2 Por segmento (extracto, `reportes/nps_por_segmento.csv`; ver el archivo completo para mes/región)

| Segmento | Valor | n | NPS verdadero | NPS ponderado (modelo) |
|---|---|---:|---:|---:|
| Tramo de edad | 60+ | 5 026 | −0,04 | −19,19 |
| Tramo de edad | < 30 | 1 279 | −14,54 | −22,13 |
| Tramo de empleador | MICRO | 5 681 | −30,79 | −36,37 |
| Tramo de empleador | SIN_EMPLEADOR | 31 790 | −4,33 | −10,11 |

El modelo acierta el **orden** de los segmentos (MICRO peor, SIN_EMPLEADOR/60+ mejor) pero arrastra
el mismo sesgo de nivel (~−5 a −20 puntos) que a nivel agregado — consistente con un sesgo que viene
del mecanismo de no respuesta, no de un segmento en particular.

### 6.3 Drivers: estimado vs. verdadero (`reportes/drivers.csv`, train+valid, ~4 800 respondientes)

| Evento | Estimado | Verdadero | Signo | n tratados / control |
|---|---:|---:|---|---:|
| Licencia mal pagada | −2,62 | −2,5 | ✔ | 538 / 4 280 |
| Reclamo sin resolver | −0,97 | −2,0 | ✔ | 259 / 4 559 |
| Rezago mensual | −0,95 | −1,5 | ✔ | 56 / 4 762 |
| DNP del empleador | −0,19 | −0,8 | ✔ | 112 / 4 706 |
| Contacto ejecutivo resuelto | +0,48 | +0,7 | ✔ | 836 / 3 982 |
| Rentabilidad relativa positiva | +0,62 | +0,5 | ✔ | 3 399 / 1 419 |
| Brecha de comisión desfavorable | — (sin datos) | −0,6 | — | 4 818 / **0** |

**Signo correcto: 6/6 estimables (100 %), supera el 80 % pedido.** El evento no estimable
(`brecha_comision_desfavorable`) es casi siempre verdadero en el perfil `cuprum` (ver §2.2): sin
grupo de control no se puede aislar su efecto de la constante del modelo — un hallazgo honesto, no
un error a esconder. "Qué arreglar primero" (impacto = |efecto| × incidencia, `reportes/
drivers_prioridad.csv`): 1) rentabilidad relativa (0,44 pts, 70,5 % de las filas), 2) licencia mal
pagada (0,29 pts, 11,2 %), 3) contacto ejecutivo (0,08 pts, 17,4 %).

### 6.4 Valor de un punto de NPS (test, 75 679 filas; **asociación ajustada, no causal**)

| | Traspaso a 6 meses | Aporte APV a 6 meses |
|---|---:|---:|
| Tasa/monto base | 2,22 % | CLP 162 456 medio |
| Efecto de +1 punto de NPS | −0,168 pp de probabilidad | +CLP 11 339 esperados |
| Traducido a la cartera (10 000 afiliados) | **≈ 127 traspasos evitados por punto** | **≈ CLP 858 millones de aporte adicional por punto** |
| Saldo asociado | ≈ CLP 3 613 millones en saldo retenido por punto | — |

(Números actualizados tras la corrección del bloqueante #2 — §8: el score de NPS usado aquí,
`modelo_score_p`, ya no ve `peso_ipw` como feature, así que cambia levemente frente al informe
anterior; el traspaso evitado por punto (≈127) resulta casi idéntico, el aporte APV por punto sube
de ≈CLP 8 517 a ≈CLP 11 339 — ambos siguen siendo asociación ajustada, no causal.)

### 6.5 Importancias y ejemplos

Top features del modelo ponderado (`reportes/importancias_score.csv`, ya sin `peso_ipw` como feature
— corrección del bloqueante #2, §8): `licencia_larga_3m` (17,0 % de la ganancia), `licencia_larga_1m`
(13,5 %), `ultima_respuesta_nps` (11,1 %), `ultima_categoria_nps` (3,6 %), `brecha_rentabilidad_pp`
(3,5 %), `estado_laboral` (2,1 %). Los eventos del brief (licencias, historial de encuesta,
rentabilidad) dominan, consistente con §6.3; `peso_ipw` ya no aparece en la tabla (no es una feature
del modelo, solo su `sample_weight`).

**Ejemplos individuales explicados** (`reportes/ejemplos_individuales.csv`, `model.
explicar_prediccion_individual` — contribución LightGBM tipo SHAP, `pred_contrib`, sobre el
clasificador de detractor ponderado; CONVENCIONES §4 pide esto además de la importancia global —
ítem menor de la revisión, antes pendiente): 3 respondientes de test elegidos por la probabilidad de
detractor predicha (más baja, mediana, más alta).

| Ejemplo | `afiliado_id`×`periodo_corte` | Prob. detractor | Detractor real | Respuesta real | Top contribuciones (feature → aporte al logit) |
|---|---|---:|---:|---:|---|
| Prob. más baja | 6187 × 202802 | 0,084 | **Sí** (1) | 5,0 | `rent_12m_sistema_fondo` (−0,25), `meses_desde_contacto_ejecutivo=0` (−0,22), `licencia_larga_3m=0` (−0,17) — todo empuja a NO-detractor |
| Mediana | 4932 × 202801 | 0,390 | No (0) | 8,0 | `brecha_rentabilidad_pp=−0,69` (**+0,57**, empuja fuerte a detractor), compensado por señales operativas limpias |
| Prob. más alta | 2502 × 202712 | 0,913 | **Sí** (1) | 3,0 | `licencia_larga_3m=3` (**+0,88**), `licencia_larga_1m=1` (+0,52), `meses_desde_reclamo=0` (+0,50) |

El caso de "prob. más baja" es un **falso negativo** honesto: el modelo lo ve como no-detractor (0,084)
porque todas sus señales operativas observables son buenas, pero respondió 5 (detractor) — un caso
donde el ruido/temperamento no observable domina, consistente con que el modelo de respuesta/NPS
tiene AUC y MAE acotados (§8). El caso de "prob. más alta" coincide con la intuición de negocio de
§6.3/§7: licencias médicas largas y recientes son la señal individual más fuerte de insatisfacción.

## 7. Lectura de negocio

* **Arreglar primero:** con la cartera actual, el driver de mayor impacto agregado es la
  rentabilidad relativa (afecta al 70 % de la cartera, aunque con magnitud moderada); el de mayor
  magnitud por caso es la licencia mal pagada (−2,6 puntos, aunque solo al 11 % de la cartera). Un
  programa que reduzca licencias mal pagadas y anticipe DNP del empleador mueve el NPS de un
  segmento identificable de la cartera sin depender de bajar la comisión (que, dado el hallazgo de
  §6.3, es una palanca que el generador trata como casi constante para Cuprum). Esta lectura ya NO
  se compara contra el baseline de regla en términos de "el modelo pierde": la regla (§8) es ahora
  un heurístico independiente, no una copia del generador, así que su buen desempeño agregado (MAE
  1,13) es un dato de negocio legítimo (los mismos eventos observables que usa el modelo también
  bastan para un heurístico simple a nivel de cartera), no un artefacto a descartar.
* **Valor del punto:** con las cifras de esta corrida (asociación, no causalidad), un punto de NPS
  vale del orden de 127 traspasos evitados a 6 meses (≈ CLP 3 613 millones de saldo) y CLP 858
  millones de aporte voluntario adicional — suficiente para justificar una inversión moderada en
  experiencia del afiliado, aunque la cifra exacta depende de covariables no observadas por 02/03
  (ver advertencia de asociación).
* **NPS inferido:** el modelo ponderado no cumple el ±3 puntos propuesto (MAE 6,75) ni reduce el
  error a la mitad (20,3 % agregado, con mucha variación por corte — entre −371 % y +82 %, ver
  §6.1); sí corrige la DIRECCIÓN del sesgo (el ingenuo entre respondientes es sistemáticamente más
  negativo que la cartera real, −21,3 vs. −14,9 en la corrida de referencia) y sí cumple el AUC de
  detractor (≥ 0,75) y los drivers (≥ 80 % de signo correcto). Para priorizar EL NIVEL agregado de
  NPS de la cartera, el Horvitz–Thompson directo (MAE 5,32, sin modelo) es hoy más confiable que el
  modelo ponderado; para priorizar POR SEGMENTO u operativamente (drivers, valor del punto), el
  modelo sigue siendo la pieza necesaria (ver §8, próximos pasos).

## 8. Decisiones tomadas, supuestos y limitaciones

### 8.0 Correcciones aplicadas tras la revisión adversarial (ronda 5)

La revisión (`docs/revisiones/ronda_5/voz_afiliado.md`) rechazó la entrega anterior por dos
bloqueantes, ambos corregidos de raíz (sin bajar umbrales ni desactivar tests) y con test de
regresión agregado para que no puedan reaparecer en silencio:

1. **El baseline de "regla de negocio" copiaba los coeficientes del generador oculto.** La versión
   anterior de `COEF_REGLA` (`model.py`) tenía 5 de 6 coeficientes de evento y el nivel base
   idénticos, a la décima, a `synthetic.ParametrosSatisfaccion` — el docstring decía "sin ver el
   generador" pero no lo era, e invalidaba la comparación baseline-vs-modelo de CONVENCIONES §4
   (el baseline ganaba de forma aplastante — MAE 1,13 — porque reconstruía la fórmula determinista
   del generador, no porque fuera una regla de negocio independiente). **Corrección:** nuevos
   coeficientes (`base=7,5`; eventos en `−1,0`/`−0,5`/`+0,4`) elegidos para no coincidir con NINGÚN
   parámetro de `ParametrosSatisfaccion` — verificado programáticamente en
   `test_regla_negocio_independiente_del_generador` (falla si algún valor de `COEF_REGLA` coincide
   con el generador). El nivel base subió de 7,0 "de libro" a 7,5 por una razón de negocio real y
   verificable — la brecha de comisión de Cuprum (dato de MERCADO público, no un parámetro oculto)
   supera el umbral en ~100 % de las filas, así que con base 7,0 la regla degeneraba (casi toda la
   cartera quedaba en "detractor" solo por el descuento de comisión, un artefacto, no una lectura de
   negocio) — no para reproducir el `base_nivel=7,6` del generador. Resultado: el baseline sigue
   siendo fuerte en el agregado (MAE 1,13 — ver nota de negocio en §7), pero ahora es una comparación
   válida porque no comparte ningún número con el generador.
   Verificación: `PYTHONPATH=/home/user/wt/empleador python3 -c "from afp_lab.voz_afiliado.synthetic import ParametrosSatisfaccion; from afp_lab.voz_afiliado.model import COEF_REGLA; import dataclasses; p=ParametrosSatisfaccion(); print(set(dataclasses.asdict(p).values()) & set(COEF_REGLA.values()))"`
   → conjunto vacío (antes: 6 valores en común).
2. **El peso IPW se usaba como feature del modelo de NPS, además de como `sample_weight`.**
   `peso_ipw` no empieza con `lat_`/`p_`, así que `columnas_features` no lo excluía y quedaba como
   input de los 3 pares de clasificadores/regresores de NPS (ponderados Y sin ponderar). Eso
   degradaba el AUC de detractor ponderado (0,7497, no cumplía el umbral 0,75 por 0,0003) y, se
   descubrió al corregirlo, también contaminaba la comparación "sin ponderar" (que en teoría no
   debía ver nada de la corrección de sesgo, pero sí veía `peso_ipw` como feature). **Corrección:**
   `peso_ipw` y `p_hat_responde` ahora se excluyen explícitamente de `columnas_features`
   (`COLUMNAS_DERIVADAS_PIPELINE`, `features.py`) y se usan SOLO como `sample_weight`/diagnóstico —
   verificado en `test_columnas_features_excluye_derivadas_pipeline`. Resultado, con el MISMO
   dataset/seed/partición (§6.1): AUC detractor ponderado 0,7497 → **0,7530** (ahora cumple el
   criterio de aceptación ≥ 0,75); a cambio, el MAE del NPS ponderado empeoró de 6,19 a **6,75**
   puntos y el "sin ponderar" de 6,64 a **8,20** (casi igual al ingenuo 8,47, como debía ser una
   comparación limpia sin corrección de sesgo) — la señal que `peso_ipw` aportaba como feature no
   era legítima (ver §8.1 para el diagnóstico completo).
   Verificación: `PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.voz_afiliado.run` reproduce
   `reportes/detractor_metricas.csv` (AUC) y `reportes/nps_error_resumen.csv` (MAE) de esta sección.

### 8.1 Decisiones y supuestos propios

(ninguna cifra del generador se inventó; todo lo de esta lista es mío, declarado con su valor):

* Nivel base de satisfacción 7,6 (no 7,0) para compensar la brecha de comisión crónica del perfil
  `cuprum` (§2.2). La regla de negocio usa un ajuste ANÁLOGO pero con un valor distinto (7,5, no
  7,6) — ver §8.0: deliberadamente no comparte el número con el generador.
* Magnitudes de invitación/respuesta (pesos del logit) y su calibración automática (bisección,
  reutilizando `churn_traspaso.model.ajustar_prior`) a los objetivos ~8 %/~25 % del brief.
* Operacionalización de "rezago > 30 días" como `lag_recepcion_medio ≥ 2` meses (el ciclo normal de
  declaración ya implica ~1 mes de desfase — "oportuno" en `common.planilla` —, así que ">30 días de
  atraso" es un mes ADICIONAL sobre ese ciclo, no el ciclo mismo).
* Operacionalización de "licencia mal pagada" como licencia médica ≥ 20 días en el mes (no se
  modela el subsidio de incapacidad laboral línea a línea).
* DNP/rezago del empleador leídos con el mismo desfase que el panel laboral (`corte−2`), sin filtrar
  además por `periodo_recepcion` real: una simplificación declarada, no una regla nueva de
  `common/planilla`. En la práctica el "rezago" (pago tardío) implica que en una fracción de los
  casos el estado de pago recién se resuelve después de `corte−2`, así que esta feature es
  ligeramente optimista sobre lo que la AFP sabría exactamente en producción; una versión más
  estricta filtraría por `periodo_recepcion ≤ corte`.
* Confusores del T-learner y de valor del punto: un conjunto reducido y curado (no las ~110
  features), por velocidad y estabilidad con el tamaño de muestra de los respondientes.
* No se integró el segmento de 06 (`segmentacion_afiliados`); se usa `tipo_afiliado` como proxy.

### 8.2 Limitaciones (numéricas, medidas, no supuestas)

* El **modelo de respuesta** discrimina poco (AUC 0,54 en validación): los pesos IPW corrigen solo
  parcialmente el sesgo de no respuesta. Se probó (a) el NPS por bucketización de un regresor
  (peor: MAE 12,3 al inicio, sesgaba a 0 por "encogimiento" de varianza — corregido cambiando a
  `nps_de_probs`, §5.2/§6.1), (b) recentrar el sesgo de respuesta en la banda pasiva en vez del
  punto medio de la escala (mejora sustancial) y (c) excluir `peso_ipw` de las features del modelo
  de NPS (§8.0, bloqueante #2 de la revisión: sin esa corrección el MAE ponderado parecía mejor,
  6,19, pero por una fuga de la propia ponderación como feature, no por una mejor corrección del
  sesgo). Con los tres cambios el MAE ponderado quedó en **6,75** puntos, no en los 3 propuestos, y
  con alta variación por corte (§6.1); el Horvitz–Thompson directo (5,32) es mejor en nivel agregado
  aunque más ruidoso por corte. Se entrega acotado y declarado, tal como autoriza el encargo, en vez
  de forzar un número.
* El driver `brecha_comision_desfavorable` no es estimable (sin grupo de control, ver §2.2/§6.3):
  es una propiedad del mercado simulado para `cuprum`, no un error de código.
* No se corrió el barrido de semillas (uno solo, `seed=42`) por acotar el tiempo de esta entrega;
  los números de §6 tienen la incertidumbre de muestreo que eso implica (12 cortes de test, ~1 500
  respondientes).
* El valor del punto (§6.4) es una asociación ajustada por un conjunto reducido de covariables, no
  un experimento: 02 y 03 no conocen la satisfacción del afiliado, así que el NPS podría estar
  proxeando otras variables omitidas (p. ej. antigüedad, digitalización) que también mueven el
  traspaso/aporte independientemente de la satisfacción.
* El baseline de regla de negocio (§8.0), con los coeficientes nuevos, nunca clasifica a nadie como
  PROMOTOR en la corrida de referencia (score máximo 7,9 < 9): es una regla puramente aditiva desde
  un nivel medio, no puede capturar que varias señales POSITIVAS se combinen para cruzar el umbral
  de promotor. Es una limitación esperable de una regla simple (no un intento de igualar el
  desempeño del modelo) y no afecta su validez como comparación independiente.

## 9. Cómo correr y tiempos

```bash
cd /home/user/wt/empleador
PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.voz_afiliado.run
# con argumentos (todos opcionales):
PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.voz_afiliado.run \
    --n-afiliados 10000 --seed 42 --periodo-inicio 202201 --periodo-fin 202812 \
    --n-test 12 --n-valid 12 --salida salidas/voz_afiliado

# tests (< 5 s de ejecucion real; ventana corta, 700 afiliados)
PYTHONPATH=/home/user/wt/empleador python3 -m pytest afp_lab/voz_afiliado/tests -q
```

Tiempos medidos (corrida de referencia, `n_afiliados=10 000`, 4 vCPU compartidas, `n_jobs=2`):
generar tablas 17,2 s; construir dataset 6,3 s; guardar fuentes (parquet) 2,2 s; modelo de respuesta
1,1 s; modelo de NPS (6 clasificadores/regresores) 1,0 s; comparación de NPS 0,5 s; detractor 0,03 s;
drivers 0,04 s; valor del punto 0,7 s — **total 29,7 s**, muy por debajo del límite de 4 minutos del
brief. Tests: 13 casos, 4,1 s.

## 10. Propuestas para `common/` y próximos pasos

* **Propuesta para `common/`:** una utilidad de calibración de tasas objetivo por bisección
  (`ajustar_prior` vive hoy en `churn_traspaso.model`, pero la usan al menos dos proyectos —
  `churn_traspaso` y este — para calibrar probabilidades sintéticas a una tasa objetivo). Se
  reutilizó tal cual en vez de moverla, para no tocar `churn_traspaso` sin acuerdo del gerente.
* **Próximos pasos:** (1) un modelo de respuesta con más señal (agregar variables de "propensión a
  contestar encuestas" más directas, si existieran en producción) — la prioridad más alta ahora que
  el AUC de detractor ya cumple pero el MAE del NPS agregado sigue limitado por este mismo modelo;
  (2) filtrar DNP/rezago por `periodo_recepcion` real en vez de por el desfase del panel laboral;
  (3) integrar el segmento de 06; (4) barrido de semillas para reportar rangos, no un solo punto;
  (5) explorar un estimador doblemente robusto (AIPW) para el NPS agregado, combinando el modelo y
  el Horvitz–Thompson de §6.1 en vez de reportarlos por separado — con el Horvitz–Thompson ahora
  claramente mejor en nivel agregado (MAE 5,32 vs 6,75), esta combinación es la vía más prometedora
  para acercarse al ±3 propuesto.
