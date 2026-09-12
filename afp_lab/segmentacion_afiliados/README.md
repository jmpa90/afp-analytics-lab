# 06 · Segmentación de afiliados

Responsable: DS-3. Ronda 2 (proyecto nuevo, `BRIEF.md`). Revisado contra `CONVENCIONES.md` §7.

## 1. Pregunta de negocio y decisión que apoya

¿Qué segmentos de comportamiento existen en la cartera de AFP Cuprum y qué acción corresponde a
cada uno (asesoría, retención, APV, producto, cobranza)? La segmentación alimenta la asignación de
ejecutivos, la priorización de campañas de retención/APV y el diseño de mensajes de asesoría
previsional — decisiones operativas mensuales, no un modelo de riesgo de un evento puntual.

## 2. Datos sintéticos

* **Universo propio**: `afp_lab.common.synth_base.generar_universo(perfil="cuprum")` — afiliados,
  panel laboral mensual, empleadores y macro, calibrados a la cartera real de Cuprum
  (`docs/contexto_afp_cuprum.md` §0/§4: mayor densidad, renta e ingreso imponible más altos, más
  jubilación/rotación ejecutiva). Es la fuente de verdad de las features "núcleo" (§4).
* **Tablas externas, sin modificar los proyectos que las generan** (`synthetic.generar_tablas_
  externas`): `afp_lab.churn_traspaso.synthetic.generar_tablas` (saldo obligatorio, fondo,
  antigüedad en la AFP, cambios de fondo, contactos, reclamos, uso digital, traspasos) y
  `afp_lab.propension_apv.synthetic.generar_tablas` (saldo voluntario, aportes voluntarios,
  campañas). Se llaman con el **mismo** `n_afiliados`/`periodo_inicio`/`seed` que el universo propio
  para que el `afiliado_id` k-ésimo exista en los tres universos.
* **Limitación de perfil, RESUELTA (seguimiento post-fusión a `main`)**: hasta la entrega anterior,
  `churn_traspaso.generar_tablas`/`propension_apv.generar_tablas` no aceptaban `perfil` y
  construían su propio universo "sistema", paralelo al `cuprum` propio (mismo `afiliado_id`
  k-ésimo, no la misma persona demográfica). Ambos ya aceptan `perfil` en `main`; ninguno acepta
  además un `universo` inyectado, pero como `generar_universo` es determinista, llamarlos con el
  MISMO `n_afiliados`/`periodo_inicio`/`periodo_fin`/`seed`/`perfil="cuprum"` que el universo
  propio reconstruye el universo IDÉNTICO dentro de 02/03 (verificado con `.equals(...)` en
  `tests/test_segmentacion.py::test_tablas_externas_mismo_universo_que_propio`). El `afiliado_id`
  k-ésimo es ahora la misma persona en los tres — ya no hay features "de segunda mano".

## 3. Features "as-of", corte de referencia y horizonte

No hay un target de horizonte único (no es un modelo de clasificación): las features se construyen
en un `periodo_corte` con la misma disciplina as-of de `CONVENCIONES.md` §3 — el panel laboral/RIM
propio hasta `corte-2`, las tablas operacionales (saldo, fondo, cambios de fondo, uso digital,
contactos, reclamos, aportes voluntarios) hasta `corte` mismo — y la validación de negocio (§6) mira
`corte+1..corte+3/6`, así que sigue habiendo una frontera de leakage que se prueba explícitamente
(`tests/test_segmentacion.py`, alterar el futuro no cambia una sola feature).

`run.py` elige el corte de referencia más reciente que deja, a la vez: 24 meses de historia
(densidad), otro corte 6 meses antes igual de completo (estabilidad temporal, §5) y 12 meses libres
hacia adelante (matriz de transición, §6). Con el rango por defecto (202201-202609) da **corte =
202509**. Toda la cartera entra (todo `tipo_afiliado`, no solo dependientes: la pregunta de negocio
es sobre TODA la cartera de Cuprum).

## 4. Features (`features.py`)

| Familia | Features | Fuente | Regla as-of |
|---|---|---|---|
| Renta y densidad | `log_renta` (RIM en `corte-2`), `densidad_12`, `densidad_24` | Panel propio (`cuprum`) | Hasta `corte-2` |
| Estabilidad laboral | `racha_cotiza_m` (meses consecutivos cotizando), `n_empleadores_12` | Panel propio | Hasta `corte-2` |
| Edad y pensión | `edad`, `meses_hasta_pension` | Afiliados propio (dato calendario, sin desfase) | — |
| Saldo obligatorio | `log_saldo_aprox`, `saldo_renta_ratio` | Panel propio, **fórmula cerrada** (ver §8, no una simulación mes a mes) | Hasta `corte-2` |
| Ahorro voluntario | `log_apv_saldo` (stock), `log_apv_flujo_6m` (flujo 6 m) | `propension_apv.saldos`/`aportes_voluntarios` | Hasta `corte` |
| Fondo | `fondo_ord` (E=0…A=4), `n_cambios_fondo_12` | `churn_traspaso.saldos`/`cambios_fondo` | Hasta `corte` |
| Uso digital | `log_uso_digital_3m` (logins+consultas, 3 m) | `churn_traspaso.uso_digital` | Hasta `corte` |
| Contacto/reclamo | `contactos_12`, `reclamos_12` | `churn_traspaso.contactos`/`reclamos` | Hasta `corte` |
| Antigüedad AFP | `log_antiguedad_afp` | `churn_traspaso.saldos` | Hasta `corte` |
| Tipo de afiliado | `es_independiente`, `es_voluntario`, `es_pensionado_activo` (dummies; DEPENDIENTE es la base) | Afiliados propio | — |

19 features numéricas (`FEATURES_CLUSTERING`) entran al espacio de clustering, reescaladas con
`RobustScaler` (mediana/IQR: robusto a la cola larga de renta/saldo). `sexo`, `region`,
`nivel_educacional`, `rubro`, `fondo`, `tipo_afiliado` quedan para perfilar segmentos (§6) y para el
clasificador explicativo (§5), no para el clustering (evita inflar la dimensión con one-hot de alta
cardinalidad).

## 5. Método

1. **k-means y GMM** (`covariance_type="diag"`), `k` en {5…9}, sobre las 19 features escaladas.
2. **Selección de k**: silueta (muestreada a 8 000 filas si hay más, por costo), BIC (solo GMM) y
   **estabilidad**: ARI de re-muestreo (5 ajustes independientes sobre el 80 % de las filas,
   promedio de ARI por par sobre la intersección — método de Ben-Hur/Elisseeff/Guyon) y ARI entre
   el corte de referencia y el corte 6 meses antes (mismo algoritmo/k, ajustados de forma
   independiente, ARI sobre los afiliados presentes en ambos). El "ganador" es el de mayor silueta
   entre los que cumplen silueta ≥ 0,20, ARI de re-muestreo ≥ 0,60 y ningún segmento < 3 %; si
   ninguno cumple las 3 a la vez, el de mayor silueta igual (marcado `cumple=False`, ver §6/§8).
3. **Fusión de segmentos chicos** (`segmentar.fusionar_segmentos_chicos`, ver §8): si el ganador
   deja algún segmento bajo el 3 %, se fusiona iterativamente con el vecino más cercano (distancia
   entre centroides en el espacio escalado) hasta que todos superen el piso. El modelo entrenado
   (`KMeans`/`GaussianMixture`) no cambia — solo se reetiquetan sus clusters — así que un corte
   futuro se asigna con el mismo modelo (`asignar_fusionado`) y cae en el mismo espacio fusionado.
4. **Clasificador explicativo** (LightGBM multiclase): reproduce el segmento desde las 19 features
   numéricas + 6 categóricas (`tipo_afiliado`, `sexo`, `region`, `nivel_educacional`, `rubro`,
   `fondo`), split aleatorio estratificado 75/25 (no temporal: el "target" es la propia etiqueta de
   clustering de ESTE corte, no un evento futuro, así que no hay riesgo de leakage que exija un
   split por tiempo).
5. **Matriz de transición**: el modelo de referencia (ya fusionado) se usa para ASIGNAR (no
   reajustar) las features de `corte+6` y `corte+12`, y se cruzan las etiquetas por afiliado.

## 6. Resultados (`n_afiliados=20 000`, seed=42, perfil `cuprum`, corte=202509)

**Corrida vigente (post-fix de perfil en 02/03, ver §2)**. Los números de la corrida anterior
(cuando 02/03 aún generaban su propio universo "sistema" paralelo) quedan en el historial de git;
aquí solo el resultado final.

### Selección de k (tabla completa; k-means y GMM, k=5…9)

| Algoritmo | k | Silueta | ARI re-muestreo | ARI temporal (corte-6) | Segmentos ≥ 3 % |
|---|---:|---:|---:|---:|---|
| **kmeans** | **5 (ganador)** | **0,271** | **0,987** | 0,465 | No |
| kmeans | 6 | 0,267 | 0,990 | 0,496 | No |
| kmeans | 7 | 0,261 | 0,965 | 0,446 | No |
| kmeans | 8 | 0,261 | 0,928 | 0,431 | No |
| kmeans | 9 | 0,174 | 0,968 | 0,443 | No |
| gmm | 5–9 | 0,10–0,23 | 0,60–0,85 | 0,23–0,47 | No |

k-means vuelve a ganar en silueta y ARI de re-muestreo en los 5 k probados. Con el universo real
de Cuprum ya compartido por los tres generadores, la silueta sube en todo el rango (antes
0,10-0,24; ahora 0,17-0,27) — la cartera real separa mejor que la mezcla con el universo "sistema"
paralelo. Sigue sin haber un k en [5,9] que deje todos los segmentos ≥ 3 % a la vez, así que se
aplica la misma fusión de segmentos chicos: **k baja de 5 a 4**, silueta final **0,2395** (antes:
k 7→6, silueta 0,2065 — mejora clara).

### Aceptación (post-fusión)

| Criterio | Antes (universo paralelo) | Ahora (universo compartido) | Umbral | Cumple |
|---|---:|---:|---:|---|
| k final | 6 (de 7) | **4 (de 5)** | 5–9 antes de fusión | — |
| Silueta | 0,2065 | **0,2395** | ≥ 0,20 | Sí |
| ARI de re-muestreo | 0,991 | 0,987 | ≥ 0,60 | Sí |
| ARI temporal, recalculado post-fusión | *(no calculado)* | **0,4655** (antes de fusión: 0,4647) | reportado, no gatilla aceptación (ver §8) | — |
| Todos los segmentos ≥ 3 % | mínimo 5,1 % | mínimo **15,0 %** | — | Sí |
| Separación de negocio (razón máx/mín) | 3/3, hasta 16,7× | **3/3**, hasta 30,7× (mín 2,08×) | ≥ 2 de 3 | Sí |
| Exactitud del clasificador | 0,9834 | **0,9812** | ≥ 0,85 | Sí |
| Runtime (20 000 afiliados) | 89,3 s | **90,8 s** | < 5 min | Sí |

Nota del gerente: la columna "Antes" reproduce la entrega anterior de DS-3. La corrida de
control de esa misma versión en `main` (commit 2c35143, `--n-afiliados 20000`) dio k 6→5,
silueta 0,2179, ARI re-muestreo 0,989 y exactitud 0,982: variación menor entre corridas de la
versión con universo paralelo, sin efecto en las conclusiones. La columna "Ahora" fue
reproducida por el gerente (k 5→4, silueta 0,2395, ARI temporal 0,4655, 3/3, exactitud 0,9812).
| Tests | 13, 2,7 s | **14, 3,1 s** | < 2 min | Sí |

### Perfil de segmentos (medianas; `n`/`%` sobre 19 575 afiliados activos al corte)

| Seg. | n (%) | Renta | Densidad 12m | Racha (m) | Edad | Antigüedad AFP (m) | Saldo aprox. | Saldo APV | Fondo típico |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 7 121 (36,4 %) | $1.794.378 | 100 % | 24 | 48 | 52 | $6.027.989 | $0 | C/D (88 %) |
| 1 | 2 940 (15,0 %) | **$3.027.956** | 100 % | 24 | 45 | 69 | **$10.258.425** | **$3.363.102** | C (51 %) |
| 2 | 4 576 (23,4 %) | $1.366.357 | 100 % | 12 | **35** | 97 | $5.276.015 | $0 | **A/B (89 %)** |
| 3 | 4 938 (25,2 %) | **$0** | **17 %** | 0 | 48 | 59 | $2.271.842 | $0 | C/D (80 %) |

### Tasas de negocio por segmento (validación de utilidad, brief §4)

| Seg. | Cese 3 m | Traspaso 6 m | Aporte APV 6 m |
|---:|---:|---:|---:|
| 0 | 5,3 % | 2,6 % | 11,8 % |
| 1 | **3,3 %** (mín.) | **3,1 %** (máx.) | **70,2 %** (máx.) |
| 2 | 6,2 % | 2,1 % | 8,4 % |
| 3 | **11,3 %** (máx.) | 1,5 % (mín.) | 2,3 % (mín.) |
| **razón máx/mín** | **3,38** | **2,08** | **30,69** |

Las 3 tasas separan con razón ≥ 2 (el brief exige al menos 2 de 3): el aporte APV lo domina el
segmento 1 (70,2 % vs. 2,3-11,8 % del resto, razón 30,7×) y el cese lo concentra el segmento 3
(11,3 % vs. 3,3-6,2 %, razón 3,4×). El traspaso separa por poco (razón 2,08×, el segmento premium
1 es el más propenso a irse) — el margen más estrecho de los tres, a diferencia de la corrida
anterior (16,7×): con el universo real, el comportamiento de traspaso de los distintos perfiles de
Cuprum es más parecido entre sí de lo que sugería mezclarlo con el universo "sistema" paralelo.

### Clasificador explicativo (LightGBM, importancias top 10 de 25 features)

| Feature | Ganancia (%) |
|---|---:|
| `log_apv_flujo_6m` | 31,9 |
| `fondo_ord` | 23,0 |
| `densidad_12` | 22,5 |
| `edad` | 7,0 |
| `densidad_24` | 3,1 |
| `log_saldo_aprox` | 3,1 |
| `meses_hasta_pension` | 2,8 |
| `log_renta` | 2,6 |
| `racha_cotiza_m` | 1,8 |
| `n_empleadores_12` | 0,7 |

Exactitud 0,9812 (train 14 681, test 4 894, split aleatorio estratificado). El flujo de APV, el
fondo y la densidad de cotización concentran el 77 % de la ganancia — mucho más concentrado que en
la corrida anterior (donde el uso digital lideraba): con el universo real, el comportamiento de
ahorro/cotización manda por sobre el uso de canales digitales.

### Matriz de transición (fracción de cada segmento de origen que termina en cada destino)

A 6 meses, la diagonal va de 70 % (segmento 1) a 83 % (segmento 2): la mayoría se queda en su
segmento; el segmento 1 (ahorradores premium) es el que más migra hacia el 0 (18 %). A 12 meses la
diagonal baja un poco más (63-82 %). Tablas completas en
`salidas/segmentacion_afiliados/reportes/transicion_{6,12}m.csv`.

## 7. Lectura de negocio: segmento → descripción → acción

| Segmento | % cartera | Descripción | Acción |
|---|---:|---|---|
| **0 · Núcleo estable, ingreso medio** | 36,4 % | El segmento más grande: renta media-alta, cotización continua (racha 24 m), fondo conservador-moderado (C/D 88 %), antigüedad AFP baja-media (4,3 años), **sin APV**. | Oferta de APV (mayor potencial sin explotar por volumen), fidelización estándar, venta cruzada de productos según rubro (salud es el más común aquí). |
| **1 · Ahorradores premium** | 15,0 % | Renta y saldo más altos de la cartera ($3,0 M / $10,3 M), **70,2 % aportó APV en 6 m** (el resto: 2-12 %), pero también la **tasa de traspaso más alta (3,1 %)** y la de cese más baja (3,3 %). | Retención prioritaria de valor (mayor riesgo de fuga en $ administrados), asesoría patrimonial dedicada, vigilar especialmente ante la licitación de stock 2027-2028 (es el segmento de mayor valor en riesgo). |
| **2 · Joven en acumulación** | 23,4 % | El más joven (edad mediana 35), mayor horizonte a la pensión (28 años), fondo A/B en el 89 % (tolerancia al riesgo propia de la etapa), racha de cotización más corta (12 m, algo más de rotación reciente), **sin APV**. | Educación financiera + oferta de APV de entrada, fidelización de largo plazo, canal digital-first. |
| **3 · Intermitentes / riesgo de inactividad** | 25,2 % | Renta y densidad actuales muy bajas (mediana de renta $0, densidad 17 %), sin racha de cotización vigente, **la tasa de cese a 3 m más alta (11,3 %)** y la de aporte APV más baja (2,3 %). | Cobranza/reactivación proactiva, prioridad de retención de cotización, revisar vías de aporte alternativas (independiente/voluntario) y elegibilidad AFC. |

## 8. Decisiones tomadas, supuestos y limitaciones

* **Perfil `cuprum` en 02/03: RESUELTO** (ver §2). Ya no hay features "de segunda mano": las 19
  features de clustering (núcleo + externas) describen a la misma persona en las tres fuentes.
* **`saldo_aprox` es una fórmula cerrada**, no una simulación mes a mes: valor futuro de una
  anualidad (`TASA_COTIZACION_OBLIGATORIA × renta media histórica`, capitalizada
  `TASA_MENSUAL_CAPITALIZACION=0,40 %` mensual real por los meses efectivamente cotizados desde
  `periodo_inicio` hasta `corte-2`). Da el orden de magnitud relativo entre afiliados (suficiente
  para segmentar), no un saldo exacto — un proyecto de capitalización dedicado lo simularía mes a
  mes con rentabilidad variable por fondo, como hacen 02/03 con SU propio universo. Esta
  limitación NO se cierra con el fix de perfil (es una decisión de diseño de este proyecto, no una
  consecuencia de usar universos distintos).
* **Silueta post-fusión (0,2395) supera con más margen el umbral (0,20)** que antes del fix de
  perfil (0,2065): con el universo real de Cuprum compartido, la cartera separa mejor en el
  espacio de 19 features que al mezclarla con el ruido de un universo "sistema" paralelo — aun así
  no es una silueta "alta" en términos absolutos (clustering de comportamiento con features
  heterogéneas y débilmente correlacionadas rara vez da siluetas > 0,3-0,4); los 3 chequeos de
  negocio del §6 son los que confirman que los segmentos "significan algo" para el negocio.
* **ARI temporal (6 meses), ahora recalculado sobre la partición FINAL (post-fusión)**: 0,4655
  (antes de fusión, mismo k=5: 0,4647 — la fusión casi no cambia el número, coherente con que solo
  reetiqueta). Sigue siendo más bajo que el ARI de re-muestreo (0,987): el criterio del brief ("ARI
  de estabilidad ≥ 0,60") se satisface con el de **re-muestreo** (Ben-Hur/Elisseeff/Guyon: "¿esta
  metodología encuentra la misma estructura si la vuelvo a estimar?"); el ARI entre cortes
  separados 6 meses mezcla eso con una deriva REAL del negocio (alguien deja de cotizar, empieza a
  aportar APV, cambia de fondo en esos 6 meses) y por eso es, y debería ser, más bajo — no se
  fuerza al umbral, se reporta tal cual.
* **Fusión de segmentos chicos** (§5/§6): sigue siendo necesaria incluso con el universo real —
  ningún (algoritmo, k) en [5,9] deja los 5-9 segmentos todos ≥ 3 % a la vez (aunque, con el
  universo real, el segmento chico que sobra es bastante menos extremo que antes: el tamaño mínimo
  post-fusión subió de 5,1 % a 15,0 %). Fusionarlo con su vecino más parecido sigue siendo más
  razonable que forzar una partición peor solo para que un k marginal cumpla el piso.
* **Todos los tipos de afiliado entran a la segmentación** (a diferencia del proyecto 04, que
  excluye independientes/voluntarios de sus modelos de riesgo): la pregunta de negocio aquí es
  sobre TODA la cartera; el segmento 3 (mayor tasa de cese) sigue teniendo una mezcla algo distinta
  de tipos de afiliado que el resto (11,6 % independientes, 7,0 % voluntarios, 2,3 % pensionados
  activos, vs. 0-0,6 % en los segmentos "activos").
* **Clasificador con exactitud 0,98**: reproduce casi perfectamente el clustering porque ve las
  MISMAS features con las que se ajustó (no es una señal de que el negocio sea trivialmente
  predecible con datos externos — es la tarea que pide el brief: explicar y asignar afiliados
  nuevos sin reclusterizar).
* **`tests/` no valida los umbrales de aceptación del brief** (silueta ≥ 0,20, etc.): esos números
  se validan en `run.py` a escala de producción (20 000 afiliados, reportados en este README); el
  suite de tests usa un universo chico (600 afiliados, unos segundos) y prueba forma del dataset,
  leakage y que las funciones de `segmentar.py`/`evaluate.py` se comporten correctamente en rango
  — sería una coincidencia, no una garantía, que un universo de 600 afiliados replique los mismos
  umbrales de un negocio real.

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.segmentacion_afiliados.run --n-afiliados 20000
python -m pytest afp_lab/segmentacion_afiliados/tests -q
```

* `run.py` (`--n-afiliados 20000`, perfil `cuprum` por defecto): **90,8 s** en la corrida de
  referencia de este informe (universo propio 2,3 s + tablas externas 7,4 s + features 1,3 s +
  selección de k 72,2 s + clasificador 1,3 s + resto < 1 s). Muy por debajo del límite de 5
  minutos.
* `pytest afp_lab/segmentacion_afiliados/tests`: **14 tests, ~3,1 s**. Muy por debajo del límite de
  2 minutos.
* `--salida` (por defecto `salidas/segmentacion_afiliados/`, en `.gitignore`):
  `dataset_segmentos.parquet` (19 575 filas con el segmento asignado) y `reportes/*.csv`
  (selección de k, perfil de segmentos, tasas de negocio, importancias del clasificador, matrices
  de transición a 6 y 12 meses).
* Semilla fija (`seed=42` por defecto) y generación determinista en las 3 fuentes (universo propio,
  `churn_traspaso`, `propension_apv`).

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/` / otros proyectos** (no se tocó nada fuera de esta carpeta):

* `churn_traspaso.generar_tablas`/`propension_apv.generar_tablas` ya aceptan `perfil` (resuelto,
  §2/§8). Sigue pendiente, como mejora ideal (no bloqueante): que acepten además un
  `universo: dict | None = None` inyectado, para no recalcular el universo común 3 veces (hoy
  ~2-3 s cada una a 20 000 afiliados — barato, pero innecesario si ya se tiene el universo a mano).
* Una utilidad común de "fusión de clusters chicos por centroide" (`segmentar.
  fusionar_segmentos_chicos`) podría vivir en `afp_lab.common` si un futuro proyecto de clustering
  la necesita — hoy es propia de este proyecto.

**Próximos pasos** (no implementados en esta ronda):

* Simular `saldo_aprox` mes a mes (como hacen 02/03) en vez de la fórmula cerrada, si se necesita
  precisión de monto y no solo de orden relativo entre afiliados.
* Explorar features derivadas específicamente para separar mejor el segmento 3 (intermitentes) de
  los segmentos "activos" con fondo/edad similares — hoy se distingue sobre todo por renta/densidad
  actuales, que son ruidosas mes a mes; una historia de 24 meses (no solo el snapshot en `corte-2`)
  podría dar una silueta más alta.
* El traspaso ahora separa por el margen más estrecho de las 3 tasas (2,08×, §6): vale la pena
  revisar si una feature adicional (p. ej. brecha de comisión percibida, o contacto reciente de la
  competencia) separaría mejor a los segmentos de mayor propensión a irse.
