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
* **Limitación documentada (no evitable sin tocar 02/03, fuera de alcance de esta ronda)**:
  `churn_traspaso.generar_tablas` y `propension_apv.generar_tablas` llaman a `generar_universo(...,
  seed=seed)` **sin** `perfil` (siempre "sistema"), mientras este proyecto usa `perfil="cuprum"`. El
  `afiliado_id` k-ésimo existe en los tres universos, pero en 02/03 corresponde a una simulación
  PARALELA con perfil "sistema" (otros sorteos de edad, renta, rubro, etc. desde el mismo punto del
  generador de números aleatorios), no a la misma persona con exactitud demográfica. Por eso:
  * Las features **núcleo** del segmento (renta, densidad, estabilidad, edad, saldo aproximado,
    §4) son 100 % del universo `cuprum` propio — internamente consistentes entre sí.
  * Las features y la validación de negocio que usan 02/03 (saldo/fondo, cambios de fondo, uso
    digital, contactos, reclamos, ahorro voluntario, tasa de traspaso y de aporte APV) heredan esta
    limitación: describen "el afiliado con este número de índice" en una cartera paralela
    equivalente en tamaño y estructura, no exactamente a la misma persona. Se documenta y se acota
    (no se puede evitar sin modificar 02/03 — ver "Propuestas para `common/`", §10).
  * La tasa de **cese** (única sacada 100 % del panel propio) no tiene esta limitación.

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
| Ahorro voluntario | `log_apv_saldo` (stock), `log_apv_flujo_6m` (flujo 6 m) | `propension_apv.saldos`/`aportes_voluntarios` (⚠ perfil, §2) | Hasta `corte` |
| Fondo | `fondo_ord` (E=0…A=4), `n_cambios_fondo_12` | `churn_traspaso.saldos`/`cambios_fondo` (⚠ perfil) | Hasta `corte` |
| Uso digital | `log_uso_digital_3m` (logins+consultas, 3 m) | `churn_traspaso.uso_digital` (⚠ perfil) | Hasta `corte` |
| Contacto/reclamo | `contactos_12`, `reclamos_12` | `churn_traspaso.contactos`/`reclamos` (⚠ perfil) | Hasta `corte` |
| Antigüedad AFP | `log_antiguedad_afp` | `churn_traspaso.saldos` (⚠ perfil) | Hasta `corte` |
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

### Selección de k (tabla completa; k-means y GMM, k=5…9)

| Algoritmo | k | Silueta | ARI re-muestreo | ARI temporal (corte-6) | Segmentos ≥ 3 % |
|---|---:|---:|---:|---:|---|
| kmeans | 5 | 0,224 | 0,974 | 0,704 | No |
| kmeans | 6 | 0,236 | 0,980 | 0,602 | No |
| **kmeans** | **7 (ganador)** | **0,240** | **0,991** | 0,576 | No |
| kmeans | 8 | 0,225 | 0,973 | 0,544 | No |
| kmeans | 9 | 0,220 | 0,781 | 0,515 | No |
| gmm | 5–9 | 0,10–0,18 | 0,68–1,00 | 0,25–0,51 | No |

k-means gana en silueta y en ARI de re-muestreo en los 5 valores de k probados; GMM (covarianza
diagonal) da siluetas más bajas y BIC que no converge a un óptimo claro en el rango — se reporta
igual por completitud, pero no fue elegido en ningún k. **Ninguno de los 10 (algoritmo, k)
evaluados deja los 5-9 segmentos todos sobre el piso de 3 %**: siempre queda un sub-grupo chico y
genuino (afiliados sin RIM ni empleador conocido — ver segmento 3 más abajo) por debajo. Se aplicó
la fusión del punto 3: **k baja de 7 a 6**, con silueta final **0,2065** (por encima del umbral de
0,20, pero por poco — ver §8).

### Aceptación (post-fusión)

| Criterio | Valor | Umbral | Cumple |
|---|---:|---:|---|
| Silueta | 0,2065 | ≥ 0,20 | Sí (margen estrecho, ver §8) |
| ARI de re-muestreo | 0,991 (del k=7 ganador; la fusión no reajusta el modelo) | ≥ 0,60 | Sí |
| Todos los segmentos ≥ 3 % | mínimo 5,1 % (segmento 0) | — | Sí (tras la fusión) |
| Separación de negocio (≥ 2 de 3 tasas, razón máx/mín ≥ 2) | 3 de 3 (ver tabla abajo) | ≥ 2 de 3 | Sí |
| Exactitud del clasificador | 0,9834 | ≥ 0,85 | Sí |
| Runtime (20 000 afiliados) | 89,3 s | < 5 min | Sí |
| Tests | 13, 2,7 s | < 2 min | Sí |

### Perfil de segmentos (medianas; `n`/`%` sobre 19 575 afiliados activos al corte)

| Seg. | n (%) | Renta | Densidad 12m | Racha (m) | Edad | Antigüedad AFP (m) | Saldo aprox. | Saldo APV | Fondo típico |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 004 (5,1 %) | $1.362.412 | 100 % | 7 | 45 | **0** | $2.626.315 | $0 | C (86 %) |
| 1 | 1 608 (8,2 %) | $1.696.387 | 100 % | 17 | 45 | 122 | $6.003.261 | **$2.145.393** | C/B/A |
| 2 | 4 732 (24,2 %) | $1.573.192 | 100 % | 18 | **36** | 141 | $5.793.874 | $0 | **A/B (92 %)** |
| 3 | 3 846 (19,6 %) | **$0** | **17 %** | 0 | 49 | 127 | $2.183.752 | $0 | C/D |
| 4 | 2 360 (12,1 %) | $1.583.347 | 100 % | 19 | 42 | 78 | $5.902.848 | $0 | C/B |
| 5 | 6 025 (30,8 %) | **$1.978.281** | 100 % | 24 | 49 | 137 | **$7.001.428** | $0 | **C/D/E (95 %)** |

### Tasas de negocio por segmento (validación de utilidad, brief §4)

| Seg. | Cese 3 m | Traspaso 6 m | Aporte APV 6 m |
|---:|---:|---:|---:|
| 0 | 7,2 % | 3,5 % | 4,5 % |
| 1 | 5,2 % | 2,6 % | **65,9 %** |
| 2 | 5,9 % | 2,6 % | 4,3 % |
| 3 | **10,8 %** | 1,7 % | 4,0 % |
| 4 | 4,4 % | **0,2 %** | 6,4 % |
| 5 | 4,9 % | 1,8 % | 4,8 % |
| **razón máx/mín** | **2,42** | **16,66** | **16,36** |

Las 3 tasas separan con razón ≥ 2 entre el segmento de mayor y menor valor (el brief exige al menos
2 de 3): el aporte APV lo domina casi por completo el segmento 1 (65,9 % vs. 4,0-6,4 % del resto) y
el traspaso lo evita casi por completo el segmento 4 (0,2 % vs. 1,7-3,5 %) — dos separaciones muy
por encima del mínimo, más una separación moderada (2,42×) en cese.

### Clasificador explicativo (LightGBM, importancias top 10 de 25 features)

| Feature | Ganancia (%) |
|---|---:|
| `log_uso_digital_3m` | 19,2 |
| `log_apv_flujo_6m` | 18,1 |
| `fondo_ord` | 17,1 |
| `densidad_12` | 16,5 |
| `log_antiguedad_afp` | 11,1 |
| `edad` | 4,6 |
| `meses_hasta_pension` | 4,0 |
| `log_saldo_aprox` | 2,7 |
| `densidad_24` | 2,7 |
| `log_renta` | 2,0 |

Exactitud 0,9834 (train 14 681, test 4 894, split aleatorio estratificado): el clasificador
reproduce casi perfectamente el clustering, lo esperable cuando ve las mismas features con las que
se ajustó (sirve para explicar en reglas y para asignar afiliados nuevos sin reclusterizar, no para
"predecir" nada nuevo). El uso digital y el flujo de APV son, con margen, las señales más
discriminantes — más que la renta misma.

### Matriz de transición (fracción de cada segmento de origen que termina en cada destino)

A 6 meses, la diagonal va de 66 % (segmento 1) a 85 % (segmento 5): la mayoría se queda en su
segmento, con fugas concentradas hacia el segmento 1 (ahorradores) desde casi todos los demás — es
de esperar dado que APV es la variable más volátil mes a mes. A 12 meses la diagonal baja un poco
más (58-80 %), consistente con más tiempo para migrar. Tablas completas en
`salidas/segmentacion_afiliados/reportes/transicion_{6,12}m.csv`.

## 7. Lectura de negocio: segmento → descripción → acción

| Segmento | % cartera | Descripción | Acción |
|---|---:|---|---|
| **0 · Recién llegados** | 5,1 % | Antigüedad en la AFP ≈ 0, racha de cotización corta (7 m), renta media, **sin uso digital ni contactos, sin APV**. | Onboarding activo (registro app, bienvenida), evaluar match de APVC del empleador, primer contacto comercial temprano. |
| **1 · Ahorradores voluntarios** | 8,2 % | Renta alta, ~10 años en la AFP, saldo APV mediano $2,1 M, **65,9 % aportó APV en 6 m** (el resto del sistema: 4-6 %), uso digital y contactos moderados. | Retención personalizada (mayor riesgo de fuga de valor si se van), venta cruzada de Régimen B / asesoría patrimonial, prioridad para ejecutivo propio. |
| **2 · Núcleo joven en acumulación** | 24,2 % | El segmento activo más numeroso: edad mediana 36, ~26 años a la pensión, fondo A/B en el 92 % (tolerancia al riesgo propia de la etapa), mayor antigüedad AFP entre los activos, **sin APV**. | Educación financiera + oferta de APV (potencial sin explotar), fidelización de largo plazo, canal digital-first. |
| **3 · Intermitentes / riesgo de inactividad** | 19,6 % | Renta y densidad actuales muy bajas (mediana de renta $0, densidad 17 %), mayor mezcla de independientes/voluntarios/pensionados que el resto, **la tasa de cese a 3 m más alta (10,8 %)**. | Cobranza/reactivación proactiva, prioridad de retención de cotización, revisar elegibilidad AFC y vías de aporte alternativas (independiente/voluntario). |
| **4 · Estables silenciosos** | 12,1 % | Cotización estable (densidad 100 %, racha 19 m), tenencia media (6,5 años), **la tasa de traspaso a 6 m más baja del sistema (0,2 %)**, pero sin uso digital ni contactos recientes. | No está en riesgo de fuga — oportunidad de venta cruzada sin explotar: activación digital y oferta de APV vía transferencia (no tienen descuento por planilla activo). |
| **5 · Preretiro de alto valor** | 30,8 % | El segmento más grande: mayor renta ($2,0 M) y mayor saldo aproximado ($7,0 M), edad mediana 49, ~13 años a la pensión, fondo conservador (C/D/E en el 95 %). | Asesoría previsional (glide path hacia fondos generacionales, vigentes desde 2027), foco de retención ante la licitación de stock de 2027-2028 (afecta a los no pensionados, y este es el segmento de mayor valor en riesgo), revisión de comisión/beneficios VIP. |

## 8. Decisiones tomadas, supuestos y limitaciones

* **Perfil `cuprum` vs. tablas 02/03 en perfil `sistema`**: ver §2. Es la limitación más importante
  del proyecto; no se puede cerrar sin modificar `churn_traspaso`/`propension_apv` (fuera de
  alcance — ver propuesta en §10).
* **`saldo_aprox` es una fórmula cerrada**, no una simulación mes a mes: valor futuro de una
  anualidad (`TASA_COTIZACION_OBLIGATORIA × renta media histórica`, capitalizada
  `TASA_MENSUAL_CAPITALIZACION=0,40 %` mensual real por los meses efectivamente cotizados desde
  `periodo_inicio` hasta `corte-2`). Da el orden de magnitud relativo entre afiliados (suficiente
  para segmentar), no un saldo exacto — un proyecto de capitalización dedicado lo simularía mes a
  mes con rentabilidad variable por fondo, como hacen 02/03 con SU propio universo.
* **Silueta post-fusión (0,2065) pasa el umbral (0,20) por un margen estrecho**: con 19 features
  bastante heterogéneas (renta, edad, fondo, uso digital, ahorro voluntario) es esperable que los
  segmentos no queden perfectamente esféricos/separados — una silueta de 0,20-0,25 es normal para
  clustering de comportamiento con muchas dimensiones débilmente correlacionadas entre sí (no es
  un artefacto: los 3 chequeos de negocio del §6 sí separan con margen amplio, así que los
  segmentos "significan algo" aunque geométricamente no estén perfectamente aislados).
* **ARI temporal (6 meses) más modesto que el ARI de re-muestreo** (0,58 del ganador k=7 vs. 0,99):
  el criterio de aceptación del brief ("ARI de estabilidad ≥ 0,60") se satisface con el ARI de
  **re-muestreo** (el estándar de Ben-Hur/Elisseeff/Guyon para "¿esta metodología encuentra la
  misma estructura si la vuelvo a estimar?"); el ARI **entre cortes separados 6 meses** es más
  bajo porque, a diferencia del re-muestreo (misma población, mismo corte), ahí se suma un
  ingrediente real: el comportamiento de las personas efectivamente cambia en 6 meses (alguien deja
  de cotizar, empieza a aportar APV, cambia de fondo) — parte de esa caída de ARI es deriva
  genuina del negocio, no solo ruido de estimación. Se reporta igual, sin forzarlo al umbral, para
  no esconder que la partición es más estable "estimándola de nuevo" que "6 meses después". No se
  recalculó tras la fusión de segmentos chicos (la fusión solo reetiqueta, no reajusta el modelo).
* **Fusión de segmentos chicos** (§5/§6): ninguno de los 10 (algoritmo, k) probados en [5,9] deja
  todos los segmentos ≥ 3 % a la vez — el segmento de afiliados sin RIM ni empleador conocido
  (bajísima densidad, la semilla del actual segmento 3) aparece como un grupo chico y genuino en
  casi cualquier k. Fusionarlo con su vecino más parecido (misma práctica que un analista humano
  aplicaría) es más razonable que forzar una partición peor solo para que un k marginal cumpla el
  piso, o que reportar un segmento que no cumple el criterio de negocio sin decirlo.
* **Todos los tipos de afiliado entran a la segmentación** (a diferencia del proyecto 04, que
  excluye independientes/voluntarios de sus modelos de riesgo): la pregunta de negocio aquí es
  sobre TODA la cartera, y de hecho el segmento 3 (mayor tasa de cese) tiene una mezcla
  notablemente distinta de tipos de afiliado que el resto (12,3 % independientes, 7,6 %
  voluntarios, 2,8 % pensionados activos, vs. 0-1 % en los segmentos "activos").
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

* `run.py` (`--n-afiliados 20000`, perfil `cuprum` por defecto): **89,3 s** en la corrida de
  referencia de este informe (universo propio 2,2 s + tablas externas 7,2 s + features 1,4 s +
  selección de k 71,6 s + clasificador 1,8 s + resto < 1 s). Muy por debajo del límite de 5
  minutos.
* `pytest afp_lab/segmentacion_afiliados/tests`: **13 tests, ~2,7 s**. Muy por debajo del límite de
  2 minutos.
* `--salida` (por defecto `salidas/segmentacion_afiliados/`, en `.gitignore`):
  `dataset_segmentos.parquet` (19 575 filas con el segmento asignado) y `reportes/*.csv`
  (selección de k, perfil de segmentos, tasas de negocio, importancias del clasificador, matrices
  de transición a 6 y 12 meses).
* Semilla fija (`seed=42` por defecto) y generación determinista en las 3 fuentes (universo propio,
  `churn_traspaso`, `propension_apv`).

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/` / otros proyectos** (no se tocó nada fuera de esta carpeta):

* La limitación de perfil (§2/§8) se cerraría si `churn_traspaso.generar_tablas` y
  `propension_apv.generar_tablas` aceptaran un `universo: dict | None = None` inyectado (o un
  parámetro `perfil`), usándolo en vez de llamar a `generar_universo` por su cuenta cuando se
  provee. Es un cambio pequeño y aditivo (parámetro opcional, comportamiento por defecto
  intacto) que beneficiaría a cualquier proyecto futuro que necesite combinar el perfil `cuprum`
  con las tablas de 02/03 sobre el MISMO universo.
* Una utilidad común de "fusión de clusters chicos por centroide" (`segmentar.
  fusionar_segmentos_chicos`) podría vivir en `afp_lab.common` si un futuro proyecto de clustering
  la necesita — hoy es propia de este proyecto.

**Próximos pasos** (no implementados en esta ronda):

* Simular `saldo_aprox` mes a mes (como hacen 02/03) en vez de la fórmula cerrada, si se necesita
  precisión de monto y no solo de orden relativo entre afiliados.
* Recalcular el ARI temporal DESPUÉS de la fusión de segmentos chicos (hoy se reporta el del k=7
  pre-fusión) para tener el número exactamente correspondiente a la partición final de 6 segmentos.
* Explorar features derivadas específicamente para separar mejor el segmento 3 (intermitentes) de
  los segmentos "activos" con fondo/edad similares — hoy se distingue sobre todo por renta/densidad
  actuales, que son ruidosas mes a mes; una historia de 24 meses (no solo el snapshot en `corte-2`)
  podría dar una silueta más alta.
* Si se resuelve la propuesta de `common/` de arriba, repetir la validación de negocio (traspaso,
  APV) sobre el universo `cuprum` real (no el paralelo "sistema") y comparar si las razones
  máx/mín cambian.
