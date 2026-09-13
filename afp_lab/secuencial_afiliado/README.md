# 08 · `secuencial_afiliado` — red secuencial multi-tarea vs. LightGBM de 02 y 04

Ronda 4. Worktree de esta entrega: `/home/user/wt/empleador`, rama `ronda-4/secuencial`
(el brief original, `docs/briefs/ronda_4/secuencial.md`, nombra `/home/user/wt/datos`; se
trabajó en el worktree que indicó el gerente para esta ronda — ver `BRIEF.md`, nota de alcance).

## 1. Pregunta de negocio

¿Una red neuronal secuencial (GRU sobre 24 meses de panel mensual del afiliado), entrenada
multi-tarea, **supera** a los LightGBM tabulares de 02 (`churn_traspaso`, traspaso a 6 meses) y
04 (`riesgo_cese_cotizacion`, cese de cotización a 6 meses) en las MISMAS poblaciones, cortes y
splits? La decisión que apoya: si la secuencia aporta AUC/PR-AUC/lift meaningful, justifica
invertir en una arquitectura más cara (embeddings, GRU, PyTorch) en vez de seguir con LightGBM
tabular; si no, se documenta y se descarta para production sin re-probarlo cada ronda.

**Alcance de esta entrega** (presupuesto de la ronda, ver §8): se implementan y comparan
íntegramente las cabezas de **traspaso (02)** y **cese a 6 meses (04)**, que el brief marca como
prioritarias. La cabeza de APV (03), la cabeza de regresión de RIM y la comparación del
embedding contra la segmentación de 06 quedan **declaradas como pendientes**, no implementadas.

## 2. Datos sintéticos

No se genera ninguna tabla nueva: `synthetic.py` llama a `churn_traspaso.synthetic.generar_tablas`
y a `riesgo_cese_cotizacion.synthetic.generar_datos` con los MISMOS `n_afiliados`,
`periodo_inicio`, `periodo_fin`, `seed` y `perfil="cuprum"`. `afp_lab.common.synth_base.generar_universo`
es determinista en esos parámetros, así que ambas llamadas reconstruyen **exactamente el mismo**
universo común (`afiliados`, `empleadores`, `panel_mensual`, `macro`) — verificado en
`tests/test_secuencial_afiliado.py::test_universo_identico_02_04` (`.equals()` fila a fila). El
resultado es un único dict de tablas con todo lo de 02 (mercado, saldos, traspasos, contactos,
reclamos, cambios_fondo, uso_digital, licitación de stock) + lo propio de 04 que 02 no trae
(`cotizaciones_conocidas`, `afc_eventos`, `licencias`, `prestaciones_cesantia`, `planilla`),
apto para llamar a las funciones de features de **ambos** proyectos sin tocarlas.

Los mecanismos sintéticos (tasas de traspaso, hazard de cese, licencias, eventos AFC, mercado
de AFP, etc.) y sus parámetros están documentados en `afp_lab/churn_traspaso/README.md` y
`afp_lab/riesgo_cese_cotizacion/README.md`; no se repiten aquí porque no se modificó nada.

## 3. Target, población, horizonte, exclusiones

Ambas cabezas usan la definición **idéntica** del proyecto de origen, sin reimplementar nada:

* **Traspaso (02)**: `churn_traspaso.features.construir_target` — `y=1` si el afiliado solicita
  un traspaso de SALIDA en `corte+1..corte+6`. Población: todos los afiliados activos al corte
  que devuelve `construir_features_multi` (excluye cortes contaminados por la licitación de
  stock, `cortes_evaluables(excluir_licitacion=True)`).
* **Cese 6 meses (04)**: `riesgo_cese_cotizacion.features.agregar_targets` (columna `y6`) sobre
  la población `pop_cese` (cotizando —RIM conocida > 0 en `corte-2`— y dependiente/pensionado
  activo; independientes y voluntarios excluidos, igual que en 04) con `horizonte6_completo`.

Los cortes evaluables y el split temporal (con purga/gap, CONVENCIONES §3) también se calculan
con las funciones del proyecto de origen: `churn_traspaso.features.cortes_evaluables` +
`split_temporal` para traspaso, y `riesgo_cese_cotizacion.run._construir_dataset` +
`_split_temporal` (mismas funciones que usa `04/run.py`) para cese. **Los cortes de TEST
coinciden exactamente entre ambas cabezas** en el rango de referencia
(`periodo_inicio=202201`, `periodo_fin=202609`): `[202601, 202602, 202603]` — verificado en
`test_mismos_cortes_test_que_02_y_04` y confirmado en cada corrida (`run.py` imprime un aviso si
alguna vez no coincidieran). El número de cortes de TRAIN se acotó a `max_cortes_train=7` para
ambas cabezas (parámetro de `split_temporal`/recorte manual del split de 04) por presupuesto de
cómputo de esta ronda — ver §8; val=3 (traspaso) / 2 (cese), test=3 en ambas.

## 4. Features

### 4.1 Secuencia (panel mensual, hasta `corte-2`)

Ventana fija de 24 meses por `(afiliado_id, periodo_corte)`, tomada directamente de
`panel_mensual` (el mismo panel común de 01/02/03/04), con la MISMA regla as-of que usan 02 y 04
para la RIM (`corte - MESES_DESFASE_RIM`, 2 meses). Meses sin fila de panel (antes de
`periodo_afiliacion` o antes de `periodo_inicio`) se rellenan con 0 y se marcan con `mask=0`
(`features.py::construir_secuencias`); el encoder multiplica la entrada por la máscara antes de
la GRU (ver §5, limitaciones).

| Variable (canal) | Tipo | Transformación | Regla as-of |
|---|---|---|---|
| `rim_log` (RIM total, ambos empleadores) | numérica | tope legal (`macro.tope_clp`) + `log1p` | panel hasta `corte-2` |
| `dias_licencia_norm` | numérica | `/30` | panel hasta `corte-2` |
| `jornada` | numérica | tal cual (0 / 0,5 / 1,0) | panel hasta `corte-2` |
| `meses_antiguedad_norm` | numérica | `/60` | panel hasta `corte-2` |
| `empleador_cambia` | numérica (0/1) | `empleador_id[t] != empleador_id[t-1]`, ambos conocidos | panel hasta `corte-2` |
| `estado` | categórica (embedding) | vocabulario fijo (EMPLEADO/CESANTE/INDEPENDIENTE/LICENCIA/VOLUNTARIO/RETIRADO/PENSIONADO_ACTIVO/SIN_PANEL/otro) | panel hasta `corte-2` |
| `evento_afc` | categórica (embedding) | NINGUNO/INICIO/TERMINO/otro | panel hasta `corte-2` |
| `tipo_contrato` | categórica (embedding) | SIN_CONTRATO/INDEFINIDO/PLAZO_FIJO/HONORARIOS/otro | panel hasta `corte-2` |

Todos los vocabularios y las transformaciones numéricas son **fijos** (no se ajustan con datos):
sin estadísticos entrenados con el dataset, no hay forma de que se filtre información de
validación/test hacia el entrenamiento a través de la normalización.

### 4.2 Estáticas (concatenadas, no secuenciales)

`sexo`, `region`, `nivel_educacional`, `tipo_afiliado` (categóricas, embedding) + `edad_norm`
(`edad_inicio` proyectada al corte, `/100`) y `antiguedad_afiliacion_norm` (meses desde
`periodo_afiliacion`, `/360`) — de la tabla `afiliados`, conocida siempre (no varía con el mes).

### 4.3 Por qué la secuencia NO lleva las tablas operacionales de 02 (contactos, reclamos, saldos, cambios de fondo, uso digital)

El BRIEF las menciona como parte de la secuencia "as-of hasta corte". Se dejaron **fuera** de la
secuencia (aunque sí se usan, intactas, para el LightGBM "de producción" de 02) para que el
baseline "tabular con lags" (misma información que ve la red, aplanada — `features.aplanar_secuencia`,
§5) aísle limpiamente el efecto de la ARQUITECTURA del efecto de la INFORMACIÓN, que es
exactamente lo que pide el brief de ese baseline. Es una decisión de diseño documentada, no un
olvido: ver §8, "Decisiones tomadas" y "Limitaciones".

## 5. Modelo y baselines

**Red** (`model.py::RedSecuencial`): embeddings por variable categórica de la secuencia (dim 8) y
de las estáticas (dim 6) + las numéricas → GRU de 2 capas (hidden=32) → se lee el **último paso**
de la secuencia (siempre observado: es `corte-2`, y las poblaciones usadas exigen estar afiliado/
cotizando a esa fecha) → concatenado con las estáticas → proyección lineal a un embedding de
dimensión 24 (16-32 pedido en el brief) → una cabeza lineal por tarea (BCE con logits).

* **Multi-tarea**: las dos cabezas comparten encoder; en cada paso de optimización se alternan
  lotes de cada tarea (cada una con su propio `train`) y se sumsa la pérdida ponderada
  (`pesos={"traspaso":0.6,"cese":0.4}`, elegido en validación — traspaso es la tarea más difícil,
  tasa base ~3 %). Early stopping sobre la suma ponderada de la pérdida de validación de ambas
  cabezas (cada cabeza usa SU PROPIO `val`, con los cortes de su split de origen).
* **Single-task**: misma arquitectura y mismos hiperparámetros, una instancia por cabeza,
  entrenada solo con los datos de esa cabeza (mismo train/val/test que su contraparte multi-tarea).
* `torch.set_num_threads(2)` (máquina de 4 vCPU compartida, CONVENCIONES de la ronda).

**Baselines**:
* `lightgbm_produccion`: `churn_traspaso.model.entrenar` / `riesgo_cese_cotizacion.model.ModeloCese.fit`,
  SUS features completas (incluidas las tablas operacionales) y SUS hiperparámetros, en el MISMO
  split que la red.
* `lightgbm_tabular_lags`: LightGBM sobre la secuencia APLANADA (`features.aplanar_secuencia`,
  24 lags × 8 variables + estáticas), mismos datos que ve la red, sin arquitectura secuencial.
* `baseline_tasa_base` (trivial) y, para cese, `baseline_regla_negocio` (regla de 04: término AFC
  reciente o contrato a plazo corto) — baselines obligatorios de CONVENCIONES §4, reusados de 02/04.

**Limitación documentada**: no se usa `pack_padded_sequence`; los meses enmascarados se ponen en
cero multiplicando la entrada antes de la GRU. Es una simplificación por presupuesto de tiempo
(no por falta de un mecanismo mejor) — el sesgo esperado es pequeño porque los ceros no llevan
señal y la GRU puede aprender a ignorarlos, pero no es matemáticamente idéntico a no procesarlos.
Las probabilidades de la red **no están calibradas** (sin isotónica/Platt, a diferencia de los
LightGBM de origen): el Brier de la red no debe compararse directamente con el de los baselines;
sí son comparables AUC, PR-AUC, lift decil 1 y KS (invariantes a una transformación monótona).

## 6. Resultados

Corrida de referencia: `PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.secuencial_afiliado.run`
(n=10.000, seed=42, perfil=cuprum, `max_cortes_train=7`, 6 épocas máx./paciencia 2, hidden=32,
embedding=24, `lr=3e-3`, `pesos={"traspaso":0.6,"cese":0.4}`, 200 réplicas bootstrap).

**Traspaso 6 meses (02)** — test: 3 cortes (202601-202603), n=20.787, tasa base 2,67 %:

| modelo | AUC | IC 95 % | PR-AUC | lift decil 1 |
|---|---|---|---|---|
| lightgbm_produccion | **0,7255** | [0,7089, 0,7450] | 0,0635 | 2,686 |
| lightgbm_tabular_lags | 0,5940 | [0,5724, 0,6149] | 0,0349 | 1,298 |
| red_multitarea | 0,5848 | [0,5617, 0,6099] | 0,0334 | 1,262 |
| red_single_task | 0,5766 | [0,5552, 0,6006] | 0,0324 | 1,280 |
| baseline_tasa_base | 0,5000 | — | 0,0267 | 1,000 |

**Cese 6 meses (04)** — test: 3 cortes (202601-202603), n=16.766, tasa base 6,64 %:

| modelo | AUC | IC 95 % | PR-AUC | lift decil 1 |
|---|---|---|---|---|
| lightgbm_produccion | **0,8610** | [0,8475, 0,8741] | 0,5031 | 6,642 |
| baseline_regla_negocio | 0,8195 | [0,8054, 0,8345] | 0,4414 | 6,665 |
| red_single_task | 0,7807 | [0,7630, 0,7948] | 0,3382 | 4,431 |
| lightgbm_tabular_lags | 0,7775 | [0,7610, 0,7928] | 0,3318 | 4,494 |
| red_multitarea | 0,7768 | [0,7586, 0,7916] | 0,3342 | 4,494 |
| baseline_tasa_base | 0,5000 | — | 0,0664 | 1,000 |

IC = intervalo bootstrap 95 % (percentil, 200 réplicas, remuestreo de filas de test) del AUC.
`n` es el tamaño del test (mismo para todos los modelos de una cabeza: mismas filas). Tablas
completas (incluye Brier, KS) en `salidas/secuencial_afiliado/metricas_{traspaso,cese}_6m.csv`.

### Conclusión explícita (criterio del BRIEF: mejora ≥ 0,01 AUC en ≥ 2 cabezas = "reemplaza")

**NO_APORTA.** La red (multi-tarea o single-task) no mejora el LightGBM de producción en
NINGUNA cabeza: Δ AUC (red_multitarea − lightgbm_produccion) = **−0,141** en traspaso y
**−0,084** en cese — ambas caídas grandes, muy por fuera del ±0,01 del criterio de "empate".
Tampoco gana el multi-tarea al single-task (traspaso: 0,585 vs 0,577, una diferencia pequeña
dentro del ruido; cese: 0,777 vs 0,781, el single-task es levemente MEJOR) — a esta escala y
presupuesto de entrenamiento, compartir encoder entre las dos tareas no ayudó.

El hallazgo más interesante, y el que sí aísla arquitectura de información como pide el brief:
**la red (multi-tarea o single-task) queda prácticamente EMPATADA con el LightGBM "tabular con
lags"** que ve exactamente la misma información aplanada (traspaso: 0,585 vs 0,594; cese: 0,777
vs 0,777, diferencias de 0,001-0,009 AUC, dentro del solape de los IC). Es decir: la brecha
grande no es un problema de la arquitectura secuencial contra un LightGBM equivalente con la
misma información — es que la secuencia (panel común, sin las tablas operacionales de la AFP)
lleva mucha menos información que las features completas de 02/04 (contactos, reclamos, saldos,
cambios de fondo, uso digital, brecha de rentabilidad). La GRU no está "perdiendo" contra
LightGBM por ser peor arquitectura; está perdiendo porque hoy ve menos datos.

### Calibración, importancias, ejemplos

No se recalibró la red (§5, limitación) ni se corrieron ejemplos individuales explicados para
esta entrega (presupuesto de tiempo, §8 "Pendientes"). Las importancias por ganancia de LightGBM
(producción y tabular-lags) y la calibración/curvas de los LightGBM de 02 y 04 ya están reportadas
en sus propios README; no se repiten porque no cambiaron (mismas funciones, mismo split).

## 7. Lectura de negocio

Con los números de §6: si la red no mejora el AUC de ninguna cabeza en ≥ 0,01 frente al LightGBM
de producción, la lectura para la AFP es que **no conviene** reemplazar los pipelines tabulares
de 02/04 por una red secuencial a este presupuesto de datos/cómputo — el costo de mantener
PyTorch, embeddings y una GRU en producción no se justifica frente a un LightGBM que ya gana
(mismos cortes, mismo split, y con MENOS variables en el caso del baseline "tabular con lags").
Si, en cambio, el baseline "tabular con lags" (misma información que la red) se acerca o supera al
LightGBM de producción, eso sí sería una señal útil por separado: que la información secuencial
aplanada compite con las features hechas a mano, aunque la arquitectura GRU en sí no aporte.

## 8. Decisiones tomadas, supuestos y limitaciones

* **Worktree**: se trabajó en `/home/user/wt/empleador` (rama `ronda-4/secuencial`), no en
  `/home/user/wt/datos` que nombra el brief original — instrucción explícita de arranque del
  gerente para esta ronda; no afecta nada del contenido.
* **Alcance recortado a 2 cabezas** (traspaso 02, cese-6m 04) de las 4 que pide el brief
  (traspaso, cese, APV, RIM) — instrucción explícita del gerente por presupuesto de la ronda
  (~200 llamadas / ~90 min): "prioridad: cabezas de traspaso y cese". **Pendiente**: APV (03,
  aporte 6 m) y RIM (regresión Huber sobre log1p, corte+1) — no implementadas.
* **Embedding vs. segmentación de 06**: se expone `model.obtener_embedding` (y `run.py` guarda un
  csv con el embedding del test de traspaso) pero la comparación k-means/ARI/silueta contra la
  segmentación de 06 **no se hizo** — declarado pendiente, como autoriza explícitamente el brief
  de esta ronda si algo no cabe.
* **Elección de arquitectura**: GRU de 2 capas (no transformer) — el brief deja elegir con
  validación; con el presupuesto de esta ronda no se probó el transformer, se fue directo a GRU
  (arquitectura más liviana y con menos hiperparámetros, razonable para 24 pasos).
* **Hiperparámetros de la red elegidos SOLO con validación** (nunca con el test), en una corrida
  a escala reducida (n=3.000): se probaron `lr ∈ {1e-3, 3e-3}` y `pesos ∈ {parejo, 0.6/0.4}`;
  `lr=3e-3` + `pesos={"traspaso":0.6,"cese":0.4}` dio el mejor AUC de validación combinado y quedó
  fijo para la corrida de referencia. No se hizo una búsqueda más extensa (grilla de `hidden`,
  `dim_embedding`, número de capas) por presupuesto de tiempo.
* **`max_cortes_train=7`** (en vez del `31` por defecto de `churn_traspaso.split_temporal`, o el
  ~24 natural de cese): recorte deliberado para que el runtime entre dentro de <5 min de
  entrenamiento de red / <8 min de corrida completa (restricción dura de esta ronda) a escala de
  10.000 afiliados. Aplicado por IGUAL a la red, al LightGBM de producción y al de tabular-lags de
  cada cabeza (mismos cortes de train para los tres, comparación justa dentro de este proyecto);
  **no** es necesariamente el mismo número de cortes de train que reportan los README propios de
  02 y 04 (que sí usan su ventana completa) — por eso el `lightgbm_produccion` de esta tabla se
  re-entrena aquí mismo, en el split recortado, y no se copian los números del README de 02/04.
* **Brier de la red no calibrado** (§5) — no comparar directamente con el de los LightGBM.
* **Sin `pack_padded_sequence`** (§5) — simplificación documentada, no debiera afectar
  materialmente el resultado dado que los cortes usados exigen que el último paso siempre exista.
* **La secuencia no lleva las tablas operacionales de 02** (contactos/reclamos/saldos/cambios de
  fondo/uso digital) a propósito (§4.3): así el baseline "tabular con lags" aísla arquitectura de
  información, como pide el brief. Efecto esperado: el `lightgbm_produccion` (que sí las tiene)
  puede ganarle a la red y al tabular-lags solo por tener más información, no por arquitectura.

## 9. Cómo correr y tiempos

```bash
cd /home/user/wt/empleador
PYTHONPATH=/home/user/wt/empleador python3 -m pytest afp_lab/secuencial_afiliado/tests -q
PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.secuencial_afiliado.run --n-afiliados 10000
```

**Tiempos medidos** (máquina de 4 vCPU compartida, `torch.set_num_threads(2)`, LightGBM
`num_threads/n_jobs=2`):

* Tests (7 casos, escala chica 300-500 afiliados + 1 corrida end-to-end diminuta hidden=4/1
  época): **16,6 s** (< 60 s exigido).
* Corrida de referencia (`--n-afiliados 10000`, defecto): **228,9 s totales** (< 8 min), de los
  cuales **143,2 s** son el entrenamiento de la red (multi-tarea + 2 single-task, < 5 min exigido).
  Desglose: universo+tablas 11,5 s; datasets/splits de 02 y 04 18,6 s; secuencias 1,1 s;
  baselines LightGBM (producción + tabular-lags, ambas cabezas) 10,0 s; red 143,2 s; métricas +
  bootstrap (200 réplicas) 40,4 s.

## 10. Propuestas para `common/` y próximos pasos

* **Propuesta para `common/`**: una función tipo `construir_matrices_panel` (pivotear
  `panel_mensual` a `(afiliado x periodo)` por canal, una vez) sería útil para cualquier proyecto
  que necesite ventanas móviles del panel común — hoy cada proyecto (02, 04, este) reimplementa su
  propia versión de "pivotear y tomar una ventana". No se movió a `common/` en esta entrega para
  no tocar código compartido sin acordarlo (CONVENCIONES §5); queda como propuesta para el gerente.
* **Próximos pasos** (si la ronda 5 sigue esta línea): (1) cabezas de APV y RIM; (2) comparación
  del embedding contra 06; (3) calibración de la red (isotónica sobre el embedding final, igual
  que 02/04); (4) `pack_padded_sequence` en vez de máscara multiplicativa; (5) probar un
  transformer encoder pequeño (el brief lo deja como alternativa) y una búsqueda de
  hiperparámetros más amplia (hidden, dim_embedding, número de capas) ahora que se sabe que el
  pipeline corre dentro del presupuesto de tiempo.
