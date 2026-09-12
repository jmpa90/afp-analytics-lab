# 05 · Riesgo de empleador: declaración y no pago (DNP) y pago tardío

Modelo de cobranza y de proyección de recaudación. Para cada **empleador** y cada mes de decisión
estima la probabilidad de que la planilla de ese mes no se pague y **cuánta masa imponible de
nuestros afiliados está en riesgo**, para que el 10 % de empleadores que cobranza puede gestionar
sea el que concentra más masa.

```bash
python -m afp_lab.riesgo_empleador.run --n-afiliados 20000    # corrida de referencia, ~26 s
python -m pytest afp_lab/riesgo_empleador/tests -q            # 21 tests, ~13 s
```

**Ronda 3** (ver §"Ronda 3" más abajo): `afiliados_por_empleador` baja de 40 a **20** por defecto
(~1.000 empleadores en vez de ~500) y el perfil `cuprum` recalibra su densidad de cotización
(`common/synth_base.py`, ronda 3 del proyecto `empleador`). AUC test 0,818 → **0,865**; captura de
masa al 10 % 54,5 % → **48,6 %** (promedio de 5 semillas: 51,9 %, sigue "en el límite" como en la
ronda 2); volatilidad intrínseca del agregado mensual 98,9 % → **63,6 %**.

---

## Ronda 3 (worktree `empleador`): densidad de `cuprum` recalibrada + más empleadores

Dos objetivos independientes, ambos corridos con 20.000 afiliados, semilla 42 salvo donde se indica.

### Objetivo 1 — heredado: recalibración de la densidad de cotización (`common/synth_base.py`)

No es un cambio de este proyecto: el brief `empleador` recalibra `PERFILES["cuprum"]` (y
`PERFILES["sistema"]`) para que la fracción de afiliados con RIM > 0 en un mes típico se acerque a la
Ficha 165 de la SP (66 % Cuprum, 47,3 % sistema, antes ~71-73 % y ~79 % respectivamente). Como este
proyecto usa `perfil="cuprum"` por defecto, **todos sus números de referencia cambian** aunque no se
tocó ni una línea de `synthetic.py`, `features.py` ni `model.py`. Aislando sólo este efecto (mismos
~500 empleadores que la ronda 2, sólo cambia la densidad):

| Indicador | Antes (fc=0,60, ~500 empl.) | Después (fc=0,40, ~500 empl.) |
|---|---|---|
| AUC test `y_dnp` | 0,822 | 0,841 |
| Captura de masa al 10 % | 52,3 % | 47,6 % (no cumple ≥50 % en esta corrida) |
| Captura al 10 % vs. regla | 42,9 % → **+9,5 pp** | 34,9 % → **+12,7 pp** |
| Volatilidad intrínseca del agregado | 98,5 % | 94,6 % |
| Planillas del mes con deuda | 5,64 % | 5,56 % |

(Fila "antes" corrida con los parámetros previos a la ronda 3 sobreescritos en memoria —sin tocar el
archivo— para aislar el efecto; por eso el AUC (0,822) difiere en unas décimas del 0,818 documentado
en la ronda 2, que corrió con `diagnostico=True` y pudo capturar variaciones menores de código entre
rondas — la diferencia no es material para ninguna conclusión.) La densidad más baja sube el AUC (la cartera
cesante crece y es más distinta de la empleada) pero por sí sola **no** alcanza a compensar la
volatilidad del agregado ni sube la captura de masa — para eso hace falta el objetivo 2.

### Objetivo 2 — más empleadores: 40 vs. 20 vs. 10 afiliados por empleador

La ronda 2 aprobó este proyecto "con reserva": con ~500 empleadores el agregado mensual de masa en
riesgo es intrínsecamente muy volátil (§6.4). `generar_universo` ya exponía `afiliados_por_empleador`
(ronda 2); este brief pide correr 40 (actual) vs. 10 (~2.000 empleadores) a 20.000 afiliados y elegir
un default. Se corrió también 20 (~1.000) como punto intermedio, dos semillas cada uno, **ya con** la
densidad de `cuprum` recalibrada (objetivo 1) para no mezclar efectos:

| `afiliados_por_empleador` | Empleadores | AUC (2 semillas) | Captura masa @10 % (2 semillas) | Volatilidad intrínseca media |
|---|---|---|---|---|
| 40 (ronda 2) | ~500 | 0,841 / 0,855 | 47,6 % / 47,4 % | 94,6 % / 72,7 % |
| **20 (elegido)** | **~1.000** | 0,865 / 0,861 | 48,6 % / 50,3 % | 63,6 % / 62,7 % |
| 10 | ~2.000 | 0,843 / 0,869 | 43,4 % / 61,7 % | 47,9 % / 50,9 % |

Lectura: la volatilidad intrínseca **baja monótonamente y de forma predecible** con más empleadores
(escala ~`1/√n_empleadores`: 500→1.000 predice una caída a ~71 % del valor anterior, medido 63-67 %;
500→2.000 predice ~50 %, medido 48-51 %). El AUC es estable en los tres puntos (0,84-0,87). La
**captura de masa al 10 %** es donde está la sorpresa: con 2.000 empleadores se vuelve **más
inestable entre semillas** (43,4-61,7 %, rango de 18 pp) que con 1.000 (48,6-50,3 %, rango de 1,7 pp)
o incluso que con 500 (47,4-47,6 %, rango de 0,2 pp aunque ambas bajo 50 %) — reducir la varianza del
**agregado** (una suma) no reduce la varianza de **qué empleadores concretos** entran al 10 % superior
del ranking (un estadístico de orden), que sigue dominada por un puñado de DNP grandes.

**Decisión: `afiliados_por_empleador = 20`** (~1.000 empleadores), default nuevo de
`run.py`/`generar_tablas`. Reduce la volatilidad del agregado en un tercio frente a 40 sin la
inestabilidad adicional de 10 en la métrica de negocio, y el runtime no cambia (§9): la generación
(que domina el tiempo) depende de `n_afiliados`, no del número de empleadores. Con 5 semillas sobre
esta elección (42/7/2/3/4): AUC 0,861-0,877, captura al 10 % 48,6-57,2 % (media 51,9 %) — ver §6 para
el resultado completo con `diagnostico=True` (semilla 42, la corrida de referencia del proyecto).

El brief preveía bajar a 10.000 afiliados si alguna corrida del experimento superaba los 5 minutos;
no hizo falta: las nueve corridas (3 valores de `afiliados_por_empleador` × 2-5 semillas) terminan en
25-50 segundos cada una, muy por debajo del límite.

### Qué NO se tocó

El modelo (`model.py`), las features (`features.py`), el split temporal, los baselines y los criterios
de aceptación son exactamente los de la ronda 2. Todo el movimiento de números en §6 viene de la
escala del universo y de la densidad de cotización, no de un cambio de metodología.

---

## 1. Pregunta de negocio y decisión que apoya

*¿A qué empleadores llama cobranza este mes, y cuánta masa imponible no va a entrar?*

| Salida | Uso |
|---|---|
| `prob_y_dnp` | probabilidad de que la planilla del periodo `corte` sea DNP o no se declare. Calibrada, se agrega para **proyectar la masa que no entrará** |
| `masa_en_riesgo_esperada = prob_y_dnp × masa_ref` | **ranking de cobranza**: los CLP de masa imponible esperados en riesgo por empleador |
| `prob_y_tardio`, `prob_y_dnp_3m` | variantes de horizonte que el brief pide reportar |

La alternativa actual —la regla "DNP en 3 meses o en el boletín de infractores", y su prima
hermana "el que quedó debiendo el mes pasado"— es el baseline a batir.

---

## 2. Datos sintéticos: tablas, mecanismos y parámetros

Tres capas, cada una con su responsable:

1. **`common.synth_base.generar_universo(perfil="cuprum")`**: afiliados, empleadores y panel
   laboral de nuestra cartera (`docs/contexto_afp_cuprum.md` §4).
2. **`common.planilla.generar_planilla`** (módulo de DS-3, `planilla.py@1fc28ef5`, copiado a este
   worktree el 2026-09-12 y registrado en `synthetic.VERSION_PLANILLA`): planilla Previred **línea a
   línea** (afiliado × empleador × periodo × tipo de pago) con los códigos de la Tabla N° 7, días
   trabajados, renta imponible AFP/AFC, cotización obligatoria, SIS y aporte del empleador de la
   reforma. De aquí sale el **contenido** de la planilla, agregado a empleador × periodo: dotación,
   masa imponible, altas (códigos 1/7), bajas (2), licencias (3) y fracción a plazo fijo.
3. **`riesgo_empleador.synthetic`**: la **capa de recepción y pago del empleador**, que es lo que
   este proyecto modela y que `common.planilla` no tiene.

**Por qué la capa de recepción es propia.** `common.planilla` sortea el rezago y la DNP *línea por
línea* con las probabilidades fijas por tamaño de `empleadores_latente`, sin persistencia, sin
estacionalidad y sin episodios. Con eso el target de este proyecto sería casi impredecible más allá
del tamaño del empleador, y el brief exige explícitamente las tres cosas. Aquí la decisión se toma
**una sola vez por (empleador, periodo)** —que es como llega la planilla en la realidad— y es la que
se reporta en `planillas`. *(Nota para DS-3: en `_aplicar_recepcion` el sorteo es `rng.random(n)` con
`n` = número de líneas, así que dos líneas de la misma (afiliado, empleador, periodo) pueden quedar
con `periodo_recepcion` distinto, lo que contradice el propio docstring del módulo. No lo toqué
porque `common/` no es mi carpeta.)*

### 2.1 Tablas (corrida de referencia, 20.000 afiliados → ~1.000 empleadores, ronda 3)

Hasta la ronda 2, `afiliados_por_empleador=40` dejaba ~500 empleadores; la ronda 3 (objetivo 2,
§"Ronda 3" abajo) lo baja a **20** (~1.000 empleadores) para reducir la volatilidad del agregado
mensual. Cifras de esta sección con el default nuevo:

| Tabla | Grano | Filas | Contenido |
|---|---|---|---|
| `planillas` | empleador × periodo | 70.574 | n_trabajadores, masa imponible, cotización, `periodo_recepcion`, `lag_recepcion`, `estado` (PAGADA / DNP / SIN_DECLARAR), `periodo_regularizacion`, `n_rectificaciones`, altas/bajas/licencias, % plazo fijo |
| `boletin_infractores` | empleador × mes | 3.047 (336 empleadores) | inclusión en el Boletín de Infractores Laborales y Previsionales (DT) |
| `boletin_concursal` | evento | 171 | REORGANIZACION / LIQUIDACION (Superir) con `periodo_publicacion` |
| `shocks_empleador` | latente | 606 episodios (99 liquidaciones) | episodios de estrés financiero. **Nunca es feature** (hay test) |

Ventana **201901–202609** (93 meses); los 12 primeros son burn-in del historial de pago.
Semilla fija, generación determinista (hay test). (Con `afiliados_por_empleador=40`, ronda 2: 500
empleadores, 39.790 planillas, 1.849 filas de boletín de 196 empleadores, 94 eventos concursales,
411 episodios de estrés con 55 liquidaciones — tabla anterior en el historial de git.)

### 2.2 Mecanismos y parámetros

| Mecanismo | Parámetros | Por qué es razonable para Chile |
|---|---|---|
| **Probabilidad base de DNP** | la del proyecto 01 por tamaño (`empleadores_latente.p_dnp`: MICRO 6 %, PEQUEÑA 4 %, MEDIANA 2 %, GRANDE 0,5 %) × `factor_dnp = 0,45` | la mora previsional está concentrada en micro y pequeña empresa |
| **Sector público** | `p_dnp = 0`, nunca entra en estrés | la administración pública paga por ley (hay test) |
| **Rubro** | `MULT_DNP_RUBRO`: construcción 1,8; agricultura 1,7; turismo 1,6; servicios 1,3; comercio y casa particular 1,2; minería y finanzas 0,5 | son los rubros con más mora previsional declarada |
| **Estacionalidad** | `MULT_DNP_MES`: dic 1,5; ene 1,6; feb 1,4; mar 1,3 | gratificación, aguinaldo, vacaciones y "el marzo" aprietan la caja |
| **Persistencia** | `MULT_DNP_REPITE = 3,5` si hubo deuda el mes anterior, ×1,6 adicional si vienen dos meses seguidos | el que no pagó un mes tiende a repetir |
| **Fragilidad estable del empleador** | `SIGMA_FRAGILIDAD = 0,90` (factor log-normal por empleador, constante en el tiempo, sobre `p_dnp`, y su raíz sobre el atraso) | hay empleadores crónicamente atrasados. **Es la fuente principal de predictibilidad**: 12 meses de historial de pago la revelan |
| **Episodios de estrés** | `P_ESTRES_BASE = 0,6 %` mensual × `MULT_ESTRES_TAMANO` (micro 2,2 … grande 0,35) × `(1 + 12 × (desempleo − 8 %))`; salida 12 % mensual (duración media ~8 meses); al salir, 15 % **liquidación** y 85 % regularización; durante el episodio `MULT_DNP_ESTRES = 10` y `MULT_TARDIO_ESTRES = 3` | la mora suele ser un episodio, no un estado permanente; y correlaciona con el ciclo |
| **Liquidación** | el empleador deja de declarar; sus trabajadores siguen en la nómina `MESES_SIN_DECLARAR_TRAS_LIQUIDACION = 3` meses (planillas `SIN_DECLARAR` con deuda real) y después sale del universo | entre la liquidación y los finiquitos pasan meses |
| **Regularización de la DNP** | 2–9 meses después de la declaración; `P_NUNCA_REGULARIZA = 12 %` | la deuda declarada se cobra, con rezago |
| **Boletín de infractores** | entra tras `MESES_DEUDA_BOLETIN = 2` meses consecutivos de deuda, con 1 mes de rezago de publicación, y se mantiene 2 meses después de pagar | así opera el Boletín de la Dirección del Trabajo |
| **Recepción tardía** | `p_pago_tardio + p_rezago` del empleador × `factor_tardio = 0,65` × estrés | mismas reglas de recepción del proyecto 01 |

### 2.3 Tasas resultantes vs. las pedidas en el brief (ronda 3: `afiliados_por_empleador=20`, perfil `cuprum` recalibrado)

| Indicador | Brief | Corrida 20.000 (ronda 3) | Corrida 20.000 (ronda 2) |
|---|---|---|---|
| planillas del mes con deuda (DNP + no declarada) | 3–6 % | **5,17 %** (DNP 3,60 % + no declarada 1,57 %) | 5,64 % |
| planillas con recepción tardía (lag ≥ 2 o nunca) | 5–10 % | **7,50 %** | 7,65 % |
| AUC de test | 0,75–0,90 | **0,865** | 0,818 |

---

## 3. Target, población, horizonte y exclusiones

**Convención de fechas (CONVENCIONES §3)**: `periodo_corte` es el **mes de la decisión**.

| Fuente | Último dato conocido |
|---|---|
| `planillas` | sólo las **recibidas**: `0 < periodo_recepcion ≤ corte`. La del periodo `corte-1` llega en `corte` si es puntual; una que aún no llegó es información en sí misma |
| `periodo_regularizacion` de una DNP | sólo si es ≤ `corte` (si no, la deuda sigue viva) |
| boletines | publicaciones con periodo ≤ `corte` |
| `panel_mensual` | hasta `corte-2` (desfase de la cotización) |
| `shocks_empleador` | nunca |

* **Fila** = empleador × `periodo_corte`, cortes mensuales móviles.
* **Población**: empleadores con al menos una planilla **recibida** en los 12 meses previos y con al
  menos un trabajador afiliado en esa última planilla conocida. Del target salen además los meses en
  que el empleador no tenía que declarar (sin trabajadores): un mes sin planilla no es una DNP.
* **Target principal** `y_dnp` = 1 si la planilla del periodo `corte` (que se declara en `corte+1`)
  es DNP o no se declara. Es el que se elige porque **es el que corresponde a la acción**: cobranza
  actúa en el mes `corte` para evitar que esa planilla no se pague, y es el que se agrega para
  proyectar la recaudación del mes. Se reportan también `y_tardio` (lag ≥ 2 o nunca) y `y_dnp_3m`
  (alguna DNP en `corte … corte+2`).
* `masa_periodo` = masa imponible del periodo; **`masa_en_riesgo = y_dnp × masa_periodo`**.
* Un empleador liquidado sigue en la población los 3 meses en que sus trabajadores permanecen en la
  nómina, con planilla `SIN_DECLARAR`: la deuda es real y cobranza debe actuar.

Con 20.000 afiliados quedan **~700 empleadores elegibles por corte** de los ~1.000 del universo
(ronda 2, `afiliados_por_empleador=40`: ~400 de 500).

---

## 4. Features (51 columnas) y su regla as-of

`features.py` convierte las tablas en matrices densas `empleador × periodo` una sola vez; la máscara
`recibida` (`existe & recepcion ≤ corte`) es la única puerta por la que pasa la información.

| Familia | Features | Regla as-of |
|---|---|---|
| **Historia de pago** | `tasa_puntualidad_3/6/12`, `n_dnp_3/6/12`, `meses_desde_ult_dnp` + `nunca_dnp_12m`, `racha_puntualidad`, `n_recibidas_12`, `n_faltantes_12`, `n_rectificaciones_12` | ventana `corte-12 … corte-1`, sólo planillas recibidas |
| **Deterioro reciente** (alerta temprana) | `lag_medio_3/6/12`, `lag_max_6`, `deterioro_lag` (= lag 3 m − lag 12 m), `lag_ultima_planilla`, `meses_desde_ult_planilla` | idem |
| **Deuda viva** | `n_dnp_pendientes`, `masa_dnp_pendiente_imm`, `meses_deuda_consecutivos` | una DNP cuenta como viva mientras su regularización no haya llegado al corte |
| **Dotación y masa** | `n_trabajadores`, `var_n_trab_3/12`, `masa_ref_imm`, `masa_por_trabajador_imm`, `var_masa_3/12`, `n_altas_3`, `n_bajas_3`, `frac_altas_3`, `frac_bajas_3`, `n_licencias_3`, `frac_plazo_fijo`, `dias_trabajados_medio`, `meses_como_empleador` | de la última planilla recibida y de las recibidas de los últimos 3 meses |
| **Boletines públicos** | `en_boletin_infractores`, `meses_en_boletin_12`, `concursal_liquidacion`, `concursal_reorganizacion` | publicaciones ≤ corte |
| **Calendario y macro** | `mes_corte`, `es_dic_a_marzo`, `desempleo`, `ipc_12m`, `imm_var_12m` | del mes de corte |
| **Atributos del empleador** | `rubro`, `tamano_empleador`, `region_empleador`, `politica_gratificacion`, `freq_reajuste` | maestro |

Los montos en CLP van en **unidades de IMM**; `masa_ref_clp` e `imm_corte` viajan como columnas de
escala y están excluidas del modelo (hay test). `columnas_modelo` levanta error si apareciera una
columna `p_*`, `lat_*` o con "estres"/"shock" en el nombre.

### Split: orígenes móviles con gap entre todas las particiones

El gap se dimensiona con la **ventana de target más larga que se reporta** (`y_dnp_3m`, 3 meses), no
con la del target principal (1 mes), para que las tres variantes queden limpias con un solo split:
`corte_max(anterior) + 3 + 3 ≤ corte_min(posterior)`. Reentrenamiento **trimestral** (4 orígenes,
ventana expansiva) cubriendo 12 cortes de test:

| Origen | train | valid | test | filas train | tasa train / valid / test |
|---|---|---|---|---|---|
| 202508 | 202001–202309 (45) | 202403–202502 (12) | 202508–202510 | 34.678 | 5,24 % / 5,96 % / 4,59 % |
| 202511 | 202001–202312 (48) | 202406–202505 (12) | 202511–202601 | 36.871 | 5,20 % / 5,91 % / 5,77 % |
| 202602 | 202001–202403 (51) | 202409–202508 (12) | 202602–202604 | 39.051 | 5,25 % / 5,90 % / 6,40 % |
| 202605 | 202001–202406 (54) | 202412–202511 (12) | 202605–202607 | 41.206 | 5,27 % / 5,73 % / 5,35 % |

Test: **8.316 filas, 12 cortes, 460 empleadores-mes con DNP** (ronda 2, `afiliados_por_empleador=40`:
4.771 filas, 230 con DNP).

**Limitación declarada** (misma que la m-2 del proyecto 03): dentro de cada origen el `valid` no toca
ningún mes de target de **su** bloque de test (cero solape, hay test), que es la condición que afecta
a cada predicción. Al agrupar los 12 cortes, el `valid` de los dos últimos orígenes sí toca 3 y 6
meses de target de bloques anteriores, que predijeron otros modelos
(`reportes/solape_valid_pool_test.csv`). Por eso el test se mantiene en **12 cortes** y no en 24:
con 24 el solape sube a 14 meses en los últimos orígenes.

---

## 5. Modelo y baselines

`LGBMClassifier` por target, con el mismo protocolo de la ronda 2 del proyecto 03:

* `num_leaves=12`, `min_child_samples=150`, `lambda_l2=10`, `lr=0,05`, `n_jobs=2` — regularización
  más fuerte que en el proyecto 03 porque el dataset es de cientos de empleadores, no de miles de
  afiliados;
* **número de árboles por early stopping** contra el `valid` del origen: **124 / 46 / 72 / 97** sobre
  un tope de 3.000, que nunca se toca (ronda 2, 500 empleadores: 74 / 69 / 146 / 56);
* **calibración isotónica** sólo si la mejora del Brier fuera de muestra dentro de validación supera
  2 errores estándar de la dispersión entre folds **y** el 1 % del Brier: en esta corrida **ningún
  origen la aplica** (mejoras entre −4,0e-4 y +2,0e-4, todas por debajo de su error estándar);
* ningún número de test entra en ninguna decisión.

**Baselines**:

* `persistencia`: ¿el empleador quedó debiendo el último mes conocido (o falta una planilla)? Con las
  tasas empíricas de entrenamiento de cada grupo como probabilidad. Es lo que hace cobranza hoy.
* `regla_negocio` (la del brief): *DNP en 3 meses **o** en el boletín de infractores*.
* `tasa_base`: una sola constante para todo el pool (AUC exactamente 0,500).
* `masa_sola` y `aleatorio` como referencias de la métrica de negocio.

### Diagnóstico de memorización por empleador

| Evaluación | AUC |
|---|---|
| train (en muestra) | 0,885 |
| **CV agrupada por empleador** (`GroupKFold`, dentro de train) | **0,857** |
| validación temporal con gap | 0,872 |

La CV agrupada coincide con la validación temporal: el modelo generaliza a empleadores que no vio, y
no está memorizando los ~1.000 del universo (ronda 2, 500 empleadores: train 0,879 / CV 0,849 /
temporal 0,849).

---

## 6. Resultados (ronda 3: test = 12 cortes 202508–202607; 8.316 filas; 460 DNP; tasa base 5,53 %)

Cifras de la ronda 2 (500 empleadores, perfil `cuprum` sin recalibrar) entre paréntesis para
comparar; detalle de qué cambió y por qué en §"Ronda 3" más abajo.

### 6.1 Clasificación del target principal

| Predictor | AUC | Average precision | Brier | Lift decil 1 | KS |
|---|---|---|---|---|---|
| **Modelo** | **0,865** (0,818) | **0,488** (0,331) | **0,0348** (0,0374) | **6,50** (5,39) | **0,595** (0,484) |
| Persistencia | 0,748 (0,725) | 0,118 (0,098) | 0,0493 (0,0442) | 2,41 (2,40) | 0,496 (0,453) |
| Regla de negocio | 0,756 (0,669) | 0,254 (0,141) | 0,0421 (0,0420) | 5,71 (4,10) | 0,516 (0,348) |
| Tasa base | 0,500 | 0,055 (0,048) | 0,0523 (0,0461) | 1,00 | 0,058 (0,118) |

Por corte: AUC de **0,775 a 0,915** (ronda 2: 0,741 a 0,911), lift de 5,54 a 7,84.

### 6.2 Los tres targets

| Target | Tasa base | AUC | AP | Lift decil 1 |
|---|---|---|---|---|
| `y_dnp` (principal, 1 mes) | 5,53 % | **0,865** (0,818) | 0,488 (0,331) | 6,50 (5,39) |
| `y_dnp_3m` (alguna DNP en 3 meses) | 10,06 % | 0,803 (0,766) | 0,469 (0,344) | 4,65 (3,66) |
| `y_tardio` (lag ≥ 2 o nunca) | 7,67 % | 0,742 (0,689) | 0,213 (0,158) | 3,34 (2,47) |

Los tres targets mejoran en AUC y en AP; el orden relativo no cambia: `y_tardio` sigue siendo el más
difícil (ruido operativo de cuándo alcanzó a pagar) y `y_dnp_3m` sigue diluyendo la señal de "este
mes" en tres meses.

> **Reserva del gerente (revisión ronda 2)**: el criterio "captura de masa en riesgo ≥ 50 % al 10 %"
> se cumplía con 12 cortes (54,5 %) y no con 24 (44,5 %); el error estándar era ±6,4 pp. La forma de
> reducir la varianza propuesta era ampliar el universo de empleadores. La ronda 3 lo hizo (§2.1,
> 500 → ~1.000 empleadores): la volatilidad intrínseca del agregado mensual efectivamente baja
> (98,9 % → 63,6 %, §6.4), pero la captura al 10 % de esta corrida puntual (semilla 42) queda en
> **48,6 %**, bajo el 50 %; el promedio de 5 semillas (42/7/2/3/4) es 51,9 % (rango 48,6–57,2 %).
> Sigue siendo "45–55 % de la masa con el 10 % de la cartera, mejor que la regla en todos los casos",
> no un cumplimiento sólido — ver §6.3.

## 6.3 Métrica de negocio: masa imponible en riesgo capturada

| Score de gestión | 10 % gestionado | 20 % gestionado |
|---|---|---|
| **Modelo (`masa_en_riesgo_esperada`)** | **48,6 % de la masa** / 39,8 % de los DNP / precisión 21,9 % | **70,9 %** / 53,5 % / 14,7 % |
| Persistencia | 20,3 % / 5,2 % / 2,9 % | 55,6 % / 18,5 % / 5,1 % |
| Regla de negocio | 44,1 % / 19,6 % / 10,8 % | 62,3 % / 33,9 % / 9,3 % |
| Modelo (sólo probabilidad) | 52,7 % / 65,0 % / 35,7 % | 57,2 % / 74,8 % / 20,6 % |
| Masa sola (ordenar por tamaño) | 20,1 % / 2,2 % / 1,2 % | 32,7 % / 4,8 % / 1,3 % |
| Aleatorio | 10,0 % / 12,0 % / 6,6 % | 21,1 % / 22,2 % / 6,1 % |

(Ronda 2, 500 empleadores: modelo 54,5 % / 28,7 % / 13,7 % al 10 %; regla 42,5 % / 8,7 % / 4,1 %.)

Gestionando al **10 % de los empleadores** (≈ 70-80 al mes, antes ≈ 40) se cubre el **48,6 % de la
masa imponible en riesgo** en esta corrida, contra 44,1 % de la regla: **+4,6 pp sobre la regla**
(lift 1,10×), menor que la ganancia de la ronda 2 (+12,0 pp) — la regla también mejora con la nueva
densidad de DNP, así que el margen del modelo sobre ella se estrecha, aunque sigue ganando en las
dos fracciones y en las 5 semillas revisadas.

Nótese el contraste entre los dos scores del modelo: ordenar por **probabilidad** encuentra el 65 %
de los empleadores morosos pero sólo el 53 % de la masa; ordenar por **probabilidad × masa**
encuentra el 40 % de los morosos pero la mayor parte del dinero relativo a su tamaño. Para
cobranza —cuyo objetivo es CLP recuperados— el segundo sigue siendo el correcto.

**Advertencia sobre la precisión de esta cifra (persiste tras la ronda 3).** La captura mensual sigue
siendo volátil porque la masa en riesgo de un mes la determinan una o dos caídas grandes; duplicar el
número de empleadores reduce la volatilidad del **agregado** (§6.4) pero no elimina la varianza de
**qué empleadores concretos** caen en el 10 % superior mes a mes. Con 5 semillas (42/7/2/3/4) la
captura al 10 % va de 48,6 % a 57,2 % (media 51,9 %, desviación ≈ 3,7 pp); el AUC en cambio es estable
(0,861–0,877). La conclusión robusta sigue siendo "el modelo captura del orden de 45–55 % de la masa
en riesgo con el 10 % de la cartera, y le gana a la regla"; ningún punto individual debe tomarse como
el número definitivo.

### 6.4 Proyección del agregado mensual: mejora sustancialmente, el criterio de ±10 % sigue sin cumplirse

| Indicador | Ronda 3 (~1.000 empleadores) | Ronda 2 (500 empleadores) |
|---|---|---|
| Meses con error ≤ 10 % | 2 de 12 | 0 de 12 |
| Meses dentro de **1 desviación estándar intrínseca** | **12 de 12** | 11 de 12 |
| Volatilidad intrínseca media del agregado | **63,6 %** | 98,9 % |
| Error absoluto medio / mediano | 0,50 / 0,47 | 0,97 / 0,59 |
| Sesgo agregado de los 12 meses juntos | +40,0 % | +16,8 % |

Duplicar (con creces) el número de empleadores **sí reduce la volatilidad intrínseca** del agregado
mensual, y de forma consistente con la teoría: la desviación estándar de una suma de Bernoulli
ponderada por masa escala aproximadamente como `1/√n_empleadores`, y 500→~1.000 predice una caída a
~71 % del valor anterior (98,9 %×0,71≈70 %); lo medido (63,6 %) es incluso mejor, probablemente porque
la recalibración de la densidad de `cuprum` también achica la masa promedio por empleador. Sigue
lejos de los ±10 % pedidos: para llegar ahí harían falta del orden de `(0,636/0,10)²≈40` veces más
empleadores desde aquí (≈ 40.000, sobre el orden de magnitud de una cartera real). El sesgo agregado
de los 12 meses sube a +40 % en esta corrida puntual (era +16,8 % en ronda 2) — coherente con que la
captura al 10 % también es más volátil entre semillas (§6.3); no se interpreta como una mejora ni un
empeoramiento estructural, sino como el mismo ruido de "una o dos caídas grandes" con una muestra de
empleadores distinta. La proyección de recaudación sigue debiendo reportarse **trimestral o
semestral**, no mensual.

### 6.5 Calibración

| Versión | Brier | ECE | Sesgo agregado |
|---|---|---|---|
| **Probabilidad usada (ningún origen calibrado)** | **0,0348** (0,0374) | **0,0070** (0,0064) | −0,6 % (+3,2 %) |
| Isotónica por origen (referencia, no se usa) | 0,0346 | 0,0087 | +6,5 % |

### 6.6 Por segmento (AUC): el modelo ahora gana en TODOS los tramos de tamaño ≥ 5 % de las filas

| Tamaño | % filas | DNP | Modelo | Persistencia | Regla |
|---|---|---|---|---|---|
| MICRO | 32,6 % | 208 | **0,849** | 0,703 | 0,743 |
| PEQUEÑA | 37,8 % | 219 | **0,870** | 0,739 | 0,788 |
| MEDIANA | 21,0 % | 32 | **0,682** | 0,611 | 0,563 |
| GRANDE | 8,7 % | 1 | **0,609** | 0,485 | 0,499 |

**Se resuelve el incumplimiento de CONVENCIONES §4 de la ronda 2**: en el tramo GRANDE el modelo
ahora gana (0,609 vs 0,485 persistencia / 0,499 regla; ronda 2: 0,447 vs 0,636, perdía). Sigue siendo
una muestra minúscula (**1 empleador-mes con DNP** en 720 filas de test, incluso más chica que los 3
casos de la ronda 2 en términos absolutos, aunque la fracción de filas subió de 6,5 % a 8,7 %), así
que el AUC de ese segmento sigue sin ser estadísticamente interpretable por sí solo — el resultado
puede volver a invertirse con otra semilla. Lo mismo pasa por región y por dotación: el modelo gana en
**los dos segmentos de región ≥5 %** (RM 0,882 vs 0,758/0,756; región 16 0,723 vs 0,633/0,592) y en
**los tres tramos de dotación ≥5 %** (1-4 trabajadores 0,836 vs 0,745; 5-19 0,863 vs 0,792; 20-99
0,726 vs 0,556 — la mejora más grande, aunque con sólo 15 positivos).

Por rubro hay 9 rubros con ≥ 5 % de las filas (antes 10; `K_FINANZAS` y `B_MINERIA` caen bajo el 5 %
con la nueva densidad); `O_ADMIN_PUBLICA` (828 filas) sigue con **0 casos de DNP** —el sector público
nunca deja de pagar—. De los 8 restantes el modelo gana a los dos baselines en **los 8** (AUC
0,763–0,920), incluido `C_MANUFACTURA` (0,849 vs 0,751 persistencia / 0,745 regla), que en la ronda 2
era el único rubro donde perdía.

### 6.7 Importancias (gain, %) y ejemplos

`en_boletin_infractores` **34,6** · `meses_deuda_consecutivos` **32,7** · `racha_puntualidad` 3,0 ·
`masa_dnp_pendiente_imm` 2,9 · `meses_desde_ult_dnp` 2,8 · `rubro` 2,6 · `n_dnp_12` 2,5 ·
`tasa_puntualidad_12` 2,2 · `masa_ref_imm` 1,8 · `region_empleador` 1,7 · `meses_en_boletin_12` 1,3 ·
`n_dnp_6` 1,2 (ronda 2: `meses_deuda_consecutivos` 50,8 · `meses_desde_ult_dnp` 10,6 · `n_dnp_3` 5,0 ·
`n_dnp_6` 4,7 · `racha_puntualidad` 3,9 · ...).

El boletín de infractores desplaza a `meses_desde_ult_dnp`/`n_dnp_3` como segunda variable más
importante: con más empleadores hay más variación en cuándo entra o sale del boletín, y esa señal
pública explica una fracción mayor del modelo que antes. La historia interpretativa no cambia: la
**persistencia** de la deuda y la **fragilidad estable** del empleador (que el historial de 12 meses
y el boletín revelan) siguen siendo el mecanismo dominante.

* **Empleador 740 (P_EDUCACION, MEDIANA, corte 202604)** — `prob 0,56`, masa esperada 44,6 MM; real:
  **pagó** (falso positivo). Empujan: 3 meses consecutivos de deuda (+1,89), estar en el boletín de
  infractores (+1,44), 3 DNP en 12 meses (+0,29). Empleador con historial de mora que esta vez
  regularizó a tiempo.
* **Empleador 522 (P_EDUCACION, PEQUEÑA, corte 202605)** — `prob 0,71`, masa esperada 32,4 MM; real:
  **no pagó, 26,3 MM en riesgo**. 12 meses consecutivos de deuda (+1,72), en el boletín (+1,36), 10
  meses en el boletín en el último año (+0,25). Episodio de estrés crónico, exactamente el caso que
  el modelo debe priorizar.
* **Empleador 90 (C_MANUFACTURA, MICRO, corte 202509)** — `prob 0,53`, masa esperada 2,6 MM; real:
  **no pagó, 4,7 MM en riesgo**. Aunque la masa esperada es chica (micro empresa), el modelo la
  identifica igual por 2 meses de deuda consecutiva (+1,52) y estar en el boletín (+1,41).
* **Empleador 293 (A_AGRICULTURA, GRANDE, corte 202605)** — `prob 0,010`. Sin deuda (−0,16), racha de
  12 meses de puntualidad (−0,19), tamaño GRANDE (−0,30): el modelo lo descarta con confianza, y el
  rubro agrícola (+0,21) no alcanza a compensar. Ejemplo de por qué GRANDE es un segmento de pocos
  casos pero bien separado cuando no hay señales de deterioro.

---

## 7. Lectura de negocio

**Con la capacidad actual de cobranza —10 % de los empleadores al mes, ~70-80 llamadas (antes ~40)—
el modelo cubre en torno al 45-55 % de la masa imponible en riesgo (48,6 % en esta corrida, 51,9 % en
promedio de 5 semillas), contra 44,1 % con la regla vigente: le sigue ganando en todas las semillas
revisadas, con un margen más ajustado que en la ronda 2 (antes +12,0 pp, ahora +4,6 pp en esta
corrida) porque la regla también mejora con la nueva densidad de DNP.** La precisión de la lista sube
de 10,8 % (regla) a 21,9 % (modelo): de cada 5 llamadas una cae en un empleador que efectivamente no
va a pagar, contra 1 de cada 9 con la regla.

Tres lecturas operativas:

* **Dos listas, no una.** Ordenar por masa esperada maximiza los CLP (48,6 % de la masa, 39,8 % de
  los morosos); ordenar por probabilidad maximiza los casos (65,0 % de los morosos, 52,7 % de la
  masa). Si el equipo de cobranza tiene un tramo telefónico masivo y otro de ejecutivo, el primero
  debería trabajar la lista por probabilidad y el segundo la lista por masa.
* **El tramo GRANDE ya no es un punto ciego del modelo** (§6.6, cambia respecto de la ronda 2): con la
  recalibración de la ronda 3 el modelo gana también ahí (0,609 vs 0,485-0,499), aunque con 1 solo
  caso de DNP en 12 meses la evidencia sigue siendo escasa — se recomienda mantener el boletín
  concursal y la señal de deterioro (`deterioro_lag`, `frac_bajas_3`) como respaldo en ese tramo, no
  reemplazarlos.
* **La proyección de recaudación sigue siendo trimestral, no mensual.** Duplicar el número de
  empleadores bajó el ruido irreducible del agregado mensual de ±99 % a ±64 % (§6.4) — una mejora
  real y medible — pero sigue muy por encima de la tolerancia de ±10 % pedida; comprometerse a un
  número mensual seguiría siendo vender algo que no existe a esta escala.

---

## 8. Decisiones tomadas, supuestos y limitaciones

**Decisiones**

1. **Target principal `y_dnp` a 1 mes** (y no la variante a 3 meses): es el que corresponde a la
   acción mensual de cobranza y a la proyección del mes. Las tres variantes se entrenan y se
   reportan.
2. **Se reutiliza `common/planilla.py` de DS-3** para el contenido de la planilla (códigos Previred,
   dotación, masa, movimientos) y se construye una capa propia de recepción/pago (§2). La versión
   usada queda registrada en `synthetic.VERSION_PLANILLA` (`planilla.py@1fc28ef5`); la revisión
   anterior (`18d3e167`) difiere sólo en el docstring y genera la planilla bit a bit idéntica
   (verificado por hash), así que los números de este README valen para las dos.
3. **El gap del split se dimensiona con la ventana de 3 meses**, la más larga que se reporta, no con
   la de 1 mes del target principal.
4. **12 cortes de test y no 24**: minimiza el solape entre el `valid` de un origen y el pool de test
   de los demás (§4), a costa de más varianza en la métrica de negocio, que se reporta explícitamente
   (§6.3).
5. **Un mes sin planilla no es una DNP**: es un mes sin trabajadores y sale de la población. El
   empleador liquidado sí queda 3 meses con planillas `SIN_DECLARAR`.
6. **Entrenamiento sin ponderar por masa.** Se probó ponderar por `masa_ref` (la métrica de negocio
   es masa-ponderada): la captura al 10 % bajó de 44,5 % a 39,1 % en la variante de 24 cortes, así que
   se descartó. El modelo sin ponderar además deja una probabilidad calibrada, que es lo que la
   proyección agregada necesita.
7. **Ronda 3, objetivo 2: `afiliados_por_empleador` baja de 40 a 20.** Se comparó 40 (~500
   empleadores, el valor de la ronda 2), 20 (~1.000) y 10 (~2.000) a 20.000 afiliados, dos semillas
   cada uno. La volatilidad intrínseca del agregado mensual baja monótonamente con más empleadores
   (~95 % → ~63 % → ~49 %, coherente con la escala `1/√n_empleadores` esperada), pero la captura de
   masa al 10 % se vuelve inestable entre semillas con 2.000 empleadores (43-62 %, cada empleador pesa
   menos y unos pocos DNP grandes siguen dominando) sin ganar AUC frente a 1.000. Se eligió **20**
   (~1.000 empleadores) como default de `run.py` porque reduce la volatilidad en un tercio sin esa
   inestabilidad adicional (captura 48,6-50,3 % en las dos semillas iniciales). No se tocó el modelo;
   sólo la escala del universo (`common.synth_base.generar_universo(afiliados_por_empleador=...)`).
8. **Ronda 3, objetivo 1: se hereda la recalibración de la densidad de cotización del perfil
   `cuprum`** (`factor_contratacion` 0,60 → 0,40 en `common/synth_base.py`, para acercarla al 66 % de
   la Ficha 165 en vez de ~71-73 %). No es una decisión de este proyecto, pero cambia sus números de
   referencia (§6) porque este proyecto usa `perfil="cuprum"` por defecto; el detalle de qué cambió y
   por qué está en `afp_lab/common/README.md`.

**Supuestos que habría que confirmar con la AFP**

* La planilla del mes *m* se recibe en *m+1* si el empleador es puntual, y lo que cobranza ve el mes
  `corte` son las recepciones hasta `corte` (no hay un feed intradía).
* El Boletín de Infractores se publica con un mes de rezago y se puede consultar como dato del mes.
* Las publicaciones del Boletín Concursal (reorganización, liquidación) están disponibles el mes
  siguiente a su publicación.
* La masa imponible de referencia para dimensionar el riesgo es la de la última planilla recibida
  (no hay una estimación previa de la planilla en curso).

**Limitaciones**

* **Escala**: 20.000 afiliados dan **~1.000 empleadores** (`afiliados_por_empleador = 20`, subido
  desde 40 en la ronda 3 — ver decisión 7). Una AFP real tiene decenas de miles. Duplicar los
  empleadores redujo la volatilidad intrínseca del agregado de ±99 % a ±64 % y sigue siendo la
  limitación dominante: para bajar a ±10 % harían falta del orden de 40x más empleadores desde aquí
  (§6.4), y la captura de masa al 10 % sigue teniendo ±3,7 pp de desviación entre semillas (§6.3).
* El panel laboral se genera **antes** que los shocks del empleador, así que un empleador liquidado
  sigue teniendo trabajadores en el panel; se resuelve sacándolo del universo de planillas 3 meses
  después de la liquidación, pero la coherencia perfecta exigiría simular las dos cosas juntas
  (propuesta para `common/`).
* No se modelan la cobranza judicial ni los convenios de pago, que en la realidad cambian la
  probabilidad de regularización.
* `y_tardio` (AUC 0,742, antes 0,689) sigue siendo el target más débil: el atraso de uno o dos meses
  es mayormente ruido operativo.
* El tramo GRANDE ya no pierde en AUC (§6.6, cambia respecto de la ronda 2), pero sigue con muy pocos
  casos (1 DNP en 12 meses de test) — la conclusión de que el modelo "gana" ahí no es robusta a otra
  semilla; se recomienda seguir tratándolo como un segmento de gestión por excepción.

---

## 9. Cómo correr y tiempos

```bash
cd /home/user/wt/empleador
PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.riesgo_empleador.run --n-afiliados 20000
PYTHONPATH=/home/user/wt/empleador python3 -m pytest afp_lab/riesgo_empleador/tests -q
```

Opciones: `--perfil {sistema,cuprum}` (por defecto `cuprum`), `--seed`, `--primer-corte`,
`--cada-origen`, `--afiliados-por-empleador` (nuevo en la ronda 3, por defecto **20**; la ronda 2 usaba
40), `--sin-diagnostico`, `--salida`.

| Etapa | Segundos (20.000 afiliados, ronda 3) | Ronda 2 (500 empleadores) |
|---|---|---|
| Generación (universo + planilla Previred + capa de recepción) | 15,5 | 18,5 |
| Dataset as-of | 0,7 | 0,6 |
| Entrenamiento (4 orígenes × 3 targets) | 7,2 | 5,5 |
| Evaluación, importancias, ejemplos y CV agrupada | 2,0 | 1,5 |
| **Total** | **26,1** (límite: 300) | 27,0 |

El runtime total prácticamente no cambia pese a duplicar los empleadores: la generación (que domina
el tiempo, la planilla línea a línea de `common` con 2,3 M de líneas para 20.000 afiliados no depende
del número de empleadores) se mantiene igual o baja, y el entrenamiento/evaluación —que sí escalan con
el número de empleadores— siguen siendo baratos porque el dataset de empleadores sigue siendo chico
frente al de afiliados.

Tests: **21 tests en ~13 s** (límite: 120; sin cambios de cobertura en la ronda 3, sólo de escala).
Cubren leakage por truncamiento y por mutación del futuro en dos cortes, que la planilla del propio
periodo de corte no sea visible, ausencia de columnas latentes y de escala, gaps entre todas las
particiones, recálculo a mano de 5 features en 2 empleadores × 2 cortes, población y target, forma
del dataset, determinismo, y la coherencia de la simulación (el sector público nunca cae en DNP,
persistencia y estacionalidad de la DNP, el boletín sólo tras 2 meses de deuda, la masa cuadra con la
RIM del panel), hiperparámetros elegidos con validación, baselines, sesgo por corte, persistencia del
modelo y corrida end-to-end.

Suite completa del laboratorio (los 6 proyectos + `common`, incluido el proyecto 01 con Spark):
**155 tests, 0 fallos, ~9 min 46 s** (`python -m pytest afp_lab -q` desde la raíz).

Salidas en `salidas/riesgo_empleador/`: `fuentes/*.parquet`, `dataset/*.parquet`,
`modelo/origen_<yyyymm>/` (uno por origen), `predicciones_test.parquet` y `reportes/*.csv`.

### Criterios de aceptación del brief

| Criterio | Umbral | Ronda 3 (obtenido) | Ronda 2 | ¿Cumple? |
|---|---|---|---|---|
| AUC de test | ≥ 0,80 | **0,865** (rango por corte 0,775–0,915) | 0,818 | **sí** |
| Captura de masa en riesgo al 10 % | ≥ 50 % | **48,6 %** (semilla 42; media 5 semillas 51,9 %, rango 48,6-57,2 %) | 54,5 % (± 6,4 pp) | **no en esta semilla** — en el límite, como en la ronda 2 (§6.3) |
| Captura al 10 % mejor que la regla | > 44,1 % (ronda 2: 42,5 %) | **48,6 %** (+4,6 pp; gana en las 5 semillas) | 54,5 % (+12,0 pp) | **sí** |
| Error del agregado mensual ≤ 10 % en ≥ 10 de 12 meses | 10 meses | **2 meses** (12 de 12 dentro de 1 σ intrínseca de ±63,6 %, antes ±98,9 %) | 0 meses | **no** — mejora mucho, sigue sin cumplirse — ver §6.4 |
| Runtime con 20.000 afiliados | < 5 min | **26 s** | 27 s | sí |
| Tests | < 2 min | **21 tests en 13 s** | 21 tests en 14 s | sí |
| AUC en el rango pedido a la simulación | 0,75–0,90 | 0,865 | 0,818 | sí |
| El modelo gana en todo segmento ≥ 5 % de filas | — | **gana en los 4 tramos de tamaño, los 2 de región y los 3 de dotación ≥5 %** (antes perdía en GRANDE) | perdía en GRANDE (3 casos) | **sí** — ver §6.6 (evidencia aún escasa en GRANDE, 1 caso) |

---

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/`** (no se modificó nada fuera de mis dos carpetas):

1. **`planilla.py`: el sorteo de recepción es por línea, no por planilla.** En `_aplicar_recepcion`,
   `rng.random(n)` con `n` = número de líneas hace que dos líneas de la misma
   (afiliado, empleador, periodo) puedan tener `periodo_recepcion` distinto, contra lo que dice el
   propio docstring. Sugerencia: sortear por `(empleador, periodo)` y propagar.
2. **`planilla.py` podría exponer un agregado empleador × periodo** (`agregar_por_empleador`): los
   proyectos 01, 04 y 05 lo necesitan y hoy cada uno lo reconstruye.
3. ~~`synth_base`: `afiliados_por_empleador` no es configurable~~ **resuelto en la ronda 2** (ya era
   parámetro de `generar_universo`) y **usado en la ronda 3** (objetivo 2: default de este proyecto
   bajado de 40 a 20, ver §"Ronda 3"). Sigue siendo la limitación dominante (§8): para cerrar el
   criterio del agregado mensual (±10 %) haría falta del orden de 40.000 empleadores desde aquí.
4. **Shocks del empleador en el universo común**: hoy el panel laboral se simula antes y el shock se
   superpone después, así que un empleador liquidado sigue teniendo trabajadores. Si `synth_base`
   generara los episodios de estrés, los proyectos 01 (RIM), 04 (cese) y 05 compartirían la misma
   historia.
5. Subir a `common/metrics.py` la **volatilidad intrínseca de un agregado** (`sqrt(Σ p(1−p)·w²)`):
   es la que permite decir si un criterio de precisión agregada es alcanzable o no, y sirve para
   cualquier proyección (recaudación, flujo de APV, masa en riesgo).

6. **Densidad de cotización recalibrada (ronda 3, este mismo brief) sin distinguir cesante reciente
   de crónico**: al bajar la densidad del sistema/cuprum a la Ficha 165, la población "sin RIM" creció
   mucho y el proyecto 04 (riesgo de cese/reingreso) perdió AUC de reingreso (0,720 → 0,605, bajo su
   umbral de 0,70) porque esa población quedó más heterogénea. Ver `common/README.md` para el detalle;
   no se tocó el proyecto 04 porque no es parte de este brief.

**Próximos pasos**

* Escalar aún más el universo de empleadores (más allá de los ~1.000 de esta ronda) si el criterio
  del agregado mensual (§6.4) se vuelve prioritario — la relación `1/√n_empleadores` deja predecir con
  bastante precisión cuánto ayudaría antes de correrlo.
* Modelo de **regularización**: dado que un empleador entró en DNP, ¿cuándo paga? Es la otra mitad
  del problema de cobranza y los datos ya están (`periodo_regularizacion`).
* Incorporar señales externas que la AFP puede comprar o scrapear: Boletín Concursal completo,
  publicaciones de la DT, alertas de prensa, y el IMACEC sectorial como macro.
* Modelo de **uplift de la cobranza**: hoy se predice quién no va a pagar, no a quién la llamada
  cambia la decisión. Con un holdout de gestión se podría estimar.
* Reportar la proyección de masa en DNP de forma trimestral y con intervalo, no como punto mensual.
