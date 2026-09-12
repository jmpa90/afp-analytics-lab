# 05 · Riesgo de empleador: declaración y no pago (DNP) y pago tardío

Modelo de cobranza y de proyección de recaudación. Para cada **empleador** y cada mes de decisión
estima la probabilidad de que la planilla de ese mes no se pague y **cuánta masa imponible de
nuestros afiliados está en riesgo**, para que el 10 % de empleadores que cobranza puede gestionar
sea el que concentra más masa.

```bash
python -m afp_lab.riesgo_empleador.run --n-afiliados 20000    # corrida de referencia, ~27 s
python -m pytest afp_lab/riesgo_empleador/tests -q            # 21 tests, ~14 s
```

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

### 2.1 Tablas (corrida de referencia, 20.000 afiliados → 500 empleadores)

| Tabla | Grano | Filas | Contenido |
|---|---|---|---|
| `planillas` | empleador × periodo | 39.790 | n_trabajadores, masa imponible, cotización, `periodo_recepcion`, `lag_recepcion`, `estado` (PAGADA / DNP / SIN_DECLARAR), `periodo_regularizacion`, `n_rectificaciones`, altas/bajas/licencias, % plazo fijo |
| `boletin_infractores` | empleador × mes | 1.849 (196 empleadores) | inclusión en el Boletín de Infractores Laborales y Previsionales (DT) |
| `boletin_concursal` | evento | 94 | REORGANIZACION / LIQUIDACION (Superir) con `periodo_publicacion` |
| `shocks_empleador` | latente | 411 episodios (55 liquidaciones) | episodios de estrés financiero. **Nunca es feature** (hay test) |

Ventana **201901–202609** (93 meses); los 12 primeros son burn-in del historial de pago.
Semilla fija, generación determinista (hay test).

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

### 2.3 Tasas resultantes vs. las pedidas en el brief

| Indicador | Brief | Corrida 20.000 |
|---|---|---|
| planillas del mes con deuda (DNP + no declarada) | 3–6 % | **5,64 %** (DNP 4,00 % + no declarada 1,64 %) |
| planillas con recepción tardía (lag ≥ 2 o nunca) | 5–10 % | **7,65 %** |
| AUC de test | 0,75–0,90 | **0,818** |

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

Con 20.000 afiliados quedan **~400 empleadores elegibles por corte** de los 500 del universo.

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
| 202508 | 202001–202309 (45) | 202403–202502 (12) | 202508–202510 | 19.557 | 6,12 % / 6,24 % / 3,59 % |
| 202511 | 202001–202312 (48) | 202406–202505 (12) | 202511–202601 | 20.796 | 6,14 % / 5,90 % / 3,93 % |
| 202602 | 202001–202403 (51) | 202409–202508 (12) | 202602–202604 | 22.040 | 6,17 % / 5,16 % / 5,88 % |
| 202605 | 202001–202406 (54) | 202412–202511 (12) | 202605–202607 | 23.269 | 6,19 % / 4,66 % / 5,90 % |

Test: **4.771 filas, 12 cortes, 230 empleadores-mes con DNP**.

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
* **número de árboles por early stopping** contra el `valid` del origen: **74 / 69 / 146 / 56** sobre
  un tope de 3.000, que nunca se toca;
* **calibración isotónica** sólo si la mejora del Brier fuera de muestra dentro de validación supera
  2 errores estándar de la dispersión entre folds **y** el 1 % del Brier: en esta corrida **ningún
  origen la aplica** (mejoras entre −4,4e-4 y +8,7e-5, todas por debajo de su error estándar);
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
| train (en muestra) | 0,879 |
| **CV agrupada por empleador** (`GroupKFold`, dentro de train) | **0,849** |
| validación temporal con gap | 0,849 |

La CV agrupada coincide con la validación temporal: el modelo generaliza a empleadores que no vio, y
no está memorizando los 500 del universo.

---

## 6. Resultados (test = 12 cortes 202508–202607; 4.771 filas; 230 DNP; tasa base 4,82 %)

### 6.1 Clasificación del target principal

| Predictor | AUC | Average precision | Brier | Lift decil 1 | KS |
|---|---|---|---|---|---|
| **Modelo** | **0,818** | **0,331** | **0,0374** | **5,39** | **0,484** |
| Persistencia | 0,725 | 0,098 | 0,0442 | 2,40 | 0,453 |
| Regla de negocio | 0,669 | 0,141 | 0,0420 | 4,10 | 0,348 |
| Tasa base | 0,500 | 0,048 | 0,0461 | 1,00 | 0,118 |

Por corte: AUC de **0,741 a 0,911**, lift de 3,63 a 7,45.

### 6.2 Los tres targets

| Target | Tasa base | AUC | AP | Lift decil 1 |
|---|---|---|---|---|
| `y_dnp` (principal, 1 mes) | 4,82 % | **0,818** | 0,331 | 5,39 |
| `y_dnp_3m` (alguna DNP en 3 meses) | 9,56 % | 0,766 | 0,344 | 3,66 |
| `y_tardio` (lag ≥ 2 o nunca) | 7,38 % | 0,689 | 0,158 | 2,47 |

`y_tardio` es el más difícil y tiene sentido: el atraso de uno o dos meses es en buena parte ruido
operativo del empleador (cuándo alcanzó a pagar), mientras la DNP es un estado financiero con
persistencia. `y_dnp_3m` pierde AUC porque diluye la señal de "este mes" en tres meses, pero su
average precision es la más alta: para una gestión trimestral es el target correcto.

### 6.3 Métrica de negocio: masa imponible en riesgo capturada

| Score de gestión | 10 % gestionado | 20 % gestionado |
|---|---|---|
| **Modelo (`masa_en_riesgo_esperada`)** | **54,5 % de la masa** / 28,7 % de los DNP / precisión 13,7 % | **72,5 %** / 40,9 % / 9,8 % |
| Persistencia | 47,6 % / 7,8 % / 3,7 % | 64,6 % / 16,5 % / 4,0 % |
| Regla de negocio | 42,5 % / 8,7 % / 4,1 % | 65,3 % / 19,6 % / 4,7 % |
| Modelo (sólo probabilidad) | 30,0 % / 53,0 % / 25,3 % | 37,2 % / 65,7 % / 15,7 % |
| Masa sola (ordenar por tamaño) | 33,7 % / 2,6 % / 1,2 % | 56,7 % / 7,8 % / 1,9 % |
| Aleatorio | 7,2 % / 9,1 % / 4,6 % | 22,2 % / 19,5 % / 5,6 % |

Gestionando al **10 % de los empleadores** (≈ 40 al mes) se cubre el **54,5 % de la masa imponible en
riesgo**, contra 42,5 % de la regla del brief y 47,6 % de la persistencia: **+12,0 pp sobre la regla**
(lift 1,28×) con una precisión 3,3× mejor.

Nótese el contraste entre los dos scores del modelo: ordenar por **probabilidad** encuentra el 53 %
de los empleadores morosos pero sólo el 30 % de la masa (son micro empresas); ordenar por
**probabilidad × masa** encuentra la mitad de los morosos pero la mayor parte del dinero. Para
cobranza —cuyo objetivo es CLP recuperados— el segundo es el correcto; si el objetivo fuera "número
de casos gestionados" habría que usar el primero.

**Advertencia sobre la precisión de esta cifra.** La captura mensual tiene una volatilidad enorme
(desviación estándar 0,22 entre meses, de 5,7 % a 93,9 %) porque la masa en riesgo de un mes la
determinan una o dos caídas grandes. Con 12 meses el error estándar de la media es ≈ **6,4 pp**, y
una corrida idéntica con 24 cortes de test da **44,5 %** en vez de 54,5 %. La conclusión robusta es
"el modelo captura del orden de 45–55 % de la masa en riesgo con el 10 % de la cartera, y le gana a
la regla en los dos casos"; el 54,5 % no debe tomarse como un punto.

### 6.4 Proyección del agregado mensual: el criterio no se cumple, y es un problema de escala

| Indicador | Valor |
|---|---|
| Meses con error ≤ 10 % | **0 de 12** |
| Meses dentro de **1 desviación estándar intrínseca** | **11 de 12** |
| Volatilidad intrínseca media del agregado | **98,9 %** |
| Error absoluto medio / mediano | 0,97 / 0,59 |
| Sesgo agregado de los 12 meses juntos | +16,8 % |

El criterio "error del agregado mensual ≤ 10 % en ≥ 10 de 12 meses" **no es alcanzable a esta
escala**, y se puede demostrar: incluso con la probabilidad *exacta*, la masa en riesgo realizada de
un mes es una suma de Bernoulli ponderada por masas muy desiguales, cuya desviación estándar es
`sqrt(Σ p(1−p)·masa²)`. Esa cantidad —calculada con las probabilidades del propio modelo— es en
promedio el **98,9 % del valor esperado**. Es decir: el ruido irreducible del agregado mensual es de
±99 %, diez veces la tolerancia pedida. Con 500 empleadores y ~20 DNP al mes, el agregado lo deciden
una o dos empresas.

Lo que sí se cumple es la versión honesta del criterio: **11 de 12 meses caen dentro de 1 σ
intrínseca** y el agregado de los 12 meses juntos tiene +16,8 % de sesgo. Para llegar a ±10 %
mensual harían falta del orden de `(0,989/0,10)² ≈ 100` veces más empleadores (≈ 50.000, que es el
orden de magnitud de la cartera real de empleadores de una AFP). La proyección de recaudación
debería reportarse **trimestral o semestral**, no mensual, mientras el universo simulado tenga este
tamaño.

### 6.5 Calibración

| Versión | Brier | ECE | Sesgo agregado |
|---|---|---|---|
| **Probabilidad usada (ningún origen calibrado)** | **0,0374** | **0,0064** | +3,2 % |
| Isotónica por origen (referencia, no se usa) | 0,0381 | 0,0063 | −9,8 % |

### 6.6 Por segmento (AUC): el modelo gana en 2 de los 4 tramos de tamaño

| Tamaño | % filas | DNP | Modelo | Persistencia | Regla |
|---|---|---|---|---|---|
| MICRO | 37,0 % | 131 | **0,831** | 0,712 | 0,692 |
| PEQUEÑA | 39,8 % | 83 | **0,781** | 0,694 | 0,645 |
| MEDIANA | 16,6 % | 13 | **0,651** | 0,569 | 0,562 |
| GRANDE | 6,5 % | **3** | 0,447 | **0,636** | 0,494 |

**Incumplimiento de CONVENCIONES §4**: en el tramo GRANDE (6,5 % de las filas, por encima del 5 %)
el modelo pierde contra la persistencia (0,447 vs 0,636). La causa es el tamaño de la muestra: hay
**3 empleadores-mes con DNP** en 312 filas de test, así que el AUC de ese segmento no es
interpretable (un caso cambia el resultado en decenas de puntos). Y es consecuencia directa de la
especificación del brief —"cae en grandes empresas y sector público (nunca DNP)"—: por construcción
casi no hay casos grandes de los que aprender. Lo honesto es decir que **para el tramo GRANDE este
modelo no aporta** y que ahí la gestión debe seguir siendo por excepción (boletines, alertas de
prensa, seguimiento comercial), no por score.

Por rubro hay 10 rubros con ≥ 5 % de las filas; uno de ellos (`O_ADMIN_PUBLICA`, 456 filas) tiene
**0 casos de DNP** —el sector público nunca deja de pagar, y la tabla lo dice explícitamente en vez
de dejar la celda vacía—. De los 9 restantes el modelo gana a los dos baselines en **8** (AUC
0,626–0,848) y pierde sólo en `C_MANUFACTURA` (0,821 vs 0,832 de la regla, 19 casos).

### 6.7 Importancias (gain, %) y ejemplos

`meses_deuda_consecutivos` **50,8** · `meses_desde_ult_dnp` 10,6 · `n_dnp_3` 5,0 · `n_dnp_6` 4,7 ·
`racha_puntualidad` 3,9 · `meses_en_boletin_12` 2,8 · `n_dnp_pendientes` 2,1 · `rubro` 2,0 ·
`masa_ref_imm` 1,7 · `masa_dnp_pendiente_imm` 1,6 · `n_faltantes_12` 1,3 · `tasa_puntualidad_12` 1,3.

Es exactamente el mecanismo simulado: la **persistencia** de la deuda y la **fragilidad estable** del
empleador (que el historial de 12 meses revela) explican la mitad del modelo; el boletín y el rubro
aportan el resto.

* **Empleador 313 (Q_SALUD, PEQUEÑA, corte 202606)** — `prob 0,54`, masa esperada 47,4 MM; real:
  **no pagó, 85,9 MM en riesgo**. Empujan: 4 meses consecutivos de deuda (+1,86), deuda el mes
  anterior (+0,44), 3 DNP en 6 meses (+0,24) y 4 DNP pendientes de regularizar (+0,22). Es el caso
  de libro: un episodio de estrés en curso.
* **Empleador 474 (B_MINERIA, MEDIANA, corte 202605)** — `prob 0,11`, masa esperada 64,6 MM; real:
  **pagó**. Un mes de deuda reciente lo sube (+1,13) pero el rubro minero (−0,19) y su tamaño lo
  bajan. Falso positivo caro: es el tipo de caso que hace que la captura de masa sea volátil.
* **Empleador 123 (K_FINANZAS, GRANDE, corte 202605)** — `prob 0,012`. 12 meses de racha de
  puntualidad (−0,12), sin deuda (−0,17), 1.117 trabajadores (−0,21). El modelo lo descarta bien,
  pero su masa es tan grande que igual aparece arriba del ranking de masa esperada.

---

## 7. Lectura de negocio

**Con la capacidad actual de cobranza —10 % de los empleadores al mes, ~40 llamadas— el modelo cubre
el 54,5 % de la masa imponible en riesgo, contra 42,5 % con la regla vigente: 12 puntos más de masa
gestionada sin gastar un peso más.** En la corrida de referencia la masa imponible en riesgo es de
**398 MM CLP al mes** en promedio, así que los 12 puntos son **~48 MM CLP al mes** de masa adicional
bajo gestión (~573 MM en el año), sin gastar un peso más. Y la
precisión de la lista sube de 4,1 % a 13,7 %: de cada 8 llamadas una cae en un empleador que
efectivamente no va a pagar, contra 1 de cada 24 con la regla.

Tres lecturas operativas:

* **Dos listas, no una.** Ordenar por masa esperada maximiza los CLP (54,5 % de la masa, 28,7 % de
  los morosos); ordenar por probabilidad maximiza los casos (53,0 % de los morosos, 30,0 % de la
  masa). Si el equipo de cobranza tiene un tramo telefónico masivo y otro de ejecutivo, el primero
  debería trabajar la lista por probabilidad y el segundo la lista por masa.
* **El tramo GRANDE no se gestiona con este modelo** (§6.6): 3 casos en 12 meses. Ahí la alerta útil
  es el boletín concursal y la señal de deterioro (`deterioro_lag`, `frac_bajas_3`), como excepción.
* **La proyección de recaudación es trimestral, no mensual.** El agregado mensual tiene ±99 % de
  ruido irreducible a esta escala (§6.4); comprometerse a un número mensual sería vender algo que no
  existe.

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

**Supuestos que habría que confirmar con la AFP**

* La planilla del mes *m* se recibe en *m+1* si el empleador es puntual, y lo que cobranza ve el mes
  `corte` son las recepciones hasta `corte` (no hay un feed intradía).
* El Boletín de Infractores se publica con un mes de rezago y se puede consultar como dato del mes.
* Las publicaciones del Boletín Concursal (reorganización, liquidación) están disponibles el mes
  siguiente a su publicación.
* La masa imponible de referencia para dimensionar el riesgo es la de la última planilla recibida
  (no hay una estimación previa de la planilla en curso).

**Limitaciones**

* **Escala**: 20.000 afiliados dan **500 empleadores** (`afiliados_por_empleador = 40` en
  `rim_nowcasting.config`). Una AFP real tiene decenas de miles. Eso limita el AUC de los segmentos
  chicos, hace inalcanzable el criterio del agregado mensual (§6.4) y le pone ±6 pp de error estándar
  a la métrica de negocio. Es la limitación dominante del proyecto.
* El panel laboral se genera **antes** que los shocks del empleador, así que un empleador liquidado
  sigue teniendo trabajadores en el panel; se resuelve sacándolo del universo de planillas 3 meses
  después de la liquidación, pero la coherencia perfecta exigiría simular las dos cosas juntas
  (propuesta para `common/`).
* No se modelan la cobranza judicial ni los convenios de pago, que en la realidad cambian la
  probabilidad de regularización.
* `y_tardio` (AUC 0,689) es débil: el atraso de uno o dos meses es mayormente ruido operativo.
* El modelo no aporta en el tramo GRANDE (§6.6).

---

## 9. Cómo correr y tiempos

```bash
cd /home/user/wt/empleador
PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.riesgo_empleador.run --n-afiliados 20000
PYTHONPATH=/home/user/wt/empleador python3 -m pytest afp_lab/riesgo_empleador/tests -q
```

Opciones: `--perfil {sistema,cuprum}` (por defecto `cuprum`), `--seed`, `--primer-corte`,
`--cada-origen`, `--sin-diagnostico`, `--salida`.

| Etapa | Segundos (20.000 afiliados) |
|---|---|
| Generación (universo + planilla Previred + capa de recepción) | 18,5 |
| Dataset as-of (66 cortes, 26,2 k filas) | 0,6 |
| Entrenamiento (4 orígenes × 3 targets) | 5,5 |
| Evaluación, importancias, ejemplos y CV agrupada | 1,5 |
| **Total** | **27,0** (límite: 300) |

El 68 % del tiempo es la planilla línea a línea de `common` (2,3 M de líneas para 20.000 afiliados);
el modelo propiamente tal es barato porque el dataset de empleadores es chico.

Tests: **21 tests en ~14 s** (límite: 120). Cubren leakage por truncamiento y por mutación del
futuro en dos cortes, que la planilla del propio periodo de corte no sea visible, ausencia de
columnas latentes y de escala, gaps entre todas las particiones, recálculo a mano de 5 features en
2 empleadores × 2 cortes, población y target, forma del dataset, determinismo, y la coherencia de la
simulación (el sector público nunca cae en DNP, persistencia y estacionalidad de la DNP, el boletín
sólo tras 2 meses de deuda, la masa cuadra con la RIM del panel), hiperparámetros elegidos con
validación, baselines, sesgo por corte, persistencia del modelo y corrida end-to-end.

Salidas en `salidas/riesgo_empleador/`: `fuentes/*.parquet`, `dataset/*.parquet`,
`modelo/origen_<yyyymm>/` (uno por origen), `predicciones_test.parquet` y `reportes/*.csv`.

### Criterios de aceptación del brief

| Criterio | Umbral | Obtenido | ¿Cumple? |
|---|---|---|---|
| AUC de test | ≥ 0,80 | **0,818** (rango por corte 0,741–0,911) | **sí** |
| Captura de masa en riesgo al 10 % | ≥ 50 % | **54,5 %** (± 6,4 pp; 44,5 % con 24 cortes) | **sí, en el límite** |
| Captura al 10 % mejor que la regla | > 42,5 % | **54,5 %** (+12,0 pp) | **sí** |
| Error del agregado mensual ≤ 10 % en ≥ 10 de 12 meses | 10 meses | **0 meses** (11 de 12 dentro de 1 σ intrínseca de ±98,9 %) | **no** — ver §6.4 |
| Runtime con 20.000 afiliados | < 5 min | **27 s** | sí |
| Tests | < 2 min | **21 tests en 14 s** | sí |
| AUC en el rango pedido a la simulación | 0,75–0,90 | 0,818 | sí |
| El modelo gana en todo segmento ≥ 5 % de filas | — | pierde en GRANDE (3 casos) | **no** — ver §6.6 |

---

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/`** (no se modificó nada fuera de mis dos carpetas):

1. **`planilla.py`: el sorteo de recepción es por línea, no por planilla.** En `_aplicar_recepcion`,
   `rng.random(n)` con `n` = número de líneas hace que dos líneas de la misma
   (afiliado, empleador, periodo) puedan tener `periodo_recepcion` distinto, contra lo que dice el
   propio docstring. Sugerencia: sortear por `(empleador, periodo)` y propagar.
2. **`planilla.py` podría exponer un agregado empleador × periodo** (`agregar_por_empleador`): los
   proyectos 01, 04 y 05 lo necesitan y hoy cada uno lo reconstruye.
3. **`synth_base`: `afiliados_por_empleador` no es configurable** desde `generar_universo` (queda
   fijo en 40 por el `Config` interno). Cualquier proyecto con grano de empleador se queda con
   `n_afiliados/40` empleadores, que es la limitación dominante de este proyecto (§8). Sugerencia:
   pasar el parámetro.
4. **Shocks del empleador en el universo común**: hoy el panel laboral se simula antes y el shock se
   superpone después, así que un empleador liquidado sigue teniendo trabajadores. Si `synth_base`
   generara los episodios de estrés, los proyectos 01 (RIM), 04 (cese) y 05 compartirían la misma
   historia.
5. Subir a `common/metrics.py` la **volatilidad intrínseca de un agregado** (`sqrt(Σ p(1−p)·w²)`):
   es la que permite decir si un criterio de precisión agregada es alcanzable o no, y sirve para
   cualquier proyección (recaudación, flujo de APV, masa en riesgo).

**Próximos pasos**

* Escalar el universo de empleadores (propuesta 3) y repetir: es lo único que mueve los criterios
  que no se cumplen.
* Modelo de **regularización**: dado que un empleador entró en DNP, ¿cuándo paga? Es la otra mitad
  del problema de cobranza y los datos ya están (`periodo_regularizacion`).
* Incorporar señales externas que la AFP puede comprar o scrapear: Boletín Concursal completo,
  publicaciones de la DT, alertas de prensa, y el IMACEC sectorial como macro.
* Modelo de **uplift de la cobranza**: hoy se predice quién no va a pagar, no a quién la llamada
  cambia la decisión. Con un holdout de gestión se podría estimar.
* Reportar la proyección de masa en DNP de forma trimestral y con intervalo, no como punto mensual.
