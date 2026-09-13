# Nowcasting de Renta Imponible (RIM) para afiliados de una AFP — PySpark

Modelo que **proyecta la Renta Imponible Mensual de cada afiliado para los 2 meses que la AFP aún no conoce**
(la cotización llega ~2 meses después de la remuneración) y **reemplaza la proyección por el dato real** cuando
el empleador declara/paga. Todo el pipeline corre sobre PySpark (tablas de decenas de millones de filas) y se
simula aquí con datos sintéticos realistas.

```
python -m afp_lab.rim_nowcasting.pipeline demo --n-afiliados 5000 --salida /tmp/afp_lab.rim_nowcasting
```

---

## 1. El problema

| Concepto | Definición usada en el código |
|---|---|
| `periodo_snapshot` (T) | Mes en que corre el modelo ("hoy"). |
| Información conocida | Movimientos con `periodo_recepcion <= T-1` (todo lo recibido hasta fin del mes anterior). |
| `periodo_corte` | `T - meses_desfase` = **T-2**: último mes de remuneración completamente conocido. El pago normal del mes *m* llega en *m+1*, se procesa y queda disponible en *m+2*. |
| Meses abiertos | `corte+1` (h=1, es T-1) y `corte+2` (h=2, es T). Son **los 2 únicos meses que se proyectan**. |
| Ciclo mensual | En T+1 llegan los pagos del mes T-1 → la fila PREDICHA se reemplaza por REAL; el mes T se re-proyecta con h=1; aparece un nuevo mes T+1 con h=2. |

La RIM real se define a partir de los **movimientos** que recibe la AFP (planillas Previred, SII, subsidios):
por cada `(afiliado, periodo, entidad pagadora, empleador)` vale el último movimiento recibido (una
**rectificación** reemplaza a la declaración), se suma sobre pagadores y se aplica el **tope imponible** del
periodo. Una **DNP** (declaración y no pago) informa RIM igual; el pago posterior no la cambia.
Esa misma función (`snapshot.rim_conocida`) sirve para la "verdad final" (`hasta_recepcion=None`) y para la
"verdad conocida en un snapshot", que es la base de todas las features. **Nunca se mira la verdad latente.**

## 2. Datos: qué tablas tiene la AFP y cómo se simulan

| Tabla | Grano | Fuente real | Uso |
|---|---|---|---|
| `cotizaciones` (movimientos) | afiliado × periodo × pagador × movimiento | Previred / recaudación (DECLARACION, RECTIFICACION, PAGO_DNP; `estado_pago` PAGADA/DNP; `periodo_recepcion`) | Verdad as-of, lags, comportamiento de pago del empleador |
| `afiliados` | afiliado | Maestro de afiliados | Sexo, edad, región, tipo (DEPENDIENTE / INDEPENDIENTE / VOLUNTARIO / PENSIONADO_ACTIVO), nivel educacional, fecha afiliación |
| `empleadores` | empleador | Maestro de empleadores (RUT, CIIU, tamaño) | Rubro, tamaño, región |
| `afc_eventos` | afiliado × periodo × evento | Seguro de Cesantía (AFC): inicio/término de relación laboral (el empleador avisa en ~10 días) | **Señal adelantada** de cesantía / nuevo empleo |
| `licencias` | afiliado × periodo | COMPIN / ISAPRE (licencias médicas) | Subsidio: parte de la RIM llega por otro pagador y con más rezago |
| `macro` | periodo | Banco Central / INE / SP | UF, tope imponible (UF y CLP), IMM, IPC 12m, desempleo |
| `verdad_mensual` | afiliado × periodo | *(sólo simulación)* | Verdad latente para generar movimientos; **no se usa en el modelo** |

### Dinámicas que reproduce el generador (`synthetic.py`)

* **Empleo**: cadena de estados (empleado / cesante / licencia / retirado) con probabilidades de término y
  contratación por rubro y mes (temporeros en agricultura Nov–Mar, retail en Nov–Dic, educación con
  términos en Ene–Feb y contrataciones en Mar, turismo en verano), contratos plazo fijo con mayor rotación,
  antigüedad que reduce la rotación, edad de retiro 60/65.
* **Sueldo**: log-normal por productividad × rubro × empleador, piso legal en el **IMM** (con masa de
  afiliados exactamente en el mínimo), **reajustes** anuales/semestrales por IPC (sector público en
  diciembre), variación del IMM (mayo/julio/enero).
* **Componentes imponibles**: gratificación legal mensual (25 % con tope 4,75 IMM) o anual (abril),
  aguinaldos de Fiestas Patrias y Navidad, remuneración variable por rubro (comisiones en comercio, turnos
  en salud, bonos en minería/finanzas en marzo), horas extra, meses parciales al entrar/salir.
* **Tope imponible** (81,6 → 89,9 UF) aplicado por pagador y en la suma.
* **Múltiples empleadores** (5 %), **licencias médicas** (la ISAPRE/FONASA paga el subsidio con 2–4 meses
  de rezago), **independientes** (toda la renta del año llega vía SII en junio del año siguiente),
  **voluntarios** (pagos esporádicos), **pensionados que siguen cotizando**.
* **Comportamiento de pago**: por tamaño de empleador, pago anticipado (0–6 %), oportuno m+1 (84–97 %),
  tardío m+2, rezagos m+3..m+8, **DNP** (0–6 %) regularizada meses después, **rectificaciones** (1–2 %).
* **Afiliados nuevos** durante la ventana (cold start) y valores macro aproximados 2022–2026.

Todo se genera con expresiones Spark (`spark.range` + `rand`) y `mapInPandas` (simulación numpy por
partición): sin colecciones al driver ni cross joins, por lo que escala a millones de afiliados cambiando
`n_afiliados` y `particiones_simulacion`.

## 3. Variables del modelo (features "as-of")

Todas se calculan con `construir_features(snapshot)` usando **sólo** movimientos con
`periodo_recepcion <= snapshot-1`. En producción esta tabla se calcula una vez al mes y se persiste
particionada por `periodo_snapshot`; el set de entrenamiento es la unión de snapshots pasados con la RIM
real que llegó después (nunca se recalculan features "hacia atrás", lo que elimina el leakage por
construcción).

| Familia | Features | Por qué importa |
|---|---|---|
| Historia propia | `rim_l1..rim_l13` (l1 = mes de corte), `media_3/6/12`, `std_6`, `cv_6`, `max_12`, `min_12`, `n_cotiza_3/6/12` (densidad), `racha_cotiza`, `meses_historia`, `meses_desde_ult_cotiza`, `meses_desde_cambio`, `n_cambios_12`, `sueldo_fijo`, `ratio_l1_l2`, `ratio_l1_media12`, `tendencia_3_12`, `al_tope_l1`, `en_imm_l1`, `dist_imm_l1` | La RIM es muy persistente; densidad, rachas y tiempo desde el último cambio capturan estabilidad laboral; los flags de tope e IMM marcan los tramos que se mueven por ley |
| Referencia | `rim_ref` (última RIM > 0 → mediana del empleador → IMM, con `ref_tipo`), `rim_ref_sobre_imm`, `rim_ref_sobre_tope`, `rim_mismo_mes_ly` (valor en target-12), `ratio_ly_ref`, `ratio_estacional_ly` (salto target-12 / target-13) | El target del regresor es `log(RIM_target / rim_ref)`; el mismo mes del año anterior captura aguinaldos, gratificación anual, bono de marzo |
| Relación laboral | `meses_con_empleador`, `n_empleadores_12`, `n_pagadores_l1`, `tiene_subsidio_l1`, `tiene_dnp_l1`, `tipo_contrato` (informado a la AFC al iniciar la relación) | Rotación, multiempleo y contratos a plazo |
| Señales adelantadas | `afc_termino_l1/l2`, `afc_inicio_l1/l2`, `meses_desde_afc_termino/inicio`, `dias_licencia_l1/l2`, `licencia_en_curso`, `rim_propia_conocida_h1` / `flag_propia_conocida_h1` (sólo filas h=2: el propio mes corte+1 ya recibido) | Un término AFC en el mes de corte implica RIM 0 el mes siguiente aunque el corte tenga RIM > 0; si la relación empezó en el corte y aún no hay planilla, el empleador del aviso AFC sirve de referencia (cold start) |
| Empleador | `rubro`, `tamano_empleador`, `emp_n_trabajadores`, `emp_crecimiento_12`, `emp_rim_mediana`, `pos_rel_empleador`, `emp_tasa_oportuno`, `emp_frac_declarado_corte`, `emp_frac_declarado_h1`, `emp_ratio_recibido_h1`, `sin_declaracion_con_emp_activo` | Si el empleador ya declaró el mes para el 90 % de sus trabajadores y no para este afiliado, es más probable un finiquito que un pago tardío; los compañeros ya recibidos para el mes abierto anticipan reajustes y aguinaldos de ese empleador |
| Afiliado | `tipo_afiliado`, `sexo`, `edad`, `meses_hasta_pension`, `region`, `nivel_educacional`, `meses_desde_afiliacion` | Segmentos con dinámicas distintas (independientes, voluntarios, pensionados, cold start) |
| Calendario y macro | `horizonte`, `mes_target`, `es_enero/marzo/abril/julio/sept/dic`, `imm_target`, `tope_target`, `ratio_imm_target_corte`, `ipc_12m_corte`, `desempleo_corte` | Estacionalidad legal (aguinaldos, gratificación, reajustes) y cambios del IMM ya legislados |

Filas ya conocidas para un mes abierto (empleadores que pagan anticipado) **no se predicen**: entran como
`REAL / PAGO_ANTICIPADO`. Si sólo llegó una parte de los pagadores esperados (multiempleo, subsidio pendiente) la
fila es `PREDICHA / PARCIAL`: se predice el total y se toma al menos lo ya recibido.

## 4. Modelo (`model.py`)

**Hurdle en dos etapas** porque ~25–35 % de los afiliado-mes son 0 (cesantía, informalidad, independientes
sin renta):

1. `GBTClassifier` → `prob_cotiza = P(RIM_target > 0)`.
2. `GBTRegressor` (pérdida absoluta, robusta a outliers) sobre filas con RIM > 0 →
   `y_reg = log(RIM_target / rim_ref)`. Modelar el **ratio** y no el nivel hace que el modelo aprenda
   "cambios" (reajuste, aguinaldo, cambio de empleador) y generalice entre tramos de renta.

Salidas por afiliado y mes abierto:

* `rim_condicional = rim_ref · exp(ŷ)` topada al tope imponible del mes (ŷ winsorizado a [-3, 2] al entrenar).
* `rim_esperada = prob_cotiza · rim_condicional · s_h` → para **agregados** (recaudación proyectada, flujo de
  fondos); `s_h` es un factor de *smearing* por horizonte (Σ real / Σ p·rim_cond en entrenamiento) que corrige el
  sesgo de estimar una mediana con pérdida absoluta.
* `rim_proyectada = rim_condicional si prob_cotiza ≥ τ_h, si no 0` → para el **dato individual** que se
  escribe en la cuenta hasta que llegue el real; `τ_h` se elige por horizonte minimizando el WAPE en
  entrenamiento (nunca en test). Ambos parámetros se guardan con el modelo (`calibracion.json`).

Un horizonte es una feature (`horizonte` ∈ {1, 2}) y cada afiliado aporta dos filas por snapshot, lo que
comparte información entre horizontes sin duplicar modelos. Las categóricas pasan por `StringIndexer`
(`handleInvalid="keep"` para categorías nuevas en producción). En producción se sustituye GBT de MLlib por
`xgboost.spark.SparkXGBRegressor` / SynapseML LightGBM con la misma interfaz `fit/transform`.

## 5. Evaluación (`evaluate.py`, `pipeline.backtest`)

* **Orígenes móviles**: se entrena con `n_snapshots_entrenamiento` snapshots anteriores a
  `test - gap_entrenamiento` y se evalúa en los `n_snapshots_test` últimos.
* La verdad de entrenamiento es la **conocida al momento de entrenar** (`rim_conocida(hasta = test-1)`) y sólo
  entran filas con **etiqueta madura** (`pipeline.etiqueta_madura`): para dependientes el target debe tener al
  menos `meses_desfase + meses_maduracion` meses (4) para que hayan llegado pagos tardíos y subsidios; para
  independientes sólo cuentan los años cuya Operación Renta ya llegó. Sin esta regla los pagos que aún no
  llegan se etiquetan como 0 y el modelo aprende a subestimar (en la primera corrida del demo esto llevó a
  predecir 0 para todos los independientes).
* La verdad de evaluación es la final (lo que terminó llegando).
* **Baselines** que hay que superar: persistencia (`rim_l1`), media de 3 meses, estacional
  (mismo mes del año anterior reajustado por la variación del IMM).
* **Métricas** por horizonte, tipo de afiliado, tramo de renta y snapshot: MAE, RMSE, **WAPE**
  (Σ|e| / Σ|y|, robusta a ceros; el MAPE sólo se reporta condicional a RIM > 0), sesgo agregado (error de la
  masa imponible proyectada), exactitud cero/no-cero, AUC y Brier del clasificador.

## 6. Reemplazo PREDICHA → REAL (`predict.py`, `reconcile.py`)

Tabla `rim_proyectada` (una fila por afiliado y periodo abierto):

| Columna | Descripción |
|---|---|
| `rim_valor` | Valor vigente (predicho o real) |
| `origen` | `PREDICHA` / `REAL` |
| `detalle_origen` | `MODELO`, `PARCIAL`, `PAGO_ANTICIPADO`, `PAGO`, `PAGO_TARDIO`, `RECTIFICACION`, `SIN_COTIZACION` |
| `prob_cotiza`, `rim_condicional`, `rim_esperada` | Salidas del modelo |
| `rim_predicha_previa` | Lo que decía el modelo antes del reemplazo (monitoreo del error) |
| `periodo_snapshot`, `periodo_reemplazo`, `version_modelo`, `fecha_calculo` | Trazabilidad |

Ciclo mensual en el snapshot T+1:

1. `reconciliar`: las PREDICHA cuyo periodo ya tiene cotización recibida pasan a REAL (`PAGO` si llegó en
   m+1, `PAGO_TARDIO` si tardó más); una REAL cuyo monto cambió (rectificación, segundo empleador que pagó tarde)
   se actualiza; un periodo más allá de la ventana de pagos tardíos sin cotización se fija en 0
   (`SIN_COTIZACION`) y si después llega un pago tardío vuelve a actualizarse. Idempotente. (En jerga AFP
   "rezago" es una cotización recibida que no se pudo imputar al afiliado; aquí no se modela.)
2. `actualizar_proyecciones`: MERGE con las nuevas proyecciones — REAL gana a PREDICHA y entre PREDICHAS gana
   el snapshot más nuevo (el h=2 del mes pasado pasa a h=1 con más información).
3. `reporte_reemplazos`: error del modelo sobre las filas recién reemplazadas (MAE, WAPE, sesgo, exactitud
   cero) → monitoreo continuo y disparador de reentrenamiento.

El equivalente en Delta Lake / Iceberg está documentado en `reconcile.py` (`MERGE INTO ... WHEN MATCHED AND
origen='PREDICHA' ... WHEN NOT MATCHED INSERT`).

## 7. Escala (tablas gigantes)

* **Particionado** de `cotizaciones` por `periodo_recepcion` (lo que llega cada mes es un append) y de
  `rim_proyectada` / `features` por `periodo_snapshot`; Z-ORDER/bucketing por `afiliado_id`.
* Lags y rolling con `Window.partitionBy(afiliado_id).orderBy(periodo)` sobre la **grilla densa**
  (afiliado × mes desde su afiliación) construida con `sequence + explode` por fila (sin cross join).
* Dimensiones (`empleadores`, `macro`) se difunden con `broadcast`; las agregaciones por empleador se hacen
  una vez por snapshot.
* Features **incrementales**: cada mes se calcula sólo el snapshot nuevo; el entrenamiento lee snapshots
  persistidos. Skew: los grandes empleadores no son problema porque las windows van por afiliado.
* Entrenamiento distribuido con MLlib (demo) o XGBoost-on-Spark / LightGBM; scoring batch de ~2 × N filas.
* Orquestación mensual (Airflow): `ingesta → reconciliar → features(T) → proyectar(T) → merge → reporte`;
  reentrenamiento trimestral o cuando el error de los reemplazos supera un umbral.

## 8. Cómo correr

```bash
pip install -e .                      # PySpark 4 necesita Java 17/21
python -m afp_lab.rim_nowcasting.pipeline demo --n-afiliados 5000 --salida /tmp/afp_lab.rim_nowcasting
python -m pytest afp_lab/rim_nowcasting/tests -q
```

Salidas en `--salida`: `fuentes/` (tablas sintéticas en parquet), `features/periodo_snapshot=*`,
`modelo/`, `rim_proyectada/` (proyección del snapshot), `rim_proyectada_<T+1>/` (tras el ciclo mensual) y
`reportes/*.csv` con métricas, importancias y reemplazos.

Otros comandos: `generar`, `backtest`, `entrenar`, `proyectar` (usar `--reusar-fuentes` para no regenerar).
Desde la ronda 3, `backtest` también deja `reportes/backtest_*.csv` (métricas globales, clasificador,
por tipo/tramo/snapshot/segmento e importancias) y acepta `--gbt-max-depth` además de `--gbt-iter`,
`--n-train`, `--n-test`, `--n-afiliados` y `--perfil {sistema,cuprum}`, de modo que cualquier corrida a
escala reducida (por ejemplo la de la suite de tests: 300 afiliados, 3/1 snapshots, GBT 5 / prof. 3) se
reproduce con el comando oficial y no con scripts fuera del repo. Para comparar dos versiones del
código, ejecutar cada una desde un directorio que no contenga `afp_lab/` (con `python -m`, el cwd va
antes que `PYTHONPATH` en `sys.path` y tapa el árbol que se quiere medir).

## 9. Estructura

```
afp_lab.rim_nowcasting/
  config.py      parámetros (desfase, horizontes, ventana de entrenamiento, GBT)
  calendario.py  periodos yyyymm y tabla macro (UF, IMM, tope, IPC, desempleo)
  synthetic.py   generador de datos sintéticos (dims + mapInPandas + movimientos con rezago)
  snapshot.py    RIM conocida "as-of", grilla densa, filas ya conocidas de meses abiertos
  features.py    feature engineering sin leakage (una fila por afiliado × horizonte)
  model.py       modelo hurdle GBT (clasificador + regresor de log-ratio)
  evaluate.py    baselines y métricas
  predict.py     scoring → tabla rim_proyectada
  reconcile.py   reemplazo PREDICHA → REAL y MERGE mensual
  pipeline.py    CLI y demo end-to-end
  tests/         pytest con Spark local
```

## 10. Resultados del demo

Corrida de referencia: `python -m afp_lab.rim_nowcasting.pipeline demo` con la configuración por defecto
(5.000 afiliados, 2022-01 a 2026-09, 12 orígenes de entrenamiento, GBT de 40 iteraciones; ~12 min en un
laptop de 4 núcleos). Backtest sobre los snapshots 2026-08 y 2026-09 (19.627 filas afiliado × horizonte).

| Predictor | WAPE h=1 | WAPE h=2 | Sesgo agregado h=1 | Exactitud cero/no-cero h=1 |
|---|---|---|---|---|
| Persistencia (`rim_l1`) | 0,312 | 0,359 | -17,6 % | 0,817 |
| Media 3 meses | 0,318 | 0,353 | -12,0 % | 0,815 |
| Estacional (mismo mes año anterior) | 0,395 | 0,400 | +13,9 % | 0,798 |
| **Modelo, dato individual** (`rim_proyectada`) | **0,196** | **0,258** | -4,1 % | **0,926** |
| **Modelo, agregados** (`rim_esperada`) | 0,226 | 0,287 | **+1,6 %** | 0,718 |

* Clasificador cotiza / no cotiza: AUC 0,969 (h=1) y 0,918 (h=2); Brier 0,055 / 0,094.
* Dependientes (80 % de la cartera): WAPE 0,115 (h=1) y 0,182 (h=2) frente a 0,215 / 0,265 de la
  persistencia; sesgo -2,6 %. Por tramo de renta el modelo gana en todos los tramos y horizontes.
* Umbral calibrado τ = 0,70 en ambos horizontes (un falso positivo cuesta la RIM completa, por eso conviene
  exigir más certeza que 0,5); smearing s ≈ 1,04.
* Puntos débiles esperables: voluntarios (cotizan al azar un 25 % de los meses: la predicción puntual
  correcta es 0 y sólo `rim_esperada` aporta) e independientes (en la simulación la renta anual se redibuja
  cada año; el modelo acierta el nivel del año anterior, WAPE 0,85).
* Features más importantes: posición relativa a la mediana del empleador, sueldo fijo, rubro, coeficiente
  de variación, ratio último/penúltimo mes, meses desde el término AFC, media de 3 meses, desempleo,
  tipo de afiliado, densidad de cotización reciente, edad, días de licencia.

Ciclo mensual simulado (snapshot 2026-10): de las 4.820 filas de agosto predichas por el modelo llegaron
2.834 pagos, 59 declaraciones sin pago (DNP) y 7 complementos de un segundo pagador; el error del modelo
sobre las filas reemplazadas de dependientes fue WAPE 0,125 con sesgo -2,2 % y 95 % de acierto cero/no-cero
(`reportes/reemplazos_202610.csv`). Las 1.923 filas de agosto que siguen `PREDICHA` son cesantes reales
más pagos tardíos aún no recibidos; 10 quedaron `PARCIAL` esperando a un segundo pagador.

## 10b. Ronda 2: perfil Cuprum

`python -m afp_lab.rim_nowcasting.pipeline demo --perfil cuprum` genera la cartera con el perfil de
`common.synth_base.PERFILES["cuprum"]` (renta 1,7× el sistema, 7 % de cotizantes en el tope, más
finanzas/minería/salud, densidad ~70 %). Backtest en los mismos snapshots (2026-08 y 2026-09):

| | Sistema | Cuprum |
|---|---|---|
| WAPE dependientes h=1 (modelo / persistencia) | 0,115 / 0,215 | 0,115 / 0,211 |
| WAPE dependientes h=2 (modelo / persistencia) | 0,182 / 0,265 | 0,221 / 0,272 |
| WAPE tramo > $2 M h=1 (modelo / persistencia) | 0,135 / 0,216 | 0,150 / 0,246 |
| AUC cotiza / no cotiza h=1 / h=2 | 0,969 / 0,918 | 0,968 / 0,919 |
| Tasa de cotización real (h=1) | 0,72 | 0,62 |
| Ciclo mensual: WAPE de lo reemplazado (dependientes, PAGO) | 0,125 | 0,126 |

Lectura: a un mes el modelo rinde igual en ambas carteras; a dos meses pierde algo en Cuprum porque la
remuneración variable (bonos de finanzas y minería, gratificación anual) pesa más y se conoce con más
rezago. El tope imponible importa más (7 % de cotizantes topados): el error en el tramo > $2 M sigue
siendo el menor en términos relativos. La masa imponible proyectada por mes sube de ~4.500 a ~6.400
millones para los mismos 5.000 afiliados.

## 10c. Ronda 3: código de movimiento de personal (Tabla N° 7)

`afp_lab/common/planilla.py` (ronda 2) ya generaba una planilla Previred completa, pero a grano
línea/concepto y sin que este proyecto la usara. Esta ronda le agrega a **la propia tabla
`cotizaciones`** de 01 (grano afiliado × período × entidad pagadora × movimiento) los campos de la
Tabla N° 7 del Compendio/Previred (ver `docs/normativa_tablas_afp.md`), reusando las mismas
constantes de `common.planilla` en vez de reescribirlas.

**Qué se agregó**

1. `synthetic.generar_movimiento_personal(cfg, verdad)`: `cod_movimiento` (0,1,2,3,4,7,11,12 — los
   mismos omitidos que `common.planilla`, 5/6/13, más el 8 que tampoco se genera aquí, ver
   limitación), `dias_trabajados` y `fecha_desde`/`fecha_hasta`, calculados con expresiones Spark
   (no `mapInPandas`) sobre el panel de verdad (`evento_afc`, `dias_licencia`), importando
   `P_PERMISO_SIN_GOCE`/`P_AUSENTISMO` directamente de `common.planilla` (mismos valores, no
   redefinidos). `generar_cotizaciones` adjunta estas columnas a la línea `EMPLEADOR` (vínculo
   principal) y `SUBSIDIO` del mismo período; la línea del segundo empleador (multiempleo) queda con
   código 0 fijo (igual que `common.planilla._lineas_segundo_empleador`); SII (independientes) y
   AFILIADO (voluntarios) quedan sin código (no declaran Previred). Una `RECTIFICACION` reemplaza el
   código por 12 (reliquidación), igual convención que `common.planilla`. Todo viaja con el mismo
   `periodo_recepcion` que ya gobierna la RIM, así que cualquier feature as-of que respete
   `periodo_recepcion <= hasta_recepcion` no puede ver estos campos antes de tiempo (tests abajo).
2. `features._features_movimiento`: 14 features nuevas — `dias_trabajados_l1`; `mov_termino_l1/l2`,
   `mov_subsidio_l1/l2`; `n_mov_termino_3/6/12`, `n_mov_subsidio_3/6/12`, `n_mov_otros_3/6/12`
   (permiso sin goce + ausentismo). Complementan (no duplican) a `afc_termino_l1`/`dias_licencia_l1`:
   estos últimos llegan por un feed más rápido (m+1), mientras que el código de movimiento viaja en
   la propia planilla y puede demorar lo mismo que la RIM (oportuno, tardío, rezago o DNP).
3. 7 tests nuevos (`afp_lab/rim_nowcasting/tests/test_afp_rim_forecast.py`): dominio de códigos
   válidos; sólo `EMPLEADOR`/`SUBSIDIO` declaran código; coherencia con `evento_afc` (INICIO/TERMINO
   del panel de verdad, con `left_anti` porque no toda combinación produce una línea en
   `cotizaciones` — una licencia de mes completo hace que esa línea se redondee a RIM 0 y
   `generar_cotizaciones` la descarte, lógica previa a esta ronda); `dias_trabajados` coherente con
   el código; `fecha_desde`/`fecha_hasta` cubren el período declarado; respeto de `periodo_recepcion`
   (agregar una declaración con recepción futura no cambia las features, mismo patrón que
   `test_rim_conocida_respeta_recepcion`); rango y consistencia interna de las nuevas features
   (ventanas 3/6/12 acumulativas). Se extendió además `test_features_sin_leakage` con las columnas
   nuevas.

**Limitación propia de esta ronda** (no de `common/planilla.py`): no se genera el código 8 (cambio
de contrato de plazo fijo a indefinido) porque requiere encadenar el estado a través de meses
consecutivos del mismo vínculo — la continuidad por afiliado que la implementación evita a
propósito (ver docstring de `generar_movimiento_personal`).

**Otras limitaciones de `_features_movimiento`**: (a) `dias_trabajados_l1` se rellena con 30 ("sin
movimiento", el valor por defecto de `generar_movimiento_personal` con código 0) cuando no hay
declaración vigente del vínculo principal en el mes de corte — incluido el caso de una licencia
médica de mes completo (`dias_licencia == 30`): ahí `generar_cotizaciones` descarta la línea
`EMPLEADOR` (RIM redondeada a 0, decisión de diseño anterior a esta ronda) y la feature no puede
distinguirlo de un mes completo trabajado; la señal queda en `dias_licencia_l1`/`licencia_en_curso`
(feed de licencias, m+1), no en el código de movimiento. Caso de baja frecuencia, documentado también
en el docstring de `test_cod_movimiento_coincide_con_evento_afc`. (b) Sólo se mira el vínculo
principal (`empleador_principal` de `serie_conocida`): el código del segundo empleador (siempre 0 en
el generador) no aporta.

**Antes / después: tests del proyecto**

| | Antes (rama base) | Después (esta ronda) |
|---|---|---|
| Tests | 11 | 18 (+7 nuevos) |
| Resultado | 11 passed | **18 passed** en una sola corrida completa de punta a punta (pasada de corrección; en la primera entrega fueron 17 passed + 1 test corregido y reverificado aislado) |
| `driver_memory` del fixture `spark` (tests) | 3 g | 6 g (subido: las features nuevas amplían el plan de `construir_features`, y los tests que llaman con `materializar=False` — sin el checkpoint que sí usa producción — canonicalizaban ese plan más veces; con 3 g el último test del archivo llegó a `OutOfMemoryError` en Catalyst. Se agregó además `.cache()` al resultado de `_features_movimiento`, igual patrón que `serie`/`abiertas`, para acotar el tamaño del plan. Contribuye también a la presión de memoria que esos tres cachés (`serie`, `abiertas`, `f_mov`) sólo se liberan con `.unpersist()` en la rama `materializar=True` de `construir_features`: los tests que llaman con `materializar=False` dejan ahora tres cachés vivos por llamada, en vez de dos, en una sesión Spark que dura toda la suite — no se liberan ahí porque el DataFrame devuelto sin checkpoint sigue dependiendo de ellos) |
| Tiempo total suite | ~13,7 min (11 tests) | 1380 s / 23,0 min (18 tests, corrida final, con otro workstream usando la máquina en paralelo; la revisión midió 1036 s / 17,3 min en aislado). El 18° test (`test_cod_movimiento_coincide_con_evento_afc`) había fallado en la primera entrega por un supuesto de test demasiado estricto, no por el código; ya corregido |

Comando: `PYTHONPATH=/home/user/wt/datos python -m pytest afp_lab/rim_nowcasting/tests -q` (o
`-k test_cod_movimiento_coincide_con_evento_afc` para repetir solo el test corregido). El entorno
de esta ronda resultó mucho más lento que un laptop de 4 núcleos (máquina de 4 vCPU compartida con
otro workstream del laboratorio corriendo en paralelo); a la misma escala (300 afiliados) esa es la
duración real observada, muy por encima del límite de "2 minutos" de CONVENCIONES — documentado
como limitación de esta ronda, no una regresión de rendimiento del código en producción (que usa
`materializar=True`, con checkpoint, y no se ve afectado).

**Antes / después: backtest con el comando oficial (`pipeline backtest`), pasada de corrección**

La primera versión de esta sección reportaba un delta medido con un script fuera del repo
(`datos_r3_eval.py`, sólo en el scratchpad de la sesión): WAPE h=1 0,490 → 0,529 y runtime +82 %.
La revisión no pudo reproducirlo y, reproducido con el comando oficial, no se sostiene. **Esos
números quedan retirados**; lo que sigue se obtiene sólo con `python -m afp_lab.rim_nowcasting.pipeline
backtest` (que ahora acepta `--gbt-max-depth` y deja `reportes/backtest_*.csv`), a la escala de
referencia de la sección 10 y para ambos perfiles.

Método. "Antes" = copia de `afp_lab/` con `synthetic.py` y `features.py` de `main` (`git show
main:…`, sólo lectura); "después" = este worktree. `pipeline.py`, `config.py` y `model.py` son los
mismos en ambos (idénticos a `main` salvo el flag y los csv agregados en esta pasada). Cada corrida en
su propio `--salida`, un solo Spark a la vez, y "antes" ejecutado con cwd fuera del worktree (ver §8:
con `python -m` el cwd tapa a `PYTHONPATH`; un primer intento de esta pasada cometió ese error y midió
dos veces el mismo árbol — se detectó porque las importancias de "antes" contenían `mov_termino_l1`;
esas corridas se descartaron, pero sirven como prueba de repetibilidad: el mismo código dos veces da
métricas idénticas y un tiempo de pared que varía ±5-11 % en esta máquina compartida).

```bash
# después (worktree)
cd /home/user/wt/datos && PYTHONPATH=/home/user/wt/datos \
  python -m afp_lab.rim_nowcasting.pipeline backtest --perfil sistema --salida /tmp/bt_ref_despues_sistema
# antes: copia del paquete con synthetic.py/features.py de main
mkdir -p /tmp/antes_rim_lab && cp -r /home/user/wt/datos/afp_lab /tmp/antes_rim_lab/ && cd /home/user/wt/datos \
  && git show main:afp_lab/rim_nowcasting/synthetic.py > /tmp/antes_rim_lab/afp_lab/rim_nowcasting/synthetic.py \
  && git show main:afp_lab/rim_nowcasting/features.py  > /tmp/antes_rim_lab/afp_lab/rim_nowcasting/features.py
cd /tmp/antes_rim_lab && PYTHONPATH=/tmp/antes_rim_lab \
  python -m afp_lab.rim_nowcasting.pipeline backtest --perfil sistema --salida /tmp/bt_ref_antes_sistema
# ídem con --perfil cuprum. Escala reducida (la de la suite de tests): agregar
#   --n-afiliados 300 --n-train 3 --n-test 1 --gbt-iter 5 --gbt-max-depth 3 --inicio 202301 --snapshot 202506 --seed 42
```

**Escala de referencia** (5.000 afiliados, 2022-01 a 2026-09, 12 orígenes de entrenamiento, GBT 40 iteraciones / profundidad 5, test 2026-08 y 2026-09; `rim_proyectada`, n=9.627 h=1 / 10.000 h=2 en Sistema):

| `rim_proyectada` | Sistema antes | Sistema después | Δ | Cuprum antes | Cuprum después | Δ |
|---|---|---|---|---|---|---|
| WAPE h=1 | 0.196 | 0.194 | -0.002 | 0.210 | 0.212 | +0.002 |
| Sesgo agregado h=1 | -4.1 % | -4.6 % | -0.5 pp | -3.4 % | -4.2 % | -0.8 pp |
| AUC cotiza/no cotiza h=1 | 0.969 | 0.969 | +0.000 | 0.968 | 0.968 | +0.000 |
| Exactitud cero/no-cero h=1 | 0.926 | 0.925 | -0.001 | 0.918 | 0.916 | -0.002 |
| WAPE h=2 | 0.258 | 0.259 | +0.001 | 0.304 | 0.270 | -0.034 |
| Sesgo agregado h=2 | -4.3 % | -4.8 % | -0.5 pp | -6.8 % | -3.4 % | +3.4 pp |
| AUC cotiza/no cotiza h=2 | 0.918 | 0.919 | +0.001 | 0.919 | 0.920 | +0.001 |
| Exactitud cero/no-cero h=2 | 0.873 | 0.872 | -0.001 | 0.868 | 0.868 | +0.000 |
| Filas train / test | 113828 / 19627 | 113828 / 19627 | — | 113390 / 19643 | 113390 / 19643 | — |
| Tiempo `pipeline backtest` (generar+features+entrenar+evaluar) | 542 s | 375 s | -30.9 % | 438 s | 323 s | -26.2 % |

**Por segmento, escala de referencia** (`reportes/backtest_metricas_por_segmento.csv`; segmentos definidos sólo con columnas que existen antes y después, para que el delta sea comparable):

**Sistema** (`rim_proyectada`; segmento = independiente / termino_reciente (aviso AFC en el corte) / con_subsidio (licencia en el corte) / resto)

| Segmento | h | n | WAPE antes | WAPE después | Δ WAPE | Sesgo antes | Sesgo después |
|---|---|---|---|---|---|---|---|
| con_subsidio | 1 | 410 | 0.169 | 0.154 | -0.015 | -2.0 % | -1.9 % |
| independiente | 1 | 808 | 0.854 | 0.868 | +0.014 | -7.1 % | -9.9 % |
| resto | 1 | 8169 | 0.119 | 0.116 | -0.002 | -3.3 % | -3.6 % |
| termino_reciente | 1 | 240 | 0.939 | 0.997 | +0.058 | -92.4 % | -98.0 % |
| con_subsidio | 2 | 425 | 0.218 | 0.206 | -0.012 | +5.1 % | +4.3 % |
| independiente | 2 | 808 | 0.840 | 0.885 | +0.044 | +4.7 % | +6.9 % |
| resto | 2 | 8525 | 0.192 | 0.188 | -0.004 | -5.0 % | -5.6 % |
| termino_reciente | 2 | 242 | 0.961 | 0.995 | +0.034 | -93.9 % | -98.5 % |

(Cuprum por segmento: la corrida "después" de referencia se hizo antes de que `backtest` guardara los csv por segmento; pendiente repetirla con `--reusar-fuentes` sobre `/tmp/bt_ref_despues_cuprum`, ~3,5 min, no se hizo para no solapar dos Spark en la máquina compartida.)

**Lectura.** A la escala de referencia el código de movimiento **no cambia el WAPE global de
Sistema** (0,196 → 0,194 en h=1; 0,258 → 0,259 en h=2) y **mejora Cuprum en h=2** (0,304 → 0,270,
con el sesgo agregado de −6,8 % a −3,4 %); en h=1 Cuprum queda igual (0,210 → 0,212). AUC y
exactitud cero/no-cero no se mueven. El modelo sí usa las features nuevas: `mov_termino_l1` (0,042)
y `n_mov_termino_3` (0,039) entran al top 15 de importancia del clasificador, pero la señal es en
gran parte redundante con el feed AFC (`mov_termino_l1` coincide con `afc_termino_l1` en el 87 % de
los casos del snapshot 2026-03), lo que explica el efecto global casi nulo. Por segmento (Sistema):
`con_subsidio` mejora (−0,015 / −0,012), `resto` queda igual (−0,002 / −0,004), los independientes
(8,4 % de las filas; no declaran planilla, así que son el control sin código) empeoran +0,014 /
+0,044, y `termino_reciente` (2,5 % de las filas) empeora +0,058 / +0,034 — en ese segmento ambas
versiones tienen sesgo ≈ −92 % … −98 %: tras un aviso AFC de término el modelo proyecta ≈ 0, pero
una parte de esos afiliados sigue con RIM > 0 en el target (reingreso, finiquito o segundo empleador);
es una debilidad previa del modelo que el código de movimiento no corrige. Con una sola semilla y dos
orígenes de test no medimos la variabilidad entre semillas, así que un delta del orden de ±0,03 en un
segmento chico no se puede declarar significativo. Conclusión honesta: **no rompe nada, ayuda a
Cuprum en h=2, neutro en Sistema**; queda como familia de features opcional y el siguiente paso es
repetir con más semillas / más orígenes de test antes de decidir si se mantiene en producción.

**Runtime.** El +82 % anterior no reproduce. Las corridas "después" a escala de referencia tardaron
358 s y 375 s (Sistema, dos corridas del mismo código) y 323 s y 340 s (Cuprum); las de "antes"
(542 s y 438 s) corrieron con otro workstream del laboratorio ocupando la máquina (load average 4,6-5,5
en 4 vCPU), así que el delta de tiempo de pared no se pudo aislar de la contención; ninguna medición
del árbol "después" supera a la de "antes" ni el presupuesto de ~12 min del `demo` de la sección 10,
por lo que no hay evidencia de un sobrecosto cercano al +30 % del brief. Las etapas internas del log
(features por snapshot: 14-19 s "después" vs. 14-16 s "antes" en las corridas menos cargadas) apuntan
a un sobrecosto de un dígito por ciento, del mismo orden que la variación entre dos corridas iguales.

**Escala reducida** (la de la suite de tests: 300 afiliados, 2023-01 a 2025-06, 3 orígenes de entrenamiento / 1 de test, GBT 5 iteraciones / profundidad 3, seed 42; perfil Sistema, `rim_proyectada`, n=296 h=1 / 300 h=2):

| `rim_proyectada` | Sistema antes | Sistema después | Δ |
|---|---|---|---|
| WAPE h=1 | 0.399 | 0.392 | -0.007 |
| Sesgo agregado h=1 | +9.8 % | +10.1 % | +0.3 pp |
| AUC h=1 | 0.925 | 0.907 | -0.018 |
| WAPE h=2 | 0.465 | 0.450 | -0.015 |
| Sesgo agregado h=2 | +13.5 % | +11.7 % | -1.8 pp |
| AUC h=2 | 0.875 | 0.871 | -0.004 |
| Tiempo `pipeline backtest` | 112 s | 96 s | -13.7 % |

A esta escala el WAPE h=1 baja 0,399 → 0,392 y h=2 0,465 → 0,450 (misma dirección que la revisión, magnitud chica); el baseline `media_3` (0,380 / 0,410) sigue ganando al modelo en ambas versiones, un problema preexistente de evaluar con 5 árboles y 1.646 filas, no de esta ronda. La corrida reducida "antes" de Cuprum abortó por un `DIVIDE_BY_ZERO` en las métricas por segmento nuevas (un segmento chico sin RIM real > 0 a 300 afiliados); se corrigió con `F.try_divide` en `evaluate.metricas` y no se repitió para no solapar dos Spark con la suite de tests — Cuprum queda evaluado a la escala de referencia (tabla anterior), que es la que pide el brief.

## 11. Supuestos y próximos pasos

* Los parámetros macro y las proporciones (rotación, DNP, pagos tardíos) son supuestos razonables para
  Chile, no series oficiales; en producción se reemplazan por los datos reales de la AFP y los ajustes se
  validan con el backtest.
* Supuestos de disponibilidad de información que conviene confirmar con la AFP: las licencias se conocen
  en m+1 (feed COMPIN/ISAPRE), los avisos AFC en m+1, el IMM y el tope del mes target están legislados al
  correr el modelo (`imm_target`, `tope_target`; si un reajuste del IMM se publica con retraso, esa
  feature debería tomar el valor vigente al corte), y `tipo_afiliado` es un atributo del maestro (no una
  inferencia).
* La etiqueta de entrenamiento está censurada a `desfase + maduracion` meses (4 con la configuración por
  defecto, cubre pagos tardíos hasta m+4 y subsidios); los pagos posteriores se etiquetan 0 en
  entrenamiento pero cuentan en la evaluación. Subir `meses_maduracion` reduce ese sesgo a costa de datos
  más antiguos.
* Convenciones documentadas: la RIM del afiliado se topa por pagador y en la suma (la devolución de
  excesos por multiempleo no se modela); el reporte de reemplazos mide contra el primer dato real que llega
  (una rectificación posterior lo corrige en el ciclo siguiente).
* Mejoras naturales: modelo específico para independientes (anual, vía Operación Renta), cuantiles
  (P10/P90) para intervalos, calibración del umbral por segmento según el costo de cada error,
  reentrenamiento automático por drift del reporte de reemplazos, y explicabilidad (SHAP) para auditoría.
* Ronda 3 (código de movimiento, sección 10c): el delta antes/después ya se midió a la escala de
  referencia (5.000 afiliados / 12 snapshots / GBT 40) para Sistema y Cuprum con el comando oficial:
  neutro en Sistema, mejora Cuprum en h=2; queda pendiente medir la variabilidad entre semillas y
  con más orígenes de test antes de decidir si la familia de features se mantiene; generar el código 8 (cambio de contrato) requeriría procesar la historia completa de cada
  afiliado en un mismo lote (no compatible con el enfoque columna-a-columna usado aquí, sí con un
  `mapInPandas` agrupado por afiliado, más costoso); evaluar además si conviene fusionar esta tabla
  con `common/planilla.py` (mismo dominio, dos implementaciones e independientes, ver "Propuestas
  para common" en la sección siguiente).

**Propuestas para `common/`**: `afp_lab/rim_nowcasting/synthetic.py` y `afp_lab/common/planilla.py`
ahora generan, por separado, el mismo concepto (código de movimiento de personal, Tabla N° 7) con
grano distinto (pagador vs. concepto) y motores distintos (Spark vs. pandas) — comparten sólo las
constantes (`P_PERMISO_SIN_GOCE`, `P_AUSENTISMO`, importadas). Valdría la pena, en una ronda futura,
extraer a `common/tablas_equivalencia.py` una única función pura `codigo_movimiento(evento_afc,
tipo_contrato, dias_licencia) -> int` que ambos módulos llamen (con `numpy` vectorizado en
`common/planilla.py` y con `F.when` — o un `pandas_udf` que envuelva la misma función — en
`rim_nowcasting`), para no mantener la misma regla de negocio dos veces.
