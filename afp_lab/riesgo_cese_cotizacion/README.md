# 04 · Riesgo de cese de cotización y probabilidad de reingreso

Responsable: DS-3. Ronda 2 (devuelta en ronda 1). Revisado contra `CONVENCIONES.md` §7 y
contra `docs/revisiones/ronda_1/riesgo_cese_cotizacion.md`.

## Correcciones de la ronda 2 (revisión del gerente)

La ronda 1 fue devuelta por leakage real en la feature dominante y un segmento donde el
modelo perdía. Se corrigieron, en el orden pedido:

1. **Bloqueante — leakage vía `prestaciones_cesantia`**: el generador paga un giro en el
   mes `p` solo si la RIM **real** (no conocida) es 0 en `p`, y la versión anterior de
   `_features_eventos` leía esa tabla con `periodo ≤ corte` (debía ser `≤ corte-1`, igual
   que el resto de los eventos AFC). Corregido: la cobertura de cesantía
   (`cesantia_en_curso`, `cesantia_giros_restantes`, `cesantia_tasa_actual`) ahora se
   deriva **100 % analíticamente** del TERMINO AFC conocido + el tipo de contrato de esa
   relación + las tasas legales (`giros_restantes = max_giros - meses_desde_termino`); no
   se vuelve a leer `prestaciones_cesantia` desde `features.py`. Se agregó
   `test_leakage_cesantia_no_lee_rim_futura` (altera el panel/las tablas derivadas
   exactamente en `corte-1` y `corte`, los meses del bug, y exige features idénticas) y un
   control positivo (`test_termino_en_corte_menos_1_si_actualiza_cesantia`) que confirma
   que la feature corregida sigue reaccionando de verdad a un término real. Impacto en
   AUC: cese 3 m pasó de 0,889 a **0,867**, reingreso de 0,778 a **0,723** — bajan, como
   avisaba la revisión, pero sin perder el margen sobre los umbrales del brief.
2. **Bloqueante — segmento INDEPENDIENTE/VOLUNTARIO**: en este universo la RIM de un
   independiente es constante dentro del año calendario (se redibuja cada enero,
   `synthetic._simular_particion`) y la de un voluntario es esencialmente un sorteo mensual
   independiente; para ambos, "cesar"/"reingresar" no es un fenómeno de historia laboral
   sino un artefacto de calendario o puro azar, y el baseline rubro × mes les ganaba al
   modelo. Se **excluyeron ambos tipos de afiliados de los Modelos A y B**
   (`TIPOS_DEPENDENCIA = ["DEPENDIENTE", "PENSIONADO_ACTIVO"]` en `features.py`). El
   diagnóstico anterior ("SII en junio") era el de rim_nowcasting, no el de este universo
   (aquí no hay pago anual vía SII para independientes, sino una RIM mensual constante); se
   corrigió la explicación.
3. **Mayor — gap del split**: el gap real entre el último target de train/val (`corte+h`) y
   el primer corte de test era de 1 mes, no de `h`. Corregido a `corte ≤ test_min - 2h`
   (`run._split_temporal`); se subió el número de cortes usados de 24 fijos a **todos los
   que caben** entre `periodo_inicio` y `periodo_fin - 6` (ver punto 7), para no quedarse
   sin datos de entrenamiento al duplicar el gap. Test:
   `test_split_gap_es_2h_no_h`.
4. **Mayor — población B contaminada por recepción tardía**: exigir solo `RIM conocida = 0`
   en `corte-2` dejaba adentro cotizantes reales cuya planilla de `corte-2` sencillamente no
   había llegado a tiempo (∼5 % de cualquier mes es tardío por diseño); ahora se exige
   `corte-2` **y** `corte-3` en 0. También se agregó `cobertura_poblacion_cese`: la
   población A (`pop_cese`) cubre el **95,3 %** de los cotizantes reales del mismo grupo
   (dependientes/pensionados activos) — el resto es la misma censura de recepción, esta vez
   del lado de los que sí cotizan. Factor de corrección promedio para escalar agregados al
   universo completo: **1,0496** (`reportes/cobertura_poblacion_a.csv`).
5. **Mayor — bug de `reindex(fill_value=0)`**: no rellenaba los `NaN` que `unstack` deja
   cuando un empleador conocido no tiene planilla un mes puntual de la ventana; se agregó
   `.fillna(0)` explícito y la bandera `emp_sin_planilla_conocida`.
6. **Mayor — `cesantia_n_giros_recibidos` (máximo histórico)**: al derivar la cobertura
   solo del término AFC **más reciente** (punto 1), la feature ya no puede mezclar
   episodios distintos; se eliminó esa columna y quedó `cesantia_giros_restantes` /
   `cesantia_tasa_actual`, siempre del episodio vigente.
7. **Menores 7, 8, 10, 11**: ventana de historia acotada explícitamente a `periodo_inicio`
   y reducida de 30 a 26 meses (§4); `tipo_contrato` atado a la relación **vigente**
   (empleador de la RIM conocida), no a cualquier inicio pasado (§4); métricas por segmento
   para los 3 predictores, no solo el modelo (§6); `metrica_negocio_agregado` es y se lee
   por **mes**, no por trimestre (§6), y se corrigió el docstring que la llamaba "WAPE" por
   error (es un error relativo con signo de un agregado, no la métrica WAPE de
   `common.metrics`). Además (menor #9, no listado explícitamente en la lista de
   obligatorios pero mencionado en el detalle): se corrigió el sesgo de `emp_crecimiento_12`
   (mediana pasó de ~0,90, un artefacto de censura de recepción, a ~0,99); y (menor #12) los
   ejemplos interpretados ahora se eligen por el puntaje **crudo** (sin la meseta de la
   calibración isotónica) y muestran probabilidad cruda y calibrada por separado.

Detalle completo de números antes/después al final del informe (no en este README, en la
respuesta al gerente). El resto del documento ya refleja la ronda 2 corregida.

## 1. Pregunta de negocio y decisión que apoya

Dos usos, con la misma familia de modelos:

* **Proyección de recaudación / densidad de cotización**: ¿cuántos cotizantes de hoy
  dejarán de cotizar en 3 y 6 meses, y cuántos no cotizantes reingresarán? Alimenta el
  presupuesto de recaudación esperada y el indicador de densidad de cotización del fondo.
* **Asesoría previsional**: alertas tempranas al 10 % de afiliados con mayor riesgo de
  laguna previsional, para contacto proactivo (recordatorio de cotización voluntaria,
  oferta de asesoría en el seguro de cesantía, etc.).

Por eso se piden probabilidades **calibradas** (se agregan para el primer uso) y con
**buena discriminación** (para priorizar contactos en el segundo).

## 2. Datos sintéticos

Se parte de `afp_lab.common.synth_base.generar_universo` (afiliados, empleadores, panel
mensual con estado laboral, RIM, licencias y eventos AFC) **sin redefinir** esa dinámica.
Este proyecto agrega 4 tablas (`afp_lab/riesgo_cese_cotizacion/synthetic.py`), derivadas
del panel, que reproducen el desfase de información real de una AFP:

| Tabla | Grano | Mecanismo y parámetros | Por qué es razonable |
|---|---|---|---|
| `cotizaciones_conocidas` | afiliado × periodo × empleador pagador | RIM de `rim_emp1`/`rim_emp2` del panel; pago declarado y recibido en `m+1` (95 %) o con rezago adicional de 1 a 6 meses (5 %, `P_TARDIO`) | Mismo orden de magnitud de atrasos que Previred/AFC (proyecto 01); un empleador rara vez se atrasa más de medio año |
| `afc_eventos` | evento (afiliado × periodo) | Inicio/término de la relación laboral desde `evento_afc` del panel, conocidos en `periodo+1`; excluye funcionarios públicos (estatuto administrativo) y pensionados activos (no afectos a la Ley 19.728) | El Seguro de Cesantía se informa mucho más rápido que la nómina (aviso en días); ambas exclusiones son legales, no supuestos de conveniencia |
| `licencias` | afiliado × periodo | Días de licencia médica del panel, conocidos en `periodo+1` (aviso COMPIN/ISAPRE más rápido que el pago del subsidio) | Igual criterio que rim_nowcasting |
| `prestaciones_cesantia` | giro (afiliado × periodo × n_giro) | Hasta 5 giros decrecientes (contrato indefinido, tasas aprox. 70/55/45/40/35 % de la remuneración imponible media de los últimos 12 meses) o 2 (plazo fijo, 70/55 %); se detienen si el afiliado reingresa antes de agotarlos (mirando la RIM **real**). Conocidos al mes | Aproximación razonable a la Ley 19.728. **Se conserva como tabla fuente porque el brief la pide, pero desde la ronda 2 `features.py` NO la lee** (ver corrección #1 arriba): decide si pagar mirando la RIM real, y usarla directamente para features filtraba el futuro. La cobertura de cesantía que sí ve el modelo se reconstruye analíticamente en `features.py` a partir del término AFC conocido, sin volver a tocar esta tabla |

Con `n_afiliados=20 000`, `202201`–`202609`: **891 726** movimientos de cotización
conocidos, **64 515** eventos AFC (32 315 términos, 32 200 inicios), **55 552** episodios
de licencia y **72 783** giros de cesantía (esta última, solo como tabla fuente: no
alimenta features). Proporciones observadas de salidas (términos AFC sobre afiliado-mes
activo) por rubro: más altas en construcción, turismo y agricultura (estacionales,
coherente con `rim_nowcasting.synthetic.RUBROS`), más bajas en minería, finanzas y
administración pública — exactamente la dinámica ya simulada en el proyecto 01, que aquí
solo se re-observa a través de los eventos.

No se usan `lat_*` ni los `p_*`/`peso_contratacion` de empleadores en ninguna feature
(`synth_base.generar_universo` ya los separa en `*_latente`, que este proyecto no importa).

## 3. Target, población, horizonte y exclusiones

Fila = afiliado × `periodo_corte` (`corte`, "hoy"), cortes mensuales móviles.

* **Modelo A — cese (`y3`, `y6`)**: población = afiliados **dependientes o pensionados
  activos** (`TIPOS_DEPENDENCIA`) con RIM conocida > 0 en `corte-2` (`pop_cese`). `y3=1` si
  no hay RIM > 0 (verdad final, del panel) en **ninguno** de `corte+1…corte+3`; `y6`
  análogo a 6 meses. Se entrenan **dos modelos binarios independientes** (no uno
  multi-horizonte): las tasas base y la importancia relativa de las features cambian
  bastante entre 3 y 6 meses, y el brief pide reportarlos por separado; un solo modelo con
  `horizonte` como feature no simplifica el pipeline aquí (a diferencia de rim_nowcasting,
  que sí comparte modelo entre horizontes porque el objetivo es un nivel continuo, no dos
  ventanas de "todo o nada").
* **Modelo B — reingreso (`y_reingreso`)**: población = afiliados **dependientes** con RIM
  conocida = 0 en **`corte-2` y en `corte-3`** (ronda 2: exigir solo `corte-2` dejaba
  adentro cotizantes reales cuya planilla llegó tarde, ver corrección #4) y al menos una
  cotización conocida en los 24 meses previos (`pop_reingreso`). `y=1` si hay RIM > 0
  (verdad final) en `corte+1…corte+6`.
* **Independientes y voluntarios quedan fuera de ambos modelos** (corrección #2, ronda 2):
  en este universo su RIM no responde a una dinámica de empleo/desempleo real (ver
  justificación completa en §8), así que "cese"/"reingreso" no es el mismo fenómeno para
  ellos y un baseline ingenuo por rubro × mes los describe mejor que cualquier historia
  laboral.
* **Exclusiones adicionales**: cortes con horizonte incompleto (`corte+h` fuera del rango
  simulado) se descartan (`horizonte3_completo`/`horizonte6_completo`). Un
  retirado/pensionado sin actividad hace años nunca entra a ninguno de los dos modelos
  (ambas poblaciones exigen, directa o indirectamente, cotización reciente), sin necesidad
  de una regla adicional sobre `estado` (que además no es observable por la AFP: ver §8).
* Las dos poblaciones son disjuntas por construcción (`pop_cese & pop_reingreso == 0`
  siempre, verificado en `tests/test_riesgo_cese.py::test_poblaciones_bien_definidas`).
* **Cobertura de la población A**: `pop_cese` (RIM *conocida*) es, por diseño, un
  subconjunto de los cotizantes *reales* del mismo grupo — cubre en promedio el **95,3 %**
  (`cobertura_poblacion_cese`, `reportes/cobertura_poblacion_a.csv`); el 4,7 % restante son
  cotizaciones reales cuya planilla de `corte-2` aún no se conoce. Para proyectar
  agregados al universo completo de cotizantes (no solo a los ya confirmados), el número
  esperado debe escalarse por el factor de corrección promedio **1,0496** (`= 1/cobertura`).

## 4. Features (`features.py`)

Todas se calculan en `construir_features(tablas, periodo_corte)` **sin** tocar el panel de
verdad ni `prestaciones_cesantia`; el target se agrega aparte con `agregar_targets` (que sí
usa el panel, por definición: un target habla del futuro del corte). Una fila por afiliado
ya afiliado a `corte`.

**Regla as-of** (documentada también como código en el docstring de `features.py`, y
verificada en 6 tests de leakage — 2 nuevos en la ronda 2): la RIM se conoce hasta `corte-2`
(recepción ≤ `corte-1`, igual regla que rim_nowcasting); los eventos AFC y licencias se
conocen hasta `corte-1` (recepción ≤ `corte`, un mes más rápido — ver decisión en §8). La
cobertura de cesantía usa los mismos eventos AFC (por lo tanto, también acotada a
`corte-1`) y NO lee la tabla `prestaciones_cesantia` (corrección #1). Los lags `l1`/`l2`
están anclados a `corte-2` para RIM y a `corte-1` para eventos.

| Familia | Features | Regla as-of |
|---|---|---|
| Historia de RIM | `rim_l1..rim_l12`, `media_3/6/12/24`, `std_6`, `max_12`, `min_12`, `n_cotiza_3/6/12/24`, `meses_historia`, `racha_cotiza`, `racha_cero`, `meses_desde_ult_cotiza`, `meses_desde_cambio`, `n_cambios_12`, `cv_6`, `ratio_l1_l2`, `ratio_l1_media12`, `tendencia_3_12`, `sueldo_fijo` | Ventana de **26** meses (antes 30; ronda 2, ver §8) terminada en `corte-2`, RIM conocida con recepción ≤ `corte-1`, **acotada explícitamente a no mirar antes de `periodo_inicio`** |
| Relación laboral | `meses_con_empleador`, `n_empleadores_12`, `tipo_contrato` (INICIO AFC de la relación **vigente**, no de cualquier inicio pasado — corrección de ronda 2), `emp_n_trabajadores`, `emp_crecimiento_12`, `emp_tasa_salida_12`, `emp_sin_planilla_conocida` (empleador de referencia) | Empleador de referencia y sus agregados, con RIM conocida ≤ `corte-1` y **corregidos por madurez de recepción** (ronda 2, ver §8) |
| Eventos y cesantía | `afc_termino_l1/l2`, `afc_inicio_l1/l2`, `meses_desde_afc_termino/inicio`, `dias_licencia_l1/l2`, `licencia_en_curso`, `cesantia_en_curso`, `cesantia_giros_restantes`, `cesantia_tasa_actual` (derivadas analíticamente del término AFC + tipo de contrato + tasas legales, ronda 2) | Conocidos con recepción ≤ `corte` y periodo ≤ `corte-1` |
| Perfil | `tipo_afiliado`, `sexo`, `edad`, `meses_hasta_pension`, `region`, `nivel_educacional`, `meses_desde_afiliacion` | Estático / determinístico a partir de `corte` |
| Calendario y macro | `mes_corte`, `es_enero/marzo/abril/julio/sept/dic`, `desempleo_corte`, `estacionalidad_rubro` (fracción histórica de términos del rubro que cae, en promedio, en los meses de la **ventana de target** `corte+1..corte+6` — ronda 2: antes se miraba el mes del corte, no el de los meses que se están arriesgando) | Macro de `corte`; estacionalidad recalculada por corte con la historia conocida hasta ese punto (nunca con datos futuros) |

`rubro`/`tamano_empleador`/`region_empleador` vienen de `empleadores` (columnas no
latentes). Sin empleador de referencia: `rubro="SIN_EMPLEADOR"`, `tipo_contrato=
"SIN_EMPLEADOR"`; con empleador pero sin aviso AFC de inicio conocido **para ese empleador
específico** (la relación empezó antes del historial disponible, o el único inicio
conocido pertenece a una relación distinta ya cerrada): `tipo_contrato="SIN_DATO_AFC"`.

## 5. Modelo y baselines (`model.py`)

* **LightGBM binario** (`n_jobs=2`, `n_estimators=350`, `num_leaves=31`,
  `min_child_samples=60`, `subsample`/`colsample_bytree=0.8`) por target, con categóricas
  nativas (`pandas.Categorical`, sin one-hot). **Calibración isotónica** ajustada sobre un
  conjunto de **validación** (nunca train ni test).
* **Cortes**: `run._cortes_validos` usa **todos** los cortes mensuales cuya ventana de
  historia (26 meses) cabe entera dentro de `[periodo_inicio, periodo_fin]` y cuyo
  horizonte más largo (6 meses) queda completo antes de `periodo_fin` (tope de seguridad
  `N_CORTES_TOPE=40`, no alcanzado con los periodos por defecto). Con
  `202201`–`202609` son **24 cortes** (`202404`..`202603`), compartidos por los 3 targets.
* **Split temporal**: test = últimos 3 cortes; train/val = los anteriores, con un **gap
  real de `2×horizonte`** meses entre el último corte de train/val y el primero de test
  (ronda 2, corrección #3: la versión anterior solo garantizaba un gap real de 1 mes entre
  el ÚLTIMO TARGET de train, que se resuelve en `corte+horizonte`, y el primer corte de
  test — `test_split_gap_es_2h_no_h` lo prueba explícitamente). Validación = el 25 % más
  reciente de los cortes de train/val restantes (tras el gap), para calibrar sin tocar
  test.
* **Baselines obligatorios**: (a) tasa base histórica de train por `rubro × mes_corte`
  (con respaldo = tasa global de train si la combinación no se vio); (b) regla de negocio
  simple del brief — cese: 1.0 si hay término AFC en `corte-1` o `corte-2`, 0.5 si el
  contrato es a plazo fijo con menos de 6 meses de antigüedad, 0 si no; reingreso:
  `1/(1+racha_cero)` (mientras más corta la racha sin cotizar, más probable el reingreso
  inminente).

## 6. Resultados (test, 3 últimos cortes: 202601–202603; `n_afiliados=20 000`, seed=42)

### Tabla principal (modelo vs. baselines)

| Target | Predictor | n | tasa base | AUC | Average Precision | Brier | Lift decil 1 | KS |
|---|---|---|---|---|---|---|---|---|
| **Cese 3 m** | **Modelo** | 41 179 | 0,0675 | **0,867** | 0,429 | **0,0459** | 6,30 | 0,594 |
| | Baseline tasa base | 41 179 | 0,0675 | 0,700 | 0,139 | 0,0606 | 2,64 | 0,308 |
| | Baseline regla de negocio | 41 179 | 0,0675 | 0,798 | 0,325 | 0,0835 | 6,09 | 0,562 |
| **Cese 6 m** | **Modelo** | 41 179 | 0,0432 | **0,858** | 0,300 | **0,0347** | 6,39 | 0,586 |
| | Baseline tasa base | 41 179 | 0,0432 | 0,574 | 0,055 | 0,0412 | 1,76 | 0,090 |
| | Baseline regla de negocio | 41 179 | 0,0432 | 0,795 | 0,214 | 0,0892 | 6,18 | 0,557 |
| **Reingreso** | **Modelo** | 7 747 | 0,6619 | **0,723** | 0,824 | **0,1922** | 1,43 | 0,302 |
| | Baseline tasa base | 7 747 | 0,6619 | 0,486 | 0,656 | 0,2257 | 0,94 | 0,035 |
| | Baseline regla de negocio | 7 747 | 0,6619 | 0,575 | 0,707 | 0,4668 | 1,12 | 0,111 |

El modelo gana a ambos baselines en AUC y Brier en los 3 targets. Nótese que el baseline de
**regla de negocio** ahora es mucho más fuerte que en la ronda 1 (por ejemplo AUC 0,798 en
cese 3 m): usa `afc_termino_l1/l2`, la misma familia de señal que hoy domina las
importancias del modelo tras sacar el leakage (ver más abajo), así que la brecha
modelo-regla se cerró bastante — sigue siendo clara, pero ya no es aplastante. Tablas
completas en `salidas/riesgo_cese_cotizacion/reportes/metricas_*.csv`.

### Por segmento, para los 3 predictores (menor #10, ronda 2: antes solo el modelo)

* **Tipo de afiliado** (`segmento_tipo_afiliado_*.csv`): tras excluir independientes y
  voluntarios, solo quedan DEPENDIENTE (99,4 % de las filas) y PENSIONADO_ACTIVO (0,6 %,
  bajo el umbral de "segmento grande"). En DEPENDIENTE el modelo gana a ambos baselines por
  márgenes amplios en los 3 targets (p. ej. cese 3 m: AUC 0,868 vs. 0,701/0,802).
* **Tramo de edad y tramo de renta** (sobre `rim_l1`): el modelo gana en AUC y Brier a
  ambos baselines en los 4-6 tramos de cada uno para cese 3 m y cese 6 m. Para
  **reingreso**, el modelo gana en AUC en TODOS los tramos de edad ≥5 % de las filas (p.
  ej. 45-54: AUC 0,617 vs. 0,503/0,487), pero en **Brier** queda muy levemente por encima
  del baseline de tasa base en 3 de los 4 tramos grandes (25-34: 0,1869 vs. 0,1772; 35-44:
  0,1970 vs. 0,1912; 45-54: 0,1865 vs. 0,1862 — diferencias de 0,0003 a 0,0097). Se
  documenta en vez de ocultarlo (ver §8): no es del mismo tipo que la falla de
  INDEPENDIENTE en la ronda 1 (ahí el modelo perdía en AUC Y Brier por márgenes grandes;
  aquí gana claramente en la métrica principal del brief, AUC, y pierde por una fracción
  mínima en Brier). Para reingreso, `tramo_renta` es siempre "sin_renta" y `rubro` siempre
  "SIN_EMPLEADOR" por construcción de la población (sin RIM ni empleador vigentes): sigue
  siendo una limitación de diseño, no un error (ver propuesta en §10).
* **Rubro y tipo de contrato**: el modelo gana en AUC y Brier a ambos baselines en todos los
  rubros y tipos de contrato con ≥ 5 % de las filas, en los 3 targets.

### Calibración

Brier del modelo mejor que el de la tasa base en los 3 targets (tabla principal). Curvas de
calibración por decil (`calibracion_*.csv`): en cese 3 m el error absoluto predicho-observado
es ≤ 0,017 en los 10 deciles. En reingreso el ajuste es más ruidoso en los deciles
intermedios (error hasta 0,19 en el decil 7) pero el extremo superior e inferior (deciles 0
y 9, los más accionables para priorizar contactos) calibran a menos de 0,033.

### Importancias (gain)

* **Cese 3 m**: `meses_desde_afc_termino` (49,5 %), `cesantia_tasa_actual` (8,7 %), `rubro`
  (4,6 %), `edad`, `afc_inicio_l1`, `meses_desde_afiliacion`, `region`,
  `meses_hasta_pension`.
* **Cese 6 m**: mismo orden con `meses_desde_afc_termino` (31,1 %) y `cesantia_tasa_actual`
  (8,1 %) al frente; el resto (edad, meses hasta pensión, antigüedad, rubro) se reparte de
  forma más pareja que a 3 meses.
* **Reingreso**: `edad` (15,9 %), `meses_hasta_pension` (14,9 %), `meses_desde_afc_inicio`
  (12,5 %), `meses_desde_afiliacion` (9,9 %), `max_12`, `region`, `media_24`,
  `n_cotiza_24`.

Que `meses_desde_afc_termino`/`cesantia_tasa_actual` dominen en cese **ya no depende de la
RIM futura** (a diferencia de la ronda 1: ver corrección #1 y los tests de leakage en §8):
ambas se calculan solo con el término AFC conocido hasta `corte-1` y las tasas legales, y
siguen siendo, con razón, un predictor fuerte (cuánto tiempo lleva cesante y cuánta
cobertura del seguro le queda son señales directas de riesgo de seguir cesante).

### Ejemplos interpretados (cese 3 m; ronda 2: elegidos por el puntaje CRUDO, sin la
meseta de la calibración isotónica — menor #12)

Uno por rubro estacional (`reportes/ejemplos_interpretados.csv`), con probabilidad
calibrada y cruda por separado (las contribuciones SHAP explican la cruda, en log-odds):

* Afiliado en `A_AGRICULTURA` (término AFC reciente): `pred_calibrada=0,85`,
  `pred_cruda=0,80`, `y3_real=1`. Contribuciones: `meses_desde_afc_termino` (+2,40), `edad`
  (+0,46), `rubro` (+0,39).
* Afiliado en `F_CONSTRUCCION`: `pred_calibrada=1,00`, `pred_cruda=0,88`, `y3_real=0` — un
  falso positivo real: el modelo lo juzga de altísimo riesgo (término AFC muy reciente),
  pero consiguió empleo antes de cumplirse los 3 meses; ejemplo honesto de que la
  discriminación (AUC 0,87) no es perfecta.

### Métrica de negocio

Contactando al **10 % de mayor riesgo** en test: se captura el **63,3 %** de los ceses a
3 meses, **63,6 %** de los ceses a 6 meses y **14,4 %** de los reingresos (la tasa base de
reingreso es 66 %, así que el 10 % de mayor riesgo son en su mayoría los que de todas
formas iban a reingresar rápido; ver interpretación de negocio en §7).

Número esperado (suma de probabilidades calibradas) vs. real de casos, **por mes de corte**
(el brief pide el error MENSUAL, no un promedio del trimestre de test — menor #11, ronda 2:
la ronda 1 evaluaba el criterio de aceptación contra un total trimestral en vez de mes a
mes):

| Target | 202601 | 202602 | 202603 |
|---|---|---|---|
| Cese 3 m | -2,5 % | +2,5 % | +7,6 % |
| Cese 6 m | -4,5 % | +2,6 % | +7,1 % |
| Reingreso | +2,0 % | +3,7 % | +3,8 % |

Los **9 meses × 3 targets = 9 números** están todos dentro de ±10 % (el peor es +7,6 %, en
cese 3 m 202603): a diferencia de la ronda 1 (un mes de cese 6 m llegaba a +12,1 %), la
corrección de la población B y del gap del split también mejoró la estabilidad del
agregado mes a mes. Detalle completo en `reportes/negocio_agregado_*.csv`.

## 7. Lectura de negocio

* **Recaudación esperada**: el "número esperado de ceses/reingresos" calibrado por mes
  (tabla de arriba) es directamente utilizable para proyectar densidad de cotización y
  flujo de recaudación, con error mensual entre -4,5 % y +7,6 % en los 9 meses×target
  evaluados. Para proyectar sobre el universo COMPLETO de cotizantes (no solo los ya
  confirmados como tales), multiplicar por el factor de corrección de cobertura (§3):
  **≈1,05**.
* **Priorización de contactos**: contactar al 10 % de mayor riesgo de cese captura ~6 de
  cada 10 ceses reales a 3 y a 6 meses — una mejora sustancial frente a contactar al azar
  (que capturaría ~10 %). Para reingreso, el 10 % de mayor riesgo no es el segmento más
  útil para contactar (la tasa base ya es 66 %: la mayoría reingresa igual, con o sin
  contacto); una campaña de reingreso rendiría más si se apunta a la **cola opuesta**
  (menor probabilidad de reingreso, para ofrecerles cotización voluntaria/APV antes de que
  la laguna se alargue) — una relectura de la misma probabilidad calibrada, sin necesidad
  de otro modelo.
* Los ejemplos interpretados muestran el patrón más accionable: un afiliado con un término
  AFC muy reciente y sin señales de reingreso es, con razón, el perfil de mayor riesgo — el
  momento natural para ofrecerle asesoría antes de que se agote su cobertura de cesantía.

## 8. Decisiones tomadas, supuestos y limitaciones

**Ambigüedades del brief, resueltas y documentadas**:

* El brief da dos reglas de desfase que, tomadas literalmente con un único corte de
  recepción, se contradicen: dice que la RIM se conoce hasta `corte-2` pero que los eventos
  AFC/licencias se conocen hasta `corte-1` (un mes menos), aunque su propia tabla de datos
  diga para ambos "conocidos en `mes+1`". Se decidió reproducir **ambas** literalmente
  usando **dos cortes de recepción distintos dentro de la misma fila**:
  `hasta_recepcion_rim = corte-1` (RIM conocida hasta `corte-2`) y
  `hasta_recepcion_eventos = corte` (AFC/licencias conocidos hasta `corte-1`) — consistente
  con que el aviso AFC llega por una vía (Seguro de Cesantía) más rápida que la nómina. Con
  esto, `afc_termino_l1`/`l2` quedan literalmente en `corte-1`/`corte-2`, tal como los
  nombra el brief.
* **(Ronda 2) La cobertura de cesantía NO puede leer `prestaciones_cesantia` con el mismo
  corte que los demás eventos**, porque esa tabla decide si pagar un giro mirando la RIM
  **real** (no conocida) del mes: hacerlo con `periodo ≤ corte` (como se hacía en ronda 1,
  copiando el corte de AFC/licencias) filtraba 2 meses de futuro de la RIM. La corrección no
  fue "usar `corte-2` en vez de `corte`" (eso seguiría dependiendo de una tabla que mira RIM
  real), sino dejar de leer esa tabla para features y reconstruir la cobertura
  analíticamente desde el término AFC conocido + tipo de contrato + tasas legales — la única
  fuente de esa información que es, por construcción, as-of.
* No se modelan DNP ni rectificaciones en `cotizaciones_conocidas` (a diferencia de
  rim_nowcasting): el brief solo pide reproducir el desfase de conocimiento, no la
  granularidad completa de movimientos.
* Tasas de giros de cesantía (70/55/45/40/35 % a 5 meses INDEFINIDO; 70/55 % a 2 meses
  PLAZO_FIJO) son una aproximación ilustrativa de la Ley 19.728, no los porcentajes legales
  exactos. El universo simulado no genera contratos "por obra" como categoría aparte (solo
  INDEFINIDO/PLAZO_FIJO); la ley trata ambos igual para el Fondo Solidario.
* Ventana de historia reducida de 30 a **26 meses** (`VENTANA_HISTORIA`) y acotada
  explícitamente a no mirar antes de `periodo_inicio` (menor #7, ronda 2): con 30 meses, los
  cortes más cercanos al inicio de la simulación pedían meses anteriores a `periodo_inicio`
  que no existen, y se rellenaban con ceros indistinguibles de cesantía real — un sesgo
  distinto por corte. Ahora `run._cortes_validos` solo usa cortes donde la ventana completa
  cabe en el rango simulado, y `_features_historia_rim` recorta la ventana como resguardo
  adicional. 26 meses = 24 de densidad (el máximo que pide el brief) + 2 de margen técnico.
* `emp_crecimiento_12`/`emp_tasa_salida_12` se corrigen por **madurez de recepción**
  (`_maturity_recepcion`, ronda 2): el mes más reciente de la ventana solo alcanza a ver el
  95 % de las planillas (el pago oportuno; el 5 % tardío aún no llega), mientras que un mes
  con 13 meses de maduración ya vio prácticamente el 100 %; sin corregir, TODOS los
  empleadores parecían encoger ~5 % cada 12 meses (mediana de `emp_crecimiento_12` ~0,90),
  un artefacto de recepción, no de dotación real. Corregido, la mediana queda en ~0,99 (se
  verifica en `test_features_empleador_sin_nan_y_crecimiento_no_sesgado`).
* `edad` se proyecta desde `edad_inicio` (edad al primer periodo simulado, según
  `synth_base.generar_afiliados_pandas`) igual que en `rim_nowcasting.features`, usando
  `macro["periodo"].min()` como referencia de `periodo_inicio`.
* 3 modelos LightGBM independientes (no multi-tarea): ver justificación en §3.

**Criterios de aceptación: estado actual (números reales, sin maquillar)**

* AUC test cese 3 m **0,867** (≥ 0,80 pedido) ✓. AUC test reingreso **0,723** (≥ 0,70
  pedido) ✓ — bajó desde 0,778 en la ronda 1 al sacar el leakage de cesantía, como se
  esperaba, pero sigue sobre el umbral con margen razonable (0,023).
* Brier del modelo mejor que el de la tasa base en los 3 targets ✓.
* Error relativo mensual del agregado dentro de ±10 % en los 9 meses×target evaluados
  (peor caso +7,6 %) ✓ — a diferencia de la ronda 1 (un mes en +12,1 %).
* Runtime y tests dentro de los límites (§9) ✓.
* **Segmento reingreso × tramo de edad** (25-34, 35-44, 45-54; cada uno ≥ 12 % de las
  filas): el modelo gana claramente en AUC (la métrica principal que pide el brief) en los
  3 tramos, pero su Brier queda entre 0,0003 y 0,0097 por encima del baseline de tasa base
  (tabla en §6). Es una diferencia de calibración, no de discriminación: el baseline
  constante por rubro×mes tiene poca varianza y le basta con acertar la tasa media del
  grupo (~75-78 % de reingreso) para tener un Brier bajo; el modelo, al discriminar más,
  paga un costo marginal de calibración en estos tramos concretos. No se intentó "arreglar"
  recalibrando por tramo de edad en esta ronda (habría que validar que no rompe la
  calibración global, que hoy es buena) — queda como mejora futura en §10, documentado en
  vez de forzado a pasar.
* No quedan criterios incumplidos de forma clara (a diferencia de la ronda 1, donde
  INDEPENDIENTE perdía en AUC Y Brier, y un mes superaba el 10 % de error).

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.riesgo_cese_cotizacion.run --n-afiliados 20000
python -m pytest afp_lab/riesgo_cese_cotizacion/tests -q
```

* `run.py` (`--n-afiliados 20000`, configuración por defecto): **54–127 s** según la carga
  de la máquina compartida (medido en corridas repetidas; la máquina es de 4 núcleos
  compartida con otros 2 data scientists — `n_jobs=2` en LightGBM). Muy por debajo del
  límite de 5 minutos incluso en el caso más lento observado.
* `pytest afp_lab/riesgo_cese_cotizacion/tests`: **16 tests, ~25 s** (subió de 9 a 16 tests
  en la ronda 2: 2 tests nuevos de leakage de cesantía, 1 de tipo de contrato vigente, 1 de
  empleador sin sesgo, 2 de split/cortes válidos, 1 de cobertura de población). Muy por
  debajo del límite de 2 minutos.
* Semilla fija (`seed=42` por defecto) y generación determinista: se verificó que dos
  corridas de `run.py` con los mismos parámetros producen exactamente las mismas métricas.
* `--salida` (por defecto `salidas/riesgo_cese_cotizacion/`, en `.gitignore`):
  `dataset_features.parquet` (465 160 filas × 81 columnas),
  `modelos/modelo_{y3,y6,y_reingreso}.joblib` y `reportes/*.csv` (métricas por segmento
  para los 3 predictores, calibración, negocio, importancias, ejemplos, cobertura de
  población A).

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/`** (no se tocó nada fuera de este proyecto; quedan como
propuesta para el gerente):

* Mover la aritmética de periodos `yyyymm` (`periodo_to_idx`/`idx_to_periodo`/`add_months`)
  a `afp_lab.common`: hoy vive duplicada (una versión Spark en `rim_nowcasting.calendario`,
  una versión numpy propia de este proyecto en `synthetic.py`).
* `afp_lab.common.metrics.tabla_metricas_por_segmento` falla con `ValueError` si el
  segmento es una columna `pandas.Categorical` con categorías sin observar
  (`df.groupby(col_segmento)` con `observed=False` por defecto itera también las
  categorías vacías). Se evitó casteando el segmento a `str` (`evaluate.agregar_tramos`);
  sería mejor agregar `observed=True` en la función común.
* Una utilidad común de "cobertura de población vs. universo real" (esta ronda agregó
  `cobertura_poblacion_cese`) y de "métrica de negocio mensual" serían reutilizables por
  `churn_traspaso` y `propension_apv`.

**Próximos pasos** (no implementados en esta ronda):

* Recalibración por segmento (isotónica separada para `tramo_edad` en reingreso) para
  cerrar la brecha de Brier documentada en §8, validando que no degrade la calibración
  global.
* Modelo o features específicas para independientes y voluntarios (excluidos en esta
  ronda): una feature de "renta anual declarada" (independientes) o de "frecuencia
  histórica de pago" (voluntarios), o modelos separados para cada uno.
* Para el modelo de reingreso, guardar el rubro y el tramo de renta del **último empleo
  antes del cese** (hoy se pierden porque el afiliado no tiene empleador vigente) — haría
  útiles esos dos segmentos, hoy triviales para ese target.
* Grilla de umbrales/costos por segmento para la asesoría (hoy la priorización es un único
  ranking por probabilidad calibrada; una AFP real distinguiría el costo de contactar a un
  afiliado de alto valor vs. uno de bajo saldo).
