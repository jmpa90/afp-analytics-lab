# 04 · Riesgo de cese de cotización y probabilidad de reingreso

Responsable: DS-3. Ronda 3 (aprobada con correcciones menores en ronda 2). Revisado contra
`CONVENCIONES.md` §7, `docs/revisiones/ronda_1/riesgo_cese_cotizacion.md` y
`docs/revisiones/ronda_2/riesgo_cese_cotizacion.md`.

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

## Correcciones de la ronda 3 (revisión del gerente: "aprobada con correcciones menores")

La ronda 2 fue **aprobada** (leakage cerrado por construcción, gap correcto, tabla
reproduce dígito a dígito), con esta lista de correcciones antes de integrar a `main`:

* **A (mayor) — cobertura del agregado de negocio**: sin independientes ni voluntarios, el
  agregado de negocio describía solo el **87,9 %** de los ceses reales a 3 meses y el
  **84,4 %** a 6 meses (medido sobre los 24 cortes completos, no solo los 3 de test — más
  estable: independientes son 7,1 % de los cotizantes activos y 8,1-12,9 % de los ceses
  reales; voluntarios son 0,6 % de los cotizantes pero su tasa de cese es **44,5 %** a 3
  meses, casi 10× la de un dependiente). Se implementó la corrección completa (no solo la
  mínima): `run._negocio_agregado_cartera_completa` proyecta independientes y voluntarios
  con un baseline de tasa base por `tipo_afiliado × mes_corte` (no por rubro: para ellos
  `rubro` es siempre "SIN_EMPLEADOR") entrenado con los mismos cortes de train/val que el
  modelo, y los suma al agregado del modelo. También se publica, por separado del factor de
  censura de recepción (1,0496), el **factor de corrección por exclusión** = 1 / cobertura:
  **1,138** (y3) y **1,184** (y6) (`reportes/diagnostico_tipo_afiliado_{y3,y6}.csv`). Hallazgo
  no buscado, documentado en vez de ocultado: sumar la proyección de independientes/
  voluntarios MEJORA la cobertura pero EMPEORA la estabilidad mensual del agregado para
  y6 (dos de los tres meses de test superan el ±10 % una vez incluidos: -24,8 % y +18,6 %),
  porque la población excluida es chica y su baseline no tiene features — ver detalle en §6
  y §8.
* **B (menor) — `emp_sin_planilla_conocida` idénticamente 0**: la definición anterior
  comparaba el empleador de referencia contra sí mismo en el mismo mes en que fue elegido
  (siempre tiene planilla ahí, por construcción — circular). Redefinida sobre toda la
  ventana de 13 meses (`hc_raw.isna().any(axis=1)`): ahora enciende en 951 de 465 160 filas.
* **G (menor, barato) — cesantía no se apagaba al reingresar**: `cesantia_en_curso`/
  `giros_restantes`/`tasa_actual` asumían "sigue cesante" incluso con evidencia YA CONOCIDA
  de reingreso (un INICIO AFC posterior al término, o RIM conocida > 0 en un mes posterior
  al término). Corregido en `construir_features` (`ya_reingreso_conocido`); docstrings
  actualizados para describir lo implementado, no solo la intención.
* **Documentación** (sin cambios de código, salvo donde se indica): la brecha de Brier del
  reingreso afecta a los tramos 25-54 años, que son el **73,1 %** de esa población, y el AUC
  global (0,72) lo sostiene el gradiente de edad/jubilación: **dentro de la edad laboral
  (25-54) el AUC cae a 0,602** (§6); se agregó `evaluate.ece`/`es_calibracion_monotona` — la
  calibración de reingreso NO es monótona por decil (ECE 0,070; cese sí, ECE 0,004-0,005) y
  se restringe su uso a agregados y ranking, no a lectura literal por individuo (§6); la
  "cola opuesta" de §7 se filtra por `meses_hasta_pension`: el 10 % de menor probabilidad de
  reingreso es **98,8 % mayores de 55 años**, así que sin ese filtro la recomendación de
  campaña era, en la práctica, solo "personas a punto de jubilar" (§7); se documenta que el
  gap `2h` deja **10 de 24** cortes de train/val para los targets a 6 meses, contra 15 con
  el gap de 1 mes de la ronda 1 (**-33 %**, "un tercio" — con `periodo_inicio=202201` fijo
  no hay más cortes disponibles para compensarlo; adelantarlo a `202101` queda como
  recomendación en §10, no aplicada esta ronda) (§5, §8); "gana en todos los rubros" se
  verificó con 2 semillas adicionales a n=20 000 (gana en las 13, ambas semillas) y con 3 a
  n=10 000 (más ruidoso: gana o empata en 2 semillas, pierde en 4 rubros con la tercera,
  ninguno coincide exactamente con los 3 que reportó la ronda 2) — la garantía es sólida a
  la escala de aceptación (n=20 000), no a n=10 000 (§6, §8); runtime medido de esta corrida
  de referencia: **50,1 s** (varió 50-152 s entre corridas por la carga de la máquina
  compartida); mediana de `emp_crecimiento_12` a nivel de fila: **1,031** (§8, §9).

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
* **Cobertura de la población A por censura de recepción**: `pop_cese` (RIM *conocida*) es,
  por diseño, un subconjunto de los cotizantes *reales del mismo grupo* — cubre en promedio
  el **95,3 %** (`cobertura_poblacion_cese`, `reportes/cobertura_poblacion_a.csv`); el 4,7 %
  restante son cotizaciones reales cuya planilla de `corte-2` aún no se conoce. Para
  proyectar agregados al universo completo de cotizantes DEPENDIENTES, el número esperado
  debe escalarse por el factor de corrección promedio **1,0496** (`= 1/cobertura`).
* **Cobertura de la población A por exclusión de tipos** (ronda 3, corrección A): lo
  anterior compara dependientes contra dependientes; independientes y voluntarios (fuera de
  ambos modelos) también cesan, y no en proporción a su peso poblacional. Sobre el dataset
  completo (24 cortes, más estable que mirar solo los 3 de test): dependientes/pensionados
  activos son el **91,7 %** de los cotizantes activos pero solo el **87,9 %** (y3) / **84,4
  %** (y6) de los ceses reales; independientes son 7,1 % de los cotizantes y 8,1-12,9 % de
  los ceses (tasa de cese 7,5 %, más alta que el 6,3 % de un dependiente); voluntarios son
  0,6 % de los cotizantes pero su tasa de cese es **44,5 %** (y3) / 18,6 % (y6) — pagan de
  forma esporádica, así que "terminan de cotizar" con mucha frecuencia sin que eso sea una
  cesantía real. Factor de corrección por exclusión (`= 1/cobertura`, **distinto** del de
  censura de recepción): **1,138** (y3), **1,184** (y6)
  (`reportes/diagnostico_tipo_afiliado_{y3,y6}.csv`). `run._negocio_agregado_cartera_completa`
  usa esta información para sumar una proyección de independientes/voluntarios (baseline
  por `tipo_afiliado × mes_corte`) al agregado del modelo — ver §6 y §7.

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
| Relación laboral | `meses_con_empleador`, `n_empleadores_12`, `tipo_contrato` (INICIO AFC de la relación **vigente**, no de cualquier inicio pasado — corrección de ronda 2), `emp_n_trabajadores`, `emp_crecimiento_12`, `emp_tasa_salida_12`, `emp_sin_planilla_conocida` (1 si en ALGÚN mes de la ventana de 13 meses el empleador de referencia no tuvo planilla conocida; corrección de ronda 3, ver §8) | Empleador de referencia y sus agregados, con RIM conocida ≤ `corte-1` y **corregidos por madurez de recepción** (ronda 2, ver §8) |
| Eventos y cesantía | `afc_termino_l1/l2`, `afc_inicio_l1/l2`, `meses_desde_afc_termino/inicio`, `dias_licencia_l1/l2`, `licencia_en_curso`, `cesantia_en_curso`, `cesantia_giros_restantes`, `cesantia_tasa_actual` (derivadas analíticamente del término AFC + tipo de contrato + tasas legales; se apagan si hay un INICIO conocido posterior al término o RIM conocida > 0 en un mes posterior al término — corrección de ronda 3, ver §8) | Conocidos con recepción ≤ `corte` y periodo ≤ `corte-1` |
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
  test. **Costo del gap correcto** (documentado en ronda 3): para los targets a 6 meses
  (`y6`, `y_reingreso`) quedan **10 de 24** cortes disponibles para train/val, contra 15 con
  el gap real de 1 mes (el bug de la ronda 1) — una caída del **33 %**. Con
  `periodo_inicio=202201` fijo no hay más cortes que "recuperar" (ya se usan todos los que
  caben, ver §8); adelantar `periodo_inicio` a `202101` daría ~12 cortes más y se deja como
  recomendación en §10, no aplicada esta ronda.
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
| **Cese 3 m** | **Modelo** | 41 179 | 0,0675 | **0,868** | 0,430 | **0,0458** | 6,32 | 0,598 |
| | Baseline tasa base | 41 179 | 0,0675 | 0,700 | 0,139 | 0,0606 | 2,64 | 0,308 |
| | Baseline regla de negocio | 41 179 | 0,0675 | 0,798 | 0,325 | 0,0835 | 6,09 | 0,562 |
| **Cese 6 m** | **Modelo** | 41 179 | 0,0432 | **0,859** | 0,312 | **0,0343** | 6,34 | 0,594 |
| | Baseline tasa base | 41 179 | 0,0432 | 0,574 | 0,055 | 0,0412 | 1,76 | 0,090 |
| | Baseline regla de negocio | 41 179 | 0,0432 | 0,795 | 0,214 | 0,0892 | 6,18 | 0,557 |
| **Reingreso** | **Modelo** | 7 747 | 0,6619 | **0,720** | 0,822 | **0,1924** | 1,43 | 0,292 |
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
  ej. 45-54: AUC 0,611 vs. 0,503/0,487), pero en **Brier** queda muy levemente por encima
  del baseline de tasa base en los 3 tramos grandes de edad laboral (25-34: 0,1871 vs.
  0,1772; 35-44: 0,1961 vs. 0,1912; 45-54: 0,1874 vs. 0,1862 — diferencias de 0,0012 a
  0,0099). Se
  documenta en vez de ocultarlo (ver §8): no es del mismo tipo que la falla de
  INDEPENDIENTE en la ronda 1 (ahí el modelo perdía en AUC Y Brier por márgenes grandes;
  aquí gana claramente en la métrica principal del brief, AUC, y pierde por una fracción
  mínima en Brier). **Documentación de ronda 3**: esos 3 tramos (25-54) son el **73,1 %**
  de la población de reingreso, y ahí es exactamente donde vive la brecha de Brier — no es
  un tramo marginal. Además, el AUC global del modelo de reingreso (0,720) está
  sostenido en gran parte por el **gradiente de edad/jubilación**: restringido a la edad
  laboral (25-54 años, ese mismo 73,1 %), el AUC cae a **0,602**; en mayores de 55 (26,9 %
  de las filas, que casi nunca reingresan) sube a 0,733. En otras palabras, el modelo
  discrimina bien "está por jubilar vs. no", pero discrimina mucho más débilmente DENTRO de
  la población en edad laboral, que es la que más importa para una campaña de reingreso.
  Para reingreso, `tramo_renta` es siempre "sin_renta" y `rubro` siempre "SIN_EMPLEADOR"
  por construcción de la población (sin RIM ni empleador vigentes): sigue siendo una
  limitación de diseño, no un error (ver propuesta en §10).
* **Rubro y tipo de contrato**: el modelo gana en AUC y Brier a ambos baselines en todos los
  rubros y tipos de contrato con ≥ 5 % de las filas, en los 3 targets, con `n_afiliados=
  20 000` y `seed=42` (la corrida de referencia) **y** con `seed=7` (verificación
  independiente para este informe, ronda 3, ítem H): gana en las 13 categorías de rubro en
  ambas semillas para `y6`. A **n=10 000** (una escala 2× menor) la comparación es más
  ruidosa: con semillas 7 y 99 el modelo gana o empata (una diferencia de 0,001 en
  K_FINANZAS con semilla 7, no una pérdida real); con semilla 2024 pierde en 4 rubros
  (A_AGRICULTURA, C_MANUFACTURA, N_SERVICIOS, P_EDUCACION; márgenes de 0,006 a 0,034 AUC) —
  ninguno coincide exactamente con los 3 que reportó la revisión de ronda 2 bajo la versión
  anterior del pipeline, consistente con que es ruido de muestra chica y no una debilidad
  estructural de un rubro puntual. La garantía "gana en todos los rubros" se sostiene con
  confianza a la escala de aceptación (`n_afiliados=20 000`) pero no debería citarse como
  válida a `n=10 000`.

### Calibración

Brier del modelo mejor que el de la tasa base en los 3 targets (tabla principal). Curvas de
calibración por decil (`calibracion_*.csv`): en cese 3 m el error absoluto predicho-observado
es ≤ 0,017 en los 10 deciles. En reingreso el ajuste es más ruidoso en los deciles
intermedios (error hasta 0,19 en el decil 7) pero el extremo superior e inferior (deciles 0
y 9, los más accionables para priorizar contactos) calibran a menos de 0,033.

**ECE y monotonía** (ronda 3, `evaluate.ece`/`es_calibracion_monotona`): cese 3 m ECE
0,0036, cese 6 m ECE 0,0048 — ambos muy bien calibrados en promedio, y `es_calibracion_
monotona` marca `False` en los tres targets (con un umbral estricto de no-decrecimiento),
pero por razones muy distintas: en cese es un único decil con una caída de ruido de ~0,001
a ~0,002 (p. ej. y3 decil 3: 0,0124 observado vs. 0,0136 del decil anterior), invisible en
la práctica. En reingreso, ECE **0,070** (20× peor) y la caída es sustancial y sostenida:
los deciles 5 a 7 predicen entre 0,76 y 0,89 pero la tasa observada se estanca/cae alrededor
de 0,71-0,72 en los tres (en vez de seguir subiendo), con errores absolutos de 0,04 a 0,18.
Conclusión práctica: la probabilidad calibrada de reingreso sirve para **agregar** (sumar
sobre una cartera, como en la métrica de negocio) y para **rankear** (ordenar por riesgo,
como en el top 10 %), pero en el rango medio-alto **no debería leerse literalmente como "la
probabilidad" de una fila individual** — un 0,89 no es más fiable que un 0,76 ahí. Cese, en
cambio, sí es fiable fila a fila (la única imperfección es la caída de ruido ya descrita).

### Importancias (gain)

* **Cese 3 m**: `cesantia_en_curso` (42,6 %), `cesantia_tasa_actual` (15,4 %), `rubro`
  (4,7 %), `edad`, `meses_desde_afc_termino`, `meses_desde_afiliacion`, `meses_hasta_pension`,
  `region`.
* **Cese 6 m**: mismo orden con `cesantia_en_curso` (37,5 %) al frente; el resto (edad,
  antigüedad, rubro, meses hasta pensión, `meses_desde_afc_termino`, tamaño del empleador) se
  reparte de forma más pareja que a 3 meses.
* **Reingreso**: `edad` (15,9 %), `meses_hasta_pension` (15,1 %), `meses_desde_afc_inicio`
  (12,4 %), `meses_desde_afiliacion` (9,9 %), `max_12`, `region`, `media_24`,
  `n_cotiza_24`.

**Cambio de ronda 3**: antes de la corrección G, `meses_desde_afc_termino` dominaba
(49,5 % en cese 3 m) y `cesantia_en_curso` era una feature secundaria; al apagar la
cobertura de cesantía cuando ya hay evidencia de reingreso, `cesantia_en_curso` pasó a ser
la señal principal (42,6 %) y `meses_desde_afc_termino` cayó a un rol menor. Tiene sentido:
antes, "cuántos meses desde el término" mezclaba a quienes seguían cesantes con quienes ya
habían vuelto a trabajar; ahora la bandera booleana ya filtra ese segundo grupo, así que es
más informativa que el conteo de meses por sí solo. Ninguna de las dos depende de la RIM
futura (a diferencia de la ronda 1: ver corrección #1 y los tests de leakage en §8): se
calculan solo con el término AFC conocido hasta `corte-1`, la evidencia de reingreso ya
conocida, y las tasas legales.

### Ejemplos interpretados (cese 3 m; ronda 2: elegidos por el puntaje CRUDO, sin la
meseta de la calibración isotónica — menor #12)

Uno por rubro estacional (`reportes/ejemplos_interpretados.csv`), con probabilidad
calibrada y cruda por separado (las contribuciones SHAP explican la cruda, en log-odds):

* Afiliado en `A_AGRICULTURA` (cesantía en curso): `pred_calibrada=0,93`,
  `pred_cruda=0,82`, `y3_real=1`. Contribuciones: `cesantia_en_curso` (+1,95), `edad`
  (+0,80), `meses_hasta_pension` (+0,77).
* Afiliado en `G_COMERCIO`: `pred_calibrada=1,00`, `pred_cruda=0,91`, `y3_real=1`.
  Contribuciones: `cesantia_en_curso` (+2,05), `edad` (+0,83), `meses_hasta_pension`
  (+0,73).

Ronda 3: tras la corrección G (§8), los 5 ejemplos de mayor puntaje crudo por rubro
resultaron todos verdaderos positivos (`y3_real=1`) — antes, con `meses_desde_afc_termino`
como señal principal, uno de los 5 (`F_CONSTRUCCION`) era un falso positivo (alguien con
término AFC reciente que ya había conseguido empleo antes de los 3 meses, algo que la
versión corregida de la feature ahora detecta y descuenta). La imperfección de la
discriminación (AUC 0,868, no 1,0) sigue siendo real y se documenta con números en la tabla
de calibración (decil 9: 0,445 predicho vs. 0,426 observado) en vez de forzar un ejemplo de
falso positivo que ya no aparece naturalmente entre los de mayor puntaje.

### Métrica de negocio

Contactando al **10 % de mayor riesgo** en test: se captura el **63,1 %** de los ceses a
3 meses, **63,5 %** de los ceses a 6 meses y **14,3 %** de los reingresos (la tasa base de
reingreso es 66 %, así que el 10 % de mayor riesgo son en su mayoría los que de todas
formas iban a reingresar rápido; ver interpretación de negocio en §7).

Número esperado (suma de probabilidades calibradas) vs. real de casos, **por mes de corte**
(el brief pide el error MENSUAL, no un promedio del trimestre de test — menor #11, ronda 2:
la ronda 1 evaluaba el criterio de aceptación contra un total trimestral en vez de mes a
mes):

| Target | 202601 | 202602 | 202603 |
|---|---|---|---|
| Cese 3 m | -2,7 % | +2,4 % | +7,2 % |
| Cese 6 m | -3,7 % | +3,1 % | +8,2 % |
| Reingreso | +1,9 % | +3,3 % | +3,8 % |

Los **9 meses × 3 targets = 9 números** están todos dentro de ±10 % (el peor es +8,2 %, en
cese 6 m 202603): a diferencia de la ronda 1 (un mes de cese 6 m llegaba a +12,1 %), la
corrección de la población B y del gap del split también mejoró la estabilidad del
agregado mes a mes. Detalle completo en `reportes/negocio_agregado_*.csv` (esta tabla es
siempre sobre la población del modelo: dependientes/pensionados activos; ver más abajo la
extensión a la cartera completa para cese).

**Cartera completa** (ronda 3, corrección A — `reportes/negocio_agregado_total_{y3,y6}.csv`):
sumando la proyección de independientes/voluntarios (baseline `tipo_afiliado × mes_corte`)
al agregado del modelo:

| Target | 202601 | 202602 | 202603 | Cobertura dependientes (test, agregada) |
|---|---|---|---|---|
| Cese 3 m, solo modelo | -2,7 % | +2,4 % | +7,2 % | — |
| Cese 3 m, **cartera completa** | -0,2 % | +1,2 % | +6,9 % | 79,2 % |
| Cese 6 m, solo modelo | -3,7 % | +3,1 % | +8,2 % | — |
| Cese 6 m, **cartera completa** | -1,5 % | **-24,8 %** | **+18,6 %** | 72,5 % |

**Hallazgo no buscado, documentado sin maquillar**: para cese 3 m, sumar la proyección de
independientes/voluntarios en realidad **mejora** el ajuste mensual (el error se acerca a
0). Para cese 6 m, en cambio, lo **empeora**: dos de los tres meses de test superan largamente
el ±10 % (-24,8 % y +18,6 %). La causa es que la población excluida es chica
(~1 130-1 140 filas por mes en test) y su baseline no tiene features de historia (solo
`tipo_afiliado × mes_corte`): el número real de voluntarios que cesan a 6 meses saltó de 331
en 202602 a 15 en 202603 en esta muestra, algo que un baseline sin historia individual no
puede anticipar. En consecuencia: el criterio de aceptación del brief (±10 % mensual) se
sigue cumpliendo con margen sobre la población que el modelo efectivamente predice
(dependientes/pensionados activos, la tabla "solo modelo"); la vista de "cartera completa"
es la más honesta para *cobertura* (informa qué fracción de los ceses reales totales queda
efectivamente proyectada) pero introduce ruido mensual adicional en `y6` que la AFP debería
conocer antes de usarla para un compromiso operativo mes a mes (ver más en §7 y §8).

## 7. Lectura de negocio

* **Recaudación esperada**: el "número esperado de ceses" calibrado por mes (tabla de §6)
  es utilizable para proyectar densidad de cotización y flujo de recaudación **sobre la
  población dependiente/pensionada** (error mensual entre -3,7 % y +8,2 % en los 9
  meses×target del modelo, sin extender a la cartera completa). Hay dos correcciones
  distintas para llegar al universo completo de TODOS los cotizantes (§3): (a) censura de
  recepción (factor **≈1,05**, corrige que `pop_cese` no ve el 100 % de los dependientes
  reales); (b) exclusión de independientes/voluntarios (factor **≈1,14-1,18**, corrige que
  ellos ni siquiera entran al modelo). Ambas se pueden aplicar juntas como un factor rápido,
  o se puede usar directamente la tabla de "cartera completa" de §6, que ya proyecta a
  independientes/voluntarios con su propio baseline en vez de un factor fijo — más completa,
  pero (ronda 3, hallazgo documentado en §6) más ruidosa mes a mes para `y6` porque esa
  población es chica y su baseline no tiene features de historia individual.
* **Priorización de contactos**: contactar al 10 % de mayor riesgo de cese captura ~6 de
  cada 10 ceses reales a 3 y a 6 meses — una mejora sustancial frente a contactar al azar
  (que capturaría ~10 %). Para reingreso, el 10 % de mayor riesgo no es el segmento más
  útil para contactar (la tasa base ya es 66 %: la mayoría reingresa igual, con o sin
  contacto); una campaña de reingreso rendiría más si se apunta a la **cola opuesta**
  (menor probabilidad de reingreso, para ofrecerles cotización voluntaria/APV antes de que
  la laguna se alargue). **Corrección de ronda 3**: esa cola opuesta, tomada tal cual, NO es
  un buen filtro para una campaña de reingreso al trabajo — el 10 % de menor probabilidad de
  reingreso es **98,8 % mayores de 55 años** (95,0 % con ≤ 5 años para pensionarse, ver §6):
  en la práctica es, casi en su totalidad, "personas a punto de jubilar", no personas en edad
  laboral con dificultades para reinsertarse. La recomendación correcta es **filtrar la cola
  opuesta por `meses_hasta_pension`** (p. ej. excluir a quienes están a ≤ 5 años de
  pensionarse) antes de usarla para una campaña de reingreso/empleo; para el segmento
  cercano a la jubilación, la oferta pertinente es otra (asesoría de retiro, no reingreso
  laboral) y esta misma probabilidad calibrada no es la herramienta adecuada para
  priorizarlos.
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
  un artefacto de recepción, no de dotación real. Corregido, la mediana medida **a nivel de
  fila** (no por empleador — cada fila pesa según cuántos afiliados comparten ese
  empleador) sobre las 318 869 filas con empleador de referencia es **1,031**
  (`test_features_empleador_sin_nan_y_crecimiento_no_sesgado` exige que quede entre 0,95 y
  1,15; el número exacto se corrigió en ronda 3, la ronda 2 citaba ~0,99 de una muestra
  chica sin ponderar por fila).
* `emp_sin_planilla_conocida` (ronda 3, corrección B): la definición de ronda 2 comparaba
  el empleador de referencia contra sí mismo en el mes en que fue elegido como referencia
  (siempre tiene planilla ahí, por construcción: circular, quedaba idénticamente 0 en las
  465 160 filas). Redefinida como "en algún OTRO mes de la ventana de 13 meses ese
  empleador no tuvo planilla conocida" (`hc_raw.isna().any(axis=1)`); enciende en 951 filas.
* La cobertura de cesantía (`cesantia_en_curso`/`giros_restantes`/`tasa_actual`) se apaga si
  hay evidencia YA CONOCIDA de reingreso (ronda 3, corrección G): un INICIO AFC conocido
  posterior al término, o RIM conocida > 0 en un mes estrictamente posterior al término
  (nunca en el propio mes del término ni antes: para un término reciente, `rim_l1` describe
  el sueldo ANTES de terminar y no debe apagar nada). Antes, la feature asumía "sigue
  cesante" indefinidamente sin revisar esta evidencia, aunque ya estuviera disponible as-of.
* `edad` se proyecta desde `edad_inicio` (edad al primer periodo simulado, según
  `synth_base.generar_afiliados_pandas`) igual que en `rim_nowcasting.features`, usando
  `macro["periodo"].min()` como referencia de `periodo_inicio`.
* 3 modelos LightGBM independientes (no multi-tarea): ver justificación en §3.
* **`periodo_inicio=202201` fijo limita cuánto se puede compensar el gap de `2h`** (ronda 3,
  documentación): con la ventana de historia de 26 meses y el rango simulado por defecto
  (`202201`-`202609`, 57 meses), solo caben 24 cortes válidos en total; el gap de `2h` deja
  10 de esos 24 para train/val de los targets a 6 meses (-33 % respecto del gap de 1 mes,
  incorrecto, de la ronda 1 — ver §5). "Usar todos los cortes que caben" (la mitigación de
  ronda 2) ya se está haciendo: no hay más margen sin cambiar el rango simulado. Adelantar
  `periodo_inicio` a `202101` (12 meses más de historia simulada) daría ~12 cortes
  adicionales; no se aplicó esta ronda porque cambia el runtime y los números de referencia
  de todo el documento, y el enunciado de la revisión lo planteó como una sugerencia a
  considerar, no como una corrección obligatoria — queda en §10.

**Criterios de aceptación: estado actual (números reales, sin maquillar)**

* AUC test cese 3 m **0,868** (≥ 0,80 pedido) ✓. AUC test reingreso **0,720** (≥ 0,70
  pedido) ✓ — bajó desde 0,778 en la ronda 1 (al sacar el leakage de cesantía) a 0,723 en
  ronda 2 y a 0,720 en ronda 3 (el apagado de cesantía por reingreso conocido, corrección
  G, también recorta un poco de señal en algunas filas de reingreso); sigue sobre el
  umbral, pero el margen ya es más ajustado (0,020) y, como documenta §6, ese AUC global
  está sostenido en gran parte por el gradiente de edad/jubilación (dentro de la edad
  laboral cae a 0,602).
* Brier del modelo mejor que el de la tasa base en los 3 targets ✓.
* Error relativo mensual del agregado (población del modelo: dependientes/pensionados)
  dentro de ±10 % en los 9 meses×target evaluados (peor caso +8,2 %) ✓ — a diferencia de la
  ronda 1 (un mes en +12,1 %). La vista de "cartera completa" (§6, corrección A) NO cumple
  ese mismo ±10 % en 2 de 3 meses de `y6` (-24,8 % y +18,6 %): es un hallazgo nuevo de esta
  ronda, no un criterio del brief (que no distingue cartera completa de la población
  modelada), pero se documenta igual porque una AFP real leería "cartera completa" como el
  número más relevante para su propio negocio.
* Runtime y tests dentro de los límites (§9) ✓.
* **Segmento reingreso × tramo de edad** (25-34, 35-44, 45-54; el **73,1 %** de esa
  población, no un tramo marginal): el modelo gana claramente en AUC (la métrica principal
  que pide el brief) en los 3 tramos, pero su Brier queda entre 0,0012 y 0,0099 por encima
  del baseline de tasa base (tabla en §6), y dentro de ese mismo rango de edad el AUC cae a
  0,602 (§6). Es una diferencia de calibración y de discriminación acotada, no un fallo del
  tamaño del de INDEPENDIENTE en ronda 1 (que perdía en AUC Y Brier por márgenes grandes en
  toda la población), pero tampoco es un matiz menor: para la parte de la población más
  relevante para una campaña de reingreso (gente en edad laboral), el modelo aporta bastante
  menos de lo que sugiere el AUC agregado. No se intentó "arreglar" recalibrando por tramo
  de edad en esta ronda (habría que validar que no rompe la calibración global) — queda
  como mejora futura en §10.
* **"Gana en todos los rubros"** se sostiene con confianza a `n_afiliados=20 000` (2
  semillas verificadas, ronda 3) pero no a `n=10 000` (3 semillas: 2 sin pérdidas claras, 1
  con pérdidas en 4 rubros) — ver detalle y números en §6. No es un criterio del brief
  (que no fija una escala mínima), pero se documenta porque la ronda 2 de revisión encontró
  pérdidas a esa escala menor y conviene dejar registrado hasta dónde llega la garantía.
* No quedan criterios del brief incumplidos de forma clara (a diferencia de la ronda 1,
  donde INDEPENDIENTE perdía en AUC Y Brier, y un mes superaba el 10 % de error); los puntos
  nuevos de esta ronda (cartera completa en `y6`, Brier/AUC de reingreso en edad laboral,
  robustez a `n=10 000`) son matices que no estaban definidos como criterios explícitos del
  brief, documentados igual por transparencia.

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.riesgo_cese_cotizacion.run --n-afiliados 20000
python -m pytest afp_lab/riesgo_cese_cotizacion/tests -q
```

* `run.py` (`--n-afiliados 20000`, configuración por defecto): **50–152 s** según la carga
  de la máquina compartida (rango medido en corridas repetidas de las rondas 2 y 3, incluida
  una corrida de **152 s** — la máquina es de 4 núcleos compartida con otros 2 data
  scientists, `n_jobs=2` en LightGBM). La corrida de referencia de este informe tomó
  **50,1 s**. Muy por debajo del límite de 5 minutos incluso en el caso más lento observado.
* `pytest afp_lab/riesgo_cese_cotizacion/tests`: **20 tests, ~15 s** (subió de 16 a 20 tests
  en la ronda 3: cobertura de cesantía que se apaga con reingreso conocido, regresión de
  `emp_sin_planilla_conocida` no-idénticamente-cero, cobertura/diagnóstico de cartera
  completa, ECE/monotonía de calibración). Muy por debajo del límite de 2 minutos.
* Semilla fija (`seed=42` por defecto) y generación determinista: se verificó que dos
  corridas de `run.py` con los mismos parámetros producen exactamente las mismas métricas
  (incluidas las nuevas tablas de cartera completa y diagnóstico).
* `--salida` (por defecto `salidas/riesgo_cese_cotizacion/`, en `.gitignore`):
  `dataset_features.parquet` (465 160 filas × 81 columnas),
  `modelos/modelo_{y3,y6,y_reingreso}.joblib` y `reportes/*.csv` (métricas por segmento
  para los 3 predictores, calibración, negocio —incluida la vista de cartera completa para
  cese—, importancias, ejemplos, cobertura de población A y diagnóstico por tipo de
  afiliado).

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
  `cobertura_poblacion_cese` y, en ronda 3, `diagnostico_cese_por_tipo_afiliado`) y de
  "métrica de negocio mensual" serían reutilizables por `churn_traspaso` y
  `propension_apv`.
* Un baseline genérico "tasa base por grupo × mes" reutilizable entre proyectos: en ronda 3
  `model.baseline_tasa_base`/`predecir_tasa_base` se generalizaron para aceptar cualquier
  lista de columnas de agrupación (no solo `rubro × mes_corte`), justamente para poder
  proyectar poblaciones sin rubro (independientes/voluntarios); podría vivir en `common`.

**Próximos pasos** (no implementados en esta ronda):

* Adelantar `periodo_inicio` a `202101` (12 meses más de historia simulada) para dar más
  margen de train/val a los targets de 6 meses bajo el gap correcto de `2h` (hoy 10 de 24
  cortes; ver §5 y §8) — evaluado y no aplicado esta ronda por su impacto en runtime y en
  todos los números de referencia del documento; sugerido explícitamente por la revisión de
  ronda 2.
* Un baseline con algo más de estructura para independientes/voluntarios en el agregado de
  "cartera completa" (§6): hoy es tasa base por `tipo_afiliado × mes_corte`, sin features de
  historia individual, y eso explica el error mensual alto de `y6` en esa vista (§6, §8);
  una feature simple como "cotizó en los últimos 12 meses" para voluntarios ya podría
  ayudar bastante.
* Recalibración por segmento (isotónica separada para `tramo_edad` en reingreso) para
  cerrar la brecha de Brier documentada en §8, validando que no degrade la calibración
  global; dado que la calibración de reingreso no es monótona (§6), conviene revisar el
  esquema de bins antes de recalibrar, no solo separar por tramo.
* Modelo o features específicas para independientes y voluntarios (excluidos de los
  modelos A/B, aunque desde ronda 3 sí se proyectan en el agregado de negocio): una feature
  de "renta anual declarada" (independientes) o de "frecuencia histórica de pago"
  (voluntarios), o modelos separados para cada uno.
* Para el modelo de reingreso, guardar el rubro y el tramo de renta del **último empleo
  antes del cese** (hoy se pierden porque el afiliado no tiene empleador vigente) — haría
  útiles esos dos segmentos, hoy triviales para ese target.
* Un segundo modelo de reingreso restringido a edad laboral (< 55 años, el 73,1 % de la
  población, ver §6), para no mezclar el gradiente de jubilación con la señal de
  reinserción laboral real — el AUC actual (0,720) es engañosamente alto por esa mezcla.
* Verificación de robustez con múltiples semillas como parte del pipeline de aceptación
  (esta ronda se hizo manualmente para el ítem H de la revisión; automatizarlo evitaría
  descubrir sorpresas de este tipo en la próxima ronda).
* Grilla de umbrales/costos por segmento para la asesoría (hoy la priorización es un único
  ranking por probabilidad calibrada; una AFP real distinguiría el costo de contactar a un
  afiliado de alto valor vs. uno de bajo saldo).
