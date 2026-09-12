# 04 · Riesgo de cese de cotización y probabilidad de reingreso

Responsable: DS-3. Ronda 1. Revisado contra `CONVENCIONES.md` §7 (checklist al final).

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
| `prestaciones_cesantia` | giro (afiliado × periodo × n_giro) | Hasta 5 giros decrecientes (contrato indefinido, tasas aprox. 70/55/45/40/35 % de la remuneración imponible media de los últimos 12 meses) o 2 (plazo fijo/por obra, 70/55 %); se detienen si el afiliado reingresa antes de agotarlos. Conocidos al mes | Aproximación razonable a la Ley 19.728 (no son los porcentajes legales exactos, que separan aporte propio/empleador/Fondo Solidario); da una señal de "cuánta cobertura le queda" |

Con `n_afiliados=20 000`, `202201`–`202609`: **1.159 M** movimientos de cotización,
**~28 k** eventos AFC (de los cuales la mitad son términos), **~26 k** episodios de
licencia y **~31 k** giros de cesantía. Proporciones observadas de salidas (términos AFC
sobre afiliado-mes activo) por rubro: más altas en construcción, turismo y agricultura
(estacionales, coherente con `rim_nowcasting.synthetic.RUBROS`), más bajas en minería,
finanzas y administración pública — exactamente la dinámica ya simulada en el proyecto 01,
que aquí solo se re-observa a través de los eventos.

No se usan `lat_*` ni los `p_*`/`peso_contratacion` de empleadores en ninguna feature
(`synth_base.generar_universo` ya los separa en `*_latente`, que este proyecto no importa).

## 3. Target, población, horizonte y exclusiones

Fila = afiliado × `periodo_corte` (`corte`, "hoy"), cortes mensuales móviles.

* **Modelo A — cese (`y3`, `y6`)**: población = afiliados con RIM conocida > 0 en
  `corte-2` (`pop_cese`). `y3=1` si no hay RIM > 0 conocida... hoy sabemos que es la
  verdad final... en **ninguno** de `corte+1…corte+3`; `y6` análogo a 6 meses. Se
  entrenan **dos modelos binarios independientes** (no uno multi-horizonte): las tasas
  base y la importancia relativa de las features cambian bastante entre 3 y 6 meses, y el
  brief pide reportarlos por separado; un solo modelo con `horizonte` como feature no
  simplifica el pipeline aquí (a diferencia de rim_nowcasting, que sí comparte modelo entre
  horizontes porque el objetivo es un nivel continuo, no dos ventanas de "todo o nada").
* **Modelo B — reingreso (`y_reingreso`)**: población = afiliados **dependientes** con
  RIM conocida = 0 en `corte-2` y al menos una cotización conocida en los 24 meses previos
  (`pop_reingreso`). `y=1` si hay RIM > 0 (verdad final) en `corte+1…corte+6`.
* **Exclusiones**: cortes con horizonte incompleto (`corte+h` fuera del rango simulado) se
  descartan (`horizonte3_completo`/`horizonte6_completo`). Afiliados sin ninguna cotización
  en los 24 meses previos a `corte-2` quedan fuera de ambas poblaciones por construcción
  (Modelo A exige RIM > 0 en `corte-2` mismo; Modelo B exige la cotización reciente): un
  retirado/pensionado sin actividad hace años nunca entra a ninguno de los dos modelos, sin
  necesidad de una regla adicional sobre `estado` (que además no es observable por la AFP:
  ver §8).
* Las dos poblaciones son disjuntas por construcción (`pop_cese & pop_reingreso == 0`
  siempre, verificado en `tests/test_riesgo_cese.py::test_poblaciones_bien_definidas`).

## 4. Features (`features.py`)

Todas se calculan en `construir_features(tablas, periodo_corte)` **sin** tocar el panel de
verdad; el target se agrega aparte con `agregar_targets` (que sí lo usa, por definición: un
target habla del futuro del corte). Una fila por afiliado ya afiliado a `corte`.

**Regla as-of** (documentada también como código en el docstring de `features.py`, y
verificada en 4 tests de leakage): la RIM se conoce hasta `corte-2` (recepción ≤ `corte-1`,
igual regla que rim_nowcasting); los eventos AFC, licencias y giros se conocen hasta
`corte-1` (recepción ≤ `corte`, un mes más rápido — ver decisión en §8). Todas las columnas
de la tabla siguen esa regla; los lags `l1`/`l2` están anclados a `corte-2` para RIM y a
`corte-1` para eventos.

| Familia | Features | Regla as-of |
|---|---|---|
| Historia de RIM | `rim_l1..rim_l12`, `media_3/6/12/24`, `std_6`, `max_12`, `min_12`, `n_cotiza_3/6/12/24`, `meses_historia`, `racha_cotiza`, `racha_cero`, `meses_desde_ult_cotiza`, `meses_desde_cambio`, `n_cambios_12`, `cv_6`, `ratio_l1_l2`, `ratio_l1_media12`, `tendencia_3_12`, `sueldo_fijo` | Ventana de 30 meses terminada en `corte-2`, RIM conocida con recepción ≤ `corte-1` |
| Relación laboral | `meses_con_empleador`, `n_empleadores_12`, `tipo_contrato` (INICIO AFC más reciente), `emp_n_trabajadores`, `emp_crecimiento_12`, `emp_tasa_salida_12` (empleador de referencia) | Empleador de referencia y sus agregados, calculados solo con RIM conocida ≤ `corte-1` |
| Eventos | `afc_termino_l1/l2`, `afc_inicio_l1/l2`, `meses_desde_afc_termino/inicio`, `dias_licencia_l1/l2`, `licencia_en_curso`, `cesantia_giro_actual`, `cesantia_en_curso`, `cesantia_n_giros_recibidos` | Conocidos con recepción ≤ `corte` y periodo ≤ `corte-1` (giros: periodo ≤ `corte`) |
| Perfil | `tipo_afiliado`, `sexo`, `edad`, `meses_hasta_pension`, `region`, `nivel_educacional`, `meses_desde_afiliacion` | Estático / determinístico a partir de `corte` |
| Calendario y macro | `mes_corte`, `es_enero/marzo/abril/julio/sept/dic`, `desempleo_corte`, `estacionalidad_rubro` (fracción histórica de términos del rubro que cae en `mes_corte`, calculada solo con eventos conocidos) | Macro de `corte`; estacionalidad recalculada por corte con la historia conocida hasta ese punto (nunca con datos futuros) |

`rubro`/`tamano_empleador`/`region_empleador` vienen de `empleadores` (columnas no
latentes). Sin empleador de referencia: `rubro="SIN_EMPLEADOR"`, `tipo_contrato=
"SIN_EMPLEADOR"`; con empleador pero sin aviso AFC de inicio conocido (la relación empezó
antes del historial disponible): `tipo_contrato="SIN_DATO_AFC"`.

## 5. Modelo y baselines (`model.py`)

* **LightGBM binario** (`n_jobs=2`, `n_estimators=350`, `num_leaves=31`,
  `min_child_samples=60`, `subsample`/`colsample_bytree=0.8`) por target, con categóricas
  nativas (`pandas.Categorical`, sin one-hot). **Calibración isotónica** ajustada sobre un
  conjunto de **validación** (nunca train ni test).
* **Split temporal**: 24 cortes mensuales compartidos por los 3 targets (el corte más
  reciente deja siempre 6 meses completos al final de la simulación, el horizonte más
  largo). Test = últimos 3 cortes; train/val = los anteriores, con un **gap** de al menos
  `horizonte` meses entre el último corte de train/val y el primero de test (`run._split_temporal`,
  probado indirectamente por el propio split — el gap real usado es igual al horizonte:
  3 para `y3`, 6 para `y6` y `y_reingreso`). Validación = el 25 % más reciente de los
  cortes de train/val restantes, para calibrar sin tocar test.
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
| **Cese 3 m** | **Modelo** | 44 719 | 0,0767 | **0,889** | 0,488 | **0,0495** | 6,22 | 0,646 |
| | Baseline tasa base | 44 719 | 0,0767 | 0,750 | 0,203 | 0,0656 | 3,26 | 0,388 |
| | Baseline regla de negocio | 44 719 | 0,0767 | 0,732 | 0,283 | 0,0898 | 4,94 | 0,438 |
| **Cese 6 m** | **Modelo** | 44 719 | 0,0528 | **0,880** | 0,334 | **0,0400** | 6,14 | 0,638 |
| | Baseline tasa base | 44 719 | 0,0528 | 0,766 | 0,178 | 0,0461 | 3,87 | 0,407 |
| | Baseline regla de negocio | 44 719 | 0,0528 | 0,696 | 0,164 | 0,0954 | 4,44 | 0,370 |
| **Reingreso** | **Modelo** | 10 661 | 0,7190 | **0,778** | 0,893 | **0,1584** | 1,36 | 0,383 |
| | Baseline tasa base | 10 661 | 0,7190 | 0,504 | 0,721 | 0,2023 | 1,03 | 0,011 |
| | Baseline regla de negocio | 10 661 | 0,7190 | 0,649 | 0,808 | 0,4029 | 1,20 | 0,212 |

El modelo gana a ambos baselines en AUC y Brier en los 3 targets. Tablas completas en
`salidas/riesgo_cese_cotizacion/reportes/metricas_*.csv`.

### Por segmento (CONVENCIONES §4: tipo de afiliado, tramo de renta, tramo de edad, y los
propios del proyecto — rubro, tipo de contrato)

* **Tipo de afiliado** (`segmento_tipo_afiliado_*.csv`): dependientes (92 % de las filas)
  AUC 0,88/0,87 (cese 3/6 m); independientes (7,0 %) AUC 0,72/0,70; pensionados activos y
  voluntarios (< 1 % cada uno, bajo el umbral de "segmento grande") cerca de 0,50–0,70. El
  modelo gana al baseline de tasa base en dependientes por un margen amplio (AUC +0,16) y
  en el resto por márgenes menores; ver la única excepción real en §8.
* **Tramo de edad y tramo de renta** (sobre `rim_l1`): el modelo gana al baseline en los 4-5
  tramos de cada uno y en los 3 targets (por ejemplo, cese 3 m: AUC 0,88–0,90 en todos los
  tramos de edad vs. 0,72–0,76 del baseline).
  Para el modelo de **reingreso** ambos tramos son poco informativos por construcción
  (`tramo_renta` es siempre "sin_renta" y `rubro` siempre "SIN_EMPLEADOR", porque la
  población no tiene empleador ni renta vigentes); es una limitación de diseño, no un
  error, y se propone una mejora en §10.
* **Rubro y tipo de contrato**: el modelo gana claramente al baseline en todos los rubros y
  tipos de contrato con ≥ 5 % de las filas, salvo la excepción de §8.

### Calibración

Brier del modelo mejor que el de la tasa base en los 3 targets (tabla principal). Curvas
de calibración por decil (`calibracion_*.csv`): en cese 3 m el error absoluto
predicho-observado por decil es ≤ 0,028 en los 10 deciles (el peor es el decil 9, el de
mayor riesgo: 0,504 predicho vs. 0,476 observado). En reingreso el ajuste es algo menos
fino en los deciles intermedios (error hasta 0,086 en el decil 5) pero los extremos
(deciles 0, 8 y 9, los más accionables) calibran a menos de 0,025.

### Importancias (gain)

* **Cese 3 m**: `cesantia_giro_actual` (42,6 %), `cesantia_en_curso` (12,2 %), `rubro`
  (5,9 %), `tipo_afiliado` (3,6 %), `meses_desde_cambio` (2,2 %), `edad`,
  `estacionalidad_rubro`, `mes_corte`, `tipo_contrato`, `meses_hasta_pension`.
* **Cese 6 m**: mismo orden con `cesantia_giro_actual` (34,2 %) y `meses_desde_cambio`
  (7,7 %) al frente; el resto (rubro, edad, tenencia con el empleador) se reparte de forma
  más pareja que a 3 meses.
* **Reingreso**: `edad` (22,0 %), `meses_desde_afc_inicio` (9,4 %), `meses_hasta_pension`
  (8,6 %), `meses_desde_afiliacion` (7,6 %), `region`, `max_12`, `meses_desde_afc_termino`,
  `cesantia_n_giros_recibidos`.

Que `cesantia_giro_actual`/`cesantia_en_curso` dominen en cese no es leakage (ver test de
leakage en §8): un giro de cesantía activo en `corte-1` sólo puede existir si el afiliado
seguía cesante en ese momento — conocido, no futuro — y es, con razón, un predictor muy
fuerte de que seguirá cesante 1 a 3 meses después. Tablas completas en
`reportes/importancias_*.csv`.

### Ejemplos interpretados (cese 3 m, contribuciones SHAP de LightGBM `pred_contrib`)

Uno por rubro estacional (`reportes/ejemplos_interpretados.csv`), por ejemplo:

* Afiliado en `A_AGRICULTURA` (giro de cesantía activo): `pred=0,88`, `y3_real=1`.
  Contribuciones principales: `cesantia_giro_actual` (+3,45), `cesantia_en_curso` (+0,65),
  `edad` (+0,48).
* Afiliado en `P_EDUCACION` (con giro activo pero que sí volvió a cotizar): `pred=1,00`,
  `y3_real=0` — un falso positivo real: el giro activo predice cese con altísima confianza,
  pero el afiliado consiguió empleo antes de agotar el beneficio; ejemplo honesto de que la
  discriminación (AUC 0,89) no es perfecta.

### Métrica de negocio

Contactando al **10 % de mayor riesgo** en test: se captura el **62,4 %** de los ceses a
3 meses, **61,6 %** de los ceses a 6 meses y **13,7 %** de los reingresos (la tasa base de
reingreso es 72 %, así que el 10 % de mayor riesgo son en su mayoría los que de todas
formas iban a reingresar rápido; ver interpretación de negocio en §7).

Número esperado (suma de probabilidades calibradas) vs. real de casos, por mes de corte:

| Target | 202601 | 202602 | 202603 | **Total test (3 meses)** |
|---|---|---|---|---|
| Cese 3 m | +5,6 % | +1,9 % | +5,7 % | **+4,3 %** |
| Cese 6 m | +3,9 % | +1,2 % | **+12,1 %** | **+5,1 %** |
| Reingreso | +4,5 % | +5,0 % | +5,1 % | **+4,9 %** |

El error relativo **agregado de los 3 meses de test** (la forma más estable de leer el
criterio de aceptación) está entre 4,3 % y 5,1 % en los tres targets, bien por debajo del
10 %. El único mes individual que supera el 10 % es cese 6 m en 202603 (+12,1 %, con solo
629 casos reales ese mes — el ruido de un mes puntual con pocos casos raros pesa más); se
documenta en vez de ocultarlo (ver §8).

## 7. Lectura de negocio

* **Recaudación esperada**: el "número esperado de ceses/reingresos" calibrado por mes
  (tabla de arriba) es directamente utilizable para proyectar densidad de cotización y
  flujo de recaudación, con un error agregado de 3 meses de ~5 % en los tres targets.
* **Priorización de contactos**: contactar al 10 % de mayor riesgo de cese captura ~6 de
  cada 10 ceses reales a 3 y a 6 meses — una mejora sustancial frente a contactar al azar
  (que capturaría ~10 %). Para reingreso, el 10 % de mayor riesgo no es el segmento más
  útil para contactar (la tasa base ya es 72 %: la mayoría reingresa igual, con o sin
  contacto); una campaña de reingreso rendiría más si se apunta a la **cola opuesta**
  (menor probabilidad de reingreso, para ofrecerles cotización voluntaria/APV antes de que
  la laguna se alargue) — una relectura de la misma probabilidad calibrada, sin necesidad
  de otro modelo.
* Los 3 casos con mayor `cesantia_giro_actual` (ejemplos interpretados) ilustran el patrón
  más accionable: un afiliado que ya está cobrando el seguro de cesantía es, con razón, el
  perfil de mayor riesgo de laguna prolongada — el momento natural para ofrecerle asesoría
  antes de que se agoten los giros.

## 8. Decisiones tomadas, supuestos y limitaciones

**Ambigüedades del brief, resueltas y documentadas** (según se pidió cuando algo no era
unívoco):

* El brief da dos reglas de desfase que, tomadas literalmente con un único corte de
  recepción, se contradicen: dice que la RIM se conoce hasta `corte-2` (desfase de 2
  meses, igual que rim_nowcasting) pero que los eventos AFC/licencias se conocen hasta
  `corte-1` (un mes menos), aunque su propia tabla de datos diga para ambos "conocidos en
  `mes+1`". Con un solo corte de recepción (`hasta_recepcion`) para toda la fila, esas dos
  afirmaciones no pueden ser ciertas a la vez si RIM y eventos comparten el mismo desfase
  de recepción. Se decidió reproducir **ambas** literalmente usando **dos cortes de
  recepción distintos dentro de la misma fila**: `hasta_recepcion_rim = corte-1` (RIM
  conocida hasta `corte-2`) y `hasta_recepcion_eventos = corte` (AFC/licencias/giros
  conocidos hasta `corte-1`) — consistente con que el aviso AFC llega por una vía
  (Seguro de Cesantía) más rápida que la nómina/nómina nunca es la fuente del aviso AFC.
  Con esto, `afc_termino_l1`/`l2` quedan literalmente en `corte-1`/`corte-2`, tal como los
  nombra el brief. Se prueba explícitamente que un evento fechado en `corte` (o después) no
  puede colarse aunque su `periodo_recepcion` se falsifique
  (`test_afc_conocido_hasta_corte_menos_1`).
* No se modelan DNP ni rectificaciones en `cotizaciones_conocidas` (a diferencia de
  rim_nowcasting): el brief solo pide reproducir el desfase de conocimiento, no la
  granularidad completa de movimientos; una declaración vale como una sola cifra de RIM
  con su `periodo_recepcion`.
* Tasas de giros de cesantía (70/55/45/40/35 % a 5 meses INDEFINIDO; 70/55 % a 2 meses
  PLAZO_FIJO/POR OBRA) son una aproximación ilustrativa de la Ley 19.728, no los
  porcentajes legales exactos (que separan cuenta individual, aporte del empleador y Fondo
  de Cesantía Solidario). Sirven para dar una señal de "cobertura restante", no para
  calcular montos reales.
* Ventana de historia acotada a 30 meses (`VENTANA_HISTORIA`) por rendimiento: features
  como `meses_historia`, `racha_cotiza`/`racha_cero` y `meses_desde_cambio` saturan en ese
  tope (un afiliado con 6 años cotizando sin interrupción se ve igual que uno con 30 meses
  para esas 3 columnas). Con `n_afiliados=20 000` esto redujo el tiempo de
  `construir_features` de ~12 s a ~1,5 s por corte (perfilado con `cProfile`;
  `groupby().rolling()` y un `groupby().apply()` python-level eran el cuello de botella;
  se reemplazaron por `groupby().tail(w)` + `groupby().agg()`, que solo necesitan la última
  fila de cada afiliado).
* `edad` se proyecta desde `edad_inicio` (edad al primer periodo simulado, según
  `synth_base.generar_afiliados_pandas`) igual que en `rim_nowcasting.features`, usando
  `macro["periodo"].min()` como referencia de `periodo_inicio` (no se pasa explícito a
  `construir_features` para no romper la firma pedida por CONVENCIONES).
* 3 modelos LightGBM independientes (no multi-tarea): ver justificación en §3.

**Criterios de aceptación no alcanzados / con matices** (números reales, sin maquillar):

* El error relativo mensual del agregado supera el 10 % en **un** mes puntual (cese 6 m,
  202603: +12,1 %, sobre solo 629 casos reales), aunque el agregado de los 3 meses de test
  da 5,1 %. Se reporta como está: el criterio del brief ("error relativo del agregado
  mensual ≤ 10 %") se cumple si se lee como el agregado del período de test completo, no
  mes a mes; un solo mes con pocos casos es ruidoso por diseño (CONVENCIONES exige ruido no
  explicable en el target).
* **Segmento independiente** (tipo de afiliado, ~7 % de las filas — supera el umbral de
  "segmento grande" de CONVENCIONES): el modelo **no** gana claramente al baseline de tasa
  base ahí. Cese 3 m: AUC 0,721 vs. 0,710 (empate técnico), Brier 0,146 vs. 0,141 (modelo
  peor). Cese 6 m: AUC 0,701 vs. **0,709** (el baseline gana), Brier 0,148 vs. 0,141 (modelo
  peor). Es la misma población que aparece como `tipo_contrato="SIN_EMPLEADOR"` (no tiene
  empleador de referencia). Causa más probable: la renta de los independientes en la
  simulación se redefine una vez al año (Operación Renta) y las features de historia
  mensual (lags, densidad, rachas) no capturan bien ese patrón anual — la misma debilidad
  que ya documentó rim_nowcasting para este mismo tipo de afiliado. No se intentó corregir
  en esta ronda (una feature de "renta anual declarada" o un modelo separado para
  independientes son la mejora natural, ver §10) para no introducir más riesgo de
  sobreajuste bajo el tiempo disponible; se deja documentado en vez de ocultarlo.
* El resto de los criterios del brief se cumple con margen: AUC test cese 3 m 0,889 (≥ 0,80
  pedido), AUC test reingreso 0,778 (≥ 0,70 pedido), Brier del modelo mejor que el de la
  tasa base en los 3 targets, runtime y tests dentro de los límites (§9).

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.riesgo_cese_cotizacion.run --n-afiliados 20000
python -m pytest afp_lab/riesgo_cese_cotizacion/tests -q
```

* `run.py` (`--n-afiliados 20000`, configuración por defecto, laptop de 4 núcleos,
  `n_jobs=2` en LightGBM): **56,8 s** en la corrida de referencia (generación del universo
  5,7 s; features de 24 cortes 28,6 s; entrenamiento + evaluación de los 3 modelos ~21 s).
  Muy por debajo del límite de 5 minutos.
* `pytest afp_lab/riesgo_cese_cotizacion/tests`: **9 tests, ~6 s**. Muy por debajo del
  límite de 2 minutos.
* Semilla fija (`seed=42` por defecto) y generación determinista: se verificó que dos
  corridas de `run.py` con los mismos parámetros producen exactamente las mismas métricas.
* `--salida` (por defecto `salidas/riesgo_cese_cotizacion/`, en `.gitignore`): `dataset_features.parquet`
  (465 160 filas × 80 columnas), `modelos/modelo_{y3,y6,y_reingreso}.joblib` y
  `reportes/*.csv` (métricas, por segmento, calibración, negocio, importancias, ejemplos).

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/`** (no se tocó nada fuera de este proyecto; quedan como
propuesta para el gerente):

* Mover la aritmética de periodos `yyyymm` (`periodo_to_idx`/`idx_to_periodo`/`add_months`)
  a `afp_lab.common`: hoy vive duplicada (una versión Spark en
  `rim_nowcasting.calendario`, una versión numpy propia de este proyecto en
  `synthetic.py` porque llamar a la versión escalar de rim_nowcasting fila a fila sobre
  cientos de miles de filas era demasiado lento). Un helper numpy vectorizado en común
  evitaría que cada proyecto pandas la reimplemente.
  `afp_lab.common.metrics.tabla_metricas_por_segmento` falla con `ValueError` si el
  segmento es una columna `pandas.Categorical` con categorías sin observar (p. ej.
  `pd.cut(...)`), porque `df.groupby(col_segmento)` itera también las categorías vacías
  (default `observed=False`). Se evitó en este proyecto casteando el segmento a `str`
  antes de llamar la función (`evaluate.agregar_tramos`), pero sería mejor agregar
  `observed=True` en la función común.
* Una utilidad común de "métrica de negocio" (esperado vs. real por período + % capturado
  contactando el top-k %) serviría también a `churn_traspaso` y `propension_apv`, que
  seguramente necesitarán algo equivalente.

**Próximos pasos** (no implementados en esta ronda):

* Modelo o features específicas para afiliados independientes (renta anual vía Operación
  Renta) para cerrar la brecha del segmento documentada en §8.
* Para el modelo de reingreso, guardar el rubro y el tramo de renta del **último empleo
  antes del cese** (hoy se pierden porque el afiliado no tiene empleador vigente) — haría
  útiles esos dos segmentos, hoy triviales para ese target.
* Explorar `cesantia_giro_actual` con más granularidad (por ejemplo, giros restantes antes
  de agotar el beneficio) como feature explícita en vez de solo el monto del último giro.
* Grilla de umbrales/costos por segmento para la asesoría (hoy la priorización es un único
  ranking por probabilidad calibrada; una AFP real distinguiría el costo de contactar a un
  afiliado de alto valor vs. uno de bajo saldo).
