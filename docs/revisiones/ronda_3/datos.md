# Revisión ronda 3 — workstream `datos` (rim_nowcasting: código de movimiento + modelo de datos regulatorio)

Revisor adversarial. Repo: `/home/user/wt/datos` (rama `ronda-3/datos`), diff contra `main`.

## Veredicto
**Aprobar con reserva.**

El diseño as-of de `cod_movimiento`/`dias_trabajados` (objetivo 1.1-1.2) es correcto y está bien
probado (7 tests nuevos + verificación manual independiente, ver abajo): no encontré leakage. El
objetivo 2 (`docs/modelo_datos_regulatorio.md`) es sólido, acotado y sin cifras inventadas.

Pero el objetivo 1.3 (delta de WAPE/sesgo/runtime, el resultado de negocio del informe) tiene un
problema serio: los números reportados NO son reproducibles (el script vive sólo en el scratchpad
de la sesión del DS, no en el repo) y, al reproducir el mismo experimento con el comando oficial y
documentado del proyecto (`pipeline backtest`) a la misma escala reducida y con la misma
configuración declarada, obtengo un runtime que sube ~7%, no +82% (la cifra que el informe usa
para decir que se excede el límite del brief), y un WAPE que en el peor caso empeora ~5 veces
menos de lo reportado (ver "Números verificados"). Además,
`--perfil cuprum` no se corrió ni una sola vez, ni siquiera a escala reducida, pese a que el CLI ya
soporta exactamente eso.

## Bloqueantes

### B1. El delta de backtest (objetivo 1.3) no es reproducible y, reproducido de forma independiente
con el comando oficial, contradice la conclusión del informe

El informe apoya su "lectura honesta" ("el WAPE no mejora... el runtime sube considerablemente,
+82%, por encima del +30% del brief") en un script ad-hoc (`datos_r3_eval.py`) que el propio DS
dice que "vive en el scratchpad de esta sesión, no en el repo". Nadie puede reproducirlo, ni el
gerente ni este revisor, exactamente como advierte la lista de cosas a refutar ("números no
reproducibles").

Reproduje el mismo experimento con el comando YA DOCUMENTADO del proyecto
(`python -m afp_lab.rim_nowcasting.pipeline backtest`), que además ya soporta escala reducida
(`--n-afiliados`, `--n-train`, `--n-test`, `--gbt-iter`) — no había necesidad de un script fuera
del repo. Con `--n-afiliados 300 --n-train 3 --n-test 1 --gbt-iter 5 --perfil sistema --inicio
202301 --snapshot 202506 --seed 42` contra una copia "antes" (`git show main:...` de
`synthetic.py`/`features.py`) y el código actual ("después"):

| | antes (CLI, depth=5 default) | después (CLI, depth=5 default) | delta |
|---|---|---|---|
| WAPE h=1 (`rim_proyectada`) | 0.435 | 0.397 | **mejora** -0.038 |
| WAPE h=2 | 0.490 | 0.447 | **mejora** -0.043 |
| AUC h=1 / h=2 | 0.910 / 0.856 | 0.933 / 0.874 | mejora |
| sesgo_agregado h=1 | +13.3% | +9.5% | mejora |
| filas train/test | 1646 / 596 | 1646 / 596 | igual |
| tiempo total | 180.7s | 192.3s | **+6.4%** |

Repetí además con `gbt_max_depth=3` (la config exacta que el informe dice haber usado, invocando
`Config`+`backtest()` directamente porque el CLI no expone profundidad):

| | antes (depth=3) | después (depth=3) | delta | informe DS |
|---|---|---|---|---|
| WAPE h=1 | 0.440 | 0.448 | +0.008 | informe: +0.039 |
| WAPE h=2 | 0.526 | 0.525 | -0.002 | informe: +0.022 |
| tiempo total (generar+features+train+eval) | 104.0s | 111.6s | **+7.3%** | informe: **+82%** |

En ningún escenario que pude reproducir el runtime se acerca a +82% (+6.4% y +7.3% en mis dos
corridas, contra +82% del informe): esta es la contradicción más clara y mejor verificada. Sobre
WAPE los resultados son mixtos pero en ningún caso tan malos como los reportados: con la
configuración por defecto del CLI (profundidad 5) el WAPE directamente MEJORA en ambos horizontes;
con profundidad 3 (la config que el informe dice haber usado) h=1 empeora en la misma dirección
que reporta el informe pero ~5 veces menos (+0.008 vs +0.039) y h=2 más bien mejora levemente
(-0.002) en vez de empeorar (+0.022 informe). No estoy afirmando que el código de movimiento
definitivamente mejore el WAPE en producción (eso requiere la escala de referencia, que nadie
corrió) — afirmo que la cifra de runtime (+82%, la que el informe usa para decir que se excede el
límite del brief) no se sostiene con el propio código del repo en esta máquina, y que la magnitud
del deterioro de WAPE reportado tampoco.

Nota adicional (contexto, no hallazgo nuevo): a esta escala de juguete el baseline `media_3` le
gana en WAPE a `rim_proyectada` tanto antes como después (0.380 vs. 0.435 antes; 0.380 vs. 0.397
después, h=1, corrida CLI depth=5) — un problema preexistente de evaluar a escala mínima, no algo
que esta ronda cause. Pero nótese que la brecha se ACORTA después (de -0.055 a -0.017), consistente
con que el código de movimiento ayuda un poco incluso a esta escala, no que la perjudique como
concluye el informe.

(Nota honesta: esta sandbox también está compartida — confirmé con `dmesg`/cgroup un OOM-kill real
de mi propia corrida de tests al competir por memoria con otro job, y vi procesos de otra sesión
corriendo en paralelo. Es plausible que el DS haya medido su +82% bajo contención similar; pero
precisamente por eso una cifra de runtime medida en una máquina compartida, con un script que no
queda en el repo, no es un número confiable para decidir nada — hay que remedirlo con el comando
oficial, en una corrida aislada, y dejar el resultado en el repo/README.)

### B2. `--perfil cuprum` no se evaluó ni una sola vez (ni a escala reducida)

El brief pide explícitamente "evaluar con `--perfil sistema` y `--perfil cuprum`". El CLI ya
soporta `--perfil cuprum` y escala reducida simultáneamente (los mismos flags usados arriba), así
que correrlo no exigía ni tiempo adicional relevante ni herramienta nueva. El informe reporta cero
resultados de Cuprum, ni siquiera a escala de juguete — no es sólo "pendiente a escala de
referencia" (que sí sería aceptable per brief, "o menor si el tiempo apremia, diciéndolo"), es una
ausencia total de un perfil que el brief pide para ambos.

## Menores

* **M1 — Afirmación incorrecta sobre `common/planilla.py` y el código 8.** El informe dice que el
  código 8 (cambio de contrato) recibe "igual tratamiento que los códigos 5/6/13 ya no generados
  por `common/planilla.py`". Es falso: `common/planilla.py` SÍ genera el código 8
  (`_movimiento_y_dias`, `afp_lab/common/planilla.py:142`: `codigo = np.where(cambia, 8, codigo)`),
  y su propio docstring lo dice ("Códigos... que este generador SÍ produce: 0,1,2,3,4,7,**8**,11,12").
  El código y el README del proyecto (sección 10c) sí lo dejan claro ("Limitación propia de esta
  ronda, no de `common/planilla.py`"); es sólo la frase del informe al gerente la que confunde
  ambos módulos.

* **M2 — Runtime de tests sigue muy por encima del límite de CONVENCIONES (2 min)**, y este round lo
  empeoró: hubo que subir `driver_memory` de 3g a 6g para evitar OOM de Catalyst. Ya era una
  violación preexistente (documentada honestamente por el DS), pero vale remarcar que la causa que
  el informe da (sólo "el plan se amplió") es incompleta: `_features_movimiento` termina con
  `.cache()` (features.py:254) que, igual que `serie`/`abiertas`, sólo se libera
  (`.unpersist()`) dentro de la rama `if materializar:` de `construir_features` (features.py
  ~404-406). Todos los tests que llaman con `materializar=False` (la mayoría de los nuevos)
  acumulan ahora TRES cachés sin liberar por llamada en vez de dos, en una sesión de Spark que vive
  toda la suite — eso también contribuye a la presión de memoria que forzó a subir a 6g, no sólo
  "el plan más ancho".

* **M3 — Comentario huérfano después de un `return`.** En `features.py`, el bloque de comentario
  que explica el `.cache()` de `_features_movimiento` (líneas ~255-259) quedó DESPUÉS del
  `return conocido.groupBy(...).agg(...).cache()`, es decir, nunca se lee en contexto — es sólo un
  detalle de estilo (el comentario es correcto, está mal ubicado), pero conviene moverlo antes del
  `return` para que no parezca código muerto al leer el diff.

* **M4 — `dias_trabajados_l1` se imputa en 30 (mes completo) incluso cuando el mes fue,
  efectivamente, una licencia médica de mes completo** (caso ya identificado por el propio DS al
  corregir `test_cod_movimiento_coincide_con_evento_afc`): si `dias_licencia == 30`, la línea
  `EMPLEADOR` de ese mes se descarta en `generar_cotizaciones` (RIM redondeada a 0), por lo que
  `_features_movimiento` no encuentra declaración vigente para ese (afiliado, periodo, empleador) y
  cae al relleno por defecto (30 = "sin movimiento"), en vez de reflejar que hubo una licencia de
  mes completo. Es una simplificación heredada de una decisión de diseño anterior a esta ronda (no
  un bug nuevo), y el caso es de baja frecuencia, pero vale documentarlo explícitamente en el
  README junto a las demás limitaciones de `_features_movimiento` (hoy sólo está en el docstring
  del test).

## Lo que sí verifiqué y quedó bien

* **As-of / leakage (objetivo 1.1-1.2):** revisé el diseño de `_features_movimiento` a mano
  (afiliado con TERMINO real → `cod_movimiento=2` en su declaración, `dias_trabajados=4`; y una
  prueba manual independiente agregando una declaración con `periodo_recepcion` futuro que no
  cambia ninguna feature) — ver "Números verificados". El código respeta el mismo
  `periodo_recepcion <= hasta_recepcion` que ya gobierna la RIM (son literalmente la misma fila
  Previred), consistente con la convención de la ronda 1 en CONVENCIONES §3. Los 7 tests nuevos
  (dominio de códigos, sólo EMPLEADOR/SUBSIDIO declaran código, coincidencia con `evento_afc`,
  `dias_trabajados` coherente, `fecha_desde`/`fecha_hasta`, respeto de recepción, rango/consistencia
  de las 14 features nuevas) prueban cosas reales, no tautologías.
* El bug que el DS encontró y corrigió en su propio test (`test_cod_movimiento_coincide_con_evento_afc`,
  supuesto de licencia de mes completo) es un fix legítimo y bien documentado, no un parche para
  esconder un problema de producción.
* `FEATURES_NUMERICAS` se importa dinámicamente en `model.py`, así que las 14 features nuevas
  efectivamente llegan al `VectorAssembler` — no hay features "muertas" declaradas pero no usadas.
* No se tocó `common/` (sólo se importan `P_PERMISO_SIN_GOCE`/`P_AUSENTISMO`, tal como pide
  CONVENCIONES §5); la propuesta de unificar `common/planilla.py` con `synthetic.py` queda
  correctamente documentada como propuesta en la sección 11, no ejecutada sin acuerdo.
* `docs/modelo_datos_regulatorio.md` (objetivo 2): una a dos páginas, referencia
  `normativa_tablas_afp.md`/`tablas_equivalencia.py`, distingue qué existe/falta y quién lo
  consumiría, sin cifras "reales" sin fuente.

## Checklist CONVENCIONES §7

- [x] Sin leakage: test presente y features as-of verificadas a mano en 2 casos — ver arriba.
- [x] Target bien definido, población/exclusiones — no se tocan (sólo se agregan features).
- [x] Dinámicas sintéticas plausibles para Chile y documentadas — reusa `P_PERMISO_SIN_GOCE`/
      `P_AUSENTISMO` ya documentados en `common/planilla.py`, con la Tabla N°7 citada.
- [~] Baselines presentes y el modelo los supera sin perder segmentos — no verificable a escala de
      referencia esta ronda (nadie la corrió); a escala de juguete el modelo no le gana a `media_3`
      en ninguno de los dos horizontes, pero esto es preexistente al proyecto 01 a esa escala, no
      algo que esta ronda cause (la brecha de hecho se acorta, ver B1).
- [x] Métricas comparables (`evaluate.metricas`/`metricas_clasificador`) y métrica de negocio — sin
      cambios de este round.
- [ ] Runtime y tests dentro de límites — tests: 18/18 pasan pero a 17m16s, muy por sobre el límite
      de 2 min de CONVENCIONES (preexistente, agravado por esta ronda: hubo que subir
      `driver_memory` 3g→6g). Runtime del backtest: no verificable a escala de referencia; a escala
      reducida el propio informe reporta +82% (> el +30% del brief) pero esa cifra no reproduce
      (ver B1) — con el comando oficial obtuve +6-7%.
- [x] README completo según plantilla — sección 10c sigue la plantilla (mecanismo, resultados,
      lectura honesta, limitaciones, propuestas para common).

## Números verificados

* **Tests, comando más barato**: `PYTHONPATH=/home/user/wt/datos python -m pytest afp_lab/rim_nowcasting/tests -q -k test_cod_movimiento_coincide_con_evento_afc`
  → `1 passed in 25.82s` (real). Coincide con el informe ("test corregido, reverificado
  aislado"; el DS reporta 16.8s, yo obtuve 25.8s en esta máquina — mismo resultado, tiempo algo
  mayor, consistente con carga compartida).
* **Suite completa** (`PYTHONPATH=/home/user/wt/datos python -m pytest afp_lab/rim_nowcasting/tests -q`),
  corrida aislada (sin otros procesos compitiendo por memoria en el cgroup): **18 passed en
  1036.31s (17m16s)**. Confirma el "18/18" del informe. (Mi primer intento de correr la suite
  completa, en paralelo con otros backtests que yo mismo había lanzado, dio 3 failed + 1 error —
  investigué con `dmesg`/cgroup y confirmé que fue un OOM-kill real de la JVM de ese test por mi
  propia contención de memoria, no un defecto del código; lo descarté y repetí en aislado.)
* **Verificación manual as-of (2 casos, independiente de los tests del DS)**: con `n_afiliados=80`,
  encontré un afiliado con evento TERMINO real (`afiliado_id=34, periodo=202301, empleador_id=6`)
  cuya declaración `EMPLEADOR` trae `cod_movimiento=2, dias_trabajados=4` (Caso 1, correcto). Con
  `n_afiliados=80, corte=202504, hasta=202505`, agregar 20 declaraciones `EMPLEADOR` del mes de
  corte con `cod_movimiento=2` y `periodo_recepcion = hasta+1` (fuera de ventana) no cambió ninguna
  fila de `_features_movimiento` (Caso 2, `pd.testing.assert_frame_equal` OK) — confirma as-of
  independientemente de los tests que trae el propio diff.
* **Backtest reducido, comando oficial (`pipeline backtest`), depth=5 (default CLI)**: antes
  WAPE h1/h2 = 0.435/0.490, después = 0.397/0.447; runtime antes 180.7s → después 192.3s (+6.4%).
* **Backtest reducido, depth=3 (config declarada por el DS, vía `Config`+`backtest()` directo
  porque el CLI no expone profundidad)**: antes WAPE h1/h2 = 0.440/0.526, después = 0.448/0.525;
  runtime antes 104.0s → después 111.6s (+7.3%). Comparar con el informe: WAPE antes/después
  0.490/0.529 (h1), 0.573/0.594 (h2); runtime antes/después 187.0s/340.6s (+82%).

