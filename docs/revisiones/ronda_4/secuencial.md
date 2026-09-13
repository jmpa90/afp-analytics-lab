# Revisión ronda 4 · 08 `secuencial_afiliado` (GRU multi-tarea vs. LightGBM de 02/04)

**Veredicto: APROBAR CON RESERVA** (revisor adversarial Sonnet). No encontré bloqueantes: no hay
leakage en `features.py` (verificado en código y con test), los targets se importan de 02/04 sin
reimplementarse, la comparación arquitectura-vs-información está construida correctamente (mismo
split para LightGBM de producción, LightGBM "tabular con lags" y la red) y las cifras del README
reproducen casi dígito a dígito con el comando exacto del README. La reserva es por alcance: 2 de
las 4 cabezas del brief, sin validar runtime a la escala de 20.000 afiliados que exige
CONVENCIONES §1, y basado en una autorización de recorte que no puedo verificar de forma
independiente desde el repo.

## Qué reproduje

- `PYTHONPATH=/home/user/wt/cese python3 -m pytest afp_lab/secuencial_afiliado/tests -q` →
  **7 passed en 15,66 s** (README: "16,6 s", < 60 s exigido) — coincide.
- `PYTHONPATH=/home/user/wt/cese python3 -m afp_lab.secuencial_afiliado.run --n-afiliados 10000`
  (comando exacto del README §9) → **212,9 s totales** (red: 126,5 s) vs. README "228,9 s
  totales (red: 143,2 s)" — dentro de la variabilidad esperada de una máquina de 4 vCPU
  compartida, ambos bajo los límites del brief.
- Filas de test: traspaso n=20.787, cese n=16.766 — coinciden exacto con el README.
- Traspaso 6 m: `lightgbm_produccion` 0,7255 [0,7089;0,7450], `lightgbm_tabular_lags` 0,5940
  [0,5724;0,6149], `red_multitarea` 0,5848 [0,5617;0,6099], `red_single_task` 0,5766
  [0,5552;0,6006] — los cuatro coinciden dígito a dígito con la tabla del README §6.
- Cese 6 m: `lightgbm_tabular_lags` 0,7775 [0,7610;0,7928], `red_multitarea` 0,7768
  [0,7586;0,7916], `red_single_task` 0,7807 [0,7630;0,7948], `baseline_regla_negocio` 0,8195
  [0,8054;0,8345] — coinciden dígito a dígito. Único número que **no** reprodujo exacto:
  `lightgbm_produccion` cese = 0,8614 [0,8492;0,8746] reproducido vs. 0,8610 [0,8475;0,8741] del
  README (diferencia ~0,0004 AUC; ver "Menores" #3).
- Deltas AUC (`red_multitarea` − `lightgbm_produccion`): reproducido **−0,1407** (traspaso) /
  **−0,0846** (cese) vs. README "−0,141 / −0,084" — coincide. Conclusión reproducida:
  `no_aporta`.

## Sin leakage (verificado en código, no solo en el README)

- `_ventana` (`afp_lab/secuencial_afiliado/features.py:136-150`) nunca lee columnas más allá de
  `pos_fin = add_months(corte, -2)`; el relleno de ventana corta es siempre por la **izquierda**.
  `construir_secuencias` (línea 178) fija `pos_fin` con `meses_desfase=MESES_DESFASE_RIM=2` antes
  de recortar, para las dos cabezas.
- `test_secuencia_sin_leakage` (`tests/test_secuencial_afiliado.py:82-106`) altera `rim_emp1` y
  `estado` en el panel posterior a `corte-2` (y borra filas futuras del afiliado 1) y confirma con
  `np.testing.assert_array_equal` que `num`/`cat`/`mask` no cambian — pasa.
- Targets NO reimplementados: `run.py:26-36` importa directamente
  `construir_dataset_traspaso`/`entrenar_lgbm_traspaso`/`predecir_lgbm_traspaso` de
  `churn_traspaso` y `_construir_dataset`/`ModeloCese.fit`/`_split_temporal` de
  `riesgo_cese_cotizacion`; `afp_lab/secuencial_afiliado/features.py` no contiene ninguna función
  de target, solo construcción de secuencia.
- Universo idéntico a 02/04: `test_universo_identico_02_04` compara con `.equals()` fila a fila
  `afiliados`/`panel_mensual`/`macro` entre `churn_traspaso.synthetic.generar_tablas` y
  `riesgo_cese_cotizacion.synthetic.generar_datos` — pasa.

## Comparación justa (arquitectura vs. información) — el punto que pedía el encargo

- **Mismo split para los tres modelos de cada cabeza**: `run.py:104-141` construye `cab_tr`/`cab_ce`
  una sola vez; `datos_tr`/`datos_ce` (usados por la red y por `lightgbm_tabular_lags`) se derivan
  de esas MISMAS filas train/val/test, y `lightgbm_produccion` se entrena sobre
  `cab_tr["train"]`/`cab_ce["train"]` (idénticas filas, solo más columnas). No hay cortes ni
  poblaciones distintas entre modelos dentro de una cabeza.
- **"Tabular con lags" ve exactamente los mismos datos que la red**: `aplanar_secuencia`
  (`features.py:213-233`) aplana los mismos tensores `num`/`cat` de la misma `SecuenciaTensores`
  que alimenta la GRU (`model.py:33-35,172`, mismas 8 columnas × 24 lags + estáticas) — confirmado
  leyendo el código, no solo la afirmación del README.
- El hallazgo central ("la brecha es de información, no de arquitectura") queda sustentado en los
  propios números reproducidos: la red empata con `lightgbm_tabular_lags` en ambas cabezas (0,585
  vs. 0,594 traspaso; 0,777 vs. 0,777 cese, dentro del solape de los IC bootstrap) y ambos pierden
  por márgenes grandes frente a `lightgbm_produccion` (que sí usa las tablas operacionales de la
  AFP). Es una lectura bien fundada, no optimista sin evidencia.
- `conclusion_vale_la_pena` (`evaluate.py:54-76`) implementa exactamente el criterio del BRIEF
  (mejora ≥0,01 AUC en ≥2 cabezas); no hay manipulación del umbral de "vale la pena".

## Bloqueantes

Ninguno.

## Menores

1. El runtime de referencia (213-229 s) solo se midió a `n_afiliados=10.000`. CONVENCIONES §1
   exige que `run.py` corra en <5 min a 20.000 afiliados, y el propio BRIEF de esta ronda pide
   "<10 min a 20.000" — nunca se corrió a esa escala en esta entrega. Si el costo escala
   aproximadamente lineal con las filas (lo esperable, dado que el cuello de botella es el
   entrenamiento por lotes de la red), a 20.000 la red rondaría ~250-290 s, aún dentro del límite
   del brief, pero es una extrapolación no verificada — recomendaría correrlo una vez a esa escala
   antes de cerrar el proyecto.
2. El alcance recortado a 2 de las 4 cabezas del brief (traspaso, cese; faltan APV y RIM) y sin la
   comparación del embedding contra 06 se atribuye en el README a una "instrucción explícita del
   gerente" — no verificable desde el repo (no hay registro de esa conversación). Es plausible
   dado el patrón de otras entregas de esta ronda, pero el gerente debería confirmar
   explícitamente que autorizó el recorte antes de dar el proyecto por completo.
3. `lightgbm_produccion` (cese) no reprodujo dígito a dígito (README 0,8610 [0,8475;0,8741] vs.
   reproducido 0,8614 [0,8492;0,8746]); es la única fila de las 11 modelo×cabeza que no calzó
   exacto. La diferencia es pequeña y no cambia ninguna conclusión, pero indica que ese
   entrenamiento LightGBM específico no es 100 % determinista con `num_threads=2` en esta
   máquina, a diferencia de los demás modelos de la tabla (incluida la red PyTorch, que sí
   reprodujo exacto pese a no fijar semilla de determinismo a nivel de kernel de CPU).

## Veredicto

**APROBAR CON RESERVA.** Sin bloqueantes: no hay leakage, la comparación arquitectura-vs-
información es rigurosa y quedó verificada en código (no solo en el README), y las cifras
reproducen casi dígito a dígito con el comando de referencia. La reserva es por alcance (2/4
cabezas, embedding vs. 06 pendiente, runtime no validado a 20.000 afiliados) y por una
autorización de recorte que el repo no permite confirmar de forma independiente — pedir
confirmación explícita del gerente sobre ese punto antes de cerrar el proyecto como completo.
