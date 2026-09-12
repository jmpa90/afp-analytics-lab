# Revisión ronda 3 · `empleador` (objetivo 1: densidad de `common`; objetivo 2: escala de `riesgo_empleador`)

**Veredicto del gerente: APROBAR CON RESERVA** (revisor adversarial Sonnet, solo lectura de git).

## Cómo se verificó

Se corrieron directamente (no se confió solo en lo declarado):

- `pytest afp_lab/common/tests -q` → 21 passed en 3,25 s (informe: 21 en 3,2 s).
- `pytest afp_lab/riesgo_empleador/tests -q` → 21 passed en 14,04 s (informe: 21 en 12,6 s).
- `pytest afp_lab/churn_traspaso/tests -q` → 33 passed en 18,64 s (informe: 33 en 15,8 s).
- `pytest afp_lab -q` (suite completa) → **155 passed en 637,33 s (10 min 37 s)**, 0 fallos (informe: 155
  passed, 0 fallos, 9 min 46 s). Confirma la cifra central (155/0); el tiempo mayor se debe a
  contención de CPU en el sandbox compartido (otro worktree corriendo Spark en paralelo, `load average`
  ~4-7 sobre 4 núcleos durante buena parte de la corrida), no al código.
- Reproduje además, con los mismos comandos y semillas del informe (o el default cuando el informe no
  precisaba semilla), los números de **cinco** proyectos distintos. Todos coincidieron, en su mayoría con
  3-4 cifras significativas:

  | Qué | Medido por mí | Informe | ¿Coincide? |
  |---|---|---|---|
  | Densidad `sistema`, n=20000, seed 42/7 | 47,78 % / 47,74 % | 47,7-47,8 % | Sí |
  | Densidad `cuprum`, n=20000, seed 42/7 | 66,42 % / 66,54 % | 66,4-66,5 % | Sí |
  | Densidad `sistema` "antes" (parámetros pre-ronda-3 reinyectados en memoria, sin tocar el archivo) | 78,50 % / 78,57 % | ~78-79 % | Sí |
  | Densidad `cuprum` "antes" (ídem) | 72,84 % / 72,81 % | ~71-73 % | Sí |
  | 05 riesgo_empleador, afpe=20 (default nuevo), seed 42: AUC / captura@10% / volatilidad / meses tolerancia / meses 1σ | 0,8648 / 48,63 % / 63,60 % / 2 de 12 / 12 de 12 | 0,8648 / 48,63 % / 63,6 % / 2/12 / 12/12 | Sí, exacto |
  | 05 ídem, seed 7: AUC / captura | 0,8611 / 50,25 % | 0,861 / 50,3 % | Sí |
  | 05 afpe=40 (escala ronda 2), seed 42: AUC / captura / volatilidad | 0,8412 / 47,61 % / 94,56 % | 0,841 / 47,6 % / 94,6 % | Sí |
  | 05 afpe=10, seed 42: AUC / captura / volatilidad | 0,8425 / 43,44 % / 47,89 % | 0,843 / 43,4 % / 47,9 % | Sí |
  | 05 runtime total (n=20000, afpe=20) | 32,1 s | 26,1 s | Cifra distinta, pero muy por debajo del límite de 300 s en ambos casos (contención de CPU, no regresión) |
  | 05 segmentos ≥5 % de filas: gana el modelo en los 4 tramos de tamaño (incl. GRANDE 0,6092 vs 0,4847/0,4986), 2 de región y 3 de dotación | Confirmado a mano sobre la salida de `run.py` | "gana en TODOS" | Sí |
  | 04 riesgo_cese_cotizacion, perfil sistema, n=20000: AUC reingreso | 0,60527 | 0,6053 | Sí, exacto — regresión real confirmada |
  | 02 churn_traspaso, perfil sistema, n=20000: AUC / captura@10% | 0,7537 / 39,05 % | 0,7537 / 39,0 % | Sí, exacto; y `gana_modelo=True` en los 13 segmentos ≥5 % de filas evaluables, incl. `tramo_edad="60+"` (donde antes perdía) |
  | 03 propension_apv, perfil sistema, n=5000: AUC / WAPE / captura@10% | 0,8428 / 0,5785 / 68,51 % | 0,8428 / 0,579 / 68,51 % | Sí, exacto |
  | 06 segmentacion_afiliados, perfil cuprum (default), n=5000: silueta post-fusión / ARI resampling / ARI temporal / razones cese-traspaso-apv | 0,2280 / 0,9789 / 0,4797 / 3,27-2,04-31,7 | 0,2280 / 0,9789 / 0,4797 / 3,27-2,04-31,7 | Sí, exacto |

No encontré ningún número fabricado, redondeado a favor, ni no reproducible. La reconstrucción de los
"antes" por monkeypatch en memoria (sin tocar `synth_base.py`) es exactamente lo que el informe declara
haber hecho, y reproduce lo mismo que yo obtuve de forma independiente.

## Revisión de código (diff completo contra `main`)

- `afp_lab/common/synth_base.py`: sólo cambian 4 valores numéricos en `PERFILES` (`p_activo_inicial`,
  `factor_contratacion` de `sistema`; `factor_contratacion` de `cuprum`), con docstring/comentario que
  cita la Ficha 165 y los valores antes/después. No toca `rim_nowcasting.synthetic._simular_particion`
  (no hizo falta). No es una feature ni afecta el as-of: son parámetros generativos, no leakage.
- `afp_lab/common/tests/test_synth_base.py` (nuevo): mide la densidad con `n=6000`, descarta 12 meses de
  burn-in y 3 finales, para ambos perfiles, con banda ±3pp y un test adicional de que `cuprum > sistema`.
  Corre en 3,2 s. Correcto y no acoplado a un valor puntual.
- `afp_lab/riesgo_empleador/{synthetic,run}.py`: sólo agregan el parámetro `afiliados_por_empleador`
  (thread-through, default local 20) hacia `common.synth_base.generar_universo`, cuyo propio default
  (40) **no se tocó** — confirmado por lectura del archivo — así que ningún otro proyecto (01/02/03/04/06)
  se ve afectado por el objetivo 2, sólo por el objetivo 1 (densidad). `features.py`, `model.py` y el
  split temporal de 05 no se tocaron: cero riesgo de leakage nuevo introducido aquí.
- `afp_lab/churn_traspaso/tests/test_churn_traspaso.py`: sólo 2 líneas de umbral cambiadas (piso de
  `tasa_traspaso_12m_min` 0,020→0,015; techo de `auc_mejor_feature_suelta` 0,72→0,76), ambas son chequeos
  de **plausibilidad del generador** (no de modelo ni de umbral de aceptación de negocio), documentadas
  inline con el número medido y el mecanismo causal (la cartera cesante crece con la nueva densidad).
  Verificado que no se tocó `churn_traspaso/synthetic.py`, `features.py` ni `model.py`. El margen que
  deja el nuevo techo (0,76) frente al AUC del modelo completo (0,88, verificado en mi propia corrida:
  la salida de `run.py --perfil sistema` da 0,7537 de AUC del modelo global — que es sobre el dataset
  completo, no el fixture del test; el 0,7286 de la nota es sobre el fixture de test más chico) sigue
  siendo amplio; no veo indicio de que esto tape un leakage real.
- No hay código muerto, prints de depuración nuevos, ni duplicación con `common/` en los archivos
  tocados (`grep` de `print(` en `synth_base.py`/`test_synth_base.py`/`riesgo_empleador/synthetic.py`:
  vacío).
- `docs/briefs/ronda_4/*` aparecen como "eliminados" en el diff contra `main` sólo porque este worktree
  se creó antes de que `main` recibiera ese commit (`63d1cb0`, posterior a `49d807d`, el punto de
  ramificación de `ronda-3/empleador`). No es una acción del DS; aviso para que el gerente lo tenga en
  cuenta al fusionar (rebase, no descartar esos archivos).

## Bloqueantes

Ninguno. Ningún hallazgo de esta revisión invalida un resultado del brief, introduce leakage, rompe un
test, o contradice lo que el brief pedía. Los dos puntos de mayor peso (la regresión de AUC de 04, y que
05 sigue "en el límite" en captura@10 %) están explícitamente dentro del alcance previsto por el brief
("es un cambio transversal... informar qué métricas cambian"; "puede quedar en el límite como en la
ronda 2") y documentados con transparencia, no ocultos ni maquillados — los trato como reserva, no como
bloqueante.

## Menores

1. **Comando de verificación inconsistente para 06.** El informe lista
   `python3 -m afp_lab.segmentacion_afiliados.run --n-afiliados 5000 --perfil sistema` como comando de
   verificación, pero los números reportados para 06 (silueta 0,2482→0,2280, ARI 0,9563→0,9789, etc.)
   se midieron con `perfil=cuprum` (el default del proyecto), como aclara el propio texto de "numeros"
   del informe ("06 segmentacion_afiliados (perfil cuprum, su default)"). Ejecutar el comando tal cual
   (con `--perfil sistema`) no reproduce esas cifras; yo reproduje corriendo sin el flag (usa el
   default `cuprum`) y ahí sí coinciden exactamente. Corregir el comando en el informe (quitar el flag
   o cambiarlo a `--perfil cuprum`) para que quien lo copie no se confunda.
2. **Regresión de 04 no reflejada en el README de 04.** El AUC de reingreso de
   `riesgo_cese_cotizacion` cae de 0,720 a 0,605 (confirmado por mí) a causa del cambio transversal en
   `common/synth_base.py`, cruzando el umbral de 0,70 que el propio `afp_lab/riesgo_cese_cotizacion/README.md`
   documenta como cumplido ("AUC test reingreso **0,720** (≥ 0,70 pedido) ✓", línea 568). El hallazgo
   está bien documentado en `common/README.md`, en el README raíz y en el informe — pero nadie tocó el
   README del propio proyecto 04 para que alguien que sólo abra ese archivo no se quede con un check
   que ya no es cierto con el `common/` actual. Sugiero pedir una nota de una línea ahí (o que quede
   como tarea explícita para el responsable de 04 en la próxima ronda), sin bloquear esta entrega.
3. **`riesgo_empleador` sigue "en el límite" en captura@10 %** con la semilla de referencia (48,6 % vs.
   el ≥50 % pedido; 51,9 % en promedio de 5 semillas). Igual que en la ronda 2, está bien documentado y
   no oculto. Como el runtime de cada corrida es de sólo 25-50 s, hay margen amplio para que el gerente
   pida una o dos semillas adicionales como parte del cierre si quiere un criterio más sólido que "en
   promedio sí, en la semilla de referencia no".
4. **Deltas de 02/03/04 medidos con una sola semilla (42) y una sola escala**, no con el protocolo
   completo de 5 semillas/12 cortes de sus propias rondas de cierre. Está declarado como limitación
   explícita en el informe; para lo que se necesita (dirección y orden de magnitud de un efecto
   heredado de un cambio ajeno a esos proyectos) es proporcional — no debería tomarse como el cierre
   formal de esos proyectos si en el futuro se quiere reabrir esa pregunta.
5. **No se corrió la demo end-to-end de `01 rim_nowcasting`** (sólo su suite pytest, incluida en los
   155). El propio brief lo dejaba como opcional ("01 con Spark solo si queda tiempo"), así que no es
   una omisión real; lo dejo anotado para que quede explícito en el cierre de la ronda.
6. **`generar_universo` (common) mantiene su propio default de `afiliados_por_empleador=40`** sin
   cambios — correcto y deseado (evita que el objetivo 2 se filtre a otros proyectos), sólo lo dejo
   registrado porque es la razón por la que 01/02/03/04/06 no cambian de escala en esta ronda, algo que
   vale la pena que el gerente confirme que era la intención.

## Números verificados

Ver la tabla en "Cómo se verificó" arriba: 15 cifras de 5 proyectos distintos (común, 05, 04, 02, 03) más
06 (silueta/ARI/razones), además de los 4 conteos de tests (common, riesgo_empleador, churn_traspaso,
suite completa), reproducidas de forma independiente con los comandos del propio informe (o el
equivalente sin `--perfil` cuando el informe se equivocó de flag, ver menor 1). Todas coinciden con lo
declarado, típicamente hasta la 3ª-4ª cifra significativa. Las únicas discrepancias son de **tiempo de
ejecución** (runtime de 05: 32,1 s vs. 26,1 s declarado; suite completa: 10:37 vs. 9:46 declarado), y en
ambos casos atribuibles a contención de CPU en este sandbox compartido (otro worktree corriendo Spark en
paralelo durante gran parte de la revisión), no a una regresión real — ambos tiempos siguen muy por
debajo de sus límites (300 s y ~10 min respectivamente).
