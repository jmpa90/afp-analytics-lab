# Revisión adversarial — `afp_lab/voz_afiliado` (ronda 5, brief "voz del afiliado")

Worktree revisado: `/home/user/wt/empleador`, rama `ronda-5/nps`. Commit del informe del DS:
`afp_lab/voz_afiliado/{synthetic,features,model,evaluate,run}.py`, `tests/test_voz_afiliado.py`,
`README.md`. Revisor: solo lectura (`git status`/`git diff main`), sin cambios de código.

## Veredicto: **rechazar**

No hay leakage temporal (as-of correcto y bien probado) ni uso de la propensión latente verdadera
para la corrección IPW — los dos riesgos más graves del brief están bien resueltos. Pero encontré
**dos bugs metodológicos concretos, reproducibles y con evidencia de archivo/línea** que invalidan
dos de los resultados más citados del informe: (1) el baseline de "regla de negocio" no es
independiente del generador — copia casi exactamente sus coeficientes verdaderos, pese a que el
código dice explícitamente "sin ver el generador" — así que su victoria sobre el modelo (MAE 1,13
vs 6,19) no prueba nada sobre reglas de negocio simples, y la narrativa del README atribuye esa
derrota a límites del modelo de respuesta cuando la causa real es que el "baseline" tiene acceso
(indebido) a la fórmula generadora; y (2) el modelo de NPS pondera por IPW **y además usa el propio
peso IPW como feature de entrada**, lo que **degrada** el AUC del clasificador de detractor y hace
que el criterio de aceptación "AUC ≥ 0,75" reporte FALLA (0,7497) cuando sacando esa única columna
el mismo modelo, con la misma semilla, **pasa** el umbral (0,7530). Ambos son arreglables en poco
tiempo (quitar `peso_ipw`/`p_hat_responde` de las features del modelo de NPS; rehacer el baseline de
regla con coeficientes que NO coincidan con `ParametrosSatisfaccion`), pero tal como está entregado,
dos números "de aceptación" están mal explicados o son directamente incorrectos.

---

## Bloqueantes

### 1. El baseline de "regla de negocio" no es independiente del generador: copia sus coeficientes verdaderos

**Archivo:** `afp_lab/voz_afiliado/model.py:169-190` (dict `COEF_REGLA` y `baseline_score_regla`),
comparado con `afp_lab/voz_afiliado/synthetic.py:44-72` (`ParametrosSatisfaccion`, los parámetros
LATENTES del generador de `lat_satisfaccion`).

**Detalle:** CONVENCIONES §4 exige un baseline "de reglas de negocio simples" que el modelo debe
superar; el sentido de ese requisito es comparar el modelo contra un heurístico plausible que un
analista de negocio armaría SIN conocer el mecanismo oculto que genera los datos. El docstring de
`baseline_score_regla` dice literalmente: *"parte de 7 y suma/resta según señales observables (**sin
ver el generador**), con coeficientes redondeados a mano"* (`model.py:181-182`). Verifiqué los
valores uno a uno contra `ParametrosSatisfaccion` (que sí es el generador, el origen de
`lat_satisfaccion`):

| Coeficiente | `COEF_REGLA` (baseline) | `ParametrosSatisfaccion` (generador, verdad latente) |
|---|---:|---:|
| nivel base | 7,6 | `base_nivel` = 7,6 |
| reclamo sin resolver | −2,0 | `ef_reclamo_sin_resolver` = −2,0 |
| rezago mensual | −1,5 | `ef_rezago_mensual` = −1,5 |
| licencia mal pagada | −1,5 | `ef_licencia_mal_pagada` = **−2,5** |
| DNP del empleador | −0,8 | `ef_dnp_empleador` = −0,8 |
| contacto ejecutivo resuelto | +0,7 | `ef_contacto_ejecutivo_resuelto` = +0,7 |
| brecha de comisión desfavorable | −0,6 | `ef_brecha_comision_desfavorable` = −0,6 |

5 de 6 coeficientes de eventos son **idénticos, a la décima**, a los parámetros latentes del
generador (solo "licencia" difiere, −1,5 vs −2,5), y el nivel base (7,6) también es exactamente el
mismo valor que `base_nivel` — de hecho el propio código lo admite para el nivel base
(`model.py:170-173`: *"mismo ajuste que `synthetic.ParametrosSatisfaccion.base_nivel`"*), pero el
README (sección 8, "Decisiones y supuestos") solo declara la reutilización del nivel base 7,6, **no**
la de los otros 5 coeficientes, y el docstring del propio código sigue afirmando "sin ver el
generador" — una contradicción interna que un lector del README no puede detectar sin comparar
ambos archivos como hice acá.

**Por qué invalida un resultado:** el "baseline de regla" gana al modelo de forma aplastante
(MAE 1,13 vs 6,19 puntos de NPS, ver `reportes/nps_error_resumen.csv`, reproducido abajo) porque
reconstruye casi exactamente la parte determinista de la fórmula que genera `lat_satisfaccion`
(le faltan el decaimiento temporal, el componente persistente, el ruido y el término de
rentabilidad — por eso no es perfecto, pero captura la mayor parte de la señal). Esto no demuestra
que "una regla de negocio simple le gana al modelo" (que es la lectura que CONVENCIONES §4 y el
README sugieren: *"Baseline regla de negocio: MAE=1.13 ... supera al modelo (ver limitaciones)"* en
el informe del DS) — demuestra que un baseline con acceso indebido a los parámetros del simulador
le gana, trivialmente, a cualquier modelo que solo vea features observables con ruido. El README
(§6.1, §8) atribuye la derrota del modelo únicamente al AUC bajo del modelo de respuesta (0,54),
sin mencionar que el rival tiene una ventaja estructural que ningún modelo entrenado con datos
observables podría igualar. Esto también contamina la tabla "qué arreglar primero" (§7): como la
regla y el generador comparten casi los mismos coeficientes, cualquier lectura de negocio que
compare "impacto según el modelo" vs "impacto según la regla" está comparando el modelo contra una
copia del oráculo, no contra un heurístico independiente.

**Cómo verificar (comando):**
```
cd /home/user/wt/empleador && PYTHONPATH=/home/user/wt/empleador python3 -c "
from afp_lab.voz_afiliado.synthetic import ParametrosSatisfaccion
from afp_lab.voz_afiliado.model import COEF_REGLA
p = ParametrosSatisfaccion()
print(p.base_nivel, p.ef_reclamo_sin_resolver, p.ef_rezago_mensual, p.ef_licencia_mal_pagada,
      p.ef_dnp_empleador, p.ef_contacto_ejecutivo_resuelto, p.ef_brecha_comision_desfavorable)
print(COEF_REGLA)"
```

---

### 2. El modelo de NPS usa su propio peso IPW como *feature*, no solo como ponderador — y eso le hace fallar el criterio de AUC que debería cumplir

**Archivo:** `afp_lab/voz_afiliado/run.py:125-126` (se agrega `d["peso_ipw"]` a `tr`/`va`/`te` ANTES
de entrenar los modelos de NPS) y `afp_lab/voz_afiliado/features.py:267-274` (`columnas_features`
solo excluye prefijos `lat_`/`p_`; `peso_ipw` no empieza con ninguno de los dos, así que **queda
adentro** de las columnas usadas para entrenar `entrenar_score`/`entrenar_detractor`/
`entrenar_promotor`, tanto en la versión ponderada como en la "sin ponderar"). El propio README lo
admite en la tabla de importancias (`README.md:239`: *"`peso_ipw` (6,4 %, usado como feature
adicional — el modelo aprende a distinguir el régimen de pocos/muchos respondientes similares)"*),
pero lo presenta como una decisión deliberada y beneficiosa, no como lo que en realidad es: una
variable derivada del propio modelo de propensión (`p_hat_responde`, que si se llamara así quedaría
excluida por el prefijo `p_` — el nombre `peso_ipw` "esquiva" el filtro por casualidad de
nomenclatura) inflando artificialmente el modelo con una señal redundante y ruidosa.

**Por qué invalida un resultado — lo reproduje yo mismo:** entrené el clasificador de detractor
ponderado (mismos hiperparámetros, mismo `seed=42`, mismo dataset de la corrida de referencia
guardada en `/tmp/voz_afiliado_rev/dataset.parquet`) con y sin la columna `peso_ipw` como feature
(dejándola SOLO como peso de muestra, que es como debería usarse):

| Versión | AUC detractor (ponderado, holdout test respondientes) | ¿Cumple ≥ 0,75? |
|---|---:|---|
| Tal como está entregado (con `peso_ipw` como feature) | **0,7497** | No (así lo reporta `aceptacion.csv`) |
| Quitando `peso_ipw` de las features (dejándola solo como peso) | **0,7530** | **Sí** |

Es decir: el criterio de aceptación "AUC de detractor (ponderado) ≥ 0,75" — uno de los 4 criterios
propuestos por el brief y el único de clasificación — está marcado como no cumplido en
`reportes/aceptacion.csv` por una diferencia de 0,0003, y esa diferencia se explica enteramente por
un bug de ingeniería de features (usar como input un valor derivado del propio pipeline de
modelado), no por una limitación de fondo del enfoque. El README (§6.1, §8) presenta el AUC de 0,75
como evidencia de que "el modelo de respuesta discrimina poco" y por eso el NPS no se puede corregir
mejor — un diagnóstico que no es falso en sí (el AUC de 0,54 del modelo de respuesta es real y sí
limita el IPW), pero el número puntual que se usa para ese diagnóstico (el AUC de detractor) está
sesgado a la baja por este bug, y con la corrección simple el criterio se cumple. Esto es
exactamente el tipo de "número de aceptación no reproducible/mal calculado" que se pidió refutar.

**Cómo verificar (reproducido con el dataset y el split de la corrida de referencia):**
```
cd /home/user/wt/empleador && PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.voz_afiliado.run
# luego comparar contra el mismo pipeline entrenando entrenar_detractor con
# cols = [c for c in columnas_features(df) if c != "peso_ipw"] en vez de columnas_features(df)
```
(script exacto usado para la comparación de arriba, ejecutado dos veces para confirmar
determinismo: AUC idéntico a 10 decimales en ambas repeticiones).

**Nota:** este mismo problema afecta a los 3 pares de modelos (score, detractor, promotor;
ponderado y sin ponderar), porque `peso_ipw` se agrega a `tr`/`va`/`te` antes de separar
train/valid/test en respondientes y se usa igual en las 6 llamadas de `run.py:136-144`. También
afecta al NPS inferido en sí (`nps_de_probs`, que promedia las probabilidades de estos mismos
clasificadores) y al valor del punto de NPS (usa `modelo_score_p`), aunque no medí cuánto cambian
esos números específicos por el presupuesto de tiempo de esta revisión — recomiendo que el DS lo
haga como parte de la corrección.

---

## Menores

* **Interpretabilidad incompleta (CONVENCIONES §4):** la plantilla pide "importancia de features
  (gain) **y 3 a 5 ejemplos individuales explicados**". El README (§6.5, `README.md:236-242`) solo
  trae la tabla de importancia global; no hay ningún ejemplo individual explicado (ni SHAP ni a
  mano), a diferencia de proyecto 02. El propio informe del DS lo reconoce en "pendientes". No lo
  elevo a bloqueante porque no invalida un resultado, pero es un ítem del checklist del gerente
  (§7 de CONVENCIONES) que no está cumplido.
* **DNP/rezago del empleador leído con el desfase del panel (`corte-2`) en vez de por
  `periodo_recepcion` real** (`features.py:14-19`, `136-149`): es una simplificación optimista
  declarada explícitamente en el README (§4, §8) y en el docstring del módulo; la dejo como menor
  porque está bien documentada y acotada, no oculta.
* **Drivers (T-learner) estimados solo sobre respondientes** (`model.py:232-265`, `resp_trva` en
  `run.py:212-213`): la ponderación IPW ayuda, pero el T-learner en sí no corrige una posible
  interacción entre "quién responde" y "el efecto del evento dentro de cada brazo tratado/control"
  más allá de reponderar la muestra; es una limitación de fondo de un estimador "simple" (como el
  propio brief permite), declarada como tal en el README §5.3/§8. No hay evidencia de que cambie el
  signo de los efectos (los 6 estimables coinciden en signo y orden de magnitud con el generador),
  así que la dejo como menor.
* **Un solo `seed` (42) corrido**; no hay barrido de semillas para reportar rango de estabilidad
  (declarado como limitación en README §8 y en el informe del DS). Los MAE/AUC reportados podrían
  moverse con otra semilla; sin barrido no se puede saber cuánto.
* **`nps_de_score` vs `nps_de_probs`:** el hallazgo documentado sobre "encogimiento" de varianza al
  bucketizar un regresor (`model.py:131-142`) es correcto y está bien resuelto (se usa
  `nps_de_probs` para el NPS inferido reportado); lo menciono solo para dejar constancia de que lo
  revisé y no encontré un error ahí — al contrario, es una de las mejores decisiones técnicas del
  proyecto.
* El evento `brecha_comision_desfavorable` queda sin poder estimarse en el T-learner (0 controles,
  `drivers.csv`) porque en el perfil `cuprum` la brecha de comisión es casi siempre desfavorable;
  está bien diagnosticado como propiedad del mercado simulado, no como bug, y correctamente excluido
  del cálculo de "% de signo correcto" (`run.py:219-221`).

---

## Verificación realizada (con evidencia de archivo/línea)

* **Tests:** `PYTHONPATH=/home/user/wt/empleador python3 -m pytest afp_lab/voz_afiliado/tests -q`
  → **11 passed en 4.07 s** (límite pedido: < 60 s). Cubren leakage, historial de encuesta,
  universo común 02==03, bucketización NPS, T-learner sintético y corrida end-to-end. No están
  vacíos: cada uno tiene aserciones sustantivas (revisé el archivo completo, 250 líneas).
* **Corrida de referencia reproducida:**
  `PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.voz_afiliado.run` (n=10 000, seed=42,
  ventana 2022-01/2028-12) → **28,82 s totales** (informe del DS: 29,5 s; diferencia normal de
  máquina, ambos muy por debajo del límite de 4 min). Números reproducidos y coincidentes con el
  informe del DS (diferencias solo de redondeo):
  * MAE ponderado (modelo IPW) = 6,191 puntos (informe: 6,19) — no cumple ±3.
  * MAE ingenuo = 8,470; MAE Horvitz-Thompson directo = 5,325; MAE regla = 1,127; MAE trivial = 11,629
    — todos calzan con `reportes/nps_error_resumen.csv` y con el informe del DS.
  * AUC detractor ponderado = 0,749735 (informe: 0,7497) — falla el umbral 0,75 por 0,0003, **pero
    ver Bloqueante #2**: sin el bug de `peso_ipw` como feature, el mismo modelo da 0,7530 (sí
    cumple), verificado con el mismo dataset/seed/partición de esta corrida.
  * Drivers: 6/6 estimables con signo correcto (100 %), magnitudes reproducidas idénticas a
    `reportes/drivers.csv` (licencia −2,62 vs −2,5 verdadero; reclamo −0,97 vs −2,0; rezago −0,95 vs
    −1,5; DNP −0,19 vs −0,8; contacto +0,48 vs +0,7; rentabilidad +0,62 vs +0,5) — **ver Bloqueante
    #1** sobre por qué el baseline de regla que se compara contra esto no es independiente.
  * Valor del punto: 126,7 traspasos evitados/punto (informe: ≈127), CLP 3.606 MM saldo/punto,
    CLP 8.517/punto de aporte APV (≈CLP 645 MM cartera) — coincide con el informe.
  * `resumen_simulacion.csv`: invitación 8,02 % (brief ~8 %), respuesta 25,14 % (brief ~25 %), NPS
    verdadero de cartera −14,9 — coherente con los supuestos declarados.
* **Leakage (código, no solo tests):** revisé `synthetic.py` (generación de la encuesta y de
  `lat_satisfaccion`) y `features.py` (`_features_propias_multi`, líneas 152-197) línea por línea: el
  historial de encuesta usa `jj_hist = max(jj-1, 0)`, que excluye estrictamente el propio corte
  (verificado también a mano en `test_historial_encuesta_excluye_el_propio_corte`); DNP/rezago del
  empleador y licencias se leen en `jj_lab = jj - MESES_DESFASE_RIM` (`corte-2`), consistente con la
  convención de fechas de CONVENCIONES §3. No encontré ninguna columna `lat_*` ni `p_*`
  (de empleador latente) llegando a `columnas_features` (`features.py:267-274`, y confirmado en
  `test_forma_dataset_y_columnas`).
* **Corrección de sesgo de no respuesta:** confirmé que `peso_ipw` en `run.py:126` se calcula sobre
  `p_hat_responde`, la salida del modelo LightGBM `entrenar_propension_respuesta` (estimado), **no**
  sobre `lat_p_responde_si_invitado` (la propensión verdadera latente); grep de
  `lat_p_responde` en todo el proyecto confirma que esa columna solo se usa dentro de `synthetic.py`
  (para generar los datos) y en un test, nunca en `model.py`/`run.py`/`evaluate.py`. Esta es la
  refutación específica que se pidió intentar y **no prosperó**: está bien implementado.
* **NPS inferido evaluado contra respondientes vs. cartera completa:** confirmé en
  `evaluate.py:34-60` que `cat_verdadera` (y por lo tanto `nps_verdadera`) se calcula sobre TODA la
  población de `df` (afiliados vigentes en el corte, no solo invitados/respondientes), usando
  `satisfaccion_latente` — no es una comparación circular contra solo respondientes. Confirmado
  también con `test_nps_por_corte_verdadero_e_ingenuo`.
* **Valor del punto presentado como causal:** no encontré ninguna frase que lo presente como
  causal; al contrario, `model.py:293-296,318-319`, `run.py:231-234` y `README.md:161-167,227,252-256`
  repiten la advertencia de que es asociación ajustada, no un experimento, de forma consistente.
* **Umbrales de aceptación:** comparé los 4 criterios codificados en `run.py:238-251`
  (±3 puntos, reducción ≥50 %, AUC≥0,75, drivers≥80 %) contra los del brief
  (`docs/briefs/ronda_5/voz_afiliado.md:47-50`): son idénticos, no hay umbrales bajados.

---

## Re-verificación (post-corrección del DS)

Worktree en el mismo estado (`ronda-5/nps`, todo sin commitear —`git status` solo muestra
`afp_lab/voz_afiliado/` y `docs/revisiones/ronda_5/` como *untracked*, nada en stage). Solo usé
`git status`/`git branch --show-current` (lectura). No corregí código.

### Veredicto de esta pasada: **aprobar_con_reserva**

Los dos bloqueantes originales están **confirmados resueltos**, con verificación independiente
(no me limité a releer el informe del DS ni el README: reentrené yo mismo con el dataset cacheado
de la revisión original y reproduje los números exactos, dígito a dígito). Dejo la aprobación
"con reserva" por un solo hallazgo nuevo, menor y no bloqueante: el comando de verificación que el
propio README/informe documentan para el bloqueante #1 **no corre tal como está escrito** (lanza
`TypeError`, ver detalle abajo) — el resultado que reportan ("conjunto vacío") no es reproducible
con ese comando literal, aunque el hecho que afirma (coeficientes independientes) sí es verdadero y
sí está bien probado por otra vía (el test de regresión, que usa una técnica distinta y correcta).

#### Bloqueante #1 — CONFIRMADO RESUELTO

* `model.py:181-193` (`COEF_REGLA`) tiene ahora `base=7,5`, `reclamos_6m=-1,0`,
  `rezago_mensual_actual=-1,0`, `licencia_larga_3m=-1,0`, `dnp_empleador_actual=-0,5`,
  `contactos_ejecutivo_propio_3m=0,4`, `brecha_comision=-0,5`. Comparé cada uno a mano contra los 7
  parámetros relevantes de `synthetic.ParametrosSatisfaccion` (`base_nivel=7,6`,
  `ef_reclamo_sin_resolver=-2,0`, `ef_rezago_mensual=-1,5`, `ef_licencia_mal_pagada=-2,5`,
  `ef_dnp_empleador=-0,8`, `ef_contacto_ejecutivo_resuelto=0,7`,
  `ef_brecha_comision_desfavorable=-0,6`): **ninguno coincide**, la ronda anterior tenía 6/7
  coincidencias. Correcto.
* `tests/test_voz_afiliado.py:229-243`
  (`test_regla_negocio_independiente_del_generador`) hace exactamente esa comparación (compara
  cada valor de `COEF_REGLA` contra el conjunto de los 7 parámetros relevantes del generador,
  incluyendo `ef_rentabilidad_relativa_positiva` aunque no se use en la regla) y pasa.
* **Hallazgo nuevo (menor, no bloqueante):** el comando de verificación documentado en
  `README.md:341` y citado en el informe del DS ("numeros", bloqueante #1) —
  `print(set(dataclasses.asdict(p).values()) & set(COEF_REGLA.values()))` — **no se puede
  ejecutar**: `ParametrosSatisfaccion.base_segmento` es un `dict` (se completa en `__post_init__`),
  y `dataclasses.asdict(p).values()` lo incluye, así que `set(...)` lanza
  `TypeError: unhashable type: 'dict'`. Lo ejecuté literalmente y confirmé el error:
  ```
  cd /home/user/wt/empleador && PYTHONPATH=/home/user/wt/empleador python3 -c "
  from afp_lab.voz_afiliado.synthetic import ParametrosSatisfaccion
  from afp_lab.voz_afiliado.model import COEF_REGLA
  import dataclasses
  p = ParametrosSatisfaccion()
  print(set(dataclasses.asdict(p).values()) & set(COEF_REGLA.values()))"
  # -> Traceback ... TypeError: unhashable type: 'dict'
  ```
  Es decir, el informe del DS afirma que este comando imprime "conjunto vacío" pero en realidad
  **no corre** — el número/resultado citado no es reproducible tal como está documentado. No cambia
  el veredicto de fondo (la independencia de `COEF_REGLA` SÍ está bien verificada, por el test de
  arriba y por mi comparación manual campo a campo), pero es una inexactitud puntual en la
  documentación que el DS debería arreglar en una próxima pasada (p. ej. comparar explícitamente los
  7 campos numéricos relevantes, como hace el test, en vez de volcar todo `asdict()` a un `set`).

#### Bloqueante #2 — CONFIRMADO RESUELTO (verificación independiente, no solo re-lectura)

* `features.py:57-58` agrega `COLUMNAS_DERIVADAS_PIPELINE = ["p_hat_responde", "peso_ipw"]` a
  `COLUMNAS_NO_FEATURE`; `columnas_features()` ahora excluye ambas explícitamente (antes solo
  `p_hat_responde` se filtraba, por casualidad, vía el prefijo `p_`). `run.py`/`evaluate.py` siguen
  usando `peso_ipw` únicamente como `sample_weight` (líneas 138-144, 235) y para el estimador
  Horvitz–Thompson (`evaluate.py:78-84`), nunca como columna de entrada — revisé ambos archivos
  completos.
* `tests/test_voz_afiliado.py:216-226`
  (`test_columnas_features_excluye_derivadas_pipeline`) agrega `peso_ipw`/`p_hat_responde` a un
  DataFrame de prueba y confirma que quedan fuera de `columnas_features()`. Pasa.
* **No me conformé con el test ni con releer el informe: reentrené yo mismo**, usando el dataset
  cacheado de la revisión original (`/tmp/voz_afiliado_rev/dataset.parquet` +
  `reportes/split.csv`, la misma corrida de referencia citada en el bloqueante original) y el
  **código actual** de `model.py`/`features.py`, comparando (a) el entrenamiento tal como lo deja
  el código de hoy (`peso_ipw` excluido de `columnas_features`) contra (b) una simulación explícita
  del bug original (llamando a `M._entrenar_booster` con `columnas_features(...) + ["peso_ipw"]`
  como columnas, bypaseando el filtro). Mismo `seed`, mismo split, mismo dataset:
  | Versión | AUC detractor ponderado (holdout test, n=1549) |
  |---|---:|
  | Código actual (`peso_ipw` excluido de features) | **0,7530290638938333** |
  | Bug simulado (`peso_ipw` reincluido como feature) | **0,7497354079185194** |

  Ambos valores **coinciden dígito a dígito** con los que el bloqueante original y el informe del
  DS reportan (0,7530 / 0,7497), y con `salidas/voz_afiliado/reportes/detractor_metricas.csv` de la
  corrida de referencia post-corrección ya presente en el worktree (`auc=0,7530290638938333`,
  generada el 2026-09-14 00:33, después de los cambios en `model.py`/`features.py`). Esto confirma
  no solo que el bug fue corregido, sino la relación causal exacta: la única diferencia entre ambas
  columnas es `peso_ipw` como feature, y remueve esa única columna basta para pasar de 0,7497 a
  0,7530 y de "no cumple" a "cumple" el umbral ≥0,75 de `aceptacion.csv`. Script usado (no modifica
  nada del proyecto, vive en `/tmp/voz_afiliado_rev/verificar_bloqueante2.py`):
  ```
  cd /home/user/wt/empleador && PYTHONPATH=/home/user/wt/empleador python3 /tmp/voz_afiliado_rev/verificar_bloqueante2.py
  ```
* Efecto lateral honesto en el MAE del NPS (documentado por el DS, no oculto): confirmé en
  `salidas/voz_afiliado/reportes/nps_error_resumen.csv` (misma corrida post-corrección) que MAE
  ponderado = 6,749741 (antes 6,19) y sin_ponderar = 8,203852 (antes 6,64, ahora casi igual al
  ingenuo 8,469808) — coincide exactamente con lo que reporta el informe del DS. Sigue sin cumplirse
  el criterio "±3 puntos" ni "reducción ≥50 % vs ingenuo" (queda en 20,3 %), pero esto **ya no es
  atribuible al bug**: es la consecuencia honesta de sacar una señal espuria, y el DS lo declara
  correctamente como limitación de fondo pendiente (AUC del modelo de respuesta = 0,54), no como
  bloqueante nuevo — coincido con esa lectura.

#### Otras verificaciones de esta pasada

* **Tests:** `PYTHONPATH=/home/user/wt/empleador python3 -m pytest afp_lab/voz_afiliado/tests -q`
  → **13 passed en 3,98 s** (antes 11 en 4,07 s; límite pedido < 60 s). Los 2 tests nuevos son
  exactamente los de regresión de los 2 bloqueantes (`test_columnas_features_excluye_derivadas_pipeline`,
  `test_regla_negocio_independiente_del_generador`); ningún test está `skip`/`xfail`.
  Corrí `grep -n "^def test_\|skip\|xfail"` sobre el archivo completo: 13 funciones `test_*`, cero
  skips/xfails.
  `test_run_end_to_end` ahora también verifica que `run.main()` genere
  `importancias_score.csv` y `ejemplos_individuales.csv` (confirmado leyendo el test).
* **Corrida de referencia:** no repetí la corrida completa (ya existe una posterior a la
  corrección en `salidas/voz_afiliado/`, con `model.py`/`features.py` modificados ANTES que los
  archivos de `reportes/` — confirmado con `stat`: código a las 00:22-00:33, reportes a las
  00:33-00:34 — así que sí refleja el código actual) y en su lugar usé la corrida más barata que
  pedía el encargo: el script dirigido de arriba (entrena solo el modelo de respuesta + un
  clasificador de detractor sobre datos ya generados, no las ~17 s de `generar_tablas`). Igual
  reviso que `reportes/tiempos.csv` de esa corrida completa da **TOTAL 27,55 s** (< 4 min) y
  `aceptacion.csv`: NPS ±3 = 6,75 (no cumple), MAE≥50 % = 79,7 % del ingenuo (no cumple), AUC≥0,75 =
  0,7530 (**cumple**, antes no), drivers≥80 % = 100 % (cumple) — 2/4, igual a lo que reporta el DS.
* **Nada de git de escritura:** confirmado (`git status`/`git branch --show-current` únicamente).

### Resumen para el gerente

Ambos bloqueantes de la ronda anterior están genuinamente corregidos de raíz (no es un maquillaje
de números ni una baja de umbral): verifiqué el mecanismo con datos y código independientes de lo
que reportó el DS, y los valores reproducen exactamente. El único hallazgo nuevo es menor y de
documentación (un comando de ejemplo en el README que no corre tal como está escrito, por un campo
`dict` no hasheable) y no afecta la validez de la corrección, que está bien probada por otra vía.
Recomiendo aprobar esta entrega; el arreglo del comando de ejemplo en README §8.0 puede ir en la
próxima pasada, junto con los pendientes ya declarados (NPS ±3, barrido de semillas) que no son
bloqueantes.
