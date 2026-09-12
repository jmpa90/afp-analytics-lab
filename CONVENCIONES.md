# Convenciones del laboratorio

Estas reglas aplican a todos los proyectos. El gerente revisa contra esta lista.

## 1. Estructura de un proyecto (`afp_lab/<proyecto>/`)

| Archivo | Contenido |
|---|---|
| `BRIEF.md` | Escrito por el gerente. No se modifica; las dudas se anotan en el README bajo "Decisiones tomadas". |
| `synthetic.py` | Tablas sintéticas propias del proyecto, construidas sobre `afp_lab.common.synth_base.generar_universo`. Parámetros numéricos explícitos y documentados. |
| `features.py` | Features "as-of" (ver §3). Una función `construir_features(tablas, periodo_corte) -> DataFrame`. |
| `model.py` | Entrenamiento, predicción y persistencia. Baselines obligatorios (§4). |
| `evaluate.py` | Métricas con `afp_lab.common.metrics`, por horizonte y por segmento. |
| `run.py` | Demo end-to-end: `python -m afp_lab.<proyecto>.run [--n-afiliados N --salida DIR]`. Debe correr en menos de 5 minutos con 20.000 afiliados en un laptop de 4 núcleos. |
| `tests/` | pytest, menos de 2 minutos. Mínimo: leakage (§3), forma del dataset, corrida end-to-end pequeña. |
| `README.md` | Resultados y supuestos (plantilla en §6). |

Salidas (parquet, csv, modelos) van a `salidas/<proyecto>/` y están en `.gitignore`. Nada real: ni RUT,
ni nombres, ni datos de clientes.

## 2. Datos sintéticos

* Partir del universo común y **no** redefinir afiliados, empleadores ni el panel laboral. Agregar tablas
  propias (eventos, contactos, aportes, etc.) coherentes con la historia laboral simulada: un cesante no
  hace APV, un traspaso ocurre con la AFP de destino, etc.
* Las columnas `lat_*` y los parámetros `p_*` de empleadores son latentes: prohibido usarlos como feature.
  Sí pueden usarse para generar la dinámica del target.
* Documentar en el README cada mecanismo simulado y sus parámetros (proporciones, tasas, elasticidades),
  con una frase de por qué es razonable para Chile.
* El target debe tener ruido no explicable: si el modelo alcanza AUC > 0,95 o WAPE < 0,05 en test, revisar
  que la simulación no sea determinista.
* Semilla fija (`seed`) y generación determinista.

## 3. Sin leakage (as-of estricto)

* Cada fila se construye en un `periodo_corte` con información conocida hasta ese mes; el target ocurre
  después (`corte+1 … corte+h`). Nunca usar columnas del panel posteriores al corte.
* Split **temporal**: entrenar con cortes antiguos, validar y testear con cortes posteriores, dejando un
  gap de al menos `h` meses entre el último target de entrenamiento y el primer corte de test.
* Test obligatorio: alterar o borrar cualquier registro posterior al corte no cambia las features.
* Convención de fechas (acordada en la ronda 1): `periodo_corte` es el **mes en que se toma la decisión**
  (se arma la lista, se proyecta). Cada fuente se lee con su propio desfase: panel laboral y RIM hasta
  `corte-2` (la cotización del mes m llega en m+2), eventos AFC y licencias hasta `corte-1`, tablas
  operacionales propias de la AFP (contactos, reclamos, aportes voluntarios, uso digital) hasta `corte`.
  El target empieza en `corte+1`. Así la población entrenada es la misma que se sirve en producción.
* El gap temporal de h meses se aplica **entre cada par de particiones** (train→valid y valid→test), no
  sólo antes de test: una validación que comparte meses de target con train no sirve para early stopping
  ni para calibrar. Validar y testear con al menos 12 cortes u orígenes móviles, y reportar por corte.
* Hiperparámetros: sólo con validación. Si un número de test influyó en una decisión, decirlo y
  re-evaluar en cortes posteriores no consultados.

## 4. Modelado y evaluación

* Stack: pandas + LightGBM + scikit-learn. PySpark sólo si la escala lo exige (proyecto 01).
* Baselines obligatorios: uno trivial (tasa base o persistencia) y uno de reglas de negocio simples.
  El modelo debe ganarles en la métrica principal y no perder en ningún segmento grande (≥ 5 % de filas).
* Métricas de `afp_lab.common.metrics`: clasificación (AUC, average precision, Brier, lift decil 1, KS),
  regresión (MAE, RMSE, WAPE, sesgo agregado). Siempre por segmento: tipo de afiliado, tramo de renta,
  tramo de edad, y los propios del proyecto.
* Calibración: si el modelo se usa para agregar (recaudación esperada, monto esperado), reportar curva de
  calibración o Brier y corregir con isotónica/Platt si hace falta.
* Interpretabilidad: importancia de features (gain) y 3 a 5 ejemplos individuales explicados.
* Métrica de negocio: traducir el resultado a la decisión (p. ej. "contactando al 10 % de mayor riesgo se
  captura el X % de los traspasos y el Y % del saldo en riesgo").

## 5. Código

* Python 3.10+, funciones puras con docstrings en español, sin notebooks en el repo.
* Sin `print` de debugging; el `run.py` imprime tablas de resultados y las guarda en `salidas/`.
* No ejecutar comandos git desde el código ni desde los agentes: el gerente integra. Y el gerente no
  ejecuta `checkout`/`reset`/`stash` sobre el árbol compartido mientras haya trabajo sin confirmar de un
  data scientist (en la ronda 1 un `git checkout -- .` revirtió ediciones en curso); desde la ronda 2 cada
  DS trabaja en su propio worktree y el gerente fusiona.
* No tocar carpetas de otros proyectos ni `common/` sin acordarlo en el README ("Propuestas para common").

## 6. Plantilla de README de proyecto

1. Pregunta de negocio y decisión que apoya.
2. Datos sintéticos: tablas, mecanismos y parámetros.
3. Definición del target, población, horizonte, exclusiones.
4. Features (tabla por familia, con la regla as-of de cada una).
5. Modelo y baselines.
6. Resultados: tabla principal, por segmento, calibración, importancias, ejemplos.
7. Lectura de negocio: qué haría la AFP con esto.
8. Decisiones tomadas, supuestos y limitaciones.
9. Cómo correr y tiempos.
10. Propuestas para `common/` y próximos pasos.

## 7. Checklist de revisión del gerente

- [ ] Sin leakage: test presente y features as-of verificadas a mano en 2 casos.
- [ ] Target bien definido y con población/exclusiones coherentes con el negocio.
- [ ] Dinámicas sintéticas plausibles para Chile y documentadas.
- [ ] Baselines presentes y el modelo los supera sin perder segmentos.
- [ ] Métricas comparables (common/metrics) y métrica de negocio.
- [ ] Runtime y tests dentro de límites; corrida reproducible con semilla.
- [ ] README completo según plantilla.
