# Brief 07 — `afp_lab/uplift_retencion` (worktree `/home/user/wt/churn`)

## Pregunta
El proyecto 02 predice quién se va. Para la licitación de stock 2027 la AFP necesita saber a
quién CONVIENE contactar: el efecto causal del contacto sobre el traspaso, no el riesgo.
Hay afiliados que se quedan igual (contacto desperdiciado), que se van igual, persuadibles, y
"sleeping dogs" a los que el contacto los despierta.

## Datos sintéticos (`synthetic.py`)
* Base: `churn_traspaso.generar_tablas(n_afiliados, periodo_inicio, periodo_fin, seed, perfil="cuprum")`
  importado sin modificar (mismo universo determinista que 02).
* Nuevo generador de campañas históricas: en cada corte mensual una campaña asigna tratamiento
  `T` (contacto de ejecutivo) con una política histórica NO aleatoria (propensión que depende de
  saldo, tramo de edad y un score heurístico de riesgo: confounding real) más un holdout
  experimental del 10 % de la cartera con asignación aleatoria (para validar sin sesgo).
* Efecto verdadero heterogéneo `lat_cate` (tabla latente, nunca feature): reduce la
  probabilidad de traspaso en persuadibles (p. ej. reclamo reciente resuelto, saldo medio, uso
  digital medio), cero en la mayoría, y la AUMENTA en un subgrupo de sleeping dogs (bajo uso
  digital y alta sensibilidad a comisión). Documentar las reglas y las magnitudes.
* El target `y` (traspaso en corte+1..corte+6) se construye con la función de 02 y luego se
  altera por `T` y `lat_cate` de forma consistente (el generador aplica el efecto a la
  probabilidad, no a la etiqueta a posteriori si eso rompe la coherencia con `traspasos`).

## Features y modelos
* Features as-of de 02 (`construir_features_multi`) más historial de contactos previos
  (conteo 6/12 m, meses desde el último) y nada del futuro.
* Propensión `e(x)` estimada con LightGBM; reportar overlap (fracción con e(x) en [0,05; 0,95]).
* Meta-learners con LightGBM: T-learner, X-learner y DR-learner (doubly robust). Implementados
  a mano con scikit-learn/LightGBM (econml y causalml no están instalados; si `pip install econml`
  funciona puede usarse como referencia, no como única implementación).
* Baseline: ranking por riesgo (el modelo de 02) usado como si fuera uplift.

## Evaluación (`evaluate.py`)
* Qini y AUUC en el holdout aleatorio (insesgado) por corte de test, con intervalos bootstrap.
* Contra la verdad sintética: correlación de Spearman entre CATE estimado y `lat_cate`, y error
  absoluto medio.
* Métrica de negocio: traspasos evitados con presupuesto de contacto del 10 % y del 20 % de la
  cartera, uplift-ranking vs riesgo-ranking vs aleatorio, y cuántos sleeping dogs se contactan.
* Aceptación propuesta: Qini > 0 en holdout aleatorio en todos los cortes de test; al 20 % de
  presupuesto el uplift-ranking evita ≥ 1,3× los traspasos del riesgo-ranking; overlap ≥ 80 %.
* Splits temporales con gap como en 02 (`split_temporal`).

## Entregables
Proyecto completo (estructura §1), README con tabla de resultados y "segmento → acción",
tests, corrida de referencia `python3 -m afp_lab.uplift_retencion.run` (< 5 min a 20 000
afiliados) e informe estructurado. Opcional: integrar el CATE en `churn_traspaso.licitacion.ranking_combinado`
sin modificar 02 (función propia en 07 que lo consume).
