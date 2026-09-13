# Revisión ronda 4 · 07 uplift_retencion (T/X/DR-learner sobre campaña confundida + holdout)

**Veredicto: APROBAR CON RESERVA** (revisor adversarial Sonnet). No encontré bloqueantes: sin leakage
en `features.py`/`synthetic.py`, target de control byte a byte igual al de 02 (verificado por test y a
mano), tests sustantivos (incluye hand-check de historial de campaña y mutación de futuro), y las cifras
del README reproducen casi exactamente con el comando exacto del README. La reserva es sobre el propio
resultado que el README ya reporta con honestidad: el criterio de aceptación principal (Qini > 0 en el
holdout aleatorio) se cumple solo por el X-learner y con un margen estadísticamente no distinguible de
cero — el proyecto lo dice explícitamente, no lo esconde, pero es la razón para no dar aprobación plena.

## Qué reproduje

- `PYTHONPATH=/home/user/wt/churn python3 -m pytest afp_lab/uplift_retencion/tests -q` →
  **13 passed en 4,66 s** (README: "13 tests, ~5 s" — coincide).
- `PYTHONPATH=/home/user/wt/churn python3 -m afp_lab.uplift_retencion.run --n-afiliados 10000 --salida
  /tmp/rev_uplift` (comando exacto del README §9) → **26,24 s** (README: 26,2 s). Cifras clave
  reproducidas dígito a dígito:
  - Overlap propensión (test): **1,00** (README: 1,00; criterio ≥0,80 cumple).
  - Spearman/MAE vs `lat_cate`: DR-learner 0,1948/0,0078; X-learner 0,1858/0,0080; T-learner
    0,1521/0,0133; riesgo 0,0544/0,0264 — coincide con README §6.2.
  - Qini agregado por corte de test (holdout aleatorio): X-learner es el único con Qini > 0 en los 3
    cortes (202709 +0,00127, 202710 +0,00143, 202711 +0,00015); T-learner, DR-learner y riesgo
    negativos en al menos un corte — coincide con README §6.3.
  - Traspasos evitados: X-learner **14,64** (10 %) / **20,85** (20 %) vs riesgo 3,23/7,59 → razón
    **4,54x / 2,75x** (README: 4,54x/2,75x, criterio ≥1,3x cumple con holgura).
  - Tabla segmento→acción: persuadible n=123 (0,67 %), sleeping_dog n=358 (1,96 %), neutral 97,36 % —
    coincide con README §6.5.
  - Tiempos: generar+construir 9,43 s, entrenar 16,31 s, Qini 0,43 s, total 26,24 s (README: 9,5/16,2/
    0,4/26,2 — coincide).

## Sin leakage (verificado a mano, además de los tests)

- `features.py` reutiliza `CT_F.construir_features_multi` sin modificar (panel laboral en `corte-2`,
  resto en `corte`, ya validado en rondas anteriores). El único insumo propio de 07,
  `_historial_campana` (`features.py:41-78`), calcula el conteo móvil de contactos usando `T_prev`
  (el tratamiento del corte **anterior**, línea 59: `T_prev = np.where(nuevo, 0.0,
  np.concatenate(([0.0], T[:-1])))`), es decir el `T` del propio corte se excluye explícitamente antes
  de armar la ventana — confirmado con el test `test_historial_campana_excluye_el_corte_actual`
  (`tests/test_uplift_retencion.py:137-160`), que hice pasar a mano con una traza de 5 cortes.
  `columnas_features` (`features.py:117-123`) excluye además `T`, `holdout` y los tramos de negocio.
- `synthetic.py`: el score de riesgo que confunde la propensión (`riesgo_latente_heuristico`,
  líneas 82-102) y los segmentos del efecto causal (`construir_efecto_causal`, líneas 157-191) usan
  solo columnas as-of ya presentes en `construir_features_multi` (`reclamos_6m`,
  `contactos_vendedor_competencia_3m`, `brecha_rentabilidad_pp`, `brecha_comision_pp`,
  `antiguedad_afp_meses`, `logins_3m`) más `lat_sensibilidad_comision` (latente por afiliado, nunca
  pasa a `columnas_features`): ninguna variable `lat_*` ni `e_hist` llega a `df` (solo a `latente`,
  ver `features.py:110-114`).
- El target de control es **exactamente** `y0` de `churn_traspaso.construir_target`
  (`synthetic.py:194-219`, `aplicar_efecto_causal`): las filas `T=0` nunca se tocan, solo se flippean
  probabilísticamente las filas `T=1` según segmento — verificado por
  `test_control_y_es_identico_a_y0_de_02` y reproducido con la corrida completa (columna `y` = `y0`
  en 100 % de las filas de control).
- `test_leakage_futuro_no_cambia_features_ni_historial` (líneas 97-134) altera/borra el futuro de 6
  tablas operacionales + panel + traspasos y confirma que ninguna feature (incluido el historial de
  campaña) cambia para un corte intermedio: pasa.
- Meta-learners y propensión se entrenan **solo con la población confundida** (fuera del holdout);
  el holdout aleatorio (10 %) se reserva íntegro para Qini/AUUC — confirmado en `model.py` y
  `run.py`, y con `test_holdout_es_aleatorio_no_confundido` (tasa de tratamiento en holdout 30-70 %,
  sin separación por riesgo).

## Metodología de los meta-learners (revisada, correcta)

T-learner (`mu1-mu0`), X-learner (imputación D1/D0 + combinación `e(x)·tau0 + (1-e(x))·tau1`,
`model.py:107-131`) y DR-learner (pseudo-outcome AIPW estándar,
`(mu1-mu0) + T(y-mu1)/e - (1-T)(y-mu0)/(1-e)`, `model.py:134-150`) coinciden con las fórmulas
estándar de la literatura (Künzel et al. / doubly robust). La convención de signo de Qini (evaluar
sobre "retención" en vez de "traspaso" para que Qini>0 = mejor que el azar) está documentada y bien
implementada en `evaluate.py:1-19,30-75` — validé el sanity test de signo conocido
(`test_qini_positivo_cuando_el_score_ordena_bien`) a mano.

## Hallazgos menores

1. **El único criterio de aceptación duro (Qini>0 en los 3 cortes) se cumple con margen mínimo y sin
   significancia estadística** (README §6.3: IC bootstrap del agregado cruza cero, peor corte
   +0,00015). El README lo declara con total honestidad ("no se puede afirmar con confianza
   estadística"), pero es la base de la reserva del veredicto: la recomendación de negocio (§7) se
   apoya en la correlación contra `lat_cate` y en la métrica de negocio oracular, ambas solo posibles
   en este sandbox, no en el holdout ciego que sería la prueba real.
2. **Sin tabla de importancia de features (gain) ni ejemplos individuales explicados**, pedido por
   CONVENCIONES §4 para el modelo principal — declarado como pendiente en README §8, no oculto, pero
   sigue faltando.
3. Escala de referencia reducida a 10.000 afiliados (no 20.000 como especifica el brief 07 original);
   está declarada y validada con una corrida adicional a 30.000 (70 s, misma conclusión cualitativa),
   pero esa corrida de 30.000 no es la que reproduje ni la que se audita por defecto.

## Checklist §7 CONVENCIONES

- [x] Sin leakage: test presente + verificado a mano (arriba).
- [x] Target bien definido: idéntico a 02, coherencia por test.
- [x] Dinámicas sintéticas documentadas y con parámetros explícitos (tabla §2.4 del README).
- [x] Baselines presentes (riesgo, aleatorio) y el uplift-ranking los supera en la métrica de negocio.
- [x] Métricas comparables + métrica de negocio (traspasos evitados, razón vs riesgo).
- [x] Runtime y tests dentro de límites; reproducible con semilla 42.
- [ ] README completo: falta importancia de features / ejemplos individuales (§4 CONVENCIONES, punto
  menor 2 arriba).
