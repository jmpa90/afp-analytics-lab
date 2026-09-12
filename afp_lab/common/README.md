# `common` · Universo sintético compartido

`afp_lab/common/synth_base.py::generar_universo` es la base que reutilizan los seis proyectos:
afiliados, empleadores y el panel mensual (estado laboral, empleador, sueldo, RIM, licencias,
eventos AFC), simulados por `rim_nowcasting.synthetic._simular_particion` con el perfil de cartera
(`PERFILES["sistema"]` o `PERFILES["cuprum"]`). Las columnas `lat_*` y los parámetros `p_*` de
empleadores son latentes: se usan para generar la dinámica, nunca como feature (`CONVENCIONES.md`
§2). Este archivo documenta los cambios **transversales** hechos aquí, porque afectan a los seis
proyectos a la vez.

## Ronda 3 (worktree `empleador`, objetivo 1): calibrar la densidad de cotización

**Problema.** La Ficha Estadística Previsional N° 165 (SP, ago-2026; `docs/contexto_afp_cuprum.md`
§0) da 5.775.070 cotizantes sobre 12.201.632 afiliados = **47,3 %** de densidad para el sistema y
**66 %** para Cuprum (66,94 % en la memoria 2025). Antes de esta ronda, `PERFILES["sistema"]` dejaba
~79 % de afiliados con RIM > 0 en un mes típico (optimista, documentado en la propia docstring desde
la ronda 2) y `PERFILES["cuprum"]` quedaba en ~71-73 % (fuera de la banda ±3 pp alrededor de 66 %).

**Mecanismo y parámetros ajustados.** La densidad de estado estacionario del panel la determina casi
enteramente `factor_contratacion` (escala la probabilidad mensual de contratación de un cesante:
`p_hire_base = 0,16 × factor_contratacion × (1 − 4·(desempleo − 0,08))`); `p_activo_inicial` (fracción
empleada al inicio de la ventana simulada) sólo corrige el sesgo transitorio de los ~57 meses de
ventana, porque con `factor_contratacion` bajo el panel tarda más en converger. Se barrieron ambos
parámetros (ver `commit`/histórico de `PERFILES`) hasta acotar la fracción con RIM > 0 en un mes
típico (periodos centrales de la ventana, sin los 12 primeros de *burn-in* ni los 3 últimos) dentro de
±3 pp del objetivo, medida con 20.000 afiliados y dos semillas (42 y 7):

| Perfil | `p_activo_inicial` | `factor_contratacion` | Densidad antes | Densidad después | Objetivo (Ficha 165) |
|---|---|---|---|---|---|
| `sistema` | 0,78 → **0,45** | 1,00 → **0,20** | ~78-79 % | **47,7-47,8 %** | 47,3 % ± 3 pp |
| `cuprum` | 0,68 (sin cambio) | 0,60 → **0,40** | ~71-73 % | **66,4-66,5 %** | 66 % ± 3 pp |

Test agregado: `afp_lab/common/tests/test_synth_base.py` mide esta fracción para ambos perfiles
(6.000 afiliados, semilla 42) y falla si se sale de la banda ±3 pp, y además exige que `cuprum` siga
con densidad mayor que `sistema` (con margen, para no acoplar el test a un valor puntual).

**Es un cambio transversal.** Ambos perfiles bajan de densidad, así que **todo lo que se mide "sobre
cotizantes" o depende del tamaño de la población activa cambia** en los seis proyectos: tasas
anualizadas, AUC (la población cesante crece y es más distinta de la empleada, lo que en general
**ayuda** a discriminar), tasas base de los targets, etc. Se corrió la suite completa
(`python -m pytest afp_lab -q`, **155 tests, 0 fallos, ~9 min 46 s**) y las corridas de referencia de
02, 03, 04, 05 y 06 con `--perfil sistema` (o el perfil que corresponda) antes/después; el detalle por
proyecto está en el informe de la ronda 3 al gerente y, para 05 (que además cambia de escala en el
objetivo 2), en `afp_lab/riesgo_empleador/README.md` §"Ronda 3". Resumen de lo más material:

* **02 (churn, sistema)**: AUC 0,698 → 0,754; captura al 10 % 35,8 % → 39,0 %; y el modelo pasa a
  ganarle a la regla en **todos** los segmentos grandes (antes perdía en 60+, 0,625 vs 0,735).
* **03 (APV, sistema)**: AUC 0,826 → 0,843; captura al 10 % 64,3 % → 68,5 %; los indicadores
  calibrados contra la Ficha (12-18 % con algún aporte, 5-8 % con aporte mensual) se mantienen dentro
  de rango (14,3 % y 6,4 %).
* **04 (cese, sistema, default del proyecto)**: AUC cese 3 m 0,868 → 0,888, cese 6 m 0,859 → 0,877
  (mejoran); **AUC reingreso 0,720 → 0,605**, cruza por debajo del umbral del brief (≥ 0,70) — ver
  limitación abajo, no se tocó ningún umbral de test ni de modelo.
* **05 (empleador, cuprum, ver su propio README)**: mejora en casi todo (AUC, segmento GRANDE,
  volatilidad del agregado), con el objetivo 2 (más empleadores) corriendo en paralelo.
* **06 (segmentación, cuprum, default del proyecto)**: cambio menor — silueta 0,248 → 0,228 (sigue
  ≥ 0,20), ARI de remuestreo 0,956 → 0,979, ARI temporal 0,456 → 0,480, separación de negocio sigue
  3/3 con razón ≥ 2 (cese 2,68→3,27, traspaso 2,72→2,04, APV 26,2→31,7 en la razón máx/mín). No
  requiere cambios en su README.
* Ningún test de la suite falló por un umbral de **modelo**; se ajustaron dos umbrales de
  **plausibilidad del generador** en `afp_lab/churn_traspaso/tests/test_churn_traspaso.py` (el piso de
  la tasa móvil de 12 meses de 0,020 a 0,015, y el techo de `auc_mejor_feature_suelta` de 0,72 a 0,76),
  ambos documentados inline con el número medido y por qué cambia con la nueva densidad.

**Limitación / punto para el gerente.** El AUC de reingreso del proyecto 04 (sistema) queda por
debajo de su umbral de aceptación (0,605 < 0,70) tras la recalibración: con menos densidad la
población de "no cotizantes" (que antes era ~21 % de los dependientes y ahora es ~53 %) se vuelve
mucho más heterogénea — mezcla cesantes de corto plazo que reingresan pronto con gente
estructuralmente fuera del mercado — y la tasa base de reingreso baja de 66 % a 23 %. Esto no se
corrigió (tocar el proyecto 04 no es parte de este brief) y se deja explícito para que el responsable
de 04 lo revise en la próxima ronda: `common/synth_base.py` no distingue "cesante reciente" de
"cesante crónico" en su generación, y esa distinción es justamente lo que el target de reingreso
necesita para seguir siendo predecible a esta densidad más realista.

## Historial de `PERFILES`

* **Ronda 2** (`generar_universo` gana `afiliados_por_empleador` configurable): sin cambios de
  densidad.
* **Ronda 3** (este cambio): ver tabla arriba. Antes de la ronda 3, `sistema` usaba
  `p_activo_inicial=0,78, factor_contratacion=1,0` y `cuprum` usaba
  `p_activo_inicial=0,68, factor_contratacion=0,60`.

## Propuestas para próximas rondas

1. Separar "cesante reciente" de "cesante crónico" en la simulación (o exponer un parámetro de
   duración media del cese) para que el proyecto 04 pueda recalibrar su target de reingreso a esta
   densidad más realista sin inventar una población nueva.
2. Si otro proyecto necesita una densidad de cotización distinta por *sub-perfil* (p. ej. sólo
   dependientes, o un tramo de edad), conviene exponer `factor_contratacion`/`p_activo_inicial` por
   segmento en vez de un único valor por perfil de cartera.
