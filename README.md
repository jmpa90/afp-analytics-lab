# afp-analytics-lab

Sandbox de modelos analíticos para una AFP, construidos con **datos sintéticos** que reproducen las dinámicas
del sistema previsional chileno. Cada proyecto responde una pregunta de negocio concreta (proyectar, segmentar,
priorizar, detectar) y deja código reproducible, tests y un README con resultados y supuestos.

## Cómo trabajamos

Equipo: un gerente de Data Analytics (define el portafolio, escribe los briefs, revisa y consolida) y tres
data scientists que trabajan en paralelo, uno por proyecto. El ciclo de cada ronda:

1. **Brief** (`afp_lab/<proyecto>/BRIEF.md`): pregunta de negocio, datos, target, features, evaluación,
   entregables y criterios de aceptación. Lo escribe el gerente.
2. **Implementación**: el data scientist construye datos sintéticos propios sobre el universo común,
   features "as-of", modelo, baselines, evaluación temporal y README de resultados.
3. **Revisión**: el gerente revisa leakage, dominio, código y métricas; las observaciones quedan en
   `docs/revisiones/` y el data scientist las corrige.
4. **Cierre**: se actualiza el tablero de abajo y se consolida lo aprendido en `CONVENCIONES.md`.

Estándares obligatorios en `CONVENCIONES.md`.

## Portafolio

| # | Proyecto | Pregunta de negocio | Tipo | Estado | Stack |
|---|---|---|---|---|---|
| 01 | `rim_nowcasting` | ¿Cuál es la renta imponible de cada afiliado en los 2 meses que aún no conocemos? | Nowcasting (hurdle GBT) | Cerrado ronda 1 | PySpark |
| 02 | `churn_traspaso` | ¿Qué afiliados se traspasarán a otra AFP en los próximos 6 meses? | Clasificación | Cerrado ronda 1 (aprobado) | pandas + LightGBM |
| 03 | `propension_apv` | ¿Quién abrirá o aumentará ahorro previsional voluntario y cuánto? | Clasificación + monto | Cerrado ronda 1 (aprobado) | pandas + LightGBM |
| 04 | `riesgo_cese_cotizacion` | ¿Quién dejará de cotizar en 3 y 6 meses, y quién volverá a cotizar? | Clasificación calibrada | Cerrado ronda 1 (aprobado) | pandas + LightGBM |
| 05 | `riesgo_empleador` | ¿Qué empleadores declararán y no pagarán, o pagarán tarde? | Clasificación (empleador) | Backlog | |
| 06 | `segmentacion_afiliados` | ¿Qué segmentos de comportamiento existen y cómo se describen? | Clustering | Backlog | |
| 07 | `proyeccion_pension` | ¿Qué pensión y tasa de reemplazo proyecta cada afiliado bajo escenarios? | Simulación + ML | Backlog | |
| 08 | `anomalias_planillas` | ¿Qué planillas tienen errores o patrones anómalos? | Detección de anomalías | Backlog | |

## Resultados de la ronda 1

Tres entregas, tres devoluciones con revisión adversarial, tres reingresos aprobados. Los números son del
test de cada proyecto con 20.000 afiliados (perfil `sistema`); detalle y limitaciones en cada README y en
`docs/revisiones/`.

| Proyecto | Métrica principal | Modelo | Mejor baseline | Métrica de negocio | Hallazgo clave de la revisión |
|---|---|---|---|---|---|
| 02 Traspaso | AUC (5 semillas) | 0,72 (0,71–0,75) | 0,60 | Contactando al 10 % se captura el 35 % de los traspasos (30–37 %), 1,8× la mejor regla | El bloque digital de la simulación "regalaba" el target (AUC 0,81 falso); corregido, el problema es legítimamente difícil |
| 03 APV | AUC (12 cortes) | 0,836 | 0,71 | Contactando al 10 % se captura el 68 % del ahorro de adquisición vs 49 % ordenando por renta; sesgo del monto esperado +1 % | Validación contaminada (sin gap) inflaba la calibración un 40 % en el decil que se contacta |
| 04 Cese | AUC cese 3 m / 6 m / reingreso | 0,87 / 0,86 / 0,72 | 0,80 / 0,80 / 0,58 | Error del agregado mensual dentro de ±10 % en 9 de 9 meses | La tabla de giros de cesantía filtraba el RIM futuro (leakage); cobertura recalculada desde el término AFC |

Lecciones incorporadas a `CONVENCIONES.md`: convención de fechas (`periodo_corte` = mes de decisión, cada
fuente con su desfase), gap de h meses entre **todas** las particiones, hiperparámetros sólo con
validación, chequeos de determinismo del simulador, y umbrales para el banco sintético distintos de los de
datos reales cuando la simulación corregida lo justifica.

## Persona: somos AFP Cuprum

Para que los proyectos tengan un contexto real, el laboratorio se pone en el lugar de **AFP Cuprum**:
la AFP más pequeña en afiliados (~5 %) pero cuarta en fondos (~17 %), con la cartera de mayores ingresos
del sistema (ingreso imponible promedio ~$2,08 M vs ~$1,22 M), líder en APV (29 % del mercado) y una
comisión de 1,44 % frente al 0,46 % de Uno. Historia, competencia, reforma de pensiones (Ley 21.735) y
la calibración del universo sintético están en `docs/contexto_afp_cuprum.md`. Desde la ronda 2 los
proyectos usan `generar_universo(perfil="cuprum")`.

Normativa que regula el contenido y los formatos de las tablas de una AFP (Compendio de Normas, TEA de la
SP, formatos de Previred): `docs/normativa_tablas_afp.md`. Cifras oficiales por AFP: `data/sp/`.

## Universo sintético común

`afp_lab/common/synth_base.py::generar_universo` genera afiliados, empleadores, un panel mensual
afiliado × mes (estado laboral, empleador, sueldo, RIM, licencias, eventos AFC) y la tabla macro
(UF, IMM, tope imponible, IPC, desempleo). Todos los proyectos parten de ahí y agregan sus propias
tablas, para que un afiliado sea "la misma persona" en todos los modelos. Las columnas `lat_*` y los
parámetros de pago del empleador son latentes del generador y **no pueden usarse como features**.

## Cómo correr

```bash
pip install -e .                       # pandas, LightGBM, scikit-learn, PySpark (solo el proyecto 01)
python -m afp_lab.<proyecto>.run       # demo end-to-end de un proyecto, salidas en salidas/<proyecto>/
python -m pytest afp_lab/<proyecto>/tests -q
```

## Estructura

```
afp_lab/
  common/        universo sintético y métricas compartidas
  <proyecto>/    BRIEF.md (gerente), README.md (resultados), synthetic.py, features.py, model.py, run.py, tests/
docs/revisiones/ revisiones del gerente por ronda
```
