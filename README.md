# afp-analytics-lab

Sandbox de modelos analíticos para una AFP, construidos con **datos sintéticos** que reproducen las dinámicas
del sistema previsional chileno. Cada proyecto responde una pregunta de negocio concreta (proyectar, segmentar,
priorizar, detectar) y deja código reproducible, tests y un README con resultados y supuestos.

## Informe ejecutivo

`docs/reporte_ejecutivo.html` es el entregable para el negocio, en formato app de un solo archivo (pantallas
Inicio, Casos, Torre de control, Roadmap, Datos y límites, Anexo y Glosario; ruteo por hash, filtros por área).
`docs/reporte_ejecutivo_documento.html` es la misma información como documento imprimible. Ambos se
regeneran con `python3 docs/build_reporte.py` (y `--documento`) a partir de `docs/reporte_datos.json`, donde
cada cifra lleva su fuente.
Revisión de exactitud en `docs/revisiones/ronda_4/reporte_ejecutivo.md`.

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
| 02 | `churn_traspaso` | ¿Qué afiliados se traspasarán a otra AFP en los próximos 6 meses? | Clasificación | Cerrado ronda 2 (Cuprum + licitación de stock) | pandas + LightGBM |
| 03 | `propension_apv` | ¿Quién abrirá o aumentará ahorro previsional voluntario y cuánto? | Clasificación + monto | Cerrado ronda 2 (Cuprum) | pandas + LightGBM |
| 04 | `riesgo_cese_cotizacion` | ¿Quién dejará de cotizar en 3 y 6 meses, y quién volverá a cotizar? | Clasificación calibrada | Cerrado ronda 2; corrección ronda 3 (reingreso, ver abajo) | pandas + LightGBM |
| 05 | `riesgo_empleador` | ¿Qué empleadores declararán y no pagarán, o pagarán tarde? | Clasificación (empleador) | Cerrado ronda 3 (más empleadores, ver abajo) | pandas + LightGBM |
| 06 | `segmentacion_afiliados` | ¿Qué segmentos de comportamiento existen y cómo se describen? | Clustering | Cerrado ronda 2 (4 segmentos) | pandas + scikit-learn |
| — | `common/planilla` | Planilla Previred sintética con códigos oficiales (modelo de datos regulatorio) | Datos | Cerrado ronda 2 | pandas |
| 07 | `uplift_retencion` | ¿A quién conviene contactar para que el contacto de verdad evite el traspaso? | Inferencia causal / uplift (T-, X-, DR-learner) | Cerrado ronda 4 | pandas + LightGBM |
| 08 | `secuencial_afiliado` | ¿Una red secuencial multi-tarea supera a los LightGBM de 02 y 04? | Red neuronal (GRU, PyTorch CPU) | Cerrado ronda 4 (no aporta, ver abajo) | PyTorch + LightGBM |
| 09 | `bandit_contacto` | ¿Qué política de contacto maximiza la retención neta de costo? | Bandits contextuales + evaluación off-policy | Cerrado ronda 4 | pandas + LightGBM |
| 10 | `pronostico_agregado` | ¿TimesFM (modelo fundacional de series) pronostica mejor las series agregadas que ETS o LightGBM? | Series de tiempo | Brief listo; en pausa hasta contar con los pesos | statsforecast + TimesFM |
| 11 | `voz_afiliado` | ¿Cuál es el NPS de toda la cartera, qué eventos lo mueven y cuánto vale un punto? | NPS inferido + causal | Cerrado ronda 5 (aprobado con reserva) | pandas + LightGBM |
| 12 | `proyeccion_pension` | ¿Qué pensión y tasa de reemplazo proyecta cada afiliado bajo escenarios? | Simulación + ML | Backlog | |
| 13 | `anomalias_planillas` | ¿Qué planillas tienen errores o patrones anómalos? | Detección de anomalías | Backlog | |

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

> **Nota ronda 3**: la tabla de arriba usa perfil `sistema` con la densidad de cotización sin calibrar
> (~79 % de afiliados con RIM > 0 en un mes típico). La ronda 3 la bajó a ~47,3 % (Ficha 165 de la SP,
> ver `afp_lab/common/README.md`), y eso mueve las tres métricas de esta tabla (medido con una sola
> semilla, no las 5/12 cortes del protocolo original, así que estos son puntos de referencia, no un
> reemplazo de la tabla): **02** AUC 0,698 → 0,754; **03** AUC 0,826 → 0,843; **04** AUC cese 3 m/6 m
> 0,868/0,859 → 0,888/0,877, **reingreso 0,720 → 0,605** (cruza bajo el umbral de 0,70 del brief —
> pendiente de revisión por el responsable de 04). Detalle completo en el informe de la ronda 3.

## Resultados de la ronda 2

Todo con `perfil="cuprum"` (20.000 afiliados) salvo indicación; detalle en cada README (§"Ronda 2") y
en `docs/revisiones/ronda_2/`.

| Proyecto | Resultado | Lectura de negocio |
|---|---|---|
| 01 RIM (Spark) | WAPE dependientes h=1 0,115 / h=2 0,221 (sistema 0,115 / 0,182) | A un mes igual; a dos meses pesa más la remuneración variable de finanzas y minería |
| 02 Traspaso | AUC 0,71–0,75 (3 semillas), captura al 10 % 30–39 % | En Cuprum el motor de fuga es el precio y la antigüedad; riesgo y saldo van en direcciones opuestas: ordenar por prob × saldo |
| 02 Licitación de stock (2027) | 9,6 % sorteados, 6,2 % del saldo administrado en riesgo (24 mil MM) | Priorizar la retención por saldo × probabilidad duplica el saldo retenido con los mismos contactos; rendimiento marginal decreciente al ampliar capacidad |
| 03 APV | AUC 0,821, captura al 10 % 55 % vs 33 % por renta | Cartera de alta renta: la renta discrimina menos y la ventaja del modelo crece (+22 pp) |
| 04 Cese | AUC cese 3 m 0,885, reingreso 0,641 | Más estabilidad y jubilación real; los códigos de la planilla no agregan sobre los eventos AFC |
| 05 Empleador (nuevo) | AUC DNP 0,82; masa en riesgo capturada al 10 % 45–55 % | Aprobado con reserva: el agregado mensual es intrínsecamente volátil con 500 empleadores — **actualizado en la ronda 3, ver abajo** |
| 06 Segmentación (nuevo) | 4 segmentos (k-means 5 + fusión), silueta 0,24, ARI re-muestreo 0,99, ARI temporal 0,47, clasificador 0,98 | Universo Cuprum compartido con 02/03; tasas de cese, traspaso y APV difieren 3,4×, 2,1× y 31× entre segmentos; tabla segmento → acción |
| Planilla regulatoria | `common/planilla.py` con códigos Tabla N° 7, recepción por declaración, DNP por empleador con persistencia | Base para que los modelos usen los mismos códigos que la planilla real |

## Resultados de la ronda 3

Tres workstreams (`churn`, `empleador`, `datos`) más una corrección de 04. Con 20.000 afiliados salvo
indicación; briefs en `docs/briefs/ronda_3/`, revisiones en `docs/revisiones/ronda_3/`.

| Cambio | Resultado | Lectura de negocio |
|---|---|---|
| Densidad de cotización recalibrada (`common/synth_base.py`, perfiles `sistema` y `cuprum`) | Densidad de un mes típico: sistema ~79 %→**47,7 %** (Ficha 165: 47,3 %), cuprum ~72 %→**66,4 %** (Ficha 165: 66 %). Sube el AUC en 02/03/04/05 (cartera cesante más grande y distinta de la empleada) y resuelve la pérdida en segmentos grandes que tenía 02; baja el AUC de reingreso de 04 (0,72→0,61, bajo su umbral) | Los indicadores "sobre cotizantes" ahora son comparables con la Ficha de la SP; el proyecto 04 necesita revisar su target de reingreso a esta densidad más realista |
| 05 Empleador: más empleadores (`afiliados_por_empleador` 40→**20**, ~500→~1.000 empleadores) | AUC 0,818→**0,865**; volatilidad intrínseca del agregado mensual 98,9 %→**63,6 %**; captura de masa al 10 % 54,5 %→48,6 % en la semilla de referencia (51,9 % promedio de 5 semillas); el modelo deja de perder en el segmento GRANDE | La proyección de recaudación sigue siendo trimestral, no mensual, pero el ruido del agregado bajó a un tercio; la relación 1/√n_empleadores predice cuánto más haría falta escalar para cerrar el criterio de ±10 % |
| 02 Churn: la licitación de stock retroalimenta la cartera; bootstrap del AUC por segmento | Sorteos parametrizables (2027-08 y 2029-08); quienes se van salen de saldos, contactos y reclamos desde el mes efectivo, sin contaminar el target. AUC 60+ 0,696 IC [0,64–0,76]; tramo de saldo 40M+ 0,667 IC [0,64–0,70], único segmento grande confiablemente bajo 0,70 | El supuesto de fuga de 0,446 % mensual sigue explícito: la SP publica el neto por AFP y el bruto del sistema, no la fuga bruta por AFP |
| 04 Cese: corrección del reingreso | Diagnóstico: en la población de reingreso rubro, tamaño y estacionalidad eran constantes y la contratación del generador no tiene memoria; features del último empleo suben el AUC solo +0,003 a +0,008 (0,61 sistema, 0,62–0,63 cuprum), sin recuperar el 0,70 | El reingreso quedó casi impredecible por diseño del generador; ajuste propuesto (probabilidad de contratación dependiente de la duración del cese) para la próxima ronda |
| 01 RIM: códigos de movimiento Previred (Tabla N°7) como features as-of; `docs/modelo_datos_regulatorio.md` | Escala de referencia (5.000 afiliados, 12 snapshots): sistema WAPE h1 0,196→0,194, h2 0,258→0,259; cuprum h1 0,210→0,212, h2 0,304→**0,270**; mejora en el segmento con subsidio (h1 0,169→0,154); término reciente sigue con sesgo ≈ −95 % en ambas versiones | Los eventos AFC ya capturan los términos: los códigos de planilla aportan poco al nowcasting de RIM salvo en subsidios y en el segundo mes de Cuprum; una sola semilla, deltas chicos no significativos |
| Suite completa del laboratorio | **155 tests, 0 fallos, ~9 min 46 s** (`python -m pytest afp_lab -q`, incluido el proyecto 01 con Spark) | Dos umbrales de plausibilidad del generador se ajustaron en `churn_traspaso` (documentados inline); ningún umbral de modelo se tocó |

## Resultados de la ronda 4 (técnicas avanzadas)

Tres proyectos nuevos, 10.000 afiliados, perfil Cuprum; briefs en `docs/briefs/ronda_4/`, revisiones en
`docs/revisiones/ronda_4/`. El objetivo era saber si la técnica paga, no que ganara.

| Proyecto | Resultado | Lectura de negocio |
|---|---|---|
| 07 Uplift causal (nuevo) | Campaña histórica confundida + holdout aleatorio 10 %; efecto verdadero heterogéneo (persuadibles, neutros, sleeping dogs). Spearman CATE estimado vs verdadero: DR 0,195, X 0,186, T 0,152, riesgo 0,054. Con 20 % de presupuesto el X-learner evita 2,75× los traspasos del ranking por riesgo (4,5× al 10 %). Qini en holdout aleatorio positivo solo para el X-learner y con IC que cruza cero | Aprobado con reserva: contactar por uplift y no por riesgo cambia a quién se llama; la validación experimental necesita un holdout mayor que el del sandbox |
| 08 Red secuencial (nuevo) | GRU multi-tarea sobre 24 meses vs LightGBM de 02/04 en los mismos cortes: traspaso 0,585 vs 0,726; cese 0,777 vs 0,861. Con la misma información, LightGBM con lags empata con la red (0,594 / 0,778) | No aporta hoy: la brecha es de información (variables operacionales y de mercado que la secuencia no incluye), no de arquitectura |
| 09 Bandit de contacto (nuevo) | Cinco políticas, 24 meses, 400 contactos/mes, 5 semillas, recompensa rezagada 6 meses, OPE con IPS/SNIPS/DR (error < 10 %). Con la señal contextual calibrada, Thompson lineal gana a ε-greedy en 3/5 semillas; solo con señal 4× más fuerte llega a 4/5; LinUCB nunca | Aprobado: con la heterogeneidad documentable hoy, un promedio simple por brazo rinde casi igual que un bandit contextual; primero hay que medir cuánta señal traen los datos reales |

## Resultados de la ronda 5

| Proyecto | Resultado | Lectura de negocio |
|---|---|---|
| 11 Voz del afiliado (nuevo) | Encuesta sintética con sesgo de no respuesta (8 % invitados, 25 % responden). NPS verdadero de la cartera entre −9,4 y −15,0 en los cortes de test (media −11,6); NPS ingenuo (solo respondientes) erra 8,5 puntos; NPS inferido con ponderación inversa 6,8; Horvitz-Thompson directo 5,3; una regla ponderada simple 1,1. AUC de detractor 0,753. Drivers con signo correcto 6/6 (licencia mal pagada −2,6 vs −2,5 real; reclamo sin resolver −1,0 vs −2,0). Un punto de NPS ≈ 127 traspasos evitados, ≈ 3.600 MM de saldo y ≈ 858 MM de APV adicional, asociación ajustada, no experimento | Aprobado con reserva: corregir el sesgo de la encuesta vale más que el modelo; el modelo de respuesta discrimina poco (AUC 0,54) y limita la corrección; el ML sirve para detectar detractores individuales, no para el agregado |

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
`afiliados_por_empleador` (configurable desde la ronda 2) fija el número de empleadores del universo.

**Ronda 3**: la densidad de cotización de los perfiles `sistema` y `cuprum` (fracción de afiliados con
RIM > 0 en un mes típico) se recalibró contra la Ficha Estadística Previsional N° 165 de la SP;
detalle, mecanismo y números antes/después en `afp_lab/common/README.md`.

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
