# Brief 06 · Segmentación de afiliados

**Responsable**: DS-3. **Ronda**: 2. **Gerente**: revisa contra `CONVENCIONES.md` §7.

## Pregunta de negocio

¿Qué segmentos de comportamiento existen en la cartera de Cuprum y qué acción corresponde a cada uno
(asesoría, retención, APV, producto, cobranza)? Necesitamos segmentos **estables en el tiempo**,
**interpretables** y que **discriminen** resultados de negocio (traspaso, aporte APV, cese de cotización).

## Datos

`generar_universo(perfil="cuprum")` más, para no duplicar mecanismos, las tablas propias de los
proyectos 02 y 03 importadas desde sus módulos (`afp_lab.churn_traspaso.synthetic`,
`afp_lab.propension_apv.synthetic`), sin modificarlos: saldos, fondo, cambios de fondo, uso digital,
contactos, reclamos, aportes voluntarios, campañas. Documentar la dependencia.

## Método

1. Features as-of por afiliado en un `periodo_corte` (renta y densidad con desfase de 2 meses; operacional
   hasta el corte): renta, densidad 12/24 m, estabilidad (racha, cambios de empleador), edad y meses a
   pensión, saldo obligatorio y su relación con la renta, ahorro voluntario (stock y flujo), fondo y
   cambios de fondo, uso digital, contactos y reclamos, antigüedad en la AFP, tipo de afiliado.
2. Escalado robusto; k-means y GMM; selección de k por silueta, BIC y **estabilidad** (ARI entre
   re-muestreos y entre cortes separados 6 meses); k final entre 5 y 9; ningún segmento < 3 %.
3. Perfil de cada segmento (medianas, tamaños, ejemplos) y nombre de negocio.
4. Validación de utilidad: tasa de traspaso a 6 m, de aporte APV a 6 m y de cese a 3 m por segmento,
   calculadas desde las tablas de los proyectos 02/03 y el panel; los segmentos deben separar al menos
   dos de esas tres tasas con razón ≥ 2 entre el mayor y el menor.
5. Clasificador supervisado (LightGBM) que reproduzca el segmento desde las features, para explicar y
   para asignar afiliados nuevos; importancias y reglas simples.
6. Matriz de transición de segmentos a 6 y 12 meses.

## Entregables y aceptación

`synthetic.py` (sólo lo propio), `features.py`, `segmentar.py`, `evaluate.py`, `run.py`, `tests/`,
`README.md` con la tabla segmento → descripción → acción. Aceptación: silueta ≥ 0,20, ARI de estabilidad
≥ 0,60, segmentos ≥ 3 %, separación de negocio como en el punto 4, clasificador con exactitud ≥ 0,85;
runtime < 5 min con 20.000 afiliados; tests < 2 min.
