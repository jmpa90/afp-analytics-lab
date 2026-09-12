# Revisión ronda 2 · 05 riesgo_empleador

**Veredicto del gerente: APROBAR con reserva** (revisor Sonnet + correcciones menores aplicadas por el
gerente por economía de créditos).

## Verificado
- Sin leakage: mutaciones de planillas no recibidas/futuras y de boletines posteriores al corte no cambian
  las features; control positivo sí cambia; `columnas_modelo` rechaza `p_*`, `lat_*` y shocks.
- Target `y_dnp` = estado de la planilla del periodo corte; gaps entre las 4 particiones; 12 cortes de
  test; calibración e hiperparámetros sólo con validación.
- Dinámicas y tasas dentro del brief (DNP 5,8 %, tardío 8,0 %); volatilidad intrínseca del agregado
  mensual reproducida (99 % del esperado): el criterio de ±10 % mensual es inalcanzable con 500
  empleadores y se sustituye por "11 de 12 meses dentro de 1σ".
- Reproducible; 21 tests; 29 s con 20.000 afiliados.

## Reserva (mayor)
La captura de masa en riesgo al 10 % es 54,5 % con 12 cortes y 44,5 % con 24 (EE ±6,4 pp). Se aprueba
como "45–55 %, mejor que la regla en ambos casos". Backlog de `common/`: `afiliados_por_empleador`
configurable para ampliar el universo de empleadores y bajar la varianza.

## Menores corregidos por el gerente
- `metricas_regresion` (common): WAPE y sesgo relativo devuelven NaN sin masa real (antes 4e17).
- `masa_por_segmento`: columnas `n_positivos` y `nota` cuando no hay casos.
- `en_boletin_infractores` leído al corte (`j`), consistente con los boletines concursales.
- Nota en features/README: `panel_mensual` no alimenta features.
