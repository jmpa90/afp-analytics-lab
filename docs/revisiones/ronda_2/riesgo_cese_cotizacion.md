# Revisión ronda 2 · 04 riesgo_cese_cotizacion

**Veredicto del gerente: APROBAR con correcciones menores** (integra a `main` tras aplicarlas).

## Verificado
- Leakage cerrado por construcción: `prestaciones_cesantia` ya no se lee; cinco pruebas independientes
  (alterar panel en corte-1/corte y regenerar tablas, RIM ×7 en meses no conocidos, quitar la tabla del
  dict) dan 0 columnas cambiadas; la cobertura analítica reimplementada por fuera coincide exacto.
- Split con gap real = h (corte a corte 2h) en ambos horizontes; isotónica sólo con validación.
- Poblaciones: sin independientes ni voluntarios; contaminación de B baja de 16,6 % a 2,2 %; cobertura de
  A 95,3 % y factor 1,0496 reproducen.
- Fixes de empleador (0 NaN; corrección de censura validada contra la verdad: sesgo −5,7 % → −0,7 %),
  tipo de contrato vigente, ventana acotada, métricas por segmento para 3 predictores, error mensual.
- Tabla ronda 1 vs 2 reproduce dígito a dígito; 9 de 9 errores mensuales dentro de ±10 %.

## Correcciones exigidas antes de integrar
- **A (mayor)**: el agregado de negocio cubre el 88 % de los ceses reales (excluye independientes y
  voluntarios, que aportan 12 % con el 7,5 % de los cotizantes). Sumar el baseline rubro × mes para esos
  tipos o declarar la cobertura y publicar el factor 1,136 separado del 1,0496 (censura de recepción).
- **B (menor)**: `emp_sin_planilla_conocida` es idénticamente 0; eliminar o redefinir sobre la ventana.
- **G (menor, barato)**: apagar `cesantia_en_curso`/`giros_restantes` cuando hay INICIO AFC posterior al
  TERMINO o `rim_l1 > 0`; corregir el docstring.
- Documentación: brecha de Brier del reingreso afecta al 73 % de esa población y el AUC 0,72 lo sostiene
  el gradiente de jubilación (C, D); filtrar la "cola opuesta" por `meses_hasta_pension` (E); decir que el
  gap 2h costó un tercio de los cortes de entrenamiento a 6 meses (F); "gana en todos los rubros" sólo con
  2–3 semillas (H); runtime y mediana de `emp_crecimiento_12` con los valores medidos.
