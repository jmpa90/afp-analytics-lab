# Brief · Modelo de datos regulatorio: planilla Previred sintética (`afp_lab/common/planilla.py`)

**Responsable**: DS-3 (primera tarea de la ronda 2). Ver `docs/normativa_tablas_afp.md`.

Objetivo: que el laboratorio tenga una tabla de cotizaciones con la forma de la planilla real (Previred /
Compendio SP), para que los modelos usen los mismos códigos que verían en producción.

1. `generar_planilla(universo, seed) -> pd.DataFrame` a partir de `panel_mensual` y
   `empleadores_latente` (sólo para el comportamiento de pago), grano **afiliado × empleador × periodo**
   con columnas: `rut_trabajador` (= afiliado_id), `empleador_id`, `periodo`, `tipo_pago` (01
   remuneraciones, 02 gratificación, 05 subsidio, 06 aguinaldo no imponible AFP…), `dias_trabajados`,
   `codigo_movimiento_personal` (Tabla N° 7: 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13) con `fecha_desde` /
   `fecha_hasta`, `tipo_trabajador`, `renta_imponible_afp` (topada), `cotizacion_obligatoria`, `sis`,
   `aporte_empleador` (reforma: 3,5 % desde 2025-08, con cronograma documentado), `renta_imponible_afc`,
   `periodo_recepcion`, `estado_pago` (PAGADA / DNP), `tipo_movimiento` (DECLARACION / RECTIFICACION /
   PAGO_DNP). Reglas: código 1 o 7 en el mes de inicio según contrato, 2 en el mes de término, 3 con
   licencia, 4 permiso sin goce (probabilidad pequeña, RIM parcial), 8 al pasar de plazo fijo a
   indefinido (probabilidad anual), 11 ausentismo, 12 reliquidación (rectificaciones), 13 no se genera.
2. `tablas_equivalencia.py` con los diccionarios de códigos y glosas (fuente: Previred, sept 2025).
3. `rezagos`: 0,5–1 % de las líneas quedan sin imputar por RUT mal informado y se imputan 2–6 meses
   después (tabla `rezagos` con `periodo_imputacion`).
4. Coherencia obligatoria con `panel_mensual` (misma RIM total, mismos empleadores) y con las reglas de
   recepción del proyecto 01 (anticipado / oportuno / tardío / DNP por tamaño de empleador). Tests.
5. No modificar `synth_base.generar_universo` ni sus salidas: el módulo es aditivo. Documentar en
   `docs/normativa_tablas_afp.md` §"Qué hacer" el mapeo columna → norma.
