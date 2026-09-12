# ¿Regula la SP cómo deben ser las tablas de movimientos de una AFP?

Respuesta corta: **la Superintendencia de Pensiones (SP) no dicta el diseño interno de las bases de datos
de una AFP (ids, nombres de tablas), pero sí fija el contenido, los códigos y los formatos de intercambio**
con tanto detalle que en la práctica el modelo de datos de cualquier AFP se construye alrededor de ellos.
Hay tres capas, todas públicas:

## 1. Compendio de Normas del Sistema de Pensiones (spensiones.cl → Compendio)

Es la norma administrativa consolidada. Lo relevante para tablas de movimientos:

| Libro / capítulo | Qué define |
|---|---|
| Libro I · Afiliación y cuentas personales: **Archivo Previsional**, **Registro de Afiliados y de Transferencias de Ahorro Previsional y de Cotizaciones de Afiliados Voluntarios**, **Traspaso electrónico de las Cuentas Personales entre AFP** (Cap. XXVII), **Anexos: Especificaciones para la transmisión de archivos** | Qué registros debe mantener la AFP por afiliado y cuenta (obligatoria, voluntaria, APV, cuenta 2, depósitos convenidos), qué campos tiene el archivo previsional, cómo se transfiere una cuenta entre AFP (archivos, plazos, conciliaciones) |
| Libro II · Cotizaciones previsionales | Planilla de declaración y pago, declaración y no pago (DNP), cobranza, **rezagos** (cotizaciones no imputables), reliquidaciones, pagos en exceso, cotizaciones de independientes vía SII/TGR |
| Libro III · Beneficios | Registros de pensiones, SCOMP, bonos |
| Libro IV · Fondos de pensiones | Multifondos / fondos generacionales, cambios de fondo (movimientos entre fondos), valor cuota |
| Libro V · Aspectos administrativos y operacionales: **Transferencia de Información**, **Informes diarios**, contabilidad | Qué archivos manda la AFP a la SP y en qué formato |

Cada capítulo trae anexos con **layouts de registro de largo fijo** (nombre del campo, posición, largo,
tipo, valores permitidos). Esos anexos son, de facto, el diccionario de datos del sistema.

## 2. TEA: Transferencia Electrónica de Archivos (spensiones.cl/apps/tea)

La SP publica la **descripción técnica de cada archivo** que las AFP, la AFC, el IPS y el DCV intercambian
con ella: nombre válido del archivo, versión, registros de largo fijo, campos, y la norma que lo origina.
Incluye, entre otros, la **Base de Datos de Afiliados, Cotizantes, Beneficiarios, Pensionados y
Fallecidos**, los saldos de cuentas personales, el archivo de afiliados, el **historial de movimientos**,
trabajos pesados y pensiones. Cada cambio de estructura se comunica por Oficio y se publica una nueva
versión. Esto es lo más cercano a "cómo deben ser las tablas": no el esquema interno, pero sí el esquema
de salida que la AFP debe poder producir mes a mes.

## 3. Formatos de Previred (previred.com → Formatos de archivos / Tablas de equivalencia)

La entrada de casi todas las cotizaciones es la planilla electrónica de Previred, cuyo formato (largo fijo
por posición o largo variable por separador, ~105 campos, actualizado con la reforma en 2025) y sus
**tablas de equivalencia** son públicos. Son los códigos que aparecen en cualquier tabla de movimientos:

| Tabla | Ejemplos de códigos |
|---|---|
| Tipo de pago | 01 remuneraciones del mes, 02 gratificación, 03 bono Ley 20.058, 04 diferencias, 05 subsidio, 06 aguinaldo/bono (no imponible AFP), … |
| Régimen previsional | AFP, IPS (antiguo), SIP |
| Tipo de trabajador | activo, pensionado que cotiza, pensionado que no cotiza, activo mayor de 65/60 |
| **Movimiento de personal** (Tabla N° 7) | 0 sin movimiento; 1 contratación a plazo indefinido; 2 retiro; 3 subsidios (licencia médica); 4 permiso sin goce de sueldo; 5 incorporación en el lugar de trabajo; 6 accidente del trabajo; 7 contratación a plazo fijo; 8 cambio de contrato de plazo fijo a indefinido; 11 otros movimientos (ausentismo); 12 reliquidación, premio, bono; 13 suspensión por acto de autoridad (Ley 21.227) |
| Códigos de AFP, ISAPRE, CCAF, mutual, fondo (A–E), forma de pago APV/APVC, tipo de contrato AFC | tablas numeradas en el mismo documento |

El registro de la planilla trae, por trabajador y periodo: RUT y DV, tipo de pago, periodo desde/hasta,
régimen, tipo de trabajador, días trabajados, tipo de línea, **código de movimiento de personal con fecha
desde/hasta**, tramo de asignación familiar, AFP, **renta imponible AFP**, cotización obligatoria,
SIS, cuenta de ahorro voluntario, renta imponible sustitutiva, tasa de pactada, aporte del empleador (reforma),
APVI/APVC (institución, régimen, monto), AFC (renta imponible, aportes), salud (ISAPRE/FONASA, renta
imponible, plan en UF), CCAF, mutual, etc.

## Qué hacer con esto en el laboratorio

1. **Alinear la tabla `cotizaciones` sintética con la planilla Previred**: agregar `tipo_pago`,
   `codigo_movimiento_personal` (0–13) con fechas, `dias_trabajados`, `tipo_trabajador`, y separar
   renta imponible AFP / AFC / salud. Los proyectos 01 y 04 ganan una señal real: el código 2 (retiro),
   3 (subsidio), 4 (permiso sin goce) y 7/8 (tipo de contrato) llegan **en la propia planilla**, es decir
   con el mismo desfase que la RIM, y el 1/5 marca ingresos.
2. **Tabla de rezagos** (Libro II): cotizaciones recibidas que no se pudieron imputar al afiliado; hoy no
   se modela y el proyecto 01 la nombra como "rezago" con otro sentido.
3. **Modelo de datos regulatorio como proyecto de backlog** (`05b_modelo_datos_regulatorio`): un esquema
   de tablas del laboratorio que reproduzca los layouts de TEA y Previred (nombres y códigos oficiales),
   para que los datos sintéticos "parezcan" los de una AFP real y los modelos se puedan portar sin
   renombrar columnas.

### Implementado (ronda 2, DS-3): `afp_lab/common/planilla.py` — mapeo columna → norma

El punto 1 y el punto 3 de arriba ya no son backlog: `afp_lab/common/planilla.py` genera, a partir
de `synth_base.generar_universo` (sin modificarlo — es aditivo), una tabla con el grano y los
códigos de la planilla Previred real. `afp_lab/common/tablas_equivalencia.py` trae los diccionarios
de códigos y glosas (fuente: Previred, tablas de equivalencia septiembre 2025). Mapeo de cada
columna de `generar_planilla(...)` a la norma que la origina:

| Columna | Norma / fuente | Nota de la implementación |
|---|---|---|
| `rut_trabajador` | Planilla Previred, campo RUT (Compendio Libro I, Archivo Previsional) | = `afiliado_id` del universo |
| `empleador_id` | Planilla Previred, campo empleador/RUT empleador | tomado de `panel_mensual.empleador_id` / `empleador2_id` (segundo empleador, multiempleo) |
| `periodo` | Planilla Previred, "período a declarar" | mes que se está declarando (no el de recepción) |
| `tipo_pago` | Previred, tabla "Tipo de pago": 01 remuneraciones, 02 gratificación, 05 subsidio, 06 aguinaldo | sólo se generan 01/02/05/06 (03 asignación familiar y 04 diferencias no tienen regla propia en el brief) |
| `dias_trabajados` | Previred, campo días trabajados del período | reducido en licencia (3), permiso sin goce (4) y ausentismo (11) |
| `codigo_movimiento_personal` | Compendio / Previred, **Tabla N° 7** (movimiento de personal) | 0,1,2,3,4,7,8,11,12 generados; 5,6,13 documentados como no generados (`tablas_equivalencia.CODIGOS_NO_GENERADOS`) — ver más abajo |
| `fecha_desde` / `fecha_hasta` | Previred, fechas del movimiento de personal | primer/último día calendario de `periodo` |
| `tipo_trabajador` | Previred, tabla "Tipo de trabajador" (activo, activo mayor, pensionado que cotiza/no cotiza) | derivado de `tipo_afiliado` + edad/sexo, no toca `synth_base` |
| `tipo_contrato_planilla` | Compendio, tipo de contrato (plazo fijo/indefinido) | copia de `panel_mensual.tipo_contrato` que sí refleja el cambio de contrato (código 8); columna propia para no tocar el panel (aditivo) |
| `renta_imponible_afp` | Previred, "renta imponible AFP" (topada al tope imponible AFP) | ya viene topada desde `rim_nowcasting.synthetic` (`np.minimum(..., tope)`); planilla.py sólo reparte esa cifra entre las líneas 01/02/05 |
| `renta_imponible_afc` | Ley 19.728 (Seguro de Cesantía) — excluye funcionarios públicos y pensionados que cotizan | misma regla que `rim_nowcasting.generar_afc_eventos`: excluye empleadores con `rubro == "O_ADMIN_PUBLICA"` y `tipo_trabajador == "PENSIONADO_COTIZA"` |
| `cotizacion_obligatoria` | Compendio Libro I, 10 % de la renta imponible AFP (cuenta individual) | `tablas_equivalencia.TASA_COTIZACION_OBLIGATORIA` |
| `sis` | Compendio, Seguro de Invalidez y Sobrevivencia (cargo empleador) | `tablas_equivalencia.TASA_SIS` |
| `aporte_empleador` | **Ley 21.735** (reforma de pensiones, mar-2025): cotización adicional del empleador | `tablas_equivalencia.tasa_aporte_empleador`: 3,5 % desde 2025-08 (único escalón dentro del horizonte simulado; cronograma completo hasta 8,5 % no confirmado en las fuentes consultadas, documentado como supuesto) |
| `periodo_recepcion` | Compendio Libro II, planilla de declaración y pago (anticipado/oportuno/tardío) | mismas probabilidades por tamaño de empleador que el proyecto 01 (`empleadores_latente.p_pago_*`) |
| `estado_pago` | Compendio Libro II, **DNP** (declarada y no pagada) | `empleadores_latente.p_dnp` |
| `tipo_movimiento` | Compendio Libro II: declaración original, rectificación, pago de una DNP | `RECTIFICACION` usa además el código 12 (reliquidación) de la Tabla N° 7 |

Códigos de la Tabla N° 7 **no generados** (`CODIGOS_NO_GENERADOS = (5, 6, 13)`), documentado, no
olvidado: 5 (incorporación en el lugar de trabajo) y 6 (accidente del trabajo) no tienen una regla
propia en `BRIEF_PLANILLA.md`; 13 (Ley 21.227, suspensión de la relación laboral por acto de
autoridad) es un mecanismo excepcional (pandemia) fuera del alcance de esta simulación.

`construir_rezagos(planilla, seed)` implementa el punto 2 de arriba (Libro II: cotizaciones
recibidas que no se pudieron imputar al RUT del trabajador): marca ~0,7 % de las líneas
`DECLARACION`/`PAGADA` y las regulariza 2–6 meses después (`periodo_imputacion`).

Limitación documentada: sólo declaran planilla los afiliados `DEPENDIENTE` o `PENSIONADO_ACTIVO`
con empleador vigente ese mes — independientes (vía SII, Operación Renta) y voluntarios (aporte
directo) no pasan por Previred y no aparecen en esta tabla. Ver el docstring de
`afp_lab/common/planilla.py` para las demás simplificaciones (líneas 03/04 no generadas, código 8
sólo afecta una columna propia `tipo_contrato_planilla`, etc.).

## Fuentes (los sitios están bloqueados para lectura directa desde la sandbox; se consultaron por buscador)
- SP · Compendio de Normas: Libro I (Archivo Previsional; Registro de Afiliados; Traspaso electrónico entre
  AFP; Anexos de transmisión de archivos), Libro II (Cotizaciones), Libro V (Transferencia de información;
  Informes diarios). https://www.spensiones.cl/portal/compendio/596/
- SP · TEA, descripción de archivos: https://www.spensiones.cl/apps/tea/indexTEA.php y
  https://www.spensiones.cl/appsSP/tea/descripArchivosTEA.php
- Previred · Formato de archivo de nómina (largo fijo por posición, versión reforma 2025) y Tablas de
  equivalencia (septiembre 2025): https://www.previred.com (sección Formatos de archivos)
- AFP Modelo · Códigos de movimiento (planilla DNP) y AFP Habitat · Movimiento de personal (portal empleadores)
