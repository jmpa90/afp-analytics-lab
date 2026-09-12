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

## Fuentes (los sitios están bloqueados para lectura directa desde la sandbox; se consultaron por buscador)
- SP · Compendio de Normas: Libro I (Archivo Previsional; Registro de Afiliados; Traspaso electrónico entre
  AFP; Anexos de transmisión de archivos), Libro II (Cotizaciones), Libro V (Transferencia de información;
  Informes diarios). https://www.spensiones.cl/portal/compendio/596/
- SP · TEA, descripción de archivos: https://www.spensiones.cl/apps/tea/indexTEA.php y
  https://www.spensiones.cl/appsSP/tea/descripArchivosTEA.php
- Previred · Formato de archivo de nómina (largo fijo por posición, versión reforma 2025) y Tablas de
  equivalencia (septiembre 2025): https://www.previred.com (sección Formatos de archivos)
- AFP Modelo · Códigos de movimiento (planilla DNP) y AFP Habitat · Movimiento de personal (portal empleadores)
