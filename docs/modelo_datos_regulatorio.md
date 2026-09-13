# Modelo de datos regulatorio — movimientos de cotización y afiliación

Esquema propuesto para el laboratorio, alineado con el Compendio de Normas de la Superintendencia
de Pensiones (SP) y con los formatos de Previred (ver `docs/normativa_tablas_afp.md` para el mapeo
norma → implementación ya hecho en `afp_lab/common/planilla.py` y
`afp_lab/common/tablas_equivalencia.py`, y la ronda 3 de este documento en `afp_lab/rim_nowcasting`).
No es el esquema físico de una AFP real (eso lo fija cada AFP); es el esquema de **salida** que el
Compendio y la TEA exigen poder producir mes a mes, adaptado a lo que el sandbox sintético puede
poblar.

## 1. Tabla `movimiento_cotizacion`

Grano: **rut_trabajador × empleador_id (o entidad pagadora) × período × tipo_pago × tipo_movimiento**.
Es la planilla de declaración y pago (Compendio Libro II) tal como la especifica Previred.

| Campo | Dominio | Clave | Norma / fuente | Dónde ya existe en el sandbox |
|---|---|---|---|---|
| `rut_trabajador` | entero (= `afiliado_id`) | PK (parcial) | Compendio Libro I, Archivo Previsional | `common.planilla.generar_planilla` (`rut_trabajador`); `rim_nowcasting.cotizaciones` (`afiliado_id`) |
| `empleador_id` | entero, nulo si el pagador no es un empleador | PK (parcial) | Previred, campo RUT empleador | ambos módulos |
| `periodo` | `yyyymm`, período **declarado** | PK (parcial) | Previred, "período a declarar" | ambos |
| `tipo_pago` | `01`..`06` (`tablas_equivalencia.TIPO_PAGO`) | PK (parcial) | Previred, tabla Tipo de pago | sólo `common.planilla` (línea a línea); `rim_nowcasting.cotizaciones` no desglosa conceptos, sólo la RIM total por pagador |
| `tipo_movimiento` | `DECLARACION`/`RECTIFICACION`/`PAGO_DNP` | PK (parcial) | Compendio Libro II | ambos |
| `entidad_pagadora` | `EMPLEADOR`/`SUBSIDIO`/`SII`/`AFILIADO` | — | Compendio Libro II (empleador, ISAPRE/COMPIN, SII para independientes, aporte directo de voluntarios) | sólo `rim_nowcasting` (`common.planilla` sólo modela el vínculo con empleador) |
| `cod_movimiento` | `0,1,2,3,4,7,11,12` (Tabla N° 7; `5,6,8,13` documentados como no generados) | — | Previred / Compendio, Tabla N° 7 | `common.planilla` (`codigo_movimiento_personal`, 0–12) y, desde esta ronda, `rim_nowcasting.cotizaciones` (`cod_movimiento`, mismo dominio salvo el 8) |
| `dias_trabajados` | 0–30 | — | Previred, días trabajados del período | ambos (desde esta ronda en `rim_nowcasting`) |
| `fecha_desde` / `fecha_hasta` | fecha calendario | — | Previred, fechas del movimiento de personal | ambos |
| `tipo_trabajador` | activo / activo mayor / pensionado que cotiza | — | Previred, tabla Tipo de trabajador | sólo `common.planilla` |
| `renta_imponible_afp` | CLP, topada al tope AFP del período | — | Previred, "renta imponible AFP" | ambos (`rim` en `rim_nowcasting`, ya suma tipo_pago) |
| `renta_imponible_afc` | CLP, 0 si no afecto (público / pensionado que cotiza) | — | Ley 19.728 | `common.planilla`; en `rim_nowcasting` sólo indirectamente vía `afc_eventos` |
| `cotizacion_obligatoria`, `sis`, `aporte_empleador` | CLP | — | Compendio Libro I; Ley 21.735 (aporte empleador, 3,5 % desde 2025-08) | sólo `common.planilla` (`tablas_equivalencia.TASA_*`) |
| `periodo_recepcion` | `yyyymm` ≥ `periodo` | — | Compendio Libro II (recepción anticipada/oportuna/tardía) | ambos |
| `estado_pago` | `PAGADA`/`DNP` | — | Compendio Libro II (DNP) | ambos |

**Cardinalidad**: ≈ (afiliados dependientes/pensionados activos con empleador vigente ese mes) ×
(1 a 3 líneas de `tipo_pago` en `common.planilla`, o 1 fila por pagador en `rim_nowcasting`) ×
(1 + rectificación 1–2 % + pago DNP 0,5–6 %). Con 5.000 afiliados y ~56 períodos son del orden de
120.000–180.000 filas en `rim_nowcasting.cotizaciones` (grano más agregado) y varias veces más en
`common.planilla` (grano por concepto).

**Quién lo consumiría**: `rim_nowcasting` (proyecto 01, nowcasting de RIM — ya lo consume, y desde
esta ronda también `cod_movimiento`/`dias_trabajados`), `riesgo_empleador` (proyecto 05, ya agrega
`common.planilla` a nivel empleador × período incluyendo altas/bajas/licencias por código de
movimiento en `_contenido_planilla`), y cualquier proyecto de recaudación esperada o mora patronal.

## 2. Tabla `movimiento_afiliacion`

Grano: **rut_afiliado × fecha del movimiento × tipo de movimiento**. Compendio Libro I: Registro de
Afiliados y de Transferencias de Ahorro Previsional, y Traspaso electrónico de Cuentas Personales
entre AFP (Cap. XXVII).

| Campo | Dominio | Norma / fuente | Dónde ya existe |
|---|---|---|---|
| `rut_afiliado` | entero | Archivo Previsional | `afiliados.afiliado_id` (todos los proyectos) |
| `fecha_movimiento` | fecha | Archivo de Traspasos / TEA | no existe: sólo hay un `periodo_afiliacion` puntual (primera afiliación), no un log de eventos |
| `tipo_movimiento` | `PRIMERA_AFILIACION`, `TRASPASO_ENTRADA`, `TRASPASO_SALIDA`, `CAMBIO_FONDO`, `PENSION`, `FALLECIMIENTO` | Compendio Libro I (traspasos) y Libro IV (cambio de fondo) | parcialmente: `churn_traspaso` (proyecto 02) simula la **decisión** de traspaso y el mercado de AFP de destino/adjudicataria de licitación, pero como tabla de eventos propia del proyecto, no como una tabla común reutilizable por otros proyectos |
| `afp_origen` / `afp_destino` | código AFP (`tablas_equivalencia`, tabla de códigos de AFP — no implementada aún) | Previred / TEA, tabla de códigos de AFP | no existe una tabla de códigos de AFP en `common/`; `churn_traspaso` usa nombres de AFP libres, no el código oficial |
| `fondo_origen` / `fondo_destino` | `A`..`E` | Compendio Libro IV (multifondos) | `churn_traspaso` tiene `fondo` como estado, no como movimiento con origen/destino |
| `plazo_dias` | entero | Compendio (plazos de conciliación) | no existe |

**Falta por completo** como tabla común: hoy cada proyecto que necesita "cambios de AFP" o "cambios
de fondo" los modela dentro de su propio `synthetic.py` (`churn_traspaso`), no hay un log de
movimientos de afiliación compartido en `common/`. Es la brecha más grande frente al Compendio.

**Quién lo consumiría**: `churn_traspaso` (proyecto 02, ya modela la decisión pero ganaría una
tabla de movimientos histórica y auditable en vez de un estado); un futuro proyecto de "cuenta 2 /
depósitos convenidos" o de proyección de saldo necesitaría los traspasos para reconstruir el saldo
por AFP a través del tiempo.

## 3. Tabla `cuenta_personal` (saldo)

Grano: **rut_afiliado × tipo de cuenta × período**. Compendio Libro I: cuenta obligatoria,
cuenta de ahorro voluntario, APV (régimen A/B), APVC, cuenta 2, depósitos convenidos.

| Campo | Dónde ya existe |
|---|---|
| Saldo cuenta obligatoria | no existe como tabla propia; se podría derivar acumulando `cotizacion_obligatoria` de `movimiento_cotizacion`, pero ningún proyecto lo hace hoy (todos trabajan con flujos mensuales, no con saldo acumulado) |
| Aportes APV/APVC, saldo, régimen (A/B), institución | `propension_apv` (proyecto 03) los modela en su propio `synthetic.py` (`_generar_apvc`, `_construir_saldos`), no en `common/` |
| Depósitos convenidos, cuenta 2 | no existe en ningún proyecto |

**Quién lo consumiría**: `propension_apv` ganaría reusar una tabla común de saldos en vez de
reconstruirla; un proyecto de proyección de pensión (mencionado como backlog en el README de 01)
la necesitaría íntegra.

## 4. Qué ya está resuelto vs. qué falta (resumen)

| Bloque del Compendio | Estado en el sandbox |
|---|---|
| Libro II, planilla de declaración y pago | **Resuelto**: `common.planilla` (grano línea/concepto) + `rim_nowcasting.cotizaciones` (grano pagador, con `cod_movimiento` desde esta ronda). Falta unificar ambos en una sola tabla `common/` — hoy son dos implementaciones independientes con grano distinto (ver "Propuestas para common" en el README de este proyecto). |
| Libro II, rezagos | **Resuelto**: `common.planilla.construir_rezagos`. |
| Libro I, Archivo Previsional (maestro de afiliados) | **Resuelto**: `common.synth_base` / `rim_nowcasting.afiliados`. |
| Libro I, Traspaso electrónico de cuentas entre AFP | **No existe como tabla común**; `churn_traspaso` sólo modela la decisión, no el archivo de traspaso (plazos, conciliación, código de AFP origen/destino). |
| Libro IV, cambio de fondo / multifondos | **Parcial**: `churn_traspaso` tiene `fondo` como atributo, no un log de movimientos; los fondos generacionales (vigentes desde abril 2027, Ley 21.735) no están modelados en ningún proyecto. |
| Libro I, cuentas voluntarias / APV / cuenta 2 | **Parcial**: sólo APV/APVC (`propension_apv`), no cuenta 2 ni depósitos convenidos. |
| Libro III, beneficios (pensiones, SCOMP) | No modelado en ningún proyecto del sandbox. |
| Libro V, informes a la SP (TEA) | No modelado; sería la capa de "qué se envía", construible sobre las tablas anteriores. |

## 5. Fuentes
Ver `docs/normativa_tablas_afp.md` (Compendio de Normas SP, TEA, Previred — los sitios están
bloqueados para lectura directa desde la sandbox, consultados por buscador) y
`docs/contexto_afp_cuprum.md` §3 (Ley 21.735, reforma de pensiones) para el detalle de cada fuente.
