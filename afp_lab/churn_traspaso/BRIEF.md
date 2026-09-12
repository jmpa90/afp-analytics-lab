# Brief 02 · Propensión a traspaso a otra AFP

**Responsable**: DS-1. **Ronda**: 1. **Gerente**: revisa contra `CONVENCIONES.md` §7.

## Pregunta de negocio

Retención: ¿qué afiliados se traspasarán a otra AFP en los próximos 6 meses? El equipo comercial puede
contactar al 10 % de la cartera al mes; queremos que ese 10 % concentre la mayor cantidad posible de
traspasos y de saldo administrado en riesgo. Segunda pregunta, menor prioridad: ¿cuál es el saldo en
riesgo esperado por afiliado (probabilidad × saldo)?

## Datos

Partir de `generar_universo` (afiliados, empleadores, panel mensual, macro). Agregar tablas sintéticas:

| Tabla | Grano | Contenido mínimo |
|---|---|---|
| `saldos` | afiliado × mes | saldo cuenta obligatoria (acumular 10 % de la RIM del panel con rentabilidad mensual del fondo elegido, más un saldo inicial coherente con edad y renta), fondo (A–E) |
| `cambios_fondo` | evento | afiliado, mes, fondo origen y destino |
| `traspasos` | evento | afiliado, mes de solicitud, mes efectivo (solicitud + 1), AFP destino (5 competidoras). **Es el target.** También traspasos de entrada para poblar afiliados nuevos. |
| `contactos` | evento | afiliado, mes, canal (ejecutivo propio, vendedor competencia, call center, campaña digital), resultado |
| `reclamos` | evento | afiliado, mes, tipo (rentabilidad, atención, comisión, pensión), estado |
| `uso_digital` | afiliado × mes | logins app/web, consultas de saldo, descarga de certificados (el "certificado de saldo" y la "cartola" se piden antes de traspasarse) |
| `mercado` | mes × AFP | comisión de cada AFP (la propia y las 5 competidoras), rentabilidad 12 m por fondo de cada AFP, cuál ganó la licitación de nuevos afiliados |

Mecanismo del target (documentar parámetros): hazard mensual de traspaso que aumente con comisión propia
relativa a la más barata, rentabilidad propia 12 m bajo el promedio del sistema, contacto de vendedor de la
competencia en los últimos 3 meses, reclamo reciente, edad joven, renta alta, antigüedad baja en la AFP,
cambios de fondo recientes (afiliado "activo"); que disminuya con contacto de ejecutivo propio, antigüedad
alta y estado cesante (un cesante rara vez se traspasa). Tasa anual de traspaso objetivo: 4–7 % de la
cartera cotizante, con estacionalidad leve (más en marzo y agosto por campañas). Ruido no explicable
suficiente para que el AUC en test quede en 0,75–0,90.

## Target y población

* Fila = afiliado × `periodo_corte`, con cortes mensuales móviles.
* Población: afiliados vigentes en la AFP al corte (no traspasados antes, no retirados, afiliación ≤ corte).
* `y = 1` si existe un traspaso con mes de solicitud en `corte+1 … corte+6`.
* Excluir del entrenamiento los cortes cuyos 6 meses no están completos.
* Información conocida al corte: panel laboral hasta `corte` (recordar que la RIM real llega con 2 meses de
  desfase: usar `corte-2` como último mes de RIM conocida y `corte` para eventos operacionales como
  contactos, reclamos, uso digital y cambios de fondo, que la AFP sí conoce al día).

## Features (mínimo)

Historia laboral y renta (as-of, con desfase de 2 meses): RIM de referencia, densidad 12 m, meses desde
último cambio de empleador, rubro, tipo de afiliado. Cuenta: saldo, fondo, cambios de fondo 3/12 m,
antigüedad en la AFP, edad, sexo, región, meses hasta edad legal de pensión. Relación: contactos por
canal 1/3/6 m, reclamos 6/12 m y estado, logins y descargas de certificado 1/3 m. Mercado: brecha de
comisión, brecha de rentabilidad 12 m del fondo del afiliado, AFP ganadora de la licitación, mes del
año.

## Modelo y evaluación

* Baselines: tasa base; regla "reclamo o contacto de competencia en 3 m, o brecha de rentabilidad < -1 pp".
* LightGBM binario; calibración si se usará para saldo en riesgo esperado.
* Split temporal por corte con gap de 6 meses; test en los últimos 3 cortes evaluables.
* Métricas `common/metrics.metricas_clasificacion` + por segmento (tramo de saldo, edad, tipo de afiliado,
  fondo) + **métrica de negocio**: % de traspasos y % de saldo en riesgo capturados contactando al 10 % y
  al 20 % de mayor score.
* Importancias e interpretación de 5 casos.

## Entregables y aceptación

`synthetic.py`, `features.py`, `model.py`, `evaluate.py`, `run.py`, `tests/`, `README.md` según
`CONVENCIONES.md`. Aceptación: checklist §7; AUC test ≥ 0,75 y lift decil 1 ≥ 3 sobre la tasa base; el
modelo supera a la regla de negocio en captura al 10 %; runtime < 5 min con 20.000 afiliados; tests < 2 min.
