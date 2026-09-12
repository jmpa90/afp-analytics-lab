# Brief 04 · Riesgo de dejar de cotizar y probabilidad de reingreso

**Responsable**: DS-3. **Ronda**: 1. **Gerente**: revisa contra `CONVENCIONES.md` §7.

## Pregunta de negocio

Dos usos: (a) proyección de recaudación y de densidad de cotización: ¿cuántos cotizantes de hoy dejarán de
cotizar en 3 y 6 meses, y cuántos no cotizantes volverán? (b) asesoría: alertas tempranas a afiliados con
alto riesgo de laguna previsional. Necesitamos probabilidades **calibradas** (se agregan) y buena
discriminación (se priorizan contactos).

## Datos

Partir de `generar_universo`; el panel mensual ya contiene la dinámica laboral (estados, eventos AFC,
licencias, rubro, estacionalidad). Agregar sólo:

| Tabla | Grano | Contenido mínimo |
|---|---|---|
| `cotizaciones_conocidas` | afiliado × mes | RIM del panel con el desfase real: conocida en `mes+2` (usar la misma regla que el proyecto 01: pago normal m+1, disponible en m+2; 5 % tardíos) |
| `afc_eventos` | evento | inicio y término de relación laboral conocidos en `mes+1` (derivar del panel: `evento_afc`) |
| `licencias` | evento | días de licencia conocidos en `mes+1` (derivar del panel) |
| `prestaciones_cesantia` | evento | giros del seguro de cesantía tras un término (5 giros decrecientes), conocidos al mes |

No inventar una dinámica nueva del target: el target se lee del panel. Sí documentar qué proporción de
salidas y reingresos hay por rubro, edad y tipo de contrato en el universo generado.

## Target y población

* Fila = afiliado × `periodo_corte`, cortes mensuales móviles. Último mes con RIM conocida = `corte-2`;
  eventos AFC y licencias conocidos hasta `corte-1`.
* **Modelo A (cese)**: población = afiliados con RIM > 0 en `corte-2`. `y3 = 1` si no hay RIM > 0 en ninguno
  de `corte+1 … corte+3`; `y6` análogo con 6 meses. Son dos targets; entrenar dos modelos o uno
  multi-horizonte, justificar.
* **Modelo B (reingreso)**: población = afiliados dependientes con RIM = 0 en `corte-2` y al menos una
  cotización en los 24 meses previos. `y = 1` si hay RIM > 0 en `corte+1 … corte+6`.
* Excluir retirados/pensionados sin cotización en 24 m; excluir cortes con horizonte incompleto.

## Features (mínimo)

Historia: lags de RIM, densidad 3/6/12/24 m, racha actual de cotización o de ceros, meses desde último
cambio, variabilidad. Relación laboral: meses con el empleador, tipo de contrato (AFC), rubro, tamaño del
empleador, tasa de salida del empleador en 12 m, crecimiento de dotación. Eventos: término AFC en
`corte-1`/`corte-2`, inicio AFC, días de licencia recientes, giros de cesantía en curso. Perfil: edad,
meses hasta pensión, sexo, región, educación, tipo de afiliado. Calendario y macro: mes del año, desempleo
del corte, estacionalidad del rubro (tasa histórica de salidas del rubro en ese mes).

## Modelo y evaluación

* Baselines: tasa base por rubro × mes; regla "término AFC reciente o contrato a plazo con < 6 meses".
* LightGBM binario por target, con calibración isotónica sobre validación; reportar Brier y curva de
  calibración por decil.
* Split temporal con gap igual al horizonte; test en los últimos 3 cortes.
* Métricas por segmento (rubro, edad, tramo de renta, tipo de contrato) + **métrica de negocio**: número
  esperado de cesantes vs real por mes de corte (error relativo) y % de salidas capturadas contactando al
  10 % de mayor riesgo.
* Importancias e interpretación de 5 casos (uno por rubro estacional).

## Entregables y aceptación

Según `CONVENCIONES.md`. Aceptación: AUC test ≥ 0,80 en cese 3 m y ≥ 0,70 en reingreso; error relativo del
agregado mensual ≤ 10 %; Brier mejor que la tasa base; runtime < 5 min con 20.000 afiliados; tests < 2 min.
