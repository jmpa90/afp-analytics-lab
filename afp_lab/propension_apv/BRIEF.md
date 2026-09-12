# Brief 03 · Propensión a ahorro previsional voluntario (APV y cotización voluntaria)

**Responsable**: DS-2. **Ronda**: 1. **Gerente**: revisa contra `CONVENCIONES.md` §7.

## Pregunta de negocio

Campañas de ahorro voluntario: ¿qué afiliados abrirán APV (o harán un aporte de cotización voluntaria) en
los próximos 6 meses, y cuánto aportarán? Se usa para priorizar contactos de asesoría y para proyectar el
flujo de ahorro voluntario. Dos salidas por afiliado: probabilidad de aportar y monto esperado (probabilidad
× monto condicional).

## Datos

Partir de `generar_universo`. Agregar tablas sintéticas:

| Tabla | Grano | Contenido mínimo |
|---|---|---|
| `aportes_voluntarios` | evento | afiliado, mes, tipo (APV régimen A, APV régimen B, cotización voluntaria, depósito convenido), monto, canal (descuento por planilla, transferencia, app), periodicidad (mensual / esporádico) |
| `apvc_empleador` | empleador × mes | si el empleador ofrece APV colectivo y el % de match |
| `campanas` | evento | afiliado, mes, canal (email, app, ejecutivo, asesor), respuesta (abrió, ignoró) |
| `saldos` | afiliado × mes | saldo obligatorio aproximado (10 % de la RIM acumulada con rentabilidad) y saldo voluntario |
| `tributario` | afiliado × año | tramo del impuesto global complementario estimado desde la renta anual (para el beneficio del régimen B) |

Mecanismo del target (documentar parámetros): la propensión crece con renta (fuerte, por el beneficio
tributario), estabilidad (densidad 12 m alta, mismo empleador), edad 35–55, nivel educacional, bonos en
marzo (rubros finanzas y minería) y aguinaldos, campaña reciente (efecto pequeño), empleador con APVC (efecto
grande), historial de aportes esporádicos; cae con cesantía, licencia larga, renta cercana al mínimo y
edad > 60. Montos: log-normales, mayores en régimen B y en tramos altos; los aportes por planilla son
mensuales y estables, los esporádicos se concentran en marzo–abril (bonos) y diciembre. Tasa objetivo:
12–18 % de los cotizantes con algún aporte en el año; 5–8 % con aporte mensual. Ruido suficiente para AUC en
test de 0,75–0,88.

## Target y población

* Fila = afiliado × `periodo_corte`, cortes mensuales móviles.
* Población: afiliados vigentes que cotizaron al menos una vez en los 12 meses previos al corte
  (información con desfase de 2 meses) y **sin aporte voluntario en los 6 meses previos** (adquisición y
  reactivación; el mantenimiento de aportantes mensuales es otro problema: excluir).
* `y = 1` si hay al menos un aporte voluntario en `corte+1 … corte+6`; `monto = suma` de esos aportes.
* Excluir cortes cuyos 6 meses no están completos.

## Features (mínimo)

Renta y estabilidad (as-of, desfase 2 meses): RIM de referencia, media 12 m, densidad, meses con el
empleador, rubro, tamaño del empleador, tipo de afiliado, tramo tributario estimado, `es_marzo` y meses a
marzo. Cuenta: saldo obligatorio, saldo voluntario histórico, meses desde último aporte, aportes en 12/24 m.
Perfil: edad, sexo, educación, región, meses hasta pensión. Campañas: contactos 1/3/6 m por canal y
respuestas. Empleador: ofrece APVC, % match, % de trabajadores del empleador con APV (as-of).

## Modelo y evaluación

* Baselines: tasa base; regla "renta > 2 M y empleador con APVC o aporte en 24 m".
* Hurdle: LightGBM binario + LightGBM regresión sobre `log(monto)` en aportantes; monto esperado = p × monto.
* Split temporal con gap de 6 meses; test en los últimos 3 cortes.
* Métricas de clasificación por segmento (tramo de renta, edad, rubro) y de regresión para el monto
  (WAPE, sesgo agregado) + **métrica de negocio**: monto esperado captado contactando al 10 % y 20 % de mayor
  monto esperado vs contactar por renta.
* Calibración de la probabilidad (Brier, curva) porque se agrega para proyectar flujo.

## Entregables y aceptación

Según `CONVENCIONES.md`. Aceptación: AUC test ≥ 0,75; lift decil 1 ≥ 3; el monto esperado supera al
ordenamiento por renta en captura al 10 %; WAPE del monto en aportantes ≤ 0,6; runtime < 5 min con 20.000
afiliados; tests < 2 min.
