# Brief 11 — `afp_lab/voz_afiliado` (NPS y experiencia; worktree `/home/user/wt/empleador`, rama `ronda-5/nps`)

## Pregunta
Solo una fracción pequeña de los afiliados contesta la encuesta de NPS, y quienes contestan no son
representativos. ¿Podemos inferir el NPS de TODA la cartera con los eventos operativos que la AFP ya
registra, decir qué eventos mueven el NPS y cuánto vale un punto de NPS en traspasos y ahorro voluntario?

## Datos sintéticos (`synthetic.py`)
* Universo común `cuprum` (10 000 afiliados, 2022-01 a 2028-12) + tablas de 02 (`churn_traspaso`:
  contactos, reclamos, uso digital, saldos, traspasos, mercado), 03 (`propension_apv`: aportes) y
  `common/planilla` (rezagos, DNP del empleador), importadas sin modificar y con el mismo `seed`.
* **Satisfacción latente** `lat_satisfaccion` (escala 0–10) por afiliado y mes: base por segmento y
  edad, más efectos de eventos operativos con decaimiento en el tiempo, con magnitudes declaradas
  como supuestos (por ejemplo: reclamo sin resolver −2,0; rezago > 30 días −1,5; licencia mal pagada
  −2,5; empleador en DNP −0,8; contacto con ejecutivo resuelto +0,7; rentabilidad relativa positiva
  +0,5; brecha de comisión desfavorable −0,6), más un componente persistente por afiliado y ruido.
* **Encuesta**: cada mes se invita a ~8 % de la cartera, con más probabilidad tras un evento (encuesta
  transaccional) y una parte relacional aleatoria. Responde ~25 %, con **sesgo de no respuesta**:
  contestan más los muy satisfechos y los muy insatisfechos, menos los jóvenes y los de bajo uso
  digital. Respuesta = redondeo de `lat_satisfaccion` + ruido, acotada a 0–10. Promotor 9–10, pasivo
  7–8, detractor 0–6; NPS = % promotores − % detractores. Guardar también la propensión verdadera de
  responder (latente) para evaluar la corrección de sesgo.
* Todo lo latente en tabla `*_latente` con prefijo `lat_`, jamás en features.

## Features (`features.py`, as-of estricto)
Eventos operativos en ventanas 1/3/6/12 meses hasta el corte (reclamos, contactos, rezagos, DNP,
licencias), uso digital, saldo y tramo, antigüedad, edad, región, segmento de 06 si aporta, mercado
(brecha de comisión y rentabilidad). Historial de encuestas anteriores solo hasta el corte (nunca la
respuesta del mismo mes ni posteriores).

## Modelos y evaluación
1. **NPS inferido**: modelo de la respuesta (0–10, o clases detractor/pasivo/promotor) entrenado con
   respondientes, ponderado por el inverso de la probabilidad estimada de responder (modelo de
   respuesta con LightGBM) para corregir el sesgo. Comparar: NPS ingenuo (solo respondientes), NPS
   inferido sin ponderar, NPS inferido ponderado, contra el **NPS verdadero de toda la cartera**
   (latente). Reportar por mes, por segmento, por tramo de edad, región y tramo de empleador. AUC
   de detractor en holdout de respondientes. Calibración por decil.
2. **Drivers en puntos de NPS**: efecto de cada evento sobre el NPS con un estimador causal simple
   (T-learner o regresión ajustada por propensión sobre "ocurrió el evento en la ventana"), comparado
   contra los efectos verdaderos del generador (signo y orden de magnitud). Tabla "qué arreglar primero".
3. **Valor de un punto de NPS**: asociación entre NPS inferido y traspaso a 6 meses (target de 02) y
   aporte APV a 6 meses (target de 03), ajustada por covariables y segmento; expresarlo en traspasos
   evitados y saldo por punto de NPS, con la advertencia explícita de que es asociación ajustada, no
   experimento (02 y 03 no conocen la satisfacción).
4. Splits temporales con gap como en 02; hiperparámetros solo con validación; semillas fijas.

## Aceptación propuesta
NPS inferido ponderado dentro de ±3 puntos del verdadero en los cortes de test; reduce el error del
NPS ingenuo al menos a la mitad; AUC de detractor ≥ 0,75; drivers con el signo correcto en ≥ 80 % de
los eventos. Runtime de referencia < 4 min a 10 000 afiliados con `n_jobs=2`; tests < 60 s.

## Entregables
Proyecto completo según `CONVENCIONES.md`: `synthetic.py`, `features.py`, `model.py`, `evaluate.py`,
`run.py`, `tests/`, `README.md` (pregunta, datos, features, modelos, tablas de resultados con
antes/después de la corrección de sesgo, drivers, valor del punto, lectura de negocio, decisiones,
limitaciones, cómo correr). Sin git desde el agente. Nada inventado: supuestos con nombre y valor.
