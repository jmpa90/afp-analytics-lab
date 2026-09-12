# Revisión ronda 2 · 03 propension_apv, addendum perfil Cuprum

**Veredicto del gerente: APROBADO** (verificación directa del gerente, sin revisor externo por
economía de créditos).

- `AJUSTE_INTERCEPTO_PERFIL = {"sistema": 0, "cuprum": 0}`: la mayor propensión de Cuprum (27 % anual,
  14 % mensual vs 15 % / 7 %) emerge de renta, tramo IGC y educación, no de un ajuste ad hoc.
- Argumento correcto de que el 29 % de participación de Cuprum en APV es stock acumulado desde 1981 y
  no flujo; techo de cordura 0,319 cuentas APV por cotizante (Ficha N° 165) respetado.
- Mezcla régimen A/B recalibrada con el 76,7 % de depósitos en régimen A de la Ficha (sistema 75 %,
  Cuprum 68 %); efecto colateral sobre el perfil sistema del orden de 0,001 en AUC, aceptado.
- Corrida `--perfil ambos` reproducida con n=4.000: `calibracion_ficha_165.csv` coherente (ingreso
  promedio simulado 0,98 del oficial; depósito promedio +43 %, brecha declarada); los cuatro criterios se
  cumplen con Cuprum (AUC 0,821, lift 4,37, WAPE 0,533, captura 55 % vs 33 % por renta a 20.000).
- Lectura de negocio: en Cuprum la ventaja del modelo sobre "ordenar por renta" sube (+22 pp) porque la
  renta discrimina menos en una cartera homogénea de alta renta; el mantenimiento de aportantes pesa más
  que la adquisición (población elegible 15 % menor).
