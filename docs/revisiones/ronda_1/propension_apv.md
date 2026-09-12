# Revisión ronda 1 · 03 propension_apv

**Veredicto del gerente: DEVOLVER** (features limpias; protocolo de validación contaminado que infla las
tres cifras con que se vende el resultado). Revisión adversarial independiente verificada ejecutando código.

## Lo que está bien
- As-of de features limpio bajo mutaciones agresivas del futuro en 6 tablas y 4 cortes (0 diferencias en
  66 columnas). Recalculo manual de 8 features en 2 casos × 2 cortes correcto. Población y target como se
  documentan, sin duplicados, sin latentes. Gap train+valid → test correcto.
- Simulación plausible y no determinista; tasas del brief cumplidas; runtime 34 s; 20 tests en 8 s.
- La conclusión comercial principal (captura al 10 % 0,68 vs 0,50 por renta) es robusta al arreglo del
  protocolo.

## Bloqueantes
1. **Validación contaminada**: no hay gap entre train y valid; 5 meses de target compartidos. Un
   memorizador por afiliado da AUC 0,90 en valid y 0,68 en test; el AUC de valid está anticorrelacionado
   con el de test (700 árboles: valid 0,891 / test 0,834; 50 árboles: test 0,845). El early stopping nunca
   corta (`n_arboles_clf = 700 = n_estimators`).
   *Corrección*: gap de h meses también entre train y valid. Probado: con `train ≤ 202311` el modelo usa
   36 árboles y mejora en todo (AUC 0,841, Brier 0,044, ECE 0,007, decil 10 0,288/0,271, sesgo +1,2 %).
2. **Isotónica rota en el decil 10** (0,380 predicho vs 0,272 real, +40 %) y empeora el test respecto de la
   probabilidad cruda; la decisión de calibrar se tomó sobre el set contaminado con una diferencia de Brier
   de 1e-4. El diagnóstico del README §6.3 (tasa base de la ventana) es falso: con la misma ventana y gap
   en train el decil 10 queda 0,288/0,271.
3. **El +16,6 % de sesgo agregado del monto es un artefacto** (deciles 1–9 subestimados −50 a −100 %,
   decil 10 +102 %; con protocolo corregido +1,2 %). Reportar sesgo por decil.
4. **Hiperparámetro elegido mirando test** (700 árboles tras ver que 2.000 empeoraban test). Fijar por
   early stopping con validación con gap y reportar test sin haberlo consultado.

## Mayores
5. **Desalineación con el momento de la decisión**: la lista se arma en `hoy = corte+2` con aportes
   conocidos sin desfase, así que 33 % (61 % en los cortes de test) del monto del target ya ocurrió y la
   exclusión "sin aporte 6 m" se sirve sobre otra población que la entrenada.
   *Corrección (convención del laboratorio)*: `periodo_corte` = mes de la decisión; panel/RIM se leen en
   `corte-2`; aportes y campañas hasta `corte`; target `corte+1 … corte+6`.
6. **Sobreajuste con memorización por afiliado** (AUC train 0,986; `region` y `meses_desde_afiliacion` con
   gain sin entrar en el target). Regularizar y usar CV agrupada por afiliado además de temporal.
7. **Smearing calculado in-sample** sobre un regresor sobreajustado (1,21 train vs 1,26 valid). Calcular en
   validación con gap.
8. **Valid y test de 3 cortes contiguos** con tasa base estacional (0,030–0,077 por corte). Evaluar en ≥ 12
   cortes u orígenes móviles y reportar por corte.
9. **Lectura de negocio sobredimensionada**: el 68,6 % es del ahorro de adquisición/reactivación (38 % del
   flujo total), no "del semestre"; y parte ya observado al contactar (5).

## Menores
m1 `truncar_tablas` débil para `tributario`; m2 `lift_decil_1` con empates (corregido en common por el
gerente); m3 la propuesta de `observed=True` ya la aplicó el gerente en common; m4 `n_empleadores_12`
cuenta transiciones; m5 ventana de licencia 3 vs 2 meses; m6 `tasa_apertura_6m` ambigua; m7 "nunca" = 99
en feature continua; m8 renta anual desde RIM topada; m9 rama muerta en `meses_con_apvc`.

## Condiciones de reingreso
Corregir 1 a 8; reescribir README §5, §6, §7 y §8 con los números nuevos y sesgo por decil; aclarar 9.
