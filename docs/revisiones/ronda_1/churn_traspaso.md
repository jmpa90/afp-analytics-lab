# Revisión ronda 1 · 02 churn_traspaso

**Veredicto del gerente: DEVOLVER** (ingeniería sólida, resultados no defendibles). Revisión adversarial
independiente reproducida y verificada con ablaciones y sensibilidad a semilla.

## Lo que está bien
- As-of sin leakage: mutaciones del futuro (RIM, panel, uso digital, traspasos falsos) sobre
  `construir_features_multi` no cambian ninguna de las 68 features. Desfase de 2 meses aplicado a todo el
  panel y al saldo. Target, población, exclusiones y split con gaps reales correctos. Sin latentes.
- Reproducible al dígito, 34 s la corrida de 20.000, tests 5,7 s. README honesto y completo.

## Bloqueantes
1. **Bloque digital = observación directa del target futuro.** Durante la "intención" (que termina en
   firma determinista en 0–4 meses) los logins se multiplican ×3,5 y la descarga de certificado ×33.
   Ablación sin bloque digital: lift decil 1 cae de 5,23 a 2,65 (no cumple ≥ 3); con parámetros plausibles
   (certificado 0,08, logins ×1,6) el AUC cae a 0,714 (no cumple ≥ 0,75). En Chile el traspaso lo tramita
   la AFP de destino, no requiere certificado de la de origen.
   *Corrección*: intención visible sólo para 30–40 % de los afiliados y con multiplicadores moderados;
   causas exógenas de descarga de certificado; chequeos de determinismo en `resumen_simulacion`.
2. **Dispersión de rentabilidad entre AFP irreal** (rango mediano 6 pp, p95 10,7 pp; real 1–2 pp) que
   genera olas de traspaso de 1,6 % a 12,6 % anual y sepulta el castigo documentado.
   *Corrección*: innovación del AR(1) a ~1/4; test sobre máximo y mínimo móvil de 12 meses de la tasa.
3. **Saldo en riesgo esperado no reproducible**: sesgo agregado de −22 % a +51 % según semilla (el −4,6 %
   del README es la semilla 42). *Corrección*: calibrar sobre la ventana reciente con horizonte cerrado o
   ajuste de prior, reportar por corte y con rango sobre ≥ 5 semillas.

## Mayores
4. Explicación del drift de tasa base ("selección de supervivencia") es falsa: es la fase de la onda del
   mercado; con seed 123 el drift se invierte.
5. Split entrena con los 21 cortes más antiguos y descarta 22 de 49; el train termina 26 meses antes del
   test. Validación contigua con purga por filas o validación deslizante.
6. Regla de negocio del brief degenerada porque la brecha de rentabilidad marca al 50 % de la cartera
   (consecuencia de 2). Mantener la regla graduada.
7. README y código no coinciden (`sigma_afiliado` 0,40 vs 0,55 documentado; dispersión "de décimos").

## Menores
8. `split_temporal` sin guardas (IndexError o train vacío en silencio con pocos cortes).
9. `construir_features_multi` descarta cortes sin historia en silencio.
10. Etiquetas de `tipo_ultimo_reclamo` desde el slice: usar catálogo fijo.
11. `common/metrics.lift_decil_1` sin desempate: **corregido por el gerente en common**.
12. `comparar_segmentos`: segmento sin positivos pasa el check con AUC NaN.
13. Métricas sin intervalo ni sensibilidad a semilla (AUC 0,768–0,837 entre semillas).
14. `meses_hasta_pension` negativo sin documentar.
15. Test de leakage tautológico (recorta antes de calcular): mutar sobre `construir_features_multi`.

## Condiciones de reingreso
Corregir 1, 2, 3, 5 y 7; documentar 4; resolver los menores. Re-correr con ≥ 5 semillas y reportar rangos.
Los umbrales del brief se mantienen sobre la simulación corregida; si no se alcanzan, reportar los números
reales y proponer umbrales con argumento.
