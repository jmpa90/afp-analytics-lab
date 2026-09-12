# Revisión ronda 1 · 04 riesgo_cese_cotizacion

**Veredicto del gerente: DEVOLVER** (diseño recuperable; leakage real en la feature dominante y un segmento
donde el modelo pierde). Revisión adversarial independiente verificada ejecutando código.

## Lo que está bien
- `construir_features` nunca toca el panel; as-of de RIM (recepción ≤ corte-1, periodo ≤ corte-2) y de
  AFC/licencias (≤ corte-1) correctos y con tests que falsifican la recepción. Tardíos realmente tardíos.
- Targets leídos del panel, poblaciones disjuntas, aritmética de periodos correcta, calibración sólo en
  validación, tablas derivadas coherentes con el panel, reproducible, rápido, métricas de `common`.
- Sin el leakage de los giros el modelo sigue cumpliendo AUC ≥ 0,80 (cese 3 m) y ≥ 0,70 (reingreso).

## Bloqueantes
1. **Leakage as-of vía `prestaciones_cesantia`**: el generador emite un giro en el mes p sólo si el RIM del
   panel en p es 0 (P = 1,0), y `_features_eventos` usa giros con periodo ≤ corte (dos meses más allá del
   límite de RIM). Alterando sólo el mes `corte` del panel cambian `cesantia_giro_actual`,
   `cesantia_n_giros_recibidos` y `cesantia_en_curso`. Es la feature #1 por gain; quitarla baja el AP 12 %.
   El README afirma lo contrario y el test de leakage no cubre `corte-1`/`corte`.
   *Corrección*: derivar la cobertura de cesantía desde el TERMINO AFC conocido + tipo de contrato + tasas
   legales (`giros_restantes = max_giros - meses_desde_termino`), sin leer RIM futura; filtrar giros con
   periodo ≤ corte-1; test que altere el panel en corte-1/corte, regenere las tablas derivadas y exija
   features idénticas.
2. **Segmento INDEPENDIENTE (6,9 % de filas)**: en el universo su RIM es constante dentro del año
   calendario, así que el target es una función del mes del corte; el baseline rubro × mes gana al modelo en
   AUC y Brier (incumple CONVENCIONES §4) y explica el 100 % del error agregado de y6 en 202603 (+43 casos
   sobre 0 reales). El diagnóstico del README ("SII en junio") no corresponde a este universo.
   *Corrección*: excluir INDEPENDIENTE y VOLUNTARIO de los modelos A y B (son problemas anuales/aleatorios,
   distintos), documentarlo y recalcular la métrica de negocio.

## Mayores
3. **Gap del split = 1 mes**, no h: usar `corte ≤ test_min - 2*h` y subir `N_CORTES`; corregir README §5.
4. **Población de B contaminada por recepción tardía**: 16,6 % de "no cotizantes" son planillas pendientes
   (target 0,96). Exigir RIM conocida = 0 en corte-2 y corte-3; reportar la cobertura de la población A
   respecto de los cotizantes reales (95,3 %) y el factor de corrección del agregado.
5. **Bug `reindex(fill_value=0)` no rellena NaN de `unstack`**: `emp_tasa_salida_12` y
   `emp_n_trabajadores` mal calculados (un empleador que se queda sin planilla da tasa 0). `fillna(0)` +
   flag `emp_sin_planilla_conocida`.
6. **`cesantia_n_giros_recibidos`** es el máximo histórico, no del episodio vigente.

## Menores
7. Ventana de 30 meses sin acotar por `periodo_inicio`: rachas saturan en tope distinto por corte (deriva
   artificial train/test). Acotar la grilla o empezar los cortes ≥ 30 meses después del inicio.
8. `tipo_contrato` describe relaciones ya terminadas (1.600 de 5.528 con TERMINO posterior).
9. `emp_crecimiento_12` sesgado por censura de recepción del último mes (mediana 0,90 para todos).
10. Métricas por segmento sólo para el modelo: iterar los 3 predictores en los CSV.
11. `metrica_negocio_agregado`: error con signo llamado WAPE; el brief pide el error mensual, no el
    trimestral.
12. Ejemplos interpretados degenerados (meseta isotónica, SHAP sobre el booster crudo).
13. Anotaciones de tipo incorrectas, columnas no usadas, rama "por obra" vacía, giros con periodo
    posterior a `periodo_fin`, estacionalidad del rubro calculada sobre el mes del corte y no sobre la
    ventana de target.

## Condiciones de reingreso
Corregir 1 a 6 y los menores 7, 8, 10 y 11; re-correr; reescribir README §2, §5, §6 y §8 con los números
nuevos y una sección explícita de lo que se corrigió.
