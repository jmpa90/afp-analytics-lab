# Revisión ronda 3 · churn_traspaso (licitación retroalimenta cartera, bootstrap por segmento, fuga Cuprum)

**Veredicto: APROBAR CON RESERVA** (revisor adversarial Sonnet). No encontré bloqueantes: sin leakage,
sin fuga hacia el target, tests sustantivos, cifras reproducidas byte a byte y las dos citas externas del
objetivo 3 verificadas de forma independiente. Dos hallazgos menores, ambos de documentación/lectura, no
de código ni de resultados.

## Qué reproduje

- `pytest afp_lab/churn_traspaso/tests -q` → **39 passed en 29,8 s** (informe: "39 passed in ~30s"; antes
  33). `pytest afp_lab/segmentacion_afiliados/tests -q` (proyecto dependiente de `generar_tablas`) → **14
  passed en 8,5 s**, sin regresión.
- `python3 -m afp_lab.churn_traspaso.run --perfil cuprum` → **52,9 s** (informe: "61,9s"; ronda 2: "45s").
  Orden de magnitud correcto (2× la ronda 2 por la segunda `generar_tablas` del objetivo 1 + bootstrap),
  bien por debajo del límite de 5 min. Métricas reproducidas **exactamente** contra el informe:
  - `metricas_test.csv`: `modelo_lgbm` AUC = 0,7141959 (informe: 0,7142); `techo_predictibilidad.csv`:
    0,738830 (informe: 0,7388).
  - `captura_negocio.csv` al 10 %: 30,0395 % traspasos / 23,0736 % saldo (informe: 30,04 % / 23,07 %).
  - `licitacion_efecto_cartera.csv`: población removida 732/728/724/722/720/718 y tasa base
    con/sin-retroalimentación por corte (2,365/2,337 % ... 0,366/0,352 %) — coinciden dígito a dígito con
    la tabla del README §14.A.
  - `resumen_simulacion.csv`: `afiliados_en_cartera_promedio` 12.023,2, `tasa_traspaso_anualizada`
    5,515 %, `tasa_salida_mensual` 0,4596 %, `traspasos_netos_total` 1.205 — coinciden con el informe.
  - `bootstrap_auc_segmento.csv`: 60+ → n=2.726, AUC=0,6963, IC[0,6429–0,7560] (cruza 0,70, no 0,5);
    saldo 40M+ → n=9.432, AUC=0,6667, IC[0,6404–0,6960] (bajo 0,70 con holgura) — coinciden con el informe.
- Diff completo (`git diff main`) revisado archivo por archivo contra CONVENCIONES §7. `git diff main
  --stat` también muestra borrado `docs/briefs/ronda_4/*`: es porque la rama `ronda-3/churn` se creó antes
  del commit `63d1cb0` (que agregó esos briefs a `main`), no algo que el DS haya tocado — el gerente lo
  verá al fusionar/rebasear, no es un hallazgo de esta revisión.
- Verificación independiente del objetivo 3 (WebSearch/WebFetch, fuera del sandbox del DS): `latercera.com`
  da efectivamente `EGRESS_BLOCKED` con WebFetch (igual que reporta el DS); WebSearch sobre el título citado
  devuelve el mismo artículo real y el mismo snippet ("Cuprum tuvo un balance neto de 4.699 traspasos...
  de los 1.014.368 traspasos totales, 44 % dentro de la misma AFP y 56 % [565.088] a otra entidad") — la
  cifra +4.699/2024 es de traspasos entre AFP, no de cambio de fondo, tal como la usa el DS. Además abrí el
  PDF `data/sp/ficha_estadistica_previsional_165_ago2026.pdf` (ya en el repo) y la Tabla N°9 muestra
  literalmente `Cuprum ... 4.128` traspasos netos de cuentas a jun-2026 y la nota (3) "el número total de
  cuentas traspasadas entre AFP en junio de 2026 fue de 56.974" — ambas cifras del informe están bien
  citadas y no son inventadas.

## Sin leakage (verificado a mano, además de los tests existentes)

- Los tests `test_sin_leakage_alterar_el_futuro_no_cambia_las_features` /
  `..._en_la_ruta_matricial` (no tocados esta ronda, `features.py` no cambió) siguen pasando.
- Hand-check propio con `generar_tablas(n_afiliados=3000, seed=42, perfil="cuprum")`: tomé un afiliado
  sorteado que decide `SE_VA` y sigue vigente el mes previo al efectivo (2028-06). (1) En un corte muy
  anterior al sorteo (2027-01) sus features (`saldo_obligatorio`, `antiguedad_afp_meses`) se construyen
  normalmente, sin verse afectadas por la salida futura. (2) En el corte igual al mes efectivo (2028-06)
  el afiliado **no aparece** en `construir_features` (0 filas): ya no está vigente. (3) Su último registro
  en `saldos` es exactamente 2028-05 (`mes_previo`), consistente con la truncación documentada. Los tres
  casos confirman que `_aplicar_salida_licitacion` no introduce fuga hacia atrás ni dejar población
  fantasma hacia adelante.
- La decisión del sorteo (`z`, `p_va`) sólo usa `antiguedad_afp_meses`/`saldo_obligatorio`/`rim`/`logins`/
  `reclamos` leídos en o antes de `periodo_sorteo` (con el mismo desfase de 2 meses del panel laboral que
  usa `features.py`): no hay información posterior al sorteo en la decisión.
- `sorteos_en_ventana` (nuevo criterio: sólo el mes de sorteo debe caber en la ventana) no cambia el
  comportamiento de la corrida por defecto: con `SORTEOS` y la ventana 2021-01→2028-12, sigue entrando
  sólo el primer sorteo (202708), igual que en la ronda 2 — confirmado leyendo el código y con la corrida
  de referencia (métricas idénticas).
- `cortes_evaluables(excluir_licitacion=True)` sigue truncando por el **primer** `periodo_efectivo` sin
  importar cuántos sorteos haya en `licitacion_stock` (usa el mínimo), verificado también con
  `test_cortes_excluyen_la_licitacion_con_dos_sorteos`.

## Tests nuevos: sustantivos, no triviales

Los 6 tests agregados sí ejercitan el comportamiento nuevo con aserciones específicas (no sólo "no
truena"): cuenta exacta de afiliados removidos, ausencia de filas nuevas en las 4 tablas operacionales
desde el mes efectivo, no doble-registro de quien ya se había ido antes, exclusividad entre primer y
segundo sorteo tras la salida secuencial, igualdad de `cortes_evaluables` con y sin segundo sorteo, y para
el bootstrap: reproducibilidad byte a byte con la misma semilla, ancho de intervalo menor en el segmento
con más eventos, y validación de errores (`metrica` inválida, `n_boot<2`).

## Hallazgos menores

1. **README §14 se contradice en el conteo de tests.** Las líneas 925 y 1090 dicen "42 tests" mientras
   que las líneas 23, 557, 916 y 1082 (y la corrida real, `pytest --collect-only` → 39) dicen "39 en
   total". Es un error de tipeo (probablemente "33 + 6 + 3" mal sumado en algún borrador), no afecta
   ningún resultado, pero conviene corregirlo antes de cerrar la ronda.
   - Evidencia: `afp_lab/churn_traspaso/README.md:925` y `:1090` ("... 42 tests ...") vs. `:916`
     ("39 en total") y `pytest --collect-only` → "39 tests collected".

2. **La tabla de `licitacion_efecto_cartera.csv` (README §14.A) no aclara que la tasa base decreciente
   corte a corte es, en su mayor parte, censura por el borde derecho de la ventana simulada, no un efecto
   del fenómeno en estudio.** Los cortes 2028-06..2028-11 quedan **excluidos del entrenamiento** por
   diseño (correcto), pero además su ventana de target (`corte+1..corte+6`) se sale cada vez más de
   `periodo_fin=2028-12`: 2028-06 observa los 6 meses completos, pero 2028-11 sólo observa 1 de 6 (el
   resto del horizonte cae en 2029, que no existe en la simulación). Reconstruí esto directamente
   (`add_months`/`periodo_to_idx`) y el patrón de meses observados (6,5,4,3,2,1) explica casi toda la
   caída de la tasa base entre filas (2,365 %→0,366 % con retroalimentación) mejor que cualquier cambio
   real de riesgo por antigüedad del sorteo. Esto **no invalida la comparación con/sin retroalimentación**
   que pide el objetivo 1 (ambas columnas sufren exactamente la misma censura por fila, así que el
   contraste relativo +1,0 a +4,0 pp sigue siendo válido) — pero un lector de negocio podría interpretar
   la tabla tal como está publicada como "el riesgo cae fuerte hacia noviembre", cuando en realidad es un
   artefacto de borde de ventana. Sugerencia: una nota en README §14.A aclarando que las columnas
   `tasa_base_*` de cortes posteriores a 2028-06 tienen ventana de target parcialmente censurada y que la
   lectura correcta es la comparación fila a fila (con vs. sin), no la tendencia entre filas.
   - Evidencia: con `HORIZONTE_MESES=6` y `periodo_fin=202812`, el corte 2028-07 ya observa sólo 5/6 meses
     de su ventana de target y 2028-11 observa 1/6 (cálculo reproducido con
     `add_months`/`periodo_to_idx` sobre `afp_lab/rim_nowcasting/calendario`).

## Notas (no son hallazgos, quedan para que el gerente las tenga a mano)

- `_aplicar_salida_licitacion` (licitacion.py) agrega el registro `SALIDA_LICITACION` pero no depura filas
  `SALIDA` voluntarias preexistentes que pudieran quedar programadas *después* de la salida forzada para
  el mismo afiliado. No afecta ninguna métrica reportada (el afiliado deja de estar vigente y por lo tanto
  fuera de la población en cualquier corte donde eso importaría, y `cortes_evaluables` ya excluye los
  cortes cuya ventana de target tocaría el sorteo), pero es una inconsistencia narrativa de los datos
  sintéticos (alguien "vuelve a traspasarse" después de ya no ser cliente) que valdría la pena limpiar si
  `licitacion.py` se mueve a `common/`.
- El informe da a entender que la cifra de la Ficha N° 165 (+4.128, jun-2026) se "halló" en la búsqueda de
  esta ronda; en realidad ya estaba documentada en el README desde la ronda 2 (línea 713 del README
  actual). El propio README §14.C lo dice correctamente ("ya estaba descargada en `data/sp/`"), así que no
  es un problema del entregable, sólo una imprecisión menor en el resumen del informe al gerente.
- El segmento `tramo_edad=60+` sigue en `segmentos_perdidos.csv` (el modelo no le gana al baseline ahí,
  AUC 0,696 vs 0,719) exactamente igual que en la ronda 2 (métricas idénticas): el bootstrap del objetivo 2
  agrega justo el contexto que faltaba (el intervalo cruza 0,70, no se puede concluir que el modelo "no
  sirve" ahí) — buen cierre del pendiente de la revisión de la ronda 2.

## Checklist CONVENCIONES §7

- [x] Sin leakage: tests presentes + 2 casos verificados a mano (arriba).
- [x] Target bien definido, población y exclusiones (`SALIDA` vs `SALIDA_LICITACION`) coherentes.
- [x] Dinámicas sintéticas plausibles y documentadas (dos citas externas verificadas de forma
      independiente).
- [x] Baselines y segmentos: sin cambios respecto de la ronda 2 (métricas idénticas); el segmento 60+
      sigue perdido pero ahora con intervalo bootstrap que contextualiza el hallazgo.
- [x] Métricas comparables y métrica de negocio: sin cambios de código, reproducidas.
- [x] Runtime y tests dentro de límite; reproducible con semilla (reproducido en esta revisión).
- [x] README completo según plantilla, con los dos matices menores de arriba.
