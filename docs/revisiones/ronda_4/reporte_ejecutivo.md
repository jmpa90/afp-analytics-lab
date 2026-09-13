# Revisión adversarial — informe ejecutivo (`docs/reporte_ejecutivo.html`)

Revisor: DS Sonnet (ronda 4). Transcrita por el gerente porque el harness del revisor no le permitió
escribir este archivo. Veredicto sobre la primera versión (commit 01b9a45 de `ronda-4/reporte`).

**Veredicto: aprobar con correcciones obligatorias; no publicar hasta resolver la #1.** La exactitud
numérica es muy alta y el tono es honesto donde debe serlo (04 reingreso, 05 masa en riesgo y 09
bandit se presentan como "en el límite" o "no cumple"). Los problemas son de proceso e integridad de
fuentes, no de dígitos.

**Cifras verificadas: 127 de 185 (100 % de portada, KPI y conclusiones de cada caso). Coincidencia
exacta: 127/127.**

## Correcciones obligatorias (orden de gravedad)

1. **Proyecto 08 no existía en la rama del informe y no tenía revisión adversarial.** El HTML y 10
   cifras del JSON citan `afp_lab/secuencial_afiliado/README.md`, pero el directorio vivía solo en
   `ronda-4/secuencial` y `docs/revisiones/ronda_4/` no tenía revisión de 08. Corrección: fusionar y
   encargar la revisión, o marcar la tarjeta como pendiente de integración.
2. **Corrección de 04 (reingreso) etiquetada "(ronda 3)" y sin revisión propia.** Se hizo en la rama
   `ronda-4/cese` (merge f468d6f). Corrección: etiquetar como corrección posterior y adjuntar revisión.
3. **La reserva de 04 invierte la secuencia del AUC de reingreso.** Decía "cayó de 0,720 a 0,608 ...
   se corrigió (+0,003 a +0,008)". Según §12 del README de 04, la recalibración lo bajó a 0,605 y la
   corrección de features lo subió a 0,608. Corrección propuesta: "cayó de 0,720 a 0,605; con las
   features del último empleo subió a 0,608 (+0,003 en la fila de referencia; rango +0,003 a +0,008 y
   un caso de −0,005)".
4. **La reserva de 04 no aclara lo que el texto promete.** La captura 63,1 % / 63,5 % y el baseline
   0,798 son anteriores a la recalibración de densidad y no se remidieron; la caja de reserva debe decirlo.
5. **Numeración 07/08/09 chocaba con la tabla de portafolio del README raíz** (07 y 08 eran backlog).
   Resuelto en `main` (28dea89): el backlog pasó a 10 y 11.
6. **"Cada entrega pasa por revisión adversarial antes de aprobarse" no era cierto** para 01 (ronda 1),
   la corrección de 04 ni 08. Acotar la frase o completar las revisiones.

## Menores

* `p05_precision_lista_modelo/regla` citan §7; están en §6.3. `p03_voluntario_pct` cita §7; aparece en §6.4.
* El KPI de la tarjeta 07 muestra 0,195 (DR-learner) junto a cifras del X-learner (0,186) sin aclarar
  que son métodos distintos.
* El AUC 0,969 de cotiza/no cotiza (01) no especifica horizonte (h=1; h=2 es 0,918).
* Anexo técnico, fila 01: "Tests/Runtime: —" sin explicación.

## Verificado sin problemas

Los 18 datos de la Ficha 165, las 4 cifras de recalibración de densidad, 7 de `rim_nowcasting`, 22 de
`churn_traspaso` (incluida la tabla de licitación 55/65/75 %), 9 de `propension_apv`, 9 de
`riesgo_cese_cotizacion`, 14 de `riesgo_empleador` (incluida la cita "45-55 %... no un cumplimiento
sólido"), 10 de `segmentacion_afiliados`, 11 de `uplift_retencion` (incluida la advertencia de que el IC
del Qini cruza cero) y 9 de `bandit_contacto` (incluido el 3/5 vs 4/5 exigido): todos coinciden
exactamente con su fuente, sin redondeos al alza ni cifras sintéticas mal marcadas.

## Decisión del gerente

Se encargan revisiones adversariales de 08 y de la corrección de 04; el informe se corrige en una sola
pasada con las seis correcciones y los menores, se regenera y se re-verifica antes de publicar.
