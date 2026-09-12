# Revisión ronda 2 · 03 propension_apv

**Veredicto del gerente: APROBAR con correcciones menores** (integra a `main` tras aplicarlas).

## Verificado
- Split: gaps perfectos dentro de cada origen y en el split global; el memorizador por afiliado rinde
  igual en valid (0,657) que en test (0,638), antes 0,90 vs 0,68.
- Early stopping real (93–181 árboles sobre tope 3.000), smearing en validación, isotónica no aplicada por
  umbral; decil 10 de 0,380/0,272 a 0,251/0,256; ECE 0,0018; sesgo del monto +1,0 % con el decil 10
  (70 % del monto) a +3,5 %.
- Convención de fechas: batería propia de mutaciones por fuente sin diferencias y control negativo que sí
  cambia; 0 % del monto del target observado al armar la lista; exclusión y target contiguos.
- Memorización acotada (train 0,853, CV agrupada 0,826); lectura de negocio corregida y ventaja sobre
  renta en 12 de 12 cortes; menores m1, m4–m9 verificados; reproducción byte a byte de 10 reportes.

## Correcciones exigidas antes de integrar
- **m-2**: el valid del origen 202601 comparte 3 meses de target con el pool de test (2,1 % del monto).
  Exigir gap del valid de cada origen contra el pool o declararlo como limitación.
- **m-6**: persistir los 4 modelos (`modelo/origen_<yyyymm>/`) o declarar que el `.pkl` es sólo el último
  origen y no reproduce el AUC agregado.
- **m-1**: baseline de tasa base por origen se convierte en escalón con AUC 0,53; usar una constante
  única o aclararlo.
- **m-3 / m-4**: acompañar el sesgo agregado (+1,0 %) y el WAPE (0,555) con el rango por corte
  (sesgo ±15 %, WAPE 0,61–0,63 en 2 cortes) y decir que la aceptación se evalúa agregada.
- **m-5**: quitar `sin_cotizacion_24m` y `sin_tramo_declarado` (constantes).
- **m-7, m-8, m-9, m-10**: etiquetar la fila de calibración de referencia, fundar el umbral del 1 % en
  la dispersión entre folds o documentarlo, cuantificar la deriva de `rim_ref_imm` (−14 %), y mostrar
  "sin aportantes" para PENSIONADO_ACTIVO.
