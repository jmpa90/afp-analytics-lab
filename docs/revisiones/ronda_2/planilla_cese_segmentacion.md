# Revisión ronda 2 · planilla regulatoria, 04 cese (addendum Cuprum) y 06 segmentación (DS-3)

**Veredicto del gerente: APROBADOS** (verificación directa del gerente por economía de créditos:
56 tests de las tres carpetas y 140 del repo completo tras la fusión; corridas reproducidas).

- **Planilla**: recepción y estado sorteados una vez por declaración (bug reportado por DS-2 corregido),
  DNP a nivel empleador × periodo con persistencia AR(1) y estacionalidad configurables, agregado
  empleador × periodo, códigos de movimiento de la Tabla N° 7. Es aditiva: no cambia `generar_universo`.
- **04 Cuprum**: cese 3 m AUC 0,885 (sistema 0,868), reingreso 0,641 (0,720): coherente con más densidad
  y jubilación real. Los códigos de planilla como features aportan ΔAUC entre −0,004 y +0,002: no entran
  a producción, documentado y fijado por test.
- **06 Segmentación**: k-means, k=7 con fusión del segmento < 3 % → 6 segmentos; silueta 0,207 (justo
  sobre 0,20), ARI de re-muestreo 0,99, separación de negocio 3/3 tasas con razón ≥ 2, clasificador
  0,983. Limitaciones aceptadas: ARI temporal 0,58 (reportado, no criterio) y tablas de 02/03 generadas
  con perfil sistema (los generadores no aceptaban `perfil` en esa rama). **Seguimiento**: regenerar con
  `perfil="cuprum"` ahora que 02 y 03 lo soportan en `main`, y recalcular ARI temporal tras la fusión.
