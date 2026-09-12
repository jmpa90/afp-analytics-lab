# Brief ronda 3 — códigos de planilla en `rim_nowcasting` y modelo de datos regulatorio (worktree `/home/user/wt/datos`)

## Objetivo 1 — códigos de movimiento Previred en el proyecto 01 (PySpark)
`afp_lab/common/planilla.py` y `tablas_equivalencia.py` implementan la planilla tipo Previred con
la Tabla N°7 de códigos de movimiento (ver `docs/normativa_tablas_afp.md`). El proyecto 01
(`afp_lab/rim_nowcasting`, PySpark 4.0.1, pandas < 3) todavía no los usa.

1. En `synthetic.generar_cotizaciones` (o en un paso posterior sobre la misma tabla) agregar
   `cod_movimiento`, `fecha_desde`, `fecha_hasta` y `dias_trabajados` coherentes con la Tabla N°7
   y con el panel de verdad (inicio y término de relación, subsidios, permiso sin goce,
   ausentismo). Reusar la lógica de `common/planilla.py` (`_movimiento_y_dias`) si es viable por
   partición con pandas; si no, replicarla con las mismas constantes. Los códigos deben
   respetar el mes de recepción ya modelado (rezagos, DNP, rectificaciones).
2. En `features.py` agregar features as-of (recepción ≤ corte y período según la regla del
   proyecto): conteos por código en 3/6/12 meses, días trabajados del último período conocido,
   flags de término y de subsidio recientes. Cero leakage: revisar `_features_eventos` como
   referencia.
3. Evaluar con `--perfil sistema` y `--perfil cuprum` a la escala de la corrida de referencia
   del README de 01 (o menor si el tiempo apremia, diciéndolo): delta de WAPE y sesgo global y
   por segmento (independientes, con subsidio, con término reciente). Runtime: máximo +30 %
   sobre lo documentado.

## Objetivo 2 — `docs/modelo_datos_regulatorio.md`
Esquema propuesto (tablas, claves, dominios, cardinalidades) para movimientos de cotización y
afiliación alineado con el Compendio de Normas de la SP y con Previred, referenciando
`docs/normativa_tablas_afp.md` y `afp_lab/common/tablas_equivalencia.py`. Una o dos páginas:
qué campos ya existen en el sandbox, cuáles faltan, y qué proyecto los consumiría.

## Entregables
Código, tests de 01 pasando más tests nuevos para los códigos y las features, README de 01
(sección "Ronda 3": antes/después), el doc, e informe estructurado al gerente (archivos,
comandos, números, pendientes). `PYTHONPATH=/home/user/wt/datos`; Java 21 disponible.
