# Brief ronda 3 — `afp_lab/churn_traspaso` (worktree `/home/user/wt/churn`)

Contexto: la ronda 2 dejó la licitación de stock 2027 simulada (`synthetic.licitacion_stock`,
`licitacion.py`, README §13) y tres pendientes anotados en `docs/revisiones/ronda_2/churn_traspaso_ronda2.md`.

## Objetivo 1 — la licitación de stock retroalimenta la cartera
Hoy el sorteo de 2027-08 adjudica afiliados a otra AFP pero la cartera los sigue simulando como
si siguieran en Cuprum (saldos, traspasos, cotizaciones). Cambiar el generador para que, desde
el mes de traspaso efectivo, los adjudicados que no ejercen el derecho a quedarse salgan de la
cartera: sin nuevos eventos propios, saldo fuera del AUM, `activo_en_afp = 0` (o el campo que
corresponda). Parametrizar la lista de sorteos (`sorteos=(202708, 202908)`) para poder simular
el segundo sorteo; extender la ventana hasta 2029-12 solo si la corrida a 20 000 afiliados sigue
bajo 5 min (si no, dejar ventana y explicar). Reportar el efecto sobre la tasa del target en
los cortes posteriores al sorteo y confirmar que `cortes_evaluables(excluir_licitacion=True)`
sigue excluyendo bien.

## Objetivo 2 — intervalos bootstrap del AUC por segmento
En `evaluate.py`, intervalos percentil (≥ 200 remuestreos, semilla fija) del AUC y de la captura
por segmento, al menos por tramo de edad (incluido 60+) y por tramo de saldo. README: tabla con
punto e intervalo, marcando dónde el intervalo cruza 0,5 o el umbral de aceptación. Los tests
deben cubrir la función (rápidos: pocos remuestreos).

## Objetivo 3 — reemplazar el supuesto de 0,446 % mensual
La tasa de salida de Cuprum es hoy un supuesto anclado al promedio del sistema (README §13).
Buscar con WebSearch cifras públicas 2024–2026 de traspasos de Cuprum (salidas, entradas,
netos, mensuales o anuales): informes de la Superintendencia de Pensiones, prensa, estudios.
`spensiones.cl` y `cuprum.cl` están bloqueados por el proxy: usar los snippets de búsqueda y
fuentes accesibles. Si se obtiene una cifra: dejarla como parámetro con fuente y fecha en
`PARAMETROS_POR_PERFIL["cuprum"]` y en el README, y recalibrar. Si no se obtiene: dejar el
parámetro explícito y documentar exactamente qué se buscó y qué se encontró. Nada inventado.

## Entregables
Código, tests (los 33 actuales siguen pasando; agregar para lo nuevo), README §14 "Ronda 3"
con antes/después, e informe estructurado al gerente (archivos tocados, comandos para
reproducir, números, pendientes). Corrida de referencia: `python3 -m afp_lab.churn_traspaso.run --perfil cuprum`.
