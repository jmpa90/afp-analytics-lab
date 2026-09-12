# Brief ronda 3 — densidad del perfil `sistema` y escala de `riesgo_empleador` (worktree `/home/user/wt/empleador`)

## Objetivo 1 — calibrar la densidad del perfil `sistema` (`afp_lab/common/synth_base.py`)
La simulación con perfil `sistema` deja ~79 % de afiliados cotizando en un mes típico; la Ficha
165 de la SP (`docs/contexto_afp_cuprum.md` §0) da 5.775.070 cotizantes sobre 12.201.632
afiliados = 47,3 % para el sistema y 66 % para Cuprum. Ajustar `p_activo_inicial`,
`factor_contratacion` (y si hace falta otro parámetro de `rim_nowcasting.synthetic._simular_particion`,
sin romper el proyecto 01) para que la fracción de afiliados con RIM > 0 en un mes típico quede
en 47 ± 3 % (sistema) y siga en 66 ± 3 % (cuprum). Agregar en `afp_lab/common/tests` un test que
mida esa fracción para ambos perfiles.

Es un cambio transversal: después correr la suite completa (`python3 -m pytest afp_lab -q`,
~10 min) y las corridas de referencia de 02, 03, 04, 05 y 06 con `--perfil sistema` (cada una
< 2 min a su escala por defecto; 01 con Spark solo si queda tiempo). Informar qué métricas
cambian y cuánto. Corregir solo los tests que fallen por supuestos del generador; no bajar
umbrales de modelos sin justificar en el informe. Actualizar la tabla "Resultados de la
ronda 1/2" del README raíz solo si un número cambia de forma material (decirlo).

## Objetivo 2 — más empleadores en `afp_lab/riesgo_empleador`
La ronda 2 aprobó 05 con reserva: el agregado mensual es intrínsecamente volátil con ~500
empleadores. `generar_universo` ya acepta `afiliados_por_empleador`. Correr a 20 000 afiliados
con 40 (actual, ≈500 empleadores) y con 10 (≈2 000), comparar volatilidad del agregado mensual,
captura y AUC; elegir un default para `run.py` y dejar tabla antes/después en el README de 05.
Si supera 5 min, usar 10 000 afiliados y decirlo.

## Entregables
Código, tests, READMEs (common, 05 y raíz si aplica) e informe estructurado al gerente
(archivos, comandos, números antes/después por proyecto, pendientes).
