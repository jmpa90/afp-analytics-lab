# Revisión ronda 4 · corrección 04 `riesgo_cese_cotizacion` (README §12, "Ronda 3 — corrección")

**Veredicto: APROBAR** (revisor adversarial Sonnet). El diagnóstico de la caída del AUC de
reingreso está sustentado con evidencia de código verificable línea por línea (no solo
narrativa), las 3 features nuevas son as-of y pasan un test de leakage dedicado, no se tocó
ningún umbral ni se desactivó ningún test, y los números "después" de la corrección reproducen
dígito a dígito con el comando exacto del README. La corrección hace exactamente lo que pedía el
brief (diagnosticar, corregir sin bajar el umbral ni tocar `common/`, proponer sin aplicar el
ajuste al generador) incluso cuando el resultado es "no alcanza" — es honesta, no esconde que el
problema de negocio sigue sin resolverse.

## Qué reproduje

- `PYTHONPATH=/home/user/wt/cese python3 -m pytest afp_lab/riesgo_cese_cotizacion/tests -q` →
  **29 passed en 23,40 s** (README: "29 en total, subieron de 25, ~19 s" — coincide, diferencia
  de tiempo atribuible a la máquina compartida).
- `git log --oneline -3 -- afp_lab/riesgo_cese_cotizacion` → un único commit de esta corrección
  (`099e22c`, "diagnostico de la caida del AUC de reingreso y features del ultimo empleo").
  `git show --stat 099e22c`: solo tocó `features.py` (+42/−3), `tests/test_riesgo_cese.py`
  (+69), `README.md` (+110) — **`model.py` y `run.py` no se tocaron**, así que no hay dónde
  hubiera podido bajarse un umbral o desactivarse un test de ese commit.
- `PYTHONPATH=/home/user/wt/cese python3 -m afp_lab.riesgo_cese_cotizacion.run --n-afiliados
  20000 --perfil ambos --seed 42` (comando exacto del README §12) → **69,9 s** (sistema) + **78,1
  s** (cuprum) = 148 s totales, cada perfil < 3 min como exige el encargo. Cifras reproducidas:
  - AUC reingreso **sistema**: 0,608108 → README `auc_reingreso_después_fix` sistema/seed42
    = **0,6081** — coincide.
  - AUC reingreso **cuprum**: 0,626123 → README **0,6261** — coincide.
  - Prevalencia test: sistema 23,08 % (README "23,1 %"), cuprum 37,93 % (README "37,9 %") —
    coincide.
  - AUC cese y3/y6 sistema: 0,886969 / 0,877994 (README "0,8870 / 0,8780") — coincide.
  - AUC cese y3/y6 cuprum: 0,889064 / 0,873204 (README "0,8891 / 0,8732") — coincide.

  Los 6 números "después" de la tabla del §12 que verifiqué reproducen dígito a dígito con el
  comando de referencia.

## Diagnóstico: evidencia de código, no solo narrativa

- **Bug de rubro/tamaño constante**: confirmado leyendo `features.py` — `rubro`/`tamano_empleador`
  se unen por `empleador_principal` (`dim_emp`, línea ~460), columna que es NaN por definición
  para toda fila con `pop_reingreso=1` (`rim_l1==0`, línea 582) y cae a `"SIN_EMPLEADOR"` tras el
  `fillna`. El fix agrega `empleador_ref_hist` (`_features_historia_rim`, `df["empleador_ref_hist"]
  = df.groupby("afiliado_id", sort=False)["empleador_principal"].ffill()`) y una segunda unión
  (`dim_emp_hist`) por esa columna.
- **Mecánica "memoryless" de `p_hire`**: confirmada en
  `afp_lab/rim_nowcasting/synthetic.py:360` — `p_hire_base = 0.16 * factor_contratacion * (1 - 4
  * (des - 0.08))`, sin ningún término que dependa de la duración del cese (`racha_cero`) — coincide
  exactamente con la descripción del README ("memoryless... solo varía por edad>55, rubro
  latente preferido y desempleo macro").
- **Calibración de `PERFILES`**: confirmada en `afp_lab/common/synth_base.py:31-49` —
  `factor_contratacion` de `sistema` 1,0→0,20, con el comentario explícito "calibrados en la
  ronda 3 ... Ficha Estadística Previsional N165" — coincide con lo que dice el README sobre el
  origen del problema.

## Sin leakage en las features nuevas

- `empleador_ref_hist` se calcula sobre `df` dentro de `_features_historia_rim`, cuya grilla
  temporal (`_grid_ventana`) está acotada a `corte_rim = corte-2` — el `ffill` nunca puede ver
  `corte-1` ni meses posteriores porque esas filas no existen en `df`.
- `rubro_ultimo_empleo`/`tamano_ultimo_empleo` se unen desde la tabla `empleadores`
  (`afp_lab/rim_nowcasting/synthetic.py:100-136`), que es **estática por empleador** (generada una
  sola vez, sin dimensión temporal) — el join no puede introducir información de meses futuros.
- `test_leakage_ultimo_empleo_no_usa_rim_futura` (nuevo, pasa): fuerza `rim=999_999_999` en
  `cotizaciones_conocidas` para `periodo > CORTE` y confirma con
  `pd.testing.assert_frame_equal` que las 3 columnas nuevas no cambian.
- `test_rubro_ultimo_empleo_coincide_con_vigente_si_cotiza_hoy` y
  `test_rubro_ultimo_empleo_disponible_en_poblacion_reingreso` (nuevos, pasan): confirman la
  semántica en las dos poblaciones (`pop_cese` y `pop_reingreso`).

## Umbral y tests: nada desactivado ni relajado

- No existe ningún `assert` de umbral 0,70 en el código de tests — es una referencia de negocio
  del brief/README, no una aserción que pudiera haberse relajado en el código.
- El diff de esta corrección no tocó `model.py` ni `run.py` (donde viviría cualquier lógica de
  decisión/umbral) — solo `features.py` (features nuevas), tests (4 nuevos) y README.
- 29 = 25 (previos, intactos) + 4 nuevos, verificado corriendo la suite completa (29 passed).
- El único `pytest.skip` presente en el archivo de tests es condicional y anterior a esta
  corrección (`test_tipo_contrato_no_describe_relacion_terminada`, para un caso de borde sin
  aviso AFC de INICIO) — no afecta ninguno de los tests de esta ronda.

## Honestidad del diagnóstico final

El README concluye que la corrección "ayuda poco" (+0,003 a +0,008 AUC, y −0,005 en una
combinación) y que el techo real es la dinámica del generador, no un vacío de features — que es
exactamente lo que pedía el punto 2 del brief ("si el problema es que la dinámica del generador
dejó el reingreso casi impredecible, decirlo con evidencia... sin aplicarlo"). La propuesta de
ajuste al generador (dar a `p_hire` una dependencia de la duración del cese) es coherente con la
evidencia de AUC por tramo de `racha_cero` que reporta el propio diagnóstico, y no se tocó
`afp_lab/common/`, respetando la restricción del brief.

## Bloqueantes

Ninguno.

## Menores

1. El "antes" de la tabla (AUC 0,6053 / 0,5931 / 0,6186 / 0,6218) se generó con un script ad-hoc
   de monkeypatch de `PERFILES` que **no quedó en el repo** (el README lo reconoce: "no se guardó
   como test... diagnóstico puntual"). No pude reproducirlo de forma independiente; solo verificar
   que el mecanismo que describe (valores antiguos de `factor_contratacion`/`p_activo_inicial`) es
   consistente con el código actual de `synth_base.py`. Sugerencia: guardar ese script (aunque sea
   como test marcado `skip` o en una carpeta de diagnósticos) para que el "antes" sea auditable en
   la próxima ronda sin depender de la narrativa del README.
2. El README menciona que `rubro_ultimo_empleo` "queda en el top-4 de importancia por ganancia
   del modelo de reingreso corregido" pero no publica la tabla de importancias antes/después
   completa en esta sección — una tabla corta habría hecho el punto 3 del diagnóstico (el bug de
   rubro constante) verificable sin tener que leer `features.py` directamente.
3. Con el AUC de reingreso en 0,608 (sistema) / 0,626 (cuprum), todavía bajo el umbral de negocio
   de 0,70, la AFP sigue sin un modelo de reingreso utilizable para campañas con la densidad
   actual del generador. El README es claro sobre esto, pero valdría la pena que el gerente tome
   una decisión explícita (pausar el uso del modelo de reingreso en producción hasta ajustar el
   generador en una próxima ronda) en vez de dejar el proyecto "corregido" de forma ambigua sobre
   si es o no utilizable hoy.

## Veredicto

**APROBAR.** El diagnóstico está sustentado con evidencia de código verificable (fórmula de
`p_hire_base`, calibración de `PERFILES`, construcción de `empleador_principal`), las features
nuevas son as-of y pasan un test de leakage dedicado además de la inspección manual, no se bajó
ningún umbral ni se desactivó ningún test (el diff ni siquiera tocó los archivos donde viviría esa
lógica), y las cifras "después" reproducen dígito a dígito con el comando de referencia del
README. Los tres menores no son bloqueantes: son sobre auditabilidad del "antes" (script no
guardado), completitud de la evidencia de importancias, y una decisión de negocio pendiente sobre
si usar o no el modelo de reingreso mientras el generador no se ajuste.
