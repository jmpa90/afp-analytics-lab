# 03 · Propensión a ahorro previsional voluntario (APV y cotización voluntaria)

Modelo *hurdle* que estima, para cada afiliado y cada `periodo_corte`, **la probabilidad de hacer
un aporte voluntario en los próximos 6 meses** y **cuánto aportará**, para priorizar contactos de
asesoría y proyectar el flujo de ahorro voluntario.

```bash
python -m afp_lab.propension_apv.run --n-afiliados 20000      # corrida de referencia, ~40 s
python -m pytest afp_lab/propension_apv/tests -q              # 20 tests, ~9 s
```

---

## 1. Pregunta de negocio y decisión que apoya

*¿A quién llama el ejecutivo este mes para abrir un APV, y cuánto ahorro voluntario vamos a captar?*

La campaña de ahorro voluntario tiene capacidad limitada (contactos de ejecutivo y asesor son
caros). El modelo entrega dos salidas por afiliado:

| Salida | Uso |
|---|---|
| `prob_aporte` | probabilidad de que aporte al menos una vez en `corte+1 … corte+6`; se agrega para **proyectar el flujo** de APV del semestre |
| `monto_esperado = prob_aporte × monto_condicional` | **ranking de contacto**: es el CLP esperado que se captura por llamar a esa persona |

La alternativa que hoy se usa en la práctica —ordenar la cartera por renta— sirve de referencia
en la métrica de negocio.

---

## 2. Datos sintéticos: tablas, mecanismos y parámetros

Todo parte de `afp_lab.common.synth_base.generar_universo` (afiliados, empleadores, panel laboral
mensual y macro); **no** se redefinen afiliados, empleadores ni panel. `synthetic.py` agrega las
tablas propias del problema, coherentes con la historia laboral simulada.

Ventana simulada **202101–202609** (69 meses). Los 12 primeros meses son *burn-in*: existen sólo
para que el primer corte (202301) ya tenga 24 meses de historial de aportes. Nada real, semilla
fija (`seed=42`), generación determinista (hay test de determinismo).

### 2.1 Tablas (corrida de referencia, 20.000 afiliados)

| Tabla | Grano | Filas | Contenido |
|---|---|---|---|
| `afiliados` / `empleadores` / `panel_mensual` / `macro` | — | 20.000 / 500 / 1.312.097 / 69 | del universo común |
| `apvc_empleador` | empleador × mes | 34.500 | `ofrece_apvc`, `pct_match` |
| `campanas` | evento | 72.373 | afiliado, mes, canal, respuesta (`ABRIO`/`IGNORO`) |
| `aportes_voluntarios` | evento | 74.746 | afiliado, mes, tipo, monto, canal, periodicidad |
| `saldos` | afiliado × mes | 1.380.000 | `saldo_obligatorio`, `saldo_voluntario` |
| `tributario` | afiliado × año | 100.000 | `renta_anual`, `tramo_igc`, `tasa_marginal_igc` |

Las columnas latentes (`lat_*`, `p_*` de empleadores) se usan sólo para generar la dinámica y
**no se devuelven** en el diccionario de tablas: el modelo no puede verlas ni por accidente (hay
test).

### 2.2 Mecanismos y parámetros

**APV colectivo del empleador** (`P_APVC_TAMANO`, `MULT_APVC_RUBRO`). Probabilidad de ofrecer APVC
por tamaño: MICRO 2 %, PEQUEÑA 8 %, MEDIANA 28 %, GRANDE 55 %; multiplicada por rubro (finanzas
1,9; minería 1,8; administración pública 1,5; agricultura 0,35; casa particular 0,20). Resultado:
**10,2 % de los empleadores** ofrece APVC al final de la ventana (46 % de las grandes, 0,9 % de las
micro), con *match* medio de **1,14 % de la renta** y 30 % de los planes sin *match*. El 35 % de
los que ofrecen lo lanza durante la ventana (adopción escalonada, para que la feature
`meses_con_apvc` tenga variación). *Razonable para Chile*: el APVC es un producto marginal
(decenas de miles de trabajadores) muy concentrado en grandes empresas formales de banca, minería
y sector público.

**Campañas** (`P_CAMPANA_BASE = 5,5 %` mensual, focalizado por renta y por APVC). Salen ~1.050
contactos por mes en la cartera de 20.000. Canales: email 45 %, app 30 %, ejecutivo 15 %, asesor
10 %; tasa de apertura por canal 22 / 34 / 62 / 55 %, aumentada por la afinidad latente al ahorro
(así la *respuesta*, no sólo el contacto, es señal). Resultado: **35 % de aperturas**.

**Aportes voluntarios.** Cada mes se evalúa un *score* logit por afiliado (`ParametrosAPV`) y de él
salen dos procesos:

* **Aporte mensual por planilla** (`intercepto_mensual = −9,15`): hazard de *apertura* del plan;
  una vez abierto se aporta todos los meses hasta que el afiliado deja de tener ingreso, cambia de
  empleador (45 % de abandono) o abandona espontáneamente (3 % mensual).
* **Aporte esporádico** (`intercepto_esporadico = −7,70`) con estacionalidad
  `{marzo +1,00; abril +0,70; diciembre +0,85; enero +0,20}` en el logit.

Coeficientes del score (todos en `ParametrosAPV`, escala logit):

| Driver | Coef. | Por qué |
|---|---|---|
| `log(RIM / IMM)` | **+0,90** | el ahorro voluntario es fuertemente creciente en renta |
| tramo del IGC (0–7) | +0,30 | beneficio tributario del régimen B = tasa marginal |
| densidad de cotización 12 m (centrada en 0,85) | +1,20 | sólo ahorra quien tiene ingreso estable |
| meses con el mismo empleador (tope 60) | +0,012/mes | estabilidad laboral |
| edad 35–55 / <30 / >60 | +0,40 / −0,25 / −0,70 | etapa de acumulación; cerca de la pensión el APV pierde atractivo |
| educación universitaria / técnica / básica | +0,45 / +0,20 / −0,30 | educación financiera |
| empleador ofrece APVC | **+1,10** | efecto grande (descuento por planilla + match) |
| *match* del empleador (por punto %) | +0,25 | |
| campaña en los últimos 3 meses | +0,20 | efecto pequeño, como en la realidad |
| contacto de ejecutivo/asesor en 6 m | +0,18 | el canal humano convierte más |
| algún aporte en los 24 meses previos | +0,70 | historial de aportes esporádicos |
| marzo–abril en finanzas/minería | +0,90 | bonos anuales |
| mes con aguinaldo imponible | +0,45 | septiembre y diciembre |
| renta < 1,15 IMM | −0,80 | no hay capacidad de ahorro |
| licencia larga reciente | −0,60 | caída de ingreso |
| independiente | −0,35 | ahorra menos por la AFP |
| **afinidad latente N(0,1)** | **+0,90** | **ruido no explicable**: fija el techo de AUC |
| cesante / retirado | prob. = 0 | un cesante no hace APV |

**Montos.** Mensual: log-normal con mediana `3,5 % de la RIM` (heterogeneidad log-normal σ 0,40
por afiliado). Esporádico: mediana `0,30 × RIM` en marzo, `0,25` en abril, `0,22` en diciembre y
`0,15` el resto del año, con σ 0,55. Ambos ×`(1 + 0,10 × tramo IGC)` y ×1,25 si es régimen B;
piso 15.000 CLP y **tope legal de 50 UF mensuales**. Mediana observada del aporte: **99.553 CLP**.
Tipo: régimen B con probabilidad `0,12 + 0,20 × tramo` (el régimen B conviene a mayor tasa
marginal), depósito convenido sólo si el empleador ofrece APVC. Mezcla resultante: régimen A
52,3 %, régimen B 31,8 %, cotización voluntaria 12,7 %, depósito convenido 3,2 %; canal descuento
por planilla 69,6 %, transferencia 20,8 %, app 9,5 %; 78,6 % de los aportes son mensuales.

**Saldos.** `saldo_obligatorio` = 10 % de la RIM capitalizado al 0,45 % mensual (~5,5 % anual),
sobre un saldo inicial estimado por edad y primera RIM. `saldo_voluntario` = aportes capitalizados
al 0,50 % mensual **partiendo de un stock inicial no nulo**: los aportantes del inicio y un 10 %
de ex-aportantes traen saldo acumulado. Sin ese stock inicial el saldo voluntario tendría una
tendencia creciente artificial que rompe la comparabilidad entre cortes (fue un hallazgo de la
primera iteración: el modelo aprendía el nivel del saldo en vez de su significado).

**Tributario.** `renta_anual` = suma de RIM del año calendario; `tramo_igc` con los tramos legales
del art. 52 de la Ley de la Renta en UTA (13,5 / 30 / 50 / 70 / 90 / 120 / 310 UTA) y UTM anual
aproximada por año. Distribución 2025: **61,6 % exento, 26,2 % tramo 1 (4 %), 9,2 % tramo 2 (8 %),
2,8 % tramo 3, 0,1 % tramo 4+** — coherente con que la mayoría de los asalariados chilenos no paga
IGC.

### 2.3 Tasas resultantes vs. las pedidas en el brief

| Indicador | Brief | Corrida 20.000 (2023 / 2024 / 2025) |
|---|---|---|
| cotizantes con algún aporte en el año | 12–18 % | **16,2 % / 16,1 % / 16,5 %** |
| cotizantes con aporte mensual | 5–8 % | **7,1 % / 7,4 % / 7,5 %** |
| AUC de test | 0,75–0,88 | **0,834** |

---

## 3. Target, población, horizonte y exclusiones

* **Fila** = afiliado × `periodo_corte`, cortes mensuales móviles.
* **`periodo_corte`** es el último mes de remuneración conocido: incorpora el **desfase de 2 meses**
  de la AFP (la cotización del mes *m* se conoce en *m+2*), igual que el proyecto 01. En producción
  `corte = hoy − 2`.
* **Población** (definida sólo con información ≤ corte):
  1. afiliado vigente: tiene registro en el panel del mes de corte y no está `RETIRADO`;
  2. cotizó al menos una vez en los 12 meses previos al corte;
  3. **sin aporte voluntario en los 6 meses previos** (corte−5 … corte). Es un problema de
     *adquisición y reactivación*; el mantenimiento de los aportantes mensuales es otro problema y
     se excluye explícitamente.
* **`y` = 1** si hay al menos un aporte voluntario en `corte+1 … corte+6`; **`monto`** = suma de
  esos aportes.
* **Exclusiones**: cortes cuyos 6 meses de target no están completos (el último corte posible es
  202603) y cortes anteriores a 202301 (falta historia).

Con 20.000 afiliados quedan **~16.000 filas elegibles por corte** (≈ 80 % de la cartera) y una tasa
de aporte de 4,9 % (train) a 5,4 % (test).

---

## 4. Features (60 columnas) y su regla as-of

Regla única: **cada fila usa sólo información con `periodo ≤ periodo_corte`**. `features.py`
convierte las tablas largas en matrices densas `afiliado × periodo` una sola vez y cada corte es
un corte de columnas; por construcción nunca se mira a la derecha del corte.

| Familia | Features | Regla as-of |
|---|---|---|
| **Renta y estabilidad** | `rim_corte`, `rim_ref`, `rim_media_3/12`, `rim_cv_12`, `rim_tendencia_3_12`, `rim_sobre_imm`, `cerca_minimo`, `densidad_6/12`, `meses_desde_ult_cotizacion`, `meses_con_empleador`, `n_empleadores_12`, `meses_cesante_12`, `dias_licencia_12`, `licencia_larga_reciente` | ventanas que terminan en el corte; el corte ya incorpora el desfase de 2 meses |
| **Tributario** | `tramo_igc_12m`, `renta_anual_12m` | renta rodante de los 12 meses hasta el corte |
| | `tramo_igc_declarado` | tabla anual; la Operación Renta del año *Y* llega en abril de *Y+1*, así que al corte sólo se conoce **Y−1 si el mes ≥ abril, Y−2 si no** (hay test) |
| **Calendario** | `mes_corte`, `es_marzo`, `meses_a_marzo`, `marzo_en_ventana`, `diciembre_en_ventana` | función determinista del corte |
| **Cuenta** | `saldo_obligatorio`, `saldo_voluntario`, `saldo_vol_sobre_rim`, `saldo_obl_sobre_rim`, `meses_desde_ult_aporte`, `n_aportes_12m/24m`, `monto_aportes_24m`, `tuvo_aporte_mensual_24m` | saldo del mes de corte; ventanas de aportes que terminan en el corte (por la exclusión de población, `meses_desde_ult_aporte ≥ 7` o 99 = nunca) |
| **Campañas** | `contactos_1m/3m/6m`, `contactos_6m_{email,app,ejecutivo,asesor}`, `aperturas_6m`, `tasa_apertura_6m`, `meses_desde_ult_contacto` | ventanas que terminan en el corte |
| **Empleador** | `ofrece_apvc`, `pct_match_apvc`, `meses_con_apvc`, `frac_apv_empleador`, `emp_n_trabajadores`, `emp_rim_mediana`, `pos_rel_empleador` | agregados calculados **con la nómina del mes de corte**; `frac_apv_empleador` = % de los trabajadores del empleador con algún aporte en los 12 meses hasta el corte |
| **Perfil** | `edad`, `sexo`, `nivel_educacional`, `region`, `meses_hasta_pension` (60 F / 65 M), `meses_desde_afiliacion`, `tipo_afiliado`, `estado_corte` | atributos del maestro y estado del mes de corte |

Categóricas (`tipo_afiliado`, `sexo`, `nivel_educacional`, `rubro`, `tamano_empleador`,
`estado_corte`, `region`) van como `category` de pandas al soporte nativo de LightGBM, con el mismo
dominio en train/valid/test.

### Split temporal y gap

| Partición | Cortes | Filas | Tasa de aporte |
|---|---|---|---|
| train | 202301–202410 (22) | 351.389 | 4,95 % |
| valid | 202411–202501 (3) | 48.750 | 6,41 % |
| **descartados por el gap** | 202502–202512 (11) | — | — |
| test | 202601–202603 (3) | 49.003 | 5,43 % |

Se aplica la lectura estricta de CONVENCIONES §3: el **último mes de target de entrenamiento**
(202507) queda **6 meses antes del primer corte de test** (202601). Cuesta 11 cortes de datos, pero
garantiza que ningún mes usado para etiquetar el entrenamiento toque la ventana de test.

---

## 5. Modelo y baselines

**Hurdle en dos etapas** (`model.py`), porque el 95 % de las filas tiene monto 0:

1. `LGBMClassifier` binario → `prob_aporte`. 700 árboles, `num_leaves=48`, `lr=0,05`,
   `min_child_samples=120`, `n_jobs=2`, early stopping (60) contra validación.
2. `LGBMRegressor` sobre `log(monto)` en las filas con aporte → `monto_condicional`, corregido por
   un factor de **smearing** `s = Σ monto real / Σ exp(ŷ)` calculado en entrenamiento
   (`s = 1,210`), porque la pérdida L2 sobre logs estima una mediana geométrica.
3. `monto_esperado = prob_aporte × monto_condicional`.

**Calibración**: regresión isotónica ajustada **en validación** (nunca en test). Se decide si
aplicarla comparando, *dentro de validación y con 3 folds fuera de muestra*, el Brier calibrado
contra el crudo (0,04221 vs 0,04231 → se aplica).

**Baselines obligatorios**:

* `tasa_base` (trivial): probabilidad constante = tasa de aporte del entrenamiento; monto esperado
  constante = tasa × monto medio de los aportantes.
* `regla_negocio` (reglas simples, la del brief): `(renta > 2 M y empleador con APVC) o aporte en
  24 m`, con las tasas y montos empíricos de entrenamiento de cada grupo como probabilidad y monto.
* `renta` (referencia comercial): ordenar por `rim_ref`; sólo entra en la métrica de negocio.
* `aleatorio`: piso de la métrica de negocio.

---

## 6. Resultados (test = 202601–202603, 49.003 filas, 2.660 aportantes)

### 6.1 Clasificación

| Predictor | AUC | Average precision | Brier | Lift decil 1 | KS |
|---|---|---|---|---|---|
| **Modelo** | **0,834** | **0,278** | 0,046 | **5,03** | **0,520** |
| Regla de negocio | 0,699 | 0,133 | 0,048 | 3,72 | 0,398 |
| Tasa base | 0,500 | 0,054 | 0,051 | 1,09 | 0,066 |

Tasa base del test: 5,43 %. El Brier del baseline trivial es menor que el del modelo por
construcción (predecir la tasa base minimiza el Brier cuando no se ordena); lo relevante es que el
modelo gana en AUC, AP, lift y KS, y que su probabilidad está calibrada (§6.3).

### 6.2 Monto

| Predictor | Vista | MAE | RMSE | WAPE | Sesgo agregado |
|---|---|---|---|---|---|
| **Modelo** | condicional en aportantes | 411.647 | 601.275 | **0,552** | −2,2 % |
| Regla de negocio | condicional en aportantes | 485.513 | 694.996 | 0,651 | −18,8 % |
| Tasa base | condicional en aportantes | 506.744 | 720.558 | 0,679 | −19,8 % |
| **Modelo** | esperado en toda la población | 63.662 | 222.434 | 1,572 | +16,6 % |
| Regla de negocio | esperado en toda la población | 62.379 | 228.810 | 1,541 | −25,4 % |

El WAPE > 1 de la vista *esperado en todos* es esperable: se predice una variable con 95 % de ceros
y cola larga; lo que importa ahí es el **sesgo agregado** (la proyección de flujo), donde el modelo
(+16,6 %) es muy superior a los baselines (−25 % / −27 %).

Monto condicional por tramo de renta (aportantes de test): WAPE 0,59 / 0,56 / 0,58 / 0,57 / 0,51 de
menor a mayor tramo; sesgo entre −23 % (tramo bajo) y +4 % (tramo alto).

### 6.3 Calibración

| Versión | Brier | ECE | Sesgo agregado |
|---|---|---|---|
| Sin calibrar | 0,0451 | 0,0125 | +7,8 % |
| Isotónica (la que se usa) | 0,0462 | 0,0155 | +13,8 % |

La curva por deciles es casi perfecta en los 9 primeros deciles (|dif| ≤ 0,9 pp) y **sobreestima en
el decil 10** (0,380 predicho vs 0,272 real). Causa: el gap de 12 meses entre validación y test —
la isotónica se ajusta al nivel de la ventana de validación, que tiene una tasa base más alta
(6,41 % vs 5,43 %). En producción, con reentrenamiento mensual, el desfase valid→scoring es de 1
mes y no de 12, así que este efecto se reduce mucho; aun así conviene **reajustar la isotónica cada
mes** con el corte maduro más reciente.

### 6.4 Por segmento (AUC; el modelo gana en todos los segmentos con ≥ 5 % de las filas)

| Tramo de renta | % filas | Modelo | Regla |
|---|---|---|---|
| hasta 600k | 19,0 % | 0,816 | 0,597 |
| 600k–1M | 37,6 % | 0,761 | 0,597 |
| 1M–1,8M | 26,0 % | 0,795 | 0,667 |
| 1,8M–3M | 11,7 % | 0,780 | 0,696 |
| sobre 3M | 5,8 % | 0,776 | 0,683 |

| Tramo de edad | % filas | Modelo | Regla |
|---|---|---|---|
| < 30 | 5,0 % | 0,835 | 0,682 |
| 30–39 | 23,2 % | 0,849 | 0,691 |
| 40–49 | 37,6 % | 0,820 | 0,692 |
| 50–59 | 27,6 % | 0,837 | 0,715 |
| 60+ | 6,6 % | 0,854 | 0,718 |

| Tipo de afiliado | % filas | Modelo | Regla |
|---|---|---|---|
| DEPENDIENTE | 88,7 % | 0,830 | 0,698 |
| INDEPENDIENTE | 8,6 % | 0,765 | 0,621 |
| PENSIONADO_ACTIVO | 0,5 % | 0,987 | 0,979 |
| VOLUNTARIO | 2,2 % | 0,496 | 0,493 |

Por rubro, el modelo gana en los 9 rubros con ≥ 5 % de las filas (AUC 0,73–0,82). Los dos
segmentos donde el modelo no discrimina son **VOLUNTARIO** (AUC 0,50; 2,2 % de las filas y sólo 5
aportantes en test) y **A_AGRICULTURA** (0,61; 3,1 %); ninguno llega al 5 % que exige CONVENCIONES §4, y
en ambos el baseline de reglas tampoco discrimina (0,49 y 0,59).

### 6.5 Importancias (gain, %)

*Clasificador*: `meses_desde_ult_aporte` 7,9 · `renta_anual_12m` 7,1 · `meses_desde_afiliacion` 5,5
· `monto_aportes_24m` 5,4 · `rim_corte` 5,2 · `saldo_obligatorio` 4,5 · `meses_con_empleador` 3,9 ·
`frac_apv_empleador` 3,9 · `edad` 3,7 · `region` 3,5 · `rubro` 3,4 · `meses_hasta_pension` 3,3 ·
`emp_n_trabajadores` 3,2 · `rim_cv_12` 2,9 · `meses_con_apvc` 2,8.

*Regresor de monto*: `rim_media_12` 26,5 · `rim_ref` 7,9 · `rim_media_3` 6,7 · `rim_sobre_imm` 5,1 ·
`rubro` 5,0 · `region` 3,9 · `marzo_en_ventana` 3,5 · `meses_a_marzo` 2,5.

Coincide con el mecanismo simulado: historial de aportes + renta + estabilidad + APVC del empleador
para la probabilidad; nivel de renta y estacionalidad de marzo para el monto. `region` aparece con
importancia moderada aunque **no** entra en la generación del target: es ruido que el árbol
aprovecha (su efecto marginal es pequeño y no daña el AUC de test).

### 6.6 Ejemplos explicados (SHAP del clasificador, `reportes/ejemplos_explicados.csv`)

* **Afiliado 8319, corte 202601** — `prob 0,72`, monto esperado 2,5 M, real: **no aportó**.
  Empujan: aporte hace 12 meses (+0,81), renta anual 41 M (+0,73), RIM al tope (+0,52), 2,8 M
  aportados en 24 m (+0,49), rubro minería (+0,39). Caso típico de "todo indica que sí" en que
  simplemente no ocurrió: es el ruido latente que el modelo no puede ver.
* **Afiliado 2853, corte 202602** — `prob 0,90`, real: **aportó 643.501**. Aporte hace 8 meses
  (+0,70), renta anual 40,9 M (+0,52), RIM al tope (+0,48), saldo voluntario 5,4 M (+0,46) y
  **36 meses con APVC en el empleador** (+0,46).
* **Afiliado 5357, corte 202601** — `prob 0,028`. Trabaja en finanzas (+0,46, el rubro ayuda) pero
  **recién entró al empleador** (0 meses, −0,12) y **nunca aportó** (`meses_desde_ult_aporte = 99`,
  −0,11): renta media y sin historial.
* **Afiliado 6320, corte 202603** — `prob 0,005`. Renta anual 3,8 M (−0,37), sin RIM en el mes de
  corte (−0,25), sin empleador (−0,14), renta en el mínimo (−0,14): cesante de renta baja.

---

## 7. Lectura de negocio

**Contactando al 10 % de la cartera elegible con mayor `monto_esperado` se captura el 68,6 % del
ahorro voluntario del semestre y el 49,1 % de los aportantes.** Ordenar por renta —lo que se hace
hoy— captura 49,8 % del monto y 30,1 % de los aportantes.

| Score de contacto | 10 % contactado | 20 % contactado |
|---|---|---|
| **Modelo (`monto_esperado`)** | **68,6 % del monto** / 49,1 % de aportantes / tasa 26,7 % | **84,4 %** / 68,4 % / 18,6 % |
| Modelo (`prob_aporte`) | 64,0 % / 49,9 % / 27,1 % | 81,9 % / 68,0 % / 18,5 % |
| Renta (práctica actual) | 49,8 % / 30,1 % / 16,3 % | 68,2 % / 47,7 % / 13,0 % |
| Regla de negocio | 48,0 % / 38,5 % / 20,9 % | 66,9 % / 55,5 % / 15,1 % |
| Aleatorio | 8,8 % / 8,8 % / 4,8 % | 19,8 % / 19,4 % / 5,3 % |

En la corrida de referencia eso son **+18,8 puntos de captura de monto al 10 %** sobre el orden por
renta (lift 1,38×), es decir ~373 MM CLP adicionales de ahorro voluntario captado en el semestre
sobre una base de 1.984 MM. Además la **tasa de aporte de los contactados sube de 16,3 % a 26,7 %**,
que es lo que mide el ejecutivo comercial.

Dos lecturas operativas adicionales:

* Ordenar por `monto_esperado` en vez de por `prob_aporte` captura 4,6 pp más de monto al 10 % (y
  algo menos de aportantes): si la meta es **CLP captados**, se usa monto esperado; si la meta es
  **número de APV abiertos**, se usa la probabilidad.
* La probabilidad calibrada permite proyectar el flujo: sumada sobre la cartera elegible da la
  cantidad esperada de nuevos aportantes del semestre con +14 % de sesgo (vs −25 % de los
  baselines).

---

## 8. Decisiones tomadas, supuestos y limitaciones

**Decisiones frente a ambigüedades del brief**

1. **Precedencia de la regla de negocio.** "renta > 2 M y empleador con APVC o aporte en 24 m" se
   implementó con la precedencia literal (`y` antes que `o`): `(renta > 2 M ∧ APVC) ∨ aporte 24 m`.
   Es además la variante más generosa (marca más gente) y por lo tanto el baseline más exigente.
2. **Gap del split.** CONVENCIONES §3 pide "gap de al menos *h* meses entre el último target de
   entrenamiento y el primer corte de test". Se aplicó literalmente (último target 202507, primer
   corte de test 202601), lo que descarta 11 cortes. La lectura laxa (gap entre *cortes*) habría
   dado más datos, pero ésta es la única que garantiza cero solapamiento.
3. **"Sin aporte voluntario en los 6 meses previos"** se interpretó como la ventana cerrada
   `corte−5 … corte`.
4. **Ventana de simulación 202101–202609** en vez de 202201: los 12 meses extra son burn-in para que
   el primer corte ya tenga 24 meses de historial de aportes. Efecto lateral conocido: la tabla
   macro del universo común extrapola hacia atrás con el tope imponible del último año configurado
   (92 UF), así que 2021 tiene un tope algo alto; sólo afecta features de historia larga en los
   cortes más antiguos, nunca al target.
5. **Tramo tributario**: se publican dos features. `tramo_igc_12m` (renta rodante, siempre
   disponible) y `tramo_igc_declarado` (tabla anual con la regla de disponibilidad de la Operación
   Renta: año *Y−1* si el corte es de abril o posterior, *Y−2* si no).
6. **Saldo voluntario inicial no nulo.** En la primera iteración toda la cartera partía con saldo
   voluntario 0 y el saldo crecía monótonamente durante la ventana, lo que hacía que "saldo alto"
   significara cosas distintas en 2023 y en 2026 (y degradaba la calibración del decil superior).
   Se añadió un stock inicial de aportantes y ex-aportantes.
7. **Calibración**: se aplica isotónica sólo si mejora el Brier fuera de muestra dentro de
   validación (3 folds). En esta corrida la mejora en validación fue mínima (0,04221 vs 0,04231) y
   en test la isotónica empeora levemente Brier/ECE: el mecanismo decide con validación, que es lo
   correcto, y el resultado se reporta tal cual.
8. **`n_estimators` del clasificador = 700** fijado a priori. Se verificó (después, como
   diagnóstico) que subir a 2.000 mejora el AUC de validación a 0,914 pero **baja el AUC de test a
   0,823**: con 12 meses de gap, el early stopping contra validación es optimista. Vale la pena
   revisar este hiperparámetro con un esquema de orígenes móviles.

**Supuestos que habría que confirmar con la AFP**

* Los aportes voluntarios se conocen sin desfase (el abono a la cuenta voluntaria es inmediato o de
  m+1), a diferencia de la cotización obligatoria que llega en m+2. Si el APV por planilla llegara
  con el mismo desfase que la cotización, habría que atrasar `meses_desde_ult_aporte` y
  `n_aportes_*` en 2 meses.
* `ofrece_apvc` y `pct_match` son atributos del convenio con el empleador, conocidos por la AFP en
  el mes en que rigen.
* Las respuestas de campaña (apertura de email, click en la app, resultado de la llamada) están
  disponibles el mismo mes.

**Limitaciones**

* Los parámetros (probabilidades de APVC, tasas de aporte, elasticidades, montos) son supuestos
  razonables para Chile, no series oficiales de la Superintendencia de Pensiones.
* El modelo no discrimina en el segmento `VOLUNTARIO` (2,2 % de las filas): en la simulación estos
  afiliados cotizan de forma errática y casi no aportan. Conviene excluirlos de la campaña o
  modelarlos aparte.
* El target es "al menos un aporte en 6 meses"; no distingue entre abrir un plan mensual y hacer un
  depósito único. El `monto_esperado` lo captura implícitamente, pero para la asesoría sería útil un
  tercer modelo de **periodicidad** (mensual vs esporádico).
* No se modela la competencia (APV en bancos, compañías de seguros y AGF), que en Chile concentra
  una parte importante del stock de APV.

---

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.propension_apv.run --n-afiliados 20000        # [--salida DIR --seed 42 --primer-corte 202301]
python -m pytest afp_lab/propension_apv/tests -q
```

Corrida de referencia (20.000 afiliados, 4 núcleos, `n_jobs=2`):

| Etapa | Segundos |
|---|---|
| Generación de las tablas sintéticas | 5,6 |
| Dataset as-of (28 cortes, 449.142 filas) | 7,7 |
| Entrenamiento (clasificador + regresor + calibración) | 20,0 |
| Evaluación, importancias y ejemplos | 2,4 |
| **Total** | **38,7** (límite: 300) |

Tests: **20 tests en ~9 s** (límite: 120). Cubren leakage por truncamiento y por alteración del
futuro en dos cortes, regla as-of del tramo tributario, ausencia de columnas latentes, gap del
split, recálculo a mano de features en 2 casos × 2 cortes, reglas de población, definición del
target, forma del dataset, determinismo, coherencia de la simulación (un cesante no aporta, el
descuento por planilla exige empleador, recursión del saldo voluntario, tasas objetivo del brief),
predicción/baselines/métricas, persistencia y corrida end-to-end.

Salidas en `salidas/propension_apv/` (en `.gitignore`):
`fuentes/*.parquet`, `dataset/{train,valid,test}.parquet`, `modelo/{modelo.pkl,calibracion.json}`,
`predicciones_test.parquet` y `reportes/*.csv` (métricas, segmentos, calibración, negocio,
importancias, ejemplos, aceptación, tiempos).

### Criterios de aceptación del brief

| Criterio | Umbral | Obtenido | ¿Cumple? |
|---|---|---|---|
| AUC de test | ≥ 0,75 | **0,834** | sí |
| Lift decil 1 | ≥ 3 | **5,03** | sí |
| WAPE del monto en aportantes | ≤ 0,6 | **0,552** | sí |
| Captura al 10 % vs orden por renta | > | **68,6 % vs 49,8 %** | sí |
| Runtime con 20.000 afiliados | < 5 min | **39 s** | sí |
| Tests | < 2 min | **9 s** | sí |
| AUC dentro del rango pedido a la simulación | 0,75–0,88 | 0,834 | sí |

---

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/`** (no se tocó nada fuera de `afp_lab/propension_apv/`):

1. `metrics.tabla_metricas_por_segmento` usa `df.groupby(col_segmento)` sin `observed=`, lo que con
   columnas `category` de pandas 2.3 genera un `FutureWarning` y, cuando el segmento es categórico,
   devuelve filas vacías para categorías sin datos. Sugerencia: `df.groupby(col, observed=True)`.
2. Falta una métrica de **captura por decil / top-k** (`captura_monto`, `captura_positivos`,
   `tasa_en_contactados`): la implementamos en `evaluate.metrica_negocio` y sirve igual para el
   churn de traspaso y para el riesgo de cese de cotización. Proponemos subirla a
   `common/metrics.py`.
3. Falta una utilidad de **curva de calibración / ECE** (`curva_calibracion`, `resumen_calibracion`),
   hoy duplicable en cualquier proyecto con salida probabilística.
4. Utilidades de panel denso (`matriz_densa`, ventanas as-of) y `split_temporal` con gap: son
   genéricas para cualquier proyecto de cortes móviles y hoy viven en `propension_apv`.
5. `synth_base` podría exponer la lista de tablas *observables* separada de las latentes (hoy el
   diccionario trae `afiliados_latente` y `empleadores_latente`, que es fácil de usar por error).

**Próximos pasos**

* Modelo de **periodicidad** (mensual vs esporádico) como tercera etapa, para que el ejecutivo sepa
  si ofrecer descuento por planilla o un depósito de marzo.
* **Uplift** en vez de propensión: hoy la campaña es una feature; con un diseño experimental
  (holdout aleatorio de contactos) se podría estimar el efecto incremental del contacto, que es la
  decisión real.
* Reentrenamiento mensual con orígenes móviles y recalibración con el último corte maduro;
  monitoreo del drift del saldo voluntario y de la tasa base.
* Cuantiles (P10/P90) del monto para dar un intervalo a la proyección de flujo.
* Segmentar el modelo para `INDEPENDIENTE` (su ciclo de renta es anual, vía Operación Renta) y
  excluir o tratar aparte a los `VOLUNTARIO`.
