# 03 · Propensión a ahorro previsional voluntario (APV y cotización voluntaria)

Modelo *hurdle* que estima, para cada afiliado y cada `periodo_corte`, **la probabilidad de hacer
un aporte voluntario en los próximos 6 meses** y **cuánto aportará**, para priorizar contactos de
asesoría y proyectar el flujo de ahorro voluntario.

```bash
python -m afp_lab.propension_apv.run --n-afiliados 20000      # corrida de referencia
python -m pytest afp_lab/propension_apv/tests -q              # 25 tests, ~16 s
```

> **Ronda 2.** Esta versión corrige los cuatro bloqueantes y los cinco mayores de la revisión de la
> ronda 1 (`docs/revisiones/ronda_1/propension_apv.md`). Los cambios de protocolo y los números
> antes/después están en **§11**.

---

## 1. Pregunta de negocio y decisión que apoya

*¿A quién llama el ejecutivo este mes para abrir un APV, y cuánto ahorro voluntario vamos a captar?*

La campaña de ahorro voluntario tiene capacidad limitada (contactos de ejecutivo y asesor son
caros). El modelo entrega dos salidas por afiliado:

| Salida | Uso |
|---|---|
| `prob_aporte` | probabilidad de que aporte al menos una vez en `corte+1 … corte+6`; se agrega para **proyectar el flujo** de adquisición de APV del semestre |
| `monto_esperado = prob_aporte × monto_condicional` | **ranking de contacto**: es el CLP esperado que se captura por llamar a esa persona |

La alternativa que hoy se usa en la práctica —ordenar la cartera por renta— sirve de referencia en
la métrica de negocio (§7).

---

## 2. Datos sintéticos: tablas, mecanismos y parámetros

Todo parte de `afp_lab.common.synth_base.generar_universo` (afiliados, empleadores, panel laboral
mensual y macro); **no** se redefinen afiliados, empleadores ni panel. `synthetic.py` agrega las
tablas propias del problema, coherentes con la historia laboral simulada.

Ventana simulada **201901–202609** (93 meses). Los 24 primeros meses son *burn-in*: existen para que
el primer mes de decisión (202101) ya tenga 24 meses de historial de aportes, y para que el stock de
aportantes mensuales llegue a régimen antes del primer corte. Nada real, semilla fija (`seed=42`),
generación determinista (hay test de determinismo).

### 2.1 Tablas (corrida de referencia, 20.000 afiliados)

| Tabla | Grano | Filas | Contenido |
|---|---|---|---|
| `afiliados` / `empleadores` / `panel_mensual` / `macro` | — | 20.000 / 500 / 1.768.096 / 93 | del universo común |
| `apvc_empleador` | empleador × mes | 46.500 | `ofrece_apvc`, `pct_match` |
| `campanas` | evento | 97.369 | afiliado, mes, canal, respuesta (`ABRIO`/`IGNORO`) |
| `aportes_voluntarios` | evento | 92.675 | afiliado, mes, tipo, monto, canal, periodicidad |
| `saldos` | afiliado × mes | 1.860.000 | `saldo_obligatorio`, `saldo_voluntario` |
| `tributario` | afiliado × año | 140.000 | `renta_anual`, `tramo_igc`, `tasa_marginal_igc` |

Las columnas latentes (`lat_*`, `p_*` de empleadores) se usan sólo para generar la dinámica y **no
se devuelven** en el diccionario de tablas: el modelo no puede verlas ni por accidente (hay test).

### 2.2 Mecanismos y parámetros

**APV colectivo del empleador** (`P_APVC_TAMANO`, `MULT_APVC_RUBRO`). Probabilidad de ofrecer APVC
por tamaño: MICRO 2 %, PEQUEÑA 8 %, MEDIANA 28 %, GRANDE 55 %; multiplicada por rubro (finanzas 1,9;
minería 1,8; administración pública 1,5; agricultura 0,35; casa particular 0,20). Resultado:
**10,2 % de los empleadores** ofrece APVC al final de la ventana (46 % de las grandes, 0,9 % de las
micro), con *match* medio de **1,14 % de la renta** y 30 % de los planes sin *match*. El 35 % de los
que ofrecen lo lanza durante la ventana (adopción escalonada, para que `meses_con_apvc` tenga
variación). *Razonable para Chile*: el APVC es un producto marginal y muy concentrado en grandes
empresas formales de banca, minería y sector público.

**Campañas** (`P_CAMPANA_BASE = 5,5 %` mensual, focalizado por renta y por APVC): ~1.047 contactos
por mes en la cartera de 20.000. Canales email 45 %, app 30 %, ejecutivo 15 %, asesor 10 %; tasa de
apertura por canal 22 / 34 / 62 / 55 %, aumentada por la afinidad latente al ahorro (así la
*respuesta*, no sólo el contacto, es señal). Resultado: **35,2 % de aperturas**.

**Aportes voluntarios.** Cada mes se evalúa un *score* logit por afiliado (`ParametrosAPV`) del que
salen dos procesos:

* **Aporte mensual por planilla** (`intercepto_mensual = −9,35`): hazard de *apertura* del plan; una
  vez abierto se aporta todos los meses hasta que el afiliado deja de tener ingreso, cambia de
  empleador (45 % de abandono) o abandona espontáneamente (3 % mensual).
* **Aporte esporádico** (`intercepto_esporadico = −7,95`) con estacionalidad
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

**Montos.** Mensual: log-normal con mediana `3,5 % de la RIM` (heterogeneidad log-normal σ 0,40 por
afiliado). Esporádico: mediana `0,30 × RIM` en marzo, `0,25` en abril, `0,22` en diciembre y `0,15`
el resto del año, con σ 0,55. Ambos ×`(1 + 0,10 × tramo IGC)` y ×1,25 si es régimen B; piso 15.000
CLP y **tope legal de 50 UF mensuales**. Mediana observada del aporte: **101.863 CLP**. Tipo:
régimen B con probabilidad `0,12 + 0,20 × tramo`; depósito convenido sólo si el empleador ofrece
APVC. Mezcla resultante: régimen A 49,8 %, régimen B 35,6 %, cotización voluntaria 11,3 %, depósito
convenido 3,3 %; canal descuento por planilla 69,3 %, transferencia 21,6 %, app 9,1 %; 79,7 % de los
aportes son mensuales.

**Saldos.** `saldo_obligatorio` = 10 % de la RIM capitalizado al 0,45 % mensual (~5,5 % anual) sobre
un saldo inicial estimado por edad y primera RIM. `saldo_voluntario` = aportes capitalizados al
0,50 % mensual **partiendo de un stock inicial no nulo**: los aportantes del inicio y un 10 % de
ex-aportantes traen saldo acumulado (si no, el saldo voluntario tendría una tendencia creciente
artificial que rompe la comparabilidad entre cortes).

**Tributario.** `renta_anual` = suma de RIM del año calendario — que es **renta imponible**, o sea
ya topada al tope imponible, no renta bruta: para las rentas altas el tramo del IGC queda
subestimado, y así está documentada la feature. Tramos legales del art. 52 de la Ley de la Renta en
UTA (13,5 / 30 / 50 / 70 / 90 / 120 / 310) con UTM anual aproximada por año. Distribución 2025:
**60,8 % exento, 26,3 % tramo 1 (4 %), 9,5 % tramo 2 (8 %), 3,2 % tramo 3, 0,1 % tramo 4+**.

### 2.3 Tasas resultantes vs. las pedidas en el brief

| Indicador | Brief | Corrida 20.000 (2023 / 2024 / 2025) |
|---|---|---|
| cotizantes con algún aporte en el año | 12–18 % | **15,0 % / 15,1 % / 15,5 %** |
| cotizantes con aporte mensual | 5–8 % | **6,7 % / 7,0 % / 7,3 %** |
| AUC de test | 0,75–0,88 | **0,836** |

---

## 3. Target, población, horizonte y exclusiones

**Convención de fechas (CONVENCIONES §3, acordada en la ronda 1).** `periodo_corte` es el **mes en
que se toma la decisión**: el mes en que se arma la lista de contacto. Cada fuente se lee con su
propio desfase y el target empieza en `corte+1`, de modo que **nada del target está observado cuando
se arma la lista**.

| Fuente | Último mes leído |
|---|---|
| panel laboral: RIM, estado, empleador, antigüedad, edad, saldo obligatorio | `corte-2` |
| licencias médicas (`dias_licencia`) | `corte-1` |
| tablas operacionales de la AFP: aportes voluntarios, campañas, saldo voluntario | `corte` |
| tributario (tabla anual) | año `Y-1` si el corte es de abril o posterior, `Y-2` si no |

* **Fila** = afiliado × `periodo_corte`, cortes mensuales móviles.
* **Población** (definida sólo con información disponible en el corte):
  1. afiliado vigente: tiene registro en el panel de `corte-2` y no está `RETIRADO`;
  2. cotizó al menos una vez en los 12 meses de panel conocidos (`corte-13 … corte-2`);
  3. **sin aporte voluntario en `corte-5 … corte`**. Es un problema de *adquisición y
     reactivación*; el mantenimiento de los aportantes mensuales es otro problema y se excluye.
* **`y` = 1** si hay al menos un aporte voluntario en `corte+1 … corte+6`; **`monto`** = suma de
  esos aportes.
* **Exclusiones**: cortes cuyos 6 meses de target no están completos (el último es 202603) y cortes
  anteriores a 202101 (falta historia de aportes y de panel).

Con 20.000 afiliados quedan **~16.000 filas elegibles por corte** (≈ 80 % de la cartera) y una tasa
de aporte de 4,9 % (entrenamiento) a 5,0 % (test).

---

## 4. Features (64 columnas) y su regla as-of

Regla única: **cada fila usa sólo información con `periodo ≤` el desfase de su fuente** (tabla de
§3). `features.py` convierte las tablas largas en matrices densas `afiliado × periodo` una sola vez
y cada corte es un corte de columnas; por construcción nunca se mira a la derecha del desfase.

Los montos en CLP se expresan en **unidades de IMM** del mes correspondiente (sufijo `_imm`): el
entrenamiento está 3–4 años antes del test y los niveles nominales crecen fuerte en ese lapso, así
que un umbral nominal aprendido en 2021 no significaría lo mismo en 2026.

| Familia | Features | Regla as-of |
|---|---|---|
| **Renta y estabilidad** | `rim_corte_imm`, `rim_ref_imm`, `rim_media_3/12_imm`, `rim_cv_12`, `rim_tendencia_3_12`, `cerca_minimo`, `densidad_6/12`, `meses_desde_ult_cotizacion` + `sin_cotizacion_24m`, `meses_con_empleador`, `n_empleadores_12`, `meses_cesante_12` | ventanas que terminan en `corte-2` |
| **Licencias** | `dias_licencia_12`, `licencia_larga_reciente` (2 meses) | ventanas que terminan en `corte-1` |
| **Tributario** | `tramo_igc_12m`, `renta_imponible_12m_imm` | renta imponible rodante de 12 meses hasta `corte-2` |
| | `tramo_igc_declarado` + `sin_tramo_declarado` | tabla anual con la regla de la Operación Renta (hay test) |
| **Calendario** | `mes_corte`, `es_marzo`, `meses_a_marzo`, `marzo_en_ventana`, `diciembre_en_ventana` | función determinista del corte |
| **Cuenta (obligatoria)** | `saldo_obligatorio_imm`, `saldo_obl_sobre_rim` | saldo de `corte-2` (se arma con cotizaciones) |
| **Cuenta (voluntaria)** | `saldo_voluntario_imm`, `saldo_vol_sobre_rim`, `meses_desde_ult_aporte` + `nunca_aporto_36m`, `n_aportes_12m/24m`, `monto_aportes_24m_imm`, `tuvo_aporte_mensual_24m` | hasta `corte` (sistema propio). Por la exclusión de población, `meses_desde_ult_aporte ≥ 7` |
| **Campañas** | `contactos_1m/3m/6m`, `contactos_6m_{email,app,ejecutivo,asesor}`, `aperturas_6m`, `tasa_apertura_6m` + `tuvo_contacto_6m`, `meses_desde_ult_contacto` + `sin_contacto_24m` | hasta `corte` |
| **Empleador** | `ofrece_apvc`, `pct_match_apvc`, `meses_con_apvc`, `frac_apv_empleador`, `emp_n_trabajadores`, `emp_rim_mediana_imm`, `pos_rel_empleador` | agregados con la nómina de `corte-2`; `frac_apv_empleador` usa los aportes hasta `corte` de esos trabajadores |
| **Perfil** | `edad`, `meses_hasta_pension`, `meses_desde_afiliacion`, `tipo_afiliado`, `sexo`, `nivel_educacional`, `region`, `estado_corte` | maestro y estado de `corte-2` |

`rim_ref_clp` e `imm_corte` viajan en el dataset como **columnas de escala** (el regresor las usa
para des-normalizar y los reportes para segmentar por tramo de renta) pero están excluidas del
conjunto de features; hay test.

Detalles de la ronda 2: las variables "meses desde …" llevan un **flag separado** para no confundir
"hace mucho" con "nunca" (antes se codificaba 99 dentro de una variable continua);
`n_empleadores_12` cuenta empleadores **distintos**, no transiciones; `tasa_apertura_6m` vale 0 con
`tuvo_contacto_6m = 0` cuando no hubo contacto.

### Split temporal: gap entre **todas** las particiones y orígenes móviles

CONVENCIONES §3 exige el gap de `h = 6` meses entre cada par de particiones, no sólo antes de test:
`corte_max(anterior) + h + h ≤ corte_min(posterior)`. La evaluación usa **orígenes móviles**
(reentrenamiento trimestral, como en producción), con 12 cortes de test y 12 de validación:

| Origen | train | valid | test | filas train | tasa train / valid / test |
|---|---|---|---|---|---|
| 202504 | 202101–202205 (17) | 202305–202404 (12) | 202504–202506 | 269.597 | 4,77 % / 4,66 % / 3,60 % |
| 202507 | 202101–202208 (20) | 202308–202407 (12) | 202507–202509 | 317.292 | 4,69 % / 4,66 % / 4,63 % |
| 202510 | 202101–202211 (23) | 202311–202410 (12) | 202510–202512 | 366.038 | 4,88 % / 4,60 % / 6,48 % |
| 202601 | 202101–202302 (26) | 202402–202501 (12) | 202601–202603 | 414.241 | 4,92 % / 4,64 % / 5,21 % |

Total de test: **192.557 filas, 12 cortes, 9.612 aportantes**. La ventana de entrenamiento es
expansiva y ningún mes de target de train o valid toca el bloque de test de su origen.

---

## 5. Modelo y baselines

**Hurdle en dos etapas** (`model.py`):

1. `LGBMClassifier` binario → `prob_aporte`. `num_leaves=16`, `min_child_samples=400`,
   `lambda_l2=10`, `lr=0,05`, `feature_fraction=0,70`, `bagging=0,70`, `n_jobs=2`.
2. `LGBMRegressor` sobre **`log(monto / rim_ref)`** (no sobre `log(monto)`) en las filas con aporte,
   winsorizado a [−4, 4]. Se modela el *ratio* para que el modelo aprenda "cuántos sueldos ahorra" y
   no un monto nominal de 2021. `monto_condicional = rim_ref × exp(ŷ) × s`.
3. `monto_esperado = prob_aporte × monto_condicional`.

**Todo lo que se elige, se elige con validación (con gap), nunca con test:**

| Decisión | Cómo se toma | Resultado en la corrida |
|---|---|---|
| número de árboles | early stopping (paciencia 100) contra el `valid` del origen, con tope 3.000 / 2.000 | **93 / 120 / 181 / 155** árboles (clasificador) y 65 / 85 / 67 / 77 (regresor): el tope nunca se toca |
| factor de *smearing* | `Σ monto / Σ (rim_ref · exp(ŷ))` **en validación** | 1,323 / 1,292 / 1,282 / 1,271 |
| ¿calibrar? | isotónica evaluada fuera de muestra con 3 folds **dentro de validación**; se aplica sólo si baja el Brier ≥ 1 % | mejoras de 0,0002 a 0,0007 (y −0,00002 en un origen) → **no se calibra en ningún origen** |

**Baselines obligatorios**:

* `tasa_base` (trivial): probabilidad constante = tasa de aporte del `train` de cada origen; monto
  esperado constante = tasa × monto medio de los aportantes de ese `train`.
* `regla_negocio` (reglas simples, la del brief): `(renta > 2 M y empleador con APVC) o aporte en
  24 m`, con las tasas y montos empíricos de entrenamiento de cada grupo.
* `renta`: ordenar por `rim_ref`; entra sólo en la métrica de negocio.
* `aleatorio`: piso de la métrica de negocio.

### Diagnóstico de memorización por afiliado

Un mismo afiliado aparece en muchos cortes; si el modelo lo memoriza, el AUC se infla sin aprender
el mecanismo. Con la regularización de la ronda 2:

| Evaluación | AUC |
|---|---|
| train (en muestra) | 0,853 |
| **CV agrupada por afiliado** (3 folds, `GroupKFold` dentro de train) | **0,826** |
| validación temporal con gap | 0,831 |

La brecha train → CV agrupada es de 2,7 puntos y la CV agrupada coincide con la validación
temporal: el modelo generaliza a afiliados que no vio.

---

## 6. Resultados (test = 12 cortes, 202504–202603; 192.557 filas, 9.612 aportantes, tasa base 4,99 %)

### 6.1 Clasificación

| Predictor | AUC | Average precision | Brier | Lift decil 1 | KS |
|---|---|---|---|---|---|
| **Modelo** | **0,836** | **0,297** | **0,0405** | **5,13** | **0,517** |
| Regla de negocio | 0,709 | 0,141 | 0,0442 | 4,05 | 0,399 |
| Tasa base | 0,530 | 0,054 | 0,0474 | 1,08 | 0,091 |

El baseline `tasa_base` no da exactamente AUC 0,5 porque cada origen usa la tasa de **su** train:
esa constante difiere entre bloques de test y ordena débilmente entre cortes. Es el baseline
correcto (es lo que haría la AFP sin modelo) y su AUC de 0,53 mide justamente ese efecto.

**Por corte** (`reportes/metricas_por_corte.csv`) la tasa base va de 3,3 % (abril) a 6,9 %
(octubre) y el modelo es estable en todos:

| Corte | 202504 | 202505 | 202506 | 202507 | 202508 | 202509 | 202510 | 202511 | 202512 | 202601 | 202602 | 202603 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tasa base | 0,033 | 0,034 | 0,041 | 0,042 | 0,041 | 0,056 | 0,069 | 0,068 | 0,057 | 0,056 | 0,058 | 0,043 |
| AUC | 0,804 | 0,798 | 0,803 | 0,814 | 0,821 | 0,836 | 0,847 | 0,852 | 0,851 | 0,847 | 0,845 | 0,830 |
| lift decil 1 | 4,78 | 4,68 | 4,55 | 4,65 | 4,82 | 5,08 | 5,07 | 5,17 | 5,34 | 5,30 | 5,14 | 4,85 |
| WAPE monto | 0,568 | 0,609 | 0,591 | 0,551 | 0,536 | 0,556 | 0,551 | 0,548 | 0,552 | 0,530 | 0,531 | 0,627 |
| sesgo monto esperado | +6 % | +6 % | +12 % | +9 % | +12 % | +4 % | −9 % | −6 % | +3 % | −3 % | −2 % | +16 % |

AUC mínimo 0,798 y máximo 0,852; sesgo del monto entre −9 % y +16 % mes a mes, sin deriva.

### 6.2 Monto

| Predictor | Vista | MAE | RMSE | WAPE | Sesgo agregado |
|---|---|---|---|---|---|
| **Modelo** | condicional en aportantes | 373.235 | 544.823 | **0,555** | +2,2 % |
| Regla de negocio | condicional en aportantes | 452.829 | 662.670 | 0,673 | −19,0 % |
| Tasa base | condicional en aportantes | 462.548 | 682.124 | 0,688 | −21,5 % |
| **Modelo** | esperado en toda la población | 50.650 | 187.266 | 1,509 | **+1,0 %** |
| Regla de negocio | esperado en toda la población | 53.791 | 203.622 | 1,602 | −19,6 % |

El WAPE > 1 de la vista *esperado en todos* es inevitable: se predice una variable con 95 % de ceros
y cola larga. Ahí lo que importa es el sesgo, y **no basta el agregado**: la ronda 1 tenía +16,6 %
agregado con los deciles 1–9 subestimados entre −50 % y −100 % y el decil 10 sobreestimado +102 %.
Ahora:

| Decil de monto esperado | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| % del monto real | 0,2 % | 0,4 % | 0,9 % | 1,0 % | 1,4 % | 2,3 % | 3,1 % | 6,7 % | 14,2 % | **69,8 %** |
| sesgo | +37 % | +14 % | −28 % | −10 % | −8 % | −12 % | +5 % | −9 % | −3 % | **+3,5 %** |

Los errores relativos grandes quedan en deciles que juntos pesan menos del 2 % del monto; el decil
que concentra el 70 % del flujo está a +3,5 %. Los deciles 1–2 sobreestiman porque el monto
condicional tiene un piso (el aporte mínimo) que no puede bajar tanto como la probabilidad.

Monto condicional por tramo de renta (aportantes de test): WAPE 0,67 / 0,58 / 0,59 / 0,56 / 0,51 de
menor a mayor tramo; sesgo −16 % (tramo bajo) a +9 % (tramo alto).

### 6.3 Calibración

La isotónica **no se aplica**: en los cuatro orígenes la mejora fuera de muestra dentro de
validación fue de 0,0002 a 0,0007 (0,02 %–0,07 % del Brier), muy por debajo del umbral de 1 %, y en
un origen fue negativa. Se usa la probabilidad cruda del clasificador.

| Versión | Brier | ECE | Sesgo agregado |
|---|---|---|---|
| **Probabilidad usada (sin calibrar)** | **0,04047** | **0,0018** | **−0,6 %** |
| Isotónica del último origen (sólo referencia) | 0,04055 | 0,0032 | −5,4 % |

La curva por deciles queda dentro de ±0,5 pp en los 10 deciles, incluido el décimo (0,251 predicho
vs 0,256 real). **Corrección del diagnóstico de la ronda 1**: allí se atribuyó la sobreestimación
del decil 10 (+40 %) a "la tasa base de la ventana de validación". Era falso. La causa era que train
y valid compartían 5 meses de target, de modo que la isotónica se ajustaba sobre probabilidades
infladas por memorización. Con el gap entre train y valid el problema desaparece **sin calibrar**.

### 6.4 Por segmento (AUC; el modelo gana en todos los segmentos con ≥ 5 % de las filas)

| Tramo de renta | % filas | Modelo | Regla |
|---|---|---|---|
| hasta 600k | 20,1 % | 0,807 | 0,613 |
| 600k–1M | 37,0 % | 0,776 | 0,634 |
| 1M–1,8M | 25,6 % | 0,783 | 0,663 |
| 1,8M–3M | 11,7 % | 0,786 | 0,706 |
| sobre 3M | 5,6 % | 0,778 | 0,698 |

| Tramo de edad | % filas | Modelo | Regla |
|---|---|---|---|
| < 30 | 3,6 % | 0,815 | 0,726 |
| 30–39 | 21,3 % | 0,839 | 0,707 |
| 40–49 | 37,7 % | 0,825 | 0,707 |
| 50–59 | 30,0 % | 0,841 | 0,710 |
| 60+ | 7,3 % | 0,856 | 0,704 |

| Tipo de afiliado | % filas | Modelo | Regla |
|---|---|---|---|
| DEPENDIENTE | 89,1 % | 0,832 | 0,711 |
| INDEPENDIENTE | 8,5 % | 0,760 | 0,554 |
| PENSIONADO_ACTIVO | 0,2 % | — (sin aportantes) | — |
| VOLUNTARIO | 2,2 % | 0,546 | 0,606 |

Por rubro el modelo gana en los 9 rubros con ≥ 5 % de las filas (AUC 0,743–0,801). El único segmento
donde el modelo **pierde** contra la regla es `VOLUNTARIO` (0,546 vs 0,606), con 2,2 % de las filas
— por debajo del 5 % que exige CONVENCIONES §4; en la simulación estos afiliados cotizan de forma
errática y casi no aportan. La recomendación operativa es excluirlos de la campaña.

### 6.5 Importancias (gain, %)

*Clasificador*: `meses_desde_ult_aporte` 17,6 · `renta_imponible_12m_imm` 13,2 · `rim_corte_imm`
10,1 · `frac_apv_empleador` 10,0 · `monto_aportes_24m_imm` 9,8 · `rim_media_3_imm` 4,0 ·
`ofrece_apvc` 3,5 · `rubro` 3,4 · `meses_con_empleador` 3,2 · `region` 2,4 · `nivel_educacional` 2,2
· `meses_con_apvc` 2,1 · `saldo_voluntario_imm` 1,7 · `pct_match_apvc` 1,7 · `edad` 1,5.

*Regresor de monto*: `rubro` 19,3 · `meses_a_marzo` 14,5 · `marzo_en_ventana` 10,9 · `region` 6,2 ·
`rim_ref_imm` 5,6 · `emp_rim_mediana_imm` 5,0 · `rim_tendencia_3_12` 4,3 · `rim_media_12_imm` 3,2.

Coincide con el mecanismo simulado: historial de aportes + renta + APVC del empleador para la
probabilidad; rubro (bonos de marzo en minería y finanzas) y estacionalidad para el monto. `region`
conserva 2,4 % de gain en el clasificador y 6,2 % en el regresor **sin entrar en la generación del
target**: es ruido que los árboles aprovechan. Bajó de 3,5 % / 3,8 % de la ronda 1 con la
regularización, pero sigue presente; quitarla sería lo correcto si no la pidiera el brief, y en
producción conviene monitorearla.

### 6.6 Ejemplos explicados (SHAP del clasificador, `reportes/ejemplos_explicados.csv`)

* **Afiliado 19076, corte 202602** — `prob 0,72`, monto esperado 2,64 M; real: **aportó 5,90 M**.
  Empujan: aporte hace 6 meses (+0,78), 17,4 IMM aportados en 24 m (+0,58), **55 % de los compañeros
  de su empleador con APV** (+0,51), renta imponible anual de 149 IMM (+0,43) y saldo voluntario de
  45 IMM (+0,36). Es el caso de reactivación puro: alguien que ahorraba, dejó de hacerlo hace medio
  año y está en una empresa donde el APV es norma.
* **Afiliado 9039, corte 202511** — `prob 0,67`, real: **aportó 2,57 M**. Aporte hace 19 meses
  (+0,59), 52 % de compañeros con APV (+0,55), renta 79 IMM (+0,41), RIM al tope (+0,38).
* **Afiliado 5500, corte 202601** — `prob 0,51`, real: **aportó 1,52 M**. Aporte hace 7 meses
  (+0,88) y empleador con APVC (+0,52), pero educación básica resta (−0,25): el modelo lo deja en el
  borde y acierta.
* **Afiliado 1746, corte 202505** — `prob 0,017`. Lleva 50 meses con el mismo empleador (+0,26, lo
  único a favor) pero **marzo está a 10 meses** (−0,14), renta imponible anual de 26,8 IMM (baja) y
  **nunca aportó en 36 meses** (−0,07). No aportó.

---

## 7. Lectura de negocio

**Qué es este 100 %.** El target cubre sólo la población de **adquisición y reactivación** (sin
aporte en los 6 meses previos). En los meses de test esa población aporta, en promedio, **539 MM CLP
por ventana de 6 meses**, que son el **38 % del flujo total de ahorro voluntario** de la AFP en esos
meses (1.409 MM por ventana equivalente); el 62 % restante lo aportan los aportantes mensuales
vigentes, que están excluidos por definición del problema y no son objeto de esta campaña.

**Contactando al 10 % de la cartera elegible con mayor `monto_esperado` se captura el 67,8 % de ese
ahorro de adquisición y el 47,4 % de los nuevos aportantes.** Ordenar por renta —lo que se hace
hoy— captura 49,4 % del monto y 30,3 % de los aportantes. Con la convención de fechas corregida,
**nada de ese monto está observado en el momento de armar la lista**.

| Score de contacto | 10 % contactado | 20 % contactado |
|---|---|---|
| **Modelo (`monto_esperado`)** | **67,8 % del monto** / 47,4 % de aportantes / tasa 23,6 % | **82,8 %** / 66,2 % / 16,5 % |
| Modelo (`prob_aporte`) | 65,1 % / 50,1 % / 25,0 % | 81,8 % / 68,1 % / 17,0 % |
| Renta (práctica actual) | 49,4 % / 30,3 % / 15,1 % | 70,8 % / 49,3 % / 12,3 % |
| Regla de negocio | 45,8 % / 38,8 % / 19,4 % | 64,6 % / 54,6 % / 13,6 % |
| Aleatorio | 11,1 % / 10,1 % / 5,1 % | 20,5 % / 19,8 % / 4,9 % |

Son **+18,4 puntos de captura al 10 %** sobre el orden por renta (lift 1,37×): ~99 MM CLP más de
ahorro de adquisición por ventana semestral sobre una base de 539 MM. Y la **tasa de conversión de
los contactados sube de 15,1 % a 23,6 %**, que es lo que mide el ejecutivo comercial.

Tres lecturas operativas:

* Ordenar por `monto_esperado` en vez de por `prob_aporte` captura 2,7 pp más de monto al 10 % pero
  2,7 pp menos de aportantes: si la meta es **CLP captados** se usa monto esperado; si la meta es
  **número de APV abiertos**, la probabilidad.
* La probabilidad está calibrada (ECE 0,0018, sesgo agregado −0,6 %): sumada sobre la cartera
  elegible proyecta el número de nuevos aportantes del semestre con menos de 1 % de error, frente a
  −20 % / −24 % de los baselines.
* El modelo no ordena a los `VOLUNTARIO` (2,2 % de la cartera elegible): conviene sacarlos de la
  lista o tratarlos con otra pieza.

---

## 8. Decisiones tomadas, supuestos y limitaciones

**Decisiones frente a ambigüedades del brief**

1. **Precedencia de la regla de negocio.** "renta > 2 M y empleador con APVC o aporte en 24 m" se
   implementó con la precedencia literal (`y` antes que `o`): `(renta > 2 M ∧ APVC) ∨ aporte 24 m`.
   Es además la variante más generosa, o sea el baseline más exigente.
2. **Gap del split.** Se aplica la lectura estricta de CONVENCIONES §3:
   `corte_max(anterior) + h + h ≤ corte_min(posterior)` para train→valid y para valid→test. Cuesta
   22 cortes de los 63 disponibles y deja el entrenamiento 3–4 años antes del test, pero es lo único
   que garantiza que la validación sirva para decidir.
3. **Orígenes móviles cada 3 meses.** La convención admite "12 cortes u orígenes móviles"; se usan
   las dos cosas: 12 cortes de test cubiertos por 4 modelos con ventana expansiva. Es el análogo de
   un reentrenamiento trimestral y evita que un único modelo cargue con toda la deriva del período.
4. **"Sin aporte voluntario en los 6 meses previos"** = la ventana cerrada `corte-5 … corte`.
5. **Ventana de simulación 201901–202609**: se necesitan 63 cortes para el split con gap entre todas
   las particiones. Efecto lateral conocido: la tabla macro del universo común extrapola hacia atrás
   con el tope imponible del último año configurado (92 UF) y con la UF y el IMM de 2022, así que
   2019–2021 tienen un régimen macro plano y un tope algo alto (3,4 % de meses al tope en 2019 vs
   7,3 % en 2022). Afecta sobre todo al entrenamiento; normalizar por IMM mitiga parte del efecto y
   la corrección de `macro_pandas` está propuesta para `common/` (§10).
6. **Tramo tributario**: se publican dos features, `tramo_igc_12m` (renta imponible rodante, siempre
   disponible) y `tramo_igc_declarado` (tabla anual con la regla de la Operación Renta). La renta
   viene **topada al tope imponible**, así que el tramo del IGC está subestimado para las rentas
   altas; en producción la AFP debería usar la renta bruta informada, no la imponible.
7. **Regresor sobre el ratio `monto / rim_ref`** en vez de sobre `log(monto)`: con train 3–4 años
   antes del test, un nivel nominal no es transferible.
8. **Calibración condicional**: la isotónica se aplica sólo si mejora el Brier fuera de muestra
   dentro de validación en al menos 1 %. En esta corrida no se aplica en ningún origen y se dice
   explícitamente; la probabilidad cruda ya queda con ECE 0,0018.

**Supuestos que habría que confirmar con la AFP**

* Los aportes voluntarios, las campañas y el saldo voluntario se conocen **sin desfase** (son
  sistemas propios de la AFP), a diferencia de la cotización obligatoria que llega en m+2 y de las
  licencias que llegan en m+1. Si el APV por planilla llegara con el mismo desfase que la
  cotización, habría que atrasar `meses_desde_ult_aporte`, `n_aportes_*` y la regla de exclusión.
* `ofrece_apvc` y `pct_match` son atributos del convenio con el empleador, conocidos en el mes en
  que rigen.
* Las respuestas de campaña (apertura de email, click en la app, resultado de la llamada) están
  disponibles el mismo mes.

**Limitaciones**

* Los parámetros (probabilidades de APVC, tasas de aporte, elasticidades, montos) son supuestos
  razonables para Chile, no series oficiales de la Superintendencia de Pensiones.
* El protocolo deja el entrenamiento 3–4 años antes del test. Es conservador: en producción, con
  reentrenamiento trimestral, la distancia real entre el último target de entrenamiento y el mes que
  se predice es de 12 meses, no de 36. Los números de §6 son por lo tanto un piso.
* `region` conserva importancia sin estar en el mecanismo: queda ruido residual pese a la
  regularización y a la CV agrupada.
* El target es "al menos un aporte en 6 meses"; no distingue abrir un plan mensual de hacer un
  depósito único. Un tercer modelo de **periodicidad** sería útil para la asesoría.
* No se modela la competencia (APV en bancos, compañías de seguros y AGF), que en Chile concentra
  buena parte del stock de APV.
* El modelo no discrimina en `VOLUNTARIO` (2,2 % de las filas) ni tiene aportantes suficientes en
  `PENSIONADO_ACTIVO` (0,2 %).

---

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.propension_apv.run --n-afiliados 20000    # [--salida DIR --seed 42 --primer-corte 202101 --cada-origen 3 --sin-diagnostico]
python -m pytest afp_lab/propension_apv/tests -q
```

Corrida de referencia (20.000 afiliados, 4 núcleos compartidos, `n_jobs=2`):

| Etapa | Máquina libre | Con los otros 2 proyectos corriendo | Peor caso observado |
|---|---|---|---|
| Generación de las tablas sintéticas | 5,0 | 6,7 | 10,8 |
| Dataset as-of (63 cortes, ~1,0 M filas) | 17,2 | 17,3 | 17,6 |
| Entrenamiento (4 orígenes × clasificador + regresor + calibración) | 52,5 | 82,4 | 195,2 |
| Evaluación, importancias, ejemplos y CV agrupada | 16,8 | 27,0 | 19,1 |
| **Total** | **95** | **135** | **245** (límite: 300) |

El runtime es reproducible en resultados pero **no en tiempo**: la máquina de 4 núcleos es
compartida con los otros dos proyectos del laboratorio y el entrenamiento de LightGBM
(`n_jobs=2`) se degrada hasta 2,4× cuando los tres corren a la vez. El peor caso medido (245 s)
se obtuvo corriendo además la suite de tests en paralelo, y sigue bajo el límite de 5 minutos.
Si hace falta margen: `--sin-diagnostico` salta la CV agrupada por afiliado (~15 s) y
`--cada-origen 6` entrena 2 orígenes en vez de 4 (la mitad del tiempo de entrenamiento, mismos
12 cortes de test, aunque con una ventana de entrenamiento algo más antigua en el segundo bloque).

Tests: **25 tests en ~16 s** (límite: 120). Cubren leakage por truncamiento (con el desfase propio
de cada fuente y neutralizando la RIM de `corte-1`) y por alteración del futuro en dos cortes; que
las licencias sí se lean en `corte-1` y sólo ellas; la regla as-of del tramo tributario; ausencia de
columnas latentes y de escala en las features; gaps entre todas las particiones y en los orígenes
móviles; recálculo a mano de 6 features en 2 casos × 2 cortes; reglas de población; que el target
empiece después del corte; forma del dataset; determinismo; coherencia de la simulación (un cesante
no aporta, el descuento por planilla exige empleador, recursión del saldo voluntario, tasas objetivo
del brief); que los hiperparámetros salgan de validación; que el regresor sea homogéneo de grado 1
en la renta; que el smearing se calcule en validación; baselines y métricas; sesgo por decil y por
corte; persistencia; y corrida end-to-end.

Salidas en `salidas/propension_apv/` (en `.gitignore`): `fuentes/*.parquet`,
`dataset/{train_ultimo_origen,valid_ultimo_origen,test}.parquet`,
`modelo/{modelo.pkl,calibracion.json}`, `predicciones_test.parquet` y `reportes/*.csv`.

### Criterios de aceptación del brief

| Criterio | Umbral | Obtenido | ¿Cumple? |
|---|---|---|---|
| AUC de test | ≥ 0,75 | **0,836** (0,798–0,852 por corte) | sí |
| Lift decil 1 | ≥ 3 | **5,13** (4,55–5,34 por corte) | sí |
| WAPE del monto en aportantes | ≤ 0,6 | **0,555** | sí |
| Captura al 10 % vs orden por renta | > | **67,8 % vs 49,4 %** | sí |
| Runtime con 20.000 afiliados | < 5 min | **95 s** con la máquina libre; **135 s** con los otros dos proyectos en paralelo; **245 s** en el peor caso medido | sí |
| Tests | < 2 min | **16 s** | sí |
| AUC dentro del rango pedido a la simulación | 0,75–0,88 | 0,836 | sí |

---

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/`** (no se tocó nada fuera de `afp_lab/propension_apv/`):

1. *(Ya aplicado por el gerente en la ronda 1)* `observed=True` en
   `metrics.tabla_metricas_por_segmento` y desempate aleatorio en `lift_decil_1`. El desempate
   importa: era el número más sensible a empates, justamente en el baseline de reglas.
2. Falta una métrica de **captura por decil / top-k** (`captura_monto`, `captura_positivos`,
   `tasa_en_contactados`): está en `evaluate.metrica_negocio` y sirve igual para el churn de
   traspaso y para el riesgo de cese.
3. Falta **curva de calibración / ECE** (`curva_calibracion`, `resumen_calibracion`) y **sesgo por
   decil** (`sesgo_monto_por_decil`): el sesgo agregado solo es engañoso en variables de cola larga,
   como quedó demostrado en la ronda 1.
4. Utilidades de panel denso (`matriz_densa`, ventanas as-of), `split_temporal` con gap entre todas
   las particiones y `origenes_moviles`: genéricas para cualquier proyecto de cortes móviles.
5. `synth_base` podría exponer las tablas *observables* separadas de las latentes (hoy el
   diccionario trae `afiliados_latente` y `empleadores_latente`, fáciles de usar por error).
6. **`macro_pandas` extrapola mal hacia atrás**: para años anteriores al primer ancla usa
   `_TOPE_UF[max(...)]` (92 UF, el valor de 2027) y congela UF e IMM. Cualquier proyecto que necesite
   burn-in anterior a 2022 hereda un tope imponible demasiado alto. Sugerencia: usar el ancla *más
   antigua* como valor por defecto hacia atrás.

**Próximos pasos**

* Modelo de **periodicidad** (mensual vs esporádico) como tercera etapa: cambia la oferta (descuento
  por planilla vs depósito de marzo).
* **Uplift** en vez de propensión: con un holdout aleatorio de contactos se puede estimar el efecto
  incremental del contacto, que es la decisión real.
* Reentrenamiento mensual en vez de trimestral y monitoreo de deriva del saldo voluntario y de la
  tasa base por corte.
* Cuantiles (P10/P90) del monto para dar un intervalo a la proyección de flujo.
* Segmentar `INDEPENDIENTE` (ciclo anual vía Operación Renta) y sacar a los `VOLUNTARIO`.
* Revisar `region` con *permutation importance* y, si no aporta fuera de muestra, quitarla.

---

## 11. Qué cambió respecto de la ronda 1

| # | Hallazgo de la revisión | Corrección | Antes → después |
|---|---|---|---|
| 1 | **Sin gap train→valid** (5 meses de target compartidos); el early stopping nunca cortaba (`n_arboles = 700 = n_estimators`) | Gap de `h` entre **cada** par de particiones; tope de árboles 3.000 y corta el early stopping | árboles 700 (= tope) → **93–181** (tope 3.000). AUC valid 0,885 (inflado) / test 0,834 → valid 0,831 / test **0,836** |
| 2 | **Isotónica rota** en el decil 10 (+40 %) y decidida sobre un valid contaminado con una mejora de 1e-4 | Umbral: sólo se calibra si la mejora fuera de muestra en valid con gap es ≥ 1 % del Brier. No se calibra, y se dice | decil 10: 0,380 vs 0,272 (+40 %) → **0,251 vs 0,256 (−2 %)**. ECE 0,0155 → **0,0018**. Brier 0,0462 → **0,0405** |
| 3 | **El +16,6 % de sesgo agregado del monto era un artefacto** (deciles 1–9 entre −50 % y −100 %, decil 10 +102 %) | Se reporta `sesgo_monto_por_decil`; regresor sobre el ratio y smearing en validación | agregado +16,6 % → **+1,0 %**; decil 10 +102 % → **+3,5 %**; peor decil con peso ≥ 3 % del monto: −100 % → **−9 %** |
| 4 | **`n_estimators` fijado mirando test** (700 tras ver que 2.000 empeoraba test) | Tope alto + early stopping contra valid con gap; ningún número de test intervino en la elección | test reportado sin haber sido consultado; el tope no se toca en ninguno de los 4 orígenes |
| 5 | **Target desalineado con la decisión**: 33 % del monto (61 % en los cortes de test) ya había ocurrido al armar la lista | Convención nueva: `corte` = mes de decisión; panel en `corte-2`, licencias en `corte-1`, operacional en `corte`, target en `corte+1…+6` | 33 % del monto ya observado → **0 %** (hay test: el target empieza en `corte+1`) |
| 6 | **Memorización por afiliado** (AUC train 0,986; `region` y `meses_desde_afiliacion` con gain sin entrar al target) | `num_leaves` 48→16, `min_child_samples` 120→400, `lambda_l2` 0→10, features en unidades de IMM, y CV **agrupada por afiliado** como diagnóstico | AUC train 0,986 → **0,853**; CV agrupada por afiliado **0,826** (antes no se medía); `region` gain 3,5 % → 2,4 %; `meses_desde_afiliacion` 5,5 % → fuera del top 15 |
| 7 | **Smearing in-sample** (1,21 train vs 1,26 valid) | Se calcula en validación con gap | smearing 1,210 (train) → **1,271** (valid). Sesgo del monto condicional −2,2 % → **+2,2 %** |
| 8 | **Valid y test de 3 cortes contiguos** con tasa base estacional (0,030–0,077) | 12 cortes de valid y 12 de test, orígenes móviles cada 3 meses, y reporte **por corte** | test 3 cortes (49 k filas) → **12 cortes (192 k filas)**; AUC por corte 0,798–0,852 reportado |
| 9 | **Lectura de negocio sobredimensionada** ("68,6 % del ahorro del semestre") | Se aclara que es el ahorro de **adquisición/reactivación** y se cuantifica su peso; el monto ya observado desapareció con la corrección 5 | "68,6 % del semestre" → **"67,8 % del ahorro de adquisición, que es el 38 % del flujo total"** |
| m1 | `truncar_tablas` débil para `tributario` | Trunca por `anio_tributario_disponible(corte)` y neutraliza RIM/estado/empleador de `corte-1` | test de leakage más estricto |
| m2, m3 | `lift_decil_1` con empates y `observed=True` | Los aplicó el gerente en `common`; §10 actualizada | — |
| m4 | `n_empleadores_12` contaba transiciones | Cuenta empleadores distintos | — |
| m5 | Ventana de licencia 3 vs 2 meses | `licencia_larga_reciente` usa 2 meses, coherente con el desfase de la fuente | — |
| m6 | `tasa_apertura_6m` ambigua sin contactos | Vale 0 y se acompaña de `tuvo_contacto_6m` | — |
| m7 | "nunca" = 99 dentro de una variable continua | Valor topado + flag (`sin_cotizacion_24m`, `nunca_aporto_36m`, `sin_contacto_24m`, `sin_tramo_declarado`) | — |
| m8 | Renta anual desde RIM topada | Renombrada `renta_imponible_12m_imm` y documentada como renta imponible, no bruta | — |
| m9 | Rama muerta en `meses_con_apvc` | Eliminada | — |

Métricas principales, ronda 1 → ronda 2. **No son comparables como "mejora del modelo"**: la ronda 1
medía sobre un protocolo contaminado y un target parcialmente observado; la ronda 2 mide sobre 12
cortes con todos los gaps y sin nada del target observado.

| Métrica | Ronda 1 | Ronda 2 |
|---|---|---|
| Cortes de test | 3 (49.003 filas) | **12 (192.557 filas)** |
| AUC | 0,834 | 0,836 |
| Lift decil 1 | 5,03 | **5,13** |
| Brier / ECE | 0,0462 / 0,0155 | **0,0405 / 0,0018** |
| Sesgo agregado de la probabilidad | +13,8 % | **−0,6 %** |
| WAPE del monto condicional | 0,552 | 0,555 |
| Sesgo agregado del monto esperado | +16,6 % (artefacto por decil) | **+1,0 %** (peor decil relevante −9 %) |
| Captura al 10 % (modelo / renta) | 68,6 % / 49,8 % | 67,8 % / 49,4 % |
| Árboles del clasificador | 700 (= tope, sin early stopping) | **93–181** (tope 3.000) |
| AUC train (memorización) | 0,986 | **0,853** |
| Runtime con 20.000 afiliados | 39 s | 95–245 s según la carga de la máquina (4 modelos en vez de 1, y 63 cortes en vez de 41) |
