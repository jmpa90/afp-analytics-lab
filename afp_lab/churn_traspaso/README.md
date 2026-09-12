# 02 · Propensión a traspaso a otra AFP (churn de afiliados)

**Ronda 2** (corrige la revisión `docs/revisiones/ronda_1/churn_traspaso.md`). Ver §11 "Qué cambió
respecto de la ronda 1".

Modelo que estima, para cada afiliado vigente y cada mes de corte, la **probabilidad de que solicite
un traspaso a otra AFP en los próximos 6 meses**, y ordena la cartera para que el equipo comercial
use su capacidad de contacto (10 % de la cartera al mes) donde más traspasos y más saldo administrado
se salvan.

```bash
python -m afp_lab.churn_traspaso.run --n-afiliados 20000                       # 43 s - 2,5 min
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 \
    --semillas-estabilidad 42,123,7,2024,99                                    # 2,5 - 4,5 min
python -m pytest afp_lab/churn_traspaso/tests -q                               # 13 - 60 s
```

---

## 1. Pregunta de negocio y decisión que apoya

| | |
|---|---|
| **Pregunta** | ¿Qué afiliados se traspasarán a otra AFP en los próximos 6 meses? |
| **Decisión** | A quién llama/visita el ejecutivo de retención. La restricción operativa es dura: el equipo contacta como máximo al **10 % de la cartera al mes**. |
| **Métrica de negocio** | % de los traspasos y % del **saldo en riesgo** capturados en ese 10 % (y en el 5 % y el 20 %, para dimensionar el equipo). |
| **Pregunta secundaria** | Saldo en riesgo esperado por afiliado = `P(traspaso) × saldo`, para presupuestar la caída de activos administrados. |

El modelo entrega un **orden** (a quién contactar) y un **nivel** (cuánto saldo está en juego). El
orden es sólido; el nivel depende de un pronóstico agregado del flujo mensual de traspasos que no se
puede sacar sólo de la historia (§6 y §8).

## 2. Datos sintéticos: tablas, mecanismos y parámetros

Se parte del universo común `afp_lab.common.synth_base.generar_universo` (afiliados, empleadores,
panel laboral mensual con estado/empleador/RIM/licencias, macro). **No se redefine nada de él.** Se
agregan 7 tablas propias, simuladas mes a mes (`synthetic.generar_tablas`, semilla fija, numpy
vectorizado sobre afiliados con un loop de 69 meses: 2021-01 a 2026-09).

| Tabla | Grano | Columnas |
|---|---|---|
| `mercado` | mes × AFP (6) | `comision`, `rent_12m_A … rent_12m_E`, `gano_licitacion`, `es_propia` |
| `saldos` | afiliado × mes vigente | `saldo_obligatorio`, `fondo` (A–E), `antiguedad_afp_meses` |
| `cambios_fondo` | evento | `fondo_origen`, `fondo_destino` |
| `traspasos` | evento | `periodo_solicitud`, `periodo_efectivo` (= solicitud+1), `afp_origen`, `afp_destino`, `tipo` (SALIDA / ENTRADA) — **es el target** |
| `contactos` | evento | `canal` (ejecutivo propio, vendedor competencia, call center, campaña digital), `resultado` |
| `reclamos` | evento | `tipo` (rentabilidad, atención, comisión, pensión), `estado` |
| `uso_digital` | afiliado × mes vigente | `logins_app`, `logins_web`, `consultas_saldo`, `descargas_certificado` |
| `afiliados_latente_churn` | afiliado | `lat_u_traspaso`, `lat_intencion_visible`, `lat_switcher` — **latentes, sólo para el diagnóstico de techo (§6); un test verifica que no entran al modelo** |

### Mercado (`AFP_ANDES` es la propia; 5 competidoras ficticias)

* **Comisiones** (% de la renta imponible, tramos en `COMISIONES`): propia 1,27 → 1,19 (2024-01) →
  1,14 (2025-07); competidoras entre 0,58 y 1,45. Es el rango real del sistema chileno, y la AFP cara
  reacciona bajando dos veces.
* **Rentabilidad 12 m por fondo y AFP**: paseo aleatorio mensual del sistema con media/volatilidad
  por multifondo (A 6,2 %/16 % anual … E 2,2 %/3,5 %) más una desviación AR(1) por AFP. La
  innovación del AR(1) está calibrada para que la **diferencia de rentabilidad 12 m entre AFP sea de
  1,5–1,8 pp (mediana) y 2,8 pp en el p95**, que es lo que se observa en Chile. La propia arrastra un
  **castigo de −1,0 pp anual entre 2024-01 y 2025-12**, que domina su ventana (brecha media −0,7 pp
  dentro contra +0,1 pp fuera; verificado en `test_castigo_de_rentabilidad_domina_su_ventana`).
* **Licitación de nuevos afiliados** cada 24 meses: la adjudicataria recibe a todos los que se
  afilian al sistema y ellos **deben permanecer 24 meses** (regla real). La propia gana 2022-08 →
  2024-07, así que en el período de test hay una cohorte recién liberada.

### Cuenta obligatoria

* Saldo inicial = `0,10 × RIM_ref × ((1+i)^m − 1)/i × densidad × lognormal(σ=0,35)`; mediana
  **8,6 millones de CLP**, media ~16 millones, coherente con los saldos por edad en Chile.
* Mensual: `saldo_t = saldo_{t−1} × (1 + r_fondo_propia_t) + 0,10 × RIM_{t−2}` — el aporte entra con
  **2 meses de desfase**, igual que la recaudación real, así que el saldo del corte es conocido al corte.
* Fondo inicial por asignación etaria por defecto (B hasta 35, C hasta 50 M / 55 H, D después) y un
  35 % de afiliados "activos" que eligen A (jóvenes) o E (mayores).

### Mecanismos de relación con la AFP

| Mecanismo | Cómo funciona | Por qué es razonable para Chile |
|---|---|---|
| **Campaña de la competencia por empleador** | con prob. 2 % mensual una empresa entra en campaña 3 meses; durante esa ventana la prob. de visita a sus afiliados se multiplica ×8 | la fuerza de venta de las AFP trabaja **por empresa** (charlas en el lugar de trabajo), no afiliado por afiliado |
| Visita de vendedor de la competencia | 1,6 % mensual base, ×2,6 en el decil superior de saldo | persiguen cuentas grandes |
| Contacto del ejecutivo propio | 1,2 % base, ×3 si hubo reclamo o visita de la competencia en 3 m | la retención es **reactiva**: llega después de la señal |
| Reclamos | 0,40 % base, ×2,2 si la rentabilidad propia va mal, ×1,6 si la comisión es cara | da ~0,12 reclamos por afiliado-año |
| **Switchers** | quien ya se cambió de AFP antes lleva +0,55 en el logit del hazard | "el que se cambió una vez se vuelve a cambiar"; es **observable** (la antigüedad en la AFP es menor que la antigüedad en el sistema) |
| Uso digital | Poisson(λ=0,55) logins, ×1,8 jóvenes/universitarios | |
| **Intención de traspaso** | sólo el **35 %** de los afiliados la manifiesta en los canales propios; para ellos los logins se multiplican ×1,7 y la prob. de pedir certificado sube en 6 pp | en Chile **el traspaso lo tramita la AFP de destino**: la de origen muchas veces no ve nada antes de recibir el aviso |
| Descarga de certificado | 1,0 % base ×2,5 en abril-mayo (Operación Renta), +0,6 pp por crédito hipotecario (30-50 años con renta media-alta), +3,0 pp a menos de 24 meses de la edad legal, +2,0 pp tras finiquito o cambio de empleador | la mayoría de las descargas tiene **causa exógena**, no el traspaso |
| Cambio de fondo | 0,55 % base, ×3 afiliado activo, ×1,8 si el fondo va mal | el que cambia de fondo es un afiliado "activo" |

### Mecanismo del target (hazard mensual, escala logit)

```
logit(h) = b0
  + b_brecha_comision · brecha de comisión (pp sobre la más barata)
  + b_brecha_rentabilidad · max(0, −brecha de rentabilidad 12 m del fondo del afiliado, en pp)
  + b_vendedor_competencia_3m · visita de la competencia en 3 m  (+ b_vendedor_interesado_3m si INTERESADO)
  + b_reclamo_6m · reclamo en 6 m                                (+ b_reclamo_desfavorable_6m si en contra)
  + b_ejecutivo_propio_3m · contacto del ejecutivo propio en 3 m
  + b_joven · (40 − edad)⁺   + b_renta · log(RIM_ref / IMM)   + b_antiguedad · log1p(años en la AFP)
  + b_cambios_fondo_12m · min(cambios de fondo 12 m, 3)
  + b_cesante · cesante      + b_retirado · retirado      + b_marzo / b_agosto
  + b_u_switcher · ya se cambió de AFP antes
  + u_afiliado ~ N(0; sigma_afiliado²)     + ε_mes ~ N(0; sigma_mes²)
```

y `h = 0` durante los 24 meses de permanencia obligatoria de los afiliados llegados por licitación.
La intención se dispara con ese hazard y la solicitud se firma con un retardo de 0 a 4 meses
(`p_retardo_intencion`); la solicitud es efectiva al mes siguiente.

### Parámetros (tabla generada desde `ParametrosChurn`; un test verifica que no se desvíe)

<!-- PARAMS:INICIO -->
| Parametro | Valor | Que representa |
|---|---|---|
| `p_antiguo_desde_afiliacion` | 0.78 | Afiliados antiguos que siempre estuvieron en la AFP propia |
| `p_antiguo_traspaso_previo` | 0.12 | Afiliados antiguos que entraron antes de la ventana simulada |
| `tasa_mensual_saldo_inicial` | 0.0035 | Rentabilidad real mensual con la que se capitaliza la historia previa |
| `densidad_cotizacion_historica` | 0.68 | Densidad de cotizacion supuesta para el saldo inicial |
| `sigma_saldo_inicial` | 0.35 | Dispersion lognormal del saldo inicial |
| `rho_ar_rentabilidad` | 0.92 | Persistencia AR(1) de la desviacion de rentabilidad de cada AFP |
| `sigma_ar_rentabilidad` | 0.00025 | Innovacion mensual del AR(1) (~0,5 pp de sd en la ventana de 12 meses) |
| `castigo_rentabilidad_propia` | -0.01 | Castigo anual de rentabilidad de la AFP propia en su ventana |
| `castigo_desde` | 202401 | Primer mes del castigo de rentabilidad |
| `castigo_hasta` | 202512 | Ultimo mes del castigo de rentabilidad |
| `p_vendedor_competencia` | 0.016 | Prob. mensual de visita de un vendedor de la competencia |
| `p_ejecutivo_propio` | 0.012 | Prob. mensual de contacto del ejecutivo propio |
| `p_call_center` | 0.03 | Prob. mensual de contacto del call center propio |
| `p_campana_digital` | 0.075 | Prob. mensual de impacto de campana digital |
| `factor_saldo_alto_competencia` | 2.6 | Multiplicador de la visita de la competencia en el decil de saldo alto |
| `factor_retencion_ejecutivo` | 3.0 | Multiplicador del contacto propio tras reclamo o visita de la competencia |
| `factor_campana_marzo_agosto` | 1.45 | Estacionalidad de campanas en marzo y agosto |
| `p_campana_empleador` | 0.02 | Prob. mensual de que la competencia haga una charla de venta en una empresa |
| `meses_campana_empleador` | 3 | Duracion en meses de esa campana por empleador |
| `factor_campana_empleador` | 8.0 | Multiplicador de la visita de la competencia durante la campana |
| `p_reclamo_base` | 0.004 | Prob. mensual base de reclamo |
| `factor_reclamo_rentabilidad_mala` | 2.2 | Multiplicador del reclamo si la rentabilidad propia va mal |
| `factor_reclamo_comision_alta` | 1.6 | Multiplicador del reclamo si la comision propia es cara |
| `lambda_login_base` | 0.55 | Logins mensuales medios (Poisson) de un afiliado tipico |
| `factor_login_joven` | 1.8 | Multiplicador de logins para jovenes y universitarios |
| `p_intencion_visible` | 0.35 | Fraccion de afiliados cuya intencion de traspaso se nota en los canales propios |
| `factor_login_intencion` | 1.7 | Multiplicador de logins durante la intencion (solo fraccion visible) |
| `p_certificado_base` | 0.01 | Prob. mensual de pedir certificado por motivo administrativo corriente |
| `p_certificado_hipotecario` | 0.006 | Prob. adicional por credito o tramite bancario (30-50 anos, renta media-alta) |
| `p_certificado_tramite_pension` | 0.03 | Prob. adicional a menos de 24 meses de la edad legal de pension |
| `p_certificado_cambio_empleo` | 0.02 | Prob. adicional tras finiquito o cambio de empleador |
| `factor_certificado_operacion_renta` | 2.5 | Estacionalidad del certificado en abril-mayo (Operacion Renta) |
| `p_certificado_intencion` | 0.06 | Prob. adicional por intencion de traspaso (solo fraccion visible) |
| `p_cambio_fondo_base` | 0.0055 | Prob. mensual base de cambio de fondo |
| `factor_cambio_fondo_activo` | 3.0 | Multiplicador del cambio de fondo para el afiliado activo |
| `b0` | -5.9 | Intercepto del hazard mensual (escala logit) |
| `b_brecha_comision` | 1.05 | Efecto de 1 pp de comision sobre la AFP mas barata |
| `b_brecha_rentabilidad` | 0.55 | Efecto de 1 pp de rentabilidad 12m bajo el promedio del sistema |
| `b_vendedor_competencia_3m` | 1.45 | Efecto de una visita de la competencia en los ultimos 3 meses |
| `b_vendedor_interesado_3m` | 0.55 | Efecto adicional si esa visita termino en INTERESADO |
| `b_reclamo_6m` | 0.95 | Efecto de un reclamo en los ultimos 6 meses |
| `b_reclamo_desfavorable_6m` | 0.4 | Efecto adicional si el reclamo se resolvio en contra |
| `b_ejecutivo_propio_3m` | -0.85 | Efecto (retencion) del contacto del ejecutivo propio en 3 meses |
| `b_joven` | 0.036 | Efecto por cada ano bajo los 40 |
| `b_renta` | 0.55 | Efecto por log(RIM de referencia / IMM) |
| `b_antiguedad` | -0.75 | Efecto por log1p(anos en la AFP) |
| `b_cambios_fondo_12m` | 0.33 | Efecto por cambio de fondo en 12 meses (tope 3) |
| `b_cesante` | -1.15 | Efecto de estar cesante |
| `b_retirado` | -1.6 | Efecto de estar retirado |
| `b_marzo` | 0.35 | Estacionalidad de marzo |
| `b_agosto` | 0.3 | Estacionalidad de agosto |
| `b_u_switcher` | 0.55 | Desplazamiento de la propension para quien ya se cambio de AFP antes |
| `sigma_afiliado` | 0.4 | Desviacion del efecto aleatorio NO observable por afiliado |
| `sigma_mes` | 0.15 | Desviacion del ruido idiosincratico mensual |
| `p_retardo_intencion` | 0.1, 0.22, 0.28, 0.25, 0.15 | Distribucion del retardo (meses) entre la intencion y la firma |
<!-- PARAMS:FIN -->

### Chequeos de plausibilidad y de determinismo (`reportes/resumen_simulacion.csv`)

| Indicador | Valor (semilla 42) | Rango en 5 semillas | Lectura |
|---|---|---|---|
| Tasa de traspaso anualizada | 4,55 % de la cartera (≈5,6 % de la cartera cotizante) | | brief: 4–7 % de la cotizante |
| Tasa móvil de 12 meses, mín / máx | 3,83 % / 5,35 % | 3,39 % / 6,86 % | ronda 1: 1,6 % / 12,6 % |
| Rango de rentabilidad 12 m entre AFP (mediana / p95) | 1,78 pp / 2,80 pp | 1,46–1,81 pp | real en Chile: 1–2 pp |
| `P(y=1 \| certificados_1m>0)` | 2,96 % (tasa base 2,25 %) | | **lift 1,32**, no 33 |
| AUC de la mejor feature suelta | 0,636 (`rim_ref`) | | muy por debajo del modelo (0,743) |
| Fracción de filas con score > 0,9 | 0,0 % (score máximo 0,374) | | ronda 1: cola saturada en 0,99 |
| Contactos / reclamos por afiliado-año | 2,03 / 0,12 | | |

## 3. Target, población, horizonte y exclusiones

* **Fila** = afiliado × `periodo_corte`, cortes mensuales móviles.
* **Target**: `y = 1` si existe un traspaso de SALIDA con **mes de solicitud** en `corte+1 … corte+6`
  (no el mes efectivo: la decisión comercial hay que tomarla antes de la firma).
* **Población**: afiliados vigentes en la AFP propia al corte (existe fila de `saldos` en ese mes).
* **Exclusiones**, todas con información conocida al corte: estado laboral `RETIRADO` en `corte−2`;
  afiliados que **ya presentaron** la solicitud en un mes ≤ corte; cortes sin los 6 meses de
  horizonte completos y los primeros 14 meses de burn-in.
* Tasa base: 2,29 % (train), 2,41 % (valid), 2,25 % (test) con la semilla 42.

## 4. Features (72 columnas, todas as-of)

Regla general: el **panel laboral se lee en `corte−2`** (la cotización del mes *m* llega en *m+2*) y
las tablas **operacionales propias de la AFP se leen en `corte`**.

| Familia | Features | Regla as-of |
|---|---|---|
| Historia laboral y renta | `rim_l1`, `rim_media_3/12`, `rim_cv_12`, `rim_ref`, `rim_tendencia_3_12`, `densidad_6/12`, `meses_desde_ult_cotiza`, `estado_laboral`, `tipo_contrato`, `rubro`, `tamano_empleador`, `meses_con_empleador`, `meses_desde_cambio_empleador`, `n_cambios_empleador_12` | ventanas que terminan en **`corte−2`** |
| Cuenta | `saldo_obligatorio`, `log_saldo`, `saldo_sobre_rim_ref`, `fondo`, `cambios_fondo_3m/12m`, `meses_desde_cambio_fondo`, `antiguedad_afp_meses` | hasta **`corte`** |
| Afiliado | `edad`, `sexo`, `region`, `nivel_educacional`, `tipo_afiliado`, `meses_desde_afiliacion`, `meses_hasta_pension`, **`meses_en_otras_afp`**, **`llego_por_traspaso`** | atributos de maestro |
| Relación comercial | `contactos_{canal}_1m/3m/6m` (4 canales), `contactos_competencia_interesado_6m`, `meses_desde_contacto_competencia`, `meses_desde_contacto_ejecutivo`, **`contactos_competencia_pares_3m`**, **`n_pares_empleador`** | hasta **`corte`**; el empleador que define a los "pares" se lee en `corte−2` |
| Reclamos | `reclamos_6m/12m`, `reclamos_abiertos_12m`, `reclamos_desfavorables_12m`, `meses_desde_reclamo`, `tipo_ultimo_reclamo` (catálogo fijo) | hasta **`corte`** |
| Uso digital | `logins_1m/3m`, `consultas_saldo_3m`, `certificados_1m/3m`, `ratio_logins_3m_12m` | hasta **`corte`** |
| Mercado | `comision_propia`, `comision_min_mercado`, `brecha_comision_pp/rel`, `rent_12m_propia_fondo`, `rent_12m_sistema_fondo`, `brecha_rentabilidad_pp`, `afp_licitacion`, `licitacion_propia`, `mes` | mes = **`corte`** |

`meses_hasta_pension` **puede ser negativo**: son los pensionados activos y quienes siguen cotizando
pasada la edad legal (60 mujeres / 65 hombres). El signo es informativo y por eso no se trunca en 0.

Las columnas `lat_*` del universo y de `afiliados_latente_churn` **no entran** (un test lo verifica).

### Cómo se garantiza que no hay leakage

`construir_features(tablas, corte)` recorta las tablas antes de calcular; `construir_features_multi`
—la ruta que usa el `run`— calcula muchos cortes de una vez sobre matrices afiliado × mes con
ventanas que sólo miran hacia atrás. Tres tests:

* `test_sin_leakage_en_la_ruta_matricial`: multiplica ×17 la RIM posterior al corte, marca todo como
  RETIRADO, borra filas del panel y de saldos, inyecta uso digital extremo y 50 traspasos falsos en
  `corte+1`, y recalcula con **`construir_features_multi`** (que no recorta nada) sobre un corte que
  **no** es el último → todas las features quedan idénticas. Este test reemplaza al de la ronda 1,
  que recortaba antes de calcular y por eso era tautológico.
* `test_sin_leakage_alterar_el_futuro_no_cambia_las_features`: lo mismo por la ruta de un corte.
* `test_features_asof_verificadas_a_mano`: `rim_l1` es la RIM del mes `corte−2` (y **no** la del
  corte) y `contactos_vendedor_competencia_3m` es el conteo exacto en `[corte−2, corte]`.

### Split temporal contiguo con purga (corrige el mayor 5)

La purga es la mínima que elimina el solapamiento: un corte `c` aporta target hasta `c+h`, así que la
partición siguiente empieza en `c+h+1`. Eso deja la validación **pegada al test** en vez de
arrastrarla `2h` meses hacia atrás como hacía la ronda 1.

| Partición | Cortes | Rango | Filas | Tasa base |
|---|---|---|---|---|
| train | 31 | 2022-03 → 2024-09 | 476.534 | 2,29 % |
| valid | 3 | 2025-04 → 2025-06 | 44.897 | 2,41 % |
| test | 3 | 2026-01 → 2026-03 | 43.278 | 2,25 % |
| descartados (purga) | 12 | 2024-10 → 2025-12 | — | — |

De los 49 cortes evaluables se usan **37 y se descartan 12** (ronda 1: 28 usados, 22 descartados, con
el train terminando 26 meses antes del test; ahora termina 16). El gap entre el último target de
entrenamiento (2025-03) y el primer corte de test (2026-01) es de 10 meses ≥ 6, como pide
CONVENCIONES §3. `split_temporal` tiene guardas: con pocos cortes levanta `ValueError` con mensaje en
vez de `IndexError` o un train vacío en silencio.

## 5. Modelo y baselines

* **LightGBM binario** (`num_leaves 48`, `lr 0,03`, `min_data_in_leaf 60`, `feature_fraction 0,75`,
  `bagging 0,80`, `lambda_l2 5`, **`num_threads 2`**), early stopping de 60 rondas por AUC de
  validación. Los hiperparámetros se eligieron comparando 5 configuraciones **por AUC de validación**
  (test mirado una sola vez, al final).
* **Dos calibraciones**, ambas ajustadas sólo con validación:
  * `uso="ranking"` → **Platt** sobre el log-odds: estrictamente monótona, no genera empates, deja
    intactos orden, AUC, lift y captura.
  * `uso="agregado"` → **isotónica por bloques** (40 bloques) para sumar `prob × saldo`.
  * `model.ajustar_prior(prob, tasa_objetivo)` permite **anclar el nivel** a un pronóstico agregado
    del flujo mensual sin tocar el orden (§6 y §8).
* **Baselines**: `baseline_tasa_base` (trivial); `baseline_regla_negocio` (la del brief: reclamo en
  3 m **o** contacto de la competencia en 3 m **o** brecha de rentabilidad 12 m < −1 pp);
  `baseline_regla_graduada` (cuenta cuántas de las tres condiciones se cumplen, 0–3). Los empates se
  rompen con un desempate aleatorio reproducible.

## 6. Resultados (20.000 afiliados, test = 43.278 filas, tasa base 2,25 %)

Todo se reporta con **rango sobre 5 semillas** (42, 123, 7, 2024, 99;
`reportes/estabilidad_semillas.csv`). La columna "semilla 42" es la corrida de referencia.

### Tabla principal

| Predictor | AUC (42) | AUC rango 5 semillas | AP | Brier | Lift decil 1 (42 / rango) | KS |
|---|---|---|---|---|---|---|
| **Modelo LightGBM** | **0,743** | **0,719 – 0,744** (mediana 0,727) | 0,081 | 0,0213 | **3,53 / 3,23 – 3,58** | 0,364 |
| Modelo, salida isotónica | 0,742 | | 0,079 | 0,0214 | 3,53 | 0,364 |
| Baseline tasa base | 0,500 | | 0,022 | 0,0220 | 0,79 | 0,037 |
| Baseline regla de negocio (brief) | 0,596 | | 0,030 | 0,281 | 1,60 | 0,192 |
| Baseline regla graduada | 0,601 | | 0,032 | 0,055 | 1,77 | 0,194 |

Estable por corte: AUC 0,745 / 0,741 / 0,741 en 2026-01, 2026-02 y 2026-03.

### Techo de predictibilidad del mundo simulado

`reportes/techo_predictibilidad.csv` entrena el mismo modelo **añadiéndole el efecto aleatorio
latente** `lat_u_traspaso` (la parte de la propensión que la AFP no puede observar). Es el techo:

| Semilla | AUC del modelo | AUC del techo | % del techo alcanzado |
|---|---|---|---|
| 42 | 0,7428 | 0,7675 | 96,8 % |
| 123 | 0,7269 | 0,7433 | 97,8 % |
| 7 | 0,7442 | 0,7705 | 96,6 % |
| 2024 | 0,7194 | 0,7465 | 96,4 % |
| 99 | 0,7236 | 0,7365 | 98,2 % |

El modelo alcanza el **96–98 % del AUC alcanzable**. Lo que queda por encima del techo no es
modelable: la mayoría de los traspasos de los próximos 6 meses los gatillan visitas, reclamos y
campañas que **todavía no han ocurrido** cuando se calcula el score.

### Métrica de negocio: captura contactando al X % de mayor score

| Predictor | 5 % — traspasos / saldo | **10 % — traspasos / saldo** | 20 % — traspasos / saldo |
|---|---|---|---|
| **Modelo (semilla 42)** | **23,2 % / 27,8 %** | **35,3 % / 38,9 %** | **52,2 % / 56,8 %** |
| Modelo, rango 5 semillas | | **32,3 – 35,8 % / 29,4 – 41,1 %** | |
| Regla graduada | 11,9 % / 16,3 % | 19,0 % / 23,7 % | 34,7 % / 42,1 % |
| Regla graduada, rango | | 17,3 – 23,3 % | |
| Regla de negocio (brief) | 8,3 % / 10,6 % | 17,6 % / 20,8 % | 33,4 % / 39,8 % |
| Regla de negocio, rango | | 12,4 – 21,1 % | |
| Tasa base (aleatorio) | 4,8 % / 5,4 % | 9,0 % / 8,8 % | 19,1 % / 18,4 % |

**El modelo gana a las dos reglas en las 5 semillas** (peor caso del modelo 32,3 % contra mejor caso
de la regla del brief 21,1 %).

### Por segmento

**El modelo gana en los 24 segmentos grandes (≥ 5 % de las filas)**, por entre +0,058 y +0,209 de
AUC (`reportes/segmentos_modelo_vs_baseline.csv`). La columna `comparable` marca los segmentos donde
el AUC es NaN por falta de positivos: esos **no** cuentan como victoria (antes pasaban en silencio).

| Segmento | AUC modelo | AUC baseline |
|---|---|---|
| Estado laboral: EMPLEADO (68,8 %) / CESANTE (15,7 %) | 0,735 / 0,768 | 0,607 / 0,596 |
| Tramo de saldo 5–15 MM (44,6 %) / 15–40 MM (29,9 %) / 40 MM+ (7,5 %) | 0,754 / 0,731 / 0,761 | 0,593 / 0,591 / 0,682 |
| Tramo de renta 0–600k (24,1 %) / 600k–1,2 MM (43,3 %) / 2,5 MM+ (8,3 %) | 0,683 / 0,738 / 0,710 | 0,568 / 0,580 / 0,622 |
| Tramo de edad <30 / 30–39 / 40–49 / 50–59 / 60+ | 0,673 / 0,728 / 0,732 / 0,769 / 0,698 | 0,615 / 0,584 / 0,592 / 0,587 / 0,562 |
| Fondo A / B / C / D / E | 0,700 / 0,719 / 0,742 / 0,787 / 0,751 | 0,577 / 0,592 / 0,595 / 0,578 / 0,575 |
| Tipo de afiliado: DEPENDIENTE (89,7 %) / INDEPENDIENTE (7,8 %) | 0,747 / 0,758 | 0,605 / 0,594 |

Los más flojos son **<30 años** (0,673, pocas filas y poca historia) y **60+** (0,698).

### Calibración y saldo en riesgo — el punto débil, con los números reales

Curva de calibración (semilla 42): la isotónica queda dentro de ±0,5 pp en los 10 deciles. El
problema **no es la forma sino el nivel**:

| Versión | Sesgo del saldo en riesgo (42) | **Rango 5 semillas** |
|---|---|---|
| Sin calibrar | +2,2 % | −15,1 % … +36,7 % |
| Isotónica (ventana reciente, horizonte cerrado) | **−3,1 %** | **−3,1 % … +38,0 %** (mediana +26,4 %) |
| **Isotónica + ancla de nivel** (`ajustar_prior`, diagnóstico) | −11,3 % | **−11,3 % … +4,8 %** (mediana −5,3 %) |

Por corte (semilla 42): −4,8 %, −4,0 %, −0,5 %. Rango del sesgo por corte entre semillas: −4,8 %
… +43,7 %.

**Diagnóstico**: la tasa base de la ventana de validación es **sistemáticamente mayor que la del
test en las 5 semillas** (2,41 vs 2,25 / 3,26 vs 2,53 / 3,32 vs 2,16 / 2,75 vs 1,83 / 2,48 vs 1,78).
No es ruido ni "selección de supervivencia": es la **fase de la onda del mercado que está en el
diseño** — la AFP propia baja la comisión en 2024-01 y 2025-07 y el castigo de rentabilidad termina
en 2025-12, de modo que el hazard cae justo en el tramo de test, igual en todas las semillas.
Ninguna calibración sobre datos pasados puede adivinar ese nivel; la tabla lo confirma: al anclar el
nivel con `ajustar_prior` el sesgo cae de +26 % (mediana) a −5 % y lo que queda es sólo la
correlación entre la probabilidad y el saldo. Por eso `saldo_en_riesgo` reporta también
`sesgo_conteo`, para separar las dos cosas.

Se probó alargar la ventana de calibración (3 → 9 → 12 cortes, `reportes` del experimento en §8):
mejora la semilla 123 (+15,9 % → +5,2 %), **no** mejora la semilla 7 (+38,0 % → +52,5 %) y cuesta
AUC en todas (−0,005 a −0,010) porque se pierden cortes de entrenamiento. Se descartó.

### Importancia de features (gain, top 12)

| # | Feature | % gain | | # | Feature | % gain |
|---|---|---|---|---|---|---|
| 1 | `meses_en_otras_afp` | 11,8 | | 7 | `meses_desde_contacto_competencia` | 4,3 |
| 2 | `antiguedad_afp_meses` | 10,6 | | 8 | `brecha_rentabilidad_pp` | 4,0 |
| 3 | `region` | 5,4 | | 9 | `rim_ref` | 3,5 |
| 4 | `meses_desde_afiliacion` | 4,9 | | 10 | `llego_por_traspaso` | 3,4 |
| 5 | `rim_media_12` | 4,7 | | 11 | `edad` | 3,4 |
| 6 | `rubro` | 4,7 | | 12 | `contactos_vendedor_competencia_3m` | 3,4 |

Ninguna feature del bloque digital entra al top 25 (en la ronda 1 ocupaban 5 de los 7 primeros
lugares y ~48 % del gain). Ahora manda el riesgo estructural: haber estado en otra AFP, la
antigüedad, el rubro y la región, y como señal próxima la presión de la competencia.

### 5 casos explicados (`reportes/casos_explicados.csv`)

| Caso | Afiliado / corte | Prob | Real | Lectura |
|---|---|---|---|---|
| `top_1` | 10164 / 2026-01 | 0,374 | 0 | 8 meses en la AFP tras 216 en otras (switcher reciente) + visita de la competencia el mes pasado + brecha de rentabilidad −0,91 pp |
| `top_2` | 14401 / 2026-03 | 0,370 | 0 | 119 meses en otras AFP, reclamo **este mes**, visita de la competencia hace 2; saldo 51 MM |
| `falso_positivo` | 9078 / 2026-03 | 0,334 | 0 | el perfil de riesgo es correcto (254 meses en otras AFP, reclamo, visita) y aun así no se fue: con 2,2 % de tasa base, el score más alto de la cartera sigue siendo un "no" en 2 de cada 3 casos |
| `positivo_mayor_saldo` | 15929 / 2026-03 | 0,125 | 1 | **171 MM CLP** en riesgo, 2 visitas de la competencia en 3 meses; el caso que justifica priorizar por `prob × saldo` y no sólo por probabilidad |
| `falso_negativo` | 7400 / 2026-03 | 0,006 | 1 | 255 meses de antigüedad, renta baja, sin señales: se fue por el efecto aleatorio no observable. Es el ruido irreducible que mide el techo de §6 |

## 7. Lectura de negocio

Con la cartera simulada (≈14.400 afiliados vigentes por mes de test, ≈238.000 MM CLP administrados):

* Contactando al **10 % de mayor score (≈1.440 afiliados/mes)** el equipo alcanza el **35 % de los
  traspasos** y el **39 % del saldo en riesgo** — unos **2.530 MM CLP mensuales** de los ~6.500 MM
  que se van. La regla del brief captura 17,6 % / 20,8 %: el modelo **duplica la captura con el mismo
  costo comercial**, y le gana en las 5 semillas.
* Subir del 10 % al 20 % de cobertura agrega 17 puntos de captura (35 % → 52 %); bajar al 5 % pierde
  12 (35 % → 23 %). El retorno es aproximadamente lineal en ese tramo, así que la cobertura óptima la
  fija el costo del contacto, no el modelo.
* Para el **saldo en riesgo esperado** la recomendación es explícita: publicar el nivel **anclado a un
  pronóstico agregado del flujo mensual de traspasos** (`model.ajustar_prior`), no al nivel de la
  última ventana cerrada. Sin ancla el error de nivel llega a +38 %; con ancla queda en ±11 %.
* Palancas accionables que el modelo ordena: **la cohorte de switchers recientes** (llegaron por
  traspaso y llevan poco tiempo: primer decil de riesgo), **las empresas en campaña de la
  competencia** (la feature `contactos_competencia_pares_3m` permite anticipar a los colegas todavía
  no visitados) y **los fondos con peor rentabilidad relativa** (campaña de comunicación por fondo).

## 8. Decisiones tomadas, supuestos y limitaciones

**Decisiones**

1. **Todo** el panel laboral se lee en `corte−2`, no sólo la RIM; el aporte a la cuenta también se
   acredita con 2 meses de desfase, así que el saldo del corte sí es conocido.
2. Universo 2021-01 → 2026-09 para tener 14 meses de burn-in y 49 cortes evaluables.
3. `mercado` en formato ancho (`rent_12m_A…E`) para respetar el grano mes × AFP del brief.
4. Se excluye a quien ya presentó la solicitud en un mes ≤ corte (ceros estructurales, ya no retenibles).
5. Los afiliados nuevos al sistema entran sólo a la adjudicataria de la licitación → ~7 % del universo
   nunca aparece en la cartera; se modela la permanencia obligatoria de 24 meses.
6. **Dos calibradores** (Platt para el orden, isotónica para el nivel) más `ajustar_prior` para el
   ancla agregada.
7. Se asume que la AFP ve la historia de RIM previa de un afiliado que entró por traspaso (llega con
   la cuenta y vía Previred). **Supuesto a confirmar**: si no fuera así habría que enmascarar las
   features laborales anteriores a `antiguedad_afp_meses`.
8. Hiperparámetros elegidos por AUC de validación entre 5 configuraciones; el test se miró una vez.
9. La ventana de calibración se dejó en 3 cortes tras comparar 3 / 9 / 12 sobre 5 semillas (§6).

**Limitaciones y criterios no alcanzados**

* **AUC 0,719–0,744 < 0,75** (criterio del brief). No se ajustó la simulación para cruzarlo. El techo
  medido del mundo simulado es 0,737–0,771, o sea que el modelo captura el 96–98 % de lo alcanzable;
  el resto son eventos posteriores al corte. **Propuesta de umbral** (§10).
* **El nivel del saldo en riesgo no es reproducible sin ancla externa**: −3 % a +38 % entre semillas.
  Es un problema de pronóstico del flujo agregado, no de calibración; queda documentado, medido por
  corte y con el mecanismo de corrección implementado.
* La **regla de negocio del brief** sigue siendo un baseline pobre pero por la razón contraria a la
  ronda 1: ahora la brecha de rentabilidad casi no varía entre afiliados dentro de un mes (sólo por
  fondo), así que la regla marca a bloques completos de la cartera. La regla graduada es el
  comparador honesto y el modelo también le gana.
* Segmentos flojos: menores de 30 (AUC 0,673) y 60+ (0,698).
* No se modela la reentrada de un afiliado que se fue y vuelve, ni el efecto de red (familiares que
  se traspasan juntos), ni el traspaso por cambio de empleador con convenio.
* Los parámetros macro, comisiones y rentabilidades son **supuestos razonables para Chile**, no
  series oficiales; las 6 AFP son ficticias.

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.churn_traspaso.run --n-afiliados 20000
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --semillas-estabilidad 42,123,7,2024,99
python -m pytest afp_lab/churn_traspaso/tests -q
```

Opciones: `--salida DIR`, `--seed`, `--periodo-inicio`, `--periodo-fin`, `--n-test`, `--n-valid`,
`--sin-fuentes`, `--sin-techo`, `--semillas-estabilidad`.

La máquina de 4 núcleos es **compartida con los otros dos proyectos**, así que se reportan las dos
mediciones observadas: con la máquina libre y con los otros dos data scientists corriendo. Todo con
`num_threads=2` y 20.000 afiliados.

| Paso | Máquina libre | Máquina cargada |
|---|---|---|
| `generar_tablas` + `construir_dataset` (37 cortes, 564.709 filas × 72 features) | 14,4 s | 17,5 s |
| Guardar fuentes y dataset en parquet | 1,8 s | 2,0 s |
| `entrenar` (LightGBM + 2 calibradores) | 9,6 s | 30,9 s |
| Evaluar (métricas, segmentos, calibración, SHAP) | 2,9 s | 3,7 s |
| `techo_predictibilidad` (un modelo extra de diagnóstico) | 13,4 s | 90,9 s |
| **Total corrida estándar** | **42,5 s** | **2 min 29 s** (límite: 5 min) |
| Barrido de 5 semillas (`--semillas-estabilidad`) | +101 s → 2 min 25 s | +141 s → 4 min 24 s |

`python -m pytest afp_lab/churn_traspaso/tests -q` → **24 tests**, 13 s con la máquina libre y 60 s
con la máquina cargada (límite: 2 min).

Si hace falta margen adicional, `--sin-techo` saca el modelo de diagnóstico (el paso más caro después
del entrenamiento) y `--semillas-estabilidad ""` (el valor por defecto) omite el barrido.

Salidas en `salidas/churn_traspaso/` (en `.gitignore`): `fuentes/*.parquet`, `dataset.parquet`,
`modelo/` y `reportes/*.csv` (parámetros, resumen de simulación y determinismo, split, métricas,
captura, por corte, por segmento, calibración, saldo en riesgo total y por corte, importancias, casos
explicados, techo de predictibilidad, estabilidad por semilla y rangos, tiempos).

## 10. Propuestas para `common/`, umbrales y próximos pasos

**Propuestas para `common/`** (no se tocó nada fuera de `afp_lab/churn_traspaso/`):

1. *(ya incorporadas por el gerente en la ronda 1: `groupby(..., observed=True)` y el desempate
   aleatorio de `lift_decil_1`.)*
2. `common/metrics.py`: agregar **captura por percentil** (`captura_top(y, score, peso, fracciones)`
   → % de eventos y % de peso capturados, con desempate reproducible). Es la métrica de negocio
   natural de cualquier proyecto de propensión; implementada en `churn_traspaso/evaluate.py`.
3. `common/metrics.py`: agregar `curva_calibracion`, `CalibradorPlatt` y `ajustar_prior` (prior
   shift). Todo proyecto que agregue montos necesita separar **orden** de **nivel**.
4. `common/calendario.py`: mover las funciones puras de periodo (`add_months`, `periodo_to_idx`,
   `rango_periodos`) fuera de `rim_nowcasting.calendario`, que importa PySpark; hoy cualquier
   proyecto pandas arrastra Spark sólo para sumar meses.
5. `common/splits.py`: `split_temporal` con purga y guardas sirve igual a los proyectos 03 y 04.
6. `common/diagnostico.py`: los **chequeos de determinismo** (`P(y|feature)`, AUC de la mejor feature
   suelta, fracción de scores saturados) deberían correr en todos los proyectos sintéticos. Son los
   que habrían detectado el bloqueante 1 de esta revisión sin necesidad de una revisión adversarial.

**Umbrales de aceptación: propuesta**

El brief pide AUC ≥ 0,75 y lift decil 1 ≥ 3. El lift se cumple en las 5 semillas (3,23–3,58); el AUC
no (0,719–0,744). Con la simulación corregida el techo medido es 0,737–0,771, así que **0,75 no es
alcanzable de forma estable** y sólo se alcanzaba antes gracias al bloque digital que regalaba el
target. Propongo, con argumento:

* Mantener **AUC ≥ 0,75 para datos reales** (una cartera real tiene mucha más heterogeneidad
  observable y persistente que un efecto aleatorio N(0; 0,40²)).
* Para el banco de pruebas sintético, reemplazarlo por **AUC ≥ 0,70 y ≥ 95 % del techo medido**
  (`reportes/techo_predictibilidad.csv`): mide lo que el modelo puede hacer, no lo que la simulación
  regala. Hoy: 0,719–0,744 de AUC y 96–98 % del techo en las 5 semillas.
* Y ascender a criterio principal la **métrica de negocio**, que es de la que depende la decisión:
  **captura al 10 % ≥ 30 % de los traspasos y ≥ 1,5 × la mejor regla de negocio**, en todas las
  semillas. Hoy: 32,3–35,8 % y 1,7–2,8 ×.

**Próximos pasos del proyecto**

* **Pronóstico del flujo agregado** de traspasos (serie mensual con comisiones y rentabilidades
  conocidas al corte) para alimentar `ajustar_prior`. Es el mayor error pendiente del nivel.
* **Uplift / incrementalidad**: el modelo predice riesgo, no el efecto del contacto. Como el ejecutivo
  propio es reactivo, el modelo puede aprender que "contacto del ejecutivo" correlaciona con riesgo.
  Un test A/B de retención y un modelo de uplift responderían a quién conviene llamar.
* **Modelo de tiempo hasta el evento** (hazard mensual discreto) para responder "¿cuándo?".
* **Alerta por empresa**: cuando `contactos_competencia_pares_3m` sube en un empleador, avisar antes
  de que la campaña alcance al resto de la dotación.
* Optimización de la lista por saldo retenido esperado sujeto a capacidad, en vez de ordenar sólo por
  probabilidad.

## 11. Qué cambió respecto de la ronda 1

| # | Hallazgo de la revisión | Qué se hizo | Evidencia |
|---|---|---|---|
| 1 (bloq.) | Bloque digital = observación del target | Intención visible sólo para el **35 %** de los afiliados; logins ×1,7 (antes ×3,5) y certificado +6 pp (antes 40 %, ×33 sobre la base); **causas exógenas** del certificado (Operación Renta, crédito, trámite de pensión, cambio de empleo); chequeos de determinismo en `resumen_simulacion` | `lift_certificado` 1,32; `auc_mejor_feature_suelta` 0,636; `frac_score > 0,9` = 0; ninguna feature digital en el top 25 de gain |
| 2 (bloq.) | Dispersión de rentabilidad irreal (6 pp) | Innovación del AR(1) a 1/4 (`sigma_ar_rentabilidad` ahora es parámetro); test del máximo/mínimo móvil de 12 meses y del dominio del castigo | rango entre AFP 1,46–1,81 pp (mediana), p95 2,80 pp; tasa móvil 12 m 3,4 %–6,9 % (antes 1,6 %–12,6 %) |
| 3 (bloq.) | Saldo en riesgo no reproducible | Calibración sobre ventana reciente contigua con horizonte cerrado; `ajustar_prior` para anclar el nivel; sesgo **por corte** y sobre 5 semillas; `sesgo_conteo` separa nivel de forma | rango con isotónica −3 % … +38 %; **con ancla −11 % … +5 %**; documentado como problema de pronóstico agregado |
| 4 (mayor) | Explicación falsa del drift | Reescrita: es la **fase de la onda del mercado** (recortes de comisión 2024-01 y 2025-07 y fin del castigo en 2025-12), idéntica en las 5 semillas (validación > test siempre) | §6, tabla de tasas base por semilla |
| 5 (mayor) | Split con dos gaps encadenados | **Split contiguo con purga**: la validación queda a 7 meses del test; train pasa de 21 a 31 cortes y los descartados de 22 a 12 | §4, `test_split_temporal_purga_y_guardas` |
| 6 (mayor) | Regla del brief degenerada | Se mantiene la regla graduada como comparador exigente y se documenta que ahora la degeneración es por bloques (la brecha de rentabilidad sólo varía por fondo) | §5 y §8 |
| 7 (mayor) | README y código no coinciden | La tabla de parámetros se **genera desde `ParametrosChurn`** (`tabla_parametros_markdown`) y un test falla si el README se desvía | `test_readme_documenta_los_parametros_vigentes` |
| 8 | `split_temporal` sin guardas | `ValueError` con mensaje cuando faltan cortes | `test_split_temporal_purga_y_guardas` |
| 9 | Cortes descartados en silencio | `construir_features_multi` levanta `ValueError` | `test_cortes_sin_historia_fallan_explicitamente` |
| 10 | Categorías desde el slice | Catálogo fijo `TIPOS_RECLAMO` y `FONDOS` | `test_catalogo_fijo_de_tipos_de_reclamo` |
| 11 | `lift_decil_1` sin desempate | *(corregido por el gerente en `common`)* | |
| 12 | Segmento con AUC NaN pasaba el check | `comparar_segmentos` agrega `comparable` / `gana_modelo`; `segmentos_no_superados` los reporta | `test_segmento_sin_positivos_no_se_da_por_ganado` |
| 13 | Métricas sin sensibilidad a semilla | Barrido de 5 semillas integrado al `run`, con tabla de rangos | `reportes/estabilidad_semillas.csv`, `estabilidad_rangos.csv` |
| 14 | `meses_hasta_pension` negativo | Documentado en §4 y en el código | `test_meses_hasta_pension_negativo_es_intencional` |
| 15 | Test de leakage tautológico | Nuevo test que muta el futuro y recalcula con `construir_features_multi` (la ruta que no recorta) | `test_sin_leakage_en_la_ruta_matricial` |

**Cambios de realismo añadidos en la ronda 2** (decididos antes de medir si cruzaban los umbrales, y
reportados igual aunque el AUC quedó bajo 0,75): campañas de venta de la competencia **por
empleador** (así trabaja la fuerza de venta en Chile) con su feature de pares, y **persistencia del
switcher** (quien ya se cambió una vez se vuelve a cambiar), que es observable porque la antigüedad
en la AFP es menor que la antigüedad en el sistema. Sin ellos el AUC quedaba en 0,696 y el lift en
3,06; con ellos, 0,743 y 3,53 con un techo de 0,767.
