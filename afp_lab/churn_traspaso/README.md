# 02 · Propensión a traspaso a otra AFP (churn de afiliados)

**Ronda 3 del brief** (`docs/briefs/ronda_3/churn.md`, todo en §14): la licitación de stock
retroalimenta la cartera, intervalos bootstrap del AUC por segmento y la búsqueda de la fuga real de
Cuprum. **Ronda 2** (`BRIEF_RONDA2.md`): perfil **Cuprum** frente al perfil sistema, **licitación de
afiliados antiguos de la Ley 21.735** y **pronóstico agregado del flujo** — todo en §13. El historial
de correcciones de la ronda 1 está en §11 y §12. Se trabaja en el worktree `/home/user/wt/churn`.

Modelo que estima, para cada afiliado vigente y cada mes de corte, la **probabilidad de que solicite
un traspaso a otra AFP en los próximos 6 meses**, y ordena la cartera para que el equipo comercial
use su capacidad de contacto (10 % de la cartera al mes) donde más traspasos y más saldo administrado
se salvan.

```bash
cd /home/user/wt/churn && export PYTHONPATH=/home/user/wt/churn
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --perfil sistema       # 61 s (ronda 2: 42 s)
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --perfil cuprum        # 60 s (ronda 2: 45 s)
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --perfil cuprum \
    --comparar-perfiles 42,123,7                                               # ~3 min (ronda 2: 2 min 42 s)
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --experimentos         # 2,2 min
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 \
    --semillas-estabilidad 42,123,7,2024,99                                    # 2,4 min
python -m pytest afp_lab/churn_traspaso/tests -q                               # 30 s (ronda 2: 18 s; 39 tests)
```

Los ~15-18 s adicionales por corrida son el reporte nuevo del objetivo 1 (una segunda llamada a
`generar_tablas` con `aplicar_salida_licitacion=False`, sólo para comparar); los tests suben de 33 a
39. Ver §14 para el detalle.

El paquete instalado en modo editable apunta al clon principal, así que **hay que exportar
`PYTHONPATH=/home/user/wt/churn`** para que importe el worktree. Las salidas del perfil cuprum van a
`salidas/churn_traspaso_cuprum/`.

---

## 1. Pregunta de negocio y decisión que apoya

| | |
|---|---|
| **Pregunta** | ¿Qué afiliados se traspasarán a otra AFP en los próximos 6 meses? |
| **Decisión** | A quién llama/visita el ejecutivo de retención. La restricción operativa es dura: el equipo contacta como máximo al **10 % de la cartera al mes**. |
| **Métrica de negocio (criterio principal)** | % de los traspasos y % del **saldo en riesgo** capturados en ese 10 % (y en el 5 % y el 20 %, para dimensionar el equipo). |
| **Pregunta secundaria** | Saldo en riesgo esperado por afiliado = `P(traspaso) × saldo`. **No entregada en esta ronda** (§7 y §8). |

El modelo entrega un **orden** (a quién contactar). El **nivel** agregado —cuántos pesos se van el mes
que viene— no se entrega: requiere un pronóstico del flujo agregado que no sale de la historia (§6).

## 2. Datos sintéticos: tablas, mecanismos y parámetros

Se parte del universo común `afp_lab.common.synth_base.generar_universo` (afiliados, empleadores,
panel laboral mensual con estado/empleador/RIM/licencias, macro). **No se redefine nada de él.** Se
agregan 7 tablas propias más una latente, simuladas mes a mes (`synthetic.generar_tablas`, semilla
fija, numpy vectorizado sobre afiliados con un loop de 69 meses: 2021-01 a 2026-09).

| Tabla | Grano | Columnas |
|---|---|---|
| `mercado` | mes × AFP (6) | `comision`, `rent_12m_A … rent_12m_E`, `gano_licitacion`, `es_propia` |
| `saldos` | afiliado × mes vigente | `saldo_obligatorio`, `fondo` (A–E), `antiguedad_afp_meses` |
| `cambios_fondo` | evento | `fondo_origen`, `fondo_destino` |
| `traspasos` | evento | `periodo_solicitud`, `periodo_efectivo` (= solicitud+1), `afp_origen`, `afp_destino`, `tipo` (SALIDA / ENTRADA) — **es el target** |
| `contactos` | evento | `canal` (ejecutivo propio, vendedor competencia, call center, campaña digital), `resultado` |
| `reclamos` | evento | `tipo` (rentabilidad, atención, comisión, pensión), `estado` |
| `uso_digital` | afiliado × mes vigente | `logins_app`, `logins_web`, `consultas_saldo`, `descargas_certificado` |
| `afiliados_latente_churn` | afiliado | `lat_u_traspaso`, `lat_intencion_visible`, `lat_switcher`, `lat_con_brecha_antiguedad`. **Latente**: sólo alimenta diagnósticos en memoria, `columnas_features` la excluye por prefijo y el `run` **no la escribe en `fuentes/`** |

### Mercado: un roster por perfil (`MercadoPerfil`)

`generar_tablas(perfil=...)` cambia a la vez el **universo común** (renta, educación, región, rubros,
edad, densidad; ver `docs/contexto_afp_cuprum.md` §4) y el **mercado** (roster de AFP, comisiones,
licitación de nuevos afiliados y ventaja de rentabilidad de la propia). `features.py` no sabe nada del
perfil: lee la AFP propia y el roster de la tabla `mercado` (`es_propia`).

| | `perfil="sistema"` (ronda 1) | `perfil="cuprum"` (Ficha N° 165, ago-2026) |
|---|---|---|
| AFP propia | AFP_ANDES (ficticia), comisión 1,27 → 1,19 → 1,14 % | **CUPRUM, 1,44 % fija** |
| Competidoras | 5 ficticias, 0,58 – 1,45 % | **UNO 0,46 %** (0,49 % hasta abr-2026), MODELO 0,58 %, PLANVITAL 1,16 %, HABITAT 1,27 %, CAPITAL 1,44 %, PROVIDA 1,45 % |
| Brecha de precio con la más barata | 0,56 – 0,69 pp | **0,98 pp** (el motor del churn) |
| Licitación de nuevos afiliados | la propia gana 2022-08 → 2024-07 | MODELO y **UNO desde 2025-10**: la propia **nunca** recibe nuevos afiliados |
| Rentabilidad de la propia | castigo de −1,0 pp anual en 2024-01 … 2025-12 | **ventaja de +0,25 pp anual en toda la ventana** (Cuprum lidera a 36 meses en los cinco fondos: A 11,26 %, B 9,42 %, C 5,99 %, D 3,06 %, E 1,36 %) |

### Mercado del perfil sistema (`AFP_ANDES` es la propia; 5 competidoras ficticias)

* **Comisiones** (% de la renta imponible): propia 1,27 → 1,19 (2024-01) → 1,14 (2025-07);
  competidoras entre 0,58 y 1,45. Es el rango real del sistema chileno, y la AFP cara recorta dos veces.
* **Rentabilidad 12 m por fondo y AFP**: paseo aleatorio mensual del sistema por multifondo (A
  6,2 %/16 % anual … E 2,2 %/3,5 %) más una desviación AR(1) por AFP. La innovación está calibrada
  para que la **diferencia de rentabilidad 12 m entre AFP sea de 1,46–1,81 pp (mediana entre semillas)
  y 2,80 pp en el p95**, que es lo que se observa en Chile. La propia arrastra un **castigo de −1,0 pp
  anual entre 2024-01 y 2025-12**, que domina su ventana: la brecha media de la propia contra el
  promedio del sistema es **−0,45 pp dentro del castigo contra +0,15 pp fuera** (semilla 42; entre
  semillas −0,45 … −1,15 dentro y −0,12 … +0,17 fuera).
* **Licitación de nuevos afiliados** cada 24 meses: la adjudicataria recibe a todos los que se afilian
  al sistema y ellos **deben permanecer 24 meses** (regla real). La propia gana 2022-08 → 2024-07.

### Cuenta obligatoria

* Saldo inicial = `0,10 × RIM_ref × ((1+i)^m − 1)/i × densidad × lognormal(σ=0,35)`; mediana
  **8,55 millones de CLP**, coherente con los saldos por edad en Chile.
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
| Reclamos | 0,40 % base, ×2,2 si la rentabilidad propia va mal, ×1,6 si la comisión es cara | da 0,12 reclamos por afiliado-año |
| **Switcher (latente) con proxy imperfecto** | el switcher activo lleva +0,55 en el logit. Es **latente**: `P(switcher \| llegó de otra AFP) = 0,60` y `P(switcher \| sin brecha de antigüedad) = 0,22`, porque parte de los que se movieron lo hicieron **una sola vez por asignación** (licitación previa, convenio del empleador) y parte de los que no registran brecha sí son switchers que todavía no se movieron | "el que se cambió una vez se vuelve a cambiar" es real, pero la brecha de antigüedad es sólo un indicio: la concordancia medida entre `llego_por_traspaso` y `lat_switcher` es **0,762** (la ronda 2 tenía 1,0 y el gerente la rechazó) |
| Uso digital | Poisson(λ=0,55) logins, ×1,8 jóvenes/universitarios | |
| **Intención de traspaso** | sólo el **35 %** de los afiliados la manifiesta en los canales propios; para ellos los logins se multiplican ×1,7 y la prob. de pedir certificado sube 6 pp | en Chile **el traspaso lo tramita la AFP de destino**: la de origen muchas veces no ve nada antes del aviso |
| Descarga de certificado | 1,0 % base ×2,5 en abril-mayo (Operación Renta), +0,6 pp por crédito hipotecario (30-50 años, renta media-alta), +3,0 pp a menos de 24 meses de la edad legal, +2,0 pp tras finiquito o cambio de empleador | la mayoría de las descargas tiene **causa exógena**, no el traspaso |
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
  + b_u_switcher · switcher latente
  + u_afiliado ~ N(0; sigma_afiliado²)     + ε_mes ~ N(0; sigma_mes²)
```

y `h = 0` durante los 24 meses de permanencia obligatoria de los afiliados llegados por licitación.
La intención se dispara con ese hazard y la solicitud se firma con un retardo de 0 a 4 meses
(`p_retardo_intencion`); la solicitud es efectiva al mes siguiente.

### Cuánto aporta cada mecanismo nuevo (contrafactual, semilla 42, 20.000 afiliados)

| Generador | AUC | Lift decil 1 | Captura al 10 % |
|---|---|---|---|
| Completo (ronda 3) | 0,720 | 3,50 | 34,9 % |
| **Sin switcher ni campañas por empleador** | **0,694** | **2,71** | **27,1 %** |

Sin esos dos mecanismos **fallarían los dos criterios de aceptación** (AUC ≥ 0,70 y captura ≥ 30 %),
lo que es consistente con lo que midió el revisor (0,685 y 2,77). Son los dos mecanismos que el
gerente revisó uno por uno en la ronda 2: aceptó las campañas por empleador y exigió degradar el
proxy del switcher, que es lo que se hizo.

### Parámetros (tabla generada desde `ParametrosChurn`; un test falla si el README se desvía)

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
| `p_switcher_dado_traspaso` | 0.6 | P(switcher activo | llego de otra AFP): el resto se movio por asignacion |
| `p_switcher_sin_traspaso` | 0.22 | P(switcher activo | sin brecha de antiguedad): switcher que aun no se movio |
| `sigma_afiliado` | 0.4 | Desviacion del efecto aleatorio NO observable por afiliado |
| `sigma_mes` | 0.15 | Desviacion del ruido idiosincratico mensual |
| `p_retardo_intencion` | 0.1, 0.22, 0.28, 0.25, 0.15 | Distribucion del retardo (meses) entre la intencion y la firma |
<!-- PARAMS:FIN -->

### Chequeos de plausibilidad y de determinismo (`reportes/resumen_simulacion.csv`)

| Indicador | Valor (semilla 42) | Rango en 5 semillas | Lectura |
|---|---|---|---|
| Tasa de traspaso anualizada | 4,72 % de la cartera | | |
| **Tasa sobre la cartera cotizante** | **6,06 %** | | brief: 4–7 % |
| Tasa móvil de 12 meses, mín / máx | 3,79 % / 5,92 % | 3,67 % / 6,97 % | ronda 1: 1,6 % / 12,6 % |
| Rango de rentabilidad 12 m entre AFP (mediana / p95) | 1,78 pp / 2,80 pp | 1,46–1,81 pp | real en Chile: 1–2 pp |
| `P(y=1 \| certificados_1m>0)` | 3,54 % (tasa base 2,22 %) | | **lift 1,60**, no 33 |
| AUC de la mejor feature suelta | 0,630 (`rim_ref`) | | modelo completo 0,720 |
| Fracción de filas con score > 0,9 | 0,0 % (score máximo 0,331) | | ronda 1: cola saturada en 0,99 |
| **Concordancia `llego_por_traspaso` vs `lat_switcher`** | **0,762** | | ronda 2: 1,0 (rechazado) |
| Contactos / reclamos por afiliado-año | 2,02 / 0,12 | | |

## 3. Target, población, horizonte y exclusiones

* **Fila** = afiliado × `periodo_corte`, cortes mensuales móviles.
* **Target**: `y = 1` si existe un traspaso de SALIDA con **mes de solicitud** en `corte+1 … corte+6`
  (no el mes efectivo: la decisión comercial hay que tomarla antes de la firma).
* **Población**: afiliados vigentes en la AFP propia al corte (existe fila de `saldos` en ese mes).
* **Exclusiones**, todas con información conocida al corte: estado laboral `RETIRADO` en `corte−2`;
  afiliados que **ya presentaron** la solicitud en un mes ≤ corte; cortes sin los 6 meses de
  horizonte completos y los primeros 14 meses de burn-in.
* Tasa base: 2,39 % (train), 2,39 % (valid), 2,22 % (test) con la semilla 42.

## 4. Features (72 columnas, todas as-of)

Regla general: el **panel laboral se lee en `corte−2`** (la cotización del mes *m* llega en *m+2*) y
las tablas **operacionales propias de la AFP se leen en `corte`**.

| Familia | Features | Regla as-of |
|---|---|---|
| Historia laboral y renta | `rim_l1`, `rim_media_3/12`, `rim_cv_12`, `rim_ref`, `rim_tendencia_3_12`, `densidad_6/12`, `meses_desde_ult_cotiza`, `estado_laboral`, `tipo_contrato`, `rubro`, `tamano_empleador`, `meses_con_empleador`, `meses_desde_cambio_empleador`, `n_cambios_empleador_12` | ventanas que terminan en **`corte−2`** |
| Cuenta | `saldo_obligatorio`, `log_saldo`, `saldo_sobre_rim_ref`, `fondo`, `cambios_fondo_3m/12m`, `meses_desde_cambio_fondo`, `antiguedad_afp_meses` | hasta **`corte`** |
| Afiliado | `edad`, `sexo`, `region`, `nivel_educacional`, `tipo_afiliado`, `meses_desde_afiliacion`, `meses_hasta_pension`, `meses_en_otras_afp`, `llego_por_traspaso` | atributos de maestro |
| Relación comercial | `contactos_{canal}_1m/3m/6m` (4 canales), `contactos_competencia_interesado_6m`, `meses_desde_contacto_competencia`, `meses_desde_contacto_ejecutivo`, `contactos_competencia_pares_3m`, `n_pares_empleador` | hasta **`corte`**; el empleador que define a los "pares" se lee en `corte−2` |
| Reclamos | `reclamos_6m/12m`, `reclamos_abiertos_12m`, `reclamos_desfavorables_12m`, `meses_desde_reclamo`, `tipo_ultimo_reclamo` (catálogo fijo) | hasta **`corte`** |
| Uso digital | `logins_1m/3m`, `consultas_saldo_3m`, `certificados_1m/3m`, `ratio_logins_3m_12m` | hasta **`corte`** |
| Mercado | `comision_propia`, `comision_min_mercado`, `brecha_comision_pp/rel`, `rent_12m_propia_fondo`, `rent_12m_sistema_fondo`, `brecha_rentabilidad_pp`, `afp_licitacion`, `licitacion_propia`, `mes` | mes = **`corte`** |

`meses_hasta_pension` **puede ser negativo**: son los pensionados activos y quienes siguen cotizando
pasada la edad legal (60 mujeres / 65 hombres). El signo es informativo y por eso no se trunca en 0.

`columnas_features` excluye **de forma dura** los identificadores, el target y cualquier columna con
prefijo `lat_` o `p_` (CONVENCIONES §2): si un diagnóstico pega una columna latente al dataset, el
modelo no la ve por construcción y no por disciplina del autor. El diagnóstico de techo (§6) tiene
que pasar su lista de columnas a mano para poder usar la latente.

### Cómo se garantiza que no hay leakage

`construir_features(tablas, corte)` recorta las tablas antes de calcular; `construir_features_multi`
—la ruta que usa el `run`— calcula muchos cortes de una vez sobre matrices afiliado × mes con
ventanas que sólo miran hacia atrás. Tres tests:

* `test_sin_leakage_en_la_ruta_matricial`: multiplica ×17 la RIM posterior al corte, marca todo como
  RETIRADO, borra filas del panel y de saldos, inyecta uso digital extremo y 50 traspasos falsos en
  `corte+1`, y recalcula con **`construir_features_multi`** (que no recorta nada) sobre un corte que
  **no** es el último → las 72 features quedan idénticas.
* `test_sin_leakage_alterar_el_futuro_no_cambia_las_features`: lo mismo por la ruta de un corte.
* `test_features_asof_verificadas_a_mano`: `rim_l1` es la RIM del mes `corte−2` (y **no** la del
  corte) y `contactos_vendedor_competencia_3m` es el conteo exacto en `[corte−2, corte]`.

### Split temporal contiguo con purga

La purga es la mínima que elimina el solapamiento: un corte `c` aporta target hasta `c+h`, así que la
partición siguiente empieza en `c+h+1`. La validación queda **a lo más `h+2` meses** del primer corte
de test, en vez de arrastrarse `2h` meses hacia atrás como en la ronda 1.

| Partición | Cortes | Rango | Filas | Tasa base |
|---|---|---|---|---|
| train | 31 | 2022-03 → 2024-09 | 471.733 | 2,39 % |
| valid | 3 | 2025-04 → 2025-06 | 44.365 | 2,39 % |
| test | 3 | 2026-01 → 2026-03 | 42.834 | 2,22 % |
| descartados (purga) | 12 | 2024-10 → 2025-12 | — | — |

De los 49 cortes evaluables se usan **37 y se descartan 12** (ronda 1: 28 y 22, con el train
terminando 26 meses antes del test; ahora 16). El gap entre el último target de entrenamiento
(2025-03) y el primer corte de test (2026-01) es de 10 meses ≥ 6, como pide CONVENCIONES §3.
`split_temporal` levanta `ValueError` con mensaje si no alcanzan los cortes.

## 5. Modelo y baselines

* **LightGBM binario** (`num_leaves 48`, `lr 0,03`, `min_data_in_leaf 60`, `feature_fraction 0,75`,
  `bagging 0,80`, `lambda_l2 5`, **`num_threads 2`**), early stopping de 60 rondas por AUC de
  validación (21 árboles en la corrida de referencia). Los hiperparámetros se eligieron entre 5
  configuraciones **por AUC de validación**; `--experimentos` reproduce la comparación y la deja en
  `reportes/hiperparametros.csv`:

| Configuración | Árboles | AUC valid | AUC test | Elegida |
|---|---|---|---|---|
| lr 0,05 · leaves 48 · min_data 200 | 13 | 0,6973 | 0,7188 | |
| lr 0,03 · leaves 64 · min_data 200 | 12 | 0,6985 | 0,7170 | |
| lr 0,05 · leaves 48 · min_data 60 | 10 | 0,6961 | 0,7148 | |
| **lr 0,03 · leaves 48 · min_data 60** | 21 | **0,7031** | 0,7195 | **sí** |
| lr 0,03 · leaves 96 · min_data 60 · ff 0,60 | 19 | 0,7027 | 0,7205 | |

* **Dos calibraciones**, ambas ajustadas sólo con validación: `uso="ranking"` → **Platt** sobre el
  log-odds (monótona, sin empates, deja intactos orden, AUC, lift y captura); `uso="agregado"` →
  **isotónica por bloques** (40 bloques). `model.ajustar_prior(prob, tasa_objetivo)` permite anclar el
  nivel a un pronóstico agregado; **queda como diagnóstico**, no como salida (§7).
* **Baselines**: `baseline_tasa_base` (trivial); `baseline_regla_negocio` (la del brief: reclamo en
  3 m **o** contacto de la competencia en 3 m **o** brecha de rentabilidad 12 m < −1 pp);
  `baseline_regla_graduada` (cuenta cuántas de las tres condiciones se cumplen, 0–3). Los empates los
  resuelve `common.metrics` por valor esperado y `captura_top` con un desempate reproducible.

## 6. Resultados (20.000 afiliados, test = 42.834 filas, tasa base 2,22 %)

Todo con **rango sobre 5 semillas** (42, 123, 7, 2024, 99; `reportes/estabilidad_semillas.csv` y
`estabilidad_rangos.csv`). La columna "semilla 42" es la corrida de referencia.

### Tabla principal

| Predictor | AUC (42) | AUC rango 5 semillas | AP | Brier | Lift decil 1 (42 / rango) | KS |
|---|---|---|---|---|---|---|
| **Modelo LightGBM** | **0,720** | **0,707 – 0,747** (mediana 0,723) | 0,081 | 0,0211 | **3,50 / 3,00 – 3,66** | 0,329 |
| Modelo, salida isotónica | 0,718 | | 0,074 | 0,0212 | 3,50 | 0,330 |
| Baseline tasa base | 0,500 | | 0,022 | 0,0217 | 1,00 | 0,037 |
| Baseline regla de negocio (brief) | 0,592 | | 0,029 | 0,271 | 1,67 | 0,185 |
| Baseline regla graduada | 0,598 | | 0,033 | 0,053 | 1,99 | 0,184 |

Estable por corte: AUC 0,713 / 0,724 / 0,722 en 2026-01, 2026-02 y 2026-03.

### Métrica de negocio: captura contactando al X % de mayor score

| Predictor | 5 % — traspasos / saldo | **10 % — traspasos / saldo** | 20 % — traspasos / saldo |
|---|---|---|---|
| **Modelo (semilla 42)** | **24,2 % / 24,8 %** | **35,0 % / 38,2 %** | **51,0 % / 54,6 %** |
| Modelo, rango 5 semillas | | **30,0 – 36,6 % / 35,6 – 38,2 %** | |
| Regla graduada | 12,6 % / 15,0 % | 19,4 % / 21,6 % | 34,7 % / 38,7 % |
| Regla graduada, rango | | 18,5 – 22,8 % | |
| Regla de negocio (brief) | 9,3 % / 9,7 % | 17,6 % / 20,3 % | 33,6 % / 35,8 % |
| Regla de negocio, rango | | 11,1 – 23,0 % | |
| Tasa base (aleatorio) | 5,7 % / 6,8 % | 11,5 % / 13,1 % | 21,9 % / 22,6 % |
| **Razón modelo / mejor regla al 10 %** | | **1,80 (42); rango 1,30 – 1,85** | |

Detalle por semilla (el caso límite es la 99, ver §8):

| Semilla | AUC | Lift decil 1 | Captura 10 % | Mejor regla | Razón | Fracción de la cartera que marca la regla del brief |
|---|---|---|---|---|---|---|
| 42 | 0,7195 | 3,50 | 34,9 % | 19,4 % | 1,80 | 26,9 % |
| 123 | 0,7348 | 3,39 | 33,9 % | 20,4 % | 1,67 | 51,0 % |
| 7 | 0,7232 | 3,28 | 32,8 % | 18,5 % | 1,77 | 49,3 % |
| 2024 | 0,7472 | 3,66 | 36,6 % | 19,8 % | 1,85 | 18,3 % |
| **99** | **0,7068** | **3,00** | **30,0 %** | **23,0 %** | **1,30** | **13,7 %** |

### Por segmento

**El modelo gana en los 23 segmentos grandes (≥ 5 % de las filas)**, por entre +0,012 y +0,156 de
AUC, y los 23 son comparables (`reportes/segmentos_modelo_vs_baseline.csv`). La columna `comparable`
marca los segmentos con AUC NaN por falta de positivos: esos **no** cuentan como victoria. Los
márgenes más estrechos son LICENCIA (+0,012), CESANTE (+0,040) y saldo 40 MM+ (+0,061).

### Techo de predictibilidad (diagnóstico, no criterio de aceptación)

`reportes/techo_predictibilidad.csv` entrena el mismo modelo añadiéndole el efecto aleatorio latente
`lat_u_traspaso`: **0,733 contra 0,720 del modelo** (semilla 42). Es un diagnóstico útil para saber
si queda señal observable sin explotar, pero **no** es criterio de aceptación: el gerente verificó
que se cumple en cualquier configuración (96,4 % en un mundo con AUC 0,685) y que sube a 99 %
simplemente bajando `sigma_afiliado`. Lo que queda por encima del techo son visitas, reclamos y
campañas que **todavía no han ocurrido** cuando se calcula el score.

### Calibración y nivel agregado

La isotónica queda dentro de ±0,5 pp en los 10 deciles, así que **la forma está bien**. El problema
es el **nivel**:

| Versión | Sesgo del saldo en riesgo (42) | Rango 5 semillas |
|---|---|---|
| Sin calibrar | −5,1 % | −16,1 % … +36,0 % |
| Isotónica (ventana reciente, horizonte cerrado) | −3,6 % | −3,6 % … +40,2 % |
| Isotónica + ancla de nivel (`ajustar_prior`, **diagnóstico**) | −9,6 % | −12,2 % … +1,2 % |

Por corte (semilla 42): −9,8 %, −0,8 %, +0,8 %; entre semillas el sesgo por corte va de −9,8 % a
+45,4 %. Que el ancla lleve el rango de 44 pp a 13 pp confirma que **casi todo el error es de nivel
agregado y no de forma**.

**Diagnóstico del drift validación → test.** La tasa base cae de validación a test en las 5
semillas. La descomposición del revisor sobre el logit medio identifica dos componentes
**sistemáticos** (presentes en 5/5 semillas) y uno aleatorio:

| Componente | Efecto en el logit medio | Sistemático |
|---|---|---|
| Recorte de comisión de la AFP propia (2024-01 y 2025-07) | −0,05 | sí, 5/5 |
| Envejecimiento de la cartera (la antigüedad media sube con el tiempo) | −0,06 | sí, 5/5 |
| Brecha de rentabilidad (fin del castigo en 2025-12) | +0,02 en la semilla 42, de signo variable | no, aleatorio |

Los dos primeros son predecibles y el modelo los tiene como features (`comision_propia`,
`brecha_comision_pp`, `antiguedad_afp_meses`), pero los árboles no extrapolan el **nivel** a
combinaciones que no vieron en entrenamiento: de ahí que el orden sea estable y el nivel no.

Se probó alargar la ventana de calibración (3 → 9 → 12 cortes, `reportes/ventana_calibracion.csv`):
en la semilla 42 el sesgo de la isotónica va de −3,6 % a +1,6 % y −0,7 %, y el AUC se mueve entre
0,7195 y 0,7244; sobre 5 semillas no dominaba a la ventana de 3 cortes (mejoraba la semilla 123 y
empeoraba la 7) y costaba entre 6 y 9 cortes de entrenamiento. Se mantuvo en 3.

### Importancia de features (gain, top 12)

| # | Feature | % gain | | # | Feature | % gain |
|---|---|---|---|---|---|---|
| 1 | `antiguedad_afp_meses` | 19,3 | | 7 | `contactos_vendedor_competencia_3m` | 3,7 |
| 2 | `rim_ref` | 10,3 | | 8 | `llego_por_traspaso` | 3,0 |
| 3 | `meses_desde_contacto_competencia` | 10,0 | | 9 | `rim_media_12` | 3,0 |
| 4 | `meses_en_otras_afp` | 8,0 | | 10 | `meses_desde_reclamo` | 2,8 |
| 5 | `meses_desde_afiliacion` | 8,0 | | 11 | `rim_media_3` | 2,8 |
| 6 | `rim_l1` | 4,3 | | 12 | `brecha_rentabilidad_pp` | 2,7 |

**Nota de redundancia**: `meses_en_otras_afp` es por construcción `meses_desde_afiliacion −
antiguedad_afp_meses`, y `llego_por_traspaso` es su indicador. Las cuatro features (1, 4, 5 y 8)
describen la misma información de antigüedad y se reparten el gain entre sí: su 38 % combinado **no**
son cuatro contribuciones independientes. Se mantienen las tres derivadas porque los árboles no
construyen la diferencia por sí solos, pero la lectura correcta es "la antigüedad en la AFP y en el
sistema es el bloque más importante del modelo". Ninguna feature del bloque digital entra al top 25
(en la ronda 1 ocupaban 5 de los 7 primeros lugares).

### 5 casos explicados (`reportes/casos_explicados.csv`)

Se eligen a propósito casos distintos (los dos de mayor score, el falso positivo más caro, el
positivo con más saldo y el falso negativo de menor score), no los 5 de mayor score, que son el mismo
perfil repetido. El score máximo de la cartera es 0,33: con una tasa base de 2,2 %, incluso el
afiliado más riesgoso se queda en 2 de cada 3 casos.

## 7. Lectura de negocio

Con la cartera simulada (≈14.280 afiliados vigentes por mes de test, ≈234.700 MM CLP administrados):

* Contactando al **10 % de mayor score (≈1.430 afiliados/mes)** el equipo alcanza el **35 % de los
  traspasos** y el **38 % del saldo en riesgo** — unos **2.390 MM CLP mensuales** de los ~6.250 MM
  que se van. La regla del brief captura 17,6 % / 20,3 %: el modelo **duplica la captura con el mismo
  costo comercial** (razón 1,80 en la semilla de referencia, 1,30–1,85 entre semillas).
* Subir del 10 % al 20 % de cobertura agrega 16 puntos de captura (35 % → 51 %); bajar al 5 % pierde
  11 (35 % → 24 %). El retorno es aproximadamente lineal en ese tramo, así que la cobertura la fija
  el costo del contacto, no el modelo.
* **El saldo en riesgo esperado no se entrega en esta ronda.** El ranking sirve para priorizar por
  valor dentro de la lista (dos afiliados con la misma probabilidad se ordenan por saldo), pero el
  **monto agregado** que se publicaría ("se van X mil millones el próximo semestre") no es utilizable:
  el sesgo va de −4 % a +40 % según la semilla y eso es un error de pronóstico del flujo agregado, no
  de calibración. `ajustar_prior` queda como diagnóstico; el pronóstico agregado está en §10.
* Palancas accionables que el modelo ordena: **la cohorte de antigüedad baja que llegó de otra AFP**
  (primer decil de riesgo), **las empresas en campaña de la competencia** (`contactos_competencia_
  pares_3m` permite anticipar a los colegas todavía no visitados) y **los fondos con peor
  rentabilidad relativa**.

## 8. Decisiones tomadas, supuestos y criterios de aceptación

**Decisiones**

1. **Todo** el panel laboral se lee en `corte−2`, no sólo la RIM; el aporte a la cuenta también se
   acredita con 2 meses de desfase, así que el saldo del corte sí es conocido.
2. Universo 2021-01 → 2026-09 para tener 14 meses de burn-in y 49 cortes evaluables.
3. `mercado` en formato ancho (`rent_12m_A…E`) para respetar el grano mes × AFP del brief.
4. Se excluye a quien ya presentó la solicitud en un mes ≤ corte (ceros estructurales, ya no retenibles).
5. Los afiliados nuevos al sistema entran sólo a la adjudicataria de la licitación → ~7 % del universo
   nunca aparece en la cartera; se modela la permanencia obligatoria de 24 meses.
6. **El switcher es latente** y `llego_por_traspaso` es un proxy imperfecto (concordancia 0,762), por
   decisión del gerente en la ronda 2.
7. Se asume que la AFP ve la historia de RIM previa de un afiliado que entró por traspaso (llega con
   la cuenta y vía Previred). **Supuesto a confirmar**: si no fuera así habría que enmascarar las
   features laborales anteriores a `antiguedad_afp_meses`.
8. Hiperparámetros elegidos por AUC de validación entre 5 configuraciones; el test se miró al final.
9. Ventana de calibración de 3 cortes tras comparar 3 / 9 / 12 (§6).

> **Ronda 2 del brief**: los criterios de esta tabla se volvieron a medir en los dos perfiles con la
> ventana y el split nuevos; el resultado está en **§13 "Criterios de aceptación de la ronda 2"**, que
> es el que hay que revisar. Lo que sigue es la medición de la ronda anterior (test 2026-01…03).

**Criterios de aceptación vigentes** (fijados por el gerente en la ronda 2): para el banco sintético,
**AUC ≥ 0,70 en todas las semillas** y, como **criterio principal**, la métrica de negocio —
**captura al 10 % ≥ 30 % de los traspasos y ≥ 1,5 × la mejor regla, en todas las semillas**. Para
datos reales se mantiene AUC ≥ 0,75. El "≥ 95 % del techo" **no** es criterio de aceptación.

| Criterio | Resultado | ¿Cumple? |
|---|---|---|
| AUC ≥ 0,70 en las 5 semillas | 0,707 / 0,720 / 0,723 / 0,735 / 0,747 | **sí, 5/5** |
| Captura al 10 % ≥ 30 % en las 5 semillas | 30,0 % / 32,8 % / 33,9 % / 34,9 % / 36,6 % | **sí, 5/5** (la semilla 99 en 29,99 %, a 0,01 pp del umbral) |
| Captura ≥ 1,5 × la mejor regla en las 5 semillas | 1,30 / 1,67 / 1,77 / 1,80 / 1,85 | **no: 4/5** (falla la semilla 99) |
| Lift decil 1 ≥ 3 (criterio original del brief) | 3,00 / 3,28 / 3,39 / 3,50 / 3,66 | 4/5 (la semilla 99 en 2,9993) |
| Modelo supera a los baselines sin perder segmentos grandes | 23/23 ganados, +0,012 a +0,156 de AUC | sí |
| Runtime < 5 min y tests < 2 min | 38 s – 2,5 min y 15 – 60 s | sí |

**Criterio no cumplido y su causa.** La razón captura/mejor regla queda en **1,30 en la semilla 99**
(umbral 1,5). La causa está medida y es del **denominador, no del modelo**: la regla del brief marca
una fracción de la cartera que oscila entre **13,7 % (semilla 99) y 51,0 % (semilla 123)**, porque la
condición "brecha de rentabilidad < −1 pp" aplica a bloques enteros de fondos y cuántos cruzan el
umbral depende del camino de rentabilidad simulado. Cuando marca poco (13,7 %) es un grupo pequeño y
bien dirigido y su captura al 10 % sube a 23,0 %, su máximo entre semillas; cuando marca la mitad de
la cartera baja a 11,1 %. El modelo, en cambio, se mueve entre 30,0 % y 36,6 %. Es decir: **el
criterio de razón es inestable porque su denominador lo es** (varía ×2), y la semilla 99 combina el
peor AUC del modelo con la mejor regla. Propuesta concreta en §10.

**Otras limitaciones**

* **Saldo en riesgo esperado: no entregado** (pregunta secundaria del brief). Sesgo −3,6 % … +40,2 %
  entre semillas; con ancla de nivel −12,2 % … +1,2 %, lo que localiza el problema en el pronóstico
  del flujo agregado.
* La **regla de negocio del brief** es un baseline pobre e inestable (ver arriba). La regla graduada
  es el comparador estable (18,5 – 22,8 %) y el modelo le gana en las 5 semillas.
* Segmentos con margen estrecho: LICENCIA (+0,012 de AUC) y CESANTE (+0,040).
* No se modela la reentrada de un afiliado que se fue y vuelve, ni el efecto de red (familiares que
  se traspasan juntos), ni el traspaso por cambio de empleador con convenio.
* Los parámetros macro, comisiones y rentabilidades son **supuestos razonables para Chile**, no
  series oficiales; las 6 AFP son ficticias.

## 9. Cómo correr y tiempos

```bash
cd /home/user/wt/churn && export PYTHONPATH=/home/user/wt/churn
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --perfil sistema
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --perfil cuprum
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --perfil cuprum --comparar-perfiles 42,123,7
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --semillas-estabilidad 42,123,7,2024,99
python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --experimentos
python -m pytest afp_lab/churn_traspaso/tests -q
```

Opciones: `--perfil {sistema,cuprum}`, `--salida DIR`, `--seed`, `--periodo-inicio`, `--periodo-fin`,
`--n-test`, `--n-valid`, `--sin-fuentes`, `--sin-techo`, `--semillas-estabilidad`,
`--comparar-perfiles`, `--experimentos`. Con `--perfil cuprum` la salida va a
`<salida>_cuprum/` para no pisar la del perfil sistema.

`reportes/estabilidad_semillas.csv` y `estabilidad_rangos.csv` **requieren
`--semillas-estabilidad`**; `hiperparametros.csv` y `ventana_calibracion.csv` **requieren
`--experimentos`**. Los cuatro archivos van incluidos en `salidas/churn_traspaso/reportes/` de la
corrida de referencia (se corrieron los tres comandos, en ese orden).

La máquina de 4 núcleos es **compartida con los otros dos proyectos**, así que se reportan las dos
mediciones observadas. Todo con `num_threads=2` y 20.000 afiliados.

| Paso | Máquina libre | Máquina cargada |
|---|---|---|
| `generar_tablas` + `construir_dataset` (37 cortes; 517.070 filas sistema / 436.728 cuprum × 72 features) | 14,4 s | 17,5 s |
| Guardar fuentes y dataset en parquet | 1,8 s | 2,0 s |
| `entrenar` (LightGBM + 2 calibradores) | 9,6 s | 30,9 s |
| Evaluar (métricas, segmentos, calibración, SHAP) | 2,9 s | 3,7 s |
| `techo_predictibilidad` (modelo extra de diagnóstico) | 13,4 s | 90,9 s |
| `licitacion_efecto_cartera` (ronda 3, objetivo 1: segunda `generar_tablas` de comparación) | 15-18 s | — |
| **Total corrida estándar** (`--perfil sistema`) | **42 s → 61 s (ronda 3)** | **2 min 29 s** (límite: 5 min) |
| **Total corrida estándar** (`--perfil cuprum`) | **45 s → 60 s (ronda 3)** | — |
| `--comparar-perfiles 42,123,7` (5 corridas extra) | +117 s → 2 min 42 s (ronda 2; sube ~1 min con el objetivo 1) | — |
| `--semillas-estabilidad` (4 corridas extra) | +102 s → 2 min 21 s | 4 min 24 s |
| `--experimentos` (5 configuraciones + 3 ventanas) | +94 s → 2 min 12 s | — |

Los tiempos de `--comparar-perfiles`, `--semillas-estabilidad` y `--experimentos` de la tabla son los
medidos en la ronda 2 (no se remidieron en la ronda 3: cada uno ya corre bien por debajo del límite de
5 min y sólo `licitacion_efecto_cartera` — que no depende de esos flags — se ejecuta una vez por
corrida completa, así que su costo ya está incluido arriba y no se multiplica por semilla o perfil).

`python -m pytest afp_lab/churn_traspaso/tests -q` → **39 tests en ~30 s** con la máquina libre (33 →
39 y 18 s → 30 s en la ronda 3: intervalos bootstrap y retroalimentación de la licitación) y del orden
de 60-70 s con la máquina cargada (límite: 2 min). `--sin-techo` y `--semillas-estabilidad ""` (el
valor por defecto) dan margen adicional si hace falta.

Salidas en `salidas/churn_traspaso/` y `salidas/churn_traspaso_cuprum/` (en `.gitignore`):
`fuentes/*.parquet` (12 tablas, incluida `licitacion_stock`; **la latente `afiliados_latente_churn` no
se escribe**), `dataset.parquet`, `modelo/` y hasta 31 `reportes/*.csv` (los de licitación, pronóstico
y bootstrap por segmento salen siempre; `comparacion_perfiles*`, `estabilidad_*`, `hiperparametros` y
`ventana_calibracion` requieren su flag).

## 10. Propuestas para `common/`, umbrales y próximos pasos

**Propuestas para `common/`** (no se tocó nada fuera de `afp_lab/churn_traspaso/`):

1. *(ya incorporadas por el gerente: `groupby(..., observed=True)` y el lift del decil 1 con empates
   por valor esperado.)*
2. `common/metrics.py`: agregar **captura por percentil** (`captura_top(y, score, peso, fracciones)`)
   con desempate reproducible. Es la métrica de negocio natural de cualquier proyecto de propensión;
   implementada en `churn_traspaso/evaluate.py`.
3. `common/metrics.py`: agregar `curva_calibracion`, `CalibradorPlatt` y `ajustar_prior`. Todo
   proyecto que agregue montos necesita separar **orden** de **nivel**.
4. `common/calendario.py`: mover las funciones puras de periodo (`add_months`, `periodo_to_idx`,
   `rango_periodos`) fuera de `rim_nowcasting.calendario`, que importa PySpark.
5. `common/splits.py`: `split_temporal` con purga y guardas sirve igual a los proyectos 03 y 04.
6. `common/diagnostico.py`: los **chequeos de determinismo** (`P(y|feature)`, AUC de la mejor feature
   suelta, fracción de scores saturados y **concordancia entre una feature y el latente que la
   inspira**) deberían correr en todos los proyectos sintéticos. Son los que habrían detectado los
   bloqueantes 1 y 2 de la ronda 1 y el proxy perfecto de la ronda 2 sin revisión adversarial.

**Umbrales: propuesta de ajuste al criterio de razón.** El criterio "≥ 1,5 × la mejor regla" tiene un
denominador que varía ×2 entre semillas por una razón ajena al modelo (la fracción de cartera que
marca la regla binaria, 13,7 % – 51,0 %). Propongo una de estas dos variantes, ambas verificables con
`reportes/estabilidad_semillas.csv`:

* medir la razón contra la **regla graduada**, cuya captura es estable (18,5 – 22,8 %): daría
  1,32 – 1,85 y la semilla 99 seguiría bajo 1,5, pero por una diferencia real y no por ruido del
  comparador; o
* exigir la razón **sobre la mediana de las semillas** (hoy 1,77) y, por semilla, el criterio
  absoluto de captura ≥ 30 % (que sí se cumple 5/5). Esta es la que recomiendo: separa "el modelo
  sirve" (captura absoluta, estable) de "el modelo es mejor que la regla" (razón, con comparador
  ruidoso).

**Próximos pasos del proyecto**

* **Pronóstico del flujo agregado** de traspasos (serie mensual con comisiones, rentabilidades y
  antigüedad media de la cartera conocidas al corte) para alimentar `ajustar_prior` y poder entregar
  el saldo en riesgo esperado. Es el entregable pendiente del brief.
* **Uplift / incrementalidad**: el modelo predice riesgo, no el efecto del contacto. Como el ejecutivo
  propio es reactivo, el modelo puede aprender que "contacto del ejecutivo" correlaciona con riesgo.
* **Modelo de tiempo hasta el evento** (hazard mensual discreto) para responder "¿cuándo?".
* **Alerta por empresa**: cuando `contactos_competencia_pares_3m` sube en un empleador, avisar antes
  de que la campaña alcance al resto de la dotación.
* Reducir la redundancia del bloque de antigüedad (4 features para la misma información) y medir si
  una sola parametrización mejora la estabilidad entre semillas.

## 11. Qué cambió respecto de la ronda 1

| # | Hallazgo de la revisión | Qué se hizo | Evidencia |
|---|---|---|---|
| 1 (bloq.) | Bloque digital = observación del target | Intención visible sólo para el **35 %**; logins ×3,5 → **×1,7**; certificado 40 % → **+6 pp** sobre una base con **causas exógenas** | `lift_certificado` 1,60 (antes ×33 de multiplicador); `auc_mejor_feature_suelta` 0,630; `frac_score>0,9` = 0; score máximo 0,331; ninguna feature digital en el top 25 de gain |
| 2 (bloq.) | Dispersión de rentabilidad irreal (6 pp) | Innovación del AR(1) a 1/4 (`sigma_ar_rentabilidad` es parámetro); tests del máximo/mínimo móvil y del dominio del castigo | rango entre AFP 1,46–1,81 pp, p95 2,80; tasa móvil 12 m 3,7 %–7,0 % (antes 1,6 %–12,6 %); castigo −0,45 pp dentro vs +0,15 pp fuera |
| 3 (bloq.) | Saldo en riesgo no reproducible | Calibración sobre ventana reciente contigua; `ajustar_prior`; sesgo por corte y sobre 5 semillas; `sesgo_conteo` separa nivel de forma. **Se declara no entregado** | isotónica −3,6 % … +40,2 %; con ancla −12,2 % … +1,2 % |
| 4 (mayor) | Explicación falsa del drift | Reescrita con la descomposición medida: recorte de comisión (−0,05 del logit) y envejecimiento de la cartera (−0,06) son sistemáticos en 5/5; la rentabilidad es aleatoria (+0,02 en la semilla 42) | §6 |
| 5 (mayor) | Split con dos gaps encadenados | **Split contiguo con purga**: validación a 7 meses del test; train 21 → 31 cortes; descartados 22 → 12 | §4, `test_split_temporal_purga_y_guardas` |
| 6 (mayor) | Regla del brief degenerada | Se mantiene la regla graduada como comparador y se documenta que la fracción marcada oscila 13,7 % – 51,0 % entre semillas | §5, §6 y §8 |
| 7 (mayor) | README y código no coinciden | Tabla de parámetros **generada desde `ParametrosChurn`**; un test falla si el README se desvía | `test_readme_documenta_los_parametros_vigentes` |
| 8 | `split_temporal` sin guardas | `ValueError` con mensaje | `test_split_temporal_purga_y_guardas` |
| 9 | Cortes descartados en silencio | `construir_features_multi` levanta `ValueError` | `test_cortes_sin_historia_fallan_explicitamente` |
| 10 | Categorías desde el slice | Catálogo fijo `TIPOS_RECLAMO` y `FONDOS` | `test_catalogo_fijo_de_tipos_de_reclamo` |
| 11 | `lift_decil_1` sin desempate | *(corregido por el gerente en `common`; ahora empates por valor esperado, score constante = 1,0)* | `metricas_test.csv`: `baseline_tasa_base` lift 1,00 |
| 12 | Segmento con AUC NaN pasaba el check | `comparable` / `gana_modelo` y `segmentos_no_superados` | `test_segmento_sin_positivos_no_se_da_por_ganado` |
| 13 | Métricas sin sensibilidad a semilla | Barrido de 5 semillas con tabla de rangos | `estabilidad_semillas.csv`, `estabilidad_rangos.csv` |
| 14 | `meses_hasta_pension` negativo | Documentado en §4 y en el código | `test_meses_hasta_pension_negativo_es_intencional` |
| 15 | Test de leakage tautológico | Test que muta el futuro y recalcula con `construir_features_multi` | `test_sin_leakage_en_la_ruta_matricial` |

## 12. Ronda 3: correcciones sobre la revisión de la ronda 2

| Punto | Antes | Ahora |
|---|---|---|
| **Umbrales** (decisión 1 del gerente) | §8/§10 proponían "AUC ≥ 0,70 **y ≥ 95 % del techo**" | El techo es **sólo diagnóstico** (§6); los criterios son AUC ≥ 0,70 en todas las semillas y la métrica de negocio como principal; ≥ 0,75 se mantiene para datos reales (§8 y §10) |
| **Proxy del switcher** (decisión 2) | `llego_por_traspaso` reproducía `lat_switcher` con concordancia **1,0** | El switcher es latente con `P(sw \| brecha)=0,60` y `P(sw \| sin brecha)=0,22`; concordancia medida **0,762**, incluida en `resumen_simulacion` y con test. Re-medido: AUC 0,743 → **0,720** (semilla 42), 0,707–0,747 en 5 semillas |
| **Saldo en riesgo** (decisión 3) | §7 lo presentaba como recomendación operativa | **No entregado en esta ronda**; `ajustar_prior` es diagnóstico y el pronóstico agregado pasó a §10 |
| **Drift** (corrección 1) | "fase de la onda del mercado", negando la composición | Descomposición medida: comisión −0,05 y **envejecimiento de la cartera** −0,06 sistemáticos en 5/5; rentabilidad aleatoria |
| **Contrafactual** (corrección 2) | "sin switcher ni campañas: 0,696 / 3,06" | Medido de nuevo con el generador corregido: **0,694 / 2,71 / captura 27,1 %**, fallarían los dos criterios (§2). Cifras del castigo (−0,45 / +0,15 pp) y tasa sobre cotizantes (**6,06 %**, indicador nuevo `tasa_traspaso_anualizada_cotizantes`) |
| **Reportes** (corrección 3) | `estabilidad_*.csv` requerían el flag y no se entregaban | Los cuatro archivos van en `reportes/`; §9 marca qué flag necesita cada uno |
| **Latentes** (corrección 4) | `columnas_features` sólo excluía por nombre; el techo pegaba `lat_u_traspaso`; la tabla latente se escribía en `fuentes/` | Exclusión **dura** por prefijo `lat_`/`p_`; el techo pasa lista de columnas explícita; la tabla latente no se persiste (3 tests) |
| **Menores** (corrección 5) | — | Test del castigo con `add_months`; docstring de `split_temporal` ("a lo más `h+2`"); **nota de redundancia** de `meses_en_otras_afp` junto a las importancias; `reportes/hiperparametros.csv` (5 configuraciones) y `ventana_calibracion.csv` vía `--experimentos` |
| **Lift de `common`** (corrección 6) | lift del baseline trivial 0,79–1,14 según el orden de las filas | `baseline_tasa_base` da exactamente **1,00**; todas las tablas de baselines se re-corrieron con la métrica nueva |

## 13. Ronda 2 del brief: perfil Cuprum, licitación de stock y pronóstico agregado

Cambios estructurales de esta ronda, antes de los resultados:

* **Ventana simulada 2021-01 → 2028-12** (96 meses, `PERIODO_INICIO`/`PERIODO_FIN`) para cubrir la
  primera licitación de afiliados antiguos (sorteo 2027-08, traspasos 2028-06). La macro de 2027-2028
  es una **extrapolación plana** del último ancla disponible (UF 40.800, IMM 560.000, tope 92 UF): no
  hay proyección real de esas series.
* **Cortes evaluables 2022-03 → 2027-11** (69). `cortes_evaluables(excluir_licitacion=True)` **trunca**
  la serie antes del primer corte cuya ventana de target contiene el mes efectivo del sorteo: los
  traspasos por licitación no son churn voluntario y ensuciarían la etiqueta. Se pierden 7 cortes.
* **Ventana de entrenamiento acotada** (`split_temporal(max_cortes_train=31)`): se usan los 31 cortes
  más recientes que respetan la purga, de modo que alargar la ventana simulada no encarece el run y
  el entrenamiento se parece a un reentrenamiento mensual. Split: train 2023-11 → 2026-05 (31),
  valid 2026-12 → 2027-02, test **2027-09 → 2027-11**, 32 cortes descartados.
* El test se movió de 2026-01…03 (rondas anteriores) a 2027-09…11, así que **los números de §6 no son
  comparables uno a uno con los de esta sección**.

### A. Perfil Cuprum vs perfil sistema (3 semillas cada uno, 20.000 afiliados)

`reportes/comparacion_perfiles.csv` y `comparacion_perfiles_rangos.csv` (comando
`--comparar-perfiles 42,123,7`). Semillas 42 / 123 / 7; entre paréntesis el rango.

| | Sistema | Cuprum | Referencia real (Ficha N° 165) |
|---|---|---|---|
| **Simulación** | | | |
| RIM de referencia, mediana del test | $897.598 (892 k – 905 k) | **$1.799.070** (1.787 k – 1.815 k) | ingreso imponible promedio $1.346.859 sistema vs $2.331.030 Cuprum (**1,73×**; la simulación da **2,0×** en la mediana) |
| Saldo mediano del test | 13,8 MM (13,8 – 15,5) | **25,4 MM** (23,5 – 26,0) | Cuprum: 17,3 % de los activos con 4,5 % de los afiliados |
| Tasa de salida mensual | 0,351 % (0,347 – 0,351) | **0,446 %** (0,434 – 0,458) | **0,47 %** mensual en el sistema (56.974 cuentas / 12,2 M afiliados) |
| Traspasos netos en la ventana | **−2.647** (−2.659 … −2.630) | **+1.307** (+1.205 … +1.376) | Cuprum fue la AFP con **más traspasos netos positivos** en jun-26 (+4.128) |
| Tasa anual sobre cotizantes | 5,44 % | 7,72 % | ver "tensión de calibración" abajo |
| Rango de rentabilidad 12 m entre AFP | 1,56 – 1,88 pp | 1,68 – 1,82 pp | 1–2 pp |
| **Modelo** | | | |
| AUC | 0,698 – 0,706 (mediana 0,701) | **0,714 – 0,754** (mediana 0,747) | |
| Average precision | 0,044 – 0,059 | 0,066 – 0,089 | |
| Lift decil 1 | 3,22 – 3,58 | 3,00 – 3,86 | |
| Brier | 0,0133 – 0,0161 | 0,0211 – 0,0273 | |
| Tasa base del test | 1,36 – 1,66 % | 2,19 – 2,84 % | |
| Techo de predictibilidad (diagnóstico) | 0,738 | 0,739 | |
| **Negocio (10 % de la cartera)** | | | |
| Captura de traspasos | 32,2 – 35,8 % | **30,0 – 38,6 %** | |
| Captura de saldo en riesgo | 30,7 – 35,3 % | **23,1 – 35,1 %** | |
| Razón contra la mejor regla | 1,52 – 1,93 (mediana 1,82) | 1,37 – 2,16 (mediana **2,04**) | |
| **Calibración** | | | |
| Sesgo del saldo en riesgo (isotónica) | −1,6 % … +6,2 % | +1,2 % … +11,4 % | |

**Lo que cambia de verdad al pasar a Cuprum**, más allá de que los niveles se muevan:

1. **El motor del churn es el precio, no la rentabilidad.** Con comisión fija en 1,44 % y ventaja de
   rentabilidad positiva, la brecha de precio de 0,98 pp con Uno es constante en el tiempo: no hay
   "ola" que explique variación entre meses, y por eso el pronóstico agregado estructural se degrada
   (ver C). En las importancias, `brecha_rentabilidad_pp` cae del 4,9 % del gain (sistema) a **0,8 %**
   (cuprum), `rim_ref` de 8,5 % a 0,9 % y `antiguedad_afp_meses` sube de 14,0 % a **31,9 %**: el modelo
   del perfil cuprum es esencialmente un modelo de antigüedad y presión comercial.
2. **Riesgo y valor están negativamente correlacionados.** En Cuprum el afiliado de saldo alto es el
   antiguo, que es el de menor riesgo: la captura de **saldo** al 10 % (23,1 % en la semilla 42) queda
   **por debajo** de la captura de traspasos (30,0 %), al revés que en el perfil sistema (35,3 % vs
   35,8 %). Consecuencia operativa directa: en Cuprum **no basta ordenar por probabilidad**, hay que
   ordenar por `probabilidad × saldo` si el objetivo es saldo retenido.
3. **La cartera no recibe afiliados nuevos** (Uno tiene la licitación desde 2025-10), así que todo el
   crecimiento viene de traspasos de entrada: por eso el perfil cuprum necesita que **36 % de los
   afiliados antiguos entren por traspaso durante la ventana** para reproducir el saldo neto positivo
   de la Ficha. Es el supuesto más fuerte del perfil.

**Tensión de calibración, resuelta por el gerente en la revisión de la ronda 2: manda la Ficha.** El
rango 4–7 % de la ronda 1 era un prior; la cifra oficial es 0,47 % mensual sobre afiliados del sistema.
Para Cuprum no hay dato público de fuga *bruta* (la Ficha sólo publica el neto, +4.128 en junio de 2026),
así que 0,446 % mensual es un **supuesto de trabajo** anclado al promedio del sistema, a reemplazar por la
serie "traspasos por AFP de origen y destino" de la SP cuando se descargue. Antecedentes: el brief de la ronda 1 pedía 4–7 %
anual **sobre cotizantes**; el de la ronda 2 pide calibrar con la Ficha, que da **0,47 % mensual sobre
afiliados** (≈5,6 % anual sobre afiliados y, con la densidad de cotizantes del sistema de 53,5 %,
≈10,5 % anual sobre cotizantes). Los dos objetivos no son compatibles. Se optó por el de la ronda 2
para el perfil cuprum (0,446 % mensual, que deja 7,7 % anual sobre cotizantes, sobre el techo del 7 %
de la ronda 1) y por dejar el perfil sistema como estaba (0,351 % mensual, 5,4 % sobre cotizantes,
25 % por debajo de la Ficha) para no romper la comparabilidad con las rondas anteriores. El test
`test_perfil_cuprum_vs_sistema` fija el rango mensual de cuprum en 0,35–0,60 %.

### B. Licitación de afiliados antiguos (Ley 21.735) — `licitacion.py`

Regla implementada: cada 2 años se sortea el **10 % de los afiliados no pensionados**; los sorteados
pasan a la **AFP de menor comisión** salvo que opten por quedarse en **30 días**. `SORTEOS` fija el
calendario (sorteo 2027-08, límite de opción 2027-09, traspaso efectivo 2028-06); el segundo sorteo
(2029-08) queda fuera de la ventana.

**Tabla `licitacion_stock`** (`simular_licitacion_stock`): afiliado, `periodo_sorteo`,
`periodo_limite_opcion`, `periodo_efectivo`, saldo, antigüedad, RIM de referencia, `p_se_va_pasiva` y
`decision` (SE_VA / SE_QUEDA) con `afp_destino`. Elegibles: vigentes al mes del sorteo con estado
laboral distinto de RETIRADO y `tipo_afiliado` distinto de PENSIONADO_ACTIVO; sorteo aleatorio simple.

La propensión a irse es un logit anclado en `p_salida_pasiva_base` (**65 %**, punto medio del rango
55–75 % del brief) y modulado por drivers conocidos al sorteo: antigüedad (−0,55 por log1p(años)),
saldo (−0,18 por log10(saldo/10 MM)), renta (−0,22), uso digital (−0,10 por login mensual), reclamo en
12 m (+0,45) y la **propensión latente al churn** del afiliado (+0,80 por desviación estándar), más
ruido (σ 0,35). `p_se_va_pasiva` es la probabilidad **sin** campaña: la retención es un escenario, no
algo ya incorporado en la tabla.

**Tamaño del evento** (`reportes/licitacion_resumen.csv`, 20.000 afiliados, semilla 42):

| Perfil | Afiliados en cartera | Sorteados | % | Saldo sorteado | % del saldo | Saldo medio sorteado vs cartera | P(se va) media | Saldo en riesgo esperado |
|---|---|---|---|---|---|---|---|---|
| Sistema | 13.714 | 1.300 | 9,5 % | 24.738 MM | 9,6 % | 19,0 vs 18,8 MM | 0,667 | **15.221 MM** |
| **Cuprum** | 12.497 | **1.194** | 9,6 % | **37.898 MM** | 9,9 % | 31,7 vs 30,7 MM | 0,681 | **23.943 MM** |

Para Cuprum, un solo sorteo pone en riesgo **23.943 MM CLP (6,2 % del saldo administrado)** en un mes,
frente a los ~2.000 MM que se van por churn voluntario en un semestre: **el sorteo es un orden de
magnitud mayor que todo el problema de retención ordinaria**.

**Escenarios de retención** (`reportes/licitacion_escenarios.csv`, perfil cuprum; campaña con
capacidad de 30 % de los sorteados = 358 contactos y efecto −1,20 en el logit):

| Salida pasiva | Capacidad | Prioridad | Saldo en riesgo | Saldo retenido | Afiliados retenidos | Saldo retenido por contacto |
|---|---|---|---|---|---|---|
| 55 % | 0 % | — | 20.826 MM | — | — | — |
| 55 % | 30 % | **saldo × prob** | 15.670 MM | **5.156 MM** | 89 | 14,4 MM |
| 55 % | 30 % | aleatorio | 18.300 MM | 2.526 MM | 85 | 7,1 MM |
| **65 %** | 0 % | — | **23.943 MM** | — | — | — |
| **65 %** | 30 % | **saldo × prob** | **18.683 MM** | **5.260 MM** | 86 | **14,7 MM** |
| 65 % | 30 % | aleatorio | 21.395 MM | 2.548 MM | 83 | 7,1 MM |
| 75 % | 0 % | — | 27.230 MM | — | — | — |
| 75 % | 30 % | saldo × prob | 22.089 MM | 5.140 MM | 81 | 14,4 MM |
| 75 % | 30 % | aleatorio | 24.802 MM | 2.428 MM | 76 | 6,8 MM |

Lecturas:

* **Priorizar por `saldo × probabilidad` duplica el saldo retenido** frente a contactar al azar
  (5.260 vs 2.548 MM con el supuesto base), con el mismo número de contactos. Los afiliados retenidos
  son casi los mismos (86 vs 83): toda la ganancia está en *a quién* se retiene, no en cuántos.
* La **sensibilidad al supuesto de salida pasiva es alta en el nivel y baja en la decisión**: el saldo
  en riesgo se mueve entre 20.826 y 27.230 MM (±13 %) según se asuma 55 % o 75 %, pero el saldo
  retenido por la campaña es prácticamente el mismo (5.140 – 5.260 MM) y la conclusión de priorizar
  por valor esperado no cambia en ningún escenario. Es decir: **la decisión es robusta al supuesto,
  el presupuesto no**.
* Con 30 % de capacidad se retiene el 22 % del saldo en riesgo. Ampliar la capacidad sigue siendo la
  palanca de mayor impacto absoluto, pero con **rendimiento marginal decreciente**: el saldo retenido por
  contacto cae de ~14 MM al 30 % a ~7 MM al 100 % (la malla de `escenarios_licitacion` incluye ahora
  10/30/50/70/100 %; `reportes/licitacion_escenarios.csv`). La inversión se justifica tramo a tramo con
  ese rendimiento marginal, no con la cota del 30 %.
* Limitación **(resuelta en la ronda 3, §14.A)**: `licitacion_stock` se generaba después de la
  cartera y no retroalimentaba `saldos`, `panel_mensual` ni `traspasos`; los afiliados que se iban en
  2028-06 seguían en la cartera simulada. No afectaba a esta ronda (los cortes se truncan antes) pero
  había que resolverlo antes de simular el segundo sorteo (2029-08) o de mover `licitacion.py` a
  `common/`.

**Combinación con el score** (`combinar_con_score`, `reportes/licitacion_ranking_combinado.csv`):
`p_ir = p_sorteo_se_va · sorteado + p_churn · (1 − sorteado)`. El resultado operativo es contundente:

| Fracción contactada | De la lista, % que es sorteo | % de los sorteados cubierto | Saldo cubierto |
|---|---|---|---|
| 5 % (595) | **100 %** | 51 % | 15.639 MM (4,1 % de la cartera) |
| 10 % (1.190) | **95,6 %** | 98 % | 38.489 MM (10,0 %) |
| 20 % (2.380) | 49 % | 100 % | 71.560 MM (18,7 %) |

Con `p_sorteo ≈ 0,68` contra `p_churn ≈ 0,03`, **el sorteo se come toda la capacidad de contacto**: la
lista del 10 % es 96 % sorteados y el churn voluntario desaparece del ranking. La recomendación es
gestionar las dos campañas con **presupuestos separados** (una para el sorteo, acotada a los 30 días
de la opción, y otra para el churn ordinario), no con una lista única.

**Por qué el modelo de churn no se reentrena con el sorteo** (docstring de `licitacion.py`): no hay
etiquetas (la primera licitación no ha ocurrido), la mecánica es distinta (la salida es *por defecto* y
lo que el afiliado decide es quedarse) y la población es aleatoria, no auto-seleccionada; meterlo como
feature contaminaría el modelo voluntario. Para aprenderlo se necesita (a) el resultado del primer
sorteo por segmento, (b) el registro de a quién se contactó y cómo, y (c) idealmente **asignación
aleatoria de la capacidad de contacto en el primer sorteo**, que permitiría medir el uplift sin
supuestos. Mientras no existan, `p_salida_pasiva` es un supuesto y todo se reporta como escenario.

### C. Pronóstico agregado del flujo (backlog de la ronda 1) — resultado negativo

`model.PronosticoFlujo` ajusta el **logit de la tasa de traspaso a 6 meses** por corte sobre variables
conocidas al corte (brecha de comisión, brecha de rentabilidad 12 m, antigüedad media de la cartera,
marzo, agosto y tendencia lineal), con dos objetivos: la tasa de **conteo** y la tasa **ponderada por
saldo** (la que corresponde al estimando cuando se suma `prob × saldo`). `ajustar_prior` acepta pesos,
así que el ancla puede fijar `Σ p·saldo / Σ saldo` en vez del conteo.

`seleccionar_pronostico` elige la especificación con un **backtest de ventana expansiva** sobre los
cortes de horizonte cerrado (24 evaluaciones fuera de muestra; el test nunca participa), compitiendo
contra el baseline honesto "constante" (la tasa media de la historia cerrada):

| Perfil | Especificación elegida | MAE relativo (backtest) | MAE de "constante" | MAE de "estructural+tendencia" |
|---|---|---|---|---|
| Sistema | `estructural` (5 variables) | **8,4 %** | 20,0 % | 10,6 % |
| Cuprum | **`constante`** | **17,2 %** | 17,2 % | 158 % |

En el perfil cuprum las especificaciones estructurales **explotan** (MAE 158 %): la brecha de comisión
es casi constante en el tiempo (1,44 % fija contra 0,46 % de Uno, con un único escalón en 2026-05), de
modo que el diseño queda casi colineal y los coeficientes son inestables. El backtest lo detecta y se
queda con la media histórica; es exactamente para eso que la tendencia compite en igualdad de
condiciones en vez de venir por defecto.

**Y el ancla no mejora el nivel** (`reportes/saldo_en_riesgo.csv`, sesgo del saldo en riesgo):

| Versión del score | Sistema (3 semillas) | Cuprum (3 semillas) |
|---|---|---|
| Sin calibrar | +17,1 % (+17,1 … +43,5) | −13,3 % (−13,3 … +10,2) |
| **Isotónica (ventana reciente cerrada)** | **−1,6 %** (−1,6 … +6,2) | **+1,2 %** (+1,2 … +11,4) |
| Isotónica + pronóstico (ponderado por saldo) | +8,5 % (−16,3 … +8,5) | −9,9 % (−9,9 … +38,5) |
| Isotónica + ancla perfecta (diagnóstico) | −4,1 % | −7,3 % |

La conclusión honesta es que **el pronóstico agregado no resuelve el problema que motivó el backlog**:
con la validación contigua al test (7 meses), la isotónica ya deja el sesgo en ±11 % y el pronóstico
sólo agrega varianza (±38 %). El problema de nivel de la ronda 1 (−4 % a +40 %) venía de que la
validación caía en otra fase de la onda del mercado, no de que faltara un modelo agregado; al mover el
test a un tramo calmo el problema desapareció por sí solo. Se deja la maquinaria (`PronosticoFlujo`,
`seleccionar_pronostico`, `ajustar_prior` con pesos, `anclar_con_pronostico`) porque es la pieza que
haría falta si la AFP quisiera publicar el nivel en un tramo de onda —y porque el backtest de
selección es reutilizable— pero **la recomendación sigue siendo usar la isotónica sobre la ventana
cerrada más reciente** y no anclar con el pronóstico. El saldo en riesgo esperado sigue **no
entregado** como salida operativa (§7).

### Criterios de aceptación de la ronda 2

| Criterio | Sistema | Cuprum | ¿Cumple? |
|---|---|---|---|
| AUC ≥ 0,70 en cada semilla | 0,6978 / 0,7006 / 0,7058 | 0,7142 / 0,7469 / 0,7539 | **no en sistema** (semilla 42 a 0,0022 del umbral); sí en cuprum 3/3 |
| Captura al 10 % ≥ 30 % en cada semilla | 35,8 / 32,7 / 32,2 % | 30,0 / 35,6 / 38,6 % | **sí, 3/3 en ambos** (cuprum semilla 42 en 30,04 %) |
| Razón ≥ 1,5 en la mediana | 1,82 | 2,04 | **sí en ambos** |
| Lift decil 1 ≥ 3 (brief ronda 1) | 3,58 / 3,27 / 3,22 | 3,00 / 3,56 / 3,86 | sí, 3/3 en ambos |
| Sin perder segmentos grandes (CONVENCIONES §4) | **21 de 23** | **21 de 22** | **no** (ver abajo) |
| Escenarios de licitación con rangos | — | 9 escenarios × 3 supuestos | sí |
| Runtime < 5 min y tests < 2 min | 42 s | 45 s (2 min 42 s con la comparación) | sí; 33 tests en 18 s |

**Criterios no cumplidos y su causa:**

1. **AUC del perfil sistema en la semilla 42: 0,6978** (umbral 0,70), con 0,7006 y 0,7058 en las otras
   dos. La causa es el cambio de ventana de test: al truncar los cortes por la licitación, el test se
   movió a 2027-09…11, donde la tasa base del perfil sistema cae a 1,36–1,66 % (contra 1,78–2,57 % en
   las rondas anteriores) porque la comisión de la propia ya está recortada y el castigo de
   rentabilidad terminó. Con menos eventos y menos dispersión de precio hay menos que ordenar. No se
   tocó ningún parámetro para cruzar el umbral. El perfil cuprum, que es el relevante para el
   laboratorio, queda en 0,714–0,754.
2. **El modelo pierde en el segmento 60+** frente a la regla graduada: sistema AUC 0,625 vs 0,735
   (−0,110, 7,0 % de las filas, **37 eventos**) y cuprum 0,696 vs 0,719 (−0,023, 7,6 %, **73
   eventos**). Con esa cantidad de eventos el error estándar del AUC es del orden de ±0,05–0,08, así
   que la diferencia del perfil cuprum está dentro del ruido y la del perfil sistema es del orden de
   1,5 errores estándar. En el perfil sistema se pierde además en renta 0–600k por −0,009 (21 % de las
   filas), que es empate práctico. Propuesta: reportar el AUC por segmento con intervalo (bootstrap) y
   no dar por perdido un segmento con menos de ~100 eventos; y si la pérdida en 60+ persiste con más
   semillas, un modelo específico para cercanos a la pensión, donde el driver es la decisión de
   jubilación y no el precio.

### Archivos nuevos y ampliados

| Archivo | Qué se agregó |
|---|---|
| `licitacion.py` | **nuevo**: `ParametrosLicitacion`, `SORTEOS`, `simular_licitacion_stock`, `escenario_licitacion`, `escenarios_licitacion`, `resumen_riesgo_licitacion`, `combinar_con_score`, `ranking_combinado`, `tabla_parametros_licitacion` |
| `synthetic.py` | `MercadoPerfil` / `MERCADOS` (roster, comisiones, licitación y ventaja de rentabilidad por perfil), `PARAMETROS_POR_PERFIL`, `PERIODO_INICIO`/`PERIODO_FIN`, `perfil` en `generar_mercado` y `generar_tablas`, tabla `licitacion_stock`, indicadores `tasa_traspaso_anualizada_cotizantes`, `tasa_salida_mensual` y `traspasos_netos_total` |
| `features.py` | AFP propia y roster leídos de la tabla `mercado` (agnóstico del perfil), `cortes_evaluables(excluir_licitacion=True)`, `split_temporal(max_cortes_train=31)` |
| `model.py` | `serie_mensual_flujo`, `PronosticoFlujo`, `ESPECIFICACIONES_PRONOSTICO`, `seleccionar_pronostico` (backtest expansivo), `anclar_con_pronostico`, `ajustar_prior(peso=...)`, `entrenar(columnas=...)` |
| `run.py` | `--perfil`, `--comparar-perfiles`, reportes de licitación y de pronóstico, salida separada por perfil |
| `tests/` | 6 tests nuevos (33 en total): perfil cuprum vs sistema, licitación de stock, monotonía de los escenarios, combinación con el score, exclusión de cortes contaminados y pronóstico con horizonte cerrado |

### Propuestas adicionales para `common/`

7. `common/synth_base.py`: el perfil `sistema` deja la densidad de cotizantes en ~79 % contra el 47 %
   real del sistema (la propia docstring lo reconoce). Mientras siga así, cualquier tasa expresada
   "sobre cotizantes" no es comparable con la Ficha. Propongo calibrar el perfil `sistema` a la
   densidad real en una ronda coordinada entre los cuatro proyectos, porque cambia los resultados de
   todos a la vez.
8. `common/`: la licitación de stock de la Ley 21.735 afecta a los cuatro proyectos (cambia la cartera
   de 2028 en adelante). `licitacion.py` está escrito sin dependencias del proyecto 02
   (`simular_licitacion_stock` recibe las tablas y el latente como argumentos) justamente para poder
   moverse a `common/` si el gerente lo decide.
9. `common/calendario.py`: la macro se extrapola plana desde 2027. Si los otros proyectos van a
   simular la reforma (cotización adicional 3,5 % → 8,5 %, fondos generacionales desde 2027-04), hace
   falta extender las anclas y agregar la tasa de cotización como serie, no como constante.

## 14. Ronda 3: la licitación retroalimenta la cartera, intervalos bootstrap por segmento y la fuga de Cuprum

Tres encargos del brief de la ronda 3 (`docs/briefs/ronda_3/churn.md`), a partir de los tres
pendientes que dejó la revisión de la ronda 2 (`docs/revisiones/ronda_2/churn_traspaso_ronda2.md`).
**Los 33 tests de la ronda 2 siguen pasando y se agregaron 6 nuevos (39 en total, ~30 s).** La corrida
de referencia (`--perfil cuprum`) sigue siendo la misma que en §13: **train 2023-11→2026-05 (31
cortes), valid 2026-12→2027-02, test 2027-09→2027-11 (35.694 filas)**; ninguno de los cambios de esta
ronda toca esos cortes, así que **las métricas principales (AUC, captura, segmentos, importancias,
casos explicados) quedan byte a byte idénticas a la ronda 2** — es la confirmación de que el arreglo
del objetivo 1 está bien acotado a los cortes que ya se excluían del entrenamiento.

```bash
cd /home/user/wt/churn && export PYTHONPATH=/home/user/wt/churn
python3 -m pytest afp_lab/churn_traspaso/tests -q                     # 42 tests, ~30 s
python3 -m afp_lab.churn_traspaso.run --perfil cuprum                  # ~60 s (antes: ~45 s; ver abajo)
```

### A. Objetivo 1 — la licitación de stock retroalimenta la cartera

**Antes** (ronda 2): `licitacion_stock` se calculaba con el sorteo, la decisión (SE_VA/SE_QUEDA) y el
riesgo de saldo, pero **no tocaba** `saldos`/`traspasos`/`uso_digital`/etc.: un afiliado sorteado que
decidía irse seguía apareciendo vigente, cotizando y acumulando saldo en la cartera simulada como si
la ley no hubiera pasado (limitación documentada explícitamente en la ronda 2, §13.B).

**Ahora**: `simular_licitacion_stock` procesa los sorteos de forma **secuencial** y, para cada uno,
saca de la cartera desde el **mes efectivo** a quien decide `SE_VA` y seguía vigente el mes anterior
(si ya se había traspasado por su cuenta antes, no hay nada que tocar ni que registrar dos veces):
sin más filas en `saldos`/`uso_digital`/`contactos`/`reclamos`/`cambios_fondo` (fuera del AUM, ya no
vigente para `features.py`, exactamente la misma convención que ya usa el traspaso voluntario) y con
un registro nuevo en `traspasos` con tipo **`SALIDA_LICITACION`** (no `SALIDA`, para no tocar el
target de churn voluntario). El procesamiento secuencial importa cuando hay más de un sorteo: quien
salió en el primero ya no aparece vigente —y por lo tanto no es elegible— en el segundo
(`test_licitacion_segundo_sorteo_parametrizable`).

`sorteos` (antes fijo a `SORTEOS`) es ahora un parámetro de `simular_licitacion_stock` y de
`generar_tablas` (`sorteos_licitacion`), y `sorteos_en_ventana` sólo exige que el **mes de sorteo**
quepa en la ventana simulada (antes exigía también el mes efectivo): así se puede simular el segundo
sorteo (2029-08) extendiendo la ventana, aunque su traspaso efectivo (2030-06) quede fuera. Con
`periodo_fin=203006` (`test_licitacion_segundo_sorteo_parametrizable`, 20.000 afiliados, perfil
cuprum) ambos sorteos se simulan y retroalimentan la cartera; con la ventana **por defecto** (hasta
2028-12) sólo el primero cae dentro y el segundo no se reporta.

**Decisión: no se extendió la ventana por defecto a 2029-12.** El brief lo condicionaba a que la
corrida siguiera bajo 5 min, y sobra margen (60 s con la ventana actual), pero extenderla NO es
gratis: `generar_mercado` saca la rentabilidad de cada AFP y fondo con llamadas `rng.normal(..., size=
n_t)` donde `n_t` es el largo de la ventana; alargar `n_t` desplaza el punto de partida en el stream
de números aleatorios de **todas** las llamadas siguientes (afp × fondo), así que la rentabilidad —y
en cascada reclamos, contactos y el hazard— de los meses que **ya estaban** en la ventana cambiaría
también. Eso rompería la comparabilidad byte a byte con la ronda 2 sin necesidad: como se confirma
abajo, el segundo sorteo (2029-08→2030-06) no altera ningún corte evaluable de todos modos (su mes
efectivo es muchísimo más tarde que el primero, que ya es el que trunca `cortes_evaluables`), así que
extender la ventana por defecto no habría agregado ninguna fila útil de entrenamiento. La capacidad de
simular el segundo sorteo queda demostrada con una ventana extendida en el test, no en la corrida de
referencia.

**`cortes_evaluables(excluir_licitacion=True)` sigue excluyendo bien**: sigue truncando por el
**primer** mes efectivo (2028-06) sin importar si hay un segundo sorteo más adelante en
`licitacion_stock` (`test_cortes_excluyen_la_licitacion_con_dos_sorteos`), porque usa el **mínimo** de
`periodo_efectivo` sobre toda la tabla. Confirmado también con la corrida de referencia: `split.csv`,
`metricas_test.csv`, `captura_negocio.csv`, `segmentos_modelo_vs_baseline.csv`,
`saldo_en_riesgo.csv` e `importancias.csv` quedan **idénticos** antes/después del cambio.

**Efecto sobre la tasa del target en los cortes posteriores al sorteo** (excluidos del entrenamiento;
`reportes/licitacion_efecto_cartera.csv`, generado comparando `aplicar_salida_licitacion=True` vs
`False` con la misma semilla): 20.000 afiliados, perfil cuprum, primer sorteo (efectivo 2028-06).

| Corte | Filas con retro. | Filas sin retro. (= ronda 2) | Afiliados removidos | Tasa base con retro. | Tasa base sin retro. |
|---|---|---|---|---|---|
| 2028-06 | 11.163 | 11.895 | 732 | 2,365 % | 2,337 % |
| 2028-07 | 11.166 | 11.894 | 728 | 1,961 % | 1,934 % |
| 2028-08 | 11.171 | 11.895 | 724 | 1,540 % | 1,505 % |
| 2028-09 | 11.180 | 11.902 | 722 | 1,145 % | 1,117 % |
| 2028-10 | 11.185 | 11.905 | 720 | 0,706 % | 0,689 % |
| 2028-11 | 11.207 | 11.925 | 718 | 0,366 % | 0,352 % |

La cartera pierde entre 718 y 732 afiliados por corte (los ~1.194 sorteados de Cuprum × ~68 % que
decide `SE_VA`, menos los pocos que ya se habían ido por su cuenta antes del mes efectivo) y la tasa
base sube entre +1,0 pp y +4,0 pp relativos (los que se van por licitación no son un sorteo perfecto
de riesgo: su probabilidad de irse incluye el efecto de la propensión latente al churn, así que
retirarlos de la cartera deja, en promedio, una población ligeramente más propensa que la que
simulaba —incorrectamente— la ronda 2). El efecto agregado sobre `resumen_simulacion.csv` (que
promedia sobre las 69 cortes evaluables 2022-03→2027-11, de los cuales sólo 6 caen después del
sorteo) es, como se esperaba, pequeño: `afiliados_en_cartera_promedio` 12.076,7 → 12.023,2 (-53,4),
`tasa_traspaso_anualizada` 5,492 % → 5,515 %, `tasa_salida_mensual` 0,4577 % → 0,4596 %.
`traspasos_netos_total` (1.205, sin cambios) no se mueve porque `SALIDA_LICITACION` no cuenta como
`SALIDA` en ese indicador, tal como se documenta.

### B. Objetivo 2 — intervalos bootstrap del AUC (y de la captura) por segmento

`evaluate.bootstrap_ic_segmento(df, prob, segmento, metrica="auc"|"captura", n_boot=200, seed=42)`
remuestrea **con reemplazo, dentro de cada valor del segmento** (no del dataset completo: el error
estándar de un segmento chico no tiene por qué mezclarse con el de uno grande), calcula el punto y el
intervalo percentil `[2,5 %, 97,5 %]`, y marca `cruza_0_5` (AUC no informativo) y
`cruza_umbral_aceptacion` (si se pasa un umbral) — la lectura que pedía la revisión de la ronda 2 para
no dar por perdido un segmento con pocos eventos (el backlog original hablaba del 60+, con 37-73
eventos según el perfil). `bootstrap_auc_segmentos` apila `tramo_edad` (incluye 60+) y `tramo_saldo`
en una sola tabla; el reporte usa `n_boot=200` (`N_BOOT_REFERENCIA`, el mínimo del brief) y los tests
usan 25-40 remuestreos para que corran rápido.

**Test (perfil cuprum, semilla 42, `reportes/bootstrap_auc_segmento.csv`)**, umbral de aceptación 0,70:

| Segmento | n | Eventos | AUC | IC 95 % | ¿Cruza 0,5? | ¿Cruza 0,70? |
|---|---|---|---|---|---|---|
| edad 30-39 | 6.183 | 182 | 0,705 | 0,666 – 0,739 | no | **sí** |
| edad 40-49 | 14.442 | 392 | 0,756 | 0,736 – 0,780 | no | no (por sobre 0,70) |
| edad 50-59 | 12.035 | 348 | 0,677 | 0,648 – 0,703 | no | **sí** |
| **edad 60+** | **2.726** | **74** | **0,696** | **0,643 – 0,756** | no | **sí** |
| edad <30 | 308 | 16 | 0,740 | 0,628 – 0,842 | no | **sí** |
| saldo 0-5M | 2.046 | 70 | 0,672 | 0,615 – 0,736 | no | **sí** |
| saldo 5-15M | 9.125 | 203 | 0,729 | 0,695 – 0,762 | no | **sí** |
| saldo 15-40M | 15.091 | 419 | 0,743 | 0,719 – 0,765 | no | no (por sobre 0,70) |
| **saldo 40M+** | **9.432** | **320** | **0,667** | **0,640 – 0,696** | no | **no (por debajo de 0,70)** |

Lectura: el segmento **60+** (2.726 filas, 74 eventos — el mismo orden que los 37 (sistema) / 73
(cuprum) que medía la revisión de la ronda 2) tiene un intervalo que **cruza 0,70 sin cruzar 0,5**: no se puede
concluir que el modelo *no sirva* ahí, pero tampoco que cumpla con holgura el criterio de aceptación;
exactamente el caso que motivó el pendiente de la ronda 2. El hallazgo más claro es **saldo 40M+**: su
intervalo completo (0,640–0,696) queda **por debajo** de 0,70 — es el único segmento grande donde el
AUC absoluto está confiablemente bajo el umbral (coherente con que en §6 ya era el margen más estrecho
contra el baseline, +0,061). Los segmentos con más filas (edad 40-49, saldo 15-40M) sí quedan
confiablemente por sobre 0,70. La tabla completa (con `captura` al 10 %, `reportes/
bootstrap_captura_segmento.csv`) va en la corrida de referencia.

### C. Objetivo 3 — el supuesto de 0,446 % mensual de Cuprum

Se buscó con WebSearch una cifra pública 2024-2026 de **traspasos brutos** (salidas) de Cuprum para
reemplazar el supuesto de trabajo (README §13.A: 0,446 %/mes, ancla al promedio del sistema completo
de la Ficha N° 165). `spensiones.cl` y `cuprum.cl` están bloqueados por el proxy de la sandbox, como
anticipa el brief, y **también `latercera.com`** (confirmado con `curl`/`WebFetch` → `EGRESS_BLOCKED`
en los tres casos): sólo quedan disponibles los snippets que devuelve WebSearch y la Ficha N° 165 que
ya estaba descargada en `data/sp/`.

**Consultas hechas** (todas el 12-sep-2026): *"AFP Cuprum traspasos 2025 salidas entradas netos
mensual Superintendencia Pensiones"*, *"Cuprum tasa de traspaso AFP 2024 2025 fuga afiliados"*,
*"'Cuprum' traspasos brutos afiliados que salen 2025 AFP origen destino"*, *"traspasos entre AFP 2025
origen destino Cuprum Habitat Capital cifras"*, *"'traspasos netos' Cuprum 2025 La Tercera OR 'Diario
Financiero' mensual"*, *"AFP traspasos mensuales 2025 Uno Cuprum ranking cuentas cambiadas"*, más dos
búsquedas de cierre sobre salidas/porcentaje mensual y sobre `df.cl`. Ninguna devolvió una cifra de
**fuga bruta mensual** de Cuprum con URL verificable.

**Lo que sí se encontró** (dos cifras NETAS —entradas menos salidas—, con fuente y fecha, cada una en
una ventana distinta y no comparable entre sí):

| Fuente | Fecha de acceso | Periodo | Traspasos netos Cuprum |
|---|---|---|---|
| Ficha Estadística Previsional N° 165 (SP, ago-2026; `data/sp/`, verificado extrayendo el PDF) | ya en el repo | junio 2026 (1 mes) | **+4.128** cuentas |
| [La Tercera — "Pensiones: fondo A lideró..."](https://www.latercera.com/pulso/noticia/pensiones-fondo-a-lidero-las-preferencias-de-traspasos-en-2024-y-el-e-fue-el-que-mas-afiliados-perdio/OA36QNL5TZFWZGUZJ7RJPBYN4U/) | 12-sep-2026 (snippet; `latercera.com` bloqueado para abrir el artículo completo) | año 2024 completo | **+4.699** cuentas |

La Ficha también confirma que el total **bruto** traspasado entre AFP en el sistema completo en junio
de 2026 fue de 56.974 cuentas (dato ya usado para el perfil `sistema`), pero, igual que reporta el
brief, la SP **sólo publica el neto por AFP**, nunca el bruto por AFP de origen. Ninguna cifra
encontrada permite aislar la fuga bruta de Cuprum específicamente, y las dos que sí se encontraron
corresponden a ventanas no comparables (un mes vs. un año calendario), así que no alcanzan ni para
recalibrar ni para construir una tendencia limpia — sólo confirman, de forma consistente con la
simulación (`traspasos_netos_total` > 0), que el neto de Cuprum es positivo.

**Decisión: el supuesto se mantiene sin cambios** (0,446 %/mes, `PARAMETROS_POR_PERFIL["cuprum"]
["b0"] = -6,32`, test `test_perfil_cuprum_vs_sistema` con rango 0,35-0,60 %). Se documenta
explícitamente en el código (comentario junto a `PARAMETROS_POR_PERFIL` en `synthetic.py`) qué se
buscó, qué se encontró y por qué no se recalibra, para que quede a mano cuando exista la serie de
traspasos por AFP de origen y destino que pide el brief.

### D. Archivos tocados

| Archivo | Qué cambió |
|---|---|
| `licitacion.py` | `simular_licitacion_stock` ahora devuelve `(licitacion_stock, tablas_actualizadas)`, procesa los sorteos secuencialmente y aplica `_aplicar_salida_licitacion`; `sorteos_en_ventana` acepta `sorteos` y sólo exige el mes de sorteo dentro de la ventana |
| `synthetic.py` | `generar_tablas(sorteos_licitacion=SORTEOS, aplicar_salida_licitacion=True)`; comentario de fuentes junto a `PARAMETROS_POR_PERFIL` (objetivo 3) |
| `evaluate.py` | `bootstrap_ic_segmento`, `bootstrap_auc_segmentos`, `N_BOOT_REFERENCIA`, `SEGMENTOS_BOOTSTRAP` |
| `run.py` | reporte `licitacion_efecto_cartera.csv` (objetivo 1) y `bootstrap_auc_segmento.csv` / `bootstrap_captura_segmento.csv` (objetivo 2) |
| `tests/` | 6 tests nuevos (39 en total): retroalimentación de la cartera, segundo sorteo parametrizable, `cortes_evaluables` con dos sorteos, bootstrap por segmento (×3: cruce de 0,5, reproducibilidad, edad+saldo); `test_run_end_to_end` ampliado para verificar los 2 reportes nuevos |

### E. Runtime y limitaciones

`python3 -m afp_lab.churn_traspaso.run --perfil cuprum` pasó de ~45 s (ronda 2) a **~60 s**: los ~15 s
adicionales son una segunda llamada a `generar_tablas` (con `aplicar_salida_licitacion=False`) sólo
para el reporte comparativo del objetivo 1, y unos segundos más de los 2×200 remuestreos bootstrap del
objetivo 2 (9 segmentos × 200 × 2 métricas). Sigue muy por debajo del límite de 5 min.
`python3 -m pytest afp_lab/churn_traspaso/tests -q` corre los 42 tests en ~30 s (límite: 2 min).

**Limitaciones de esta ronda**:

* El objetivo 3 queda **acotado**: no se obtuvo la fuga bruta mensual de Cuprum (limitación de las
  fuentes públicas accesibles, no de tiempo), así que el parámetro no se recalibró. Ver §C.
* El segundo sorteo (2029-08) sólo se simula con una ventana extendida en un test; la corrida de
  referencia no lo incluye (decisión justificada en §A, no es una limitación de tiempo).
* `bootstrap_ic_segmento` no aplica desempate reproducible en `metrica="captura"` (a diferencia de
  `captura_top`): con probabilidades continuas de LightGBM los empates son prácticamente inexistentes,
  pero si se usara con un score con muchos valores repetidos habría que agregarlo.
