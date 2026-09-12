# 02 · Propensión a traspaso a otra AFP (churn de afiliados)

Modelo que estima, para cada afiliado vigente y cada mes de corte, la **probabilidad de que solicite
un traspaso a otra AFP en los próximos 6 meses**, y ordena la cartera para que el equipo comercial
use su capacidad de contacto (10 % de la cartera al mes) donde más traspasos y más saldo administrado
se salvan.

```bash
python -m afp_lab.churn_traspaso.run --n-afiliados 20000
python -m pytest afp_lab/churn_traspaso/tests -q
```

---

## 1. Pregunta de negocio y decisión que apoya

| | |
|---|---|
| **Pregunta** | ¿Qué afiliados se traspasarán a otra AFP en los próximos 6 meses? |
| **Decisión** | A quién llama/visita el ejecutivo de retención. La restricción operativa es dura: el equipo contacta como máximo al **10 % de la cartera al mes**. |
| **Métrica de negocio** | % de los traspasos y % del **saldo en riesgo** capturados en ese 10 % (y en el 5 % y el 20 %, para dimensionar el equipo). |
| **Pregunta secundaria** | Saldo en riesgo esperado por afiliado = `P(traspaso) × saldo`, para presupuestar la caída de activos administrados y priorizar por valor y no sólo por probabilidad. |

El modelo no decide el traspaso: entrega un **orden** (a quién contactar) y un **nivel** (cuánto saldo
está en juego). Por eso se calibran dos salidas distintas (ver §5).

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

### Mercado (`AFP_ANDES` es la propia; 5 competidoras ficticias)

* **Comisiones** (% de la renta imponible, tramos en `COMISIONES`): propia 1,27 → 1,19 (2024-01) →
  1,14 (2025-07); competidoras entre 0,58 y 1,45. *Por qué:* el rango real del sistema chileno va de
  ~0,58 % a ~1,45 % y la AFP cara reacciona bajando la comisión.
* **Rentabilidad 12 m por fondo y AFP**: paseo aleatorio mensual del sistema con media/volatilidad
  por multifondo (A 6,2 %/16 % anual … E 2,2 %/3,5 %) más una desviación AR(1) por AFP de pocos
  décimos de punto. La propia arrastra un **castigo de −1,0 pp anual entre 2024-01 y 2025-12**.
  *Por qué:* las diferencias de rentabilidad entre AFP son chicas y persistentes, y un mal tramo de
  rentabilidad relativa es el gatillante clásico de una ola de traspasos.
* **Licitación de nuevos afiliados** cada 24 meses (`LICITACIONES`): la adjudicataria recibe a todos
  los que se afilian al sistema y ellos **deben permanecer 24 meses** (regla real). La propia gana la
  licitación 2022-08 → 2024-07, así que en el período de test hay una cohorte recién liberada, joven
  y de antigüedad baja.

### Cuenta obligatoria

* Saldo inicial = `0,10 × RIM_ref × ((1+i)^m − 1)/i × densidad × lognormal(σ=0,35)` con
  `i = 0,35 %` mensual real, `densidad = 0,68` y `m` = meses de afiliación previos. Da una mediana de
  **8,8 millones de CLP** y una media de ~17 millones, coherente con los saldos por edad en Chile.
* Mensual: `saldo_t = saldo_{t−1} × (1 + r_fondo_propia_t) + 0,10 × RIM_{t−2}` — el aporte entra con
  **2 meses de desfase**, igual que la recaudación real, así que el saldo del corte es conocido al corte.
* Fondo inicial por asignación etaria por defecto (B hasta 35, C hasta 50 M / 55 H, D después) y un
  35 % de afiliados "activos" que eligen A (jóvenes) o E (mayores).

### Relación con la AFP

| Mecanismo | Parámetro base mensual | Modulación | Por qué es razonable |
|---|---|---|---|
| Contacto de **vendedor de la competencia** | 0,016 | ×2,6 en el decil superior de saldo, ×1,45 en marzo y agosto | la fuerza de venta persigue cuentas grandes y las campañas son estacionales |
| Contacto de **ejecutivo propio** | 0,012 | ×3 si hubo reclamo o visita de la competencia en 3 m, ×2,2 por saldo alto | la retención es **reactiva**: llega después de la señal de riesgo |
| Call center / campaña digital | 0,030 / 0,075 | campaña ×1,45 en marzo y agosto | contacto masivo, poco dirigido |
| **Reclamos** | 0,0040 | ×2,2 si la rentabilidad propia va >0,5 pp bajo el sistema, ×1,6 si la comisión está >0,5 pp sobre la más barata, ×1,8 sobre 55 años | ~0,13 reclamos por afiliado-año, del orden de lo que reporta la industria |
| **Uso digital** | Poisson(λ=0,55) logins, ×1,8 jóvenes/universitarios | **×3,5 durante la intención de traspaso** | el que está por irse revisa su cuenta |
| **Descarga de certificado** | 0,012 | **0,40 durante la intención** | el certificado de saldo y la cartola se piden justo antes de traspasarse |
| **Cambio de fondo** | 0,0055 | ×3 afiliado activo, ×1,8 si la rentabilidad del fondo va mal | el que cambia de fondo es un afiliado "activo", más propenso a moverse |

### Mecanismo del target (hazard mensual, escala logit, `ParametrosChurn`)

```
logit(h) = −6,20
  + 1,05 · brecha_comisión (pp sobre la más barata)
  + 0,55 · max(0, −brecha_rentabilidad_12m del fondo del afiliado, en pp)
  + 1,45 · contacto de la competencia en 3 m   + 0,55 · si el resultado fue INTERESADO
  + 0,95 · reclamo en 6 m                      + 0,40 · si se resolvió en contra
  − 0,85 · contacto de ejecutivo propio en 3 m
  + 0,036 · (40 − edad)⁺        + 0,55 · log(RIM_ref / IMM)      − 0,75 · log1p(años en la AFP)
  + 0,33 · min(cambios de fondo 12 m, 3)
  − 1,15 · cesante              − 1,60 · retirado
  + 0,35 (marzo) + 0,30 (agosto)
  + u_afiliado ~ N(0; 0,55²)    + ε_mes ~ N(0; 0,15²)
```

y `h = 0` durante los 24 meses de permanencia obligatoria de los afiliados llegados por licitación.

**La secuencia importa**: el afiliado primero recibe contactos y reclamos; si se dispara el hazard
entra en un estado latente de *intención* con un retardo de 0 a 4 meses
(`p = 0,10 / 0,22 / 0,28 / 0,25 / 0,15`); durante la intención sube su uso digital y pide el
certificado; sólo al final firma la solicitud, que es efectiva al mes siguiente. Así las señales
digitales son **anteriores** al evento y no contemporáneas (que sería leakage disfrazado).

`u_afiliado` es un efecto aleatorio **no observable**: es lo que fija el techo de AUC y evita que la
simulación sea determinista (CONVENCIONES §2).

**Chequeos de la corrida de referencia** (`reportes/resumen_simulacion.csv`): tasa de traspaso
anualizada **4,33 %** de la cartera y **5,28 % sobre la cartera cotizante** (el brief pide 4–7 %),
3.852 salidas y 2.489 entradas, 1,91 contactos y 0,13 reclamos por afiliado-año, saldo mediano
8,8 MM CLP.

## 3. Target, población, horizonte y exclusiones

* **Fila** = afiliado × `periodo_corte`, cortes mensuales móviles.
* **Target**: `y = 1` si existe un traspaso de SALIDA con **mes de solicitud** en `corte+1 … corte+6`
  (no el mes efectivo: la decisión comercial hay que tomarla antes de la firma).
* **Población**: afiliados vigentes en la AFP propia al corte (existe fila de `saldos` en ese mes, lo
  que implica afiliación a la AFP ≤ corte y traspaso aún no efectivo).
* **Exclusiones**, todas con información conocida al corte:
  * `estado` laboral `RETIRADO` en `corte−2` (pensionados fuera de la fuerza de trabajo);
  * afiliados que **ya presentaron** la solicitud de traspaso en un mes ≤ corte (están saliendo: no
    son un objetivo de retención y contaminarían la etiqueta con ceros estructurales);
  * cortes sin los 6 meses de horizonte completos (`cortes_evaluables`) y los primeros 14 meses de
    burn-in, que no tienen historia de 12 meses.
* Tasa base resultante: **3,1 %** en entrenamiento, **1,43 %** en test (la caída es real, ver §8).

## 4. Features (68 columnas, todas as-of)

Regla general: el **panel laboral se lee en `corte−2`** (la cotización del mes *m* llega en *m+2*,
igual que en el proyecto 01) y las tablas **operacionales propias de la AFP se leen en `corte`**
(contactos, reclamos, uso digital, cambios de fondo, saldos y mercado, que la AFP conoce al día).

| Familia | Features | Regla as-of |
|---|---|---|
| Historia laboral y renta | `rim_l1`, `rim_media_3`, `rim_media_12`, `rim_cv_12`, `rim_ref`, `rim_tendencia_3_12`, `densidad_6/12`, `meses_desde_ult_cotiza`, `estado_laboral`, `tipo_contrato`, `rubro`, `tamano_empleador`, `meses_con_empleador`, `meses_desde_cambio_empleador`, `n_cambios_empleador_12` | ventanas que terminan en **`corte−2`** |
| Cuenta | `saldo_obligatorio`, `log_saldo`, `saldo_sobre_rim_ref`, `fondo`, `cambios_fondo_3m/12m`, `meses_desde_cambio_fondo`, `antiguedad_afp_meses` | hasta **`corte`** |
| Afiliado | `edad`, `sexo`, `region`, `nivel_educacional`, `tipo_afiliado`, `meses_desde_afiliacion`, `meses_hasta_pension` (60 M / 65 H) | atributos de maestro |
| Relación comercial | `contactos_{canal}_1m/3m/6m` (4 canales), `contactos_competencia_interesado_6m`, `meses_desde_contacto_competencia`, `meses_desde_contacto_ejecutivo` | hasta **`corte`** |
| Reclamos | `reclamos_6m/12m`, `reclamos_abiertos_12m`, `reclamos_desfavorables_12m`, `meses_desde_reclamo`, `tipo_ultimo_reclamo` | hasta **`corte`** |
| Uso digital | `logins_1m/3m`, `consultas_saldo_3m`, `certificados_1m/3m`, `ratio_logins_3m_12m` | hasta **`corte`** |
| Mercado | `comision_propia`, `comision_min_mercado`, `brecha_comision_pp`, `brecha_comision_rel`, `rent_12m_propia_fondo`, `rent_12m_sistema_fondo`, `brecha_rentabilidad_pp` (del fondo del afiliado), `afp_licitacion`, `licitacion_propia`, `mes` | mes = **`corte`** |

Las columnas `lat_*` del universo y los parámetros `p_*` de empleadores **no entran** (el universo ya
las separa y un test lo verifica).

### Cómo se garantiza que no hay leakage

`construir_features(tablas, periodo_corte)` **primero recorta las tablas** (`recortar_tablas`: panel
hasta `corte−2`, operacionales hasta `corte`, traspasos hasta la solicitud ≤ corte) y recién después
calcula: cualquier registro posterior simplemente no existe para el cálculo.
`construir_features_multi` calcula muchos cortes de una vez sobre matrices afiliado × mes con
ventanas que sólo miran hacia atrás (`cumsum` desplazado, "meses desde" y forward-fill), que es lo
que usa el `run`. Los tests cierran el círculo:

* `test_sin_leakage_alterar_el_futuro_no_cambia_las_features`: multiplica ×17 la RIM posterior al
  corte, marca todo como RETIRADO, borra la mitad de las filas futuras, inyecta uso digital y 50
  traspasos falsos en `corte+1` → **todas las features quedan idénticas**;
* `test_features_multi_igual_a_corte_unico`: la ruta matricial coincide exactamente con la ruta de
  un corte recortado (así el test anterior también cubre la ruta rápida);
* `test_features_asof_verificadas_a_mano`: dos casos a mano — `rim_l1` es la RIM del mes `corte−2` (y
  **no** la del corte) y `contactos_vendedor_competencia_3m` es el conteo exacto en
  `[corte−2, corte]`.

### Split temporal (gap de 6 meses)

| Partición | Cortes | Rango | Filas | Tasa base |
|---|---|---|---|---|
| train | 21 | 2022-03 → 2023-11 | 319.588 | 3,12 % |
| valid | 3 | 2024-11 → 2025-01 | 44.870 | 1,64 % |
| test | 3 | 2026-01 → 2026-03 | 43.448 | 1,43 % |

El último target de entrenamiento (2024-05) queda 6 meses antes del primer corte de validación, y el
último target de validación (2025-07) queda 6 meses antes del primer corte de test (`split_temporal`,
verificado en `test_split_temporal_respeta_el_gap`).

## 5. Modelo y baselines

* **LightGBM binario** (`num_leaves 48`, `lr 0,05`, `min_data_in_leaf 200`, `feature_fraction 0,75`,
  `bagging 0,80`, `lambda_l2 5`, **`num_threads 2`**), early stopping de 60 rondas por AUC de
  validación. Categóricas nativas de LightGBM. Se detuvo en **151 árboles**.
* **Dos calibraciones**, ambas ajustadas sólo con validación, porque el score se usa para dos cosas:
  * `uso="ranking"` → **Platt** sobre el log-odds: estrictamente monótona, no genera empates, deja
    intactos el orden, el AUC, el lift y la captura. Es la que ordena la lista de contacto.
  * `uso="agregado"` → **isotónica por bloques** (40 bloques de validación, no observación a
    observación, para que ningún punto de la curva descanse en un puñado de casos). Ajusta mucho
    mejor el nivel para sumar `prob × saldo`, a costa de aplanar el score (el *average precision* cae
    de 0,351 a 0,268), por lo que **no** se usa para ordenar.
* **Baselines obligatorios**:
  * `baseline_tasa_base` (trivial): la tasa base de entrenamiento para todos.
  * `baseline_regla_negocio` (reglas del brief): reclamo en los últimos 3 meses **o** contacto de
    vendedor de la competencia en 3 meses **o** brecha de rentabilidad 12 m < −1 pp.
  * `baseline_regla_graduada` (añadido): cuenta cuántas de las tres condiciones se cumplen (0–3).
    Rompe los empates de la regla binaria y **ordena bastante mejor**; se reporta para no comparar el
    modelo contra un baseline artificialmente débil. Los empates se rompen con un desempate aleatorio
    reproducible (`captura_top(seed=7)`).

## 6. Resultados (20.000 afiliados, semilla 42, test = 43.448 filas, tasa base 1,43 %)

### Tabla principal

| Predictor | AUC | Average precision | Brier | Lift decil 1 | KS |
|---|---|---|---|---|---|
| **Modelo LightGBM (Platt)** | **0,806** | **0,351** | **0,0106** | **5,23** | **0,458** |
| Modelo, salida isotónica (agregados) | 0,804 | 0,268 | 0,0126 | 5,24 | 0,457 |
| Baseline tasa base | 0,500 | 0,014 | 0,0144 | 1,02 | 0,039 |
| Baseline regla de negocio (brief) | 0,589 | 0,018 | 0,5868 | 1,19 | 0,178 |
| Baseline regla graduada | 0,624 | 0,022 | 0,0961 | 2,27 | 0,182 |

Estable por corte: AUC 0,795 / 0,801 / 0,823 en 2026-01, 2026-02 y 2026-03.

### Métrica de negocio: captura contactando al X % de mayor score

| Predictor | 5 % — traspasos / saldo | **10 % — traspasos / saldo** | 20 % — traspasos / saldo |
|---|---|---|---|
| **Modelo** | **44,4 % / 46,7 %** | **52,3 % / 56,6 %** | **65,0 % / 66,3 %** |
| Regla graduada | 18,1 % / 22,7 % | 24,0 % / 29,9 % | 35,8 % / 40,8 % |
| Regla de negocio (brief) | 7,4 % / 8,5 % | 15,5 % / 16,7 % | 28,1 % / 30,0 % |
| Tasa base (aleatorio) | 6,1 % / 7,2 % | 11,6 % / 12,0 % | 21,5 % / 21,1 % |

### Por segmento (AUC modelo vs mejor baseline de reglas; segmentos ≥ 5 % de las filas)

**El modelo gana en los 24 segmentos grandes**, por entre +0,10 y +0,24 de AUC
(`reportes/segmentos_modelo_vs_baseline.csv`; el `run` imprime una alerta si algún segmento se
pierde, y en esta corrida el archivo `segmentos_perdidos.csv` no se genera).

| Segmento | AUC modelo | AUC baseline |
|---|---|---|
| Estado laboral: EMPLEADO (68,7 %) | 0,801 | 0,625 |
| Estado laboral: CESANTE (15,9 %) | 0,697 | 0,598 |
| Tramo de saldo 5–15 MM (43,4 %) / 15–40 MM (30,7 %) / 40 MM+ (8,7 %) | 0,813 / 0,804 / 0,804 | 0,612 / 0,661 / 0,623 |
| Tramo de renta 0–600k (23,8 %) / 600k–1,2 MM (43,6 %) / 2,5 MM+ (8,8 %) | 0,742 / 0,819 / 0,762 | 0,602 / 0,607 / 0,620 |
| Tramo de edad <30 (5,3 %) / 30–39 / 40–49 / 50–59 / 60+ | 0,755 / 0,808 / 0,819 / 0,789 / 0,723 | 0,624 / 0,565 / 0,628 / 0,622 / 0,510 |
| Fondo A / B / C / D / E | 0,808 / 0,800 / 0,785 / 0,791 / 0,795 | 0,608 / 0,572 / 0,568 / 0,548 / 0,676 |
| Tipo de afiliado: DEPENDIENTE (89,6 %) / INDEPENDIENTE (7,9 %) | 0,804 / 0,841 | 0,624 / 0,619 |

Los puntos más débiles son **cesantes** (AUC 0,697: por construcción casi no se traspasan, así que
queda poco que ordenar) y **60+** (0,723).

### Calibración y saldo en riesgo

| Versión del score | Saldo en riesgo esperado (MM CLP, 3 cortes) | Real | Sesgo agregado |
|---|---|---|---|
| Sin calibrar | 14.863 | 12.753 | **+16,6 %** |
| Platt (ranking) | 15.039 | 12.753 | **+17,9 %** |
| **Isotónica por bloques (agregados)** | 12.166 | 12.753 | **−4,6 %** |

La sobreestimación del score crudo es puro **drift de tasa base**: validación tiene 1,64 % y test
1,43 %. La isotónica lo corrige casi por completo; por eso el saldo en riesgo esperado se reporta con
esa salida y el ranking con Platt. Curva por decil en `reportes/calibracion.csv`: la isotónica está
dentro de ±0,5 pp en 9 de 10 deciles y subestima 2,1 pp en el decil superior.

### Importancia de features (gain, top 13)

| # | Feature | % gain | | # | Feature | % gain |
|---|---|---|---|---|---|---|
| 1 | `logins_1m` | 17,8 | | 8 | `region` | 3,4 |
| 2 | `brecha_rentabilidad_pp` | 7,8 | | 9 | `certificados_1m` | 3,4 |
| 3 | `ratio_logins_3m_12m` | 7,8 | | 10 | `rubro` | 3,3 |
| 4 | `certificados_3m` | 7,7 | | 11 | `meses_desde_afiliacion` | 2,9 |
| 5 | `consultas_saldo_3m` | 6,8 | | 12 | `rim_ref` | 2,0 |
| 6 | `antiguedad_afp_meses` | 5,9 | | 13 | `edad` | 1,9 |
| 7 | `logins_3m` | 4,8 | | | | |

El bloque digital (logins + consultas + certificados) concentra ~48 % del gain, la brecha de
rentabilidad del fondo del afiliado 7,8 % y la antigüedad/tiempo en la AFP ~9 %. Es el orden
esperado: el uso digital es la señal *próxima* al evento y el resto describe el riesgo estructural.

### 5 casos explicados (`reportes/casos_explicados.csv`)

Se eligen a propósito casos distintos, no los 5 de mayor score (que son el mismo perfil repetido):

| Caso | Afiliado / corte | Prob | Real | Lectura |
|---|---|---|---|---|
| `top_1` | 7824 / 2026-03 | 0,997 | 1 | 13 consultas de saldo y **2 descargas de certificado** en 3 meses (+2,65 y +2,00 en logit): está armando la carpeta del traspaso |
| `top_2` | 1872 / 2026-03 | 0,997 | 1 | mismo patrón digital; saldo 7,8 MM |
| `falso_positivo` | 14613 / 2026-01 | 0,962 | 0 | 9 logins en el mes y un reclamo hace 2 meses, pero **259 meses de antigüedad** (−0,40) y sin descarga de certificado: se quedó. Es un falso positivo *barato*: saldo 118 MM, vale la llamada igual |
| `positivo_mayor_saldo` | 14134 / 2026-03 | 0,933 | 1 | 108 MM CLP en riesgo, certificado descargado en el mes; el caso que justifica ordenar también por saldo |
| `falso_negativo` | 649 / 2026-02 | 0,002 | 1 | 288 meses de antigüedad, renta baja (270 k), sin empleador y sin actividad digital: el traspaso viene del efecto aleatorio no observable `u_afiliado`, que es exactamente el ruido irreducible de la simulación |

## 7. Lectura de negocio

Con la cartera simulada (15.451 afiliados vigentes en promedio; 14.483 y ≈250.000 MM CLP
administrados en cada mes de test):

* Contactando al **10 % de mayor score (≈1.450 afiliados/mes)** el equipo alcanza el **52 % de los
  traspasos** y el **57 % del saldo en riesgo** — unos **2.400 MM CLP mensuales** de los ~4.250 MM que
  se van. La regla de negocio actual del brief captura 15,5 % / 16,7 %; el modelo **multiplica por
  3,4 la captura de traspasos y por 3,4 la de saldo con el mismo costo comercial**.
* Si el equipo sólo alcanza el **5 %**, todavía captura el 44 % de los traspasos: el retorno marginal
  entre el 5 % y el 10 % ya es decreciente, lo que sugiere dimensionar el equipo cerca del 10 % y no
  del 20 %.
* La salida isotónica permite publicar un **saldo en riesgo esperado mensual** con sesgo agregado de
  −4,6 %, que sirve para presupuestar activos administrados y para priorizar por valor
  (`prob × saldo`) cuando dos afiliados tienen la misma probabilidad.
* Las palancas accionables que el modelo ordena son las del negocio: la **descarga del certificado de
  saldo dispara la alerta a días del traspaso** (vale la pena un gatillo operacional inmediato, no
  esperar al scoring mensual), la **brecha de rentabilidad del fondo del afiliado** identifica al
  grupo estructuralmente en riesgo (campaña de comunicación por fondo) y la **antigüedad baja** marca
  a la cohorte de licitación recién liberada.

## 8. Decisiones tomadas, supuestos y limitaciones

**Decisiones ante ambigüedades del brief**

1. **"Panel laboral hasta `corte`" vs "RIM con 2 meses de desfase"**: se aplicó el criterio estricto
   y **todo** el panel laboral (RIM, estado, empleador, contrato, antigüedad) se lee en `corte−2`. Un
   afiliado puede quedar cesante en `corte−1` sin que la AFP lo sepa por la vía de la cotización;
   asumir lo contrario habría sido leakage encubierto. El aporte a la cuenta también se acredita con
   2 meses de desfase, de modo que el saldo del corte sí es conocido al corte.
2. **Ventana del universo**: el brief no fija el período. Se generó 2021-01 → 2026-09 (69 meses) en
   vez de arrancar en 2022-01 para tener 14 meses de burn-in y aun así 21 cortes de entrenamiento
   respetando dos gaps de 6 meses.
3. **`mercado` mes × AFP**: la rentabilidad 12 m por fondo se guardó en formato ancho
   (`rent_12m_A … rent_12m_E`) para respetar el grano mes × AFP que pide el brief en vez de abrir una
   tabla mes × AFP × fondo.
4. **Exclusión de solicitudes ya presentadas**: el brief sólo dice "no traspasados antes". Se excluyó
   además a quien ya firmó la solicitud en un mes ≤ corte (información conocida al corte). Sin esa
   exclusión se agregan ~0,4 % de filas que son ceros estructurales y que ensucian tanto la etiqueta
   como la lista de contacto (ya no son retenibles).
5. **Afiliados nuevos al sistema**: en Chile entran obligatoriamente a la adjudicataria de la
   licitación, así que los que se afilian fuera de la ventana en que gana la propia **no aparecen en
   la cartera** (7,0 % del universo queda sin ninguna fila). Es lo correcto, pero reduce el tamaño
   efectivo del panel.
6. **Historia laboral de los afiliados que entran por traspaso**: se asume que la AFP ve su historia
   de RIM previa (llega con el traspaso de la cuenta y vía Previred). Si la AFP real no la tuviera,
   habría que enmascarar las features laborales anteriores a `antiguedad_afp_meses`. Es un supuesto a
   confirmar.
7. **Dos calibradores en vez de uno**: la isotónica sola degradaba el ranking (AP 0,351 → 0,268) y
   Platt solo no corregía el nivel (sesgo +17,9 %). Se guardan ambos y cada uso toma el que
   corresponde.
8. **Baseline de reglas**: la regla binaria del brief marca al **59 % de la cartera** de test (la
   brecha de rentabilidad < −1 pp aplica a fondos completos), así que a nivel de score sólo tiene dos
   valores y su captura al 10 % es prácticamente aleatoria dentro del grupo marcado. Se reporta con
   desempate aleatorio reproducible (`captura_top(seed=7)`) y se agregó la variante graduada (0–3
   condiciones) como baseline exigente; el modelo también le gana a esa.

**Limitaciones**

* **Drift de tasa base**: la tasa cae de 3,12 % (train) a 1,43 % (test). Es un efecto de selección
  realista — los afiliados con `u_afiliado` alto se traspasan temprano y la cartera remanente es más
  fiel, y además la antigüedad promedio sube — pero implica que la calibración envejece rápido: en
  producción la isotónica hay que reajustarla cada mes con la cohorte madura más reciente.
* **La cola alta del score es demasiado segura** (probabilidades de 0,99). En la simulación, la
  combinación "descarga de certificado + logins disparados" es casi determinista; en la realidad el
  certificado se pide también por otros motivos (crédito hipotecario, trámites). El AUC global
  (0,806) sí queda en el rango 0,75–0,90 que pide el brief, pero el modelo se apoya más en la señal
  digital de lo que probablemente ocurre con datos reales.
* Segmentos flojos: cesantes (AUC 0,697) y 60+ (0,723); son poblaciones con muy pocos eventos.
* No se modela la **reentrada** de un afiliado que se fue y vuelve, ni el traspaso de cuentas de APV
  o el cambio de AFP por cambio de empleador con convenio; tampoco el efecto de red (familiares que
  se traspasan juntos).
* Los parámetros macro, las comisiones y las rentabilidades son **supuestos razonables para Chile**,
  no series oficiales; y las 6 AFP son ficticias.

**No cumplido**: ninguno de los criterios de aceptación del brief (AUC ≥ 0,75, lift decil 1 ≥ 3, el
modelo supera a la regla de negocio en la captura al 10 %, runtime < 5 min, tests < 2 min) quedó
fuera de rango en la corrida de referencia.

## 9. Cómo correr y tiempos

```bash
cd /home/user/afp-analytics-lab
python -m afp_lab.churn_traspaso.run --n-afiliados 20000          # salidas/churn_traspaso
python -m pytest afp_lab/churn_traspaso/tests -q
```

Opciones: `--salida DIR`, `--seed`, `--periodo-inicio`, `--periodo-fin`, `--n-test`, `--n-valid`,
`--sin-fuentes` (no escribe las tablas sintéticas en parquet).

| Paso | Segundos (20.000 afiliados, 4 núcleos, `num_threads=2`) |
|---|---|
| `generar_tablas` (universo común + 7 tablas propias) | 6,9 |
| Guardar fuentes en parquet | 0,9 |
| `construir_dataset` (27 cortes, 407.906 filas × 68 features) | 6,7 |
| `entrenar` (LightGBM + 2 calibradores) | 8,2 |
| Evaluar (métricas, segmentos, calibración, SHAP) | 2,3 |
| **Total** | **26,6 s** (límite: 5 min) |

`python -m pytest afp_lab/churn_traspaso/tests -q` → **14 tests en 6,3 s** (límite: 2 min).

Salidas en `salidas/churn_traspaso/` (en `.gitignore`): `fuentes/*.parquet`, `dataset.parquet`,
`modelo/` (`lightgbm.txt`, `calibrador_platt.csv`, `calibrador_isotonico.csv`, `modelo.json`) y
`reportes/*.csv` (resumen de simulación, split, métricas, captura, por corte, por segmento,
calibración, saldo en riesgo, importancias, casos explicados, tiempos).

## 10. Propuestas para `common/` y próximos pasos

**Propuestas para `common/`** (no se tocó nada fuera de `afp_lab/churn_traspaso/`):

1. `common/metrics.py`: `tabla_metricas_por_segmento` usa `df.groupby(col)`, que con columnas
   `category` devuelve grupos vacíos en pandas 2.x y revienta las métricas. Propongo
   `groupby(col, observed=True)`. Mientras tanto este proyecto castea los segmentos a `str` en
   `evaluate.agregar_segmentos`.
2. `common/metrics.py`: agregar una función de **captura por percentil**
   (`captura_top(y, score, peso, fracciones)` → % de eventos y % de peso capturados) con desempate
   reproducible. Es la métrica de negocio natural de cualquier proyecto de propensión (churn, APV,
   cese de cotización) y hoy cada proyecto la reescribe. La implementación está en
   `churn_traspaso/evaluate.py:captura_top` y es trivial de generalizar.
3. `common/metrics.py`: agregar `curva_calibracion(y, prob, n_bins)` y un `CalibradorPlatt`
   compartido; la calibración se pide en CONVENCIONES §4 para todo proyecto que agregue montos.
4. `common/calendario.py`: hoy `add_months`, `periodo_to_idx` y `rango_periodos` viven en
   `rim_nowcasting.calendario`, que importa PySpark. Todo proyecto pandas termina importando el
   módulo de otro proyecto (y arrastrando Spark). Propongo mover las funciones puras de periodo a
   `common/calendario.py` y dejar en `rim_nowcasting` sólo las expresiones Spark.
5. `common/splits.py`: `split_temporal` con gap (implementado en `features.py`) sirve igual para los
   proyectos 03 y 04.

**Próximos pasos del proyecto**

* **Gatillo diario** para la descarga de certificado de saldo: el scoring mensual llega tarde para la
  señal más fuerte; conviene una alerta operacional en cuanto se descarga.
* **Modelo de tiempo hasta el evento** (hazard mensual discreto en vez de una ventana fija de 6
  meses): permite responder "¿cuándo?" y no sólo "¿se va?", y usar horizontes distintos por campaña.
* **Uplift / incrementalidad**: hoy el modelo predice riesgo, no el efecto del contacto. Como en la
  simulación el ejecutivo propio es reactivo (llega *después* de la señal), el modelo puede aprender
  que "contacto del ejecutivo" correlaciona con riesgo. Un test A/B de retención y un modelo de
  uplift responderían a quién conviene llamar, que es la pregunta real.
* **Optimización de la lista**: maximizar saldo retenido esperado sujeto a capacidad (`prob × saldo ×
  efectividad del canal`) en vez de ordenar sólo por probabilidad.
* Recalibración mensual automática y monitoreo de drift de la tasa base (la caída de 3,1 % a 1,4 %
  entre train y test muestra que es el riesgo operacional principal).
* Explicabilidad SHAP persistida por afiliado para que el ejecutivo vea el motivo en su pantalla.
