# Brief 05 · Riesgo de empleador: declaración y no pago (DNP) y pago tardío

**Responsable**: DS-2. **Ronda**: 2. **Gerente**: revisa contra `CONVENCIONES.md` §7.

## Pregunta de negocio

Cobranza y proyección de recaudación: ¿qué empleadores no pagarán (DNP) o pagarán tarde la planilla del
próximo mes, y cuánta masa imponible de nuestros afiliados está en riesgo? El área de cobranza puede
gestionar al 10 % de los empleadores al mes; queremos que ese 10 % concentre la mayor masa en riesgo.
Segunda salida: masa imponible esperada en DNP por mes (para proyección de recaudación).

## Datos

Partir de `generar_universo(perfil="cuprum")`. El universo trae `empleadores_latente` con parámetros de
pago (`p_pago_*`, `p_dnp`, `p_rectificacion`): se usan **sólo para generar**, nunca como features. Si
`afp_lab/common/planilla.py` existe, usar su tabla de planillas; si no, construir en `synthetic.py`:

| Tabla | Grano | Contenido mínimo |
|---|---|---|
| `planillas` | empleador × periodo | n_trabajadores, masa imponible, `periodo_recepcion` (m+1 normal; anticipado; tardío; sin planilla), `estado` (PAGADA / DNP / SIN_DECLARAR), fecha de regularización de la DNP, rectificaciones |
| `boletin_infractores` | empleador × mes | inclusión en el Boletín de Infractores Laborales y Previsionales (público, DT) tras k meses de deuda |
| `boletin_concursal` | evento | reorganización / liquidación (Superir), con fecha de publicación |
| `shocks_empleador` | latente | episodios de estrés financiero que elevan la probabilidad de DNP y tardanza durante varios meses, con recuperación o quiebra |

Mecanismo del target (documentar parámetros): la DNP tiene **persistencia** (un empleador en DNP tiende a
repetir), sube en diciembre–marzo (gratificación, vacaciones), en micro y pequeñas empresas, en
construcción, agricultura, turismo y servicios; cae en grandes empresas y sector público (nunca DNP).
Los episodios de estrés terminan en regularización (pago de la deuda) o en liquidación. Tasa objetivo:
3–6 % de las planillas del mes en DNP; 5–10 % con recepción tardía (≥ m+2). Ruido para AUC 0,75–0,90.

## Target y población

* Fila = empleador × `periodo_corte` (mes de decisión). Conocido: planillas con `periodo_recepcion ≤ corte`
  (la planilla del mes `corte-1` llega en `corte` si es puntual), boletines publicados ≤ corte.
* Población: empleadores con al menos una planilla en los 12 meses previos y ≥ 1 trabajador afiliado.
* `y_dnp = 1` si la planilla del periodo `corte` (que debería llegar en `corte+1`) es DNP o no se declara
  en `corte+1`; `y_tardio = 1` si llega con lag ≥ 2 o no llega. Variante a 3 meses: alguna DNP en
  `corte … corte+2`. Elegir la principal y justificar; reportar ambas.
* `masa_en_riesgo = y · masa imponible del periodo`.

## Features (mínimo)

Historia de pago del empleador (as-of): tasa de puntualidad 3/6/12 m, DNP en 3/6/12 m, meses desde la
última DNP, lag promedio de recepción, rectificaciones, racha de puntualidad; tamaño y dotación (n
trabajadores, variación 3/12 m, altas/bajas AFC), masa imponible y su variación, % contratos plazo fijo,
rubro, región, antigüedad como empleador en la AFP; boletines (infractores, concursal); calendario y
macro (mes, desempleo, IMACEC si se agrega). Nada de `p_*` ni de shocks latentes.

## Modelo y evaluación

* Baselines: persistencia (estado de la última planilla conocida); regla "DNP en 3 m o boletín".
* LightGBM binario calibrado; split temporal con gap = horizonte; ≥ 12 cortes de test u orígenes móviles.
* Métricas de `common/metrics` por segmento (tamaño, rubro, región) + **negocio**: masa imponible en
  riesgo capturada gestionando al 10 % y 20 % de los empleadores; error del agregado mensual de masa en
  DNP esperada vs real (±10 %).

## Entregables y aceptación

Según `CONVENCIONES.md`. AUC test ≥ 0,80; captura de masa en riesgo al 10 % ≥ 50 % y mejor que la regla;
error del agregado mensual ≤ 10 % en ≥ 10 de 12 meses; runtime < 5 min con 20.000 afiliados; tests < 2 min.
