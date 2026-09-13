# Brief 10 — `afp_lab/pronostico_agregado` (worktree `/home/user/wt/churn`, rama `ronda-5/timesfm`)

## Pregunta
¿Un modelo fundacional de series de tiempo (TimesFM de Google, sin entrenar) pronostica mejor que lo
que una AFP usaría hoy (estacional ingenuo, ETS/Theta, LightGBM global) las series agregadas que
importan a tesorería, operaciones y comercial? ¿Y sirve como baseline honesto para la renta
imponible por afiliado del proyecto 01?

## Modelo y licencias (verificado en Hugging Face, 2026-09-13)
* `google/timesfm-2.5-200m-pytorch` (Apache 2.0, usable en producción; paquete PyPI `timesfm`).
* `google/timesfm-3.0-pytorch` (24-ago-2026, licencia no comercial): solo evaluación, nunca producción.
* **Restricción del sandbox:** el proxy bloquea `huggingface.co`. El código debe cargar los pesos desde
  una ruta local configurable (`TIMESFM_CHECKPOINT`, default `models/timesfm-2.5-200m/`) y, si no
  existen, degradar con claridad: correr toda la evaluación con los baselines y dejar la columna
  TimesFM como "pendiente de pesos", nunca inventar. Documentar en el README cómo obtener los pesos
  (descarga desde Hugging Face en una máquina con acceso y copia a esa carpeta) y probar el adaptador
  con un checkpoint sintético diminuto o con mocks en los tests.

## Series (todas del universo sintético común, perfil `cuprum`, 20 000 afiliados salvo indicación)
1. **Recaudación mensual** (suma de RIM cotizada) total, por rubro, por región y por tramo de tamaño de
   empleador: ~25 series mensuales de 2022-01 a 2028-12 (`panel_mensual` + `common/planilla` para
   el mes de recepción, así la serie es "lo que llegó", no "lo devengado").
2. **Traspasos mensuales** de salida y de entrada (tablas de 02 `churn_traspaso`) total y por tramo
   de edad; incluir el efecto de la licitación 2027 en la ventana.
3. **Reclamos y contactos diarios** (tablas de 02/06) para dotación: series diarias con
   estacionalidad semanal, agregadas también a semanal.
4. **Declaraciones sin pago (DNP) mensuales** por tamaño de empleador (`common/planilla`).
5. **Serie real, si es alcanzable sin inventar:** traspasos netos por AFP de la SP. Como
   `spensiones.cl` está bloqueado, usar solo lo que ya está en `docs/contexto_afp_cuprum.md` como
   punto de referencia; si no hay serie, decirlo y no incluirla.

## Protocolo
* Backtesting de origen móvil: ≥ 12 orígenes mensuales (o 8 semanales para las diarias), horizontes
  1, 3, 6 y 12 meses; para diarias, 7 y 28 días. Misma ventana de contexto para todos los modelos.
* Modelos: estacional ingenuo, ETS y Theta (`statsforecast`), LightGBM global con lags y calendario,
  TimesFM 2.5 zero-shot (con cuantiles). Sin ajuste de hiperparámetros con el test.
* Métricas: WAPE, MASE, sesgo; cobertura empírica de los intervalos 80 % y 90 % (TimesFM y ETS) y
  su ancho. Tabla por familia de serie y por horizonte, con la mejor en negrita y una lectura de
  negocio por familia.
* Baseline en 01: para una muestra de 2 000 afiliados, serie mensual de RIM (rim_emp1+rim_emp2) con
  contexto as-of hasta corte−2 y horizontes 1 y 2 (mismos horizontes de 01); comparar TimesFM vs
  último valor conocido, media de 3 meses y LightGBM con lags; reportar WAPE global y por tipo de
  afiliado y explicar por qué gana o pierde frente a los números del README de 01 (sin pretender
  igualdad de protocolo).
* Runtime: corrida de referencia < 6 min en CPU (`torch.set_num_threads(2)`), tests < 60 s con
  series pequeñas y sin pesos reales.

## Entregables
Proyecto completo según `CONVENCIONES.md` (§1 estructura, §6 README): `synthetic.py` (construcción
de series), `features.py` (ventanas y calendario), `model.py` (adaptadores: naive, ETS/Theta,
LightGBM global, TimesFM local), `evaluate.py`, `run.py`, `tests/`, `README.md` con tablas,
cobertura, lectura de negocio, licencias y "cómo se desplegaría". Nada de git desde el agente.
