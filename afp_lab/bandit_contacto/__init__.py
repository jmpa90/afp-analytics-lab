"""Proyecto 09 - Bandit contextual de contacto para retencion (ronda 4).

Pregunta de negocio: cada mes, para cada afiliado identificado como "en riesgo" de traspaso,
decidir entre {sin contacto, email, llamada, visita de ejecutivo, oferta de ahorro/APV} para
maximizar la retencion neta de costo. Se aprende online (bandit contextual) y se valida offline
antes de "desplegar" con evaluacion off-policy (IPS, SNIPS, doubly robust).

Modulos:
    synthetic.py  reutiliza `afp_lab.churn_traspaso.synthetic.generar_tablas` (universo + tablas
                  operacionales de 02) y agrega las tablas propias de este proyecto: preferencias
                  de no-contacto (observable) y la respuesta latente por brazo (`lat_`, nunca
                  visible para las politicas).
    features.py   reutiliza `afp_lab.churn_traspaso.features.construir_features_multi` como
                  contexto y agrega el score de riesgo (heuristica de negocio) y la matriz
                  numerica estandarizada que usan LinUCB/Thompson lineal.
    model.py      politicas: heuristica de logging (eps=0.1), baselines (sin contacto,
                  top-k-por-riesgo-visita), epsilon-greedy, LinUCB, Thompson lineal, Thompson con
                  bootstrap de LightGBM; simulacion online mensual con recompensa rezagada.
    evaluate.py   evaluacion off-policy (IPS, SNIPS, DR) contra el valor verdadero del simulador.
    run.py        demo end-to-end (`python -m afp_lab.bandit_contacto.run`).

Todos los supuestos economicos y de diseno numerados abajo se documentan tambien en README.md
("Datos sinteticos" y "Decisiones tomadas").
"""
from __future__ import annotations

# el horizonte de resolucion de la recompensa es el mismo que el target de 02 (6 meses): se deja
# como constante propia (no se importa) porque aqui mide otra cosa (retraso de la recompensa del
# bandit, no la ventana de un target de clasificacion), aunque el valor coincida por diseno del brief.
HORIZONTE_MESES = 6
MESES_DESFASE_RIM = 2  # idem 02: la cotizacion del mes m se conoce en m+2 (usado por el contexto)

BRAZOS = ("SIN_CONTACTO", "EMAIL", "LLAMADA", "VISITA", "APV")
N_BRAZOS = len(BRAZOS)
IDX_BRAZO = {b: i for i, b in enumerate(BRAZOS)}

# --------------------------------------------------------------------- supuestos economicos
# Declarados como pide el brief: nombre, valor y por que. Todo en CLP nominales (sin indexar).
COSTO_OFERTA_APV_CLP = 7_000  # SUPUESTO: incentivo/beneficio promedio otorgado en la oferta de APV
COSTOS_BRAZOS_CLP = {
    "SIN_CONTACTO": 0,
    "EMAIL": 200,          # SUPUESTO (brief): envio automatizado, costo marginal bajo
    "LLAMADA": 2_000,      # SUPUESTO (brief): call center
    "VISITA": 15_000,      # SUPUESTO (brief): visita de ejecutivo, el brazo mas caro
    "APV": 3_000 + COSTO_OFERTA_APV_CLP,  # SUPUESTO (brief): "3.000 CLP mas costo de la oferta"
}
TASA_COMISION_ANUAL = 0.0144  # SUPUESTO (brief): 1,44% anual sobre la RIM (remuneracion imponible)

# --------------------------------------------------------------------- restricciones operativas
PRESUPUESTO_MENSUAL_CONTACTOS = 400  # SUPUESTO: tope operativo mensual (4% de 10.000 afiliados);
# ver README "Decisiones tomadas": la poblacion "en riesgo" de cada mes se define como el top-N por
# score de riesgo (N = este presupuesto), asi que el presupuesto acota directamente el tamano del
# pool mensual en vez de exigir un segundo paso de racionamiento.
MESES_ENTRE_CONTACTOS = 3  # maximo un contacto (brazo pagado) cada 3 meses por afiliado
EPSILON_LOGGING = 0.10     # exploracion de la politica historica (heuristica por riesgo)
P_OPT_OUT = 0.03           # SUPUESTO: fraccion de afiliados que pidio no ser contactada

# --------------------------------------------------------------------- respuesta latente
TIPOS_RESPUESTA = ("PERSUADIBLE", "INDIFERENTE", "SLEEPING_DOG")
P_TIPO_RESPUESTA = (0.30, 0.45, 0.25)  # SUPUESTO: mezcla de heterogeneidad al contacto (coherente con 07)

# delta en log-odds de traspaso a 6 meses que produce CADA brazo pagado, segun el tipo latente del
# afiliado (SUPUESTOS del generador, ver README p.2): negativo = el contacto retiene, positivo = el
# contacto empuja al traspaso (sleeping dogs). SIN_CONTACTO siempre tiene delta 0 (linea base).
DELTA_LOGODDS = {
    "PERSUADIBLE": {"EMAIL": -0.10, "LLAMADA": -0.35, "VISITA": -0.55, "APV": -0.65},
    "INDIFERENTE": {"EMAIL": -0.02, "LLAMADA": -0.03, "VISITA": -0.03, "APV": -0.05},
    "SLEEPING_DOG": {"EMAIL": 0.05, "LLAMADA": 0.25, "VISITA": 0.45, "APV": 0.10},
}
