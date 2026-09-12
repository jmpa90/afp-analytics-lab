"""Tablas de equivalencia de la planilla Previred / Compendio de Normas de la Superintendencia
de Pensiones (fuente: Previred, tablas de equivalencia septiembre 2025; ver
`docs/normativa_tablas_afp.md`). Códigos y glosas oficiales para que los proyectos del
laboratorio usen los mismos valores que verían en producción, sin tener que redescubrirlos.

Este módulo es puramente declarativo (diccionarios); `planilla.py` es el que genera datos.
"""
from __future__ import annotations

# --------------------------------------------------------------------- tipo de pago (línea)
TIPO_PAGO = {
    "01": "Remuneraciones del mes",
    "02": "Gratificación",
    "03": "Bono Ley 20.058 (asignación familiar)",
    "04": "Diferencias de remuneraciones",
    "05": "Subsidio (licencia médica)",
    "06": "Aguinaldo / bono (no imponible AFP)",
}

# --------------------------------------------------- movimiento de personal (Tabla N° 7)
MOVIMIENTO_PERSONAL = {
    0: "Sin movimiento",
    1: "Contratación a plazo indefinido",
    2: "Retiro (término de la relación laboral)",
    3: "Subsidio (licencia médica)",
    4: "Permiso sin goce de sueldo",
    5: "Incorporación en el lugar de trabajo",
    6: "Accidente del trabajo",
    7: "Contratación a plazo fijo",
    8: "Cambio de contrato de plazo fijo a indefinido",
    11: "Otros movimientos (ausentismo)",
    12: "Reliquidación, premio, bono",
    13: "Suspensión por acto de autoridad (Ley 21.227)",
}
# Códigos que este generador NO produce (documentado, no es un olvido): 5 y 6 no tienen una regla
# propia en el brief; 13 (Ley 21.227, suspensión de la relación laboral por acto de autoridad) es
# un mecanismo excepcional (pandemia) fuera del alcance de esta simulación.
CODIGOS_NO_GENERADOS = (5, 6, 13)

# ------------------------------------------------------------------------- tipo de trabajador
TIPO_TRABAJADOR = {
    "ACTIVO": "Trabajador activo",
    "ACTIVO_MAYOR": "Activo mayor de 65 años (hombre) o 60 años (mujer)",
    "PENSIONADO_COTIZA": "Pensionado que cotiza",
    "PENSIONADO_NO_COTIZA": "Pensionado que no cotiza",
}

# ------------------------------------------------------------------------------ estado / tipo
ESTADO_PAGO = {"PAGADA": "Pagada", "DNP": "Declarada y no pagada"}
TIPO_MOVIMIENTO_PLANILLA = {
    "DECLARACION": "Declaración original",
    "RECTIFICACION": "Rectificación de una declaración anterior",
    "PAGO_DNP": "Pago posterior de una declaración y no pago (DNP)",
}

# ------------------------------------------------------------- aporte adicional del empleador
# Ley 21.735 (reforma de pensiones, publicada 26-mar-2025): cotización adicional de cargo del
# empleador, gradual. Cifra inicial verificada (docs/contexto_afp_cuprum.md §3): 3,5 % desde las
# remuneraciones de agosto de 2025 (cotización total sube a 13,5 %). El cronograma completo hasta
# 8,5 % NO está confirmado por la SP en las fuentes consultadas ("verificar cronograma"); como el
# horizonte simulado del laboratorio no pasa de 202609, solo este primer escalón cae dentro de la
# ventana de datos y es el único que se implementa. Se documenta como supuesto, no como norma.
APORTE_EMPLEADOR_CRONOGRAMA = [
    (202508, 0.035),
]


def tasa_aporte_empleador(periodo: int) -> float:
    """Tasa de aporte adicional del empleador vigente en `periodo` (Ley 21.735)."""
    vigente = 0.0
    for p, tasa in APORTE_EMPLEADOR_CRONOGRAMA:
        if periodo >= p:
            vigente = tasa
    return vigente


# ------------------------------------------------------------------------- tasas de cotizacion
TASA_COTIZACION_OBLIGATORIA = 0.10   # 10 % de la renta imponible AFP (cuenta individual)
TASA_SIS = 0.0153                    # Seguro de Invalidez y Sobrevivencia, de cargo del empleador
