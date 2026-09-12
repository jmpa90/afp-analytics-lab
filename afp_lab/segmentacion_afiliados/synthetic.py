"""Datos de `segmentacion_afiliados` (proyecto 06).

Este proyecto **no redefine mecanismos**: parte del universo comun con el perfil `cuprum`
(nuestra propia cartera, `afp_lab.common.synth_base.generar_universo`) para las features
"nucleo" de comportamiento (renta, densidad, estabilidad laboral, edad, saldo obligatorio) y
reutiliza, **sin modificarlos**, los generadores de los proyectos 02 (`churn_traspaso`) y 03
(`propension_apv`) para todo lo que ya simulan y el brief pide no duplicar: saldo/fondo,
cambios de fondo, uso digital, contactos, reclamos, antiguedad en la AFP, aportes
voluntarios. `evaluate.py` usa ademas las tablas `traspasos` (02) y `aportes_voluntarios`
(03) para la validacion de utilidad de negocio (tasa de traspaso / aporte APV por segmento).

**Limitacion documentada (no se puede evitar sin modificar 02/03, fuera de alcance)**:
`churn_traspaso.generar_tablas` y `propension_apv.generar_tablas` llaman a
`synth_base.generar_universo(..., seed=seed)` **sin** parametro `perfil` (siempre
"sistema"), mientras que este proyecto usa `perfil="cuprum"` para su propio universo. Con
el mismo `n_afiliados`, `periodo_inicio` y `seed`, el `afiliado_id` k-esimo existe en los
tres universos y las features nucleo (de nuestro universo `cuprum`) describen al mismo
afiliado en todos los cortes, pero el `afiliado_id` k-esimo de las tablas 02/03
corresponde a una simulacion PARALELA con perfil `sistema` (otros sorteos de edad, renta,
rubro, etc. desde el mismo punto del generador de numeros aleatorios) y no a la misma
persona con exactitud demografica. Se documenta y se acota su uso a las features y a la
validacion de negocio que el brief pide explicitamente sacar de esas tablas (no a las
features nucleo del segmento, que son 100 % del universo `cuprum` propio). Ver README
"Decisiones tomadas" y la propuesta a `common/` (que 02/03 acepten un `universo` inyectado
o un parametro `perfil`) en "Propuestas para `common/`".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso.synthetic import generar_tablas as _generar_tablas_churn
from afp_lab.common.synth_base import generar_universo
from afp_lab.propension_apv.synthetic import generar_tablas as _generar_tablas_apv

# --------------------------------------------------------------- aritmetica de periodos ----
def periodo_to_idx(p):
    p = np.asarray(p)
    return (p // 100 - 2000) * 12 + (p % 100 - 1)


def idx_to_periodo(i):
    i = np.asarray(i)
    return (2000 + i // 12) * 100 + (i % 12 + 1)


def add_months(p, k):
    return idx_to_periodo(periodo_to_idx(p) + k)


TASA_COTIZACION_OBLIGATORIA = 0.10   # 10 % de la RIM a la cuenta de capitalizacion obligatoria
TASA_MENSUAL_CAPITALIZACION = 0.0040  # rentabilidad real mensual supuesta para el saldo aproximado
# (mismo orden de magnitud que churn_traspaso.ParametrosChurn.tasa_mensual_saldo_inicial=0,0035 y
# propension_apv.RENT_MENSUAL_OBLIGATORIO=0,0045: ~5 % real anual, tipico de un fondo C/D de largo plazo)


def generar_datos(n_afiliados: int = 5000, periodo_inicio: int = 202201, periodo_fin: int = 202609,
                  seed: int = 42, perfil: str = "cuprum") -> dict[str, pd.DataFrame]:
    """Universo comun (perfil `cuprum` por defecto): fuente de verdad de las features nucleo
    del segmento (renta, densidad, estabilidad, edad, saldo aproximado). No agrega ninguna
    tabla propia nueva (el brief pide `synthetic.py` con "solo lo propio": lo propio de este
    proyecto es la RE-USO documentado, no un mecanismo nuevo de simulacion)."""
    return generar_universo(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                            periodo_fin=periodo_fin, seed=seed, perfil=perfil)


def generar_tablas_externas(n_afiliados: int = 5000, periodo_inicio: int = 202201,
                            periodo_fin: int = 202609, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Tablas de los proyectos 02 y 03, importadas sin modificarlas, con el MISMO
    `n_afiliados`/`periodo_inicio`/`seed` que `generar_datos` para que el `afiliado_id`
    k-esimo exista en los tres universos (ver limitacion documentada arriba). Namespacing
    con prefijo `churn_`/`apv_` para no confundir columnas homonimas (ambos proyectos
    generan una tabla `saldos`, con columnas distintas)."""
    churn = _generar_tablas_churn(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                  periodo_fin=periodo_fin, seed=seed)
    apv = _generar_tablas_apv(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                              periodo_fin=periodo_fin, seed=seed)
    return {
        "churn_saldos": churn["saldos"],
        "churn_cambios_fondo": churn["cambios_fondo"],
        "churn_contactos": churn["contactos"],
        "churn_reclamos": churn["reclamos"],
        "churn_uso_digital": churn["uso_digital"],
        "churn_traspasos": churn["traspasos"],
        "apv_saldos": apv["saldos"],
        "apv_aportes": apv["aportes_voluntarios"],
        "apv_campanas": apv["campanas"],
    }
