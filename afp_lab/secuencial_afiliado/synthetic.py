"""Tablas del proyecto 08: combina, SIN modificarlas, las tablas propias de 02
(`churn_traspaso`) y 04 (`riesgo_cese_cotizacion`).

`afp_lab.common.synth_base.generar_universo` es determinista en
`(n_afiliados, periodo_inicio, periodo_fin, seed, perfil, afiliados_por_empleador)`: al llamar
a `churn_traspaso.synthetic.generar_tablas` y `riesgo_cese_cotizacion.synthetic.generar_datos`
con los MISMOS argumentos (ninguno de los dos toca `lote` ni `afiliados_por_empleador`), ambos
reconstruyen exactamente el mismo universo comun (`afiliados`, `empleadores`, `panel_mensual`,
`macro`). Se verifica en
`tests/test_secuencial_afiliado.py::test_universo_identico_02_04`. Este modulo se queda con el
universo + tablas de 02 (que trae ademas mercado/saldos/traspasos/contactos/reclamos/
cambios_fondo/uso_digital/licitacion_stock) y le agrega solo las tablas propias de 04 que no
existen en 02 (cotizaciones_conocidas, afc_eventos, licencias, prestaciones_cesantia,
planilla), quedando un unico dict apto para `churn_traspaso.features` Y
`riesgo_cese_cotizacion.features` a la vez.

No se genera nada de 03 (`propension_apv`) en esta entrega: la cabeza de APV queda pendiente
(ver README, "Pendientes").
"""
from __future__ import annotations

import pandas as pd

from afp_lab.churn_traspaso.synthetic import generar_tablas as _generar_tablas_churn
from afp_lab.riesgo_cese_cotizacion.synthetic import generar_datos as _generar_datos_cese

# Mismo rango que usan por defecto 02 y 04 (57 meses: da los 24 de ventana de secuencia mas
# margen de sobra para burn-in, desfase y horizonte de 6 meses en ambas puntas).
PERIODO_INICIO = 202201
PERIODO_FIN = 202609

# Tablas propias de 04 que no trae 02 (el resto de 04 -afiliados/empleadores/macro/panel_
# mensual- es identico al universo de 02 y no hace falta duplicarlo).
_TABLAS_CESE_EXTRA = ("cotizaciones_conocidas", "afc_eventos", "licencias",
                     "prestaciones_cesantia", "planilla")


def generar_tablas(n_afiliados: int = 10_000, periodo_inicio: int = PERIODO_INICIO,
                   periodo_fin: int = PERIODO_FIN, seed: int = 42,
                   perfil: str = "cuprum") -> dict[str, pd.DataFrame]:
    """Universo comun + tablas de 02 y 04, en un solo dict (ninguna tabla se modifica).

    `perfil="cuprum"` (BRIEF.md): cartera de Cuprum, ver `afp_lab.common.synth_base.PERFILES` y
    `docs/contexto_afp_cuprum.md`.
    """
    t_churn = _generar_tablas_churn(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                    periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    t_cese = _generar_datos_cese(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                 periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    tablas = dict(t_churn)
    for k in _TABLAS_CESE_EXTRA:
        if k in t_cese and t_cese[k] is not None:
            tablas[k] = t_cese[k]
    return tablas
