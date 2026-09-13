"""Tablas propias del proyecto 09, construidas sobre el universo y las tablas de 02.

`generar_tablas_bandit` reutiliza `afp_lab.churn_traspaso.synthetic.generar_tablas` TAL CUAL (no se
copia ni se modifica: mismo universo `cuprum`, mismas tablas operacionales) y agrega:

* `preferencias_contacto`: tabla OBSERVABLE (no latente) con los afiliados que pidieron no ser
  contactados. Es un hecho operacional conocido por la AFP (CRM), no una probabilidad de respuesta.
* `respuesta_latente`: tabla LATENTE (prefijo `lat_` en sus columnas de probabilidad y tipo) con la
  probabilidad VERDADERA de traspaso a 6 meses bajo cada brazo, para cada fila de contexto
  (afiliado x periodo_corte) que participa en la simulacion. Nunca se pasa a `features.py` ni a las
  politicas: solo la usa el simulador/oraculo para calcular recompensas y regret.

Mecanismo de heterogeneidad (coherente con el brief de uplift de la ronda 4, generador propio):
cada afiliado tiene un tipo latente fijo en el tiempo -- PERSUADIBLE, INDIFERENTE o SLEEPING_DOG
(`TIPOS_RESPUESTA`, proporciones `P_TIPO_RESPUESTA`) -- que determina cuanto mueve cada brazo el
log-odds de traspaso respecto del riesgo base (`DELTA_LOGODDS`). El riesgo base es una funcion
logistica de variables as-of de negocio (brecha de comision, reclamos, interes por la competencia,
densidad de cotizacion, antiguedad) mas ruido gaussiano no explicable (CONVENCIONES §2: evita
AUC > 0,95).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso.synthetic import generar_tablas as _generar_tablas_churn

from . import BRAZOS, DELTA_LOGODDS, P_OPT_OUT, P_TIPO_RESPUESTA, TIPOS_RESPUESTA

TABLAS_CHURN = ("afiliados", "empleadores", "panel_mensual", "macro", "mercado", "saldos",
                "cambios_fondo", "traspasos", "contactos", "reclamos", "uso_digital",
                "licitacion_stock")


def generar_tablas_bandit(n_afiliados: int = 10_000, periodo_inicio: int = 202201,
                          periodo_fin: int = 202612, seed: int = 42,
                          perfil: str = "cuprum") -> dict[str, pd.DataFrame]:
    """Universo + tablas de 02 (`churn_traspaso`, sin modificar) mas las tablas propias de 09.

    Ventana por defecto 2022-01..2026-12 (60 meses): alcanza para el burn-in de features (14 m,
    igual que 02), >= 24 meses de simulacion y los 6 meses de resolucion de la ultima recompensa,
    con margen. Menos meses que el default de 02 (96) para que la generacion sea mas rapida a la
    escala de referencia de esta ronda (10.000 afiliados).
    """
    tablas = _generar_tablas_churn(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                   periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    tablas = {k: tablas[k] for k in TABLAS_CHURN}
    tablas["preferencias_contacto"] = generar_preferencias_contacto(tablas["afiliados"], seed)
    tablas["respuesta_latente_tipo"] = generar_tipo_respuesta(tablas["afiliados"], seed)
    return tablas


def generar_preferencias_contacto(afiliados: pd.DataFrame, seed: int,
                                  p_opt_out: float = P_OPT_OUT) -> pd.DataFrame:
    """Afiliados que pidieron no ser contactados (tabla operacional observable, CRM de la AFP).

    Simplificacion documentada: la exclusion es fija durante toda la ventana simulada (no depende
    del periodo); en la realidad estas solicitudes llegan de a poco en el tiempo, pero como aqui
    solo se usa para filtrar el pool mensual, tratarla como un flag persistente por afiliado alcanza
    y mantiene el generador simple.
    """
    rng = np.random.default_rng(seed + 9001)
    ids = afiliados["afiliado_id"].to_numpy()
    opt_out = rng.random(len(ids)) < p_opt_out
    return pd.DataFrame({"afiliado_id": ids, "solicito_no_contacto": opt_out})


def generar_tipo_respuesta(afiliados: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Tipo de respuesta latente al contacto por afiliado (fijo en el tiempo). Tabla LATENTE."""
    rng = np.random.default_rng(seed + 9002)
    ids = afiliados["afiliado_id"].to_numpy()
    tipo = rng.choice(TIPOS_RESPUESTA, size=len(ids), p=P_TIPO_RESPUESTA)
    return pd.DataFrame({"afiliado_id": ids, "lat_tipo_respuesta": tipo})


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def calcular_respuesta_latente(contexto: pd.DataFrame, tipo_respuesta: pd.DataFrame,
                               seed: int) -> pd.DataFrame:
    """Probabilidad VERDADERA de traspaso a 6 meses bajo cada brazo (tabla latente).

    `contexto`: una fila por (afiliado_id, periodo_corte), con las columnas as-of de
    `churn_traspaso.features` (nunca se usan aqui las columnas `lat_`/`p_` del propio contexto,
    solo variables de negocio observables). `tipo_respuesta`: tabla de `generar_tipo_respuesta`.

    Devuelve un DataFrame con `afiliado_id`, `periodo_corte`, `lat_tipo_respuesta` y una columna
    `lat_p_churn_<BRAZO>` por brazo. El orden de filas es el de `contexto.reset_index(drop=True)`
    (la funcion es determinista dado ese orden y `seed`, no depende del pool mensual).
    """
    df = contexto.merge(tipo_respuesta, on="afiliado_id", how="left")
    n = len(df)
    rng = np.random.default_rng(seed + 9003)

    brecha = df["brecha_comision_rel"].fillna(0.0).clip(-1, 3).to_numpy()
    reclamos = np.log1p(df["reclamos_6m"].fillna(0.0).to_numpy())
    interes_comp = np.log1p(df["contactos_competencia_interesado_6m"].fillna(0.0).to_numpy())
    dens = df["densidad_12"].fillna(0.0).to_numpy()
    tenure = np.log1p(df["antiguedad_afp_meses"].fillna(0.0).clip(lower=0).to_numpy())
    ruido = rng.normal(0.0, 0.55, n)  # ruido no explicable (CONVENCIONES §2)

    logit0 = (-2.6 + 1.10 * brecha + 0.55 * reclamos + 0.65 * interes_comp
              - 0.35 * dens - 0.10 * tenure + ruido)
    p0 = np.clip(_expit(logit0), 0.01, 0.6)

    out = {"afiliado_id": df["afiliado_id"].to_numpy(), "periodo_corte": df["periodo_corte"].to_numpy(),
          "lat_tipo_respuesta": df["lat_tipo_respuesta"].to_numpy(),
          "lat_p_churn_SIN_CONTACTO": p0}
    tipo = df["lat_tipo_respuesta"].to_numpy()
    for brazo in BRAZOS:
        if brazo == "SIN_CONTACTO":
            continue
        delta = np.array([DELTA_LOGODDS[t][brazo] for t in tipo])
        out[f"lat_p_churn_{brazo}"] = np.clip(_expit(logit0 + delta), 0.01, 0.6)
    return pd.DataFrame(out)
