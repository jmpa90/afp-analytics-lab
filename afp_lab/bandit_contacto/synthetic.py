"""Tablas propias del proyecto 09, construidas sobre el universo y las tablas de 02.

`generar_tablas_bandit` reutiliza `afp_lab.churn_traspaso.synthetic.generar_tablas` TAL CUAL (no se
copia ni se modifica: mismo universo `cuprum`, mismas tablas operacionales) y agrega:

* `preferencias_contacto`: tabla OBSERVABLE (no latente) con los afiliados que pidieron no ser
  contactados. Es un hecho operacional conocido por la AFP (CRM), no una probabilidad de respuesta.
* `respuesta_latente`: tabla LATENTE (prefijo `lat_` en sus columnas de probabilidad y tipo) con la
  probabilidad VERDADERA de traspaso a 6 meses bajo cada brazo, para cada fila de contexto
  (afiliado x periodo_corte) que participa en la simulacion. Nunca se pasa a `features.py` ni a las
  politicas: solo la usa el simulador/oraculo para calcular recompensas y regret.

Mecanismo de heterogeneidad (coherente con el brief de uplift de la ronda 4, generador propio,
corregido tras revision del gerente -- ver "Decisiones tomadas" del README): cada fila
(afiliado, periodo_corte) tiene un tipo latente -- PERSUADIBLE, INDIFERENTE o SLEEPING_DOG
(`TIPOS_RESPUESTA`) -- que determina cuanto mueve cada brazo el log-odds de traspaso respecto del
riesgo base (`DELTA_LOGODDS`). El tipo YA NO es un sorteo independiente del contexto: es una funcion
logit (softmax de 3 clases) de variables as-of observables (`generar_tipo_respuesta`,
`TIPO_RESPUESTA_COEF`) mas un componente de ruido persistente por afiliado ("personalidad", fijo en
el tiempo) y uno transitorio por fila -- asi que un bandit contextual EN PRINCIPIO puede aprender a
distinguir personas, aunque el ruido deja un piso de heterogeneidad no explicable. El riesgo base
(bajo SIN_CONTACTO) es, por separado, una funcion logistica de variables as-of de negocio (brecha de
comision, reclamos, interes por la competencia, densidad de cotizacion, antiguedad) mas ruido
gaussiano no explicable (CONVENCIONES §2: evita AUC > 0,95).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso.synthetic import generar_tablas as _generar_tablas_churn

from . import (BRAZOS, DELTA_LOGODDS, P_OPT_OUT, TIPO_RESPUESTA_COEF, TIPO_RESPUESTA_INTERCEPTO,
              TIPO_RESPUESTA_RUIDO_AFILIADO_SD, TIPO_RESPUESTA_RUIDO_FILA_SD, TIPOS_RESPUESTA)

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

    El tipo de respuesta latente NO se genera aqui: depende del contexto as-of (`generar_tipo_respuesta`
    necesita `logins_3m`, `reclamos_6m`, etc. por corte), que todavia no existe en este punto. Se
    calcula en `model.preparar_entorno`, una vez construido el contexto de la ventana simulada.
    """
    tablas = _generar_tablas_churn(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                   periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    tablas = {k: tablas[k] for k in TABLAS_CHURN}
    tablas["preferencias_contacto"] = generar_preferencias_contacto(tablas["afiliados"], seed)
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


def _z(col: pd.Series) -> np.ndarray:
    """Z-score de una columna dentro del propio lote (media/desviacion del lote pasado)."""
    x = col.fillna(0.0).to_numpy(dtype=np.float64)
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma if sigma > 1e-6 else 1.0)


def generar_tipo_respuesta(contexto: pd.DataFrame, seed: int,
                          coef: dict = TIPO_RESPUESTA_COEF,
                          intercepto: dict = TIPO_RESPUESTA_INTERCEPTO,
                          sd_afiliado: float = TIPO_RESPUESTA_RUIDO_AFILIADO_SD,
                          sd_fila: float = TIPO_RESPUESTA_RUIDO_FILA_SD,
                          fuerza_senal: float = 1.0) -> pd.DataFrame:
    """Tipo de respuesta latente (softmax de 3 clases, INDIFERENTE de referencia) por FILA de
    contexto (afiliado_id, periodo_corte): tabla LATENTE, funcion suave de variables as-of + ruido.

    Cada logit (PERSUADIBLE, SLEEPING_DOG; ver `TIPO_RESPUESTA_COEF` para el detalle y el porque de
    cada signo) combina variables as-of ESTANDARIZADAS (z-score dentro del lote de `contexto`) mas
    dos componentes de ruido: uno persistente POR AFILIADO (mismo valor en todos sus cortes, una
    "personalidad" fija) y uno transitorio por fila. Con la calibracion de `TIPO_RESPUESTA_INTERCEPTO`
    la mezcla resultante queda cerca de `P_TIPO_RESPUESTA` (30/45/25%) en promedio, pero cada
    afiliado se desvia de ese promedio segun su contexto -- por eso un bandit contextual puede, en
    principio, aprender a distinguirlos (a diferencia de un sorteo puramente al azar por afiliado).

    `fuerza_senal` (SUPUESTO del barrido de sensibilidad, README "Sensibilidad a la senal
    contextual"): multiplica los pesos de `coef` (NO el intercepto ni el ruido). `0.0` anula el
    termino de contexto por completo -- el tipo queda determinado solo por intercepto + ruido, es
    decir, independiente del contexto observable ("ablacion aleatoria", equivalente en espiritu a la
    primera version de esta ronda); `1.0` es la calibracion vigente; `>1` amplifica la senal (mundo
    hipotetico con mas heterogeneidad explicable por contexto que la calibrada).
    """
    n = len(contexto)
    rng = np.random.default_rng(seed + 9002)
    af_ids = contexto["afiliado_id"].to_numpy()
    max_id = int(af_ids.max())

    campos = {v for d in coef.values() for v in d}
    z = {}
    for feat in campos:
        col = contexto[feat].fillna(0.0)
        if feat.startswith("brecha_"):
            col = col.clip(-5, 5)  # brechas de comision/rentabilidad: acotar outliers
        z[feat] = _z(col)

    ruido_af = {"PERSUADIBLE": rng.normal(0.0, sd_afiliado, max_id + 1),
               "SLEEPING_DOG": rng.normal(0.0, sd_afiliado, max_id + 1)}
    ruido_fila = {"PERSUADIBLE": rng.normal(0.0, sd_fila, n),
                 "SLEEPING_DOG": rng.normal(0.0, sd_fila, n)}

    logits = {"INDIFERENTE": np.zeros(n)}
    for tipo in ("PERSUADIBLE", "SLEEPING_DOG"):
        base = intercepto[tipo] + ruido_af[tipo][af_ids] + ruido_fila[tipo]
        for feat, w in coef[tipo].items():
            base = base + (fuerza_senal * w) * z[feat]
        logits[tipo] = base

    orden = ["PERSUADIBLE", "INDIFERENTE", "SLEEPING_DOG"]
    m = np.stack([logits[t] for t in orden], axis=1)
    m = m - m.max(axis=1, keepdims=True)
    p = np.exp(m)
    p = p / p.sum(axis=1, keepdims=True)
    u = rng.random(n)
    idx = (u[:, None] < np.cumsum(p, axis=1)).argmax(axis=1)
    tipo = np.array(orden)[idx]
    return pd.DataFrame({"afiliado_id": af_ids, "periodo_corte": contexto["periodo_corte"].to_numpy(),
                         "lat_tipo_respuesta": tipo})


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def calcular_respuesta_latente(contexto: pd.DataFrame, tipo_respuesta: pd.DataFrame,
                               seed: int) -> pd.DataFrame:
    """Probabilidad VERDADERA de traspaso a 6 meses bajo cada brazo (tabla latente).

    `contexto`: una fila por (afiliado_id, periodo_corte), con las columnas as-of de
    `churn_traspaso.features` (nunca se usan aqui las columnas `lat_`/`p_` del propio contexto,
    solo variables de negocio observables). `tipo_respuesta`: tabla de `generar_tipo_respuesta`
    (misma granularidad afiliado x periodo_corte).

    Devuelve un DataFrame con `afiliado_id`, `periodo_corte`, `lat_tipo_respuesta` y una columna
    `lat_p_churn_<BRAZO>` por brazo. El orden de filas es el de `contexto.reset_index(drop=True)`
    (la funcion es determinista dado ese orden y `seed`, no depende del pool mensual).
    """
    df = contexto.merge(tipo_respuesta, on=["afiliado_id", "periodo_corte"], how="left")
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
