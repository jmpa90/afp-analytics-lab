"""Ensamblado del dataset causal: features as-of de 02 + historial de campana + target alterado.

`construir_dataset_causal` es la funcion principal. Devuelve DOS tablas:

* `df` -- una fila por (afiliado_id, periodo_corte), con las features as-of de
  `churn_traspaso.construir_features_multi` (reutilizadas sin modificar), el historial de la
  campana propia (`campana_contactos_6m`, `campana_contactos_12m`, `campana_meses_desde_ultimo`,
  todas estrictamente anteriores al corte), el tratamiento de ESTE corte (`T`), el flag `holdout`
  (diseno experimental, no un feature) y el target `y` YA alterado por el efecto causal. Es lo que
  ve el modelo (`columnas_features` excluye `T`, `holdout`, `y`, etc. igual que en 02).
* `latente` -- la verdad del simulador que ningun analista real observaria: `segmento`, `lat_cate`
  (CATE verdadero), `y0_control_potencial` (target de 02 sin campana) y las variables latentes por
  afiliado. Solo para evaluacion (`evaluate.py`), nunca se une al dataset de entrenamiento.

Historial de campana: se calcula sobre TODOS los cortes evaluables (no solo el split), igual que
`construir_features_multi`, para que la ventana de 6/12 meses no salte por encima de la purga del
split -- exactamente el patron de `churn_traspaso.run.preparar` (arma el dataset completo y despues
filtra train/valid/test por `periodo_corte`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso import features as CT_F
from afp_lab.churn_traspaso.evaluate import agregar_segmentos

from . import HORIZONTE_MESES
from .synthetic import (ParametrosCampana, aplicar_efecto_causal, asignar_tratamiento,
                        construir_efecto_causal, riesgo_latente_heuristico,
                        variables_latentes_afiliado)

VENTANAS_HISTORIAL_CAMPANA = (6, 12)
# columnas que ademas del target/ids hay que excluir de `columnas_features`: el tratamiento y el
# flag de holdout son variables de diseno/decision, no covariables; los tramos de negocio (si algun
# dia se agregan al dataset principal) tampoco son features numericas utilizables tal cual.
COLUMNAS_NO_FEATURE_EXTRA = ("T", "holdout", "tramo_saldo", "tramo_edad", "tramo_renta")


# --------------------------------------------------------------------- historial de campana (as-of)
def _historial_campana(trat: pd.DataFrame, ventanas: tuple[int, ...] = VENTANAS_HISTORIAL_CAMPANA) -> pd.DataFrame:
    """Conteo movil de contactos de campana y meses desde el ultimo, EXCLUYENDO el corte actual.

    `trat` trae (afiliado_id, periodo_corte, T) para todos los cortes evaluables: cada afiliado
    aparece en un bloque CONTIGUO de cortes mensuales mientras esta vigente en la cartera (igual que
    en `construir_features_multi`), asi que una ventana movil por posicion de fila equivale a una
    ventana movil calendario.
    """
    d = trat.sort_values(["afiliado_id", "periodo_corte"]).reset_index(drop=True)
    afiliado = d["afiliado_id"].to_numpy()
    T = d["T"].to_numpy(dtype=float)
    n = len(d)
    nuevo = np.empty(n, dtype=bool)
    nuevo[0] = True
    if n > 1:
        nuevo[1:] = afiliado[1:] != afiliado[:-1]

    # T del corte ANTERIOR (0 en el primer corte de cada afiliado: no hay historial previo)
    T_prev = np.where(nuevo, 0.0, np.concatenate(([0.0], T[:-1])))
    s = pd.Series(T_prev, index=d.index)
    for v in ventanas:
        roll = s.groupby(afiliado).rolling(v, min_periods=1).sum()
        d[f"campana_contactos_{v}m"] = roll.reset_index(level=0, drop=True).to_numpy()

    # meses desde el ultimo contacto, INCLUYENDO el corte actual (0 si T=1 este corte)...
    tope = 99.0
    meses_incl = np.empty(n, dtype=np.float64)
    cur = tope
    for i in range(n):
        if nuevo[i]:
            cur = tope
        cur = 0.0 if T[i] == 1.0 else min(cur + 1.0, tope)
        meses_incl[i] = cur
    # ...y se corre un corte hacia adelante para que la fila actual solo vea el pasado
    meses_prev = np.where(nuevo, tope, np.concatenate(([tope], meses_incl[:-1])))
    d["campana_meses_desde_ultimo"] = meses_prev
    return d[["afiliado_id", "periodo_corte", "T",
             *[f"campana_contactos_{v}m" for v in ventanas], "campana_meses_desde_ultimo"]]


# --------------------------------------------------------------------- dataset causal
def construir_dataset_causal(tablas: dict[str, pd.DataFrame], cortes: list[int], seed: int,
                             par: ParametrosCampana | None = None,
                             horizonte: int = HORIZONTE_MESES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features as-of de 02 + historial de campana + T + target `y` alterado por el efecto causal.

    Devuelve `(df, latente)`: `df` es lo que ve el modelo; `latente` es la verdad del simulador
    (`segmento`, `lat_cate`, `y0_control_potencial`) para evaluar contra ella.
    """
    par = par or ParametrosCampana()
    base = CT_F.construir_features_multi(tablas, cortes)                 # reutilizado, sin modificar
    base_seg = agregar_segmentos(base)                                    # copia aparte: tramo_saldo, etc.

    riesgo = riesgo_latente_heuristico(base_seg, par)
    latentes_af = variables_latentes_afiliado(tablas["afiliados"], seed, par)
    trat = asignar_tratamiento(base_seg, riesgo, latentes_af, seed, par)
    causal = construir_efecto_causal(base_seg, riesgo, latentes_af, par)

    hist = _historial_campana(trat[["afiliado_id", "periodo_corte", "T"]])
    df = base.merge(hist.drop(columns=["T"]), on=["afiliado_id", "periodo_corte"], how="left")
    df = df.merge(trat[["afiliado_id", "periodo_corte", "T", "holdout"]],
                  on=["afiliado_id", "periodo_corte"], how="left")
    df["T"] = df["T"].fillna(0).astype(np.int8)
    df["holdout"] = df["holdout"].fillna(False).astype(bool)

    df = CT_F.construir_target(tablas, df, horizonte=horizonte)          # agrega "y" = y0 (identico a 02)
    y0 = df["y"].to_numpy().copy()
    df["y"] = aplicar_efecto_causal(y0, df["T"].to_numpy(), causal["segmento"].to_numpy(), seed, par)

    latente = causal.merge(trat[["afiliado_id", "periodo_corte", "holdout", "e_hist"]],
                           on=["afiliado_id", "periodo_corte"])
    latente["y0_control_potencial"] = y0
    latente = latente.merge(latentes_af, on="afiliado_id", how="left")
    return df.reset_index(drop=True), latente.reset_index(drop=True)


def columnas_features(df: pd.DataFrame) -> list[str]:
    """Columnas utilizables por los modelos: features as-of de 02 + historial de campana.

    Excluye ademas de `CT_F.columnas_features` (que ya saca ids, target y prefijos `lat_`/`p_`) el
    tratamiento (`T`), el flag de holdout y los tramos de negocio (si llegaran a colarse).
    """
    return [c for c in CT_F.columnas_features(df) if c not in COLUMNAS_NO_FEATURE_EXTRA]
