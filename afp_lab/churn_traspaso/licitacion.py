"""Licitacion de afiliados ANTIGUOS de la Ley 21.735 (reforma de pensiones, marzo 2025).

La regla: cada 2 anos se sortea el **10 % de los afiliados no pensionados** del sistema y esos
afiliados pasan a la AFP de **menor comision**, salvo que ejerzan la opcion de quedarse dentro de
**30 dias**. Primera licitacion en **agosto de 2027**, con traspasos efectivos **desde junio de
2028** (`SORTEOS`).

Para Cuprum es el mayor riesgo de fuga de su historia: un afiliado premium sorteado "se va por
defecto" salvo que la AFP lo retenga, y la cartera de Cuprum tiene el saldo medio mas alto del
sistema.

Este modulo tiene tres partes:

1. `simular_licitacion_stock`: la tabla sintetica `licitacion_stock` (un registro por afiliado
   sorteado: mes de sorteo, plazo de opcion, mes efectivo, probabilidad de irse y decision).
2. `escenarios_licitacion`: afiliados y saldo en riesgo por sorteo, efecto de una campana de
   retencion con capacidad limitada y priorizada por `saldo x P(se va)`, y sensibilidad al supuesto
   de salida pasiva.
3. `combinar_con_score`: `p_ir = p_sorteo_se_va · sorteado + p_churn · (1 − sorteado)`.

**Por que el modelo de churn no se reentrena con el sorteo**: el sorteo es un evento nuevo, sin
historia (la primera licitacion todavia no ocurre), con una mecanica distinta del traspaso voluntario
(la salida es *por defecto* y lo que el afiliado decide es **quedarse**) y con una poblacion elegida
al azar, no auto-seleccionada. Entrenar con el no es posible: no hay etiquetas. Ademas, meterlo como
feature contaminaria el modelo voluntario, cuya poblacion y mecanismo no cambian. Lo que se necesita
para aprenderlo: (a) los resultados del primer sorteo (tasa real de ejercicio de la opcion por
segmento), (b) el registro de quien fue contactado y como, para estimar el efecto de la retencion, y
(c) idealmente una asignacion aleatoria de la capacidad de contacto en el primer sorteo, que permite
medir el uplift sin supuestos. Mientras no existan, la probabilidad de salida pasiva es un
**supuesto** y por eso todo se reporta como escenario con rango.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from afp_lab.rim_nowcasting.calendario import add_months

# (mes de sorteo, mes limite para ejercer la opcion, mes del traspaso efectivo)
# Ley 21.735: primera licitacion agosto 2027, opcion de 30 dias, traspasos desde junio 2028; luego
# cada 2 anos. El segundo sorteo (2029-08) queda fuera de la ventana simulada.
SORTEOS = ((202708, 202709, 202806), (202908, 202909, 203006))
FRACCION_SORTEADA = 0.10          # 10 % de los afiliados no pensionados
DECISIONES = ("SE_VA", "SE_QUEDA")


@dataclass(frozen=True)
class ParametrosLicitacion:
    """Supuestos de la decision del afiliado sorteado (ver README, seccion "Ronda 2")."""

    # Salida pasiva: probabilidad media de irse si la AFP no hace nada. El brief fija el rango de
    # trabajo en 55-75 %; el caso base es el punto medio.
    p_salida_pasiva_base: float = 0.65
    # Efecto (en logit) de cada driver de la propension a QUEDARSE
    b_antiguedad: float = -0.55        # por log1p(anos en la AFP): mas antiguedad -> se queda
    b_saldo: float = -0.18             # por log10(saldo / 10 MM): mas saldo -> mas involucrado
    b_renta: float = -0.22             # por log(RIM_ref / IMM)
    b_churn_latente: float = 0.80      # por desviacion estandar de la propension latente al churn
    b_uso_digital: float = -0.10       # por login mensual promedio (afiliado enganchado)
    b_reclamo_12m: float = 0.45        # un reclamo reciente empuja a irse
    # Campana de retencion: la AFP contacta a una fraccion de los sorteados dentro de los 30 dias
    efecto_contacto_retencion: float = -1.20   # en logit, sobre la probabilidad de irse
    capacidad_contacto: float = 0.30           # fraccion de los sorteados que se alcanza a contactar
    sigma_ruido: float = 0.35                  # heterogeneidad no explicada de la decision

    def __post_init__(self) -> None:
        if not 0.0 < self.p_salida_pasiva_base < 1.0:
            raise ValueError("p_salida_pasiva_base debe estar en (0, 1)")
        if not 0.0 <= self.capacidad_contacto <= 1.0:
            raise ValueError("capacidad_contacto debe estar en [0, 1]")


DESCRIPCION_PARAMETROS_LICITACION = {
    "p_salida_pasiva_base": "P(se va) media del sorteado sin contacto de retencion (brief: 55-75 %)",
    "b_antiguedad": "Efecto en el logit de irse por log1p(anos en la AFP)",
    "b_saldo": "Efecto por log10(saldo / 10 millones)",
    "b_renta": "Efecto por log(RIM de referencia / IMM)",
    "b_churn_latente": "Efecto por desviacion estandar de la propension latente al churn",
    "b_uso_digital": "Efecto por login mensual promedio (afiliado enganchado con la AFP)",
    "b_reclamo_12m": "Efecto de un reclamo en los ultimos 12 meses",
    "efecto_contacto_retencion": "Efecto en el logit del contacto de retencion dentro de los 30 dias",
    "capacidad_contacto": "Fraccion de los sorteados que la campana alcanza a contactar",
    "sigma_ruido": "Heterogeneidad no explicada de la decision",
}


def tabla_parametros_licitacion(par: ParametrosLicitacion | None = None) -> pd.DataFrame:
    par = par or ParametrosLicitacion()
    return pd.DataFrame([
        {"parametro": c.name, "valor": getattr(par, c.name),
         "descripcion": DESCRIPCION_PARAMETROS_LICITACION.get(c.name, "")}
        for c in fields(par)
    ])


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def sorteos_en_ventana(periodo_inicio: int, periodo_fin: int) -> list[tuple[int, int, int]]:
    """Sorteos cuyo mes de sorteo y mes efectivo caben en la ventana simulada."""
    return [s for s in SORTEOS if periodo_inicio <= s[0] and s[2] <= periodo_fin]


# --------------------------------------------------------------------- tabla sintetica
def simular_licitacion_stock(tablas: dict[str, pd.DataFrame], u_latente: pd.Series,
                             periodo_inicio: int, periodo_fin: int, seed: int,
                             par: ParametrosLicitacion | None = None,
                             fraccion_sorteada: float = FRACCION_SORTEADA) -> pd.DataFrame:
    """Tabla `licitacion_stock`: un registro por afiliado sorteado en cada licitacion de stock.

    La poblacion elegible es la de afiliados **vigentes en la AFP al mes del sorteo y no
    pensionados** (estado laboral distinto de RETIRADO y `tipo_afiliado` distinto de
    PENSIONADO_ACTIVO). El sorteo es aleatorio simple, como manda la ley.

    La decision se simula con una propension a IRSE en escala logit, anclada en
    `p_salida_pasiva_base` y modulada por antiguedad, saldo, renta, uso digital, reclamos recientes y
    la propension latente al churn del afiliado. `p_se_va_pasiva` es la probabilidad **sin** contacto
    de retencion: la campana es un escenario que se evalua despues (`escenarios_licitacion`), no algo
    que ya este en la tabla.
    """
    par = par or ParametrosLicitacion()
    rng = np.random.default_rng(seed + 909)
    saldos, panel = tablas["saldos"], tablas["panel_mensual"]
    afiliados = tablas["afiliados"].set_index("afiliado_id")
    macro = tablas["macro"].set_index("periodo")
    reclamos = tablas["reclamos"]
    dig = tablas["uso_digital"]

    filas = []
    for periodo_sorteo, periodo_opcion, periodo_efectivo in sorteos_en_ventana(periodo_inicio, periodo_fin):
        vig = saldos[saldos["periodo"] == periodo_sorteo]
        if not len(vig):
            continue
        corte_lab = add_months(periodo_sorteo, -2)
        pan = panel[panel["periodo"] == corte_lab].set_index("afiliado_id")
        estado = pan["estado"].reindex(vig["afiliado_id"]).to_numpy()
        tipo = afiliados["tipo_afiliado"].reindex(vig["afiliado_id"]).to_numpy()
        elegible = (estado != "RETIRADO") & (tipo != "PENSIONADO_ACTIVO")
        cand = vig[elegible].reset_index(drop=True)
        if not len(cand):
            continue
        n_sorteo = int(round(fraccion_sorteada * len(cand)))
        idx = rng.choice(len(cand), size=n_sorteo, replace=False)
        sub = cand.iloc[np.sort(idx)].reset_index(drop=True)
        af = sub["afiliado_id"].to_numpy()

        # --- drivers de la decision, todos conocidos al mes del sorteo
        antig = sub["antiguedad_afp_meses"].to_numpy(dtype=float)
        saldo = sub["saldo_obligatorio"].to_numpy(dtype=float)
        rim = (pan["rim_emp1"].reindex(af).fillna(0).to_numpy()
               + pan["rim_emp2"].reindex(af).fillna(0).to_numpy())
        imm = float(macro.loc[corte_lab, "imm"])
        renta_rel = np.clip(np.log(np.maximum(rim, imm * 0.3) / imm), -1.0, 2.5)
        ventana_dig = [add_months(periodo_sorteo, -k) for k in range(12)]
        d = dig[dig["periodo"].isin(ventana_dig)]
        logins = (d.assign(_l=d["logins_app"] + d["logins_web"]).groupby("afiliado_id")["_l"].mean()
                  .reindex(af).fillna(0.0).to_numpy())
        ventana_rec = [add_months(periodo_sorteo, -k) for k in range(12)]
        rec = (reclamos[reclamos["periodo"].isin(ventana_rec)].groupby("afiliado_id").size()
               .reindex(af).fillna(0).to_numpy() > 0).astype(float)
        u = u_latente.reindex(af).fillna(0.0).to_numpy()
        u_std = u / max(float(np.std(u_latente.to_numpy())), 1e-9)

        z = (_logit(par.p_salida_pasiva_base)
             + par.b_antiguedad * (np.log1p(antig / 12.0) - np.log1p(antig.mean() / 12.0))
             + par.b_saldo * (np.log10(np.maximum(saldo, 1.0) / 1e7))
             + par.b_renta * (renta_rel - renta_rel.mean())
             + par.b_uso_digital * (logins - logins.mean())
             + par.b_reclamo_12m * rec
             + par.b_churn_latente * u_std
             + rng.normal(0, par.sigma_ruido, len(af)))
        p_va = 1.0 / (1.0 + np.exp(-z))
        decision = np.where(rng.random(len(af)) < p_va, "SE_VA", "SE_QUEDA")
        destino_defecto = _afp_mas_barata(tablas["mercado"], periodo_sorteo)

        filas.append(pd.DataFrame({
            "afiliado_id": af,
            "periodo_sorteo": periodo_sorteo,
            "periodo_limite_opcion": periodo_opcion,
            "periodo_efectivo": periodo_efectivo,
            "saldo_obligatorio": saldo,
            "antiguedad_afp_meses": antig,
            "rim_referencia": rim,
            "p_se_va_pasiva": p_va,
            "decision": decision,
            "afp_destino": np.where(decision == "SE_VA", destino_defecto, ""),
        }))
    if not filas:
        return pd.DataFrame(columns=["afiliado_id", "periodo_sorteo", "periodo_limite_opcion",
                                     "periodo_efectivo", "saldo_obligatorio", "antiguedad_afp_meses",
                                     "rim_referencia", "p_se_va_pasiva", "decision", "afp_destino"])
    return pd.concat(filas, ignore_index=True)


def _afp_mas_barata(mercado: pd.DataFrame, periodo: int) -> str:
    """AFP de menor comision en el mes del sorteo: el destino por defecto de la ley."""
    m = mercado[mercado["periodo"] == periodo]
    if not len(m):
        m = mercado[mercado["periodo"] == mercado["periodo"].max()]
    m = m[m["es_propia"] == 0]
    return str(m.loc[m["comision"].idxmin(), "afp"])


# --------------------------------------------------------------------- escenarios
def _prioridad_retencion(df: pd.DataFrame) -> np.ndarray:
    """Orden de la campana: saldo x probabilidad de irse (valor esperado en riesgo)."""
    return -(df["saldo_obligatorio"].to_numpy(dtype=float) * df["p_se_va_pasiva"].to_numpy(dtype=float))


def escenario_licitacion(lic: pd.DataFrame, p_salida_pasiva: float, capacidad: float,
                         efecto_contacto: float, par: ParametrosLicitacion,
                         prioridad: str = "saldo_x_prob", seed: int = 11) -> dict:
    """Un escenario: reescala la propension al supuesto de salida pasiva y aplica la campana.

    `p_salida_pasiva` reemplaza el ancla del caso base desplazando el logit de todos los sorteados
    (mantiene el orden y la dispersion). `capacidad` es la fraccion contactada y `prioridad` decide a
    quien: `saldo_x_prob` (valor esperado en riesgo), `saldo`, `prob` o `aleatorio`.
    """
    if not len(lic):
        return {}
    rng = np.random.default_rng(seed)
    p0 = lic["p_se_va_pasiva"].to_numpy(dtype=float)
    z = np.log(np.clip(p0, 1e-9, 1 - 1e-9) / (1 - np.clip(p0, 1e-9, 1 - 1e-9)))
    z = z - _logit(par.p_salida_pasiva_base) + _logit(p_salida_pasiva)
    p_sin = 1.0 / (1.0 + np.exp(-z))

    saldo = lic["saldo_obligatorio"].to_numpy(dtype=float)
    n = len(lic)
    k = int(round(capacidad * n))
    if prioridad == "saldo_x_prob":
        orden = np.argsort(-(saldo * p_sin))
    elif prioridad == "saldo":
        orden = np.argsort(-saldo)
    elif prioridad == "prob":
        orden = np.argsort(-p_sin)
    elif prioridad == "aleatorio":
        orden = rng.permutation(n)
    else:
        raise ValueError(f"prioridad desconocida: {prioridad}")
    contactado = np.zeros(n, dtype=bool)
    contactado[orden[:k]] = True

    p_con = 1.0 / (1.0 + np.exp(-(z + efecto_contacto * contactado)))
    return {
        "p_salida_pasiva": p_salida_pasiva,
        "capacidad_contacto": capacidad,
        "prioridad": prioridad,
        "efecto_contacto_logit": efecto_contacto,
        "n_sorteados": n,
        "n_contactados": int(contactado.sum()),
        "saldo_sorteado_mm": float(saldo.sum()) / 1e6,
        "afiliados_esperados_se_van_sin_campana": float(p_sin.sum()),
        "afiliados_esperados_se_van_con_campana": float(p_con.sum()),
        "afiliados_retenidos": float((p_sin - p_con).sum()),
        "saldo_en_riesgo_sin_campana_mm": float((p_sin * saldo).sum()) / 1e6,
        "saldo_en_riesgo_con_campana_mm": float((p_con * saldo).sum()) / 1e6,
        "saldo_retenido_mm": float(((p_sin - p_con) * saldo).sum()) / 1e6,
        "saldo_retenido_por_contacto_mm": float(((p_sin - p_con) * saldo).sum()) / 1e6 / max(k, 1),
        "frac_saldo_sorteado_en_riesgo_sin_campana": float((p_sin * saldo).sum() / max(saldo.sum(), 1.0)),
        "frac_saldo_sorteado_en_riesgo_con_campana": float((p_con * saldo).sum() / max(saldo.sum(), 1.0)),
    }


def escenarios_licitacion(lic: pd.DataFrame, par: ParametrosLicitacion | None = None,
                          supuestos_salida: tuple[float, ...] = (0.55, 0.65, 0.75),
                          capacidades: tuple[float, ...] = (0.0, 0.30),
                          prioridades: tuple[str, ...] = ("saldo_x_prob", "aleatorio")) -> pd.DataFrame:
    """Malla de escenarios por sorteo: supuesto de salida pasiva x capacidad x criterio de prioridad."""
    par = par or ParametrosLicitacion()
    filas = []
    for periodo_sorteo, sub in lic.groupby("periodo_sorteo", observed=True):
        for p_sal in supuestos_salida:
            for cap in capacidades:
                for prio in (prioridades if cap > 0 else ("sin_campana",)):
                    e = escenario_licitacion(sub, p_sal, cap, par.efecto_contacto_retencion, par,
                                             prioridad="saldo_x_prob" if prio == "sin_campana" else prio)
                    if not e:
                        continue
                    e["periodo_sorteo"] = int(periodo_sorteo)
                    e["prioridad"] = prio
                    filas.append(e)
    cols = ["periodo_sorteo", "p_salida_pasiva", "capacidad_contacto", "prioridad"]
    d = pd.DataFrame(filas)
    return d[cols + [c for c in d.columns if c not in cols]]


def resumen_riesgo_licitacion(lic: pd.DataFrame, cartera: pd.DataFrame,
                              par: ParametrosLicitacion | None = None) -> pd.DataFrame:
    """Tamano del evento: sorteados y saldo sorteado contra el total de la cartera de ese mes."""
    par = par or ParametrosLicitacion()
    filas = []
    for periodo_sorteo, sub in lic.groupby("periodo_sorteo", observed=True):
        c = cartera[cartera["periodo"] == periodo_sorteo]
        saldo_cartera = float(c["saldo_obligatorio"].sum())
        p = sub["p_se_va_pasiva"].to_numpy(dtype=float)
        saldo = sub["saldo_obligatorio"].to_numpy(dtype=float)
        filas.append({
            "periodo_sorteo": int(periodo_sorteo),
            "periodo_efectivo": int(sub["periodo_efectivo"].iloc[0]),
            "afiliados_cartera": int(len(c)),
            "afiliados_sorteados": int(len(sub)),
            "frac_afiliados_sorteados": len(sub) / max(len(c), 1),
            "saldo_cartera_mm": saldo_cartera / 1e6,
            "saldo_sorteado_mm": float(saldo.sum()) / 1e6,
            "frac_saldo_sorteado": float(saldo.sum()) / max(saldo_cartera, 1.0),
            "saldo_medio_sorteado_mm": float(saldo.mean()) / 1e6,
            "saldo_medio_cartera_mm": saldo_cartera / max(len(c), 1) / 1e6,
            "p_se_va_media": float(p.mean()),
            "decision_se_va_observada": float((sub["decision"] == "SE_VA").mean()),
            "saldo_en_riesgo_esperado_mm": float((p * saldo).sum()) / 1e6,
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------- combinacion con el score
def combinar_con_score(df: pd.DataFrame, prob_churn: np.ndarray, lic: pd.DataFrame,
                       periodo_sorteo: int, p_sorteo_se_va: np.ndarray | float | None = None
                       ) -> pd.DataFrame:
    """`p_ir = p_sorteo_se_va · sorteado + p_churn · (1 − sorteado)`.

    El sorteo es un evento **determinista y conocido** (la AFP sabe quien fue sorteado el mismo mes),
    asi que no se predice: se combina. Para los sorteados la probabilidad de irse la manda la ley y
    el supuesto de salida pasiva; para el resto de la cartera sigue mandando el score de churn.
    Devuelve el dataframe con `sorteado`, `p_churn`, `p_sorteo` y `p_ir`.
    """
    sorteados = lic[lic["periodo_sorteo"] == periodo_sorteo].set_index("afiliado_id")
    out = df[["afiliado_id", "periodo_corte", "saldo_obligatorio"]].copy()
    out["p_churn"] = np.asarray(prob_churn, dtype=float)
    out["sorteado"] = out["afiliado_id"].isin(sorteados.index).astype(int)
    if p_sorteo_se_va is None:
        p_s = out["afiliado_id"].map(sorteados["p_se_va_pasiva"]).to_numpy(dtype=float)
    elif np.isscalar(p_sorteo_se_va):
        p_s = np.where(out["sorteado"] == 1, float(p_sorteo_se_va), np.nan)
    else:
        p_s = np.asarray(p_sorteo_se_va, dtype=float)
    out["p_sorteo"] = p_s
    out["p_ir"] = np.where(out["sorteado"] == 1, np.nan_to_num(p_s), out["p_churn"])
    return out


def ranking_combinado(comb: pd.DataFrame, fraccion: float = 0.10) -> pd.DataFrame:
    """Composicion de la lista de contacto cuando el sorteo entra al ranking.

    Responde la pregunta operativa: si el equipo sigue contactando al X % de mayor `p_ir`, cuanta de
    esa capacidad se la lleva el sorteo y cuanto saldo cubre.
    """
    n = len(comb)
    k = max(1, int(round(fraccion * n)))
    orden = np.argsort(-comb["p_ir"].to_numpy(dtype=float))
    sel = comb.iloc[orden[:k]]
    saldo = comb["saldo_obligatorio"].to_numpy(dtype=float)
    return pd.DataFrame([{
        "fraccion_contactada": fraccion,
        "n_contactados": k,
        "n_sorteados_en_la_lista": int(sel["sorteado"].sum()),
        "frac_de_la_lista_que_es_sorteo": float(sel["sorteado"].mean()),
        "frac_de_los_sorteados_cubierta": float(sel["sorteado"].sum() / max(comb["sorteado"].sum(), 1)),
        "saldo_cubierto_mm": float(sel["saldo_obligatorio"].sum()) / 1e6,
        "frac_saldo_cartera_cubierto": float(sel["saldo_obligatorio"].sum() / max(saldo.sum(), 1.0)),
        "p_ir_media_lista": float(sel["p_ir"].mean()),
    }])
