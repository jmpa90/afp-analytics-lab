"""Perfiles de segmento, validacion de utilidad de negocio, clasificador explicativo y
matriz de transicion (`BRIEF.md` §3-6).

Las tasas de negocio (traspaso, APV, cese) se leen de las tablas fuente (proyectos 02/03 y
el panel propio), NUNCA de features: son el resultado observado que valida si los segmentos
"separan algo real", no una prediccion. Sujeto a la limitacion de perfil documentada en
`synthetic.py` para las dos primeras (traspaso viene de `churn_traspasos`, aporte APV de
`apv_aportes`, ambas de universos "sistema" paralelos); el cese sale del panel `cuprum`
propio y no tiene esa limitacion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES_CLUSTERING
from .synthetic import add_months

FEATURES_CLASIFICADOR_CAT = ["tipo_afiliado", "sexo", "region", "nivel_educacional", "rubro", "fondo"]
PERFIL_NUMERICO = ["renta", "densidad_12", "racha_cotiza_m", "edad", "meses_hasta_pension",
                  "saldo_aprox", "saldo_renta_ratio", "apv_saldo_voluntario", "antiguedad_afp_meses",
                  "n_cambios_fondo_12", "contactos_12", "reclamos_12"]


# --------------------------------------------------------------------- targets de negocio
def _cese_3m(panel: pd.DataFrame, corte: int) -> pd.Series:
    """1 si el afiliado deja de cotizar (RIM = 0) en `corte+1..corte+3`, dado que cotizaba en
    `corte-2` (misma poblacion base que el proyecto 04); NaN si no cotizaba en `corte-2` (no
    aplica) o si el horizonte se sale del rango simulado."""
    corte_rim = add_months(corte, -2)
    horizonte = [add_months(corte, h) for h in (1, 2, 3)]
    p = panel[["afiliado_id", "periodo", "rim_emp1", "rim_emp2"]].copy()
    p["rim"] = p["rim_emp1"].fillna(0.0) + p["rim_emp2"].fillna(0.0)
    wide = p.pivot_table(index="afiliado_id", columns="periodo", values="rim", fill_value=0.0)
    if corte_rim not in wide.columns or any(h not in wide.columns for h in horizonte):
        return pd.Series(np.nan, index=wide.index, dtype=float)
    base = wide[corte_rim] > 0
    cese = (wide[horizonte] <= 0).all(axis=1)
    return pd.Series(np.where(base, cese.astype(float), np.nan), index=wide.index)


def _traspaso_6m(churn_traspasos: pd.DataFrame, churn_saldos: pd.DataFrame, corte: int) -> pd.Series:
    """1 si hay una solicitud de SALIDA (a la competencia) en `corte+1..corte+6`, entre los
    afiliados que, en la simulacion PARALELA de `churn_traspaso` (ver limitacion de perfil),
    ya estaban en la AFP propia a `corte`; NaN para el resto (no aplica: nunca estuvieron)."""
    horizonte = [add_months(corte, h) for h in range(1, 7)]
    vigentes = pd.Index(churn_saldos.loc[churn_saldos["periodo"] <= corte, "afiliado_id"].unique(),
                        name="afiliado_id")
    if len(vigentes) == 0:
        return pd.Series(dtype=float)
    salidas = set(churn_traspasos.loc[
        (churn_traspasos["tipo"] == "SALIDA") & churn_traspasos["periodo_solicitud"].isin(horizonte),
        "afiliado_id"].unique())
    return pd.Series(vigentes.isin(salidas).astype(float), index=vigentes)


def _apv_6m(apv_aportes: pd.DataFrame, af_ids: np.ndarray, corte: int) -> pd.Series:
    """1 si hay algun aporte voluntario en `corte+1..corte+6` (universo paralelo de
    `propension_apv`, misma limitacion de perfil); se evalua sobre TODOS los afiliados del
    segmento (no hay poblacion "no aplica" para este target)."""
    horizonte = [add_months(corte, h) for h in range(1, 7)]
    con_aporte = set(apv_aportes.loc[apv_aportes["periodo"].isin(horizonte), "afiliado_id"].unique())
    idx = pd.Index(af_ids, name="afiliado_id")
    return pd.Series(idx.isin(con_aporte).astype(float), index=idx)


def construir_targets_negocio(tablas: dict[str, pd.DataFrame], tablas_externas: dict[str, pd.DataFrame],
                              af_ids: np.ndarray, corte: int) -> pd.DataFrame:
    """Tabla `afiliado_id x {cese_3m, traspaso_6m, apv_6m}` para validar la separacion de
    negocio de los segmentos. `corte` debe dejar el horizonte completo dentro del rango
    simulado (igual chequeo que en `run.py`, ver `cortes_evaluables_negocio`)."""
    cese = _cese_3m(tablas["panel_mensual"], corte)
    traspaso = _traspaso_6m(tablas_externas["churn_traspasos"], tablas_externas["churn_saldos"], corte)
    apv = _apv_6m(tablas_externas["apv_aportes"], af_ids, corte)
    out = pd.DataFrame(index=pd.Index(af_ids, name="afiliado_id"))
    out["cese_3m"] = cese.reindex(out.index)
    out["traspaso_6m"] = traspaso.reindex(out.index)
    out["apv_6m"] = apv.reindex(out.index)
    return out.reset_index()


def tasas_por_segmento(df_labels: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Tasa de cese/traspaso/APV por segmento, con el tamano de la base de cada una (no
    todas las filas de un segmento "aplican" a las 3 tasas, ver docstrings de arriba)."""
    m = df_labels[["afiliado_id", "segmento"]].merge(targets, on="afiliado_id", how="left")
    filas = []
    for seg, g in m.groupby("segmento"):
        filas.append({
            "segmento": seg, "n": len(g),
            "tasa_cese_3m": g["cese_3m"].mean(skipna=True), "n_base_cese_3m": int(g["cese_3m"].notna().sum()),
            "tasa_traspaso_6m": g["traspaso_6m"].mean(skipna=True),
            "n_base_traspaso_6m": int(g["traspaso_6m"].notna().sum()),
            "tasa_apv_6m": g["apv_6m"].mean(skipna=True), "n_base_apv_6m": int(g["apv_6m"].notna().sum()),
        })
    return pd.DataFrame(filas).sort_values("segmento").reset_index(drop=True)


def separacion_negocio(tabla_tasas: pd.DataFrame) -> dict:
    """Brief §4: los segmentos deben separar al menos 2 de las 3 tasas con razon (max/min)
    >= 2. Una tasa minima de 0 (algun segmento con 0 casos) cuenta como separacion perfecta
    (razon infinita) siempre que el maximo sea > 0."""
    razones, cumplen = {}, {}
    for col in ("tasa_cese_3m", "tasa_traspaso_6m", "tasa_apv_6m"):
        vals = tabla_tasas[col].dropna()
        vals = vals[vals.notna()]
        if len(vals) < 2 or vals.max() <= 0:
            razon = float("nan")
        elif vals.min() <= 0:
            razon = float("inf")
        else:
            razon = float(vals.max() / vals.min())
        razones[col] = razon
        cumplen[col] = bool(np.isfinite(razon) and razon >= 2.0) or razon == float("inf")
    n_ok = sum(cumplen.values())
    return {"razones": razones, "cumplen": cumplen, "n_tasas_con_razon_2": n_ok, "cumple": n_ok >= 2}


# --------------------------------------------------------------------- perfil de segmentos
def perfil_segmentos(df_labels: pd.DataFrame) -> pd.DataFrame:
    """Medianas/modas por segmento (para nombrarlos y describirlos en el README)."""
    n_total = len(df_labels)
    filas = []
    for seg, g in df_labels.groupby("segmento"):
        fila = {"segmento": seg, "n": len(g), "pct": 100.0 * len(g) / n_total}
        for c in PERFIL_NUMERICO:
            fila[f"mediana_{c}"] = float(g[c].median())
        for c, nombre in (("tipo_afiliado", "moda_tipo_afiliado"), ("fondo", "moda_fondo"),
                         ("rubro", "moda_rubro"), ("nivel_educacional", "moda_nivel_educacional")):
            moda = g[c].mode()
            fila[nombre] = moda.iat[0] if len(moda) else None
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("segmento").reset_index(drop=True)


# --------------------------------------------------------------------- clasificador explicativo
def entrenar_clasificador(df_labels: pd.DataFrame, seed: int = 42, test_size: float = 0.25) -> dict:
    """LightGBM multiclase que reproduce el segmento desde las features (brief §5): explica
    la particion en reglas simples y permite asignar afiliados nuevos sin re-clusterizar.
    Split aleatorio estratificado (no temporal): el "target" es la propia etiqueta de
    clustering de ESTE corte, no un evento futuro, asi que no hay riesgo de leakage temporal."""
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    cols_num = list(FEATURES_CLUSTERING)
    cols_cat = list(FEATURES_CLASIFICADOR_CAT)
    X = df_labels[cols_num + cols_cat].copy()
    for c in cols_cat:
        X[c] = X[c].astype("category")
    y = df_labels["segmento"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                             min_child_samples=30, n_jobs=2, random_state=seed, verbose=-1)
    clf.fit(X_train, y_train, categorical_feature=cols_cat)
    pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, pred))

    gain = clf.booster_.feature_importance(importance_type="gain")
    importancias = pd.DataFrame({"feature": cols_num + cols_cat, "ganancia": gain})
    importancias["ganancia_pct"] = 100 * importancias["ganancia"] / max(importancias["ganancia"].sum(), 1e-9)
    importancias = importancias.sort_values("ganancia", ascending=False).reset_index(drop=True)

    return {"accuracy": acc, "importancias": importancias, "modelo": clf,
           "n_train": len(X_train), "n_test": len(X_test)}


# --------------------------------------------------------------------- matriz de transicion
def matriz_transicion(af_ids_a: np.ndarray, labels_a: np.ndarray,
                      af_ids_b: np.ndarray, labels_b: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Matriz de transicion de segmentos entre dos cortes (filas = origen, normalizadas a
    1), restringida a los afiliados presentes en ambos (misma persona, mismo universo
    `cuprum`: sin la limitacion de perfil de las tablas 02/03)."""
    da = pd.Series(labels_a, index=af_ids_a)
    db = pd.Series(labels_b, index=af_ids_b)
    comunes = da.index.intersection(db.index)
    tab = pd.crosstab(da.loc[comunes], db.loc[comunes])
    tab_norm = tab.div(tab.sum(axis=1), axis=0)
    return tab_norm, len(comunes)
