"""Features "as-of" del proyecto 11 (CONVENCIONES §3).

Reutiliza SIN MODIFICAR:

* `churn_traspaso.features.construir_features_multi` -- mercado (comision, rentabilidad), saldo y
  fondo, antiguedad, contactos y reclamos por canal/ventana, uso digital, edad, region, tipo de
  afiliado, rubro y tamano del empleador. Mismo desfase de panel (`corte-2`).
* `propension_apv.features.construir_features` (`solo_elegibles=False`, para no perder poblacion) --
  un subconjunto de columnas propias de ahorro voluntario (APVC del empleador, historial de aporte,
  tramo del IGC), con su mismo desfase.

Y agrega, ya de este proyecto:

* DNP y rezago de recepcion del empleador (`common/planilla`, via `agregado_empleador`), leidos con
  el MISMO desfase que el panel laboral (`corte-2`): simplificacion documentada en el README/README
  "Decisiones y limitaciones" -- la fecha real en que la AFP conoce el estado de pago
  (`periodo_recepcion`) puede ser algo posterior a `corte-2` en los casos de pago mas tardio, que son
  justamente los que esta feature quiere capturar; una version mas estricta filtraria por
  `periodo_recepcion <= corte`.
* Licencias medicas largas (`dias_licencia`, panel laboral, `corte-2`).
* Historial de encuestas: SOLO hasta `corte-1` (nunca la respuesta del propio corte ni posteriores).

`construir_dataset` agrega el target de la encuesta de ESTE corte (invitado/respondio/respuesta).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso import features as CT_F
from afp_lab.propension_apv import features as APV_F
from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx, rango_periodos

from . import HORIZONTE_MESES, MESES_DESFASE_RIM
from .synthetic import DETRACTOR_HASTA, PROMOTOR_DESDE

VENTANAS = (1, 3, 6, 12)

# subconjunto de columnas de propension_apv que se agregan (evita colisionar con las de churn_traspaso
# y con los ids/target; se renombran con el prefijo `apv_`)
COLUMNAS_APV_EXTRA = (
    "ofrece_apvc", "pct_match_apvc", "meses_con_apvc", "tuvo_aporte_mensual_24m", "n_aportes_12m",
    "meses_desde_ult_aporte", "nunca_aporto_36m", "tramo_igc_12m", "saldo_voluntario_imm",
    "contactos_6m", "tasa_apertura_6m",
)

COLUMNAS_CATEGORICAS = list(CT_F.COLUMNAS_CATEGORICAS) + ["ultima_categoria_nps"]
COLUMNAS_ID = ["afiliado_id", "periodo_corte"]
COLUMNAS_TARGET_ENCUESTA = ["invitado", "respondio", "respuesta", "categoria", "es_detractor", "es_promotor"]
# Columnas derivadas del propio pipeline de modelado (NO features, aunque `run.py` las agrega al
# DataFrame para usarlas como `sample_weight`/diagnostico): `p_hat_responde` ya queda excluida por el
# prefijo `p_`, pero `peso_ipw` = clip(1/p_hat_responde) NO empieza con `lat_`/`p_` y se filtraba solo
# por casualidad de nombre -- si se deja como feature, el clasificador de NPS puede "leer" el propio
# peso de muestra (que es funcion del modelo de respuesta) en vez de aprender de senales observables,
# lo que degrada el AUC de detractor (encontrado en la correccion de ronda 5: 0.7497 con `peso_ipw`
# como feature vs 0.7530 sin ella, mismo dataset/seed/particion). Debe usarse SOLO como `sample_weight`.
COLUMNAS_DERIVADAS_PIPELINE = ["p_hat_responde", "peso_ipw"]
COLUMNAS_NO_FEATURE = COLUMNAS_ID + COLUMNAS_TARGET_ENCUESTA + COLUMNAS_DERIVADAS_PIPELINE


# --------------------------------------------------------------------- adaptadores de reuso
def _tablas_para_churn(tablas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    claves = ("afiliados", "empleadores", "panel_mensual", "macro", "mercado", "saldos",
             "cambios_fondo", "traspasos", "contactos", "reclamos", "uso_digital")
    return {k: tablas[k] for k in claves}


def _tablas_para_apv(tablas: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "afiliados": tablas["afiliados"], "empleadores": tablas["empleadores"],
        "panel_mensual": tablas["panel_mensual"], "macro": tablas["macro"],
        "apvc_empleador": tablas["apvc_empleador"], "campanas": tablas["campanas_apv"],
        "aportes_voluntarios": tablas["aportes_voluntarios"], "saldos": tablas["saldos_apv"],
        "tributario": tablas["tributario"],
    }


# --------------------------------------------------------------------- helpers matriciales (propios)
def _matriz(df: pd.DataFrame, col: str, afil_pos: dict, per_pos: dict,
           dtype=np.float32, relleno=0.0, agregar: bool = False) -> np.ndarray:
    m = np.full((len(afil_pos), len(per_pos)), relleno, dtype=dtype)
    if df is None or not len(df):
        return m
    d = df[df["periodo"].isin(per_pos) & df["afiliado_id"].isin(afil_pos)]
    if not len(d):
        return m
    i = d["afiliado_id"].map(afil_pos).to_numpy()
    j = d["periodo"].map(per_pos).to_numpy()
    if agregar:
        np.add.at(m, (i, j), d[col].to_numpy())
    else:
        m[i, j] = d[col].to_numpy()
    return m


def _suma_movil(mat: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(mat, axis=1, dtype=np.float64)
    out = c.copy()
    if k < mat.shape[1]:
        out[:, k:] = c[:, k:] - c[:, :-k]
    return out.astype(np.float32)


def _meses_desde(flag: np.ndarray, tope: int = 99) -> np.ndarray:
    n, t = flag.shape
    out = np.full((n, t), tope, dtype=np.float32)
    actual = np.full(n, tope, dtype=np.float32)
    for j in range(t):
        actual = np.where(flag[:, j], 0.0, np.minimum(actual + 1.0, tope))
        out[:, j] = actual
    return out


def _ultimo_valor(valores: np.ndarray, valido: np.ndarray, relleno: float = np.nan) -> np.ndarray:
    n, t = valores.shape
    out = np.full((n, t), relleno, dtype=np.float32)
    actual = np.full(n, relleno, dtype=np.float32)
    for j in range(t):
        actual = np.where(valido[:, j], valores[:, j], actual)
        out[:, j] = actual
    return out


# --------------------------------------------------------------------- DNP/rezago/licencia + encuesta
def _features_propias_multi(tablas: dict[str, pd.DataFrame], cortes: list[int]) -> pd.DataFrame:
    """DNP/rezago del empleador, licencias e historial de encuesta, para todos los `cortes` a la vez."""
    afiliados = tablas["afiliados"].sort_values("afiliado_id").reset_index(drop=True)
    panel = tablas["panel_mensual"]
    p_min = int(min(panel["periodo"].min(), tablas["encuesta"]["periodo"].min()
                    if len(tablas["encuesta"]) else panel["periodo"].min()))
    p_max = max(cortes)
    periodos = rango_periodos(p_min, p_max)
    per_pos = {p: t for t, p in enumerate(periodos)}
    af_ids = afiliados["afiliado_id"].to_numpy()
    afil_pos = {a: i for i, a in enumerate(af_ids)}
    n, n_t = len(af_ids), len(periodos)
    desf = MESES_DESFASE_RIM

    pan = panel[panel["periodo"].isin(per_pos)].copy()
    pan["empleador_id"] = pan["empleador_id"].fillna(0).astype("int64")
    emp_id = _matriz(pan, "empleador_id", afil_pos, per_pos, dtype=np.int64, relleno=0)
    dias_lic = _matriz(pan, "dias_licencia", afil_pos, per_pos)

    ae = tablas["agregado_empleador"]
    ae = ae[ae["periodo"].isin(per_pos)]
    n_emp = int(max(emp_id.max(), int(ae["empleador_id"].max()) if len(ae) else 0)) + 1
    dnp_emp = np.zeros((n_emp, n_t), dtype=np.float32)
    lag_emp = np.zeros((n_emp, n_t), dtype=np.float32)
    if len(ae):
        ie = ae["empleador_id"].to_numpy()
        ip = ae["periodo"].map(per_pos).to_numpy()
        dnp_emp[ie, ip] = ae["frac_dnp"].to_numpy(dtype=np.float32)
        lag_emp[ie, ip] = ae["lag_recepcion_medio"].to_numpy(dtype=np.float32)
    emp_id_seguro = np.clip(emp_id, 0, n_emp - 1)
    col_idx = np.arange(n_t)[None, :]
    dnp_flag = (dnp_emp[emp_id_seguro, col_idx] > 0) & (emp_id > 0)
    rezago_flag = (lag_emp[emp_id_seguro, col_idx] >= 2.0) & (emp_id > 0)
    licencia_larga = dias_lic >= 20

    # ---- encuesta: historial estrictamente ANTERIOR al corte (nunca el propio corte)
    enc = tablas["encuesta"]
    invitado_m = _matriz(enc.assign(_u=1), "_u", afil_pos, per_pos)
    respondio_m = _matriz(enc.assign(_u=enc["respondio"]), "_u", afil_pos, per_pos)
    resp_val_m = _matriz(enc[enc["respondio"] == 1], "respuesta", afil_pos, per_pos, relleno=np.nan)
    cat_cod = {"DETRACTOR": 1, "PASIVO": 2, "PROMOTOR": 3}
    enc_resp = enc[enc["respondio"] == 1].assign(_c=enc.loc[enc["respondio"] == 1, "categoria"].map(cat_cod))
    cat_m = _matriz(enc_resp, "_c", afil_pos, per_pos, dtype=np.float32, relleno=0.0)

    ii_list, jj_list = [], []
    idx_cortes = [per_pos[c] for c in cortes]
    for t in idx_cortes:
        idx = np.arange(n)
        ii_list.append(idx)
        jj_list.append(np.full(n, t))
    ii = np.concatenate(ii_list)
    jj = np.concatenate(jj_list)
    jj_lab = jj - desf

    datos: dict[str, np.ndarray] = {"afiliado_id": af_ids[ii],
                                    "periodo_corte": np.array(periodos, dtype=np.int64)[jj]}
    for k in VENTANAS:
        datos[f"dnp_empleador_{k}m"] = (_suma_movil(dnp_flag.astype(np.float32), k) / k)[ii, jj_lab]
        datos[f"rezago_mensual_{k}m"] = (_suma_movil(rezago_flag.astype(np.float32), k) / k)[ii, jj_lab]
        datos[f"licencia_larga_{k}m"] = _suma_movil(licencia_larga.astype(np.float32), k)[ii, jj_lab]
    datos["dnp_empleador_actual"] = dnp_flag.astype(np.float32)[ii, jj_lab]
    datos["rezago_mensual_actual"] = rezago_flag.astype(np.float32)[ii, jj_lab]

    # historial de encuesta: ventana hasta corte-1 (se excluye la columna del propio corte `jj`)
    jj_hist = np.maximum(jj - 1, 0)
    tiene_hist = jj >= 1
    for k in VENTANAS:
        n_inv = _suma_movil(invitado_m, k)[ii, jj_hist]
        n_resp = _suma_movil(respondio_m, k)[ii, jj_hist]
        datos[f"encuesta_invitaciones_{k}m"] = np.where(tiene_hist, n_inv, 0.0)
        datos[f"encuesta_respuestas_{k}m"] = np.where(tiene_hist, n_resp, 0.0)
    meses_desde_resp, nunca_resp = _meses_desde(respondio_m > 0), None
    datos["meses_desde_ultima_respuesta_nps"] = np.where(tiene_hist, meses_desde_resp[ii, jj_hist], 99.0)
    ultima_resp = _ultimo_valor(resp_val_m, respondio_m > 0, relleno=np.nan)
    datos["ultima_respuesta_nps"] = np.where(tiene_hist, ultima_resp[ii, jj_hist], np.nan)
    ultima_cat_cod = _ultimo_valor(cat_m, cat_m > 0, relleno=0.0)
    inv_cat = {v: k for k, v in cat_cod.items()}
    cod_hist = np.where(tiene_hist, ultima_cat_cod[ii, jj_hist], 0.0).astype(np.int64)
    datos["ultima_categoria_nps"] = pd.Categorical(
        [inv_cat.get(c, "SIN_HISTORIA") for c in cod_hist], categories=["SIN_HISTORIA", *cat_cod])
    datos["tiene_historia_nps"] = tiene_hist.astype(np.int8)

    return pd.DataFrame(datos)


# --------------------------------------------------------------------- API principal
def construir_features_multi(tablas: dict[str, pd.DataFrame], cortes: list[int]) -> pd.DataFrame:
    """Features as-of de todos los `cortes` a la vez (una fila por afiliado vigente x corte)."""
    cortes = sorted(int(c) for c in cortes)
    base = CT_F.construir_features_multi(_tablas_para_churn(tablas), cortes)

    tablas_apv = _tablas_para_apv(tablas)
    denso_apv = APV_F.preparar_panel(tablas_apv)
    partes_apv = []
    for corte in cortes:
        f = APV_F.construir_features(tablas_apv, corte, denso=denso_apv, solo_elegibles=False)
        cols = ["afiliado_id", "periodo_corte", *COLUMNAS_APV_EXTRA]
        partes_apv.append(f[cols])
    apv_extra = pd.concat(partes_apv, ignore_index=True).rename(
        columns={c: f"apv_{c}" for c in COLUMNAS_APV_EXTRA})

    propias = _features_propias_multi(tablas, cortes)

    df = base.merge(apv_extra, on=["afiliado_id", "periodo_corte"], how="left")
    df = df.merge(propias, on=["afiliado_id", "periodo_corte"], how="left")
    df["ultima_categoria_nps"] = df["ultima_categoria_nps"].astype("category")
    return df


def construir_features(tablas: dict[str, pd.DataFrame], periodo_corte: int) -> pd.DataFrame:
    """Features as-of de un unico corte (API pedida por CONVENCIONES §1)."""
    return construir_features_multi(tablas, [periodo_corte])


def construir_target_encuesta(tablas: dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """Agrega el resultado de la encuesta de ESTE `periodo_corte` (invitado/respondio/respuesta)."""
    enc = tablas["encuesta"].rename(columns={"periodo": "periodo_corte"})
    df = df.merge(enc[["afiliado_id", "periodo_corte", "respondio", "respuesta", "categoria"]],
                  on=["afiliado_id", "periodo_corte"], how="left")
    df["invitado"] = df["respondio"].notna().astype(np.int8)
    df["respondio"] = df["respondio"].fillna(0).astype(np.int8)
    df["es_detractor"] = (df["categoria"] == "DETRACTOR").astype(np.int8)
    df["es_promotor"] = (df["categoria"] == "PROMOTOR").astype(np.int8)
    return df


def construir_dataset(tablas: dict[str, pd.DataFrame], cortes: list[int]) -> pd.DataFrame:
    """Features as-of + resultado de la encuesta, para una lista de cortes."""
    df = construir_features_multi(tablas, cortes)
    return construir_target_encuesta(tablas, df)


def cortes_evaluables(tablas: dict[str, pd.DataFrame], horizonte: int = HORIZONTE_MESES,
                      meses_burn_in: int = 14) -> list[int]:
    """Cortes con >= `meses_burn_in` de historia y con los `horizonte` meses de target (valor del
    punto: traspaso/aporte a 6 meses) completos dentro de la ventana simulada."""
    periodos = sorted(tablas["saldos"]["periodo"].unique().tolist())
    primero = add_months(int(periodos[0]), meses_burn_in)
    ultimo = add_months(int(periodos[-1]), -horizonte)
    return list(rango_periodos(primero, ultimo))


def split_temporal(cortes: list[int], n_test: int = 12, n_valid: int = 12,
                   horizonte: int = HORIZONTE_MESES, max_cortes_train: int | None = None) -> dict[str, list[int]]:
    """Split temporal con purga (CONVENCIONES §3): reusa `churn_traspaso.features.split_temporal`
    ("Splits temporales con gap como en 02", brief)."""
    return CT_F.split_temporal(cortes, n_test=n_test, n_valid=n_valid, horizonte=horizonte,
                               max_cortes_train=max_cortes_train)


PREFIJOS_PROHIBIDOS = ("lat_", "p_")


def columnas_features(df: pd.DataFrame) -> list[str]:
    """Columnas utilizables por los modelos: excluye ids, target de la encuesta, cualquier columna
    con prefijo `lat_`/`p_` (CONVENCIONES §2) y las derivadas del propio pipeline de modelado
    (`peso_ipw`, `p_hat_responde` -- ver `COLUMNAS_DERIVADAS_PIPELINE`)."""
    return [c for c in df.columns
            if c not in COLUMNAS_NO_FEATURE and not c.startswith(PREFIJOS_PROHIBIDOS)]
