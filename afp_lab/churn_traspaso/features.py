"""Features "as-of" del proyecto 02 y armado del dataset (afiliado x periodo_corte).

Reglas de disponibilidad de informacion (CONVENCIONES §3 y BRIEF):

* `panel_mensual` (historia laboral y RIM): la cotizacion del mes *m* se conoce en *m+2*, por lo
  que el ultimo mes laboral usable en un corte es `corte - 2` (`MESES_DESFASE_RIM`).
* `saldos`, `contactos`, `reclamos`, `uso_digital`, `cambios_fondo`, `mercado`: son sistemas
  operacionales propios de la AFP, se conocen al dia -> ultimo mes usable `corte`.
* El target vive en `corte+1 .. corte+6` y nunca toca las features.

`construir_features(tablas, periodo_corte)` **recorta las tablas antes de calcular nada**
(`recortar_tablas`), de modo que el leakage es imposible por construccion: cualquier registro
posterior al corte simplemente no entra al calculo. `construir_features_multi` calcula muchos
cortes de una vez sobre matrices (afiliado x mes) con ventanas moviles hacia atras; el test
`test_features_multi_igual_a_corte_unico` verifica que ambas rutas dan exactamente lo mismo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx, rango_periodos

from . import HORIZONTE_MESES, MESES_DESFASE_RIM
from .synthetic import AFP_PROPIA, AFP_TODAS, CANALES, FONDOS

EDAD_LEGAL_PENSION = {"F": 60, "M": 65}

# columnas categoricas del dataset (LightGBM las usa como 'category')
COLUMNAS_CATEGORICAS = [
    "tipo_afiliado", "sexo", "nivel_educacional", "region", "fondo", "rubro",
    "tamano_empleador", "tipo_contrato", "estado_laboral", "afp_licitacion",
    "tipo_ultimo_reclamo",
]
COLUMNAS_ID = ["afiliado_id", "periodo_corte"]
# columnas que no son features (identificadores, target y auxiliares de negocio/segmento)
COLUMNAS_NO_FEATURE = COLUMNAS_ID + ["y", "periodo_solicitud_target", "afp_destino_target"]


# --------------------------------------------------------------------- recorte as-of
def recortar_tablas(tablas: dict[str, pd.DataFrame], periodo_corte: int,
                    meses_desfase: int = MESES_DESFASE_RIM) -> dict[str, pd.DataFrame]:
    """Deja solo la informacion conocida por la AFP al cierre de `periodo_corte`.

    El panel laboral se corta en `corte - meses_desfase` (desfase real de la cotizacion);
    el resto de las tablas operacionales, en `corte`.
    """
    corte_lab = add_months(periodo_corte, -meses_desfase)
    out = dict(tablas)
    out["panel_mensual"] = tablas["panel_mensual"][tablas["panel_mensual"]["periodo"] <= corte_lab]
    for nombre in ("saldos", "uso_digital", "contactos", "reclamos", "cambios_fondo", "mercado", "macro"):
        if nombre in tablas and len(tablas[nombre]):
            out[nombre] = tablas[nombre][tablas[nombre]["periodo"] <= periodo_corte]
    tr = tablas["traspasos"]
    out["traspasos"] = tr[tr["periodo_solicitud"] <= periodo_corte] if len(tr) else tr
    return out


# --------------------------------------------------------------------- helpers matriciales
def _suma_movil(mat: np.ndarray, k: int) -> np.ndarray:
    """Suma de los ultimos `k` meses INCLUYENDO la columna actual (solo mira hacia atras)."""
    c = np.cumsum(mat, axis=1, dtype=np.float64)
    out = c.copy()
    if k < mat.shape[1]:
        out[:, k:] = c[:, k:] - c[:, :-k]
    return out.astype(np.float32)


def _meses_desde(flag: np.ndarray, tope: int = 99) -> np.ndarray:
    """Meses transcurridos desde la ultima vez que `flag` fue verdadero (incluye el mes actual)."""
    n, t = flag.shape
    out = np.full((n, t), tope, dtype=np.float32)
    actual = np.full(n, tope, dtype=np.float32)
    for j in range(t):
        actual = np.where(flag[:, j], 0.0, np.minimum(actual + 1.0, tope))
        out[:, j] = actual
    return out


def _ultimo_valor(valores: np.ndarray, valido: np.ndarray, relleno: float = np.nan) -> np.ndarray:
    """Ultimo valor observado hasta cada mes (forward fill hacia adelante, nunca hacia atras)."""
    n, t = valores.shape
    out = np.full((n, t), relleno, dtype=np.float32)
    actual = np.full(n, relleno, dtype=np.float32)
    for j in range(t):
        actual = np.where(valido[:, j], valores[:, j], actual)
        out[:, j] = actual
    return out


def _matriz(df: pd.DataFrame, col_valor: str, afil_pos: dict, per_pos: dict,
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
        np.add.at(m, (i, j), d[col_valor].to_numpy())
    else:
        m[i, j] = d[col_valor].to_numpy()
    return m


# --------------------------------------------------------------------- features
def construir_features_multi(tablas: dict[str, pd.DataFrame], cortes: list[int]) -> pd.DataFrame:
    """Features as-of para varios cortes de una vez (una fila por afiliado vigente x corte).

    Todas las ventanas moviles miran hacia atras y las familias laborales se leen en `corte-2`.
    """
    cortes = sorted(int(c) for c in cortes)
    afiliados = tablas["afiliados"].sort_values("afiliado_id").reset_index(drop=True)
    panel, saldos = tablas["panel_mensual"], tablas["saldos"]
    p_min = int(min(panel["periodo"].min(), saldos["periodo"].min()))
    p_max = max(cortes)
    periodos = rango_periodos(p_min, p_max)
    per_pos = {p: t for t, p in enumerate(periodos)}
    af_ids = afiliados["afiliado_id"].to_numpy()
    afil_pos = {a: i for i, a in enumerate(af_ids)}
    n, n_t = len(af_ids), len(periodos)
    desf = MESES_DESFASE_RIM

    # ---------------------------------------------------------------- 1. panel laboral (corte-2)
    pan = panel[panel["periodo"].isin(per_pos)].copy()
    pan["empleador_id"] = pan["empleador_id"].fillna(0).astype("int64")
    rim = (_matriz(pan, "rim_emp1", afil_pos, per_pos) + _matriz(pan, "rim_emp2", afil_pos, per_pos))
    cotiza = (rim > 0).astype(np.float32)
    estado = _matriz(pan, "estado", afil_pos, per_pos, dtype=object, relleno="SIN_INFO")
    emp_id = _matriz(pan, "empleador_id", afil_pos, per_pos, dtype=np.float64, relleno=0.0)
    antig_emp = _matriz(pan, "meses_antiguedad", afil_pos, per_pos)
    edad_pan = _matriz(pan, "edad", afil_pos, per_pos, dtype=np.float32, relleno=0.0)
    contrato = _matriz(pan, "tipo_contrato", afil_pos, per_pos, dtype=object, relleno="SIN_CONTRATO")
    contrato = np.where(pd.isna(contrato), "SIN_CONTRATO", contrato)

    n_cot_6, n_cot_12 = _suma_movil(cotiza, 6), _suma_movil(cotiza, 12)
    suma_3, suma_12 = _suma_movil(rim, 3), _suma_movil(rim, 12)
    suma_sq_12 = _suma_movil(rim ** 2, 12)
    media_3 = suma_3 / 3.0
    media_12 = suma_12 / 12.0
    var_12 = np.maximum(suma_sq_12 / 12.0 - media_12 ** 2, 0.0)
    rim_ref = _ultimo_valor(rim, cotiza > 0, relleno=np.nan)
    cambio_emp = np.zeros((n, n_t), dtype=bool)
    cambio_emp[:, 1:] = (emp_id[:, 1:] != emp_id[:, :-1]) & (emp_id[:, 1:] > 0)

    lab = {
        "rim_l1": rim,
        "rim_media_3": media_3,
        "rim_media_12": media_12,
        "rim_cv_12": np.where(media_12 > 0, np.sqrt(var_12) / np.maximum(media_12, 1.0), 0.0),
        "rim_ref": rim_ref,
        "rim_tendencia_3_12": np.where(media_12 > 0, media_3 / np.maximum(media_12, 1.0), 1.0),
        "densidad_6": n_cot_6 / 6.0,
        "densidad_12": n_cot_12 / 12.0,
        "meses_desde_ult_cotiza": _meses_desde(cotiza > 0),
        "meses_con_empleador": antig_emp,
        "meses_desde_cambio_empleador": _meses_desde(cambio_emp),
        "n_cambios_empleador_12": _suma_movil(cambio_emp.astype(np.float32), 12),
        "edad_panel": edad_pan,
    }
    lab_cat = {"estado_laboral": estado, "tipo_contrato": contrato, "_emp_id": emp_id}

    # ---------------------------------------------------------------- 2. cuenta y vigencia
    vig = _matriz(saldos, "afiliado_id", afil_pos, per_pos, dtype=np.float32, relleno=0.0) > 0
    saldo = _matriz(saldos, "saldo_obligatorio", afil_pos, per_pos, dtype=np.float32)
    antig_afp = _matriz(saldos, "antiguedad_afp_meses", afil_pos, per_pos)
    fondo_cod = saldos.assign(_f=saldos["fondo"].map({f: i + 1 for i, f in enumerate(FONDOS)}))
    fondo = _matriz(fondo_cod, "_f", afil_pos, per_pos, dtype=np.int8, relleno=0)

    cf = tablas["cambios_fondo"]
    cf_flag = _matriz(cf.assign(_u=1) if len(cf) else cf, "_u", afil_pos, per_pos, agregar=True)
    cuenta = {
        "saldo_obligatorio": saldo,
        "log_saldo": np.log1p(np.maximum(saldo, 0.0)),
        "antiguedad_afp_meses": antig_afp,
        "cambios_fondo_3m": _suma_movil(cf_flag, 3),
        "cambios_fondo_12m": _suma_movil(cf_flag, 12),
        "meses_desde_cambio_fondo": _meses_desde(cf_flag > 0),
    }

    # ---------------------------------------------------------------- 3. relacion con la AFP
    con = tablas["contactos"]
    rel: dict[str, np.ndarray] = {}
    canal_mat = {}
    for canal in CANALES:
        sub = con[con["canal"] == canal].assign(_u=1) if len(con) else con
        m = _matriz(sub, "_u", afil_pos, per_pos, agregar=True)
        canal_mat[canal] = m
        base = canal.lower()
        rel[f"contactos_{base}_1m"] = m
        rel[f"contactos_{base}_3m"] = _suma_movil(m, 3)
        rel[f"contactos_{base}_6m"] = _suma_movil(m, 6)
    rel["meses_desde_contacto_competencia"] = _meses_desde(canal_mat["VENDEDOR_COMPETENCIA"] > 0)
    rel["meses_desde_contacto_ejecutivo"] = _meses_desde(canal_mat["EJECUTIVO_PROPIO"] > 0)
    if len(con):
        sub = con[(con["canal"] == "VENDEDOR_COMPETENCIA") & (con["resultado"] == "INTERESADO")].assign(_u=1)
    else:
        sub = con
    m_int = _matriz(sub, "_u", afil_pos, per_pos, agregar=True)
    rel["contactos_competencia_interesado_6m"] = _suma_movil(m_int, 6)

    rec = tablas["reclamos"]
    rec_u = rec.assign(_u=1) if len(rec) else rec
    m_rec = _matriz(rec_u, "_u", afil_pos, per_pos, agregar=True)
    rec_ab = rec[rec["estado"] == "ABIERTO"].assign(_u=1) if len(rec) else rec
    rec_df = rec[rec["estado"] == "RESUELTO_DESFAVORABLE"].assign(_u=1) if len(rec) else rec
    rel["reclamos_6m"] = _suma_movil(m_rec, 6)
    rel["reclamos_12m"] = _suma_movil(m_rec, 12)
    rel["reclamos_abiertos_12m"] = _suma_movil(_matriz(rec_ab, "_u", afil_pos, per_pos, agregar=True), 12)
    rel["reclamos_desfavorables_12m"] = _suma_movil(_matriz(rec_df, "_u", afil_pos, per_pos, agregar=True), 12)
    rel["meses_desde_reclamo"] = _meses_desde(m_rec > 0)
    if len(rec):
        tipos = sorted(rec["tipo"].unique())
        cod = {t: i + 1 for i, t in enumerate(tipos)}
        m_tipo = _matriz(rec.assign(_t=rec["tipo"].map(cod)), "_t", afil_pos, per_pos,
                         dtype=np.float32, relleno=0.0)
        tipo_rec_cod = _ultimo_valor(m_tipo, m_tipo > 0, relleno=0.0)
        etiquetas_reclamo = np.array(["SIN_RECLAMO"] + tipos, dtype=object)
    else:
        tipo_rec_cod = np.zeros((n, n_t), dtype=np.float32)
        etiquetas_reclamo = np.array(["SIN_RECLAMO"], dtype=object)

    dig = tablas["uso_digital"]
    logins = (_matriz(dig, "logins_app", afil_pos, per_pos) + _matriz(dig, "logins_web", afil_pos, per_pos))
    consultas = _matriz(dig, "consultas_saldo", afil_pos, per_pos)
    certif = _matriz(dig, "descargas_certificado", afil_pos, per_pos)
    logins_3, logins_12 = _suma_movil(logins, 3), _suma_movil(logins, 12)
    rel.update({
        "logins_1m": logins,
        "logins_3m": logins_3,
        "consultas_saldo_3m": _suma_movil(consultas, 3),
        "certificados_1m": certif,
        "certificados_3m": _suma_movil(certif, 3),
        "ratio_logins_3m_12m": np.where(logins_12 > 0, logins_3 / np.maximum(logins_12 / 4.0, 0.25), 1.0),
    })

    # ---------------------------------------------------------------- 4. mercado
    merc = tablas["mercado"]
    merc = merc[merc["periodo"].isin(per_pos)]
    piv_com = merc.pivot(index="periodo", columns="afp", values="comision").reindex(periodos).ffill()
    com_propia = piv_com[AFP_PROPIA].to_numpy(dtype=np.float32)
    com_min = piv_com[list(AFP_TODAS)].min(axis=1).to_numpy(dtype=np.float32)
    rent_propia_f, rent_prom_f = {}, {}
    for f in FONDOS:
        piv = merc.pivot(index="periodo", columns="afp", values=f"rent_12m_{f}").reindex(periodos).ffill()
        rent_propia_f[f] = piv[AFP_PROPIA].to_numpy(dtype=np.float32)
        rent_prom_f[f] = piv[list(AFP_TODAS)].mean(axis=1).to_numpy(dtype=np.float32)
    gan = merc[merc["gano_licitacion"] == 1].set_index("periodo")["afp"].reindex(periodos).ffill().bfill()
    afp_licit = gan.to_numpy(dtype=object)

    # ---------------------------------------------------------------- 5. armado del largo
    idx_cortes = [per_pos[c] for c in cortes]
    tr = tablas["traspasos"]
    sal = tr[tr["tipo"] == "SALIDA"] if len(tr) else tr
    mes_salida = pd.Series(np.inf, index=af_ids, dtype=float)
    if len(sal):
        primera = sal.groupby("afiliado_id")["periodo_solicitud"].min()
        mes_salida.loc[primera.index] = [per_pos.get(int(p), np.inf) for p in primera.to_numpy()]
    mes_salida_arr = mes_salida.to_numpy()

    ii_list, jj_list = [], []
    for t in idx_cortes:
        t_lab = t - desf
        if t_lab < 0:
            continue
        # poblacion: vigente al corte, con solicitud de salida aun no presentada y no retirado
        ok = vig[:, t] & (mes_salida_arr > t) & (estado[:, t_lab] != "RETIRADO")
        idx = np.where(ok)[0]
        ii_list.append(idx)
        jj_list.append(np.full(len(idx), t))
    if not ii_list:
        return pd.DataFrame()
    ii = np.concatenate(ii_list)
    jj = np.concatenate(jj_list)
    jj_lab = jj - desf

    datos: dict[str, np.ndarray] = {
        "afiliado_id": af_ids[ii],
        "periodo_corte": np.array(periodos, dtype=np.int64)[jj],
    }
    for nombre, m in lab.items():
        datos[nombre] = m[ii, jj_lab]
    datos["estado_laboral"] = lab_cat["estado_laboral"][ii, jj_lab]
    datos["tipo_contrato"] = lab_cat["tipo_contrato"][ii, jj_lab]
    emp_corte = lab_cat["_emp_id"][ii, jj_lab]
    for nombre, m in cuenta.items():
        datos[nombre] = m[ii, jj]
    for nombre, m in rel.items():
        datos[nombre] = m[ii, jj]
    datos["tipo_ultimo_reclamo"] = etiquetas_reclamo[tipo_rec_cod[ii, jj].astype(np.int64)]

    fondo_idx = fondo[ii, jj]
    datos["fondo"] = np.where(fondo_idx > 0, np.array(["?"] + list(FONDOS))[fondo_idx], "SIN_FONDO")
    rp = np.stack([rent_propia_f[f] for f in FONDOS])          # (5, n_t)
    rs = np.stack([rent_prom_f[f] for f in FONDOS])
    k = np.clip(fondo_idx - 1, 0, len(FONDOS) - 1)
    datos["rent_12m_propia_fondo"] = rp[k, jj]
    datos["rent_12m_sistema_fondo"] = rs[k, jj]
    datos["brecha_rentabilidad_pp"] = rp[k, jj] - rs[k, jj]
    datos["comision_propia"] = com_propia[jj]
    datos["comision_min_mercado"] = com_min[jj]
    datos["brecha_comision_pp"] = com_propia[jj] - com_min[jj]
    datos["brecha_comision_rel"] = (com_propia[jj] - com_min[jj]) / np.maximum(com_min[jj], 0.01)
    datos["afp_licitacion"] = afp_licit[jj]
    datos["licitacion_propia"] = (afp_licit[jj] == AFP_PROPIA).astype(np.int8)
    datos["mes"] = (np.array(periodos, dtype=np.int64)[jj] % 100).astype(np.int8)

    df = pd.DataFrame(datos)

    # atributos estables del afiliado y del empleador
    af = afiliados.set_index("afiliado_id")
    df = df.join(af[["sexo", "region", "nivel_educacional", "tipo_afiliado", "periodo_afiliacion",
                     "edad_inicio"]], on="afiliado_id")
    idx_corte = df["periodo_corte"].map(periodo_to_idx)
    df["meses_desde_afiliacion"] = idx_corte - df["periodo_afiliacion"].map(periodo_to_idx)
    edad = np.where(df["edad_panel"].to_numpy() > 0, df["edad_panel"].to_numpy(),
                    df["edad_inicio"].to_numpy() + (jj_lab // 12))
    df["edad"] = edad.astype(np.int16)
    edad_legal = df["sexo"].map(EDAD_LEGAL_PENSION).to_numpy()
    df["meses_hasta_pension"] = ((edad_legal - df["edad"].to_numpy()) * 12).astype(np.int32)
    df.drop(columns=["edad_panel", "edad_inicio", "periodo_afiliacion"], inplace=True)

    emp = tablas["empleadores"].set_index("empleador_id")[["rubro", "tamano_empleador"]]
    emp_s = pd.Series(np.nan_to_num(emp_corte, nan=0.0).astype("int64"), name="empleador_id")
    df["rubro"] = emp_s.map(emp["rubro"]).fillna("SIN_EMPLEADOR").to_numpy()
    df["tamano_empleador"] = emp_s.map(emp["tamano_empleador"]).fillna("SIN_EMPLEADOR").to_numpy()

    # ratios derivados utiles y segmentos de negocio
    ref = df["rim_ref"].to_numpy(dtype=np.float64)
    df["saldo_sobre_rim_ref"] = np.where(ref > 0, df["saldo_obligatorio"].to_numpy() / np.maximum(ref, 1.0), np.nan)
    df["rim_ref"] = df["rim_ref"].astype(np.float32)
    df["region"] = df["region"].astype("int16")
    for c in COLUMNAS_CATEGORICAS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df.reset_index(drop=True)


def construir_features(tablas: dict[str, pd.DataFrame], periodo_corte: int) -> pd.DataFrame:
    """Features as-of de un unico corte (API pedida por CONVENCIONES §1).

    Recorta las tablas a lo conocido al corte y delega en `construir_features_multi`.
    """
    return construir_features_multi(recortar_tablas(tablas, periodo_corte), [periodo_corte])


# --------------------------------------------------------------------- target y dataset
def construir_target(tablas: dict[str, pd.DataFrame], df: pd.DataFrame,
                     horizonte: int = HORIZONTE_MESES) -> pd.DataFrame:
    """Agrega `y` = 1 si el afiliado solicita un traspaso de SALIDA en `corte+1 .. corte+h`."""
    tr = tablas["traspasos"]
    sal = tr[tr["tipo"] == "SALIDA"][["afiliado_id", "periodo_solicitud", "afp_destino"]] if len(tr) else None
    df = df.copy()
    if sal is None or not len(sal):
        df["y"] = 0
        df["periodo_solicitud_target"] = pd.NA
        df["afp_destino_target"] = pd.NA
        return df
    sal = sal.sort_values("periodo_solicitud").drop_duplicates("afiliado_id", keep="first")
    m = df[["afiliado_id", "periodo_corte"]].merge(sal, on="afiliado_id", how="left")
    ini = m["periodo_corte"].map(periodo_to_idx) + 1
    fin = m["periodo_corte"].map(periodo_to_idx) + horizonte
    sol = m["periodo_solicitud"].map(lambda p: periodo_to_idx(int(p)) if pd.notna(p) else np.nan)
    dentro = (sol >= ini) & (sol <= fin)
    df["y"] = dentro.fillna(False).astype(np.int8).to_numpy()
    df["periodo_solicitud_target"] = np.where(dentro.fillna(False), m["periodo_solicitud"], np.nan)
    df["afp_destino_target"] = np.where(dentro.fillna(False), m["afp_destino"], None)
    return df


def cortes_evaluables(tablas: dict[str, pd.DataFrame], horizonte: int = HORIZONTE_MESES,
                      meses_burn_in: int = 14) -> list[int]:
    """Cortes con historia suficiente (>= `meses_burn_in`) y con los `horizonte` meses completos."""
    periodos = sorted(tablas["saldos"]["periodo"].unique().tolist())
    p_ini, p_fin = periodos[0], periodos[-1]
    primero = add_months(int(p_ini), meses_burn_in)
    ultimo = add_months(int(p_fin), -horizonte)
    return [p for p in rango_periodos(primero, ultimo)]


def construir_dataset(tablas: dict[str, pd.DataFrame], cortes: list[int],
                      horizonte: int = HORIZONTE_MESES) -> pd.DataFrame:
    """Features as-of + target para una lista de cortes."""
    df = construir_features_multi(tablas, cortes)
    return construir_target(tablas, df, horizonte)


def columnas_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in COLUMNAS_NO_FEATURE]


def split_temporal(cortes: list[int], n_test: int = 3, n_valid: int = 3,
                   horizonte: int = HORIZONTE_MESES) -> dict[str, list[int]]:
    """Split temporal con gap de `horizonte` meses entre el ultimo target y el corte siguiente.

    test  = los ultimos `n_test` cortes evaluables.
    valid = `n_valid` cortes cuyo ultimo target queda `horizonte` meses antes del primer test.
    train = todos los cortes anteriores que respetan el mismo gap frente a validacion.
    """
    cortes = sorted(cortes)
    test = cortes[-n_test:]
    limite_valid = add_months(test[0], -2 * horizonte)          # corte + h + gap(h) <= primer test
    candidatos = [c for c in cortes if c <= limite_valid]
    valid = candidatos[-n_valid:]
    limite_train = add_months(valid[0], -2 * horizonte)
    train = [c for c in cortes if c <= limite_train]
    return {"train": train, "valid": valid, "test": test}
