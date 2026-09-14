"""Tablas sinteticas del proyecto 11 (voz del afiliado: NPS y experiencia).

Parte del universo comun `cuprum` y reutiliza SIN MODIFICAR las tablas de:

* `afp_lab.churn_traspaso.synthetic.generar_tablas` (mercado, saldos, cambios de fondo, traspasos,
  contactos, reclamos, uso digital);
* `afp_lab.propension_apv.synthetic.generar_tablas` (APVC del empleador, campanas, aportes
  voluntarios, saldo voluntario, tributario);
* `afp_lab.common.planilla` (`generar_planilla`, `construir_rezagos`, `construir_agregado_empleador`)
  para el rezago de recepcion y la DNP del empleador.

Ambas llamadas a `generar_tablas` reciben el MISMO `n_afiliados`, `periodo_inicio`, `periodo_fin`,
`seed` y `perfil`; como `generar_universo` es una funcion pura de esos argumentos, el universo comun
(`afiliados`, `empleadores`, `panel_mensual`, `macro`) que arma cada una es identico (se verifica en
los tests). `empleadores_latente` (necesario para `generar_planilla`, que no es parte del universo
"publico" que exponen 02/03) se reconstruye aparte con `generar_empleadores_pandas`, que es la MISMA
funcion pura que usa `generar_universo` internamente (mismo seed): barato, porque no vuelve a
simular el panel laboral mes a mes.

Este modulo agrega la capa propia del proyecto: la **satisfaccion latente** (`lat_satisfaccion`,
0-10) por afiliado y mes, y la **encuesta de NPS** (invitacion + respuesta, con sesgo de no
respuesta). Nada de esto se usa en 02/03/planilla; solo LEE sus tablas.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso.model import ajustar_prior
from afp_lab.churn_traspaso.synthetic import FONDOS
from afp_lab.churn_traspaso.synthetic import generar_tablas as generar_tablas_churn
from afp_lab.common.planilla import construir_agregado_empleador, construir_rezagos, generar_planilla
from afp_lab.propension_apv.synthetic import generar_tablas as generar_tablas_apv
from afp_lab.rim_nowcasting.calendario import periodo_to_idx, rango_periodos
from afp_lab.rim_nowcasting.config import Config
from afp_lab.rim_nowcasting.synthetic import generar_empleadores_pandas

PROMOTOR_DESDE = 9     # 9-10
DETRACTOR_HASTA = 6    # 0-6 (7-8 = pasivo)


@dataclass(frozen=True)
class ParametrosSatisfaccion:
    """Coeficientes del generador de satisfaccion latente y de la encuesta (todos documentados en
    el README, seccion "Datos sinteticos"). Las magnitudes de los eventos son las del BRIEF
    (ronda 5, `docs/briefs/ronda_5/voz_afiliado.md`); el resto son supuestos de trabajo razonables
    para una escala 0-10, declarados aqui para que se puedan recalibrar sin tocar la logica.
    """

    # --- nivel base --------------------------------------------------------------------
    # 7.6 (no 7.0): Cuprum tiene una brecha de comision CRONICA (-0.6, casi siempre activa dado que
    # es la AFP mas cara del perfil "cuprum" en `churn_traspaso.MERCADOS`), asi que el nivel base
    # se sube para que el promedio neto de la cartera quede cerca de 7 (frontera pasivo/promotor) y
    # no todo el mundo caiga a detractor solo por el precio.
    base_nivel: float = 7.6                 # nivel medio de una cartera moderadamente satisfecha
    base_segmento: dict = None              # se completa en __post_init__ (no literal mutable)
    base_joven: float = -0.3                # < 30 anos
    base_mayor: float = 0.4                 # >= 60 anos
    sigma_persistente: float = 1.0          # componente fijo por afiliado (humor/temperamento base)
    sigma_mes: float = 0.8                  # ruido idiosincratico mes a mes

    # --- efectos de eventos (magnitudes del brief; escala logit no, escala PUNTOS de NPS) -----
    ef_reclamo_sin_resolver: float = -2.0
    ef_rezago_mensual: float = -1.5         # empleador con recepcion tardia (>= 1 mes, "> 30 dias")
    ef_licencia_mal_pagada: float = -2.5
    ef_dnp_empleador: float = -0.8
    ef_contacto_ejecutivo_resuelto: float = 0.7
    ef_rentabilidad_relativa_positiva: float = 0.5
    ef_brecha_comision_desfavorable: float = -0.6

    # --- decaimiento (vida media en meses) de los efectos NO contemporaneos -------------------
    vida_media_evento: float = 2.0
    tope_meses_evento: int = 6
    vida_media_dnp: float = 1.0             # la DNP del empleador se "olvida" mas rapido
    tope_meses_dnp: int = 3
    umbral_brecha_comision_pp: float = 0.30  # pp sobre la mas barata para considerarla "desfavorable"
    umbral_licencia_mal_pagada_dias: int = 20  # dias de licencia en el mes para la etiqueta "mal pagada"
    sigma_respuesta: float = 0.6            # ruido propio de contestar la encuesta (redondeo + humor del dia)

    # --- encuesta ------------------------------------------------------------------------------
    p_invitacion_objetivo: float = 0.08     # brief: ~8 % de la cartera invitada cada mes
    p_respuesta_objetivo: float = 0.25      # brief: ~25 % de los invitados responde
    peso_evento_reciente_invitacion: float = 2.0   # logit: encuesta transaccional tras un evento
    peso_extremo_respuesta: float = 0.35   # por punto FUERA de la banda pasiva (7-8), ver `_extremidad`
    peso_joven_respuesta: float = -0.5     # jovenes responden menos
    peso_uso_digital_bajo_respuesta: float = -0.4  # bajo uso digital responde menos

    def __post_init__(self) -> None:
        if self.base_segmento is None:
            object.__setattr__(self, "base_segmento", {
                "DEPENDIENTE": 0.0, "INDEPENDIENTE": -0.3,
                "VOLUNTARIO": 0.2, "PENSIONADO_ACTIVO": 0.3,
            })


DESCRIPCION_PARAMETROS = {
    "base_nivel": "Nivel medio de satisfaccion (0-10) de una cartera moderadamente satisfecha",
    "base_joven": "Ajuste de nivel base para menores de 30 anos",
    "base_mayor": "Ajuste de nivel base para mayores o iguales a 60 anos",
    "sigma_persistente": "Desviacion del componente fijo (temperamento) por afiliado",
    "sigma_mes": "Desviacion del ruido idiosincratico mes a mes",
    "ef_reclamo_sin_resolver": "Efecto de un reclamo ABIERTO o RESUELTO_DESFAVORABLE",
    "ef_rezago_mensual": "Efecto de que el empleador reciba/pague con >= 1 mes de atraso (~> 30 dias)",
    "ef_licencia_mal_pagada": "Efecto de una licencia larga (proxy de subsidio mal/tarde pagado)",
    "ef_dnp_empleador": "Efecto de que el empleador este en DNP ese mes",
    "ef_contacto_ejecutivo_resuelto": "Efecto de un contacto del ejecutivo propio con resultado positivo",
    "ef_rentabilidad_relativa_positiva": "Efecto de que el fondo del afiliado rente sobre el promedio del mercado",
    "ef_brecha_comision_desfavorable": "Efecto de que la comision propia supere a la mas barata en >= umbral",
    "vida_media_evento": "Vida media (meses) del decaimiento de los eventos no contemporaneos",
    "tope_meses_evento": "Ventana maxima (meses) del decaimiento de esos eventos",
    "vida_media_dnp": "Vida media (meses) del efecto de la DNP del empleador",
    "tope_meses_dnp": "Ventana maxima (meses) del efecto de la DNP",
    "umbral_brecha_comision_pp": "Brecha de comision (pp) para considerarla desfavorable",
    "umbral_licencia_mal_pagada_dias": "Dias de licencia en el mes para la etiqueta licencia mal pagada",
    "sigma_respuesta": "Ruido propio de la respuesta de la encuesta (ademas del redondeo)",
    "p_invitacion_objetivo": "Fraccion mensual objetivo de la cartera invitada a la encuesta",
    "p_respuesta_objetivo": "Fraccion objetivo de los invitados que responde",
    "peso_evento_reciente_invitacion": "Peso (logit) de un evento operativo reciente en la invitacion",
    "peso_extremo_respuesta": "Peso (logit) de la distancia fuera de la banda pasiva (7-8) en la respuesta",
    "peso_joven_respuesta": "Peso (logit) de ser joven (< 30) en la respuesta",
    "peso_uso_digital_bajo_respuesta": "Peso (logit) de bajo uso digital en la respuesta",
}


def tabla_parametros(par: "ParametrosSatisfaccion | None" = None) -> pd.DataFrame:
    par = par or ParametrosSatisfaccion()
    filas = []
    for campo in fields(par):
        if campo.name == "base_segmento":
            continue
        v = getattr(par, campo.name)
        filas.append({"parametro": campo.name, "valor": v,
                      "descripcion": DESCRIPCION_PARAMETROS.get(campo.name, "")})
    for seg, v in par.base_segmento.items():
        filas.append({"parametro": f"base_segmento[{seg}]", "valor": v,
                      "descripcion": "Ajuste de nivel base por tipo_afiliado"})
    return pd.DataFrame(filas)


EFECTOS_VERDADEROS_EVENTOS = {
    "reclamo_sin_resolver": ("ef_reclamo_sin_resolver", -2.0),
    "rezago_mensual": ("ef_rezago_mensual", -1.5),
    "licencia_mal_pagada": ("ef_licencia_mal_pagada", -2.5),
    "dnp_empleador": ("ef_dnp_empleador", -0.8),
    "contacto_ejecutivo_resuelto": ("ef_contacto_ejecutivo_resuelto", 0.7),
    "rentabilidad_relativa_positiva": ("ef_rentabilidad_relativa_positiva", 0.5),
    "brecha_comision_desfavorable": ("ef_brecha_comision_desfavorable", -0.6),
}


# --------------------------------------------------------------------------- utilidades matriciales
def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _extremidad(sat: np.ndarray) -> np.ndarray:
    """Distancia FUERA de la banda pasiva (7-8): 0 dentro de la banda, positiva y creciente hacia
    ambos lados. Ancla el sesgo de no respuesta ("contestan mas los muy satisfechos y los muy
    insatisfechos") en las fronteras REALES de promotor/detractor, no en el punto medio de la escala
    (5): con un nivel base ~7 (banda pasiva), usar `|sat-5|` trataria a un pasivo alto (8) como mas
    "extremo" que un detractor moderado (6), que es lo opuesto de lo que dice el brief."""
    return np.where(sat >= PROMOTOR_DESDE, sat - (PROMOTOR_DESDE - 1),
                    np.where(sat <= DETRACTOR_HASTA, (DETRACTOR_HASTA + 1) - sat, 0.0))


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


def _efecto_decaido(flag: np.ndarray, magnitud: float, vida_media_meses: float, tope: int) -> np.ndarray:
    """`magnitud * decaimiento^k` sumado sobre los ultimos `tope` meses (incluye el mes actual)."""
    decay = 0.5 ** (1.0 / vida_media_meses)
    out = np.zeros(flag.shape, dtype=np.float64)
    f = flag.astype(np.float64)
    for k in range(tope + 1):
        w = magnitud * (decay ** k)
        if k == 0:
            out += w * f
        else:
            out[:, k:] += w * f[:, :-k]
    return out


def _empleadores_latente(n_afiliados: int, periodo_inicio: int, periodo_fin: int, seed: int,
                         afiliados_por_empleador: int = 40) -> pd.DataFrame:
    """Reconstruye `empleadores_latente` sin volver a simular el panel laboral (mismo seed que
    `synth_base.generar_universo`, que arma su `Config` con los mismos argumentos)."""
    cfg = Config(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio, periodo_snapshot=periodo_fin,
                seed=seed, afiliados_por_empleador=afiliados_por_empleador)
    return generar_empleadores_pandas(cfg)


# --------------------------------------------------------------------------- generador principal
def generar_tablas(n_afiliados: int = 10_000, periodo_inicio: int = 202201, periodo_fin: int = 202812,
                   seed: int = 42, perfil: str = "cuprum",
                   par: ParametrosSatisfaccion | None = None) -> dict[str, pd.DataFrame]:
    """Universo `cuprum` + tablas de 02/03/planilla (reutilizadas) + satisfaccion latente y encuesta.

    Devuelve un dict con el universo comun, las tablas de `churn_traspaso` (`mercado`, `saldos`,
    `cambios_fondo`, `traspasos`, `contactos`, `reclamos`, `uso_digital`), las de `propension_apv`
    (`apvc_empleador`, `campanas_apv`, `aportes_voluntarios`, `saldos_apv`, `tributario`), las de
    `common/planilla` (`planilla`, `rezagos`, `agregado_empleador`) y las propias (`encuesta`,
    `satisfaccion_latente`).
    """
    par = par or ParametrosSatisfaccion()

    tablas_churn = generar_tablas_churn(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                        periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    tablas_apv = generar_tablas_apv(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                    periodo_fin=periodo_fin, seed=seed, perfil=perfil)

    afiliados = tablas_churn["afiliados"]
    panel = tablas_churn["panel_mensual"]
    emp_lat = _empleadores_latente(n_afiliados, periodo_inicio, periodo_fin, seed)
    planilla = generar_planilla({"panel_mensual": panel, "afiliados": afiliados,
                                 "empleadores_latente": emp_lat}, seed)
    rezagos = construir_rezagos(planilla, seed)
    agregado_empleador = construir_agregado_empleador(planilla)

    periodos = rango_periodos(periodo_inicio, periodo_fin)
    per_pos = {p: t for t, p in enumerate(periodos)}
    af_ids = afiliados["afiliado_id"].to_numpy()
    afil_pos = {a: i for i, a in enumerate(af_ids)}
    n, n_t = len(af_ids), len(periodos)

    # ---- vigente en la cartera (mismo indicador que 02: presencia en `saldos`)
    saldos = tablas_churn["saldos"]
    vig = _matriz(saldos.assign(_u=1), "_u", afil_pos, per_pos, relleno=0.0) > 0
    fondo_cod = saldos.assign(_f=saldos["fondo"].map({f: i for i, f in enumerate(FONDOS)}))
    fondo_mat = _matriz(fondo_cod, "_f", afil_pos, per_pos, dtype=np.int8, relleno=-1)

    # ---- eventos base (churn): reclamo sin resolver, contacto ejecutivo resuelto
    rec = tablas_churn["reclamos"]
    if len(rec):
        m_rec_sr = rec[rec["estado"].isin(["ABIERTO", "RESUELTO_DESFAVORABLE"])].assign(_u=1)
    else:
        m_rec_sr = rec
    flag_reclamo_sr = _matriz(m_rec_sr, "_u", afil_pos, per_pos, agregar=True) > 0

    con = tablas_churn["contactos"]
    if len(con):
        m_ep_ok = con[(con["canal"] == "EJECUTIVO_PROPIO") & (con["resultado"].isin(["CONTACTADO", "INTERESADO"]))
                     ].assign(_u=1)
    else:
        m_ep_ok = con
    flag_ejecutivo_ok = _matriz(m_ep_ok, "_u", afil_pos, per_pos, agregar=True) > 0

    # ---- licencia mal pagada (proxy): dias_licencia >= umbral en el mes (panel, SIN desfase: es
    # para la verdad latente, no para features observables)
    pan_c = panel[panel["periodo"].isin(per_pos)]
    dias_lic = _matriz(pan_c, "dias_licencia", afil_pos, per_pos, dtype=np.float32)
    flag_licencia_mal_pagada = dias_lic >= par.umbral_licencia_mal_pagada_dias

    # ---- empleador: DNP y rezago de recepcion (agregado_empleador, sin desfase: verdad latente)
    emp_id_mat = _matriz(pan_c.assign(_e=pan_c["empleador_id"].fillna(0).astype("int64")), "_e",
                         afil_pos, per_pos, dtype=np.int64, relleno=0)
    ae = agregado_empleador
    emp_ids_u = np.union1d(ae["empleador_id"].to_numpy(), [0])
    emp_row = {int(e): i for i, e in enumerate(emp_ids_u)}
    n_emp = len(emp_ids_u)
    dnp_emp = np.zeros((n_emp, n_t), dtype=np.float32)
    lag_emp = np.zeros((n_emp, n_t), dtype=np.float32)
    tasa_rez_emp = np.zeros((n_emp, n_t), dtype=np.float32)
    ae_p = ae[ae["periodo"].isin(per_pos)]
    if len(ae_p):
        ie = ae_p["empleador_id"].map(emp_row).to_numpy()
        ip = ae_p["periodo"].map(per_pos).to_numpy()
        dnp_emp[ie, ip] = ae_p["frac_dnp"].to_numpy(dtype=np.float32)
        lag_emp[ie, ip] = ae_p["lag_recepcion_medio"].to_numpy(dtype=np.float32)
    if len(rezagos):
        rez_c = rezagos.assign(_u=1)
        tot = ae_p[["empleador_id", "periodo", "n_lineas"]] if len(ae_p) else ae_p
        rez_cnt = rez_c.groupby(["empleador_id", "periodo"])["_u"].sum().reset_index()
        rc = rez_cnt.merge(tot, on=["empleador_id", "periodo"], how="left") if len(tot) else rez_cnt.assign(n_lineas=np.nan)
        rc["tasa"] = rc["_u"] / rc["n_lineas"].replace(0, np.nan)
        rc = rc.dropna(subset=["tasa"])
        if len(rc):
            ie = rc["empleador_id"].map(emp_row).to_numpy()
            ip = rc["periodo"].map(per_pos).to_numpy()
            valido = ~pd.isna(ie) & ~pd.isna(ip)
            tasa_rez_emp[ie[valido].astype(int), ip[valido].astype(int)] = rc.loc[valido, "tasa"].to_numpy(np.float32)

    emp_row_af = np.array([emp_row.get(int(e), emp_row[0]) for e in emp_id_mat.ravel()]).reshape(emp_id_mat.shape)
    flag_dnp_empleador = dnp_emp[emp_row_af, np.arange(n_t)[None, :]] > 0
    # el ciclo normal de declaracion ya implica ~1 mes de desfase entre `periodo` y `periodo_recepcion`
    # ("oportuno" en `_aplicar_recepcion`); ">30 dias" de atraso RESPECTO de ese ciclo normal es
    # lag_recepcion_medio >= 2 (equivale a "tardio" o al "rezago" verdadero de `TAMANO_PAGO`), no >= 1.
    flag_rezago_mensual = lag_emp[emp_row_af, np.arange(n_t)[None, :]] >= 2.0

    # ---- mercado: rentabilidad relativa y brecha de comision (contemporaneas, sin decaimiento)
    merc = tablas_churn["mercado"]
    merc_propia = merc[merc["es_propia"] == 1].set_index("periodo")
    com_propia = merc_propia["comision"].reindex(periodos).to_numpy(dtype=np.float32)
    com_min = merc.groupby("periodo")["comision"].min().reindex(periodos).to_numpy(dtype=np.float32)
    brecha_com = com_propia - com_min
    rent_propia = np.stack([merc_propia[f"rent_12m_{f}"].reindex(periodos).to_numpy(dtype=np.float32)
                            for f in FONDOS])                                            # (5, n_t)
    rent_prom = np.stack([merc.groupby("periodo")[f"rent_12m_{f}"].mean().reindex(periodos)
                          .to_numpy(dtype=np.float32) for f in FONDOS])
    brecha_rent_fondo = rent_propia - rent_prom                                          # (5, n_t)
    fondo_seguro = np.clip(fondo_mat, 0, len(FONDOS) - 1)
    brecha_rent_af = brecha_rent_fondo[fondo_seguro, np.arange(n_t)[None, :]]
    flag_rent_positiva = (brecha_rent_af > 0) & (fondo_mat >= 0)
    flag_comision_desfavorable = np.broadcast_to(brecha_com[None, :] > par.umbral_brecha_comision_pp, (n, n_t))

    # ---- satisfaccion latente
    tipo_af = afiliados["tipo_afiliado"].to_numpy()
    base_seg = np.array([par.base_segmento.get(t, 0.0) for t in tipo_af])
    edad_mat = _matriz(pan_c, "edad", afil_pos, per_pos, dtype=np.float32)
    edad0 = afiliados["edad_inicio"].to_numpy().astype(np.float32)
    for t in range(n_t):
        edad_mat[:, t] = np.where(edad_mat[:, t] > 0, edad_mat[:, t], edad0 + t // 12)
    base_edad = np.where(edad_mat < 30, par.base_joven, np.where(edad_mat >= 60, par.base_mayor, 0.0))

    rng = np.random.default_rng(seed + 501)
    persistente = rng.normal(0.0, par.sigma_persistente, n)
    ruido_mes = rng.normal(0.0, par.sigma_mes, (n, n_t))

    lat = (par.base_nivel + base_seg[:, None] + base_edad + persistente[:, None]
          + _efecto_decaido(flag_reclamo_sr, par.ef_reclamo_sin_resolver, par.vida_media_evento, par.tope_meses_evento)
          + _efecto_decaido(flag_rezago_mensual, par.ef_rezago_mensual, par.vida_media_evento, par.tope_meses_evento)
          + _efecto_decaido(flag_licencia_mal_pagada, par.ef_licencia_mal_pagada, par.vida_media_evento, par.tope_meses_evento)
          + _efecto_decaido(flag_dnp_empleador, par.ef_dnp_empleador, par.vida_media_dnp, par.tope_meses_dnp)
          + _efecto_decaido(flag_ejecutivo_ok, par.ef_contacto_ejecutivo_resuelto, par.vida_media_evento, par.tope_meses_evento)
          + par.ef_rentabilidad_relativa_positiva * flag_rent_positiva
          + par.ef_brecha_comision_desfavorable * flag_comision_desfavorable
          + ruido_mes)
    lat = np.clip(lat, 0.0, 10.0)

    # ---- encuesta: invitacion (transaccional + relacional) y respuesta (con sesgo de no respuesta)
    evento_reciente = flag_reclamo_sr | flag_ejecutivo_ok | (_matriz(
        con.assign(_u=1) if len(con) else con, "_u", afil_pos, per_pos, agregar=True) > 0)
    rng2 = np.random.default_rng(seed + 502)
    z_invita = par.peso_evento_reciente_invitacion * evento_reciente.astype(float)
    p_invita = np.zeros((n, n_t))
    if vig.any():
        p_invita[vig] = ajustar_prior(_sigmoide(z_invita[vig]), par.p_invitacion_objetivo)
    invitado = vig & (rng2.random((n, n_t)) < p_invita)

    logins_mat = _matriz(tablas_churn["uso_digital"], "logins_app", afil_pos, per_pos) + \
        _matriz(tablas_churn["uso_digital"], "logins_web", afil_pos, per_pos)
    uso_bajo = logins_mat <= 0
    joven = edad_mat < 30
    z_resp = (par.peso_extremo_respuesta * _extremidad(lat)
             + par.peso_joven_respuesta * joven
             + par.peso_uso_digital_bajo_respuesta * uso_bajo)
    lat_p_responde = np.zeros((n, n_t))
    if invitado.any():
        lat_p_responde[invitado] = ajustar_prior(_sigmoide(z_resp[invitado]), par.p_respuesta_objetivo)
    respondio = invitado & (rng2.random((n, n_t)) < lat_p_responde)

    ruido_resp = rng2.normal(0.0, par.sigma_respuesta, (n, n_t))
    respuesta_cont = np.clip(np.round(lat + ruido_resp), 0, 10)

    ii, jj = np.nonzero(invitado)
    per_arr = np.array(periodos)
    tipo_invitacion = np.where(evento_reciente[ii, jj], "TRANSACCIONAL", "RELACIONAL")
    resp_ok = respondio[ii, jj]
    resp_val = np.where(resp_ok, respuesta_cont[ii, jj], np.nan)
    categoria = np.where(~resp_ok, None,
                         np.where(resp_val >= PROMOTOR_DESDE, "PROMOTOR",
                                  np.where(resp_val <= DETRACTOR_HASTA, "DETRACTOR", "PASIVO")))
    encuesta = pd.DataFrame({
        "afiliado_id": af_ids[ii],
        "periodo": per_arr[jj],
        "tipo_invitacion": tipo_invitacion,
        "respondio": resp_ok.astype(np.int8),
        "respuesta": resp_val,
        "categoria": categoria,
    }).sort_values(["periodo", "afiliado_id"]).reset_index(drop=True)

    ii2, jj2 = np.nonzero(vig)
    satisfaccion_latente = pd.DataFrame({
        "afiliado_id": af_ids[ii2],
        "periodo": per_arr[jj2],
        "lat_satisfaccion": lat[ii2, jj2].astype(np.float32),
        "lat_p_responde_si_invitado": lat_p_responde[ii2, jj2].astype(np.float32),
    })

    return {
        "afiliados": afiliados,
        "empleadores": tablas_churn["empleadores"],
        "panel_mensual": panel,
        "macro": tablas_churn["macro"],
        "mercado": tablas_churn["mercado"],
        "saldos": saldos,
        "cambios_fondo": tablas_churn["cambios_fondo"],
        "traspasos": tablas_churn["traspasos"],
        "contactos": tablas_churn["contactos"],
        "reclamos": tablas_churn["reclamos"],
        "uso_digital": tablas_churn["uso_digital"],
        "apvc_empleador": tablas_apv["apvc_empleador"],
        "campanas_apv": tablas_apv["campanas"],
        "aportes_voluntarios": tablas_apv["aportes_voluntarios"],
        "saldos_apv": tablas_apv["saldos"],
        "tributario": tablas_apv["tributario"],
        "planilla": planilla,
        "rezagos": rezagos,
        "agregado_empleador": agregado_empleador,
        "encuesta": encuesta,
        "satisfaccion_latente": satisfaccion_latente,
    }


def resumen_simulacion(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Chequeos de plausibilidad: tasas de invitacion/respuesta y NPS verdadero de la cartera."""
    lat = tablas["satisfaccion_latente"]
    enc = tablas["encuesta"]
    n_vig = len(lat)
    n_inv = len(enc)
    n_resp = int(enc["respondio"].sum())
    cat_true = pd.cut(lat["lat_satisfaccion"].round().clip(0, 10),
                      bins=[-1, DETRACTOR_HASTA, 8, 11], labels=["DETRACTOR", "PASIVO", "PROMOTOR"])
    nps_true = float((cat_true == "PROMOTOR").mean() * 100 - (cat_true == "DETRACTOR").mean() * 100)
    resp = enc[enc["respondio"] == 1]
    nps_naive = float((resp["categoria"] == "PROMOTOR").mean() * 100
                      - (resp["categoria"] == "DETRACTOR").mean() * 100) if len(resp) else float("nan")
    filas = [
        {"indicador": "afiliado_meses_vigentes", "valor": float(n_vig), "detalle": ""},
        {"indicador": "frac_invitada_mensual", "valor": float(n_inv / max(n_vig, 1)), "detalle": "brief: ~8 %"},
        {"indicador": "frac_respuesta_entre_invitados", "valor": float(n_resp / max(n_inv, 1)), "detalle": "brief: ~25 %"},
        {"indicador": "nps_verdadero_cartera", "valor": nps_true, "detalle": "lat_satisfaccion redondeada, toda la cartera"},
        {"indicador": "nps_ingenuo_respondientes", "valor": nps_naive, "detalle": "sesgo de no respuesta esperado"},
        {"indicador": "frac_transaccional", "valor": float((enc["tipo_invitacion"] == "TRANSACCIONAL").mean())
         if len(enc) else float("nan"), "detalle": ""},
        {"indicador": "satisfaccion_media_cartera", "valor": float(lat["lat_satisfaccion"].mean()), "detalle": ""},
    ]
    return pd.DataFrame(filas)
