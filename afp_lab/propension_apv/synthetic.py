"""Tablas sinteticas del proyecto 03 (ahorro previsional voluntario).

Parte del universo comun (`afp_lab.common.synth_base.generar_universo`: afiliados, empleadores,
panel laboral mensual y macro) y le agrega las tablas propias del problema, todas coherentes con la
historia laboral simulada (un cesante no aporta, el descuento por planilla existe solo si hay
empleador, el APV colectivo depende del empleador, etc.):

| Tabla | Grano | Contenido |
|---|---|---|
| `apvc_empleador` | empleador x mes | si el empleador ofrece APV colectivo (APVC) y el % de match |
| `campanas` | evento | afiliado, mes, canal, respuesta (ABRIO / IGNORO) |
| `aportes_voluntarios` | evento | afiliado, mes, tipo, monto, canal, periodicidad |
| `saldos` | afiliado x mes | saldo obligatorio (10 % de la RIM capitalizada) y saldo voluntario |
| `tributario` | afiliado x anio | renta anual y tramo del Impuesto Global Complementario (IGC) |

Todos los parametros numericos estan explicitos en las constantes de modulo y en
`ParametrosAPV`; el README documenta por que cada uno es razonable para Chile.

La ventana por defecto parte en 201901 aunque el laboratorio trabaja desde 202201: los primeros
24 meses son *burn-in* (para que el primer corte tenga 24 meses de historial de aportes) y los
63 cortes resultantes permiten el split con gap entre cada par de particiones que pide
CONVENCIONES §3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from afp_lab.common.synth_base import generar_universo
from afp_lab.rim_nowcasting.calendario import periodo_to_idx, rango_periodos

# --------------------------------------------------------------------------- calendario
PERIODO_INICIO = 201901          # incluye 24 meses de burn-in para el historial de aportes
PERIODO_FIN = 202609             # "hoy" del laboratorio (igual que el proyecto 01)
# Desfases de disponibilidad de cada fuente (CONVENCIONES §3, convencion de fechas de la ronda 1).
# `periodo_corte` es el MES DE LA DECISION; cada fuente se lee con su propio rezago.
MESES_DESFASE = 2                # panel laboral / RIM: la cotizacion del mes m se conoce en m+2
MESES_DESFASE_LICENCIA = 1       # licencias medicas (feed COMPIN/ISAPRE) y eventos AFC
MESES_DESFASE_OPERACIONAL = 0    # aportes voluntarios, campanas, saldos: sistemas propios
HORIZONTE = 6                    # ventana del target: corte+1 .. corte+6
# Rezago con el que se trunca cada tabla en los tests de leakage
DESFASE_POR_TABLA = {"panel_mensual": MESES_DESFASE, "apvc_empleador": MESES_DESFASE,
                     "campanas": MESES_DESFASE_OPERACIONAL,
                     "aportes_voluntarios": MESES_DESFASE_OPERACIONAL,
                     "saldos": MESES_DESFASE_OPERACIONAL}

# --------------------------------------------------------------------------- dominios
TIPOS_APORTE = ("APV_REGIMEN_A", "APV_REGIMEN_B", "COTIZACION_VOLUNTARIA", "DEPOSITO_CONVENIDO")
CANALES_APORTE = ("DESCUENTO_PLANILLA", "TRANSFERENCIA", "APP")
PERIODICIDADES = ("MENSUAL", "ESPORADICO")
CANALES_CAMPANA = ("EMAIL", "APP", "EJECUTIVO", "ASESOR")
RESPUESTAS_CAMPANA = ("ABRIO", "IGNORO")

ESTADOS = {"EMPLEADO": 1, "LICENCIA": 2, "INDEPENDIENTE": 3, "VOLUNTARIO": 4,
           "CESANTE": 5, "RETIRADO": 6}
ESTADOS_INV = {v: k for k, v in ESTADOS.items()}
ESTADOS_INV[0] = "SIN_DATO"
# estados en que el afiliado puede hacer un aporte voluntario (un cesante no hace APV)
ESTADOS_CON_INGRESO = (1, 2, 3, 4)
# estados con relacion laboral dependiente (unico caso con descuento por planilla)
ESTADOS_DEPENDIENTES = (1, 2)

# --------------------------------------------------------------------------- tributario
# UTM anual promedio aproximada (SII). UTA = 12 x UTM.
UTM_ANUAL = {2019: 49_000, 2020: 50_000, 2021: 52_000, 2022: 57_000, 2023: 63_000,
             2024: 65_500, 2025: 68_000, 2026: 70_500}
# Tramos del IGC (Ley de la Renta, art. 52): limite superior en UTA y tasa marginal.
TRAMOS_IGC_UTA = ((13.5, 0.000), (30.0, 0.040), (50.0, 0.080), (70.0, 0.135),
                  (90.0, 0.230), (120.0, 0.304), (310.0, 0.350), (np.inf, 0.400))

# --------------------------------------------------------------------------- APVC
# Probabilidad de que un empleador ofrezca APV colectivo, por tamano.
# (En Chile el APVC es marginal y muy concentrado en empresas grandes y formales.)
P_APVC_TAMANO = {"MICRO": 0.02, "PEQUENA": 0.08, "MEDIANA": 0.28, "GRANDE": 0.55}
# Multiplicador por rubro: banca/mineria/sector publico ofrecen mucho mas; agro y casa particular casi nada.
MULT_APVC_RUBRO = {"K_FINANZAS": 1.9, "B_MINERIA": 1.8, "O_ADMIN_PUBLICA": 1.5, "Q_SALUD": 1.1,
                   "A_AGRICULTURA": 0.35, "T_CASA_PARTICULAR": 0.20, "I_TURISMO": 0.5,
                   "F_CONSTRUCCION": 0.6}
P_APVC_ADOPCION_EN_VENTANA = 0.35   # fraccion de los que ofrecen APVC que lo lanzan durante la ventana
P_APVC_SIN_MATCH = 0.30             # ofrecen el vehiculo pero sin aporte del empleador
MATCH_APVC_MEDIA = 0.015            # match medio como fraccion de la renta (1,5 %)
MATCH_APVC_SIGMA = 0.45             # dispersion log-normal del match

# --------------------------------------------------------------------------- campanas
P_CAMPANA_BASE = 0.055              # cobertura mensual media de campanas sobre afiliados vigentes
CAMPANA_PESO_RENTA = 0.55           # la AFP focaliza campanas en rentas altas
CAMPANA_PESO_APVC = 0.45            # y en empleadores con APVC
P_CANAL_CAMPANA = {"EMAIL": 0.45, "APP": 0.30, "EJECUTIVO": 0.15, "ASESOR": 0.10}
P_APERTURA_CANAL = {"EMAIL": 0.22, "APP": 0.34, "EJECUTIVO": 0.62, "ASESOR": 0.55}
APERTURA_PESO_AFINIDAD = 0.45       # quien tiene mas afinidad al ahorro abre mas (senal util, no leakage)

# --------------------------------------------------------------------------- saldos
RENT_MENSUAL_OBLIGATORIO = 0.0045   # ~5,5 % anual nominal (fondo C)
RENT_MENSUAL_VOLUNTARIO = 0.0050    # ~6,2 % anual nominal (mezcla A/B)
TASA_COTIZACION_OBLIGATORIA = 0.10  # 10 % de la RIM a la cuenta de capitalizacion individual
SALDO_INICIAL_FACTOR = 0.55         # capitalizacion implicita del saldo previo a la ventana
SALDO_INICIAL_SIGMA = 0.55
# La cartera de APV no parte en cero: al inicio de la ventana ya hay aportantes con saldo
# acumulado y ex-aportantes que dejaron de aportar (si no, el saldo voluntario tendria una
# tendencia artificial creciente que rompe la comparabilidad entre cortes).
P_EX_APORTANTE_INICIAL = 0.10       # afiliados que aportaron antes de la ventana y ya no aportan
MESES_APORTE_PREVIOS = (6, 72)      # antiguedad del habito de ahorro al inicio de la ventana
SALDO_VOL_INICIAL_SIGMA = 0.40

# --------------------------------------------------------------------------- montos
TASA_AHORRO_MENSUAL = 0.035         # mediana del aporte mensual como fraccion de la RIM
TASA_AHORRO_MENSUAL_SIGMA = 0.40
FACTOR_ESPORADICO = {3: 0.30, 4: 0.25, 12: 0.22}   # marzo-abril (bonos) y diciembre
FACTOR_ESPORADICO_OTRO = 0.15
FACTOR_ESPORADICO_SIGMA = 0.55
ELASTICIDAD_MONTO_TRAMO = 0.10      # +10 % de monto por tramo del IGC (beneficio tributario)
FACTOR_MONTO_REGIMEN_B = 1.25       # el regimen B (rebaja de base imponible) mueve montos mayores
TOPE_APORTE_UF = 50.0               # tope legal mensual del APV (50 UF)
APORTE_MINIMO = 15_000.0

# --------------------------------------------------------------------------- estacionalidad
ESTACIONALIDAD_ESPORADICO = {1: 0.20, 3: 1.00, 4: 0.70, 12: 0.85}  # resto 0
RUBROS_BONO_MARZO = ("K_FINANZAS", "B_MINERIA")


@dataclass(frozen=True)
class ParametrosAPV:
    """Coeficientes del *score* latente de propension (escala logit).

    El score entra en dos hazards mensuales: abrir un aporte **mensual** por planilla
    (`intercepto_mensual`) y hacer un aporte **esporadico** (`intercepto_esporadico`).
    """

    # --- drivers observables -------------------------------------------------
    b_renta: float = 0.90          # elasticidad a log(RIM / IMM); driver mas fuerte
    b_tramo_igc: float = 0.30      # beneficio tributario del regimen B (tramo 0..7)
    b_densidad: float = 1.20       # estabilidad: densidad de cotizacion 12 m (centrada en 0,85)
    b_antiguedad: float = 0.012    # por mes con el mismo empleador (tope 60 meses)
    b_edad_35_55: float = 0.40     # etapa de acumulacion
    b_edad_menor_30: float = -0.25
    b_edad_mayor_60: float = -0.70
    b_edu_universitaria: float = 0.45
    b_edu_tecnica: float = 0.20
    b_edu_basica: float = -0.30
    b_apvc: float = 1.10           # empleador con APVC: efecto grande
    b_match: float = 0.25          # por punto porcentual de match
    b_campana_3m: float = 0.20     # campana reciente: efecto pequeno
    b_campana_asesor: float = 0.18 # extra si el contacto fue de ejecutivo/asesor
    b_historial: float = 0.70      # hizo algun aporte en los 24 meses previos
    b_bono_marzo: float = 0.90     # marzo-abril en finanzas/mineria (bonos)
    b_aguinaldo: float = 0.45      # meses con aguinaldo imponible
    b_cerca_minimo: float = -0.80  # renta a menos de 1,15 IMM
    b_licencia_larga: float = -0.60
    b_independiente: float = -0.35 # el independiente ahorra menos via AFP
    # --- ruido no explicable -------------------------------------------------
    b_afinidad_latente: float = 0.90   # N(0,1) por afiliado: mantiene el AUC en 0,75-0,88
    # --- interceptos (calibran las tasas objetivo del brief) -----------------
    intercepto_esporadico: float = -7.95   # ~12-18 % de cotizantes con algun aporte en el anio
    intercepto_mensual: float = -9.35      # ~5-8 % con aporte mensual
    intercepto_stock_inicial: float = -5.75  # stock de aportantes mensuales al inicio de la ventana
    p_salida_mensual: float = 0.030        # abandono mensual del aporte por planilla
    p_salida_cambio_empleador: float = 0.45


# =========================================================================== utilidades
def indexar_periodos(periodos: list[int]) -> dict[int, int]:
    """yyyymm -> posicion en la ventana simulada."""
    return {p: i for i, p in enumerate(periodos)}


def matriz_densa(df: pd.DataFrame, col: str, n_af: int, n_per: int,
                 idx_af: np.ndarray, idx_per: np.ndarray, dtype=np.float32, default=0):
    """Matriz densa (afiliado x periodo) a partir de una tabla larga.

    `idx_af` e `idx_per` son las posiciones ya calculadas de cada fila de `df`.
    """
    m = np.full((n_af, n_per), default, dtype=dtype)
    m[idx_af, idx_per] = df[col].to_numpy()
    return m


def tramo_igc(renta_anual: np.ndarray, anio: int) -> np.ndarray:
    """Tramo del Impuesto Global Complementario (0 = exento ... 7 = 40 %) segun la renta anual."""
    uta = 12.0 * UTM_ANUAL.get(anio, UTM_ANUAL[max(UTM_ANUAL)])
    limites = np.array([t[0] for t in TRAMOS_IGC_UTA[:-1]]) * uta
    return np.searchsorted(limites, np.asarray(renta_anual, float), side="right").astype(np.int8)


def tasa_marginal_igc(tramo: np.ndarray) -> np.ndarray:
    tasas = np.array([t[1] for t in TRAMOS_IGC_UTA])
    return tasas[np.asarray(tramo, int)]


def anio_tributario_disponible(periodo_corte: int) -> int:
    """Ultimo ano tributario conocido en el corte.

    La Operacion Renta del ano *Y* se declara en abril de *Y+1*: en un corte de abril o
    posterior se conoce *Y-1*; antes de abril, solo *Y-2*.
    """
    anio, mes = periodo_corte // 100, periodo_corte % 100
    return anio - 1 if mes >= 4 else anio - 2


def _sigmoide(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# probabilidad de elegir el regimen B (rebaja de la base imponible): crece con el tramo del IGC,
# porque el beneficio del regimen B es la tasa marginal y el del regimen A es una bonificacion fija.
P_REGIMEN_B_BASE = 0.12
P_REGIMEN_B_POR_TRAMO = 0.20
P_COTIZACION_VOLUNTARIA = 0.18   # entre quienes no eligen regimen B


def _elegir_tipo(tramo: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Indice en TIPOS_APORTE (0 = APV A, 1 = APV B, 2 = cotizacion voluntaria)."""
    p_b = np.clip(P_REGIMEN_B_BASE + P_REGIMEN_B_POR_TRAMO * np.asarray(tramo, float), 0.0, 0.85)
    return np.where(u < p_b, 1, np.where(u < p_b + (1 - p_b) * (1 - P_COTIZACION_VOLUNTARIA), 0, 2))


# =========================================================================== APVC
def _generar_apvc(empleadores: pd.DataFrame, periodos: list[int], rng: np.random.Generator
                  ) -> pd.DataFrame:
    """APV colectivo por empleador y mes (con adopcion escalonada durante la ventana)."""
    n = len(empleadores)
    p = np.array([P_APVC_TAMANO[t] for t in empleadores["tamano_empleador"]])
    p = p * np.array([MULT_APVC_RUBRO.get(r, 1.0) for r in empleadores["rubro"]])
    ofrece = rng.random(n) < np.clip(p, 0.0, 0.90)
    # 35 % de los que ofrecen lo lanzan en algun mes de la ventana; el resto ya lo tenia
    adopta_en_ventana = ofrece & (rng.random(n) < P_APVC_ADOPCION_EN_VENTANA)
    mes_inicio = np.where(adopta_en_ventana, rng.integers(1, len(periodos), n), 0)
    match = rng.lognormal(np.log(MATCH_APVC_MEDIA), MATCH_APVC_SIGMA, n)
    match = np.where(rng.random(n) < P_APVC_SIN_MATCH, 0.0, np.clip(match, 0.002, 0.06))

    per = np.asarray(periodos)
    activo = ofrece[:, None] & (np.arange(len(periodos))[None, :] >= mes_inicio[:, None])
    filas = pd.DataFrame({
        "empleador_id": np.repeat(empleadores["empleador_id"].to_numpy(), len(periodos)),
        "periodo": np.tile(per, n),
        "ofrece_apvc": activo.ravel().astype(np.int8),
        "pct_match": np.where(activo, match[:, None], 0.0).ravel().round(4),
    })
    return filas


# =========================================================================== generador principal
def generar_tablas(n_afiliados: int = 5_000, periodo_inicio: int = PERIODO_INICIO,
                   periodo_fin: int = PERIODO_FIN, seed: int = 42,
                   params: ParametrosAPV | None = None) -> dict[str, pd.DataFrame]:
    """Universo comun + tablas de ahorro voluntario. Determinista dado `seed`.

    Devuelve un diccionario con `afiliados`, `empleadores`, `panel_mensual`, `macro`,
    `apvc_empleador`, `campanas`, `aportes_voluntarios`, `saldos` y `tributario`.
    Las columnas latentes (`lat_*`, `p_*` de empleadores) se usan solo aqui y **no** se
    devuelven: el modelo no puede verlas.
    """
    params = params or ParametrosAPV()
    rng = np.random.default_rng(seed + 103)

    universo = generar_universo(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                                periodo_fin=periodo_fin, seed=seed)
    afiliados = universo["afiliados"]
    afil_lat = universo["afiliados_latente"]
    empleadores = universo["empleadores"]
    panel = universo["panel_mensual"]
    macro = universo["macro"]

    periodos = rango_periodos(periodo_inicio, periodo_fin)
    n_per, n_af = len(periodos), len(afiliados)
    pos_per = indexar_periodos(periodos)

    # ------------------------------------------------------------ matrices densas del panel
    idx_af = panel["afiliado_id"].to_numpy() - 1
    idx_per = panel["periodo"].map(pos_per).to_numpy()
    rim = (matriz_densa(panel.assign(_r=panel["rim_emp1"] + panel["rim_emp2"]), "_r",
                        n_af, n_per, idx_af, idx_per))
    estado = matriz_densa(panel.assign(_e=panel["estado"].map(ESTADOS)), "_e",
                          n_af, n_per, idx_af, idx_per, dtype=np.int8)
    antig = matriz_densa(panel, "meses_antiguedad", n_af, n_per, idx_af, idx_per, dtype=np.int16)
    edad = matriz_densa(panel, "edad", n_af, n_per, idx_af, idx_per, dtype=np.int16)
    dias_lic = matriz_densa(panel, "dias_licencia", n_af, n_per, idx_af, idx_per, dtype=np.int16)
    aguinaldo = matriz_densa(panel, "aguinaldo", n_af, n_per, idx_af, idx_per)
    emp_idx = matriz_densa(panel.assign(_e=panel["empleador_id"].fillna(0).astype("int64")), "_e",
                           n_af, n_per, idx_af, idx_per, dtype=np.int64, default=0) - 1

    cotiza = (rim > 0).astype(np.float32)
    csum_rim = np.concatenate([np.zeros((n_af, 1), np.float64), np.cumsum(rim, axis=1, dtype=np.float64)], axis=1)
    csum_cot = np.concatenate([np.zeros((n_af, 1), np.float64), np.cumsum(cotiza, axis=1, dtype=np.float64)], axis=1)

    # ------------------------------------------------------------ APVC por empleador
    apvc = _generar_apvc(empleadores, periodos, rng)
    n_emp = len(empleadores)
    apvc_ofrece = apvc["ofrece_apvc"].to_numpy().reshape(n_emp, n_per).astype(np.int8)
    apvc_match = apvc["pct_match"].to_numpy().reshape(n_emp, n_per).astype(np.float32)

    # ------------------------------------------------------------ atributos del afiliado
    rubro_emp = empleadores["rubro"].to_numpy()
    bono_marzo_emp = np.isin(rubro_emp, RUBROS_BONO_MARZO)
    edu = afiliados["nivel_educacional"].to_numpy()
    efecto_edu = np.select([edu == "UNIVERSITARIA", edu == "TECNICA", edu == "BASICA"],
                           [params.b_edu_universitaria, params.b_edu_tecnica, params.b_edu_basica], 0.0)
    es_independiente = (afiliados["tipo_afiliado"].to_numpy() == "INDEPENDIENTE").astype(float)
    # afinidad latente al ahorro: ruido no explicable por ninguna feature
    lat_afinidad = rng.normal(0.0, 1.0, n_af)
    # heterogeneidad del monto (tambien latente)
    lat_tasa = rng.lognormal(0.0, TASA_AHORRO_MENSUAL_SIGMA, n_af)
    # (afil_lat solo se usa para confirmar el orden de los afiliados; no entra al score)
    assert (afil_lat["afiliado_id"].to_numpy() == afiliados["afiliado_id"].to_numpy()).all()

    imm_v = macro.set_index("periodo").loc[periodos, "imm"].to_numpy(float)
    uf_v = macro.set_index("periodo").loc[periodos, "uf"].to_numpy(float)
    mes_v = np.array([p % 100 for p in periodos])
    anio_v = np.array([p // 100 for p in periodos])

    # ------------------------------------------------------------ estado del proceso
    aportante_mensual = np.zeros(n_af, dtype=bool)
    tasa_mensual = TASA_AHORRO_MENSUAL * lat_tasa
    tipo_mensual = np.zeros(n_af, dtype=np.int8)      # indice en TIPOS_APORTE
    canal_mensual = np.zeros(n_af, dtype=np.int8)
    ult_aporte = np.full(n_af, -999, dtype=np.int32)
    ult_contacto = np.full(n_af, -999, dtype=np.int32)
    ult_contacto_asesor = np.full(n_af, -999, dtype=np.int32)
    emp_previo = np.full(n_af, -1, dtype=np.int64)

    filas_aportes: list[pd.DataFrame] = []
    filas_campanas: list[pd.DataFrame] = []
    aportes_mat = np.zeros((n_af, n_per), dtype=np.float32)

    for j in range(n_per):
        imm, uf, mes, anio = imm_v[j], uf_v[j], mes_v[j], anio_v[j]
        est = estado[:, j]
        presente = est > 0
        con_ingreso = np.isin(est, ESTADOS_CON_INGRESO)
        dependiente = np.isin(est, ESTADOS_DEPENDIENTES)

        ini = max(0, j - 11)
        suma_rim_12 = csum_rim[:, j + 1] - csum_rim[:, ini]
        n_cot_12 = csum_cot[:, j + 1] - csum_cot[:, ini]
        rim_med_12 = suma_rim_12 / np.maximum(n_cot_12, 1.0)
        densidad_12 = n_cot_12 / min(12, j + 1)
        rim_ref = np.where(rim[:, j] > 0, rim[:, j], rim_med_12)

        tramo = tramo_igc(suma_rim_12, int(anio))

        e = emp_idx[:, j]
        tiene_emp = e >= 0
        ofrece = np.where(tiene_emp, apvc_ofrece[np.clip(e, 0, None), j], 0).astype(float)
        match = np.where(tiene_emp, apvc_match[np.clip(e, 0, None), j], 0.0)
        bono_marzo = np.where(tiene_emp, bono_marzo_emp[np.clip(e, 0, None)], False) & (mes in (3, 4))

        # ---------------------------------------------------- campanas del mes
        peso = _sigmoide(-2.9 + CAMPANA_PESO_RENTA * np.log(np.maximum(rim_ref, 1.0) / imm)
                         + CAMPANA_PESO_APVC * ofrece)
        peso_medio = peso[presente].mean() if presente.any() else 1.0
        p_contacto = np.clip(peso / max(float(peso_medio), 1e-9) * P_CAMPANA_BASE, 0.0, 0.35)
        objetivo = presente & (rng.random(n_af) < p_contacto)
        n_obj = int(objetivo.sum())
        if n_obj:
            ids_obj = np.where(objetivo)[0]
            canal = rng.choice(list(P_CANAL_CAMPANA), size=n_obj, p=list(P_CANAL_CAMPANA.values()))
            p_abre = np.array([P_APERTURA_CANAL[c] for c in canal])
            p_abre = _sigmoide(np.log(p_abre / (1 - p_abre)) + APERTURA_PESO_AFINIDAD * lat_afinidad[ids_obj])
            abrio = rng.random(n_obj) < p_abre
            filas_campanas.append(pd.DataFrame({
                "afiliado_id": ids_obj + 1,
                "periodo": periodos[j],
                "canal": canal,
                "respuesta": np.where(abrio, "ABRIO", "IGNORO"),
            }))
            ult_contacto[ids_obj] = j
            asesor = np.isin(canal, ["EJECUTIVO", "ASESOR"])
            ult_contacto_asesor[ids_obj[asesor]] = j

        # ---------------------------------------------------- score de propension
        edad_j = edad[:, j].astype(float)
        efecto_edad = np.where((edad_j >= 35) & (edad_j <= 55), params.b_edad_35_55, 0.0)
        efecto_edad += np.where(edad_j < 30, params.b_edad_menor_30, 0.0)
        efecto_edad += np.where(edad_j > 60, params.b_edad_mayor_60, 0.0)
        lic_larga = (dias_lic[:, j] >= 20) | (dias_lic[:, max(0, j - 1)] >= 20)

        z = (params.b_renta * np.log(np.maximum(rim_ref, 1.0) / imm)
             + params.b_tramo_igc * tramo
             + params.b_densidad * (densidad_12 - 0.85)
             + params.b_antiguedad * np.minimum(antig[:, j], 60)
             + efecto_edad
             + efecto_edu
             + params.b_apvc * ofrece
             + params.b_match * (match * 100.0)
             + params.b_campana_3m * ((j - ult_contacto) <= 2)
             + params.b_campana_asesor * ((j - ult_contacto_asesor) <= 5)
             + params.b_historial * ((j - ult_aporte) <= 24)
             + params.b_bono_marzo * bono_marzo
             + params.b_aguinaldo * (aguinaldo[:, j] > 0)
             + params.b_cerca_minimo * (rim_ref < 1.15 * imm)
             + params.b_licencia_larga * lic_larga
             + params.b_independiente * es_independiente
             + params.b_afinidad_latente * lat_afinidad)

        # ---------------------------------------------------- stock inicial de aportantes mensuales
        if j == 0:
            aportante_mensual = con_ingreso & dependiente & (
                rng.random(n_af) < _sigmoide(z + params.intercepto_stock_inicial))
            tipo_mensual = _elegir_tipo(tramo, rng.random(n_af)).astype(np.int8)
            canal_mensual = np.zeros(n_af, dtype=np.int8)
            ex_aportante = (~aportante_mensual) & (rng.random(n_af) < P_EX_APORTANTE_INICIAL)
            meses_prev = rng.integers(*MESES_APORTE_PREVIOS, n_af)
            base_vol = tasa_mensual * np.maximum(rim_ref, imm) * meses_prev
            saldo_vol_inicial = (np.where(aportante_mensual, base_vol,
                                          np.where(ex_aportante, 0.5 * base_vol, 0.0))
                                 * rng.lognormal(0.0, SALDO_VOL_INICIAL_SIGMA, n_af))

        # ---------------------------------------------------- salidas del aporte mensual
        cambio_emp = (e != emp_previo) & (emp_previo >= 0)
        sale = aportante_mensual & (
            ~con_ingreso
            | (cambio_emp & (rng.random(n_af) < params.p_salida_cambio_empleador))
            | (rng.random(n_af) < params.p_salida_mensual))
        aportante_mensual &= ~sale

        # ---------------------------------------------------- entradas al aporte mensual
        candidato = con_ingreso & dependiente & ~aportante_mensual
        entra = candidato & (rng.random(n_af) < _sigmoide(z + params.intercepto_mensual))
        if entra.any():
            ids = np.where(entra)[0]
            t = _elegir_tipo(tramo[ids], rng.random(len(ids)))
            dc = (ofrece[ids] > 0) & (tramo[ids] >= 2) & (rng.random(len(ids)) < 0.12)
            tipo_mensual[ids] = np.where(dc, 3, t).astype(np.int8)
            canal_mensual[ids] = (rng.random(len(ids)) < 0.15).astype(np.int8)  # 0=planilla, 1=transferencia
            aportante_mensual[ids] = True

        # ---------------------------------------------------- aportes mensuales del mes
        paga_mensual = aportante_mensual & con_ingreso
        # ---------------------------------------------------- aportes esporadicos del mes
        z_esp = z + params.intercepto_esporadico + ESTACIONALIDAD_ESPORADICO.get(int(mes), 0.0)
        esporadico = con_ingreso & ~aportante_mensual & (rng.random(n_af) < _sigmoide(z_esp))

        tope_mes = TOPE_APORTE_UF * uf
        if paga_mensual.any():
            ids = np.where(paga_mensual)[0]
            base = tasa_mensual[ids] * np.maximum(rim_ref[ids], imm)
            base *= (1.0 + ELASTICIDAD_MONTO_TRAMO * tramo[ids])
            base *= np.where(tipo_mensual[ids] == 1, FACTOR_MONTO_REGIMEN_B, 1.0)
            monto = np.clip(base * rng.lognormal(0.0, 0.12, len(ids)), APORTE_MINIMO, tope_mes)
            canal = np.where(canal_mensual[ids] == 0, "DESCUENTO_PLANILLA", "TRANSFERENCIA")
            canal = np.where(np.isin(est[ids], ESTADOS_DEPENDIENTES), canal, "TRANSFERENCIA")
            filas_aportes.append(pd.DataFrame({
                "afiliado_id": ids + 1, "periodo": periodos[j],
                "tipo": np.array(TIPOS_APORTE)[tipo_mensual[ids]],
                "monto": np.round(monto, 0), "canal": canal, "periodicidad": "MENSUAL",
            }))
            aportes_mat[ids, j] += monto.astype(np.float32)
            ult_aporte[ids] = j

        if esporadico.any():
            ids = np.where(esporadico)[0]
            factor = FACTOR_ESPORADICO.get(int(mes), FACTOR_ESPORADICO_OTRO)
            t = _elegir_tipo(tramo[ids], rng.random(len(ids)))
            dc = (ofrece[ids] > 0) & (tramo[ids] >= 2) & (rng.random(len(ids)) < 0.08)
            t = np.where(dc, 3, t)
            base = factor * np.maximum(rim_ref[ids], imm) * (1.0 + ELASTICIDAD_MONTO_TRAMO * tramo[ids])
            base *= np.where(t == 1, FACTOR_MONTO_REGIMEN_B, 1.0)
            monto = np.clip(base * rng.lognormal(0.0, FACTOR_ESPORADICO_SIGMA, len(ids)),
                            APORTE_MINIMO, tope_mes)
            canal = np.where(rng.random(len(ids)) < 0.55, "TRANSFERENCIA", "APP")
            filas_aportes.append(pd.DataFrame({
                "afiliado_id": ids + 1, "periodo": periodos[j],
                "tipo": np.array(TIPOS_APORTE)[t],
                "monto": np.round(monto, 0), "canal": canal, "periodicidad": "ESPORADICO",
            }))
            aportes_mat[ids, j] += monto.astype(np.float32)
            ult_aporte[ids] = j

        emp_previo = np.where(presente, e, emp_previo)

    # ------------------------------------------------------------ tablas finales
    aportes = (pd.concat(filas_aportes, ignore_index=True) if filas_aportes
               else pd.DataFrame(columns=["afiliado_id", "periodo", "tipo", "monto", "canal", "periodicidad"]))
    aportes = aportes.sort_values(["periodo", "afiliado_id"]).reset_index(drop=True)
    aportes["afiliado_id"] = aportes["afiliado_id"].astype("int64")
    aportes["periodo"] = aportes["periodo"].astype("int32")
    aportes["monto"] = aportes["monto"].astype("float64")

    campanas = (pd.concat(filas_campanas, ignore_index=True) if filas_campanas
                else pd.DataFrame(columns=["afiliado_id", "periodo", "canal", "respuesta"]))
    campanas = campanas.sort_values(["periodo", "afiliado_id"]).reset_index(drop=True)
    campanas["afiliado_id"] = campanas["afiliado_id"].astype("int64")
    campanas["periodo"] = campanas["periodo"].astype("int32")

    saldos = _construir_saldos(rim, aportes_mat, edad, periodos, rng, saldo_vol_inicial)
    tributario = _construir_tributario(rim, periodos, afiliados["afiliado_id"].to_numpy())

    return {
        "afiliados": afiliados,
        "empleadores": empleadores,
        "panel_mensual": panel,
        "macro": macro,
        "apvc_empleador": apvc,
        "campanas": campanas,
        "aportes_voluntarios": aportes,
        "saldos": saldos,
        "tributario": tributario,
    }


def _construir_saldos(rim: np.ndarray, aportes_mat: np.ndarray, edad: np.ndarray,
                      periodos: list[int], rng: np.random.Generator,
                      saldo_vol_inicial: np.ndarray) -> pd.DataFrame:
    """Saldo obligatorio (10 % de la RIM capitalizado) y saldo voluntario, afiliado x mes."""
    n_af, n_per = rim.shape
    # saldo inicial: anos cotizados aproximados x 10 % de la primera RIM conocida, capitalizado
    primera_rim = np.zeros(n_af)
    for j in range(n_per):
        falta = primera_rim == 0
        if not falta.any():
            break
        primera_rim[falta] = rim[falta, j]
    edad0 = edad[:, 0].astype(float)
    edad0 = np.where(edad0 > 0, edad0, 30.0)
    meses_previos = np.maximum(edad0 - 25.0, 0.0) * 12.0
    saldo0 = (TASA_COTIZACION_OBLIGATORIA * primera_rim * meses_previos * SALDO_INICIAL_FACTOR
              * rng.lognormal(0.0, SALDO_INICIAL_SIGMA, n_af))

    obl = np.zeros((n_af, n_per), dtype=np.float32)
    vol = np.zeros((n_af, n_per), dtype=np.float32)
    s_o = saldo0.astype(np.float64)
    s_v = np.asarray(saldo_vol_inicial, dtype=np.float64)
    for j in range(n_per):
        s_o = s_o * (1.0 + RENT_MENSUAL_OBLIGATORIO) + TASA_COTIZACION_OBLIGATORIA * rim[:, j]
        s_v = s_v * (1.0 + RENT_MENSUAL_VOLUNTARIO) + aportes_mat[:, j]
        obl[:, j] = s_o
        vol[:, j] = s_v
    return pd.DataFrame({
        "afiliado_id": np.repeat(np.arange(1, n_af + 1, dtype="int64"), n_per),
        "periodo": np.tile(np.asarray(periodos, dtype="int32"), n_af),
        "saldo_obligatorio": np.round(obl.ravel(), 0),
        "saldo_voluntario": np.round(vol.ravel(), 0),
    })


def _construir_tributario(rim: np.ndarray, periodos: list[int], afiliado_ids: np.ndarray) -> pd.DataFrame:
    """Renta anual y tramo del IGC por afiliado y anio calendario (solo anos completos)."""
    anios = sorted({p // 100 for p in periodos})
    pos = indexar_periodos(periodos)
    filas = []
    for anio in anios:
        cols = [pos[p] for p in periodos if p // 100 == anio]
        if len(cols) < 12:          # anio incompleto en la ventana simulada
            continue
        renta = rim[:, cols].sum(axis=1)
        tramo = tramo_igc(renta, anio)
        filas.append(pd.DataFrame({
            "afiliado_id": afiliado_ids,
            "anio": np.int32(anio),
            "renta_anual": np.round(renta, 0),
            "tramo_igc": tramo.astype("int8"),
            "tasa_marginal_igc": tasa_marginal_igc(tramo),
        }))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame(
        columns=["afiliado_id", "anio", "renta_anual", "tramo_igc", "tasa_marginal_igc"])


def truncar_tablas(tablas: dict[str, pd.DataFrame], periodo_corte: int) -> dict[str, pd.DataFrame]:
    """Copia de `tablas` sin nada que no este disponible en el mes de decision `periodo_corte`.

    Aplica el desfase propio de cada fuente (`DESFASE_POR_TABLA`) y, ademas, **neutraliza las
    columnas del panel del mes `corte-1`**: de ese mes solo se conocen las licencias medicas, de
    modo que si alguna feature leyera la RIM, el estado o el empleador de `corte-1` el test de
    leakage lo detectaria. `macro` no se trunca porque el IMM y el tope imponible estan
    legislados por adelantado.
    """
    corte_idx = periodo_to_idx(periodo_corte)
    fuera: dict[str, pd.DataFrame] = {}
    for nombre, df in tablas.items():
        if nombre == "macro":
            fuera[nombre] = df.copy()
        elif nombre == "panel_mensual":
            idx = df["periodo"].map(periodo_to_idx)
            g = df[idx <= corte_idx - MESES_DESFASE_LICENCIA].copy().reset_index(drop=True)
            ultimo = g["periodo"].map(periodo_to_idx) == corte_idx - MESES_DESFASE_LICENCIA
            cols_monto = ["sueldo_base", "rim_emp1", "rim_emp2", "gratificacion", "aguinaldo",
                          "variable", "jornada"]
            g.loc[ultimo, cols_monto] = 0.0
            g.loc[ultimo, "estado"] = "CESANTE"
            g.loc[ultimo, ["empleador_id", "empleador2_id"]] = pd.NA
            g.loc[ultimo, "meses_antiguedad"] = 0
            fuera[nombre] = g
        elif nombre == "tributario":
            fuera[nombre] = df[df["anio"] <= anio_tributario_disponible(periodo_corte)
                               ].reset_index(drop=True)
        elif "periodo" in df.columns:
            lag = DESFASE_POR_TABLA.get(nombre, MESES_DESFASE_OPERACIONAL)
            idx = df["periodo"].map(periodo_to_idx)
            fuera[nombre] = df[idx <= corte_idx - lag].reset_index(drop=True)
        else:
            fuera[nombre] = df.copy()
    return fuera
