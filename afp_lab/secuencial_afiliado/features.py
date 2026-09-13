"""Features "as-of" del proyecto 08: secuencias mensuales (panel hasta `corte-2`, igual regla
que 02/04) + estaticas, y armado del dataset de cada cabeza REUSANDO las funciones de target y
de split de 02 (`churn_traspaso`) y 04 (`riesgo_cese_cotizacion`) sin modificarlas ni
reimplementarlas.

Reglas as-of (identicas a 02/04, CONVENCIONES.md #3):
    * panel_mensual (RIM, estado, tipo_contrato, jornada, dias_licencia, evento_afc): conocido
      hasta `corte - MESES_DESFASE_RIM` (2 meses, la cotizacion de *m* se conoce en *m+2*).
    * El target de cada cabeza vive en `corte+1 .. corte+horizonte` y se construye con la
      funcion de su proyecto de origen (`churn_traspaso.features.construir_target`,
      `riesgo_cese_cotizacion.features.agregar_targets`); aqui no se reimplementa nada de eso.

La secuencia es deliberadamente MAS ANGOSTA que las features tabulares completas de 02/04: solo
lleva las variables mensuales del panel comun (BRIEF.md las lista: RIM, estado, tipo de
contrato, jornada, licencias, evento AFC, empleador cambia) + estaticas de `afiliados`. NO
incluye las tablas operacionales propias de 02 (contactos, reclamos, cambios_fondo, saldos,
uso_digital): esas alimentan el LightGBM "de produccion" de 02/04 (con sus propias features),
pero dejarlas fuera de la secuencia permite que el baseline "tabular con lags" (mismas
variables, aplanadas) aisle el efecto de la arquitectura del efecto de la informacion (BRIEF.md,
seccion Baselines) frente al encoder secuencial. Documentado tambien en el README, limitacion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from afp_lab.rim_nowcasting.calendario import add_months, periodo_to_idx, rango_periodos

from . import MESES_DESFASE_RIM, VENTANA_SECUENCIA

# --------------------------------------------------------------------- vocabularios (fijos,
# no se ajustan con datos: cualquier categoria no vista cae en el ultimo indice de la lista).
ESTADOS = ["SIN_PANEL", "EMPLEADO", "CESANTE", "INDEPENDIENTE", "LICENCIA", "VOLUNTARIO",
          "RETIRADO", "PENSIONADO_ACTIVO", "OTRO_ESTADO"]
EVENTOS_AFC = ["NINGUNO", "INICIO", "TERMINO", "OTRO_EVENTO"]
TIPOS_CONTRATO_PANEL = ["SIN_CONTRATO", "INDEFINIDO", "PLAZO_FIJO", "HONORARIOS", "OTRO_CONTRATO"]
SEXOS = ["F", "M", "OTRO_SEXO"]
NIVELES_EDUC = ["BASICA", "MEDIA", "TECNICA", "UNIVERSITARIA", "OTRO_NIVEL"]
TIPOS_AFILIADO = ["DEPENDIENTE", "INDEPENDIENTE", "VOLUNTARIO", "PENSIONADO_ACTIVO", "OTRO_TIPO"]
N_REGIONES = 17  # regiones de Chile van de 1 a 16; 0 = sin dato

CANALES_NUM = ["rim_log", "dias_licencia_norm", "jornada", "meses_antiguedad_norm", "empleador_cambia"]
CANALES_CAT = ["estado", "evento_afc", "tipo_contrato"]
VOCAB_CAT = {"estado": ESTADOS, "evento_afc": EVENTOS_AFC, "tipo_contrato": TIPOS_CONTRATO_PANEL}

CANALES_STATIC_NUM = ["edad_norm", "antiguedad_afiliacion_norm"]
CANALES_STATIC_CAT = ["sexo", "region", "nivel_educacional", "tipo_afiliado"]
VOCAB_STATIC_CAT = {"sexo": SEXOS, "region": list(range(N_REGIONES)), "nivel_educacional": NIVELES_EDUC,
                    "tipo_afiliado": TIPOS_AFILIADO}


def _codificar(valores: np.ndarray, vocab: list) -> np.ndarray:
    """Codigo entero de `valores` (array de objetos) segun `vocab` (lista fija); lo no
    reconocido (incluido NaN) cae en el ULTIMO indice de `vocab` (categoria "otro"/sentinela).
    Determinista, no se ajusta con datos: sin fuga posible entre splits."""
    codigos = pd.Categorical(pd.Series(valores.ravel()), categories=vocab[:-1]).codes
    codigos = np.where(codigos < 0, len(vocab) - 1, codigos)
    return codigos.reshape(valores.shape).astype(np.int16)


@dataclass
class MatricesPanel:
    """Panel comun pivoteado UNA vez a (n_afiliados x periodos) por canal; `construir_secuencias`
    recorta de aqui la ventana de 24 meses para cada corte solicitado (sin volver a leer el
    panel crudo por cada fila)."""
    periodos: list[int]
    pos: dict[int, int]
    n_af: int
    rim: np.ndarray                # (n_af, T) float32, ya con log1p y tope aplicados
    dias_licencia: np.ndarray      # (n_af, T) float32, normalizado (/30)
    jornada: np.ndarray            # (n_af, T) float32
    meses_antiguedad: np.ndarray   # (n_af, T) float32, normalizado (/60)
    empleador_cambia: np.ndarray   # (n_af, T) float32 (0/1)
    estado_cod: np.ndarray         # (n_af, T) int16
    evento_afc_cod: np.ndarray     # (n_af, T) int16
    tipo_contrato_cod: np.ndarray  # (n_af, T) int16
    existe: np.ndarray             # (n_af, T) bool: hubo fila de panel ese mes


def _pivotar(panel: pd.DataFrame, col: str, n_af: int, periodos: list[int]) -> np.ndarray:
    """(n_af, len(periodos)) alineado por posicion: fila `i` = afiliado_id `i+1` (los
    afiliado_id del universo comun son 1..n_af, consecutivos; `synth_base.generar_afiliados_pandas`)."""
    idx = pd.MultiIndex.from_arrays([panel["afiliado_id"].to_numpy(), panel["periodo"].to_numpy()])
    s = pd.Series(panel[col].to_numpy(), index=idx)
    s = s[~s.index.duplicated(keep="last")]
    completo = pd.MultiIndex.from_product([range(1, n_af + 1), periodos])
    return s.reindex(completo).to_numpy().reshape(n_af, len(periodos))


def construir_matrices_panel(tablas: dict[str, pd.DataFrame], n_afiliados: int) -> MatricesPanel:
    """Pivotea `panel_mensual` una sola vez; `n_afiliados` debe ser el `n_afiliados` con el que
    se generó el universo (afiliado_id corre 1..n_afiliados)."""
    panel = tablas["panel_mensual"]
    macro = tablas["macro"]
    periodos = rango_periodos(int(macro["periodo"].min()), int(macro["periodo"].max()))
    tope = macro.set_index("periodo")["tope_clp"].reindex(periodos).to_numpy(dtype=float)

    estado_raw = _pivotar(panel, "estado", n_afiliados, periodos)
    existe = pd.notna(estado_raw)
    rim1 = _pivotar(panel, "rim_emp1", n_afiliados, periodos).astype(float)
    rim2 = _pivotar(panel, "rim_emp2", n_afiliados, periodos).astype(float)
    rim = np.nan_to_num(rim1, nan=0.0) + np.nan_to_num(rim2, nan=0.0)
    rim = np.minimum(rim, tope[None, :])  # mismo tope legal que usa 02 (afiliado_id x periodo)
    dias_lic = np.nan_to_num(_pivotar(panel, "dias_licencia", n_afiliados, periodos).astype(float), nan=0.0)
    jornada = np.nan_to_num(_pivotar(panel, "jornada", n_afiliados, periodos).astype(float), nan=0.0)
    meses_ant = np.nan_to_num(_pivotar(panel, "meses_antiguedad", n_afiliados, periodos).astype(float), nan=0.0)
    evento_raw = _pivotar(panel, "evento_afc", n_afiliados, periodos)
    contrato_raw = _pivotar(panel, "tipo_contrato", n_afiliados, periodos)

    empleador_raw = _pivotar(panel, "empleador_id", n_afiliados, periodos)
    emp_prev = np.empty_like(empleador_raw)
    emp_prev[:, 0] = np.nan
    emp_prev[:, 1:] = empleador_raw[:, :-1]
    ambos_conocidos = pd.notna(empleador_raw) & pd.notna(emp_prev)
    cambia = np.zeros(empleador_raw.shape, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        distinto = (empleador_raw != emp_prev)
    cambia[ambos_conocidos & distinto] = 1.0

    return MatricesPanel(
        periodos=periodos, pos={p: i for i, p in enumerate(periodos)}, n_af=n_afiliados,
        rim=np.log1p(np.clip(rim, 0, None)).astype(np.float32),
        dias_licencia=(dias_lic / 30.0).astype(np.float32),
        jornada=jornada.astype(np.float32),
        meses_antiguedad=(meses_ant / 60.0).astype(np.float32),
        empleador_cambia=cambia,
        estado_cod=_codificar(np.where(existe, estado_raw, None), ESTADOS),
        evento_afc_cod=_codificar(evento_raw, EVENTOS_AFC),
        tipo_contrato_cod=_codificar(contrato_raw, TIPOS_CONTRATO_PANEL),
        existe=existe,
    )


def _ventana(mat: np.ndarray, filas: np.ndarray, pos_fin: int, largo: int, relleno: float) -> np.ndarray:
    """Columnas `[pos_fin-largo+1 .. pos_fin]` de `mat[filas, :]`, rellenando por la IZQUIERDA
    con `relleno` si la ventana se sale del rango simulado (afiliado sin historia suficiente
    antes de `periodo_inicio`; nunca se rellena por la derecha: `pos_fin` es siempre `<=`
    ultima columna disponible porque los cortes usados son "evaluables" de cada proyecto)."""
    n_t = mat.shape[1]
    pos_ini = pos_fin - largo + 1
    out = np.full((len(filas), largo), relleno, dtype=mat.dtype)
    src_ini = max(pos_ini, 0)
    src_fin = min(pos_fin, n_t - 1)
    largo_valido = src_fin - src_ini + 1
    if largo_valido > 0:
        dst_ini = src_ini - pos_ini
        out[:, dst_ini:dst_ini + largo_valido] = mat[filas][:, src_ini:src_fin + 1]
    return out


@dataclass
class SecuenciaTensores:
    """Tensores alineados fila a fila con `meta` (mismo orden), listos para `model.py`
    (que los convierte a `torch.Tensor`; aqui quedan como numpy, sin dependencia de torch)."""
    meta: pd.DataFrame                      # afiliado_id, periodo_corte
    num: np.ndarray                         # (N, VENTANA, len(CANALES_NUM)) float32
    cat: np.ndarray                         # (N, VENTANA, len(CANALES_CAT)) int16
    mask: np.ndarray                        # (N, VENTANA) float32, 1 = mes observado
    static_num: np.ndarray                  # (N, len(CANALES_STATIC_NUM)) float32
    static_cat: np.ndarray                  # (N, len(CANALES_STATIC_CAT)) int16


def construir_secuencias(mats: MatricesPanel, afiliados: pd.DataFrame, pares: pd.DataFrame,
                         ventana: int = VENTANA_SECUENCIA,
                         meses_desfase: int = MESES_DESFASE_RIM) -> SecuenciaTensores:
    """`pares`: DataFrame con (al menos) `afiliado_id`, `periodo_corte` -- las filas a construir,
    ya filtradas por poblacion/split de la cabeza que corresponda. El orden de salida es el
    mismo que el de `pares` (reset_index)."""
    pares = pares[["afiliado_id", "periodo_corte"]].reset_index(drop=True)
    n = len(pares)
    num = np.zeros((n, ventana, len(CANALES_NUM)), dtype=np.float32)
    cat = np.zeros((n, ventana, len(CANALES_CAT)), dtype=np.int16)
    mask = np.zeros((n, ventana), dtype=np.float32)

    for corte, grupo in pares.groupby("periodo_corte"):
        pos_fin = mats.pos[int(add_months(int(corte), -meses_desfase))]
        filas_af = (grupo["afiliado_id"].to_numpy() - 1)  # afiliado_id -> fila 0-based
        idx_salida = grupo.index.to_numpy()
        num[idx_salida, :, 0] = _ventana(mats.rim, filas_af, pos_fin, ventana, 0.0)
        num[idx_salida, :, 1] = _ventana(mats.dias_licencia, filas_af, pos_fin, ventana, 0.0)
        num[idx_salida, :, 2] = _ventana(mats.jornada, filas_af, pos_fin, ventana, 0.0)
        num[idx_salida, :, 3] = _ventana(mats.meses_antiguedad, filas_af, pos_fin, ventana, 0.0)
        num[idx_salida, :, 4] = _ventana(mats.empleador_cambia, filas_af, pos_fin, ventana, 0.0)
        cat[idx_salida, :, 0] = _ventana(mats.estado_cod, filas_af, pos_fin, ventana, 0)
        cat[idx_salida, :, 1] = _ventana(mats.evento_afc_cod, filas_af, pos_fin, ventana, 0)
        cat[idx_salida, :, 2] = _ventana(mats.tipo_contrato_cod, filas_af, pos_fin, ventana, 0)
        existe_f = _ventana(mats.existe.astype(np.float32), filas_af, pos_fin, ventana, 0.0)
        mask[idx_salida, :] = existe_f

    af = afiliados.set_index("afiliado_id").reindex(pares["afiliado_id"])
    corte_idx = pares["periodo_corte"].map(periodo_to_idx).to_numpy()
    afil_idx = af["periodo_afiliacion"].map(periodo_to_idx).to_numpy()
    edad_actual = af["edad_inicio"].to_numpy() + (corte_idx - afil_idx) // 12  # proyeccion simple, igual criterio que 02/04
    antiguedad_meses = np.clip(corte_idx - afil_idx, 0, None)

    static_num = np.stack([
        (edad_actual / 100.0).astype(np.float32),
        (antiguedad_meses / 360.0).astype(np.float32),
    ], axis=1)
    static_cat = np.stack([
        _codificar(af["sexo"].to_numpy(), SEXOS),
        _codificar(af["region"].to_numpy(), list(range(N_REGIONES)) + ["OTRA_REGION"]),
        _codificar(af["nivel_educacional"].to_numpy(), NIVELES_EDUC),
        _codificar(af["tipo_afiliado"].to_numpy(), TIPOS_AFILIADO),
    ], axis=1)

    return SecuenciaTensores(meta=pares, num=num, cat=cat, mask=mask,
                             static_num=static_num, static_cat=static_cat)


def aplanar_secuencia(t: SecuenciaTensores) -> pd.DataFrame:
    """Aplana la secuencia a columnas anchas `<canal>_lag<k>` (k=1 el mes mas reciente,
    `corte-2`; k=VENTANA el mas antiguo) + estaticas: el baseline LightGBM "tabular con lags"
    (BRIEF.md) ve EXACTAMENTE las mismas variables que la red, sin la arquitectura secuencial."""
    ventana = t.num.shape[1]
    cols: dict[str, np.ndarray] = {}
    for i, canal in enumerate(CANALES_NUM):
        for k in range(ventana):
            cols[f"{canal}_lag{ventana - k}"] = t.num[:, k, i]
    for i, canal in enumerate(CANALES_CAT):
        for k in range(ventana):
            cols[f"{canal}_lag{ventana - k}"] = t.cat[:, k, i].astype(str)
    for i, canal in enumerate(CANALES_STATIC_NUM):
        cols[canal] = t.static_num[:, i]
    for i, canal in enumerate(CANALES_STATIC_CAT):
        cols[canal] = t.static_cat[:, i].astype(str)
    df = pd.DataFrame(cols)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    return df


def columnas_categoricas_planas(ventana: int = VENTANA_SECUENCIA) -> list[str]:
    cols = [f"{canal}_lag{k}" for canal in CANALES_CAT for k in range(1, ventana + 1)]
    cols += list(CANALES_STATIC_CAT)
    return cols
