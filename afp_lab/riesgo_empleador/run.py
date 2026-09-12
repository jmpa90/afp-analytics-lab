"""Demo end-to-end del proyecto 05 (riesgo de empleador).

    python -m afp_lab.riesgo_empleador.run --n-afiliados 20000

Genera el universo (perfil cuprum) + la planilla Previred + la capa de recepcion/pago, construye el
dataset as-of de empleadores, entrena con **origenes moviles** (reentrenamiento trimestral con gap
entre todas las particiones), compara con los baselines, y escribe metricas por segmento y por
corte, calibracion, metrica de negocio (masa en riesgo capturada) y el error del agregado mensual
en `salidas/riesgo_empleador/`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import features as F
from . import model as M
from .synthetic import AFILIADOS_POR_EMPLEADOR, HORIZONTE_LARGO, PERFIL, VERSION_PLANILLA, generar_tablas

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DEFECTO = RAIZ / "salidas" / "riesgo_empleador"
PRIMER_CORTE = 202001     # necesita 12 meses de historia de planillas (burn-in desde 201901)
N_TEST = 12   # minimo que pide CONVENCIONES §3; mantenerlo bajo minimiza el solape entre el
              # valid de un origen y el pool de test de los demas (ver README §4)
N_VALID = 12
CADA_ORIGEN = 3


def _imprimir(titulo: str, df: pd.DataFrame, n: int | None = None) -> None:
    print(f"\n=== {titulo} ===")
    with pd.option_context("display.width", 220, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print((df.head(n) if n else df).to_string(index=False))


def _guardar(df: pd.DataFrame, ruta: Path, nombre: str) -> None:
    ruta.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta / f"{nombre}.csv", index=False)


def ejecutar(n_afiliados: int = 5_000, salida: Path = SALIDA_DEFECTO, seed: int = 42,
             primer_corte: int = PRIMER_CORTE, n_test: int = N_TEST, n_valid: int = N_VALID,
             cada_origen: int = CADA_ORIGEN, perfil: str = PERFIL,
             afiliados_por_empleador: int = AFILIADOS_POR_EMPLEADOR,
             diagnostico: bool = True) -> dict:
    t0 = time.perf_counter()
    salida = Path(salida)
    tiempos: dict[str, float] = {}

    # ------------------------------------------------------------------ 1. datos sinteticos
    tablas = generar_tablas(n_afiliados=n_afiliados, seed=seed, perfil=perfil,
                            afiliados_por_empleador=afiliados_por_empleador)
    tiempos["generacion"] = time.perf_counter() - t0
    fuentes = salida / "fuentes"
    fuentes.mkdir(parents=True, exist_ok=True)
    for nombre, df in tablas.items():
        df.to_parquet(fuentes / f"{nombre}.parquet", index=False)
    resumen_fuentes = pd.DataFrame(
        [{"tabla": k, "filas": len(v), "columnas": v.shape[1]} for k, v in tablas.items()])

    pl = tablas["planillas"]
    tasas = pl["estado"].value_counts(normalize=True)
    resumen_sim = pd.DataFrame([{
        "perfil": perfil, "version_planilla": VERSION_PLANILLA, "afiliados_por_empleador": afiliados_por_empleador,
        "empleadores": int(pl["empleador_id"].nunique()), "planillas": len(pl),
        "share_pagada": float(tasas.get("PAGADA", 0.0)),
        "share_dnp": float(tasas.get("DNP", 0.0)),
        "share_sin_declarar": float(tasas.get("SIN_DECLARAR", 0.0)),
        "share_con_deuda": float(tasas.get("DNP", 0.0) + tasas.get("SIN_DECLARAR", 0.0)),
        "share_recepcion_tardia": float((pl["lag_recepcion"] >= 2).mean()),
        "episodios_estres": len(tablas["shocks_empleador"]),
        "liquidaciones": int((tablas["shocks_empleador"]["desenlace"] == "LIQUIDACION").sum())
        if len(tablas["shocks_empleador"]) else 0,
        "empleadores_en_boletin": int(tablas["boletin_infractores"]["empleador_id"].nunique())
        if len(tablas["boletin_infractores"]) else 0,
    }])

    # ------------------------------------------------------------------ 2. dataset as-of
    t = time.perf_counter()
    denso = F.preparar_panel(tablas)
    cortes = F.cortes_disponibles(tablas, primer_corte, HORIZONTE_LARGO)
    origenes = F.origenes_moviles(cortes, n_test=n_test, n_valid=n_valid, gap=HORIZONTE_LARGO,
                                  horizonte=HORIZONTE_LARGO, cada=cada_origen)
    usados = sorted({c for o in origenes for c in o["train"] + o["valid"] + o["test"]})
    datos = F.construir_dataset(tablas, usados, denso=denso)
    idx = {c: g.index for c, g in datos.groupby("periodo_corte", observed=True)}
    tiempos["dataset"] = time.perf_counter() - t

    cortes_test = [c for o in origenes for c in o["test"]]
    test = datos.loc[np.concatenate([idx[c] for c in cortes_test])].reset_index(drop=True)

    # ------------------------------------------------------------------ 3. origenes moviles
    t = time.perf_counter()
    params = M.ParametrosModelo(seed=seed)
    filas_split, filas_ajuste, trozos, filas_modelos = [], [], [], []
    modelo = train_ult = valid_ult = train_primero = None
    for o in origenes:
        tr = datos.loc[np.concatenate([idx[c] for c in o["train"]])]
        va = datos.loc[np.concatenate([idx[c] for c in o["valid"]])]
        te = datos.loc[np.concatenate([idx[c] for c in o["test"]])]
        if train_primero is None:
            train_primero = tr
        m = M.entrenar(tr, va, params)
        ruta_m = M.guardar(m, salida / "modelo" / f"origen_{o['origen']}")
        filas_split.append({
            "origen": o["origen"],
            "train": f"{min(o['train'])}-{max(o['train'])}", "n_train": len(o["train"]),
            "valid": f"{min(o['valid'])}-{max(o['valid'])}", "n_valid": len(o["valid"]),
            "test": f"{min(o['test'])}-{max(o['test'])}",
            "filas_train": len(tr), "filas_test": len(te),
            "tasa_train": float(tr["y_dnp"].mean()), "tasa_valid": float(va["y_dnp"].mean()),
            "tasa_test": float(te["y_dnp"].mean())})
        filas_ajuste.append({
            "origen": o["origen"], "arboles": m.n_arboles(), "tope": params.n_estimators,
            "early_stopping_corto": m.n_arboles() < params.n_estimators,
            "aplica_isotonica": bool(m.usa_calibracion[M.TARGET_PRINCIPAL]),
            "mejora_media_entre_folds": m.mejora_media[M.TARGET_PRINCIPAL],
            "error_estandar_entre_folds": m.mejora_sem[M.TARGET_PRINCIPAL]})
        filas_modelos.append({"origen": o["origen"], "ruta": str(ruta_m.relative_to(salida)),
                              "predice_cortes": ",".join(str(c) for c in o["test"]),
                              "arboles": m.n_arboles()})
        trozos.append({
            "modelo": M.predecir(m, te),
            "persistencia": M.baseline_persistencia(train_primero, te),
            "regla_negocio": M.baseline_regla(train_primero, te),
            "tasa_base": M.baseline_tasa_base(train_primero, te)})
        modelo, train_ult, valid_ult = m, tr, va
    tiempos["entrenamiento"] = time.perf_counter() - t

    pred = {k: pd.concat([z[k] for z in trozos], ignore_index=True) for k in trozos[0]}
    for k in pred:
        pred[k].index = test.index
    resumen_split = pd.DataFrame(filas_split)
    ajuste = pd.DataFrame(filas_ajuste)
    modelos_por_origen = pd.DataFrame(filas_modelos)
    solape = F.solape_valid_pool_test(origenes, HORIZONTE_LARGO)

    # ------------------------------------------------------------------ 4. evaluacion
    t = time.perf_counter()
    clasificacion = ev.tabla_clasificacion(test, pred)
    targets = ev.tabla_clasificacion_todos_los_targets(test, pred["modelo"])
    por_corte = ev.metricas_por_corte(test, pred["modelo"])
    agregado = ev.error_agregado_mensual(por_corte)

    segmentos = {}
    for seg in ("tamano_empleador", "rubro", "region_empleador", "tramo_dotacion"):
        segmentos[seg] = ev.comparar_segmentos(test, pred, seg)
    masa_seg = ev.masa_por_segmento(test, pred["modelo"], "tamano_empleador")

    calib = ev.curva_calibracion(test["y_dnp"].to_numpy(), pred["modelo"]["prob_y_dnp"].to_numpy())
    n_cal = int(ajuste["aplica_isotonica"].sum())
    resumen_calib = pd.DataFrame([
        {"version": f"probabilidad usada ({n_cal} de {len(ajuste)} origenes calibrados)",
         **ev.resumen_calibracion(test["y_dnp"].to_numpy(), pred["modelo"]["prob_y_dnp"].to_numpy())},
        {"version": "isotonica por origen (referencia)",
         **ev.resumen_calibracion(test["y_dnp"].to_numpy(),
                                  pred["modelo"]["prob_isotonica_y_dnp"].to_numpy())},
    ])
    decision_calib = ajuste[["origen", "aplica_isotonica", "mejora_media_entre_folds",
                            "error_estandar_entre_folds"]].assign(
        n_sigmas_exigidas=M.N_SIGMAS_CALIBRACION,
        mejora_relativa_minima=M.MEJORA_MINIMA_CALIBRACION)

    scores = {
        "modelo_masa_esperada": pred["modelo"]["masa_en_riesgo_esperada"].to_numpy(),
        "modelo_probabilidad": pred["modelo"]["prob_y_dnp"].to_numpy(),
        "regla_negocio": pred["regla_negocio"]["masa_en_riesgo_esperada"].to_numpy(),
        "persistencia": pred["persistencia"]["masa_en_riesgo_esperada"].to_numpy(),
        "masa_sola": test["masa_ref_clp"].to_numpy(),
        "aleatorio": np.random.default_rng(seed).random(len(test)),
    }
    negocio = ev.metrica_negocio(test, scores)
    resumen_neg = ev.resumen_negocio(negocio)
    aceptacion = ev.chequear_aceptacion(clasificacion, negocio, por_corte, agregado)

    imp = M.importancias(modelo)
    ejemplos = M.explicar(modelo, test, n_ejemplos=5, seed=seed)
    memorizacion = (M.diagnostico_memorizacion(train_ult, valid_ult, modelo) if diagnostico
                    else pd.DataFrame())
    tiempos["evaluacion"] = time.perf_counter() - t
    tiempos["total"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ 5. salidas
    rep = salida / "reportes"
    for nombre, df in (("resumen_fuentes", resumen_fuentes), ("resumen_simulacion", resumen_sim),
                       ("resumen_split", resumen_split), ("ajuste_modelo", ajuste),
                       ("modelos_por_origen", modelos_por_origen),
                       ("solape_valid_pool_test", solape),
                       ("metricas_clasificacion", clasificacion), ("metricas_por_target", targets),
                       ("metricas_por_corte", por_corte), ("error_agregado_mensual", agregado),
                       ("masa_por_tamano", masa_seg), ("curva_calibracion", calib),
                       ("resumen_calibracion", resumen_calib),
                       ("decision_calibracion", decision_calib),
                       ("metrica_negocio", negocio), ("metrica_negocio_resumen", resumen_neg),
                       ("aceptacion", aceptacion), ("importancias", imp),
                       ("ejemplos_explicados", ejemplos), ("tiempos", pd.DataFrame([tiempos]))):
        _guardar(df, rep, nombre)
    for seg, tabla in segmentos.items():
        _guardar(tabla, rep, f"metricas_por_{seg}")
    if len(memorizacion):
        _guardar(memorizacion, rep, "diagnostico_memorizacion")

    (salida / "dataset").mkdir(parents=True, exist_ok=True)
    for nombre, df in (("train_ultimo_origen", train_ult), ("valid_ultimo_origen", valid_ult),
                       ("test", test)):
        df.reset_index(drop=True).to_parquet(salida / "dataset" / f"{nombre}.parquet", index=False)
    test.assign(prob_dnp=pred["modelo"]["prob_y_dnp"].to_numpy(),
                masa_en_riesgo_esperada=pred["modelo"]["masa_en_riesgo_esperada"].to_numpy()
                )[["empleador_id", "periodo_corte", "y_dnp", "y_tardio", "y_dnp_3m",
                   "masa_periodo", "masa_en_riesgo", "prob_dnp", "masa_en_riesgo_esperada",
                   "tamano_empleador", "rubro"]].to_parquet(
        salida / "predicciones_test.parquet", index=False)

    # ------------------------------------------------------------------ 6. impresion
    _imprimir("Tablas sinteticas", resumen_fuentes)
    _imprimir("Simulacion: tasas de DNP y recepcion tardia", resumen_sim.T.reset_index())
    _imprimir("Origenes moviles (gap de 3 meses entre cada par de particiones)", resumen_split)
    _imprimir("Ajuste por origen (arboles por early stopping contra validacion)", ajuste)
    _imprimir("Modelos persistidos (uno por origen)", modelos_por_origen)
    _imprimir("Solape de meses de target: valid vs su test y vs el pool", solape)
    if len(memorizacion):
        _imprimir("Diagnostico de memorizacion por empleador", memorizacion)
    _imprimir("Clasificacion del target principal (y_dnp) en test", clasificacion)
    _imprimir("Los tres targets con el modelo", targets)
    _imprimir("Metricas y agregado de masa por corte de test", por_corte)
    _imprimir("Error del agregado mensual de masa en DNP", agregado)
    for seg, tabla in segmentos.items():
        _imprimir(f"AUC por {seg} (test)", tabla)
    _imprimir("Masa en riesgo real vs esperada por tamano", masa_seg)
    _imprimir("Decision de calibracion (en validacion con gap, 3 folds)", decision_calib)
    _imprimir("Calibracion en test", resumen_calib)
    _imprimir("Curva de calibracion (deciles)", calib)
    _imprimir("Metrica de negocio: masa en riesgo capturada al 10 % y 20 %", negocio)
    _imprimir("Metrica de negocio: modelo vs regla", resumen_neg)
    _imprimir("Importancias (top 15, target principal)",
              imp[imp["target"] == M.TARGET_PRINCIPAL].head(15))
    _imprimir("Ejemplos explicados (SHAP)", ejemplos)
    _imprimir("Criterios de aceptacion del brief (se evaluan sobre el agregado)", aceptacion)
    print("\n=== Tiempos (s) ===")
    for k, v in tiempos.items():
        print(f"{k:>16}: {v:8.1f}")
    print(f"\nSalidas en {salida}")

    return {"modelo": modelo, "clasificacion": clasificacion, "targets": targets,
            "por_corte": por_corte, "agregado": agregado, "negocio": negocio,
            "resumen_neg": resumen_neg, "aceptacion": aceptacion, "tiempos": tiempos,
            "origenes": origenes, "resumen_sim": resumen_sim, "solape": solape,
            "memorizacion": memorizacion, "ajuste": ajuste, "pred": pred,
            "datos": {"train": train_ult, "valid": valid_ult, "test": test}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Demo de riesgo de empleador (DNP y pago tardio)")
    ap.add_argument("--n-afiliados", type=int, default=5_000)
    ap.add_argument("--salida", type=Path, default=SALIDA_DEFECTO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primer-corte", type=int, default=PRIMER_CORTE)
    ap.add_argument("--perfil", choices=("sistema", "cuprum"), default=PERFIL)
    ap.add_argument("--cada-origen", type=int, default=CADA_ORIGEN)
    ap.add_argument("--afiliados-por-empleador", type=int, default=AFILIADOS_POR_EMPLEADOR,
                    help="afiliados por empleador (define el numero de empleadores; ronda 3 §2.4)")
    ap.add_argument("--sin-diagnostico", action="store_true",
                    help="omite la CV agrupada por empleador")
    args = ap.parse_args(argv)
    ejecutar(n_afiliados=args.n_afiliados, salida=args.salida, seed=args.seed,
             primer_corte=args.primer_corte, cada_origen=args.cada_origen, perfil=args.perfil,
             afiliados_por_empleador=args.afiliados_por_empleador,
             diagnostico=not args.sin_diagnostico)


if __name__ == "__main__":
    main()
