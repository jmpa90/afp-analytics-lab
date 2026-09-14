"""Demo end-to-end del proyecto 11 (voz del afiliado).

    python -m afp_lab.voz_afiliado.run --n-afiliados 10000 --salida salidas/voz_afiliado

Genera el universo `cuprum` + tablas de 02/03/planilla + satisfaccion latente y encuesta, arma el
dataset as-of, entrena el modelo de respuesta y el de NPS (sin ponderar / ponderado), compara contra
el NPS verdadero e ingenuo, estima los drivers (T-learner) y el valor de un punto de NPS.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from afp_lab.churn_traspaso import features as CT_F
from afp_lab.propension_apv import features as APV_F

from . import HORIZONTE_MESES, PERIODO_FIN, PERIODO_INICIO
from . import evaluate as E
from . import features as F
from . import model as M
from .synthetic import generar_tablas, resumen_simulacion, tabla_parametros

SALIDA_POR_DEFECTO = "salidas/voz_afiliado"


def _mostrar(titulo: str, df: pd.DataFrame, ruta: Path | None = None, max_filas: int = 40) -> None:
    print(f"\n=== {titulo} ===")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df.head(max_filas).to_string(index=False))
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False)


def _target_traspaso(tablas: dict, df: pd.DataFrame, horizonte: int) -> np.ndarray:
    tablas_churn = F._tablas_para_churn(tablas)
    t = CT_F.construir_target(tablas_churn, df[["afiliado_id", "periodo_corte"]].copy(), horizonte=horizonte)
    return t["y"].to_numpy()


def _target_apv(tablas: dict, df: pd.DataFrame, horizonte: int) -> np.ndarray:
    tablas_apv = F._tablas_para_apv(tablas)
    partes = []
    for corte in sorted(df["periodo_corte"].unique().tolist()):
        t = APV_F.construir_target(tablas_apv, int(corte), horizonte=horizonte)
        t = t.assign(periodo_corte=int(corte))
        partes.append(t)
    tabla = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(
        columns=["afiliado_id", "periodo_corte", "y", "monto"])
    m = df[["afiliado_id", "periodo_corte"]].merge(tabla, on=["afiliado_id", "periodo_corte"], how="left")
    return m["monto"].fillna(0.0).to_numpy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo de voz del afiliado (NPS inferido)")
    ap.add_argument("--n-afiliados", type=int, default=10_000)
    ap.add_argument("--salida", type=str, default=SALIDA_POR_DEFECTO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--periodo-inicio", type=int, default=PERIODO_INICIO)
    ap.add_argument("--periodo-fin", type=int, default=PERIODO_FIN)
    ap.add_argument("--n-test", type=int, default=12)
    ap.add_argument("--n-valid", type=int, default=12)
    ap.add_argument("--max-cortes-train", type=int, default=None)
    ap.add_argument("--sin-fuentes", action="store_true", help="no escribir las tablas sinteticas")
    args = ap.parse_args(argv)

    salida = Path(args.salida)
    rep = salida / "reportes"
    rep.mkdir(parents=True, exist_ok=True)
    t_ini = time.perf_counter()
    tiempos: list[dict] = []

    def marca(paso: str, t0: float) -> None:
        tiempos.append({"paso": paso, "segundos": round(time.perf_counter() - t0, 2)})

    par_tab = tabla_parametros()
    par_tab.to_csv(rep / "parametros.csv", index=False)
    print(f"Parametros de la simulacion: {len(par_tab)} -> reportes/parametros.csv")

    # ------------------------------------------------------------------ 1. datos y dataset
    t0 = time.perf_counter()
    tablas = generar_tablas(n_afiliados=args.n_afiliados, periodo_inicio=args.periodo_inicio,
                            periodo_fin=args.periodo_fin, seed=args.seed)
    marca("generar_tablas", t0)

    t0 = time.perf_counter()
    cortes = F.cortes_evaluables(tablas, horizonte=HORIZONTE_MESES)
    split = F.split_temporal(cortes, n_test=args.n_test, n_valid=args.n_valid, horizonte=HORIZONTE_MESES,
                             max_cortes_train=args.max_cortes_train)
    ds = F.construir_dataset(tablas, split["train"] + split["valid"] + split["test"])
    partes = {k: ds[ds["periodo_corte"].isin(split[k])].reset_index(drop=True) for k in ("train", "valid", "test")}
    marca("construir_dataset", t0)

    if not args.sin_fuentes:
        t0 = time.perf_counter()
        (salida / "fuentes").mkdir(parents=True, exist_ok=True)
        for nombre, tabla in tablas.items():
            tabla.to_parquet(salida / "fuentes" / f"{nombre}.parquet", index=False)
        ds.to_parquet(salida / "dataset.parquet", index=False)
        marca("guardar_fuentes", t0)

    tr, va, te = partes["train"], partes["valid"], partes["test"]
    resumen_split = pd.DataFrame([
        {"particion": k, "n_cortes": len(split[k]), "corte_min": min(split[k]), "corte_max": max(split[k]),
         "n_filas": int(len(d)), "n_invitados": int(d["invitado"].sum()), "n_respondientes": int(d["respondio"].sum())}
        for k, d in (("train", tr), ("valid", va), ("test", te))
    ] + [{"particion": "descartados (purga)", "n_cortes": len(split["descartados"]),
         "corte_min": min(split["descartados"]) if split["descartados"] else None,
         "corte_max": max(split["descartados"]) if split["descartados"] else None,
         "n_filas": 0, "n_invitados": 0, "n_respondientes": 0}])
    _mostrar("Split temporal (con purga de 6 meses entre particiones)", resumen_split, rep / "split.csv")

    _mostrar("Chequeos de la simulacion (invitacion, respuesta, NPS verdadero)",
            resumen_simulacion(tablas), rep / "resumen_simulacion.csv")

    # ------------------------------------------------------------------ 2. modelo de respuesta
    t0 = time.perf_counter()
    modelo_resp = M.entrenar_propension_respuesta(tr, va, seed=args.seed)
    for nombre, d in (("train", tr), ("valid", va), ("test", te)):
        d["p_hat_responde"] = M.predecir_propension_respuesta(modelo_resp, d)
        d["peso_ipw"] = M.pesos_ipw(d["p_hat_responde"].to_numpy())
    marca("modelo_respuesta", t0)

    va_inv = va[va["invitado"] == 1]
    auc_prop_resp = float(roc_auc_score(va_inv["respondio"], va_inv["p_hat_responde"]))
    print(f"AUC del modelo de respuesta (invitados de validacion): {auc_prop_resp:.3f}")

    # ------------------------------------------------------------------ 3. modelo de NPS (score + detractor)
    t0 = time.perf_counter()
    tr_r, va_r, te_r = (d[d["respondio"] == 1].reset_index(drop=True) for d in (tr, va, te))
    modelo_score_np = M.entrenar_score(tr_r, va_r, seed=args.seed)
    modelo_score_p = M.entrenar_score(tr_r, va_r, seed=args.seed,
                                      pesos_train=tr_r["peso_ipw"].to_numpy(), pesos_valid=va_r["peso_ipw"].to_numpy())
    modelo_det_np = M.entrenar_detractor(tr_r, va_r, seed=args.seed)
    modelo_det_p = M.entrenar_detractor(tr_r, va_r, seed=args.seed,
                                        pesos_train=tr_r["peso_ipw"].to_numpy(), pesos_valid=va_r["peso_ipw"].to_numpy())
    modelo_prom_np = M.entrenar_promotor(tr_r, va_r, seed=args.seed)
    modelo_prom_p = M.entrenar_promotor(tr_r, va_r, seed=args.seed,
                                        pesos_train=tr_r["peso_ipw"].to_numpy(), pesos_valid=va_r["peso_ipw"].to_numpy())
    marca("modelo_nps", t0)

    # Score de regresion (0-10): se usa para drivers/valor del punto y para los baselines "trivial"
    # y "regla" (su formula ya tiene suficiente dispersion para bucketizar sin sesgo). El NPS
    # INFERIDO propiamente tal (sin_ponderar/ponderado) se agrega con los DOS clasificadores
    # (promotor/detractor) en probabilidad, no bucketizando el score de regresion -- ver
    # `model.nps_de_probs`.
    scores_test = {
        "trivial": M.baseline_score_trivial(tr_r, te),
        "regla": M.baseline_score_regla(te),
    }
    probs_test = {
        "sin_ponderar": (M.predecir_promotor(modelo_prom_np, te), M.predecir_detractor(modelo_det_np, te)),
        "ponderado": (M.predecir_promotor(modelo_prom_p, te), M.predecir_detractor(modelo_det_p, te)),
    }
    prob_det_test_resp = {
        "sin_ponderar": M.predecir_detractor(modelo_det_np, te_r),
        "ponderado": M.predecir_detractor(modelo_det_p, te_r),
        "trivial": M.baseline_detractor_trivial(tr_r, te_r),
        "regla": M.baseline_detractor_regla(te_r),
    }

    # ------------------------------------------------------------------ 4. comparacion de NPS
    t0 = time.perf_counter()
    prep = E.preparar_comparacion(tablas, te, scores=scores_test, probs=probs_test)
    tabla_por_corte = E.nps_por(prep, "periodo_corte")
    _mostrar("NPS por corte de test: verdadero, ingenuo e inferido (sin ponderar / ponderado / baselines)",
            tabla_por_corte, rep / "nps_por_corte.csv")
    err = E.resumen_error(tabla_por_corte, "nps_verdadera")
    _mostrar("Error absoluto medio (puntos de NPS) de cada version vs el verdadero, y reduccion vs el ingenuo",
            err, rep / "nps_error_resumen.csv")

    prep_seg = E.agregar_segmentos(prep)
    tablas_seg = []
    for seg in E.SEGMENTOS:
        t = E.nps_por(prep_seg, seg)
        t.insert(0, "segmento", seg)
        t = t.rename(columns={seg: "valor"})
        tablas_seg.append(t)
    tabla_segmentos = pd.concat(tablas_seg, ignore_index=True)
    _mostrar("NPS por segmento (mes, tramo de edad, region, tramo de empleador)", tabla_segmentos,
            rep / "nps_por_segmento.csv", max_filas=60)
    marca("comparacion_nps", t0)

    # ------------------------------------------------------------------ 5. AUC de detractor y calibracion
    t0 = time.perf_counter()
    filas_det = []
    calibs = []
    for nombre, prob in prob_det_test_resp.items():
        m = E.metricas_detractor(te_r, prob)
        m["version"] = nombre
        filas_det.append(m)
        c = E.curva_calibracion(te_r["es_detractor"].to_numpy(), prob)
        c.insert(0, "version", nombre)
        calibs.append(c)
    tabla_det = pd.DataFrame(filas_det).set_index("version").reset_index()
    _mostrar("Clasificador de detractor en el holdout de respondientes (test)", tabla_det,
            rep / "detractor_metricas.csv")
    _mostrar("Calibracion por decil del clasificador de detractor", pd.concat(calibs, ignore_index=True),
            rep / "detractor_calibracion.csv", max_filas=40)
    marca("detractor", t0)

    _mostrar("Importancia de features (gain) - modelo de NPS ponderado", M.importancias(modelo_score_p, 20),
            rep / "importancias_score.csv")

    # 3 ejemplos individuales explicados (CONVENCIONES SS4: importancia global Y ejemplos puntuales),
    # con el clasificador de detractor ponderado: el de prob mas alta, el de prob mas baja y uno
    # mediano, sobre el holdout de respondientes de test.
    prob_det_p_te_r = prob_det_test_resp["ponderado"]
    orden_prob = np.argsort(prob_det_p_te_r)
    posiciones = {"prob_detractor_mas_baja": orden_prob[0], "prob_detractor_mediana": orden_prob[len(orden_prob) // 2],
                 "prob_detractor_mas_alta": orden_prob[-1]}
    ejemplos = []
    for etiqueta, pos in posiciones.items():
        fila = te_r.iloc[int(pos)]
        tabla = M.explicar_prediccion_individual(modelo_det_p, te_r, int(pos), top=5)
        tabla.insert(0, "ejemplo", etiqueta)
        tabla.insert(1, "afiliado_id", int(fila["afiliado_id"]))
        tabla.insert(2, "periodo_corte", int(fila["periodo_corte"]))
        tabla.insert(3, "prob_detractor_ponderado", float(prob_det_p_te_r[pos]))
        tabla.insert(4, "es_detractor_real", int(fila["es_detractor"]))
        tabla.insert(5, "respuesta_real_0_10", float(fila["respuesta"]))
        ejemplos.append(tabla)
    tabla_ejemplos = pd.concat(ejemplos, ignore_index=True)
    _mostrar("Ejemplos individuales explicados (contribucion LightGBM, modelo de detractor ponderado)",
            tabla_ejemplos, rep / "ejemplos_individuales.csv", max_filas=30)

    # ------------------------------------------------------------------ 6. drivers (T-learner)
    t0 = time.perf_counter()
    resp_trva = pd.concat([tr_r, va_r], ignore_index=True)
    drivers = M.estimar_drivers(resp_trva, pesos=resp_trva["peso_ipw"].to_numpy())
    _mostrar("Drivers: efecto estimado (T-learner) vs efecto verdadero del generador", drivers,
            rep / "drivers.csv")
    prioridad = M.tabla_que_arreglar_primero(drivers, resp_trva)
    _mostrar("Que arreglar primero (impacto agregado = |efecto| x incidencia)", prioridad,
            rep / "drivers_prioridad.csv")
    frac_signo_ok = float(drivers["signo_correcto"].dropna().mean()) if drivers["signo_correcto"].notna().any() \
        else float("nan")
    print(f"Drivers con signo correcto: {frac_signo_ok:.1%} (criterio de aceptacion: >= 80 %)")
    marca("drivers", t0)

    # ------------------------------------------------------------------ 7. valor del punto de NPS
    t0 = time.perf_counter()
    score_valor = M.predecir_score(modelo_score_p, te)
    y_traspaso = _target_traspaso(tablas, te, HORIZONTE_MESES)
    monto_apv = _target_apv(tablas, te, HORIZONTE_MESES)
    valor_traspaso = M.valor_punto_traspaso(te, score_valor, y_traspaso)
    valor_apv = M.valor_punto_apv(te, score_valor, monto_apv)
    _mostrar("Valor de un punto de NPS: traspaso a 6 meses (asociacion ajustada, NO un experimento)",
            pd.DataFrame([valor_traspaso]), rep / "valor_punto_traspaso.csv")
    _mostrar("Valor de un punto de NPS: aporte APV a 6 meses (asociacion ajustada, NO un experimento)",
            pd.DataFrame([valor_apv]), rep / "valor_punto_apv.csv")
    marca("valor_del_punto", t0)

    # ------------------------------------------------------------------ 8. chequeo de aceptacion
    mae_ponderado = float(err.set_index("version").loc["ponderado", "mae_puntos_nps"]) if "ponderado" in err["version"].to_numpy() else float("nan")
    mae_ingenuo = float(err.set_index("version").loc["ingenua", "mae_puntos_nps"]) if "ingenua" in err["version"].to_numpy() else float("nan")
    auc_detractor_ponderado = float(tabla_det.set_index("version").loc["ponderado", "auc"])
    aceptacion = pd.DataFrame([
        {"criterio": "NPS ponderado dentro de +-3 puntos del verdadero (MAE por corte)",
         "valor": mae_ponderado, "cumple": bool(mae_ponderado <= 3.0)},
        {"criterio": "MAE ponderado <= 50% del MAE ingenuo",
         "valor": mae_ponderado / mae_ingenuo if mae_ingenuo else float("nan"),
         "cumple": bool(mae_ponderado <= 0.5 * mae_ingenuo) if mae_ingenuo else False},
        {"criterio": "AUC de detractor (ponderado) >= 0.75", "valor": auc_detractor_ponderado,
         "cumple": bool(auc_detractor_ponderado >= 0.75)},
        {"criterio": "Drivers con signo correcto >= 80%", "valor": frac_signo_ok,
         "cumple": bool(frac_signo_ok >= 0.80) if frac_signo_ok == frac_signo_ok else False},
    ])
    _mostrar("Chequeo de los criterios de aceptacion del brief", aceptacion, rep / "aceptacion.csv")

    tiempos.append({"paso": "TOTAL", "segundos": round(time.perf_counter() - t_ini, 2)})
    _mostrar("Tiempos", pd.DataFrame(tiempos), rep / "tiempos.csv")
    print(f"\nSalidas en {salida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
