"""Demo end-to-end del proyecto 07 (uplift de retencion).

    PYTHONPATH=/home/user/wt/churn python3 -m afp_lab.uplift_retencion.run \\
        --n-afiliados 10000 --salida salidas/uplift_retencion

Genera el universo y el target de `churn_traspaso` (sin modificarlo), arma la campana historica
confundida + holdout aleatorio, entrena la propension y los tres meta-learners (T-, X-, DR-learner)
mas los baselines (riesgo, aleatorio), y mide Qini/AUUC en el holdout, la correlacion contra el CATE
verdadero y la metrica de negocio (traspasos evitados al 10 %/20 % de presupuesto de contacto).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from afp_lab.churn_traspaso import features as CT_F
from afp_lab.churn_traspaso.synthetic import PERIODO_FIN, PERIODO_INICIO, generar_tablas

from . import HORIZONTE_MESES, N_AFILIADOS_REFERENCIA
from . import evaluate as E
from . import features as F
from . import model as M
from .synthetic import ParametrosCampana

SALIDA_POR_DEFECTO = "salidas/uplift_retencion"
METODOS_UPLIFT = ("t_learner", "x_learner", "dr_learner")
METODOS_TODOS = METODOS_UPLIFT + ("riesgo", "aleatorio")
# Convencion de signo para RANKEAR a quien contactar (Qini y la metrica de negocio): `y` es traspaso
# (resultado MALO), asi que el CATE de un persuadible es NEGATIVO (el contacto reduce `y`) y el de un
# sleeping dog es POSITIVO (lo aumenta). La "prioridad de contacto" tiene que ser mayor cuanto MAS
# negativo el CATE estimado, por eso los tres meta-learners se usan con signo invertido (-tau_hat); el
# riesgo (mayor riesgo = contactar primero, la estrategia ingenua) y el aleatorio no se invierten. El
# CATE crudo (sin invertir) se sigue usando tal cual para comparar contra `lat_cate` (misma convencion
# de signo que la verdad del simulador).
SIGNO_PRIORIDAD_CONTACTO = {"t_learner": -1.0, "x_learner": -1.0, "dr_learner": -1.0,
                           "riesgo": 1.0, "aleatorio": 1.0}


def _mostrar(titulo: str, df: pd.DataFrame, ruta: Path | None = None, max_filas: int = 40) -> None:
    print(f"\n=== {titulo} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(df.head(max_filas).to_string(index=False))
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False)


def preparar(n_afiliados: int, seed: int, periodo_inicio: int, periodo_fin: int, n_test: int,
            n_valid: int, perfil: str = "cuprum", par: ParametrosCampana | None = None) -> dict:
    """Genera tablas, arma el dataset causal completo y lo parte en train/valid/test."""
    tablas = generar_tablas(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                            periodo_fin=periodo_fin, seed=seed, perfil=perfil)
    cortes = CT_F.cortes_evaluables(tablas, horizonte=HORIZONTE_MESES)
    split = CT_F.split_temporal(cortes, n_test=n_test, n_valid=n_valid, horizonte=HORIZONTE_MESES)
    ds, latente = F.construir_dataset_causal(tablas, cortes, seed=seed, par=par)
    partes = {k: ds[ds["periodo_corte"].isin(split[k])].reset_index(drop=True)
             for k in ("train", "valid", "test")}
    lat_partes = {k: latente[latente["periodo_corte"].isin(split[k])].reset_index(drop=True)
                 for k in ("train", "valid", "test")}
    return {"tablas": tablas, "split": split, "dataset": ds, "latente": latente,
           "partes": partes, "lat_partes": lat_partes}


def entrenar_todo(partes: dict, seed: int) -> dict:
    """Propension + T-/X-/DR-learner (poblacion confundida) + baseline de riesgo (poblacion completa)."""
    tr, va = partes["train"], partes["valid"]
    tr_c, va_c = tr[~tr["holdout"]].reset_index(drop=True), va[~va["holdout"]].reset_index(drop=True)
    prop = M.estimar_propension(tr_c, seed=seed)
    t_learner = M.entrenar_t_learner(tr_c, va_c, seed=seed)
    x_learner = M.entrenar_x_learner(tr_c, va_c, prop, seed=seed)
    dr_learner = M.entrenar_dr_learner(tr_c, va_c, prop, seed=seed)
    riesgo = M.entrenar_baseline_riesgo(tr, va, seed=seed)
    return {"propension": prop, "t_learner": t_learner, "x_learner": x_learner,
           "dr_learner": dr_learner, "riesgo": riesgo}


def puntuar(modelos: dict, df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "t_learner": M.predecir_t_learner(modelos["t_learner"], df),
        "x_learner": M.predecir_x_learner(modelos["x_learner"], df),
        "dr_learner": M.predecir_dr_learner(modelos["dr_learner"], df),
        "riesgo": M.predecir_baseline_riesgo(modelos["riesgo"], df),
        "aleatorio": rng.random(len(df)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo de uplift de retencion (a quien conviene contactar)")
    ap.add_argument("--n-afiliados", type=int, default=N_AFILIADOS_REFERENCIA)
    ap.add_argument("--salida", type=str, default=SALIDA_POR_DEFECTO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perfil", type=str, default="cuprum", choices=("sistema", "cuprum"))
    ap.add_argument("--periodo-inicio", type=int, default=PERIODO_INICIO)
    ap.add_argument("--periodo-fin", type=int, default=PERIODO_FIN)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--n-valid", type=int, default=3)
    args = ap.parse_args(argv)

    salida = Path(args.salida)
    rep = salida / "reportes"
    rep.mkdir(parents=True, exist_ok=True)
    t_ini = time.perf_counter()
    tiempos: list[dict] = []

    def marca(paso: str, t0: float) -> None:
        tiempos.append({"paso": paso, "segundos": round(time.perf_counter() - t0, 2)})

    # ------------------------------------------------------------------ 1. datos y dataset causal
    t0 = time.perf_counter()
    prep = preparar(args.n_afiliados, args.seed, args.periodo_inicio, args.periodo_fin,
                    args.n_test, args.n_valid, args.perfil)
    marca("generar_tablas + construir_dataset_causal", t0)
    partes, lat_partes, split = prep["partes"], prep["lat_partes"], prep["split"]
    tr, va, te = partes["train"], partes["valid"], partes["test"]
    lat_te = lat_partes["test"]
    print(f"Perfil: {args.perfil} | afiliados: {args.n_afiliados} | ventana {args.periodo_inicio}-{args.periodo_fin}")

    resumen_split = pd.DataFrame([
        {"particion": k, "n_cortes": len(split[k]), "corte_min": min(split[k]), "corte_max": max(split[k]),
         "n_filas": int(len(d)), "tasa_base_y": float(d["y"].mean()), "frac_tratados": float(d["T"].mean()),
         "frac_holdout": float(d["holdout"].mean())}
        for k, d in (("train", tr), ("valid", va), ("test", te))
    ] + [{"particion": "descartados (purga)", "n_cortes": len(split["descartados"]),
         "corte_min": min(split["descartados"]) if split["descartados"] else None,
         "corte_max": max(split["descartados"]) if split["descartados"] else None,
         "n_filas": 0, "tasa_base_y": float("nan"), "frac_tratados": float("nan"), "frac_holdout": float("nan")}])
    _mostrar("Split temporal (identico patron a 02) y composicion de la campana", resumen_split,
             rep / "split.csv")

    seg_test = lat_te["segmento"].value_counts(normalize=True).rename("frac").reset_index()
    seg_test.columns = ["segmento", "frac"]
    _mostrar("Composicion de segmentos (verdad del simulador) en test", seg_test, rep / "segmentos_test.csv")

    # ------------------------------------------------------------------ 2. modelos
    t0 = time.perf_counter()
    modelos = entrenar_todo(partes, seed=args.seed)
    marca("entrenar (propension + T/X/DR-learner + riesgo)", t0)

    # ------------------------------------------------------------------ 3. overlap
    e_test = M.predecir_propension(modelos["propension"], te)
    overlap = E.overlap_propension(e_test)
    _mostrar("Overlap de la propension estimada e(x) en test",
             pd.DataFrame([{"overlap_0_05_0_95": overlap, "criterio_aceptacion": ">= 0.80",
                            "cumple": overlap >= 0.80}]), rep / "overlap.csv")

    # ------------------------------------------------------------------ 4. Qini/AUUC (holdout aleatorio)
    t0 = time.perf_counter()
    scores_test = puntuar(modelos, te, seed=args.seed)
    prioridad_test = {n: SIGNO_PRIORIDAD_CONTACTO[n] * s for n, s in scores_test.items()}
    te_hold = te[te["holdout"]].copy()
    mask_hold = te["holdout"].to_numpy()
    for nombre, s in prioridad_test.items():
        te_hold[f"score_{nombre}"] = s[mask_hold]

    qini_general = pd.DataFrame([
        {"metodo": nombre, **E.bootstrap_qini(te_hold, f"score_{nombre}", seed=args.seed)}
        for nombre in METODOS_TODOS
    ])
    _mostrar("Qini/AUUC en el holdout aleatorio de test (con IC bootstrap)", qini_general,
             rep / "qini_general.csv")

    qini_corte = pd.concat([
        E.qini_por_corte(te_hold, f"score_{nombre}").assign(metodo=nombre) for nombre in METODOS_TODOS
    ], ignore_index=True)
    _mostrar("Qini/AUUC por corte de test (holdout aleatorio)", qini_corte, rep / "qini_por_corte.csv",
             max_filas=60)
    marca("qini_auuc", t0)

    # ------------------------------------------------------------------ 5. correlacion vs verdad sintetica
    corr = pd.DataFrame([
        {"metodo": nombre, **E.correlacion_cate(lat_te["lat_cate"].to_numpy(), scores_test[nombre])}
        for nombre in METODOS_TODOS
    ])
    _mostrar("Correlacion (Spearman) y MAE del CATE estimado vs el verdadero (lat_cate)", corr,
             rep / "correlacion_cate.csv")

    # ------------------------------------------------------------------ 6. metrica de negocio
    negocio = pd.concat([
        E.traspasos_evitados(lat_te, prioridad_test[nombre], fracciones=(0.10, 0.20)).assign(ranking=nombre)
        for nombre in METODOS_TODOS
    ], ignore_index=True)
    _mostrar("Traspasos evitados contactando al 10 %/20 % de la cartera (uplift vs riesgo vs aleatorio)",
             negocio.sort_values(["frac_contactada", "ranking"]), rep / "traspasos_evitados.csv")

    razon = (negocio[negocio["ranking"] == "x_learner"].set_index("frac_contactada")["traspasos_evitados"]
            / negocio[negocio["ranking"] == "riesgo"].set_index("frac_contactada")["traspasos_evitados"]
            .replace(0, np.nan))
    _mostrar("Razon traspasos evitados: uplift (X-learner) / riesgo, por presupuesto",
             razon.rename("razon_uplift_vs_riesgo").reset_index(), rep / "razon_uplift_vs_riesgo.csv")

    tabla_seg = E.tabla_segmento_accion(lat_te, scores_test["x_learner"])
    _mostrar('Tabla "segmento -> accion" (verdad del simulador + CATE estimado por el X-learner)',
             tabla_seg, rep / "segmento_accion.csv")

    tiempos.append({"paso": "TOTAL", "segundos": round(time.perf_counter() - t_ini, 2)})
    _mostrar("Tiempos", pd.DataFrame(tiempos), rep / "tiempos.csv")
    print(f"\nSalidas en {salida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
