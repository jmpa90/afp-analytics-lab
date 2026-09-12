"""Demo end-to-end del proyecto 02 (propension a traspaso).

    python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --salida salidas/churn_traspaso

Genera el universo sintetico, arma el dataset as-of, entrena LightGBM con split temporal,
compara contra los baselines obligatorios e imprime/guarda todas las tablas de resultados.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from . import HORIZONTE_MESES
from . import evaluate as E
from . import features as F
from . import model as M
from .synthetic import generar_tablas, resumen_simulacion

SALIDA_POR_DEFECTO = "salidas/churn_traspaso"
TABLAS_FUENTE = ("afiliados", "empleadores", "panel_mensual", "macro", "mercado", "saldos",
                 "cambios_fondo", "traspasos", "contactos", "reclamos", "uso_digital")


def _mostrar(titulo: str, df: pd.DataFrame, ruta: Path | None = None, max_filas: int = 40) -> None:
    """Imprime una tabla de resultados y la guarda en csv."""
    print(f"\n=== {titulo} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(df.head(max_filas).to_string(index=False))
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo de propension a traspaso a otra AFP")
    ap.add_argument("--n-afiliados", type=int, default=20_000)
    ap.add_argument("--salida", type=str, default=SALIDA_POR_DEFECTO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--periodo-inicio", type=int, default=202101)
    ap.add_argument("--periodo-fin", type=int, default=202609)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--n-valid", type=int, default=3)
    ap.add_argument("--sin-fuentes", action="store_true", help="no escribir las tablas sinteticas")
    args = ap.parse_args(argv)

    salida = Path(args.salida)
    rep = salida / "reportes"
    rep.mkdir(parents=True, exist_ok=True)
    t_ini = time.perf_counter()
    tiempos: list[dict] = []

    def marca(paso: str, t0: float) -> None:
        tiempos.append({"paso": paso, "segundos": round(time.perf_counter() - t0, 2)})

    # ------------------------------------------------------------------ 1. datos
    t0 = time.perf_counter()
    tablas = generar_tablas(n_afiliados=args.n_afiliados, periodo_inicio=args.periodo_inicio,
                            periodo_fin=args.periodo_fin, seed=args.seed)
    marca("generar_tablas", t0)
    _mostrar("Chequeos de la simulacion", resumen_simulacion(tablas), rep / "resumen_simulacion.csv")
    if not args.sin_fuentes:
        t0 = time.perf_counter()
        (salida / "fuentes").mkdir(parents=True, exist_ok=True)
        for nombre in TABLAS_FUENTE:
            tablas[nombre].to_parquet(salida / "fuentes" / f"{nombre}.parquet", index=False)
        marca("guardar_fuentes", t0)

    # ------------------------------------------------------------------ 2. dataset as-of
    t0 = time.perf_counter()
    cortes = F.cortes_evaluables(tablas, horizonte=HORIZONTE_MESES)
    split = F.split_temporal(cortes, n_test=args.n_test, n_valid=args.n_valid, horizonte=HORIZONTE_MESES)
    usados = split["train"] + split["valid"] + split["test"]
    ds = F.construir_dataset(tablas, usados, horizonte=HORIZONTE_MESES)
    marca("construir_dataset", t0)
    ds.to_parquet(salida / "dataset.parquet", index=False)

    tr = ds[ds["periodo_corte"].isin(split["train"])]
    va = ds[ds["periodo_corte"].isin(split["valid"])]
    te = ds[ds["periodo_corte"].isin(split["test"])].copy()
    resumen_split = pd.DataFrame([
        {"particion": k, "n_cortes": len(v), "corte_min": min(v), "corte_max": max(v),
         "n_filas": int(len(d)), "tasa_base": float(d["y"].mean())}
        for k, v, d in (("train", split["train"], tr), ("valid", split["valid"], va),
                        ("test", split["test"], te))
    ])
    _mostrar("Split temporal (gap de 6 meses entre el ultimo target y el corte siguiente)",
             resumen_split, rep / "split.csv")

    # ------------------------------------------------------------------ 3. modelo
    t0 = time.perf_counter()
    modelo = M.entrenar(tr, va, seed=args.seed)
    marca("entrenar", t0)
    M.guardar_modelo(modelo, salida / "modelo")

    prob = M.predecir(modelo, te, uso="ranking")
    prob_bruta = M.predecir(modelo, te, uso="bruto")
    prob_agregado = M.predecir(modelo, te, uso="agregado")
    predicciones = {
        "modelo_lgbm": prob,
        "modelo_lgbm_sin_calibrar": prob_bruta,
        "modelo_lgbm_isotonica": prob_agregado,
        "baseline_tasa_base": M.baseline_tasa_base(tr, te),
        "baseline_regla_negocio": M.baseline_regla_negocio(te),
        "baseline_regla_graduada": M.baseline_regla_graduada(te),
    }
    te_seg = E.agregar_segmentos(te)   # `te` conserva los dtypes categoricos que necesita LightGBM

    # ------------------------------------------------------------------ 4. evaluacion
    t0 = time.perf_counter()
    principal = E.tabla_comparativa(te_seg, predicciones)
    _mostrar("Metricas en test (modelo vs baselines)", principal, rep / "metricas_test.csv")

    captura = E.captura_top(te_seg, predicciones)
    _mostrar("Metrica de negocio: captura contactando al X % de mayor score",
             captura.sort_values(["frac_contactada", "predictor"]), rep / "captura_negocio.csv")

    por_corte = E.resumen_por_corte(te_seg, prob)
    _mostrar("Modelo por corte de test", por_corte, rep / "metricas_por_corte.csv")

    seg = E.metricas_por_segmento(te_seg, predicciones)
    seg.to_csv(rep / "metricas_por_segmento.csv", index=False)
    comp = E.comparar_segmentos(seg, "modelo_lgbm", "baseline_regla_graduada", "auc")
    grandes = comp[comp["segmento_grande"]].copy()
    _mostrar("AUC por segmento: modelo vs mejor baseline de reglas (segmentos >= 5 % de las filas)",
             grandes[["segmento", "valor", "n", "frac_filas", "auc", "auc_baseline", "dif"]],
             rep / "segmentos_modelo_vs_baseline.csv")
    perdidos = grandes[grandes["dif"] < 0]
    if len(perdidos):
        _mostrar("ATENCION: segmentos grandes donde el modelo NO supera al baseline", perdidos,
                 rep / "segmentos_perdidos.csv")

    calibs = []
    for etiqueta, p in (("sin_calibrar", prob_bruta), ("platt_ranking", prob),
                        ("isotonica_agregado", prob_agregado)):
        c = E.curva_calibracion(te_seg["y"].to_numpy(), p)
        c.insert(0, "version", etiqueta)
        calibs.append(c)
    _mostrar("Calibracion por decil de score", pd.concat(calibs, ignore_index=True),
             rep / "calibracion.csv", max_filas=40)

    riesgo = pd.concat([
        E.saldo_en_riesgo(te_seg, prob_bruta).assign(version="sin_calibrar"),
        E.saldo_en_riesgo(te_seg, prob).assign(version="platt_ranking"),
        E.saldo_en_riesgo(te_seg, prob_agregado).assign(version="isotonica_agregado"),
    ], ignore_index=True)
    _mostrar("Saldo en riesgo esperado vs real (test completo)", riesgo, rep / "saldo_en_riesgo.csv")

    _mostrar("Importancia de features (gain)", M.importancias(modelo, 25), rep / "importancias.csv")
    _mostrar("Casos explicados (5 afiliados representativos del test)",
             M.explicar_casos(modelo, te, prob, n_casos=5), rep / "casos_explicados.csv", max_filas=30)
    marca("evaluar", t0)

    tiempos.append({"paso": "TOTAL", "segundos": round(time.perf_counter() - t_ini, 2)})
    _mostrar("Tiempos", pd.DataFrame(tiempos), rep / "tiempos.csv")
    print(f"\nSalidas en {salida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
