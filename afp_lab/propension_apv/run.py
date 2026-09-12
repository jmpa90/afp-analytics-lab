"""Demo end-to-end del proyecto 03.

    python -m afp_lab.propension_apv.run --n-afiliados 20000

Genera el universo sintetico, construye el dataset as-of de todos los cortes, entrena el
hurdle, lo compara con los baselines, calcula metricas por segmento, calibracion y metrica
de negocio, y escribe todo en `salidas/propension_apv/`.
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
from .synthetic import HORIZONTE, generar_tablas

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DEFECTO = RAIZ / "salidas" / "propension_apv"
PRIMER_CORTE = 202301          # 24 meses de historia (12 de burn-in + 12 de panel) antes del corte


def _imprimir(titulo: str, df: pd.DataFrame, n: int | None = None) -> None:
    print(f"\n=== {titulo} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print((df.head(n) if n else df).to_string(index=False))


def _guardar(df: pd.DataFrame, ruta: Path, nombre: str) -> None:
    ruta.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta / f"{nombre}.csv", index=False)


def ejecutar(n_afiliados: int = 5_000, salida: Path = SALIDA_DEFECTO, seed: int = 42,
             primer_corte: int = PRIMER_CORTE, n_test: int = 3, n_valid: int = 3) -> dict:
    t0 = time.perf_counter()
    salida = Path(salida)
    tiempos: dict[str, float] = {}

    # ------------------------------------------------------------------ 1. datos sinteticos
    tablas = generar_tablas(n_afiliados=n_afiliados, seed=seed)
    tiempos["generacion"] = time.perf_counter() - t0
    fuentes = salida / "fuentes"
    fuentes.mkdir(parents=True, exist_ok=True)
    for nombre, df in tablas.items():
        df.to_parquet(fuentes / f"{nombre}.parquet", index=False)

    resumen_fuentes = pd.DataFrame(
        [{"tabla": k, "filas": len(v), "columnas": v.shape[1]} for k, v in tablas.items()])

    # ------------------------------------------------------------------ 2. dataset as-of
    t = time.perf_counter()
    denso = F.preparar_panel(tablas)
    cortes = F.cortes_disponibles(tablas, primer_corte, HORIZONTE)
    split = F.split_temporal(cortes, n_test=n_test, n_valid=n_valid, gap=HORIZONTE,
                             horizonte=HORIZONTE)
    datos = F.construir_dataset(tablas, split["train"] + split["valid"] + split["test"],
                                HORIZONTE, denso=denso)
    train = datos[datos["periodo_corte"].isin(split["train"])].reset_index(drop=True)
    valid = datos[datos["periodo_corte"].isin(split["valid"])].reset_index(drop=True)
    test = datos[datos["periodo_corte"].isin(split["test"])].reset_index(drop=True)
    tiempos["dataset"] = time.perf_counter() - t

    resumen_split = pd.DataFrame([
        {"particion": n, "cortes": f"{min(c)}-{max(c)}", "n_cortes": len(c), "filas": len(d),
         "tasa_aporte": float(d["y"].mean()), "monto_total": float(d["monto"].sum())}
        for n, c, d in (("train", split["train"], train), ("valid", split["valid"], valid),
                        ("test", split["test"], test))]
        + [{"particion": "descartados_por_gap", "cortes": (f"{min(split['descartados'])}-"
                                                           f"{max(split['descartados'])}"),
            "n_cortes": len(split["descartados"]), "filas": 0, "tasa_aporte": np.nan,
            "monto_total": np.nan}])

    # ------------------------------------------------------------------ 3. modelo
    t = time.perf_counter()
    modelo = M.entrenar(train, valid, M.ParametrosModelo(seed=seed))
    tiempos["entrenamiento"] = time.perf_counter() - t
    M.guardar(modelo, salida / "modelo")

    pred = {
        "modelo": M.predecir(modelo, test),
        "tasa_base": M.baseline_tasa_base(train, test),
        "regla_negocio": M.baseline_regla(train, test),
    }
    # ------------------------------------------------------------------ 4. evaluacion
    t = time.perf_counter()
    clasificacion = ev.tabla_clasificacion(test, pred)
    monto = ev.tabla_monto(test, pred)

    segmentos = {}
    for seg in ("tramo_renta", "tramo_edad", "tipo_afiliado", "rubro"):
        segmentos[seg] = ev.comparar_segmentos(test, pred, seg)
    monto_seg = ev.monto_por_segmento(test, pred["modelo"], "tramo_renta")

    calib = ev.curva_calibracion(test["y"].to_numpy(), pred["modelo"]["prob_aporte"].to_numpy())
    decision_calib = pd.DataFrame([{
        "brier_valid_crudo": modelo.brier_valid_crudo,
        "brier_valid_isotonica_cv": modelo.brier_valid_calibrado,
        "aplica_isotonica": bool(modelo.usa_calibracion)}])
    resumen_calib = pd.DataFrame([
        {"version": "sin_calibrar", **ev.resumen_calibracion(test["y"].to_numpy(),
                                                             pred["modelo"]["prob_cruda"].to_numpy())},
        {"version": "isotonica", **ev.resumen_calibracion(test["y"].to_numpy(),
                                                          pred["modelo"]["prob_aporte"].to_numpy())},
    ])

    scores = {
        "modelo_monto_esperado": pred["modelo"]["monto_esperado"].to_numpy(),
        "modelo_probabilidad": pred["modelo"]["prob_aporte"].to_numpy(),
        "regla_negocio": pred["regla_negocio"]["monto_esperado"].to_numpy(),
        "renta": test["rim_ref"].to_numpy(),
        "aleatorio": np.random.default_rng(seed).random(len(test)),
    }
    negocio = ev.metrica_negocio(test, scores)
    resumen_neg = ev.resumen_negocio(negocio)
    aceptacion = ev.chequear_aceptacion(clasificacion, monto, negocio)

    imp = M.importancias(modelo)
    ejemplos = M.explicar(modelo, test, n_ejemplos=5, seed=seed)
    tiempos["evaluacion"] = time.perf_counter() - t
    tiempos["total"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ 5. salidas
    rep = salida / "reportes"
    _guardar(resumen_fuentes, rep, "resumen_fuentes")
    _guardar(resumen_split, rep, "resumen_split")
    _guardar(clasificacion, rep, "metricas_clasificacion")
    _guardar(monto, rep, "metricas_monto")
    _guardar(monto_seg, rep, "metricas_monto_por_tramo_renta")
    for seg, tabla in segmentos.items():
        _guardar(tabla, rep, f"metricas_por_{seg}")
    _guardar(calib, rep, "curva_calibracion")
    _guardar(resumen_calib, rep, "resumen_calibracion")
    _guardar(decision_calib, rep, "decision_calibracion")
    _guardar(negocio, rep, "metrica_negocio")
    _guardar(resumen_neg, rep, "metrica_negocio_resumen")
    _guardar(aceptacion, rep, "aceptacion")
    _guardar(imp, rep, "importancias")
    _guardar(ejemplos, rep, "ejemplos_explicados")
    _guardar(pd.DataFrame([tiempos]), rep, "tiempos")

    (salida / "dataset").mkdir(parents=True, exist_ok=True)
    for nombre, df in (("train", train), ("valid", valid), ("test", test)):
        df.to_parquet(salida / "dataset" / f"{nombre}.parquet", index=False)
    test.assign(**{c: pred["modelo"][c] for c in
                   ("prob_aporte", "monto_condicional", "monto_esperado")}
                )[["afiliado_id", "periodo_corte", "y", "monto", "prob_aporte",
                   "monto_condicional", "monto_esperado", "tramo_renta", "tramo_edad"]
                  ].to_parquet(salida / "predicciones_test.parquet", index=False)

    # ------------------------------------------------------------------ 6. impresion
    _imprimir("Tablas sinteticas", resumen_fuentes)
    _imprimir("Split temporal (gap de 6 meses entre el ultimo target de train y el test)",
              resumen_split)
    _imprimir("Clasificacion en test: modelo vs baselines", clasificacion)
    _imprimir("Monto en test", monto)
    for seg, tabla in segmentos.items():
        _imprimir(f"AUC por {seg} (test)", tabla)
    _imprimir("Monto condicional por tramo de renta (aportantes de test)", monto_seg)
    _imprimir("Decision de calibracion (tomada en validacion, con 3 folds)", decision_calib)
    _imprimir("Calibracion de la probabilidad en test", resumen_calib)
    _imprimir("Curva de calibracion (deciles de probabilidad)", calib)
    _imprimir("Metrica de negocio: captura contactando al 10 % y 20 %", negocio)
    _imprimir("Metrica de negocio: modelo vs orden por renta", resumen_neg)
    _imprimir("Importancias (top 15 del clasificador)",
              imp[imp["etapa"] == "clasificacion"].head(15))
    _imprimir("Importancias (top 10 del regresor de monto)",
              imp[imp["etapa"] == "monto"].head(10))
    _imprimir("Ejemplos explicados (SHAP del clasificador)", ejemplos)
    _imprimir("Criterios de aceptacion del brief", aceptacion)
    print("\n=== Tiempos (s) ===")
    for k, v in tiempos.items():
        print(f"{k:>16}: {v:8.1f}")
    print(f"\nSalidas en {salida}")

    return {"modelo": modelo, "clasificacion": clasificacion, "monto": monto, "negocio": negocio,
            "aceptacion": aceptacion, "tiempos": tiempos, "split": split,
            "datos": {"train": train, "valid": valid, "test": test}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Demo de propension a APV / cotizacion voluntaria")
    ap.add_argument("--n-afiliados", type=int, default=5_000)
    ap.add_argument("--salida", type=Path, default=SALIDA_DEFECTO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primer-corte", type=int, default=PRIMER_CORTE)
    args = ap.parse_args(argv)
    ejecutar(n_afiliados=args.n_afiliados, salida=args.salida, seed=args.seed,
             primer_corte=args.primer_corte)


if __name__ == "__main__":
    main()
