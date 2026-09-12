"""Demo end-to-end de `riesgo_cese_cotizacion`.

    python -m afp_lab.riesgo_cese_cotizacion.run --n-afiliados 20000

Genera el universo sintetico, construye features "as-of" para una ventana de cortes
mensuales, entrena los 3 modelos (cese 3 m, cese 6 m, reingreso) con calibracion
isotonica, los compara contra los baselines obligatorios y guarda tablas de resultados,
modelos e importancias en `--salida` (por defecto `salidas/riesgo_cese_cotizacion/`).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import (agregar_tramos, cobertura_poblacion_cese, curva_calibracion,
                       diagnostico_cese_por_tipo_afiliado, ece, es_calibracion_monotona, evaluar,
                       evaluar_segmentos_multi, metrica_negocio_agregado, metrica_negocio_captura)
from .features import TIPOS_DEPENDENCIA, TIPOS_EXCLUIDOS, VENTANA_HISTORIA, agregar_targets, construir_features
from .model import (ModeloCese, baseline_tasa_base, predecir_regla_negocio_cese,
                    predecir_regla_negocio_reingreso, predecir_tasa_base)
from .synthetic import add_months, generar_datos, idx_to_periodo, periodo_to_idx

N_TEST = 3        # ultimos cortes reservados para test (CONVENCIONES: "ultimos 3 cortes")
N_CORTES_TOPE = 40   # techo de cortes a usar (control de runtime); normalmente no se alcanza
FRAC_VAL = 0.25   # fraccion (de los cortes de train+val) usada para calibrar
HORIZONTE_MAX = 6  # el mayor horizonte de los 3 targets (y6, y_reingreso)

TARGETS = {
    "y3": {"horizonte": 3, "poblacion": "pop_cese", "nombre": "Cese 3 meses"},
    "y6": {"horizonte": 6, "poblacion": "pop_cese", "nombre": "Cese 6 meses"},
    "y_reingreso": {"horizonte": 6, "poblacion": "pop_reingreso", "nombre": "Reingreso"},
}

RUBROS_ESTACIONALES = ["A_AGRICULTURA", "G_COMERCIO", "P_EDUCACION", "I_TURISMO", "F_CONSTRUCCION"]


def _cortes_validos(periodo_inicio: int, periodo_fin: int) -> list[int]:
    """Cortes mensuales para los que la ventana de historia (`VENTANA_HISTORIA` meses)
    cabe entera dentro de `[periodo_inicio, periodo_fin]` y el horizonte mas largo (6
    meses) queda completo antes de `periodo_fin` (menor #7, ronda 1: antes se usaban
    siempre 24 cortes fijos sin chequear que la ventana no se saliera de rango). Se toman
    TODOS los cortes validos hasta un techo (`N_CORTES_TOPE`) por control de runtime."""
    corte_min = add_months(periodo_inicio, VENTANA_HISTORIA + 1)   # deja +2 para corte_rim=corte-2
    corte_max = add_months(periodo_fin, -HORIZONTE_MAX)
    idx_min, idx_max = periodo_to_idx(corte_min), periodo_to_idx(corte_max)
    if idx_max < idx_min:
        raise ValueError(f"Rango [{periodo_inicio},{periodo_fin}] insuficiente para "
                         f"VENTANA_HISTORIA={VENTANA_HISTORIA} y horizonte maximo={HORIZONTE_MAX}.")
    idx_ini = max(idx_min, idx_max - N_CORTES_TOPE + 1)
    return [int(idx_to_periodo(i)) for i in range(idx_ini, idx_max + 1)]


def _construir_dataset(tablas: dict[str, pd.DataFrame], periodo_inicio: int, periodo_fin: int) -> tuple[pd.DataFrame, list[int]]:
    cortes = _cortes_validos(periodo_inicio, periodo_fin)
    filas = [construir_features(tablas, c) for c in cortes]
    dataset = pd.concat(filas, ignore_index=True)
    return agregar_targets(dataset, tablas["panel_mensual"], periodo_fin), cortes


def _split_temporal(dataset: pd.DataFrame, cortes: list[int], horizonte: int) -> dict[str, pd.DataFrame]:
    """Split temporal con gap >= horizonte ENTRE EL ULTIMO TARGET DE TRAIN/VAL Y EL PRIMER
    CORTE DE TEST (CONVENCIONES #3): el ultimo corte de train/val "resuelve" su target en
    `corte+horizonte`, que debe quedar al menos `horizonte` meses antes de `test_min`, es
    decir `corte <= test_min - 2*horizonte` (mayor #3, ronda 1: la version anterior solo
    exigia `corte <= test_min - horizonte - 1`, dejando un gap real de 1 mes entre el
    ULTIMO TARGET de train y el primer corte de test, no de `horizonte` meses). Test =
    ultimos N_TEST cortes; el resto, tras el gap, se reparte en train/val (val = el ultimo
    FRAC_VAL de los cortes candidatos, para calibrar)."""
    cortes_ordenados = sorted(cortes)
    test_cortes = cortes_ordenados[-N_TEST:]
    test_min_idx = periodo_to_idx(min(test_cortes))
    candidatos = [c for c in cortes_ordenados if periodo_to_idx(c) <= test_min_idx - 2 * horizonte]
    if len(candidatos) < 4:
        raise ValueError(f"Solo {len(candidatos)} cortes de train/val disponibles para horizonte="
                         f"{horizonte} (gap=2h); sube N_CORTES_TOPE o el rango de periodos simulado.")
    n_val = max(2, round(len(candidatos) * FRAC_VAL))
    train_cortes, val_cortes = candidatos[:-n_val], candidatos[-n_val:]
    return {
        "train": dataset[dataset["periodo_corte"].isin(train_cortes)],
        "val": dataset[dataset["periodo_corte"].isin(val_cortes)],
        "test": dataset[dataset["periodo_corte"].isin(test_cortes)],
        "cortes": {"train": train_cortes, "val": val_cortes, "test": test_cortes},
    }


def _regla_negocio(target: str, df: pd.DataFrame) -> np.ndarray:
    return predecir_regla_negocio_cese(df) if target != "y_reingreso" else predecir_regla_negocio_reingreso(df)


def _entrenar_y_evaluar(target: str, cfg: dict, dataset: pd.DataFrame, cortes: list[int],
                        salida: Path) -> dict:
    horiz_col = "horizonte3_completo" if cfg["horizonte"] == 3 else "horizonte6_completo"
    pob = dataset[(dataset[cfg["poblacion"]] == 1) & dataset[horiz_col]].copy()
    pob = agregar_tramos(pob)
    splits = _split_temporal(pob, cortes, cfg["horizonte"])
    train, val, test = splits["train"], splits["val"], splits["test"]

    modelo = ModeloCese.fit(train, val, target)
    modelo.save(str(salida / "modelos"))
    test = test.copy()
    test["pred_modelo"] = modelo.predict_proba(test)
    test["pred_modelo_crudo"] = modelo.predict_proba_cruda(test)

    tasas, tasa_global = baseline_tasa_base(pd.concat([train, val]), target)
    test["pred_base_tasa"] = predecir_tasa_base(tasas, tasa_global, test)
    test["pred_base_regla"] = _regla_negocio(target, test)

    predictores = {"modelo": "pred_modelo", "baseline_tasa_base": "pred_base_tasa",
                  "baseline_regla_negocio": "pred_base_regla"}
    metricas = {nombre: evaluar(test, target, col) for nombre, col in predictores.items()}
    tabla_metricas = pd.DataFrame(metricas).T.reset_index().rename(columns={"index": "predictor"})

    # menor #10 (ronda 1): metricas por segmento para los 3 predictores, no solo el modelo,
    # para poder verificar "no perder en ningun segmento grande" contra AMBOS baselines.
    segmentos = evaluar_segmentos_multi(test, target, predictores)
    calib = curva_calibracion(test, target, "pred_modelo")
    # Documentacion (ronda 2 de revision, C/D): el ECE resume la calibracion en un numero,
    # pero no dice si la curva es monotona por decil; si NO lo es (reingreso), la
    # probabilidad calibrada sirve para agregar y para rankear, no como lectura literal
    # por individuo (se documenta en README, no se fuerza una recalibracion en esta ronda).
    calib_ece = ece(calib)
    calib_monotona = es_calibracion_monotona(calib)
    # metrica de negocio: el brief pide el error MENSUAL (uno por corte de test), no un
    # promedio/total trimestral (menor #11, ronda 1); `negocio_agregado` ya es por corte.
    negocio_agregado = metrica_negocio_agregado(test, target, "pred_modelo")
    negocio_captura = metrica_negocio_captura(test, target, "pred_modelo", frac=0.10)
    importancias = modelo.importancias(20)

    n_por_split = {k: len(v) for k, v in splits.items() if k != "cortes"}
    return {
        "target": target, "nombre": cfg["nombre"], "tabla_metricas": tabla_metricas,
        "segmentos": segmentos, "calibracion": calib, "ece": calib_ece, "calibracion_monotona": calib_monotona,
        "negocio_agregado": negocio_agregado,
        "negocio_captura": negocio_captura, "importancias": importancias, "n_por_split": n_por_split,
        "cortes_split": splits["cortes"], "test_df": test, "modelo": modelo, "horiz_col": horiz_col,
    }


def _negocio_agregado_cartera_completa(dataset: pd.DataFrame, r: dict, target: str, horiz_col: str) -> pd.DataFrame:
    """Extiende `r["negocio_agregado"]` (solo dependientes/pensionados activos, la
    poblacion del modelo) sumando una proyeccion simple de independientes y voluntarios
    (excluidos del modelo: ver `TIPOS_EXCLUIDOS`) con un baseline de tasa base por
    `tipo_afiliado x mes_corte` (no por rubro: para ellos `rubro` es siempre
    "SIN_EMPLEADOR"), entrenado y evaluado con los MISMOS cortes de train/val/test que el
    modelo (`r["cortes_split"]`, sin gap extra: son solo un baseline, no un modelo que
    pueda sobreajustar).

    Mayor A (ronda 2 de revision): sin esto, el agregado de negocio solo describe el
    ~88% de los ceses reales de la cartera completa (dependientes/pensionados son ~92,5%
    de los cotizantes activos pero solo ~88% de quienes de verdad cesan, porque los
    voluntarios cesan con tasa mucho mas alta que su peso poblacional)."""
    cortes = r["cortes_split"]
    excluidos = dataset[dataset["tipo_afiliado"].isin(TIPOS_EXCLUIDOS) & (dataset["rim_l1"] > 0)
                        & dataset[horiz_col]]
    train_val_cortes = cortes["train"] + cortes["val"]
    train_ex = excluidos[excluidos["periodo_corte"].isin(train_val_cortes)]
    test_ex = excluidos[excluidos["periodo_corte"].isin(cortes["test"])].copy()

    cols_grupo = ["tipo_afiliado", "mes_corte"]
    tasas, tasa_global = baseline_tasa_base(train_ex, target, cols=cols_grupo)
    test_ex["pred_base"] = predecir_tasa_base(tasas, tasa_global, test_ex, cols=cols_grupo)

    dep = r["negocio_agregado"].set_index("periodo_corte")
    ex_agg = (test_ex.groupby("periodo_corte")
             .agg(n_esperado_excluidos=("pred_base", "sum"), n_real_excluidos=(target, "sum"),
                  n_filas_excluidos=(target, "size")))
    out = dep.join(ex_agg, how="left").fillna(0.0)
    out["n_esperado_total"] = out["n_esperado"] + out["n_esperado_excluidos"]
    out["n_real_total"] = out["n_real"] + out["n_real_excluidos"]
    out["error_relativo_total"] = ((out["n_esperado_total"] - out["n_real_total"])
                                   / out["n_real_total"].clip(lower=1e-9))
    out["cobertura_dependientes"] = out["n_real"] / out["n_real_total"].clip(lower=1e-9)
    return out.reset_index()


def _ejemplos_por_rubro(resultado_y3: dict, top: int = 5) -> pd.DataFrame:
    """Un ejemplo por rubro estacional, con su probabilidad CALIBRADA (la que se usaria en
    la practica) y sus 3 features de mayor aporte segun `pred_contrib` de LightGBM.

    Menor #12 (ronda 1): antes se ordenaba por la probabilidad CALIBRADA, que la
    calibracion isotonica suele aplanar en el extremo superior (varias filas MUY distintas
    caen en la misma meseta, p. ej. 0,8846 para varios rubros): eso hacia parecer que el
    modelo daba la misma respuesta para todo el mundo. Ahora se ordena por el puntaje
    CRUDO (continuo, sin meseta) para elegir un ejemplo genuinamente distinto por rubro, y
    se muestran ambas probabilidades por separado para que no se confundan: las
    contribuciones SHAP (`pred_contrib`) explican el puntaje crudo (en escala log-odds,
    `sum(contrib) = logit(pred_cruda)`), no la probabilidad calibrada."""
    test, modelo = resultado_y3["test_df"], resultado_y3["modelo"]
    filas = []
    for rubro in RUBROS_ESTACIONALES:
        cand = test[test["rubro"] == rubro]
        if cand.empty:
            continue
        fila = cand.sort_values("pred_modelo_crudo", ascending=False).iloc[[0]]
        filas.append(fila)
        if len(filas) >= top:
            break
    if not filas:
        return pd.DataFrame()
    ejemplos = pd.concat(filas, ignore_index=True)
    from .model import _preparar_X   # uso interno: misma matriz que ve el modelo
    X = _preparar_X(ejemplos)
    contrib = modelo.clasificador.booster_.predict(X, pred_contrib=True)
    cols = modelo.columnas
    salida = []
    for i, (_, fila) in enumerate(ejemplos.iterrows()):
        c = pd.Series(contrib[i, :-1], index=cols).sort_values(key=np.abs, ascending=False)
        top3 = ", ".join(f"{f}={v:+.3f}" for f, v in c.head(3).items())
        salida.append({"afiliado_id": fila["afiliado_id"], "rubro": fila["rubro"],
                       "periodo_corte": fila["periodo_corte"], "pred_calibrada": fila["pred_modelo"],
                       "pred_cruda": fila["pred_modelo_crudo"], "y3_real": fila["y3"],
                       "top3_aportes_log_odds": top3})
    return pd.DataFrame(salida)


def main(n_afiliados: int = 5000, periodo_inicio: int = 202201, periodo_fin: int = 202609,
        seed: int = 42, salida: str = "salidas/riesgo_cese_cotizacion") -> dict:
    t0 = time.time()
    salida_dir = Path(salida)
    (salida_dir / "reportes").mkdir(parents=True, exist_ok=True)
    (salida_dir / "modelos").mkdir(parents=True, exist_ok=True)

    print(f"Generando universo sintetico (n_afiliados={n_afiliados}, seed={seed})...")
    tablas = generar_datos(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                           periodo_fin=periodo_fin, seed=seed)
    print(f"  listo en {time.time() - t0:.1f}s")

    t1 = time.time()
    dataset, cortes = _construir_dataset(tablas, periodo_inicio, periodo_fin)
    print(f"Features construidas: {dataset.shape} en {time.time() - t1:.1f}s "
         f"({len(cortes)} cortes, {cortes[0]}..{cortes[-1]})")

    resultados = {}
    for target, cfg in TARGETS.items():
        t2 = time.time()
        r = _entrenar_y_evaluar(target, cfg, dataset, cortes, salida_dir)
        resultados[target] = r
        r["tabla_metricas"].to_csv(salida_dir / "reportes" / f"metricas_{target}.csv", index=False)
        for seg, tabla in r["segmentos"].items():
            tabla.to_csv(salida_dir / "reportes" / f"segmento_{seg}_{target}.csv", index=False)
        r["calibracion"].to_csv(salida_dir / "reportes" / f"calibracion_{target}.csv", index=False)
        r["negocio_agregado"].to_csv(salida_dir / "reportes" / f"negocio_agregado_{target}.csv", index=False)
        r["importancias"].to_csv(salida_dir / "reportes" / f"importancias_{target}.csv", index=False)
        print(f"\n=== {cfg['nombre']} ({target}) === filas train/val/test: {r['n_por_split']} "
             f"[{time.time() - t2:.1f}s]")
        print(r["tabla_metricas"].to_string(index=False))
        print(f"Calibracion: ECE={r['ece']:.4f}, monotona por decil={r['calibracion_monotona']}")
        print("Negocio - agregado por corte (test, solo poblacion del modelo):")
        print(r["negocio_agregado"].to_string(index=False))
        print(f"Negocio - top 10% de riesgo: {r['negocio_captura']}")

        if cfg["poblacion"] == "pop_cese":
            # mayor A (ronda 2): agregado extendido a toda la cartera (+ independientes/voluntarios)
            neg_total = _negocio_agregado_cartera_completa(dataset, r, target, r["horiz_col"])
            neg_total.to_csv(salida_dir / "reportes" / f"negocio_agregado_total_{target}.csv", index=False)
            r["negocio_agregado_total"] = neg_total
            print("Negocio - agregado por corte, CARTERA COMPLETA (+ independientes/voluntarios proyectados):")
            print(neg_total.to_string(index=False))
            n_dep_test = neg_total["n_real"].sum()
            n_tot_test = neg_total["n_real_total"].sum()
            print(f"Cobertura de dependientes/pensionados sobre el total de ceses reales (3 meses de "
                 f"test, agregados): {n_dep_test / n_tot_test:.4f}")

            diag = diagnostico_cese_por_tipo_afiliado(dataset, target, r["horiz_col"])
            diag.to_csv(salida_dir / "reportes" / f"diagnostico_tipo_afiliado_{target}.csv", index=False)
            print(f"Diagnostico por tipo_afiliado (dataset completo, {len(cortes)} cortes):")
            print(diag.to_string(index=False))

    ejemplos = _ejemplos_por_rubro(resultados["y3"])
    if not ejemplos.empty:
        ejemplos.to_csv(salida_dir / "reportes" / "ejemplos_interpretados.csv", index=False)
        print("\nEjemplos interpretados (cese 3 m, uno por rubro estacional):")
        print(ejemplos.to_string(index=False))

    # cobertura de la poblacion A vs. los cotizantes reales (mayor #4, ronda 1)
    cobertura = cobertura_poblacion_cese(dataset, tablas["panel_mensual"], TIPOS_DEPENDENCIA)
    cobertura.to_csv(salida_dir / "reportes" / "cobertura_poblacion_a.csv", index=False)
    print("\nCobertura de pop_cese vs. cotizantes reales (dependientes/pensionados activos), por corte:")
    print(cobertura.to_string(index=False))
    print(f"Cobertura promedio: {cobertura['cobertura'].mean():.4f} "
         f"(factor de correccion promedio para agregados: {cobertura['factor_correccion'].mean():.4f})")

    dataset.to_parquet(salida_dir / "dataset_features.parquet", index=False)
    total = time.time() - t0
    print(f"\nRuntime total: {total:.1f}s. Salidas en {salida_dir.resolve()}")
    return resultados


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-afiliados", type=int, default=5000)
    p.add_argument("--periodo-inicio", type=int, default=202201)
    p.add_argument("--periodo-fin", type=int, default=202609)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salida", type=str, default="salidas/riesgo_cese_cotizacion")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(n_afiliados=args.n_afiliados, periodo_inicio=args.periodo_inicio, periodo_fin=args.periodo_fin,
        seed=args.seed, salida=args.salida)
