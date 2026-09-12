"""Demo end-to-end del proyecto 02 (propension a traspaso).

    python -m afp_lab.churn_traspaso.run --n-afiliados 20000 --salida salidas/churn_traspaso

Genera el universo sintetico, arma el dataset as-of, entrena LightGBM con split temporal contiguo y
purgado, compara contra los baselines obligatorios y mide el techo de predictibilidad del mundo
simulado.

    # barrido de estabilidad (rangos en vez de puntos): ~3 minutos adicionales
    python -m afp_lab.churn_traspaso.run --n-afiliados 20000 \
        --semillas-estabilidad 42,123,7,2024,99
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
from .synthetic import generar_tablas, resumen_simulacion, tabla_parametros

SALIDA_POR_DEFECTO = "salidas/churn_traspaso"
SEMILLAS_ESTABILIDAD = (42, 123, 7, 2024, 99)
# `afiliados_latente_churn` NO se escribe: es latente y solo alimenta diagnosticos en memoria
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


def preparar(n_afiliados: int, seed: int, periodo_inicio: int, periodo_fin: int,
             n_test: int, n_valid: int) -> tuple[dict, dict, pd.DataFrame, dict]:
    """Genera las tablas, arma el dataset as-of y devuelve las particiones del split temporal."""
    tablas = generar_tablas(n_afiliados=n_afiliados, periodo_inicio=periodo_inicio,
                            periodo_fin=periodo_fin, seed=seed)
    cortes = F.cortes_evaluables(tablas, horizonte=HORIZONTE_MESES)
    split = F.split_temporal(cortes, n_test=n_test, n_valid=n_valid, horizonte=HORIZONTE_MESES)
    ds = F.construir_dataset(tablas, split["train"] + split["valid"] + split["test"],
                             horizonte=HORIZONTE_MESES)
    partes = {k: ds[ds["periodo_corte"].isin(split[k])] for k in ("train", "valid", "test")}
    return tablas, split, ds, partes


def resumen_corrida(tablas: dict, split: dict, partes: dict, seed: int) -> dict:
    """Metricas clave de una corrida completa (una fila del barrido de semillas)."""
    tr, va, te = partes["train"], partes["valid"], partes["test"]
    modelo = M.entrenar(tr, va, seed=seed)
    prob = M.predecir(modelo, te, uso="ranking")
    prob_agg = M.predecir(modelo, te, uso="agregado")
    prob_bruta = M.predecir(modelo, te, uso="bruto")
    m = E.tabla_comparativa(te, {"m": prob}).iloc[0]
    preds = {"modelo": prob, "regla": M.baseline_regla_negocio(te),
             "regla_graduada": M.baseline_regla_graduada(te)}
    cap = E.captura_top(te, preds, fracciones=(0.10,)).set_index("predictor")
    por_corte = E.saldo_en_riesgo_por_corte(te, prob_agg)
    tasas = resumen_simulacion(tablas).set_index("indicador")["valor"]
    return {
        "seed": seed,
        "tasa_base_valid": float(va["y"].mean()),
        "tasa_base_test": float(te["y"].mean()),
        "auc": float(m["auc"]),
        "average_precision": float(m["average_precision"]),
        "brier": float(m["brier"]),
        "lift_decil_1": float(m["lift_decil_1"]),
        "ks": float(m["ks"]),
        "captura_10_modelo": float(cap.loc["modelo", "pct_traspasos_capturados"]),
        "captura_10_regla": float(cap.loc["regla", "pct_traspasos_capturados"]),
        "captura_10_regla_graduada": float(cap.loc["regla_graduada", "pct_traspasos_capturados"]),
        "captura_10_saldo_modelo": float(cap.loc["modelo", "pct_saldo_riesgo_capturado"]),
        "razon_captura_modelo_vs_mejor_regla": float(
            cap.loc["modelo", "pct_traspasos_capturados"]
            / max(cap.loc["regla", "pct_traspasos_capturados"],
                  cap.loc["regla_graduada", "pct_traspasos_capturados"], 1e-9)),
        "frac_cartera_marcada_por_la_regla": float(M.baseline_regla_negocio(te).mean()),
        "sesgo_saldo_sin_calibrar": float(E.saldo_en_riesgo(te, prob_bruta)["sesgo_relativo"].iloc[0]),
        "sesgo_saldo_isotonica": float(E.saldo_en_riesgo(te, prob_agg)["sesgo_relativo"].iloc[0]),
        "sesgo_saldo_con_ancla": float(E.saldo_en_riesgo(
            te, M.ajustar_prior(prob_agg, float(te["y"].mean())))["sesgo_relativo"].iloc[0]),
        "sesgo_saldo_corte_min": float(por_corte["sesgo_relativo"].min()),
        "sesgo_saldo_corte_max": float(por_corte["sesgo_relativo"].max()),
        "tasa_traspaso_12m_min": float(tasas["tasa_traspaso_12m_min"]),
        "tasa_traspaso_12m_max": float(tasas["tasa_traspaso_12m_max"]),
        "rango_rentabilidad_pp": float(tasas["rango_rentabilidad_12m_entre_afp_mediana_pp"]),
        "cortes_train": len(split["train"]),
        "cortes_descartados": len(split["descartados"]),
    }


def techo_predictibilidad(tablas: dict, partes: dict, seed: int, auc_modelo: float) -> pd.DataFrame:
    """AUC alcanzable si el modelo pudiera ver el efecto aleatorio latente `lat_u_traspaso`.

    Es el techo del mundo simulado: lo que queda por encima de ese numero es ruido que ocurre
    DESPUES del corte (visitas, reclamos y campanas que todavia no existen cuando se calcula el
    score). La columna latente se usa SOLO para este diagnostico y nunca entra al modelo.
    """
    lat = tablas["afiliados_latente_churn"].set_index("afiliado_id")["lat_u_traspaso"]
    partes2 = {}
    for k, d in partes.items():
        d = d.copy()
        d["u_latente_diagnostico"] = d["afiliado_id"].map(lat).astype("float32")
        partes2[k] = d
    # lista de columnas EXPLICITA: `columnas_features` excluye de forma dura los prefijos lat_/p_,
    # asi que el diagnostico tiene que pedir su columna a mano (y nunca puede colarse al modelo real)
    cols = F.columnas_features(partes["train"]) + ["u_latente_diagnostico"]
    modelo = M.entrenar(partes2["train"], partes2["valid"], seed=seed, columnas=cols)
    prob = M.predecir(modelo, partes2["test"])
    m = E.tabla_comparativa(partes2["test"], {"techo": prob}).iloc[0]
    return pd.DataFrame([
        {"escenario": "modelo (solo informacion observable al corte)", "auc": auc_modelo},
        {"escenario": "techo: + efecto aleatorio latente del afiliado", "auc": float(m["auc"])},
    ])


# Las 5 configuraciones que se compararon, en valores absolutos (la 4 es la que quedo por defecto
# en `model.PARAMS_LGBM`). Se eligen por AUC de VALIDACION; el AUC de test se reporta solo para que
# se vea que la eleccion no se hizo con test.
GRID_HIPERPARAMETROS = (
    {"learning_rate": 0.05, "num_leaves": 48, "min_data_in_leaf": 200, "feature_fraction": 0.75},
    {"learning_rate": 0.03, "num_leaves": 64, "min_data_in_leaf": 200, "feature_fraction": 0.75},
    {"learning_rate": 0.05, "num_leaves": 48, "min_data_in_leaf": 60, "feature_fraction": 0.75},
    {"learning_rate": 0.03, "num_leaves": 48, "min_data_in_leaf": 60, "feature_fraction": 0.75},
    {"learning_rate": 0.03, "num_leaves": 96, "min_data_in_leaf": 60, "feature_fraction": 0.60},
)
VENTANAS_CALIBRACION = (3, 9, 12)


def experimento_hiperparametros(partes: dict, seed: int) -> pd.DataFrame:
    """Las 5 configuraciones comparadas, elegidas por AUC de VALIDACION (el test se reporta aparte)."""
    from sklearn.metrics import roc_auc_score

    tr, va, te = partes["train"], partes["valid"], partes["test"]
    filas = []
    for g in GRID_HIPERPARAMETROS:
        base = dict(M.PARAMS_LGBM)
        base.update(g)
        m = M.entrenar(tr, va, seed=seed, params=g)
        filas.append({
            "configuracion": ", ".join(f"{k}={v}" for k, v in g.items()),
            "es_default_actual": g == {k: M.PARAMS_LGBM[k] for k in g},
            "learning_rate": base["learning_rate"], "num_leaves": base["num_leaves"],
            "min_data_in_leaf": base["min_data_in_leaf"], "feature_fraction": base["feature_fraction"],
            "arboles": m["mejor_iteracion"],
            "auc_valid": float(roc_auc_score(va["y"], M.predecir(m, va))),
            "auc_test": float(roc_auc_score(te["y"], M.predecir(m, te))),
        })
    d = pd.DataFrame(filas)
    d["elegida"] = d["auc_valid"] == d["auc_valid"].max()
    return d


def experimento_ventana_calibracion(n_afiliados: int, semillas: list[int], periodo_inicio: int,
                                    periodo_fin: int, n_test: int) -> pd.DataFrame:
    """Sesgo del saldo en riesgo segun el largo de la ventana de calibracion (3 / 9 / 12 cortes)."""
    filas = []
    for sd in semillas:
        for nv in VENTANAS_CALIBRACION:
            _, sp, _, pt = preparar(n_afiliados, sd, periodo_inicio, periodo_fin, n_test, nv)
            tr, va, te = pt["train"], pt["valid"], pt["test"]
            m = M.entrenar(tr, va, seed=sd)
            fila = {"seed": sd, "cortes_calibracion": nv, "cortes_train": len(sp["train"]),
                    "tasa_base_valid": float(va["y"].mean()), "tasa_base_test": float(te["y"].mean()),
                    "auc": float(E.tabla_comparativa(te, {"m": M.predecir(m, te)})["auc"].iloc[0])}
            for nom, uso in (("sin_calibrar", "bruto"), ("isotonica", "agregado")):
                p = M.predecir(m, te, uso=uso)
                fila[f"sesgo_saldo_{nom}"] = float(E.saldo_en_riesgo(te, p)["sesgo_relativo"].iloc[0])
            filas.append(fila)
    return pd.DataFrame(filas)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo de propension a traspaso a otra AFP")
    ap.add_argument("--n-afiliados", type=int, default=20_000)
    ap.add_argument("--salida", type=str, default=SALIDA_POR_DEFECTO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--periodo-inicio", type=int, default=202101)
    ap.add_argument("--periodo-fin", type=int, default=202609)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--n-valid", type=int, default=3)
    ap.add_argument("--semillas-estabilidad", type=str, default="",
                    help=f"barrido de estabilidad, p.ej. {','.join(map(str, SEMILLAS_ESTABILIDAD))}; "
                         "vacio (por defecto) lo omite para que la corrida estandar sea corta")
    ap.add_argument("--sin-fuentes", action="store_true", help="no escribir las tablas sinteticas")
    ap.add_argument("--sin-techo", action="store_true", help="omitir el diagnostico de techo")
    ap.add_argument("--experimentos", action="store_true",
                    help="reproduce la seleccion de hiperparametros y el experimento de ventana de "
                         "calibracion (reportes/hiperparametros.csv y ventana_calibracion.csv)")
    args = ap.parse_args(argv)

    salida = Path(args.salida)
    rep = salida / "reportes"
    rep.mkdir(parents=True, exist_ok=True)
    t_ini = time.perf_counter()
    tiempos: list[dict] = []

    def marca(paso: str, t0: float) -> None:
        tiempos.append({"paso": paso, "segundos": round(time.perf_counter() - t0, 2)})

    par = tabla_parametros()
    par.to_csv(rep / "parametros.csv", index=False)
    print(f"Parametros de la simulacion: {len(par)} (fuente unica ParametrosChurn -> "
          f"reportes/parametros.csv, tabla del README generada desde ahi)")

    # ------------------------------------------------------------------ 1. datos y dataset
    t0 = time.perf_counter()
    tablas, split, ds, partes = preparar(args.n_afiliados, args.seed, args.periodo_inicio,
                                         args.periodo_fin, args.n_test, args.n_valid)
    marca("generar_tablas + construir_dataset", t0)
    if not args.sin_fuentes:
        t0 = time.perf_counter()
        (salida / "fuentes").mkdir(parents=True, exist_ok=True)
        for nombre in TABLAS_FUENTE:
            tablas[nombre].to_parquet(salida / "fuentes" / f"{nombre}.parquet", index=False)
        ds.to_parquet(salida / "dataset.parquet", index=False)
        marca("guardar_fuentes", t0)

    tr, va, te = partes["train"], partes["valid"], partes["test"]
    resumen_split = pd.DataFrame(
        [{"particion": k, "n_cortes": len(split[k]), "corte_min": min(split[k]), "corte_max": max(split[k]),
          "n_filas": int(len(d)), "tasa_base": float(d["y"].mean())}
         for k, d in (("train", tr), ("valid", va), ("test", te))]
        + [{"particion": "descartados (purga)", "n_cortes": len(split["descartados"]),
            "corte_min": min(split["descartados"]) if split["descartados"] else None,
            "corte_max": max(split["descartados"]) if split["descartados"] else None,
            "n_filas": 0, "tasa_base": float("nan")}])
    _mostrar("Split temporal contiguo con purga de 6 meses entre particiones", resumen_split,
             rep / "split.csv")

    # ------------------------------------------------------------------ 2. modelo
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
    te_seg = E.agregar_segmentos(te)

    # ------------------------------------------------------------------ 3. evaluacion
    t0 = time.perf_counter()
    _mostrar("Chequeos de la simulacion y de determinismo",
             resumen_simulacion(tablas, te, prob), rep / "resumen_simulacion.csv", max_filas=30)

    principal = E.tabla_comparativa(te_seg, predicciones)
    _mostrar("Metricas en test (modelo vs baselines)", principal, rep / "metricas_test.csv")

    captura = E.captura_top(te_seg, predicciones)
    _mostrar("Metrica de negocio: captura contactando al X % de mayor score",
             captura.sort_values(["frac_contactada", "predictor"]), rep / "captura_negocio.csv")

    _mostrar("Modelo por corte de test", E.resumen_por_corte(te_seg, prob),
             rep / "metricas_por_corte.csv")

    seg = E.metricas_por_segmento(te_seg, predicciones)
    seg.to_csv(rep / "metricas_por_segmento.csv", index=False)
    comp = E.comparar_segmentos(seg, "modelo_lgbm", "baseline_regla_graduada", "auc")
    grandes = comp[comp["segmento_grande"]]
    _mostrar("AUC por segmento: modelo vs mejor baseline de reglas (segmentos >= 5 % de las filas)",
             grandes[["segmento", "valor", "n", "frac_filas", "auc", "auc_baseline", "dif",
                      "comparable", "gana_modelo"]],
             rep / "segmentos_modelo_vs_baseline.csv")
    perdidos = E.segmentos_no_superados(comp)
    if len(perdidos):
        _mostrar("ATENCION: segmentos grandes no superados (o no comparables)", perdidos,
                 rep / "segmentos_perdidos.csv")

    calibs = []
    for etiqueta, p in (("sin_calibrar", prob_bruta), ("platt_ranking", prob),
                        ("isotonica_agregado", prob_agregado)):
        c = E.curva_calibracion(te_seg["y"].to_numpy(), p)
        c.insert(0, "version", etiqueta)
        calibs.append(c)
    _mostrar("Calibracion por decil de score", pd.concat(calibs, ignore_index=True),
             rep / "calibracion.csv", max_filas=40)

    # diagnostico: cuanto del sesgo queda si el nivel agregado estuviera bien anclado (el orden del
    # score no cambia). Usa la tasa realizada del test, asi que NO es una salida de produccion.
    prob_anclado = M.ajustar_prior(prob_agregado, float(te["y"].mean()))
    riesgo = pd.concat([
        E.saldo_en_riesgo(te_seg, prob_bruta).assign(version="sin_calibrar"),
        E.saldo_en_riesgo(te_seg, prob).assign(version="platt_ranking"),
        E.saldo_en_riesgo(te_seg, prob_agregado).assign(version="isotonica_agregado"),
        E.saldo_en_riesgo(te_seg, prob_anclado).assign(version="isotonica+ancla (diagnostico)"),
    ], ignore_index=True)
    _mostrar("Saldo en riesgo esperado vs real (test completo)", riesgo, rep / "saldo_en_riesgo.csv")
    _mostrar("Saldo en riesgo por corte (salida isotonica)",
             E.saldo_en_riesgo_por_corte(te_seg, prob_agregado), rep / "saldo_en_riesgo_por_corte.csv")

    _mostrar("Importancia de features (gain)", M.importancias(modelo, 25), rep / "importancias.csv")
    _mostrar("Casos explicados (5 afiliados representativos del test)",
             M.explicar_casos(modelo, te, prob, n_casos=5), rep / "casos_explicados.csv", max_filas=30)
    marca("evaluar", t0)

    # ------------------------------------------------------------------ 4. techo y estabilidad
    if not args.sin_techo:
        t0 = time.perf_counter()
        auc_modelo = float(principal.set_index("predictor").loc["modelo_lgbm", "auc"])
        _mostrar("Techo de predictibilidad del mundo simulado",
                 techo_predictibilidad(tablas, partes, args.seed, auc_modelo),
                 rep / "techo_predictibilidad.csv")
        marca("techo_predictibilidad", t0)

    semillas = [int(x) for x in args.semillas_estabilidad.split(",") if x.strip()]
    if args.experimentos:
        t0 = time.perf_counter()
        _mostrar("Seleccion de hiperparametros (criterio: AUC de validacion)",
                 experimento_hiperparametros(partes, args.seed), rep / "hiperparametros.csv")
        _mostrar("Largo de la ventana de calibracion vs sesgo del saldo en riesgo",
                 experimento_ventana_calibracion(args.n_afiliados, semillas or [args.seed],
                                                 args.periodo_inicio, args.periodo_fin, args.n_test),
                 rep / "ventana_calibracion.csv", max_filas=20)
        marca("experimentos", t0)

    if semillas:
        t0 = time.perf_counter()
        filas = []
        for sd in semillas:
            if sd == args.seed:
                filas.append(resumen_corrida(tablas, split, partes, sd))
            else:
                tb, sp, _, pt = preparar(args.n_afiliados, sd, args.periodo_inicio, args.periodo_fin,
                                         args.n_test, args.n_valid)
                filas.append(resumen_corrida(tb, sp, pt, sd))
        est = pd.DataFrame(filas)
        _mostrar(f"Estabilidad sobre {len(semillas)} semillas", est, rep / "estabilidad_semillas.csv")
        rangos = est.drop(columns=["seed"]).agg(["min", "median", "max"]).T.reset_index()
        rangos.columns = ["metrica", "min", "mediana", "max"]
        _mostrar("Rangos entre semillas", rangos, rep / "estabilidad_rangos.csv", max_filas=30)
        marca("estabilidad_semillas", t0)

    tiempos.append({"paso": "TOTAL", "segundos": round(time.perf_counter() - t_ini, 2)})
    _mostrar("Tiempos", pd.DataFrame(tiempos), rep / "tiempos.csv")
    print(f"\nSalidas en {salida.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
