"""Demo end-to-end del proyecto 03.

    python -m afp_lab.propension_apv.run --n-afiliados 20000

Genera el universo sintetico, construye el dataset as-of de todos los cortes, entrena el
hurdle, lo compara con los baselines, calcula metricas por segmento y por corte, calibracion,
sesgo del monto por decil, diagnostico de memorizacion y metrica de negocio, y escribe todo en
`salidas/propension_apv/<perfil>/`.

`--perfil` elige la cartera del universo comun (ver `docs/contexto_afp_cuprum.md`):
`sistema` (promedio de las AFP), `cuprum` (nuestra cartera) o `ambos` (corre los dos y escribe la
tabla de comparacion `comparacion_perfiles.csv`).
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
from .synthetic import ANCLAS_FICHA_165, HORIZONTE, PERFIL_DEFECTO, generar_tablas

RAIZ = Path(__file__).resolve().parents[2]
SALIDA_DEFECTO = RAIZ / "salidas" / "propension_apv"
# primer mes de decision: necesita 24 meses de historial de aportes y 2 de desfase de panel
PRIMER_CORTE = 202101
N_TEST = 12       # CONVENCIONES §3: validar y testear con al menos 12 cortes
N_VALID = 12
CADA_ORIGEN = 3   # se reentrena cada 3 meses (origenes moviles), como en produccion


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
             cada_origen: int = CADA_ORIGEN, diagnostico: bool = True,
             perfil: str = PERFIL_DEFECTO) -> dict:
    t0 = time.perf_counter()
    salida = Path(salida)
    tiempos: dict[str, float] = {}

    # ------------------------------------------------------------------ 1. datos sinteticos
    tablas = generar_tablas(n_afiliados=n_afiliados, seed=seed, perfil=perfil)
    tiempos["generacion"] = time.perf_counter() - t0
    fuentes = salida / "fuentes"
    fuentes.mkdir(parents=True, exist_ok=True)
    for nombre, df in tablas.items():
        df.to_parquet(fuentes / f"{nombre}.parquet", index=False)

    resumen_fuentes = pd.DataFrame(
        [{"tabla": k, "filas": len(v), "columnas": v.shape[1]} for k, v in tablas.items()])
    calibracion = calibrar_contra_ficha(tablas, perfil)

    # ------------------------------------------------------------------ 2. dataset as-of
    t = time.perf_counter()
    denso = F.preparar_panel(tablas)
    cortes = F.cortes_disponibles(tablas, primer_corte, HORIZONTE)
    origenes = F.origenes_moviles(cortes, n_test=n_test, n_valid=n_valid, gap=HORIZONTE,
                                  horizonte=HORIZONTE, cada=cada_origen)
    usados = sorted({c for o in origenes for c in o["train"] + o["valid"] + o["test"]})
    datos = F.construir_dataset(tablas, usados, HORIZONTE, denso=denso)
    por_corte_idx = {c: g.index for c, g in datos.groupby("periodo_corte", observed=True)}
    tiempos["dataset"] = time.perf_counter() - t

    cortes_test = [c for o in origenes for c in o["test"]]
    test = datos.loc[np.concatenate([por_corte_idx[c] for c in cortes_test])].reset_index(drop=True)

    # ------------------------------------------------------------------ 3. modelos (origenes moviles)
    t = time.perf_counter()
    params = M.ParametrosModelo(seed=seed)
    filas_split, filas_ajuste, trozos, filas_modelos = [], [], [], []
    modelo = train_ult = valid_ult = None
    # m-1: el baseline trivial usa UNA sola constante para todo el pool de test (la tasa del
    # train del primer origen, que es la unica conocida cuando arranca la campana). Si se usara
    # la tasa de cada origen la constante cambiaria entre bloques y el "baseline trivial" se
    # convertiria en un escalon con AUC > 0,5, que no es un baseline trivial.
    train_primer_origen = None
    for o in origenes:
        tr = datos.loc[np.concatenate([por_corte_idx[c] for c in o["train"]])]
        va = datos.loc[np.concatenate([por_corte_idx[c] for c in o["valid"]])]
        te = datos.loc[np.concatenate([por_corte_idx[c] for c in o["test"]])]
        if train_primer_origen is None:
            train_primer_origen = tr
        m = M.entrenar(tr, va, params)
        n_clf, n_reg = m.n_arboles
        ruta_m = M.guardar(m, salida / "modelo" / f"origen_{o['origen']}")
        filas_modelos.append({
            "origen": o["origen"], "ruta": str(ruta_m.relative_to(salida)),
            "predice_cortes": ",".join(str(c) for c in o["test"]),
            "arboles_clf": n_clf, "arboles_reg": n_reg})
        filas_split.append({
            "origen": o["origen"],
            "train": f"{min(o['train'])}-{max(o['train'])}", "n_train": len(o["train"]),
            "valid": f"{min(o['valid'])}-{max(o['valid'])}", "n_valid": len(o["valid"]),
            "test": f"{min(o['test'])}-{max(o['test'])}", "n_test": len(o["test"]),
            "filas_train": len(tr), "filas_test": len(te),
            "tasa_train": float(tr["y"].mean()), "tasa_valid": float(va["y"].mean()),
            "tasa_test": float(te["y"].mean())})
        filas_ajuste.append({
            "origen": o["origen"], "arboles_clf": n_clf, "tope_clf": params.clf_n_estimators,
            "arboles_reg": n_reg, "tope_reg": params.reg_n_estimators,
            "early_stopping_corto": n_clf < params.clf_n_estimators,
            "smearing_valid": m.smearing, "aplica_isotonica": bool(m.usa_calibracion),
            "mejora_brier_valid": 1.0 - m.brier_valid_calibrado / max(m.brier_valid_crudo, 1e-12),
            "mejora_media_entre_folds": m.mejora_brier_media,
            "error_estandar_entre_folds": m.mejora_brier_sem})
        trozos.append({
            "modelo": M.predecir(m, te),
            "tasa_base": M.baseline_tasa_base(train_primer_origen, te),
            "regla_negocio": M.baseline_regla(tr, te)})
        modelo, train_ult, valid_ult = m, tr, va
    tiempos["entrenamiento"] = time.perf_counter() - t
    modelos_por_origen = pd.DataFrame(filas_modelos)

    pred = {k: pd.concat([z[k] for z in trozos], ignore_index=True) for k in trozos[0]}
    for k in pred:
        pred[k].index = test.index
    resumen_split = pd.DataFrame(filas_split)
    ajuste = pd.DataFrame(filas_ajuste)
    solape = F.solape_valid_pool_test(origenes, HORIZONTE)

    # ------------------------------------------------------------------ 4. evaluacion
    t = time.perf_counter()
    clasificacion = ev.tabla_clasificacion(test, pred)
    monto = ev.tabla_monto(test, pred)
    por_corte = ev.metricas_por_corte(test, pred["modelo"])
    sesgo_decil = ev.sesgo_monto_por_decil(test, pred["modelo"])

    segmentos = {}
    for seg in ("tramo_renta", "tramo_edad", "tipo_afiliado", "rubro"):
        segmentos[seg] = ev.comparar_segmentos(test, pred, seg)
    monto_seg = ev.monto_por_segmento(test, pred["modelo"], "tramo_renta")

    calib = ev.curva_calibracion(test["y"].to_numpy(), pred["modelo"]["prob_aporte"].to_numpy())
    decision_calib = ajuste[["origen", "aplica_isotonica", "mejora_brier_valid",
                             "mejora_media_entre_folds", "error_estandar_entre_folds"]].assign(
        n_sigmas_exigidas=M.N_SIGMAS_CALIBRACION,
        mejora_relativa_minima=M.MEJORA_MINIMA_CALIBRACION)
    n_cal = int(ajuste["aplica_isotonica"].sum())
    etiqueta = (f"probabilidad usada ({n_cal} de {len(ajuste)} origenes calibrados)"
                if n_cal else "probabilidad usada (ningun origen calibrado)")
    resumen_calib = pd.DataFrame([
        {"version": etiqueta, **ev.resumen_calibracion(
            test["y"].to_numpy(), pred["modelo"]["prob_aporte"].to_numpy())},
        # referencia: cada bloque calibrado con la isotonica de SU origen (m-7)
        {"version": "isotonica por origen (referencia, no se usa)", **ev.resumen_calibracion(
            test["y"].to_numpy(), pred["modelo"]["prob_isotonica"].to_numpy())},
    ])

    scores = {
        "modelo_monto_esperado": pred["modelo"]["monto_esperado"].to_numpy(),
        "modelo_probabilidad": pred["modelo"]["prob_aporte"].to_numpy(),
        "regla_negocio": pred["regla_negocio"]["monto_esperado"].to_numpy(),
        "renta": test["rim_ref_clp"].to_numpy(),
        "aleatorio": np.random.default_rng(seed).random(len(test)),
    }
    negocio = ev.metrica_negocio(test, scores)
    resumen_neg = ev.resumen_negocio(negocio)
    contexto = ev.contexto_negocio(test, tablas, HORIZONTE)
    aceptacion = ev.chequear_aceptacion(clasificacion, monto, negocio, por_corte)
    deriva = ev.deriva_features(train_primer_origen, test, M.columnas_modelo_de(modelo))
    meses_solapados = sorted({int(m) for fila in solape["meses_solapados"] if fila
                              for m in fila.split(",")})
    solape_monto = ev.solape_monto_valid_pool(test, tablas, meses_solapados, HORIZONTE)

    imp = M.importancias(modelo)
    ejemplos = M.explicar(modelo, test, n_ejemplos=5, seed=seed)
    memorizacion = (M.diagnostico_memorizacion(train_ult, valid_ult, modelo) if diagnostico
                    else pd.DataFrame())
    tiempos["evaluacion"] = time.perf_counter() - t
    tiempos["total"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ 5. salidas
    rep = salida / "reportes"
    _guardar(resumen_fuentes, rep, "resumen_fuentes")
    _guardar(calibracion, rep, "calibracion_ficha_165")
    _guardar(resumen_split, rep, "resumen_split")
    _guardar(modelos_por_origen, rep, "modelos_por_origen")
    _guardar(solape, rep, "solape_valid_pool_test")
    _guardar(solape_monto, rep, "solape_monto_valid_pool_test")
    _guardar(deriva, rep, "deriva_features")
    _guardar(ajuste, rep, "ajuste_modelo")
    _guardar(clasificacion, rep, "metricas_clasificacion")
    _guardar(monto, rep, "metricas_monto")
    _guardar(por_corte, rep, "metricas_por_corte")
    _guardar(sesgo_decil, rep, "sesgo_monto_por_decil")
    _guardar(monto_seg, rep, "metricas_monto_por_tramo_renta")
    for seg, tabla in segmentos.items():
        _guardar(tabla, rep, f"metricas_por_{seg}")
    _guardar(calib, rep, "curva_calibracion")
    _guardar(resumen_calib, rep, "resumen_calibracion")
    _guardar(decision_calib, rep, "decision_calibracion")
    _guardar(negocio, rep, "metrica_negocio")
    _guardar(resumen_neg, rep, "metrica_negocio_resumen")
    _guardar(contexto, rep, "contexto_negocio")
    _guardar(aceptacion, rep, "aceptacion")
    _guardar(imp, rep, "importancias")
    _guardar(ejemplos, rep, "ejemplos_explicados")
    if len(memorizacion):
        _guardar(memorizacion, rep, "diagnostico_memorizacion")
    _guardar(pd.DataFrame([tiempos]), rep, "tiempos")

    (salida / "dataset").mkdir(parents=True, exist_ok=True)
    for nombre, df in (("train_ultimo_origen", train_ult), ("valid_ultimo_origen", valid_ult),
                       ("test", test)):
        df.reset_index(drop=True).to_parquet(salida / "dataset" / f"{nombre}.parquet", index=False)
    test.assign(**{c: pred["modelo"][c] for c in
                   ("prob_aporte", "monto_condicional", "monto_esperado")}
                )[["afiliado_id", "periodo_corte", "y", "monto", "prob_aporte",
                   "monto_condicional", "monto_esperado", "tramo_renta", "tramo_edad"]
                  ].to_parquet(salida / "predicciones_test.parquet", index=False)

    # ------------------------------------------------------------------ 6. impresion
    print(f"\n### PERFIL: {perfil} ###")
    _imprimir("Tablas sinteticas", resumen_fuentes)
    _imprimir("Calibracion contra la Ficha Estadistica Previsional N 165 (SP, ago 2026)", calibracion)
    _imprimir("Origenes moviles (gap de 6 meses entre cada par de particiones)", resumen_split)
    _imprimir("Ajuste por origen (arboles elegidos por early stopping contra validacion)", ajuste)
    _imprimir("Modelos persistidos (uno por origen; el .pkl de un origen NO reproduce el agregado)",
              modelos_por_origen)
    _imprimir("Solape de meses de target: valid de cada origen vs su test y vs el pool", solape)
    _imprimir("Monto del pool de test en los meses solapados (limitacion conocida m-2)", solape_monto)
    if len(memorizacion):
        _imprimir("Diagnostico de memorizacion por afiliado", memorizacion)
    _imprimir("Clasificacion en test: modelo vs baselines", clasificacion)
    _imprimir("Clasificacion y monto por corte de test", por_corte)
    _imprimir("Monto en test", monto)
    _imprimir("Sesgo del monto esperado por decil", sesgo_decil)
    for seg, tabla in segmentos.items():
        _imprimir(f"AUC por {seg} (test)", tabla)
    _imprimir("Monto condicional por tramo de renta (aportantes de test)", monto_seg)
    _imprimir("Decision de calibracion (tomada en validacion con gap, 3 folds)", decision_calib)
    _imprimir("Calibracion de la probabilidad en test", resumen_calib)
    _imprimir("Curva de calibracion (deciles de probabilidad)", calib)
    _imprimir("Metrica de negocio: captura contactando al 10 % y 20 %", negocio)
    _imprimir("Metrica de negocio: modelo vs orden por renta", resumen_neg)
    _imprimir("Contexto: cuanto del flujo total es la poblacion de adquisicion", contexto)
    _imprimir("Importancias (top 15 del clasificador)",
              imp[imp["etapa"] == "clasificacion"].head(15))
    _imprimir("Importancias (top 10 del regresor de monto)",
              imp[imp["etapa"] == "monto"].head(10))
    _imprimir("Ejemplos explicados (SHAP del clasificador)", ejemplos)
    _imprimir("Deriva de nivel train (primer origen) -> test en las features numericas", deriva)
    _imprimir("Criterios de aceptacion del brief (se evaluan sobre el agregado)", aceptacion)
    print("\n=== Tiempos (s) ===")
    for k, v in tiempos.items():
        print(f"{k:>16}: {v:8.1f}")
    print(f"\nSalidas en {salida}")

    return {"modelo": modelo, "clasificacion": clasificacion, "monto": monto, "negocio": negocio,
            "aceptacion": aceptacion, "tiempos": tiempos, "origenes": origenes,
            "por_corte": por_corte, "sesgo_decil": sesgo_decil, "memorizacion": memorizacion,
            "perfil": perfil, "calibracion": calibracion, "resumen_split": resumen_split,
            "contexto": contexto, "resumen_calib": resumen_calib, "negocio_resumen": resumen_neg,
            "ajuste": ajuste, "pred": pred, "solape": solape, "solape_monto": solape_monto,
            "deriva": deriva, "modelos_por_origen": modelos_por_origen,
            "datos": {"train": train_ult, "valid": valid_ult, "test": test}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Demo de propension a APV / cotizacion voluntaria")
    ap.add_argument("--n-afiliados", type=int, default=5_000)
    ap.add_argument("--salida", type=Path, default=SALIDA_DEFECTO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--primer-corte", type=int, default=PRIMER_CORTE)
    ap.add_argument("--perfil", choices=("sistema", "cuprum", "ambos"), default=PERFIL_DEFECTO,
                    help="cartera del universo comun; 'ambos' corre los dos y compara")
    ap.add_argument("--cada-origen", type=int, default=CADA_ORIGEN,
                    help="meses entre reentrenamientos (origenes moviles); subirlo acorta el runtime")
    ap.add_argument("--sin-diagnostico", action="store_true",
                    help="omite la CV agrupada por afiliado (diagnostico de memorizacion)")
    args = ap.parse_args(argv)
    perfiles = ("sistema", "cuprum") if args.perfil == "ambos" else (args.perfil,)
    resultados = {}
    for perfil in perfiles:
        resultados[perfil] = ejecutar(
            n_afiliados=args.n_afiliados, salida=Path(args.salida) / perfil, seed=args.seed,
            primer_corte=args.primer_corte, cada_origen=args.cada_origen,
            diagnostico=not args.sin_diagnostico, perfil=perfil)
    if len(resultados) > 1:
        comp = comparar_perfiles(resultados)
        Path(args.salida).mkdir(parents=True, exist_ok=True)
        comp.to_csv(Path(args.salida) / "comparacion_perfiles.csv", index=False)
        largo = comp.set_index("perfil").T.reset_index().rename(columns={"index": "indicador"})
        largo.to_csv(Path(args.salida) / "comparacion_perfiles_vertical.csv", index=False)
        _imprimir("Comparacion de perfiles (sistema vs cuprum)", largo)
        print(f"\nComparacion en {Path(args.salida) / 'comparacion_perfiles.csv'}")



# =========================================================================== perfiles
def calibrar_contra_ficha(tablas: dict[str, pd.DataFrame], perfil: str) -> pd.DataFrame:
    """Compara el universo simulado con las anclas de la Ficha Estadistica Previsional N 165.

    Devuelve una fila por indicador con el valor simulado, el valor oficial y el objetivo del
    perfil, de modo que la calibracion quede auditable. Se mide sobre los tres ultimos anos
    calendario completos de la ventana simulada.
    """
    from .synthetic import ANCLAS_FICHA_165 as A
    from .synthetic import CUENTAS_APV_POR_COTIZANTE_SISTEMA as TECHO

    panel = tablas["panel_mensual"]
    panel = panel.assign(rim=panel["rim_emp1"] + panel["rim_emp2"], anio=panel["periodo"] // 100)
    ap = tablas["aportes_voluntarios"]
    ap = ap.assign(anio=ap["periodo"] // 100)
    anios = [a for a in (2023, 2024, 2025) if a in set(panel["anio"])]

    tasas, mensuales = [], []
    for a in anios:
        cot = set(panel[(panel["anio"] == a) & (panel["rim"] > 0)]["afiliado_id"])
        x = ap[ap["anio"] == a]
        if not cot:
            continue
        tasas.append(len(set(x["afiliado_id"]) & cot) / len(cot))
        mensuales.append(len(set(x[x["periodicidad"] == "MENSUAL"]["afiliado_id"]) & cot) / len(cot))
    tasa = float(np.mean(tasas)) if tasas else np.nan
    mensual = float(np.mean(mensuales)) if mensuales else np.nan

    ult = panel[panel["anio"] == max(anios)] if anios else panel
    rim_media = float(ult.loc[ult["rim"] > 0, "rim"].mean())
    apv = ap[ap["tipo"].isin(["APV_REGIMEN_A", "APV_REGIMEN_B"])]
    share_a = float((apv["tipo"] == "APV_REGIMEN_A").mean()) if len(apv) else np.nan

    objetivo_rim = (A["ingreso_imponible_promedio_cuprum"] if perfil == "cuprum"
                    else A["ingreso_imponible_promedio_sistema"])
    filas = [
        {"indicador": "ingreso imponible promedio de cotizantes (CLP)",
         "simulado": rim_media, "oficial_ficha_165": objetivo_rim,
         "nivel_del_dato_oficial": f"{perfil} (Ficha §0)",
         "comentario": f"ratio simulado/oficial {rim_media / objetivo_rim:.2f}"},
        {"indicador": "% de depositos en regimen A",
         "simulado": share_a, "oficial_ficha_165": A["share_depositos_regimen_a"],
         "nivel_del_dato_oficial": "sistema",
         "comentario": "mezcla A/B calibrada con esta ancla (P_REGIMEN_B_BASE / _POR_TRAMO)"},
        {"indicador": "deposito promedio (CLP)",
         "simulado": float(ap["monto"].mean()), "oficial_ficha_165": A["deposito_promedio_clp"],
         "nivel_del_dato_oficial": "sistema",
         "comentario": "brecha conocida: los aportes esporadicos de la simulacion son grandes"},
        {"indicador": "deposito mediano (CLP)",
         "simulado": float(ap["monto"].median()), "oficial_ficha_165": A["deposito_promedio_clp"],
         "nivel_del_dato_oficial": "sistema (promedio, no mediana)",
         "comentario": "referencia, no ancla"},
        {"indicador": "cotizantes con algun aporte en el anio",
         "simulado": tasa, "oficial_ficha_165": TECHO,
         "nivel_del_dato_oficial": "sistema: cuentas APV en AFP por cotizante (techo, no tasa)",
         "comentario": "la Ficha no publica aportantes por AFP: el techo es 0,319 cuentas/cotizante"},
        {"indicador": "cotizantes con aporte mensual",
         "simulado": mensual, "oficial_ficha_165": np.nan,
         "nivel_del_dato_oficial": "no publicado",
         "comentario": "objetivo del brief: 5-8 % sistema"},
    ]
    return pd.DataFrame(filas)


def comparar_perfiles(resultados: dict[str, dict]) -> pd.DataFrame:
    """Tabla unica con los indicadores clave de cada perfil (encargo 1 del BRIEF_RONDA2)."""
    filas = []
    for perfil, r in resultados.items():
        clas = r["clasificacion"].set_index("predictor")
        monto = r["monto"].set_index(["predictor", "vista"])
        cal = r["calibracion"].set_index("indicador")["simulado"]
        neg = r["negocio"].set_index(["score", "fraccion_contactada"])
        rc = r["resumen_calib"].iloc[0]
        pc = r["por_corte"]
        test = r["datos"]["test"]
        filas.append({
            "perfil": perfil,
            # --- universo simulado
            "rim_media_cotizantes": cal["ingreso imponible promedio de cotizantes (CLP)"],
            "tasa_anual_aportantes": cal["cotizantes con algun aporte en el anio"],
            "tasa_aporte_mensual": cal["cotizantes con aporte mensual"],
            "share_regimen_a": cal["% de depositos en regimen A"],
            "deposito_medio": cal["deposito promedio (CLP)"],
            # --- dataset
            "filas_test": len(test), "tasa_base_test": float(test["y"].mean()),
            "aportantes_test": int(test["y"].sum()),
            # --- clasificacion
            "auc": clas.loc["modelo", "auc"], "auc_regla": clas.loc["regla_negocio", "auc"],
            "auc_min_corte": pc["auc"].min(), "auc_max_corte": pc["auc"].max(),
            "lift_decil_1": clas.loc["modelo", "lift_decil_1"],
            "average_precision": clas.loc["modelo", "average_precision"],
            # --- monto
            "wape_condicional": monto.loc[("modelo", "condicional_en_aportantes"), "wape"],
            "sesgo_monto_esperado": monto.loc[("modelo", "esperado_en_todos"), "sesgo_agregado"],
            # --- calibracion
            "brier": rc["brier"], "ece": rc["ece"], "sesgo_prob": rc["sesgo_agregado"],
            # --- negocio
            "captura_10pct": neg.loc[("modelo_monto_esperado", 0.10), "captura_monto"],
            "captura_10pct_renta": neg.loc[("renta", 0.10), "captura_monto"],
            "conversion_contactados_10pct": neg.loc[("modelo_monto_esperado", 0.10),
                                                    "tasa_aporte_contactados"],
            "share_adquisicion_del_flujo": float(r["contexto"]["share_adquisicion_sobre_flujo_total"].iloc[0]),
            "runtime_s": r["tiempos"]["total"],
        })
    return pd.DataFrame(filas)

if __name__ == "__main__":
    main()
