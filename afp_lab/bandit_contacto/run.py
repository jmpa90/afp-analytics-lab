"""Demo end-to-end del proyecto 09 (bandit contextual de contacto para retencion).

    PYTHONPATH=/home/user/wt/empleador python3 -m afp_lab.bandit_contacto.run \\
        --n-afiliados 10000 --seeds 42,7,123,2024,99 --salida salidas/bandit_contacto

Genera el universo sintetico (`churn_traspaso`, sin modificar) + las tablas propias de este
proyecto, arma el contexto as-of, corre la simulacion online mensual (7 politicas x N semillas x
24 meses con recompensa rezagada) y la evaluacion off-policy (IPS/SNIPS/DR vs el valor verdadero).
Imprime las tablas de resultados y las guarda en `--salida`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from . import COSTOS_BRAZOS_CLP, PRESUPUESTO_MENSUAL_CONTACTOS, TASA_COMISION_ANUAL
from . import evaluate as E
from . import model as M
from .features import cortes_evaluables
from .synthetic import generar_tablas_bandit

SALIDA_POR_DEFECTO = "salidas/bandit_contacto"
N_MESES_SIMULACION = 24
SEEDS_POR_DEFECTO = "42,7,123,2024,99"


def _mostrar(titulo: str, df: pd.DataFrame, ruta: Path | None = None, max_filas: int = 40) -> None:
    print(f"\n=== {titulo} ===")
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df.head(max_filas).to_string(index=False))
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta, index=False)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Demo del bandit contextual de contacto (proyecto 09)")
    ap.add_argument("--n-afiliados", type=int, default=10_000)
    ap.add_argument("--periodo-inicio", type=int, default=202201)
    ap.add_argument("--periodo-fin", type=int, default=202612)
    ap.add_argument("--n-meses-simulacion", type=int, default=N_MESES_SIMULACION)
    ap.add_argument("--seeds", type=str, default=SEEDS_POR_DEFECTO,
                    help="semillas de la simulacion online, separadas por coma")
    ap.add_argument("--seed-mundo", type=int, default=42,
                    help="semilla del universo/contexto/respuesta latente (fija entre semillas)")
    ap.add_argument("--perfil", type=str, default="cuprum", choices=("sistema", "cuprum"))
    ap.add_argument("--salida", type=str, default=SALIDA_POR_DEFECTO)
    args = ap.parse_args(argv)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    salida = Path(args.salida)
    t0 = time.time()

    tablas = generar_tablas_bandit(n_afiliados=args.n_afiliados, periodo_inicio=args.periodo_inicio,
                                   periodo_fin=args.periodo_fin, seed=args.seed_mundo,
                                   perfil=args.perfil)
    t1 = time.time()
    print(f"[1/4] tablas sinteticas generadas en {t1 - t0:.1f}s "
         f"({args.n_afiliados} afiliados, {args.periodo_inicio}-{args.periodo_fin})")

    cortes = cortes_evaluables(tablas)
    cortes_sim = cortes[-args.n_meses_simulacion:]
    if len(cortes_sim) < args.n_meses_simulacion:
        raise ValueError(f"solo hay {len(cortes_sim)} cortes evaluables, se pidieron "
                         f"{args.n_meses_simulacion}; ampliar --periodo-fin")

    entorno = M.preparar_entorno(tablas, cortes_sim, seed_mundo=args.seed_mundo)
    t2 = time.time()
    print(f"[2/4] contexto + respuesta latente en {t2 - t1:.1f}s "
         f"({len(entorno['contexto'])} filas afiliado x corte, {len(cortes_sim)} cortes de "
         f"{cortes_sim[0]} a {cortes_sim[-1]})")

    curvas, logueado, congeladas = M.simular_todas(entorno, cortes_sim, seeds)
    t3 = time.time()
    print(f"[3/4] simulacion online ({len(seeds)} semillas x {len(M.TODAS_LAS_POLITICAS)} "
         f"politicas x {len(cortes_sim)} meses) en {t3 - t2:.1f}s")

    resumen = E.resumen_online(curvas)
    _mostrar(f"Resumen online a {len(cortes_sim)} meses (media +/- IC95% entre {len(seeds)} semillas)",
             resumen, salida / "resumen_online.csv")

    ope = E.evaluar_off_policy(entorno, logueado, congeladas)
    _mostrar("Evaluacion off-policy: valor verdadero vs IPS/SNIPS/DR (error relativo)", ope,
             salida / "ope.csv")
    t4 = time.time()
    print(f"[4/4] evaluacion off-policy en {t4 - t3:.1f}s (log: {len(logueado)} decisiones)")

    if "THOMPSON_LINEAL" in congeladas:
        _mostrar("Interpretabilidad: pesos de Thompson lineal (top 4 |peso| por brazo)",
                 E.importancia_lineal(congeladas["THOMPSON_LINEAL"]),
                 salida / "importancia_thompson_lineal.csv")
    if "THOMPSON_BOOTSTRAP_LGBM" in congeladas:
        _mostrar("Interpretabilidad: importancia (gain) de Thompson-LGBM",
                 E.importancia_lgbm(congeladas["THOMPSON_BOOTSTRAP_LGBM"]),
                 salida / "importancia_thompson_lgbm.csv")
    _mostrar("Ejemplos individuales (politica de logging)", E.ejemplos_individuales(entorno, logueado),
             salida / "ejemplos_individuales.csv")

    salida.mkdir(parents=True, exist_ok=True)
    curvas.to_csv(salida / "curvas_online.csv", index=False)
    logueado.to_csv(salida / "logueado.csv", index=False)

    print(f"\nsupuestos economicos: costos por brazo (CLP) {COSTOS_BRAZOS_CLP}, "
         f"tasa de comision anual {TASA_COMISION_ANUAL}, "
         f"presupuesto mensual de contactos {PRESUPUESTO_MENSUAL_CONTACTOS}")
    print(f"\nruntime total: {time.time() - t0:.1f}s -- salidas en {salida}/")
    return {"resumen_online": resumen, "ope": ope, "curvas": curvas, "logueado": logueado}


if __name__ == "__main__":
    main()
