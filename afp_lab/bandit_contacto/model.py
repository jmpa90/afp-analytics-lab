"""Politicas de contacto y simulacion online mensual con recompensa rezagada.

`preparar_entorno` precomputa UNA vez (para todas las semillas y politicas) todo lo que es exogeno
a la simulacion: contexto as-of, matriz numerica estandarizada, respuesta latente por brazo,
comision esperada y el pool ordenado por riesgo de cada corte. Lo unico que cambia por semilla es
el azar de exploracion/observacion (que brazo elige cada politica, si el afiliado traspasa dado el
brazo). Esto reduce mucho el costo: generar el universo y el contexto es lo caro (una vez), simular
7 politicas x 5 semillas x 24 meses es aritmetica de arrays chicos (pool <= `PRESUPUESTO_MENSUAL_CONTACTOS`).

Restricciones del brief aplicadas en `seleccionar_pool`: presupuesto mensual de contactos (el pool
"en riesgo" de cada mes es, por diseno, el top-N por `score_riesgo` -- ver README "Decisiones
tomadas"), maximo un contacto pagado cada `MESES_ENTRE_CONTACTOS` y exclusion de quienes pidieron no
ser contactados (`preferencias_contacto`).

Recompensa rezagada: el resultado (traspasa o no en 6 meses) se sortea EN EL MOMENTO de la decision
con la probabilidad verdadera (`lat_p_churn_<brazo>`) -- el simulador/oraculo lo sabe de inmediato,
por eso "recompensa neta acumulada" y "regret" se pueden acumular mes a mes sin esperar --, pero la
politica solo recibe esa observacion (`actualizar`) `HORIZONTE_MESES` despues, como en la realidad.
"""
from __future__ import annotations

import zlib
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

from afp_lab.rim_nowcasting.calendario import periodo_to_idx

from . import (BRAZOS, COSTOS_BRAZOS_CLP, EPSILON_LOGGING, HORIZONTE_MESES, IDX_BRAZO,
              MESES_ENTRE_CONTACTOS, N_BRAZOS, PRESUPUESTO_MENSUAL_CONTACTOS, TASA_COMISION_ANUAL)
from .features import CONTEXTO_NUMERICO, ajustar_escalador, construir_contexto, transformar_contexto
from .synthetic import calcular_respuesta_latente, generar_tipo_respuesta

COSTO_VEC = np.array([COSTOS_BRAZOS_CLP[b] for b in BRAZOS], dtype=np.float64)


def _hash_estable(s: str) -> int:
    """Hash determinista (no depende de PYTHONHASHSEED) para derivar semillas por politica."""
    return zlib.crc32(s.encode("utf-8")) % 997


IDX_VISITA = IDX_BRAZO["VISITA"]

BASELINES = ("SIN_CONTACTO", "TOPK_RIESGO_VISITA")
LOGGING = "HEURISTICA_LOGGING"
ALGORITMOS = ("EPSILON_GREEDY", "LINUCB", "THOMPSON_LINEAL", "THOMPSON_BOOTSTRAP_LGBM")
TODAS_LAS_POLITICAS = BASELINES + (LOGGING,) + ALGORITMOS


# --------------------------------------------------------------------- entorno (una vez)
def preparar_entorno(tablas: dict[str, pd.DataFrame], cortes_sim: list[int],
                     seed_mundo: int = 42) -> dict:
    """Precomputa el contexto, la respuesta latente y el pool por corte (fijo entre semillas)."""
    contexto = construir_contexto(tablas, cortes_sim).reset_index(drop=True)
    mu, sigma = ajustar_escalador(contexto)
    X = transformar_contexto(contexto, mu, sigma)
    raw = contexto[CONTEXTO_NUMERICO].fillna(0.0).to_numpy(dtype=np.float64)

    tipo_tabla = generar_tipo_respuesta(contexto, seed_mundo)
    lat = calcular_respuesta_latente(contexto, tipo_tabla, seed_mundo)
    lat_p = lat[[f"lat_p_churn_{b}" for b in BRAZOS]].to_numpy(dtype=np.float64)
    tipo_resp = lat["lat_tipo_respuesta"].to_numpy()

    rim_ref = contexto["rim_ref"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    comision = TASA_COMISION_ANUAL * rim_ref * 12.0

    n_afiliados = int(tablas["afiliados"]["afiliado_id"].max())
    opt_out = np.zeros(n_afiliados + 1, dtype=bool)
    pref = tablas["preferencias_contacto"]
    opt_out[pref.loc[pref["solicito_no_contacto"], "afiliado_id"].to_numpy()] = True

    af_ids = contexto["afiliado_id"].to_numpy()
    corte_arr = contexto["periodo_corte"].to_numpy()
    score = contexto["score_riesgo"].to_numpy()
    por_corte = {}
    for c in cortes_sim:
        idx = np.where(corte_arr == c)[0]
        por_corte[c] = idx[np.argsort(-score[idx])]

    return {"contexto": contexto, "X": X, "raw": raw, "lat_p": lat_p, "tipo_resp": tipo_resp,
           "comision": comision, "opt_out": opt_out, "af_ids": af_ids, "corte": corte_arr,
           "score_riesgo": score, "por_corte": por_corte, "n_afiliados": n_afiliados,
           "d": X.shape[1], "d_raw": raw.shape[1]}


def seleccionar_pool(entorno: dict, corte: int, ultimo_contacto: np.ndarray,
                     presupuesto: int = PRESUPUESTO_MENSUAL_CONTACTOS) -> np.ndarray:
    """Poblacion "en riesgo" del mes: top `presupuesto` por riesgo, excluyendo opt-out y a quien se
    contacto (brazo pagado) hace menos de `MESES_ENTRE_CONTACTOS`. `por_corte[corte]` ya viene
    ordenado por riesgo descendente, asi que el filtrado booleano preserva ese orden."""
    idx_corte = entorno["por_corte"][corte]
    if len(idx_corte) == 0:
        return idx_corte
    af_ids = entorno["af_ids"][idx_corte]
    corte_idx = periodo_to_idx(corte)
    elegible = (~entorno["opt_out"][af_ids]) & ((corte_idx - ultimo_contacto[af_ids]) >= MESES_ENTRE_CONTACTOS)
    return idx_corte[elegible][:presupuesto]


# --------------------------------------------------------------------- politicas
class PoliticaSinContacto:
    """Baseline: nunca contacta a nadie."""
    nombre = "SIN_CONTACTO"

    def elegir(self, entorno, pos, rng):
        return np.zeros(len(pos), dtype=np.int8)

    def actualizar(self, entorno, pos, arms, rewards):
        pass

    def propension_matriz(self, entorno, pos, eps_eval: float = 0.05):
        m = len(pos)
        p = np.full((m, N_BRAZOS), eps_eval / N_BRAZOS)
        p[:, 0] += 1 - eps_eval
        return p


class PoliticaTopKVisita:
    """Baseline de negocio: a todo el pool (top-K por riesgo) se le hace una visita de ejecutivo."""
    nombre = "TOPK_RIESGO_VISITA"

    def elegir(self, entorno, pos, rng):
        return np.full(len(pos), IDX_VISITA, dtype=np.int8)

    def actualizar(self, entorno, pos, arms, rewards):
        pass

    def propension_matriz(self, entorno, pos, eps_eval: float = 0.05):
        m = len(pos)
        p = np.full((m, N_BRAZOS), eps_eval / N_BRAZOS)
        p[:, IDX_VISITA] += 1 - eps_eval
        return p


class PoliticaHeuristicaLogging:
    """Politica historica de logging: terciles de riesgo dentro del pool -> brazo, con exploracion
    epsilon uniforme sobre los 5 brazos. Devuelve tambien la propension de la accion registrada
    (necesaria para IPS/SNIPS/DR). No aprende (es la politica ya desplegada que se esta auditando)."""
    nombre = LOGGING

    def __init__(self, eps: float = EPSILON_LOGGING):
        self.eps = eps

    def elegir(self, entorno, pos, rng):
        m = len(pos)
        rango = np.arange(m) / max(m, 1)  # pos ya viene ordenado por riesgo descendente
        tier = np.where(rango < 1 / 3, IDX_BRAZO["VISITA"],
                        np.where(rango < 2 / 3, IDX_BRAZO["LLAMADA"], IDX_BRAZO["EMAIL"]))
        explora = rng.random(m) < self.eps
        aleatorio = rng.integers(0, N_BRAZOS, m)
        arms = np.where(explora, aleatorio, tier).astype(np.int8)
        propension = np.where(arms == tier, (1 - self.eps) + self.eps / N_BRAZOS, self.eps / N_BRAZOS)
        return arms, propension, tier.astype(np.int8)

    def actualizar(self, entorno, pos, arms, rewards):
        pass


class PoliticaEpsilonGreedy:
    """Epsilon-greedy NO contextual (promedio de recompensa por brazo, clasico)."""
    nombre = "EPSILON_GREEDY"

    def __init__(self, eps: float = 0.1):
        self.eps = eps
        self.n = np.zeros(N_BRAZOS)
        self.media = np.zeros(N_BRAZOS)

    def elegir(self, entorno, pos, rng):
        m = len(pos)
        greedy = int(np.argmax(self.media))
        explora = rng.random(m) < self.eps
        aleatorio = rng.integers(0, N_BRAZOS, m)
        return np.where(explora, aleatorio, greedy).astype(np.int8)

    def actualizar(self, entorno, pos, arms, rewards):
        for a in np.unique(arms):
            mask = arms == a
            k = mask.sum()
            self.n[a] += k
            self.media[a] += (rewards[mask].sum() - k * self.media[a]) / self.n[a]

    def propension_matriz(self, entorno, pos, eps_eval: float = 0.05):
        m = len(pos)
        p = np.full((m, N_BRAZOS), self.eps / N_BRAZOS)
        p[:, int(np.argmax(self.media))] += 1 - self.eps
        return p


class _PoliticaLineal:
    """Ridge lineal por brazo (A = X'X + lam*I, b = X'r): base comun de LinUCB y Thompson lineal."""

    def __init__(self, d: int, lam: float = 1.0):
        self.d = d
        self.A = np.tile(lam * np.eye(d), (N_BRAZOS, 1, 1))
        self.b = np.zeros((N_BRAZOS, d))

    def _mu_cov(self, a: int) -> tuple[np.ndarray, np.ndarray]:
        Ainv = np.linalg.inv(self.A[a])
        return Ainv @ self.b[a], Ainv

    def actualizar(self, entorno, pos, arms, rewards):
        X = entorno["X"][pos]
        for a in np.unique(arms):
            mask = arms == a
            Xa, ra = X[mask], rewards[mask]
            self.A[a] += Xa.T @ Xa
            self.b[a] += Xa.T @ ra


class PoliticaLinUCB(_PoliticaLineal):
    nombre = "LINUCB"

    def __init__(self, d: int, alpha: float = 1.0, lam: float = 1.0):
        super().__init__(d, lam)
        self.alpha = alpha

    def elegir(self, entorno, pos, rng):
        X = entorno["X"][pos]
        ucb = np.empty((len(pos), N_BRAZOS))
        for a in range(N_BRAZOS):
            mu, Ainv = self._mu_cov(a)
            var = np.einsum("ij,jk,ik->i", X, Ainv, X)
            ucb[:, a] = X @ mu + self.alpha * np.sqrt(np.maximum(var, 0.0))
        return np.argmax(ucb, axis=1).astype(np.int8)

    def propension_matriz(self, entorno, pos, eps_eval: float = 0.05):
        arms = self.elegir(entorno, pos, None)
        p = np.full((len(pos), N_BRAZOS), eps_eval / N_BRAZOS)
        p[np.arange(len(pos)), arms] += 1 - eps_eval
        return p


class PoliticaThompsonLineal(_PoliticaLineal):
    nombre = "THOMPSON_LINEAL"

    def __init__(self, d: int, v: float = 1.0, lam: float = 1.0):
        super().__init__(d, lam)
        self.v = v

    def elegir(self, entorno, pos, rng):
        X = entorno["X"][pos]
        m = len(pos)
        valor = np.empty((m, N_BRAZOS))
        for a in range(N_BRAZOS):
            mu, Ainv = self._mu_cov(a)
            theta = rng.multivariate_normal(mu, (self.v ** 2) * Ainv, size=m)
            valor[:, a] = np.einsum("ij,ij->i", theta, X)
        return np.argmax(valor, axis=1).astype(np.int8)

    def propension_matriz(self, entorno, pos, eps_eval: float = 0.05):
        # se congela con la MEDIA posterior (no una muestra) para tener una distribucion evaluable
        # de forma determinista: aproximacion documentada en el README.
        X = entorno["X"][pos]
        valor = np.stack([X @ self._mu_cov(a)[0] for a in range(N_BRAZOS)], axis=1)
        arms = np.argmax(valor, axis=1)
        p = np.full((len(pos), N_BRAZOS), eps_eval / N_BRAZOS)
        p[np.arange(len(pos)), arms] += 1 - eps_eval
        return p


LGBM_PARAMS = dict(n_estimators=60, max_depth=4, num_leaves=15, learning_rate=0.1,
                   min_child_samples=20, n_jobs=2, verbosity=-1)


class PoliticaThompsonBootstrapLGBM:
    """Thompson sampling aproximado con un ensamble bootstrap de `B` LightGBM (una arma extra al
    contexto numerico crudo, en vez de 5 x B modelos por brazo, para que el refit sea barato).
    Se reentrena cada `refit_cada` meses sobre las ultimas `buffer_max` observaciones resueltas."""
    nombre = "THOMPSON_BOOTSTRAP_LGBM"

    def __init__(self, d_raw: int, B: int = 3, refit_cada: int = 6, buffer_max: int = 4000,
                seed: int = 0):
        self.B, self.refit_cada, self.buffer_max = B, refit_cada, buffer_max
        self.rng_boot = np.random.default_rng(seed + 777)
        self.buffer_X: list[np.ndarray] = []
        self.buffer_a: list[np.ndarray] = []
        self.buffer_r: list[np.ndarray] = []
        self.modelos: list[lgb.LGBMRegressor] | None = None

    def _long(self, raw: np.ndarray) -> np.ndarray:
        m = len(raw)
        rep = np.repeat(raw, N_BRAZOS, axis=0)
        brazo_idx = np.tile(np.arange(N_BRAZOS), m).reshape(-1, 1)
        return np.hstack([rep, brazo_idx])

    def _predicciones(self, raw: np.ndarray) -> np.ndarray:
        """(B, m, n_brazos) con la prediccion de cada miembro del ensamble."""
        largo = self._long(raw)
        m = len(raw)
        return np.stack([mod.predict(largo).reshape(m, N_BRAZOS) for mod in self.modelos])

    def elegir(self, entorno, pos, rng):
        m = len(pos)
        if self.modelos is None:
            return rng.integers(0, N_BRAZOS, m).astype(np.int8)
        raw = entorno["raw"][pos]
        preds = self._predicciones(raw)
        b_elegido = rng.integers(0, self.B, m)
        valor = preds[b_elegido, np.arange(m), :]
        return np.argmax(valor, axis=1).astype(np.int8)

    def actualizar(self, entorno, pos, arms, rewards):
        self.buffer_X.append(entorno["raw"][pos])
        self.buffer_a.append(arms.astype(np.float64))
        self.buffer_r.append(rewards)

    def tick(self, mes_i: int):
        if self.refit_cada <= 0 or (mes_i + 1) % self.refit_cada != 0 or not self.buffer_X:
            return
        X = np.vstack(self.buffer_X)
        a = np.concatenate(self.buffer_a).reshape(-1, 1)
        r = np.concatenate(self.buffer_r)
        if len(r) > self.buffer_max:
            X, a, r = X[-self.buffer_max:], a[-self.buffer_max:], r[-self.buffer_max:]
            self.buffer_X, self.buffer_a, self.buffer_r = [X], [a.ravel()], [r]
        if len(r) < 30:
            return
        Xa = np.hstack([X, a])
        modelos = []
        for _ in range(self.B):
            idx = self.rng_boot.integers(0, len(r), len(r))
            mod = lgb.LGBMRegressor(**LGBM_PARAMS)
            mod.fit(Xa[idx], r[idx])
            modelos.append(mod)
        self.modelos = modelos

    def propension_matriz(self, entorno, pos, eps_eval: float = 0.05):
        m = len(pos)
        if self.modelos is None:
            return np.full((m, N_BRAZOS), 1.0 / N_BRAZOS)
        preds = self._predicciones(entorno["raw"][pos]).mean(axis=0)  # promedio del ensamble
        arms = np.argmax(preds, axis=1)
        p = np.full((m, N_BRAZOS), eps_eval / N_BRAZOS)
        p[np.arange(m), arms] += 1 - eps_eval
        return p


def crear_politica(nombre: str, entorno: dict, seed: int):
    if nombre == "SIN_CONTACTO":
        return PoliticaSinContacto()
    if nombre == "TOPK_RIESGO_VISITA":
        return PoliticaTopKVisita()
    if nombre == LOGGING:
        return PoliticaHeuristicaLogging()
    if nombre == "EPSILON_GREEDY":
        return PoliticaEpsilonGreedy()
    if nombre == "LINUCB":
        return PoliticaLinUCB(entorno["d"])
    if nombre == "THOMPSON_LINEAL":
        return PoliticaThompsonLineal(entorno["d"])
    if nombre == "THOMPSON_BOOTSTRAP_LGBM":
        return PoliticaThompsonBootstrapLGBM(entorno["d_raw"], seed=seed)
    raise ValueError(f"politica desconocida: {nombre}")


# --------------------------------------------------------------------- simulacion online
def simular_politica_online(entorno: dict, nombre: str, cortes_sim: list[int], seed: int
                            ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Corre `nombre` durante `cortes_sim` con la semilla `seed`. Devuelve la curva mensual
    acumulada y, solo para `HEURISTICA_LOGGING`, el dataset logueado (para OPE)."""
    rng = np.random.default_rng(seed * 1000 + _hash_estable(nombre))
    politica = crear_politica(nombre, entorno, seed)
    pendientes: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    ultimo_contacto = np.full(entorno["n_afiliados"] + 1, -10_000, dtype=np.int64)
    acumulado = dict(reward=0.0, costo=0.0, regret=0.0, churn=0, n=0, sleeping_dog=0)
    curva, log_filas = [], []

    for mes_i, corte in enumerate(cortes_sim):
        corte_idx = periodo_to_idx(corte)
        listos = pendientes.pop(corte_idx, [])
        if listos:
            pos_r = np.array([x[0] for x in listos])
            arms_r = np.array([x[1] for x in listos])
            rew_r = np.array([x[2] for x in listos])
            politica.actualizar(entorno, pos_r, arms_r, rew_r)
        if hasattr(politica, "tick"):
            politica.tick(mes_i)

        pos_pool = seleccionar_pool(entorno, corte, ultimo_contacto)
        if len(pos_pool) == 0:
            curva.append({"mes": mes_i + 1, "corte": corte, **acumulado})
            continue

        propension_logueada, tier_logueado = None, None
        salida = politica.elegir(entorno, pos_pool, rng)
        if isinstance(salida, tuple):
            arms, propension_logueada, tier_logueado = salida
        else:
            arms = salida

        lat_p_pool = entorno["lat_p"][pos_pool]
        comision = entorno["comision"][pos_pool]
        p_elegido = lat_p_pool[np.arange(len(pos_pool)), arms]
        traspasa = rng.random(len(pos_pool)) < p_elegido
        costo = COSTO_VEC[arms]
        reward = np.where(traspasa, 0.0, comision) - costo

        valor_esperado = (1 - lat_p_pool) * comision[:, None] - COSTO_VEC[None, :]
        reward_oraculo = valor_esperado.max(axis=1)
        reward_esperado_elegido = valor_esperado[np.arange(len(pos_pool)), arms]

        af_ids = entorno["af_ids"][pos_pool]
        ultimo_contacto[af_ids] = np.where(arms != 0, corte_idx, ultimo_contacto[af_ids])
        for i in range(len(pos_pool)):
            pendientes[corte_idx + HORIZONTE_MESES].append((pos_pool[i], arms[i], reward[i]))

        tipo_pool = entorno["tipo_resp"][pos_pool]
        acumulado["reward"] += float(reward.sum())
        acumulado["costo"] += float(costo.sum())
        acumulado["regret"] += float((reward_oraculo - reward_esperado_elegido).sum())
        acumulado["churn"] += int(traspasa.sum())
        acumulado["n"] += int(len(pos_pool))
        acumulado["sleeping_dog"] += int(((arms != 0) & (tipo_pool == "SLEEPING_DOG")).sum())
        curva.append({"mes": mes_i + 1, "corte": corte, **acumulado})

        if propension_logueada is not None:
            for i in range(len(pos_pool)):
                log_filas.append({"pos": int(pos_pool[i]), "afiliado_id": int(af_ids[i]),
                                  "periodo_corte": int(corte), "brazo": BRAZOS[arms[i]],
                                  "arm_idx": int(arms[i]), "tier_idx": int(tier_logueado[i]),
                                  "propension_logueada": float(propension_logueada[i]),
                                  "reward": float(reward[i]), "costo": float(costo[i]),
                                  "traspaso": bool(traspasa[i])})

    curva_df = pd.DataFrame(curva)
    log_df = pd.DataFrame(log_filas) if log_filas else None
    return curva_df, log_df


def simular_todas(entorno: dict, cortes_sim: list[int], seeds: tuple[int, ...]
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Corre las 7 politicas para cada semilla. Devuelve:

    * `curvas`: DataFrame largo (politica, seed, mes, corte, reward, costo, regret, churn, n,
      sleeping_dog) acumulados.
    * `logueado`: dataset de la politica de logging en la PRIMERA semilla (usado por `evaluate.py`).
    * `politicas_congeladas`: politicas candidatas ya entrenadas al final de la primera semilla
      (estado congelado, usado para calcular su distribucion de accion en `evaluate.py`).
    """
    curvas = []
    logueado = None
    congeladas = {}
    for seed in seeds:
        for nombre in TODAS_LAS_POLITICAS:
            rng_state_seed = seed
            curva, log_df = simular_politica_online(entorno, nombre, cortes_sim, rng_state_seed)
            curva.insert(0, "politica", nombre)
            curva.insert(1, "seed", seed)
            curvas.append(curva)
    curvas_df = pd.concat(curvas, ignore_index=True)

    # dataset logueado + politicas congeladas: de la PRIMERA semilla, para que evaluate.py evalue
    # sobre un unico snapshot de datos historicos (asi funcionaria en produccion: se audita un lote
    # ya loggeado con las politicas candidatas ya entrenadas, no con un promedio de "paralelos").
    seed0 = seeds[0]
    for nombre in (LOGGING,) + ALGORITMOS:
        politica = crear_politica(nombre, entorno, seed0)
        _rerun_hasta_convergencia(entorno, politica, cortes_sim, seed0)
        if nombre == LOGGING:
            _, logueado = simular_politica_online(entorno, nombre, cortes_sim, seed0)
        else:
            congeladas[nombre] = politica
    return curvas_df, logueado, congeladas


def _rerun_hasta_convergencia(entorno, politica, cortes_sim, seed):
    """Re-simula `politica` desde cero (mismo camino que `simular_politica_online`) pero devolviendo
    el OBJETO politica ya entrenado (con su estado al final de la ventana), para congelarlo y usarlo
    en la evaluacion off-policy. Duplica el bucle en vez de reusar `simular_politica_online` para no
    tener que "inyectar" una politica ya creada en esa funcion (que crea la suya via `crear_politica`,
    mas simple de mantener separada dado el alcance de este proyecto)."""
    rng = np.random.default_rng(seed * 1000 + _hash_estable(politica.nombre))
    pendientes: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    ultimo_contacto = np.full(entorno["n_afiliados"] + 1, -10_000, dtype=np.int64)
    for mes_i, corte in enumerate(cortes_sim):
        corte_idx = periodo_to_idx(corte)
        listos = pendientes.pop(corte_idx, [])
        if listos:
            pos_r = np.array([x[0] for x in listos])
            arms_r = np.array([x[1] for x in listos])
            rew_r = np.array([x[2] for x in listos])
            politica.actualizar(entorno, pos_r, arms_r, rew_r)
        if hasattr(politica, "tick"):
            politica.tick(mes_i)
        pos_pool = seleccionar_pool(entorno, corte, ultimo_contacto)
        if len(pos_pool) == 0:
            continue
        salida = politica.elegir(entorno, pos_pool, rng)
        arms = salida[0] if isinstance(salida, tuple) else salida
        lat_p_pool = entorno["lat_p"][pos_pool]
        comision = entorno["comision"][pos_pool]
        p_elegido = lat_p_pool[np.arange(len(pos_pool)), arms]
        traspasa = rng.random(len(pos_pool)) < p_elegido
        costo = COSTO_VEC[arms]
        reward = np.where(traspasa, 0.0, comision) - costo
        af_ids = entorno["af_ids"][pos_pool]
        ultimo_contacto[af_ids] = np.where(arms != 0, corte_idx, ultimo_contacto[af_ids])
        for i in range(len(pos_pool)):
            pendientes[corte_idx + HORIZONTE_MESES].append((pos_pool[i], arms[i], reward[i]))
