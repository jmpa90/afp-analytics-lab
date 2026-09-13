"""Red secuencial multi-tarea (PyTorch, CPU) + entrenamiento del baseline LightGBM "tabular con
lags" (mismas variables que la secuencia, aplanadas). Los LightGBM "de produccion" de 02 y 04 NO
se reimplementan aqui: `run.py` llama directo a `churn_traspaso.model.entrenar` y
`riesgo_cese_cotizacion.model.ModeloCese.fit`.

Arquitectura (BRIEF.md): embeddings por variable categorica de la secuencia (estado, evento_afc,
tipo_contrato) + las numericas, GRU de 2 capas -> ultimo paso (siempre valido: los cortes usados
son "evaluables", el mes `corte-2` existe si el afiliado califica para esa poblacion) ->
concatenado con las estaticas (embeddings + numericas) -> proyeccion lineal a un embedding de
16-32 (`dim_embedding`) -> una cabeza lineal por tarea. Multi-tarea = todas las cabezas activas
compartiendo encoder, con lotes de cada tarea alternados y perdida BCE ponderada; single-task =
la misma arquitectura con una sola cabeza activa (instancia separada, mismo split que su
contraparte multi-tarea). Early stopping por perdida de validacion combinada.

Los meses no observados de la secuencia se enmascaran multiplicando la entrada por la mascara
(no se usa `pack_padded_sequence`): es una simplificacion documentada en el README (limitaciones)
por presupuesto de tiempo de la ronda; el ultimo paso de la ventana (`corte-2`) es el que se lee
como resumen y ese sí existe siempre para las poblaciones usadas.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .features import (CANALES_CAT, CANALES_NUM, CANALES_STATIC_CAT, CANALES_STATIC_NUM,
                       SecuenciaTensores, VOCAB_CAT, VOCAB_STATIC_CAT, aplanar_secuencia,
                       columnas_categoricas_planas)

SEED_TORCH = 42


@dataclass
class DatosCabeza:
    """Tensores + etiqueta de una cabeza, ya separados en train/val/test (mismos cortes/split
    que el proyecto de origen de esa cabeza)."""
    train: SecuenciaTensores
    y_train: np.ndarray
    val: SecuenciaTensores
    y_val: np.ndarray
    test: SecuenciaTensores
    y_test: np.ndarray


def _a_tensores(t: SecuenciaTensores) -> tuple[torch.Tensor, ...]:
    return (torch.from_numpy(t.num), torch.from_numpy(t.cat).long(), torch.from_numpy(t.mask),
           torch.from_numpy(t.static_num), torch.from_numpy(t.static_cat).long())


class RedSecuencial(nn.Module):
    """Encoder GRU compartido + una cabeza lineal por tarea en `cabezas`."""

    def __init__(self, cabezas: list[str], hidden: int = 32, capas: int = 2,
                dim_embedding: int = 24, dim_emb_cat: int = 8, dim_emb_static: int = 6):
        super().__init__()
        self.emb_cat = nn.ModuleList([nn.Embedding(len(VOCAB_CAT[c]), dim_emb_cat) for c in CANALES_CAT])
        self.emb_static = nn.ModuleList([nn.Embedding(len(VOCAB_STATIC_CAT[c]), dim_emb_static)
                                         for c in CANALES_STATIC_CAT])
        in_dim = len(CANALES_NUM) + dim_emb_cat * len(CANALES_CAT)
        self.gru = nn.GRU(in_dim, hidden, num_layers=capas, batch_first=True)
        static_dim = len(CANALES_STATIC_NUM) + dim_emb_static * len(CANALES_STATIC_CAT)
        self.proyeccion = nn.Sequential(nn.Linear(hidden + static_dim, dim_embedding), nn.ReLU())
        self.cabezas = nn.ModuleDict({c: nn.Linear(dim_embedding, 1) for c in cabezas})

    def embedding(self, num, cat, mask, static_num, static_cat) -> torch.Tensor:
        emb_seq = [self.emb_cat[i](cat[..., i]) for i in range(cat.shape[-1])]
        x = torch.cat([num] + emb_seq, dim=-1) * mask.unsqueeze(-1)
        out, _ = self.gru(x)
        rep = out[:, -1, :]
        emb_static = [self.emb_static[i](static_cat[..., i]) for i in range(static_cat.shape[-1])]
        z = torch.cat([rep, static_num] + emb_static, dim=-1)
        return self.proyeccion(z)

    def forward(self, num, cat, mask, static_num, static_cat, cabeza: str) -> torch.Tensor:
        z = self.embedding(num, cat, mask, static_num, static_cat)
        return self.cabezas[cabeza](z).squeeze(-1)


def entrenar_red(datos: dict[str, DatosCabeza], cabezas: list[str] | None = None, seed: int = SEED_TORCH,
                 epocas: int = 6, paciencia: int = 2, lr: float = 2e-3, batch_size: int = 512,
                 hidden: int = 32, dim_embedding: int = 24, pesos: dict[str, float] | None = None,
                 n_threads: int = 2) -> dict:
    """Entrena la red (multi-tarea si `cabezas` tiene >1 elemento, single-task si tiene 1).

    Early stopping sobre la SUMA ponderada de la perdida BCE de validacion de cada cabeza
    (cada cabeza usa su propio `val`, con los mismos cortes que su split de origen)."""
    torch.manual_seed(seed)
    torch.set_num_threads(n_threads)
    cabezas = cabezas or list(datos.keys())
    pesos = pesos or {c: 1.0 / len(cabezas) for c in cabezas}

    red = RedSecuencial(cabezas=cabezas, hidden=hidden, dim_embedding=dim_embedding)
    opt = torch.optim.Adam(red.parameters(), lr=lr)

    gens = {c: torch.Generator().manual_seed(seed + i) for i, c in enumerate(cabezas)}
    loaders = {}
    for c in cabezas:
        tens = _a_tensores(datos[c].train) + (torch.from_numpy(datos[c].y_train).float(),)
        loaders[c] = DataLoader(TensorDataset(*tens), batch_size=batch_size, shuffle=True,
                                generator=gens[c])
    val_tens = {c: _a_tensores(datos[c].val) + (torch.from_numpy(datos[c].y_val).float(),) for c in cabezas}

    mejor_val, mejor_estado, sin_mejora = float("inf"), None, 0
    historia = []
    for _epoca in range(epocas):
        red.train()
        iters = {c: iter(loaders[c]) for c in cabezas}
        n_pasos = max(len(loaders[c]) for c in cabezas)
        for _paso in range(n_pasos):
            opt.zero_grad()
            loss_total = torch.zeros(())
            for c in cabezas:
                try:
                    *ent, y = next(iters[c])
                except StopIteration:
                    iters[c] = iter(loaders[c])
                    *ent, y = next(iters[c])
                pred = red(*ent, c)
                loss_total = loss_total + pesos[c] * F.binary_cross_entropy_with_logits(pred, y)
            loss_total.backward()
            opt.step()

        red.eval()
        with torch.no_grad():
            val_loss = sum(pesos[c] * F.binary_cross_entropy_with_logits(
                red(*val_tens[c][:-1], c), val_tens[c][-1]).item() for c in cabezas)
        historia.append(val_loss)
        if val_loss < mejor_val - 1e-4:
            mejor_val, sin_mejora = val_loss, 0
            mejor_estado = copy.deepcopy(red.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(mejor_estado)
    return {"modelo": red, "historia_val": historia, "mejor_val": mejor_val, "cabezas": cabezas}


def predecir_red(red: RedSecuencial, tensores: SecuenciaTensores, cabeza: str) -> np.ndarray:
    red.eval()
    with torch.no_grad():
        logits = red(*_a_tensores(tensores), cabeza)
        return torch.sigmoid(logits).numpy()


def obtener_embedding(red: RedSecuencial, tensores: SecuenciaTensores) -> np.ndarray:
    red.eval()
    with torch.no_grad():
        return red.embedding(*_a_tensores(tensores)).numpy()


# --------------------------------------------------------------- baseline "tabular con lags"
PARAMS_LGBM_LAGS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "num_threads": 2, "verbosity": -1, "force_row_wise": True,
}


def entrenar_lgbm_lags(train: SecuenciaTensores, y_train: np.ndarray, val: SecuenciaTensores,
                       y_val: np.ndarray, seed: int = 42, n_arboles: int = 400) -> dict:
    """LightGBM sobre la secuencia APLANADA (mismas variables que ve la red, sin arquitectura
    secuencial): baseline fuerte que separa el efecto de la arquitectura del de la informacion
    (BRIEF.md)."""
    x_train, x_val = aplanar_secuencia(train), aplanar_secuencia(val)
    cat_cols = columnas_categoricas_planas()
    for c in cat_cols:
        cats = pd.api.types.union_categoricals([x_train[c], x_val[c]]).categories
        x_train[c] = x_train[c].cat.set_categories(cats)
        x_val[c] = x_val[c].cat.set_categories(cats)
    p = dict(PARAMS_LGBM_LAGS, seed=seed)
    dtrain = lgb.Dataset(x_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(x_val, label=y_val, categorical_feature=cat_cols, reference=dtrain, free_raw_data=False)
    booster = lgb.train(p, dtrain, num_boost_round=n_arboles, valid_sets=[dval], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
    return {"booster": booster, "columnas": list(x_train.columns), "categoricas": cat_cols,
           "mejor_iteracion": int(booster.best_iteration or n_arboles)}


def predecir_lgbm_lags(modelo: dict, tensores: SecuenciaTensores) -> np.ndarray:
    x = aplanar_secuencia(tensores)
    for c in modelo["categoricas"]:
        x[c] = x[c].astype("category")
    x = x[modelo["columnas"]]
    return np.asarray(modelo["booster"].predict(x, num_iteration=modelo["mejor_iteracion"]), dtype=float)
