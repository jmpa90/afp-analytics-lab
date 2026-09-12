# Ronda 4 — técnicas avanzadas (briefs)

Objetivo de la ronda: probar en el sandbox tres técnicas que van más allá del GBDT tabular y
medir con honestidad si le sirven a la AFP. Tres proyectos nuevos, cada uno con un DS Sonnet en
su worktree, revisor adversarial y pasada de corrección (máximo 3 agentes en paralelo).

| Proyecto | Carpeta | Técnica | Pregunta de negocio |
|---|---|---|---|
| 07 | `afp_lab/uplift_retencion` | Inferencia causal / uplift (T-, X-, DR-learner) | ¿A quién contactar para que el contacto de verdad evite el traspaso? |
| 08 | `afp_lab/secuencial_afiliado` | Red neuronal secuencial multi-tarea (GRU / transformer, PyTorch CPU) | ¿Una representación secuencial del afiliado supera a los LightGBM de 02 y 04 y sirve de embedding? |
| 09 | `afp_lab/bandit_contacto` | Reinforcement learning: bandits contextuales + evaluación off-policy | ¿Qué política de contacto (a quién, canal, oferta) maximiza la retención neta de costo? |

Reglas comunes (además de `CONVENCIONES.md` y `docs/briefs/ronda_3/README.md`):

* Estructura §1 de CONVENCIONES: `synthetic.py`, `features.py`, `model.py`, `evaluate.py`, `run.py`,
  `tests/`, `README.md` (plantilla §6). Tests rápidos (< 60 s por proyecto).
* Reusar el universo común (`afp_lab/common/synth_base.generar_universo(..., perfil="cuprum")`) y las
  tablas y definiciones de target de 02 (`churn_traspaso`), 03 y 04 importándolas, sin copiarlas ni
  modificarlas. La definición del target debe ser IDÉNTICA a la del proyecto de origen.
* Splits temporales por corte con gap (§3). Hiperparámetros solo con validación. Semillas fijas.
* Toda "verdad" latente del generador (efectos causales, respuesta por brazo) vive en tablas
  `*_latente` y nunca entra a features (prefijo `lat_`, excluido como en 02).
* Los resultados se reportan aunque la técnica pierda contra el baseline: el objetivo es saber,
  no ganar.
* Nada de git desde los agentes; el gerente integra. `PYTHONPATH=<worktree>`.
* Costos y supuestos económicos (costo de un contacto, margen de comisión) se declaran como
  parámetros con nombre y valor en el README, marcados como supuestos.
