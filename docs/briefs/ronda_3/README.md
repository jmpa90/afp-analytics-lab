# Ronda 3 — briefs

Tres workstreams en paralelo (un DS Sonnet por workstream, un revisor adversarial por entrega,
una pasada de corrección). Reglas comunes, además de `CONVENCIONES.md`:

* Cada DS trabaja SOLO en su worktree (`/home/user/wt/<ws>`), con `PYTHONPATH=/home/user/wt/<ws>`
  y rama `ronda-3/<ws>`. **Nada de git** (ni add, ni commit, ni checkout, ni stash): el gerente
  integra. Excepción para revisores: `git diff main` / `git status` de solo lectura.
* Datos 100 % sintéticos. Cifras reales solo con fuente pública citada (URL y fecha) o desde
  `docs/contexto_afp_cuprum.md`; nunca inventadas.
* Sin leakage: as-of estricto (§3 de CONVENCIONES), gap entre particiones, hiperparámetros
  solo con validación.
* Entregable = código + tests + README del proyecto (sección "Ronda 3") + informe corto con
  números antes/después reproducibles por comando.
* Runtime: cada corrida de referencia < 5 min a la escala documentada; si no, bajar la escala
  y decirlo.
* Nunca `pkill -f` con patrones amplios (mata la shell de otros); matar por PID.
