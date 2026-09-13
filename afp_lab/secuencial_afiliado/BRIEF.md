# Brief 08 — `afp_lab/secuencial_afiliado` (worktree `/home/user/wt/datos`)

> Copia literal de `docs/briefs/ronda_4/secuencial.md` (no se modifica; las dudas y el
> acotamiento de alcance quedan en el README, sección "Decisiones tomadas").
>
> Nota de alcance (ver README): el worktree efectivo de esta entrega fue
> `/home/user/wt/empleador`, rama `ronda-4/secuencial`, según la instrucción de arranque del
> gerente para esta ronda; no cambia nada del contenido del brief.

## Pregunta
¿Una red neuronal secuencial sobre la historia mensual del afiliado, entrenada multi-tarea,
supera a los LightGBM tabulares de 02 (traspaso 6 m) y 04 (cese) y produce un embedding útil
para 06 (segmentación)? Se reporta el resultado aunque pierda.

## Datos
* Universo común perfil `cuprum` + tablas de 02, 03 y 04 importadas sin modificar. Targets
  construidos con las funciones de esos proyectos (definiciones idénticas): traspaso 6 m (02),
  cese (04, su horizonte), aporte APV 6 m (03) y RIM del mes corte+1 (regresión sobre
  log1p, desde el panel).
* Secuencia por afiliado y corte: ventana de 24 meses con las variables mensuales disponibles
  as-of (panel hasta corte−2 por la regla del proyecto 01: RIM, estado, tipo de contrato,
  jornada, licencias, evento AFC, empleador cambia; tablas operacionales de 02/03/04 hasta
  corte: contactos, reclamos, cambios de fondo, uso digital, saldos). Meses no observados
  enmascarados. Categóricas con embeddings; estáticas (sexo, edad, región, educación,
  antigüedad) concatenadas.
* Cortes y splits: los mismos cortes evaluables y el mismo `split_temporal` con gap que 02
  para que la comparación sea justa; misma semilla.

## Modelo (`model.py`, PyTorch CPU, ya instalado)
* GRU de 2 capas (o transformer encoder pequeño, elegir con validación) con 4 cabezas: 3 de
  clasificación (BCE) y 1 de regresión (Huber sobre log1p), pérdida ponderada; early stopping
  en validación; semilla fija; entrenamiento < 10 min a 20 000 afiliados con `torch.set_num_threads`.
* Ablación: single-task por cabeza vs multi-task.
* Embedding: la salida del encoder (dimensión 16–32) por afiliado en el corte de referencia.

## Baselines
* LightGBM de 02 y 04 con SUS features y SUS funciones, en los mismos cortes/splits.
* LightGBM "tabular con lags" sobre las mismas variables de la secuencia aplanadas (baseline
  fuerte: separa el efecto de la arquitectura del efecto de la información).

## Evaluación
* Por cabeza: AUC, PR-AUC, lift decil 1 (usar `common.metrics`), calibración; WAPE para RIM.
* Tabla red vs baselines con intervalos bootstrap sobre el test; criterio "vale la pena" =
  mejora ≥ 0,01 AUC en ≥ 2 cabezas, o igual AUC con menos ingeniería de features.
* Embedding: k-means sobre el embedding vs la segmentación de 06 (ARI, silueta, separación de
  tasas de negocio), sin modificar 06.

## Entregables
Proyecto completo, README con resultados y conclusión explícita (¿reemplaza, complementa o no
aporta?), tests rápidos (modelo diminuto, pocas épocas), corrida de referencia
`python3 -m afp_lab.secuencial_afiliado.run` e informe estructurado.
