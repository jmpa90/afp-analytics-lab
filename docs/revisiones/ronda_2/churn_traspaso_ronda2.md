# Revisión ronda 2 · 02 churn_traspaso (perfil Cuprum, licitación de stock, pronóstico agregado)

**Veredicto del gerente: APROBAR con menores** (revisor Sonnet; menores aplicados por el gerente).

## Verificado
- Sin fuga: el target sale sólo de `traspasos`; `licitacion_stock` no entra como feature ni contamina el
  target (cortes truncados antes de 2028-06); latentes excluidos por prefijo; la decisión del sorteo se
  simula con el latente sólo en el mundo sintético.
- `combinar_con_score` y `ranking_combinado` hacen lo documentado; escenario saldo × prob vs aleatorio
  reproducido (2,0–2,1× de saldo retenido en los tres supuestos de salida pasiva).
- `seleccionar_pronostico` sólo usa train+valid; resultado negativo bien medido.
- Sistema vs Cuprum reproducido dígito a dígito; AUC 0,698 en sistema/semilla 42 es determinista y
  explicable por la ventana de test (tasa base 1,66 %); 33 tests; reproducible.

## Decisiones y menores aplicados
- Calibración de la fuga de Cuprum: manda la Ficha (0,47 %/mes sistema); 0,446 %/mes es supuesto de
  trabajo hasta tener la serie de traspasos por AFP de origen. Rango 4–7 % de la ronda 1 queda como prior.
- Malla de capacidad de la campaña con 10/30/50/70/100 % y lectura de rendimiento marginal decreciente
  (14 MM por contacto al 30 % → 7 MM al 100 %).
- Limitación documentada: `licitacion_stock` no retroalimenta la cartera simulada.
- `contexto_afp_cuprum.md` §4: aclarado 1,74× (universo común) vs 2,0× (ventana 2027 del proyecto 02).
- Pendiente de backlog: intervalos bootstrap del AUC por segmento (60+ con < 100 eventos).
