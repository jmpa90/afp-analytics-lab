# Revisión ronda 2 · 02 churn_traspaso

**Veredicto del gerente: APROBAR con correcciones menores**, con tres decisiones tomadas aquí.

## Verificado
- Bloque digital sin interruptor: ratio logins k=1 ×1,3 (antes ×3,4), certificado ×1,7 (antes ×27), lift
  del certificado 1,32, bloque digital 2 % del gain, score máximo 0,37.
- Mercado: rango de rentabilidad entre AFP 1,5–1,9 pp (antes 6), tasa móvil 3,4–6,9 % (antes 1,6–12,6 %),
  castigo domina su ventana en 6/6 semillas.
- Leakage limpio con mutación agresiva incluidas las features nuevas (0 de 72 cambian). Split 37/12 cortes
  usados/descartados, 0 meses de target compartidos, validación a 7 meses del test. Calibración y early
  stopping sólo con validación. 24/24 segmentos grandes ganados. Reproducción al dígito, 58 s, 24 tests.

## Decisiones del gerente
1. **Umbral de AUC para el banco sintético**: se acepta **AUC ≥ 0,70 en todas las semillas** y se asciende
   la métrica de negocio a criterio principal: **captura al 10 % ≥ 30 % de los traspasos y ≥ 1,5 × la
   mejor regla, en todas las semillas** (hoy 32–36 % y 1,7–2,8×). Para datos reales se mantiene ≥ 0,75.
   El "≥ 95 % del techo" **no** es criterio de aceptación (se cumple en cualquier configuración y depende
   de `sigma_afiliado`); `techo_predictibilidad` queda como diagnóstico.
2. **Mecanismos nuevos**: las campañas de la competencia por empleador son plausibles (los vendedores
   visitan empresas) y se aceptan. La persistencia del switcher es plausible, pero `llego_por_traspaso`
   reproduce el latente con concordancia 1,0: debe degradarse a proxy imperfecto (parte de los
   switchers se afilió directamente y parte de los no switchers llegó por un traspaso único).
3. **Saldo en riesgo esperado (pregunta secundaria)**: se declara **no entregado en esta ronda**; el
   ancla queda como diagnóstico y el pronóstico agregado del flujo pasa al backlog.

## Correcciones exigidas antes de integrar
- Mayor 5: reescribir el diagnóstico del drift de §6 con la descomposición medida (recorte de comisión y
  envejecimiento de la cartera como parte sistemática; rentabilidad como parte aleatoria).
- Mayor 3 / menor 9: corregir el contrafactual de §11 (0,685 / 2,77, fallarían ambos criterios) y las
  cifras del castigo (−0,45 / +0,15) y de la tasa (5,83 %).
- Menor 6: entregar `estabilidad_semillas.csv` y `estabilidad_rangos.csv` con la corrida por defecto o
  marcar el flag en §9.
- Menor 7: excluir `lat_*`/`p_*` en `columnas_features`; el diagnóstico del techo pasa columnas explícitas;
  no escribir `afiliados_latente_churn` en `fuentes/`.
- Menores 8, 10, 11, 12: `add_months` en el test del castigo; docstring de `split_temporal`; nota de
  redundancia de `meses_en_otras_afp`; `reportes/hiperparametros.csv` y el experimento de ventana.
- Menor 13 (common, corregido por el gerente): lift del decil 1 con empates por valor esperado.

## Cierre (ronda 3 de correcciones)
- Verificado: 27 tests en 15 s; corrida de 20.000 en 30 s; AUC 0,720 (0,707–0,747 en 5 semillas), lift
  3,50 (3,00–3,66), captura al 10 % 35,0 % (30,0–36,6 %), razón vs mejor regla 1,80 (1,30–1,85),
  23/23 segmentos grandes ganados; concordancia proxy/latente del switcher 0,762; latentes excluidos por
  prefijo; saldo en riesgo declarado no entregado.
- **Decisión sobre la razón ≥ 1,5**: falla en 1 de 5 semillas (1,30) porque la regla del brief marca
  entre 14 % y 51 % de la cartera según la semilla (denominador inestable). Se acepta la propuesta del DS:
  el criterio absoluto (captura ≥ 30 %) se exige por semilla; la razón ≥ 1,5 se mide sobre la mediana de
  semillas (1,77). Queda registrado en README §8 del proyecto.
- **Incidente de proceso**: a las 16:56 el gerente ejecutó `git checkout -- .` para limpiar el árbol
  antes de fusionar `origin/main` y revirtió las ediciones no confirmadas de DS-1, que las re-aplicó y
  verificó byte a byte. Regla nueva en CONVENCIONES §5: mientras haya data scientists trabajando, el
  gerente no ejecuta checkout/reset/stash sobre el árbol compartido; en la ronda 2 cada DS trabaja en su
  propio worktree.
