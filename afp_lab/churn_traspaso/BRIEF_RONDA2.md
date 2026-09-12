# Brief 02 · Traspaso — ronda 2 (addendum al BRIEF.md)

**Responsable**: DS-1. Parte del código aprobado en la ronda 1 (rama `main`). Tres encargos:

## A. Perfil Cuprum
Correr todo el proyecto con `generar_universo(perfil="cuprum")` (ver `docs/contexto_afp_cuprum.md`) y
comparar contra el perfil `sistema` en una tabla única (simulación, métricas, negocio, 3 semillas cada
uno). El `run.py` acepta `--perfil {sistema,cuprum}`; el README documenta ambos y explica las
diferencias con el contexto real de Cuprum (comisión 1,44 %, competencia de precio de Uno/Modelo, cartera
de mayor renta y mayor antigüedad, traspasos netos positivos en 2026). Calibrar la tasa de traspaso del
perfil Cuprum con la Ficha N° 165 (`data/sp/`): 56.974 cuentas traspasadas al mes en el sistema
(~0,47 % mensual) y Cuprum con saldo neto positivo.

## B. Licitación de afiliados antiguos (Ley 21.735)
Cada 2 años se sortea el 10 % de los afiliados no pensionados del sistema y pasan a la AFP de menor
comisión salvo que opten por quedarse en 30 días; primera licitación agosto de 2027, traspasos desde
junio de 2028. Extender la ventana simulada hasta 2028-12 (la macro extrapola) y construir:
1. Tabla `licitacion_stock`: afiliado, fecha de sorteo, plazo de opción, decisión (se queda / se va),
   AFP de destino. La decisión se simula con una propensión a quedarse que dependa de antigüedad, saldo,
   renta, contacto de retención de la AFP y del score de churn latente; supuesto base 55–75 % se van si
   no hay contacto y documentar rangos.
2. Módulo `licitacion.py` con escenarios: afiliados y saldo administrado en riesgo por sorteo, efecto de
   una campaña de retención con capacidad limitada (p. ej. 30 % de los sorteados contactados) priorizada
   por saldo × probabilidad de irse, y sensibilidad al supuesto de salida pasiva.
3. El modelo de churn no se reentrena con el sorteo (no hay historia): el sorteo es un evento
   determinista que se combina con el score (`p_ir = p_sorteo_se_va · sorteado + p_churn · (1 − sorteado)`).
   Documentar por qué y qué se necesitaría para aprenderlo.

## C. Backlog de la ronda 1
Pronóstico agregado mensual del flujo de traspasos (serie con comisión, brecha de rentabilidad y
estacionalidad) para anclar el nivel del saldo en riesgo; reportar sesgo por corte con y sin ancla.

Criterios de aceptación: los de la ronda 1 (AUC ≥ 0,70 por semilla, captura ≥ 30 % por semilla, razón
≥ 1,5 en la mediana) en ambos perfiles o explicación de por qué no; escenarios de licitación con rangos;
runtime y tests dentro de límites; README con sección "Ronda 2".
