# Brief — informe ejecutivo autocontenido "Qué ofrece Data Analytics" (entregable final)

## Para quién y para qué
Gerencia general, comercial, operaciones y riesgo de AFP Cuprum (persona del sandbox). Debe
poder leerse en 15 minutos en celular o impreso, sin conocer modelos, y responder: qué ofrece el
equipo, qué problema resuelve cada capacidad, qué evidencia hay, cómo se usaría y qué falta
para producción.

## Formato
* Un solo archivo `docs/reporte_ejecutivo.html`, autocontenido: CSS inline, gráficos en SVG
  inline (barras, líneas, embudo) generados con un script Python del repo
  (`docs/build_reporte.py`) que lee los CSV de resultados de cada proyecto; sin librerías
  externas ni URLs de terceros; funciona a 400 px de ancho; imprimible.
* Español, tono ejecutivo, frases cortas. Sin jerga en el cuerpo; anexo técnico al final.
* Mockups de dashboards como bloques HTML/CSS estáticos (tarjetas KPI, tabla de alertas,
  gráfico), uno por caso de uso y uno de "torre de control" que integra todo.

## Estructura
1. Portada y resumen ejecutivo (una pantalla): 5 mensajes, 5 números.
2. Qué ofrece el equipo: catálogo de capacidades (enriquecer, proyectar, segmentar, priorizar,
   decidir, cumplir normativa), y cómo se construyó (sandbox sintético, revisión adversarial,
   convenciones sin leakage).
3. Un caso de uso por proyecto (01 a 09, más planilla regulatoria y modelo de datos): problema
   de negocio, qué hace, resultado en el sandbox (con la métrica traducida a negocio),
   cómo lo usaría el área, mockup de dashboard, camino a producción (datos reales que se
   necesitan, validación, riesgos y regulación), y las reservas honestas.
4. Roadmap 6–12 meses y qué se pide al negocio (datos, patrocinio, pilotos).
5. Supuestos, reservas y límites (todo lo sintético marcado como tal).
6. Anexo técnico por proyecto (tabla: datos, modelo, métricas, tests, runtime, revisión).
7. Glosario.

## Reglas de veracidad
* Cada número del sandbox sale de la tabla de resultados del README del proyecto o de
  `docs/revisiones/`; se cita la sección. Está prohibido redondear hacia arriba o presentar un
  resultado "en el límite" como cumplido: usar las mismas palabras que la revisión.
* Números reales solo de la Ficha 165 (`docs/contexto_afp_cuprum.md`) y de fuentes ya citadas
  en el repo, con su fecha.
* Cada caso de uso lleva su reserva principal visible, no escondida en el anexo.

## Proceso
Lo escribe un DS; otro DS lo revisa contra los READMEs (exactitud de cada número, ninguna
exageración, lectura en celular); el gerente aprueba y publica.
