#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera docs/reporte_ejecutivo.html a partir de docs/reporte_datos.json.

Solo biblioteca estándar (json, os, html.parser para el chequeo de etiquetas). Sin
dependencias externas, sin llamadas de red. Ejecutar desde cualquier lugar:

    python3 docs/build_reporte.py

Regla de disciplina de este script: ningún número aparece escrito a mano en el HTML.
Todo número visible se obtiene con n(clave)/num(clave), que lo lee de
docs/reporte_datos.json y lo marca como "usado" — al final el script avisa si sobra
algún dato sin usar. Así el JSON es la única fuente de verdad para cifras.
"""
import json
import os
from html.parser import HTMLParser

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "reporte_datos.json")
OUT_PATH = os.path.join(BASE, "reporte_ejecutivo.html")
DOC_OUT_PATH = os.path.join(BASE, "reporte_ejecutivo_documento.html")

with open(DATA_PATH, "r", encoding="utf-8") as fh:
    _RAW = json.load(fh)
D = {k: v for k, v in _RAW.items() if not k.startswith("_")}
USED = set()


def n(key):
    """'valor'+'unidad' del dato `key`, tal como debe leerse en el texto."""
    USED.add(key)
    item = D[key]
    return f"{item['valor']}{item['unidad']}"


def num(key):
    """Valor numérico (float) del dato `key`, para usar en gráficos."""
    USED.add(key)
    item = D[key]
    if "valor_num" in item:
        return float(item["valor_num"])
    return float(str(item["valor"]).replace(".", "").replace(",", "."))


def fuente(key):
    USED.add(key)
    return D[key]["fuente"]


def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --------------------------------------------------------------------------
# Gráficos SVG inline. Todos usan var(--token) para heredar el tema (claro/oscuro)
# sin generar dos versiones de color.
# --------------------------------------------------------------------------

def _fmt3(v):
    return f"{v:.3f}".replace(".", ",")


def _fmt_pct0(v):
    return f"{v:.0f}"


def _fmt1(v):
    return f"{v:.1f}".replace(".", ",")


def _fmt_money0(v):
    return f"{v:,.0f}".replace(",", ".")


def grouped_bar_chart(categories, series, colors, *, fmt=None, unit_suffix="",
                       width=640, height=300, value_max=None, aria=""):
    fmt = fmt or _fmt3
    series_names = list(series.keys())
    n_series = len(series_names)
    n_cat = len(categories)
    all_vals = [v for vals in series.values() for v in vals]
    vmax = value_max if value_max is not None else (max(all_vals) * 1.28 if all_vals else 1)
    m_left, m_right, m_top, m_bottom = 58, 14, 40, 44
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    group_w = plot_w / max(n_cat, 1)
    gap_frac = 0.26
    bar_w = (group_w * (1 - gap_frac)) / max(n_series, 1)
    parts = []
    parts.append(
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" '
        f'role="img" aria-label="{esc(aria)}" xmlns="http://www.w3.org/2000/svg">'
    )
    n_grid = 4
    for i in range(n_grid + 1):
        gy = m_top + plot_h - (plot_h * i / n_grid)
        gval = vmax * i / n_grid
        parts.append(
            f'<line x1="{m_left}" y1="{gy:.1f}" x2="{width - m_right}" y2="{gy:.1f}" '
            f'stroke="var(--border)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{m_left - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="var(--muted)" font-family="var(--font-mono)">{fmt(gval)}{unit_suffix}</text>'
        )
    for ci, cat in enumerate(categories):
        gx0 = m_left + ci * group_w + group_w * gap_frac / 2
        for si, sname in enumerate(series_names):
            val = series[sname][ci]
            bx = gx0 + si * bar_w
            bh = (val / vmax) * plot_h if vmax else 0
            by = m_top + plot_h - bh
            color = colors[sname]
            bw = bar_w * 0.82
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{max(bh, 0):.1f}" '
                f'fill="{color}" rx="2"/>'
            )
            label = f"{fmt(val)}{unit_suffix}"
            ly = by - 6
            parts.append(
                f'<text x="{bx + bw / 2:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="10.3" '
                f'fill="var(--ink)" font-family="var(--font-mono)">{esc(label)}</text>'
            )
        cx = m_left + ci * group_w + group_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{m_top + plot_h + 20:.1f}" text-anchor="middle" '
            f'font-size="12" fill="var(--ink)">{esc(cat)}</text>'
        )
    parts.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h:.1f}" x2="{width - m_right}" '
        f'y2="{m_top + plot_h:.1f}" stroke="var(--ink)" stroke-width="1.2"/>'
    )
    lx, ly = m_left, 18
    for sname in series_names:
        parts.append(f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="11" height="11" fill="{colors[sname]}" rx="2"/>')
        parts.append(f'<text x="{lx + 16:.1f}" y="{ly:.1f}" font-size="11.5" fill="var(--ink)">{esc(sname)}</text>')
        lx += 16 + 6.6 * len(sname) + 26
    parts.append("</svg>")
    return "".join(parts)


def line_chart(x_labels, series, colors, *, fmt=None, unit_suffix="", width=640, height=280):
    fmt = fmt or _fmt3
    all_vals = [v for vals in series.values() for v in vals]
    vmin, vmax = min(all_vals), max(all_vals)
    if vmax == vmin:
        vmax = vmin + 1
    pad = (vmax - vmin) * 0.35
    vmin -= pad
    vmax += pad
    m_left, m_right, m_top, m_bottom = 58, 14, 36, 40
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    ncat = len(x_labels)

    def xpos(i):
        return m_left + (plot_w * i / (ncat - 1) if ncat > 1 else plot_w / 2)

    def ypos(v):
        return m_top + plot_h - (v - vmin) / (vmax - vmin) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    n_grid = 4
    for i in range(n_grid + 1):
        gy = m_top + plot_h * i / n_grid
        gval = vmax - (vmax - vmin) * i / n_grid
        parts.append(f'<line x1="{m_left}" y1="{gy:.1f}" x2="{width - m_right}" y2="{gy:.1f}" stroke="var(--border)"/>')
        parts.append(
            f'<text x="{m_left - 8}" y="{gy + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="var(--muted)" font-family="var(--font-mono)">{fmt(gval)}{unit_suffix}</text>'
        )
    for i, lbl in enumerate(x_labels):
        parts.append(
            f'<text x="{xpos(i):.1f}" y="{m_top + plot_h + 20:.1f}" text-anchor="middle" '
            f'font-size="12" fill="var(--ink)">{esc(lbl)}</text>'
        )
    for sname, vals in series.items():
        pts = " ".join(f"{xpos(i):.1f},{ypos(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[sname]}" stroke-width="2.6"/>')
        for i, v in enumerate(vals):
            parts.append(f'<circle cx="{xpos(i):.1f}" cy="{ypos(v):.1f}" r="4" fill="{colors[sname]}"/>')
    lx, ly = m_left, 18
    for sname in series:
        parts.append(f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="11" height="11" fill="{colors[sname]}" rx="2"/>')
        parts.append(f'<text x="{lx + 16:.1f}" y="{ly:.1f}" font-size="11.5" fill="var(--ink)">{esc(sname)}</text>')
        lx += 16 + 6.6 * len(sname) + 30
    parts.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h:.1f}" x2="{width - m_right}" '
        f'y2="{m_top + plot_h:.1f}" stroke="var(--ink)" stroke-width="1.2"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Componentes de UI reutilizables
# --------------------------------------------------------------------------

def kpi_card(number_text, label, sub=""):
    subhtml = f'<div class="kpi-sub">{esc(sub)}</div>' if sub else ""
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-num">{number_text}</div>'
        f'<div class="kpi-label">{esc(label)}</div>'
        f"{subhtml}"
        "</div>"
    )


def chip(text, level="info"):
    return f'<span class="chip chip-{level}"><span class="chip-dot" aria-hidden="true"></span>{esc(text)}</span>'


def mini_dashboard(titulo, kpis, filas_alerta, chart_svg=None, chart_caption=""):
    """kpis: list[(numero_html, etiqueta)]; filas_alerta: list[(severidad, nombre, detalle, valor)]"""
    kpi_html = "".join(
        f'<div class="md-kpi"><div class="md-kpi-num">{num}</div><div class="md-kpi-label">{esc(lbl)}</div></div>'
        for num, lbl in kpis
    )
    filas_html = "".join(
        f'<tr class="sev-{sev}"><td><span class="stripe stripe-{sev}" aria-hidden="true"></span>{esc(nombre)}</td>'
        f'<td>{esc(detalle)}</td><td class="num">{esc(valor)}</td>'
        f'<td>{chip("atención" if sev == "warn" else ("crítico" if sev == "critical" else "ok"), sev if sev != "critical" else "critico")}</td></tr>'
        for sev, nombre, detalle, valor in filas_alerta
    )
    chart_html = ""
    if chart_svg:
        chart_html = f'<div class="md-chart"><figure>{chart_svg}<figcaption>{esc(chart_caption)}</figcaption></figure></div>'
    return f"""
<div class="mockup" role="group" aria-label="Mockup: {esc(titulo)}">
  <div class="mockup-bar"><span class="mockup-dot"></span><span class="mockup-dot"></span><span class="mockup-dot"></span>
    <span class="mockup-title">{esc(titulo)}</span></div>
  <div class="mockup-body">
    <div class="md-kpis">{kpi_html}</div>
    <div class="md-table-wrap">
      <table class="md-table">
        <thead><tr><th>Alerta</th><th>Detalle</th><th>Valor</th><th>Estado</th></tr></thead>
        <tbody>{filas_html}</tbody>
      </table>
    </div>
    {chart_html}
  </div>
</div>"""


def card_reserva(texto):
    return f'<div class="reserva"><span class="reserva-tag">Reserva principal</span><p>{texto}</p></div>'


def card_fuente(texto):
    return f'<p class="fuente-cita">Fuente: {esc(texto)}</p>'


# --------------------------------------------------------------------------
# CSS (tokens de tema, layout, componentes). Cadena normal (no f-string): las
# llaves de CSS no deben confundirse con interpolación de Python.
# --------------------------------------------------------------------------

CSS = """
:root{
  --bg:#F3F5F7; --ink:#14202B; --surface:#FFFFFF; --surface-2:#E7EAEF;
  --border:#D3DAE1; --muted:#55636E;
  --accent:#B5651D; --accent-ink:#FFFFFF;
  --accent-2:#3F8A7B; --accent-2-ink:#FFFFFF;
  --good:#2E7D46; --good-bg:#E4F3E8;
  --warn:#9A6400; --warn-bg:#FBEDD3;
  --critical:#B3261E; --critical-bg:#FBE4E1;
  --font-display:"Bricolage Grotesque","Segoe UI",system-ui,-apple-system,sans-serif;
  --font-body:"Source Sans 3","Segoe UI",-apple-system,Roboto,sans-serif;
  --font-mono:"IBM Plex Mono","SFMono-Regular",Consolas,"Liberation Mono",monospace;
  --measure:68ch;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#101720; --ink:#E7EEF3; --surface:#161F29; --surface-2:#1E2A35;
    --border:#2B3A47; --muted:#9FB1BE;
    --accent:#E0964F; --accent-ink:#1A1206;
    --accent-2:#5FBBA9; --accent-2-ink:#08201C;
    --good:#63C77F; --good-bg:#123420;
    --warn:#E3AE45; --warn-bg:#3A2B0C;
    --critical:#E88680; --critical-bg:#3B1613;
  }
}
:root[data-theme="dark"]{
  --bg:#101720; --ink:#E7EEF3; --surface:#161F29; --surface-2:#1E2A35;
  --border:#2B3A47; --muted:#9FB1BE;
  --accent:#E0964F; --accent-ink:#1A1206;
  --accent-2:#5FBBA9; --accent-2-ink:#08201C;
  --good:#63C77F; --good-bg:#123420;
  --warn:#E3AE45; --warn-bg:#3A2B0C;
  --critical:#E88680; --critical-bg:#3B1613;
}
*{box-sizing:border-box;}
html{-webkit-text-size-adjust:100%;}
body{
  margin:0; background:var(--bg); color:var(--ink); font-family:var(--font-body);
  font-size:16px; line-height:1.55; padding-inline:16px;
}
a{color:var(--accent);}
a:focus-visible, button:focus-visible, [tabindex]:focus-visible{
  outline:3px solid var(--accent-2); outline-offset:2px; border-radius:3px;
}
h1,h2,h3,h4{font-family:var(--font-display); line-height:1.15; letter-spacing:-0.01em;}
h1{font-size:clamp(1.7rem,4vw,2.6rem); margin:0 0 .3em;}
h2{font-size:clamp(1.3rem,3vw,1.7rem); margin:2.2em 0 .6em; padding-top:.2em; border-top:1px solid var(--border);}
h3{font-size:1.15rem; margin:1.6em 0 .5em;}
h4{font-size:1rem; margin:1.2em 0 .4em;}
p{margin:.6em 0;}
.mono, .kpi-num, .md-kpi-num, .num, table td.num{font-family:var(--font-mono); font-variant-numeric:tabular-nums;}
.medida{max-width:var(--measure);}
.wrap{max-width:1180px; margin:0 auto;}
.layout{display:flex; gap:2.2rem; align-items:flex-start; padding-block:1.4rem 4rem;}
nav.indice{
  position:sticky; top:1rem; flex:0 0 220px; align-self:flex-start;
  background:var(--surface); border:1px solid var(--border); border-radius:10px;
  padding:.9rem .9rem; max-height:calc(100vh - 2rem); overflow-y:auto;
}
nav.indice h2{margin:0 0 .5em; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); border:none; padding:0;}
nav.indice ul{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:.15rem;}
nav.indice a{display:block; padding:.32rem .5rem; border-radius:6px; color:var(--ink); text-decoration:none; font-size:.9rem;}
nav.indice a:hover{background:var(--surface-2);}
main{flex:1 1 auto; min-width:0;}
section{margin-bottom:2.4rem;}
.badge{display:inline-block; font-family:var(--font-mono); font-size:.72rem; font-weight:600;
  background:var(--surface-2); color:var(--muted); border:1px solid var(--border); border-radius:999px;
  padding:.15em .65em; margin-right:.5em;}
.badge-accent{background:var(--accent); color:var(--accent-ink); border-color:var(--accent);}
.kpi-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:.9rem; margin:1.2rem 0;}
.kpi-grid.cinco{grid-template-columns:repeat(auto-fit,minmax(160px,1fr));}
.kpi-card{background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1rem 1.05rem;
  border-top:4px solid var(--accent);}
.kpi-num{font-size:1.7rem; font-weight:600; color:var(--accent); line-height:1.1;}
.kpi-label{margin-top:.35rem; font-size:.92rem;}
.kpi-sub{margin-top:.25rem; font-size:.78rem; color:var(--muted);}
.mensajes{list-style:none; margin:1rem 0; padding:0; display:grid; gap:.7rem;}
.mensajes li{background:var(--surface); border:1px solid var(--border); border-left:5px solid var(--accent-2);
  border-radius:8px; padding:.75rem .95rem;}
.chip{display:inline-flex; align-items:center; gap:.4em; font-size:.78rem; font-weight:600; padding:.2em .65em .22em;
  border-radius:999px; white-space:nowrap;}
.chip-dot{width:.5em; height:.5em; border-radius:50%; display:inline-block;}
.chip-ok{background:var(--good-bg); color:var(--good);}
.chip-ok .chip-dot{background:var(--good);}
.chip-warn, .chip-atención{background:var(--warn-bg); color:var(--warn);}
.chip-warn .chip-dot, .chip-atención .chip-dot{background:var(--warn);}
.chip-critical, .chip-critico{background:var(--critical-bg); color:var(--critical);}
.chip-critical .chip-dot, .chip-critico .chip-dot{background:var(--critical);}
.chip-info{background:var(--surface-2); color:var(--muted);}
.chip-info .chip-dot{background:var(--muted);}
.grid-cap{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:.9rem; margin:1rem 0;}
.cap-card{background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.9rem 1rem;}
.cap-card h4{margin:.1em 0 .3em; color:var(--accent-2);}
.proyecto{background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1.3rem 1.4rem 1.5rem;
  margin:1.3rem 0 2.1rem; scroll-margin-top:1rem;}
.proyecto.encurso{border-style:dashed; border-color:var(--accent-2);}
.proyecto-head{display:flex; flex-wrap:wrap; align-items:baseline; gap:.6rem; margin-bottom:.2rem;}
.proyecto-head h3{margin:0;}
dl.campos{display:grid; grid-template-columns:max-content 1fr; gap:.35rem 1rem; margin:.7rem 0;}
dl.campos dt{font-weight:600; color:var(--muted); font-size:.85rem; padding-top:.15rem;}
dl.campos dd{margin:0;}
.reserva{background:var(--warn-bg); border:1px solid var(--warn); border-radius:10px; padding:.75rem .95rem; margin:1rem 0;}
.reserva-tag{display:inline-block; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
  color:var(--warn); margin-bottom:.3rem;}
.reserva p{margin:.2rem 0 0;}
.fuente-cita{font-size:.78rem; color:var(--muted); border-top:1px dashed var(--border); padding-top:.5rem; margin-top:1rem;}
.chart-wrap{background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.8rem .9rem 1rem; margin:1rem 0;}
.chart-wrap figure{margin:0;}
.chart-wrap figcaption{font-size:.82rem; color:var(--muted); margin-top:.4rem;}
.mockup{border:1px solid var(--border); border-radius:12px; overflow:hidden; margin:1rem 0; background:var(--surface-2);}
.mockup-bar{display:flex; align-items:center; gap:.4rem; background:var(--surface); padding:.5rem .8rem; border-bottom:1px solid var(--border);}
.mockup-dot{width:9px; height:9px; border-radius:50%; background:var(--border);}
.mockup-title{margin-left:.5rem; font-size:.82rem; font-weight:600; color:var(--muted);}
.mockup-body{padding:.9rem;}
.md-kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:.6rem; margin-bottom:.8rem;}
.md-kpi{background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:.55rem .6rem;}
.md-kpi-num{font-size:1.15rem; font-weight:600; color:var(--accent-2);}
.md-kpi-label{font-size:.74rem; color:var(--muted); margin-top:.15rem;}
.md-table-wrap{overflow-x:auto; border:1px solid var(--border); border-radius:8px; background:var(--surface);}
table.md-table, table.tabla{width:100%; border-collapse:collapse; font-size:.86rem;}
table.md-table th, table.md-table td, table.tabla th, table.tabla td{
  padding:.42rem .6rem; border-bottom:1px solid var(--border); text-align:left; vertical-align:top;}
table.md-table th, table.tabla th{font-size:.74rem; text-transform:uppercase; letter-spacing:.03em; color:var(--muted);}
.stripe{display:inline-block; width:5px; height:.9em; margin-right:.5em; border-radius:2px; vertical-align:-1px;}
.stripe-ok{background:var(--good);} .stripe-warn{background:var(--warn);} .stripe-critical{background:var(--critical);}
tr.sev-critical td:first-child{font-weight:600;}
.md-chart{margin-top:.8rem;}
.tabla-wrap{overflow-x:auto; margin:1rem 0;}
.tabla th, .tabla td{white-space:normal;}
.roadmap{list-style:none; counter-reset:paso; margin:1rem 0; padding:0; display:grid; gap:.8rem;}
.roadmap li{position:relative; padding-left:2.6rem; background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding-top:.7rem; padding-bottom:.75rem; padding-right:.9rem;}
.roadmap li::before{counter-increment:paso; content:counter(paso); position:absolute; left:.75rem; top:.65rem;
  width:1.5rem; height:1.5rem; border-radius:50%; background:var(--accent); color:var(--accent-ink);
  font-family:var(--font-mono); font-weight:700; font-size:.85rem; display:flex; align-items:center; justify-content:center;}
.roadmap h4{margin:0 0 .25rem;}
.glosario dt{font-weight:700; font-family:var(--font-display); margin-top:.8rem;}
.glosario dd{margin:.15rem 0 0;}
footer.pie{color:var(--muted); font-size:.8rem; margin:2.5rem 0 1rem; padding-top:1rem; border-top:1px solid var(--border);}
@media (max-width: 880px){
  .layout{flex-direction:column; gap:1.1rem; align-items:stretch;}
  main{width:100%; max-width:100%;}
  nav.indice{position:static; flex:none; width:100%; max-height:none;}
  nav.indice ul{flex-direction:row; flex-wrap:wrap;}
  nav.indice a{font-size:.82rem; padding:.28rem .5rem;}
}
@media print{
  nav.indice{display:none;}
  .layout{display:block; padding:0;}
  body{padding-inline:0;}
  a{color:inherit; text-decoration:none;}
  .proyecto, .kpi-card, .mockup{break-inside:avoid;}
  .chart-wrap{break-inside:avoid;}
}
"""

# --------------------------------------------------------------------------
# Colores de series (siempre var(--token), heredan tema claro/oscuro)
# --------------------------------------------------------------------------
COL_MODELO = "var(--accent)"
COL_BASE = "var(--muted)"
COL_A2 = "var(--accent-2)"
COL_A2_SOFT = "var(--accent-2)"
COL_WARN = "var(--warn)"


# --------------------------------------------------------------------------
# Portada y resumen ejecutivo
# --------------------------------------------------------------------------

def sec_portada():
    kpis = [
        (n("p02_captura10_traspasos"),
         "de los traspasos capturados contactando al 10 % de la cartera",
         f'razón {n("p02_razon_vs_regla")} sobre la mejor regla · churn_traspaso'),
        (n("p03_captura10_monto"), "del ahorro voluntario de adquisición capturado contactando al 10 %",
         f'vs. {n("p03_captura10_monto_renta")} ordenando por renta (práctica actual) · propension_apv'),
        (n("p04_captura10_cese3m"), "de los ceses a 3 meses capturados contactando al 10 % de mayor riesgo",
         "riesgo_cese_cotizacion"),
        (n("p05_captura10_masa"), "de la masa imponible en riesgo capturada gestionando al 10 % de los empleadores",
         f'promedio de 5 semillas: {n("p05_captura10_masa_promedio5")} · riesgo_empleador'),
        (f'{n("p02_licitacion_saldo_riesgo")}', "en riesgo en un solo sorteo de la licitación de 2027",
         f'{n("p02_licitacion_saldo_pct")} del saldo administrado de Cuprum · churn_traspaso'),
    ]
    kpi_html = "".join(kpi_card(num, lbl, sub) for num, lbl, sub in kpis)
    APP_RAW["kpis"] = [{"valor": v, "label": lbl, "sub": sub} for v, lbl, sub in kpis]

    mensajes = [
        "El equipo de Data Analytics completó nueve capacidades analíticas y una actualización sobre la "
        "primera, todas sobre un mismo laboratorio sintético de AFP Cuprum y con revisión adversarial "
        "antes de aprobarse.",
        "Cada modelo mejora la práctica comercial actual con la misma capacidad de contacto: no se pide más "
        "gente, se pide priorizar mejor a quién llamar, visitar o escribir.",
        f'La licitación de afiliados de {n("f165_licitacion_anio")} (Ley 21.735) es el mayor riesgo de fuga de '
        "la historia de Cuprum, y ya existe una simulación de escenarios de retención lista para usar.",
        f'Ninguna cifra de este informe describe a un afiliado real: todo el laboratorio usa datos sintéticos, '
        f'calibrados contra la Ficha Estadística Previsional N° {n("f165_numero")} de la Superintendencia de '
        f'Pensiones ({n("f165_fecha_anio")}).',
        "Tres técnicas nuevas ya se probaron en el sandbox: separar causalmente a quién conviene contactar "
        "ayuda de verdad; una red secuencial para el afiliado y usar bandits para decidir el canal todavía "
        "no le ganan a lo simple.",
    ]
    APP_RAW["mensajes"] = list(mensajes)
    mensajes_html = "".join(f"<li>{m}</li>" for m in mensajes)

    return f"""
<section id="portada">
  <p class="badge badge-accent">Informe ejecutivo</p>
  <h1>Qué ofrece Data Analytics a AFP Cuprum</h1>
  <p class="medida">Un resumen, en una pantalla, de qué construyó el equipo, qué tan bien funciona en un
  sandbox {n("meta_pct_sintetico")}&nbsp;% sintético y qué falta para llevarlo a producción. Pensado para leerse en quince
  minutos, en el celular o impreso.</p>

  <h2 style="border-top:none; margin-top:1rem;">Cinco mensajes</h2>
  <ul class="mensajes medida">{mensajes_html}</ul>

  <h2>Cinco números</h2>
  <div class="kpi-grid cinco">{kpi_html}</div>
  <p class="fuente-cita">Todas las cifras de esta portada son sintéticas (sandbox del laboratorio), salvo el
  saldo en riesgo de la licitación, que también es una simulación sobre el universo sintético calibrado
  contra la Ficha 165. Detalle y fuente exacta de cada número en su ficha de proyecto y en
  <code>docs/reporte_datos.json</code>.</p>
</section>
"""


# --------------------------------------------------------------------------
# Catálogo de capacidades
# --------------------------------------------------------------------------

def sec_capacidades():
    chart_densidad = grouped_bar_chart(
        ["Sistema", "Cuprum"],
        {
            "Antes (ronda 2)": [num("dens_sistema_antes"), num("dens_cuprum_antes")],
            "Después (ronda 3)": [num("dens_sistema_despues"), num("dens_cuprum_despues")],
            "Ficha 165 (real)": [num("f165_densidad_sistema"), num("f165_densidad_cuprum_memoria")],
        },
        {"Antes (ronda 2)": COL_BASE, "Después (ronda 3)": COL_A2, "Ficha 165 (real)": COL_MODELO},
        fmt=_fmt1, unit_suffix=" %", width=620, height=260, value_max=100,
        aria="Densidad de cotización del universo sintético antes y después de la recalibración, vs. la Ficha 165",
    )
    capacidades = [
        ("Enriquecer", "Reconstruir hoy el dato que todavía no llegó",
         "rim_nowcasting proyecta la renta imponible de los dos meses que la AFP aún no conoce, y la "
         "reemplaza por el dato real en cuanto el empleador declara."),
        ("Proyectar", "Estimar un agregado del mes o del trimestre que viene",
         "riesgo_cese_cotizacion y riesgo_empleador proyectan cuántos afiliados dejarán de cotizar y "
         "cuánta masa imponible no se pagará, para el presupuesto de recaudación."),
        ("Segmentar", "Describir la cartera en grupos que se comportan distinto",
         "segmentacion_afiliados divide la cartera en cuatro perfiles de comportamiento, cada uno con "
         "una acción comercial distinta."),
        ("Priorizar", "Ordenar la cartera para una capacidad de contacto limitada",
         "churn_traspaso, propension_apv, riesgo_cese_cotizacion y riesgo_empleador ordenan a quién "
         "llamar, visitar o gestionar primero, con el mismo equipo de siempre."),
        ("Decidir", "Elegir la acción que de verdad cambia el resultado",
         "uplift_retencion separa a quién conviene contactar de quién simplemente tiene riesgo; "
         "bandit_contacto aprende, contacto a contacto, qué canal y oferta usar."),
        ("Cumplir normativa", "Hablar el mismo idioma que la planilla real",
         "La planilla Previred sintética de common/ usa los códigos oficiales de la Tabla N° 7, para que "
         "los modelos entrenen con la misma estructura de datos que verían en producción."),
    ]
    APP_RAW["capacidades"] = [{"titulo": t, "sub": s, "desc": d} for t, s, d in capacidades]
    APP_RAW["construccion"] = {
        "chart_densidad": chart_densidad,
        "kpis": [
            {"valor": n("meta_tests_total"), "label": "tests automatizados en la suite del laboratorio", "sub": f'{n("meta_tests_fallos")} fallos · {n("meta_tests_runtime")}'},
            {"valor": n("f165_densidad_sistema"), "label": "densidad de cotización del sistema (Ficha 165)", "sub": "el universo sintético se recalibró contra esta cifra en la ronda 3"},
            {"valor": n("f165_densidad_cuprum_memoria"), "label": "densidad de cotización propia de Cuprum", "sub": "la más alta del sistema; memoria 2025"},
        ],
        "chart_caption": (
            f'Densidad de cotización del universo sintético antes y después de la recalibración de la ronda 3, '
            f'contra el objetivo real de la Ficha 165 (sistema {n("f165_densidad_sistema")}, Cuprum {n("f165_densidad_cuprum_memoria")}).'
        ),
    }
    cap_html = "".join(
        f'<div class="cap-card"><h4>{esc(t)}</h4><p style="font-size:.86rem;color:var(--muted);margin:0 0 .3rem;">{esc(s)}</p>'
        f'<p style="font-size:.9rem;margin:0;">{d}</p></div>'
        for t, s, d in capacidades
    )

    return f"""
<section id="capacidades">
  <h2>Qué ofrece el equipo</h2>
  <p class="medida">Seis capacidades, construidas sobre un mismo laboratorio de datos sintéticos. Cada una
  responde una pregunta de negocio concreta y deja código, tests y un README con resultados y supuestos.</p>
  <div class="grid-cap">{cap_html}</div>

  <h3>Cómo se construyó</h3>
  <p class="medida">Todos los proyectos parten del mismo universo sintético (afiliados, empleadores, panel
  laboral mensual y macroeconomía) calibrado contra la Ficha 165 de la Superintendencia de Pensiones, para
  que un afiliado sea "la misma persona" en todos los modelos. Cada proyecto nuevo pasa por una revisión
  adversarial independiente antes de aprobarse: se buscan fugas de información desde el futuro (leakage),
  reglas de negocio degeneradas y afirmaciones que no reproducen. El detalle de qué se revisó y con qué
  veredicto está en <code>docs/revisiones/</code> (por proyecto, en el anexo técnico). El resultado se
  reporta aunque la técnica pierda contra lo simple &mdash; es el criterio que se siguió con las tres
  técnicas más nuevas del portafolio (causalidad, red secuencial y bandits).</p>
  <div class="kpi-grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr));">
    {kpi_card(n("meta_tests_total"), "tests automatizados en la suite del laboratorio", f'{n("meta_tests_fallos")} fallos · {n("meta_tests_runtime")}')}
    {kpi_card(n("f165_densidad_sistema"), "densidad de cotización del sistema (Ficha 165)", "el universo sintético se recalibró contra esta cifra en la ronda 3")}
    {kpi_card(n("f165_densidad_cuprum_memoria"), "densidad de cotización propia de Cuprum", "la más alta del sistema; memoria 2025")}
  </div>
  <div class="chart-wrap"><figure>{chart_densidad}<figcaption>Densidad de cotización del universo
  sintético antes y después de la recalibración de la ronda 3, contra el objetivo real de la Ficha 165
  (sistema {n("f165_densidad_sistema")}, Cuprum {n("f165_densidad_cuprum_memoria")}).</figcaption></figure></div>
  <p class="fuente-cita">Sin jerga de modelos en esta página: el detalle técnico de cada proyecto (datos,
  modelo, métricas, tests, tiempos de ejecución y revisión) está en el anexo técnico, al final del informe.</p>
</section>
"""


# --------------------------------------------------------------------------
# Casos de uso: una tarjeta por proyecto
# --------------------------------------------------------------------------

CASE_REGISTRY = []
APP_RAW = {}


def proyecto_card(*, id_, numero, nombre, titulo, problema, que_hace, resultado_html,
                   uso_html, mockup_html, produccion_html, reserva_html, fuente_texto,
                   estado_chip=""):
    CASE_REGISTRY.append(dict(
        id_=id_, numero=numero, nombre=nombre, titulo=titulo, problema=problema,
        que_hace=que_hace, resultado_html=resultado_html, uso_html=uso_html,
        mockup_html=mockup_html, produccion_html=produccion_html,
        reserva_html=reserva_html, fuente_texto=fuente_texto, estado_chip=estado_chip,
    ))
    chip_html = f'<span class="chip chip-info">{esc(estado_chip)}</span>' if estado_chip else ""
    return f"""
<article class="proyecto" id="{id_}">
  <div class="proyecto-head">
    <span class="badge badge-accent">{esc(numero)}</span>
    <h3>{esc(titulo)}</h3>
    <span class="badge">{esc(nombre)}</span>
    {chip_html}
  </div>
  <dl class="campos medida">
    <dt>Problema</dt><dd>{problema}</dd>
    <dt>Qué hace</dt><dd>{que_hace}</dd>
    <dt>Resultado en el sandbox</dt><dd>{resultado_html}</dd>
    <dt>Cómo lo usaría el área</dt><dd>{uso_html}</dd>
  </dl>
  {mockup_html}
  <h4>Camino a producción</h4>
  <div class="medida">{produccion_html}</div>
  {reserva_html}
  {card_fuente(fuente_texto)}
</article>
"""


def _card02():
    chart = grouped_bar_chart(
        ["10 % (referencia)", "20 % (doble equipo)"],
        {"Captura de traspasos": [num("p02_captura10_traspasos"), num("p02_captura20_traspasos")]},
        {"Captura de traspasos": COL_MODELO},
        fmt=_fmt1, unit_suffix=" %", width=560, height=220, value_max=100,
        aria="Captura de traspasos al 10 % y al 20 % de cobertura",
    )
    resultado = (
        f'Con el modelo, contactando a {n("cap10")} de la cartera se captura el {n("p02_captura10_traspasos")} '
        f'de los traspasos y el {n("p02_captura10_saldo")} del saldo en riesgo &mdash; {n("p02_razon_vs_regla")} '
        f'la captura de la mejor regla de negocio (AUC {n("p02_auc_modelo")} del modelo vs. {n("p02_auc_baseline")} '
        f'de esa regla, semilla de referencia, perfil sistema). En la cartera Cuprum el AUC va de '
        f'{n("p02_cuprum_auc_rango")} según la semilla. Subir la cobertura a {n("cap20")} agrega captura '
        f'({n("p02_captura20_traspasos")} de los traspasos) con rendimiento decreciente.'
    )
    mockup = mini_dashboard(
        "Torre de retención · churn_traspaso",
        [(n("p02_captura10_traspasos"), "traspasos capturados (10 % cartera)"),
         (n("p02_razon_vs_regla"), "veces la mejor regla"),
         (n("p02_licitacion_saldo_riesgo"), "en riesgo, licitación 2027")],
        [
            ("critical", "Sorteo licitación 2027", "afiliado premium, alto saldo, sin campaña asignada", n("p02_licitacion_saldo_pct") + " del saldo Cuprum"),
            ("warn", "Antigüedad baja + llegó por traspaso", "primer decil de riesgo de churn voluntario", n("p02_captura10_traspasos") + " del riesgo capturado aquí"),
            ("ok", "Empresa en campaña de la competencia", "colegas del mismo empleador, aún sin visitar", "priorizar por pares"),
        ],
        chart, f'Captura de traspasos al 10 % y al 20 % de cobertura (perfil sistema, semilla {n("semilla42")}).',
    )
    chart_licitacion = grouped_bar_chart(
        [f'{n("p02_esc_pct_baja")} se va', f'{n("p02_esc_pct_ref")} (referencia)', f'{n("p02_esc_pct_alta")} se va'],
        {
            "Prioridad saldo\u00d7probabilidad": [num("p02_esc55_prioridad"), num("p02_escenario_saldo_retenido_valor"), num("p02_esc75_prioridad")],
            "Contacto aleatorio": [num("p02_esc55_azar"), num("p02_escenario_saldo_retenido_azar"), num("p02_esc75_azar")],
        },
        {"Prioridad saldo\u00d7probabilidad": COL_MODELO, "Contacto aleatorio": COL_BASE},
        fmt=_fmt_money0, width=620, height=250,
        aria="Saldo retenido en la licitacion de 2027 segun el supuesto de salida pasiva y la forma de priorizar",
    )
    chart_licitacion_html = (
        '<div class="chart-wrap"><figure>' + chart_licitacion +
        f'<figcaption>Saldo retenido (MM CLP) con {n("p02_esc_capacidad")} de capacidad de contacto sobre los '
        f'sorteados de la licitación de {n("f165_licitacion_anio")}, según el supuesto de cuántos se van sin '
        f'campaña ({n("p02_esc_pct_baja")}/{n("p02_esc_pct_ref")}/{n("p02_esc_pct_alta")}) y cómo se prioriza. '
        "El sorteo elige al azar quién queda expuesto; a quién CONTACTAR dentro de ese "
        "grupo sí se puede elegir.</figcaption></figure></div>"
    )
    produccion = (
        "Se necesita la tabla real de solicitudes de traspaso (SIAP/Previred), el historial de contactos "
        "del equipo comercial y las comisiones y rentabilidades publicadas de las seis AFP. Antes de "
        f'producción: repetir la validación temporal con gap de {n("horizonte6")} y exigir AUC ≥ '
        f'{n("p02_auc_umbral_real")} con datos reales (hoy el umbral del sandbox es {n("p02_auc_modelo")}). '
        "Riesgo regulatorio principal: la licitación de stock de la Ley 21.735 cambia las reglas de "
        "competencia cada dos años y debe gestionarse con presupuesto de contacto separado del churn "
        "voluntario."
    )
    reserva = card_reserva(
        "El saldo en riesgo esperado (probabilidad × saldo, agregado a nivel de cartera) no se entrega en "
        f'esta ronda: el sesgo del pronóstico agregado va de {n("p02_sesgo_min")} a {n("p02_sesgo_max")} entre semillas, y es un '
        "problema de nivel del pronóstico, no de la calibración del modelo. El ranking para priorizar "
        "contactos sí es confiable; el número que se publicaría ('se van X mil millones el próximo "
        "semestre') todavía no lo es."
    )
    return proyecto_card(
        id_="p02", numero="02", nombre="churn_traspaso",
        titulo="Traspaso a otra AFP",
        problema="¿Qué afiliados se traspasarán a otra AFP en los próximos seis meses, y a quién debería "
                 "llamar o visitar el ejecutivo de retención con la capacidad de contacto que existe hoy?",
        que_hace="Ordena a la cartera vigente por probabilidad de traspaso a seis meses, usando historia "
                 "laboral, antigüedad en la AFP, comisión y rentabilidad relativa, y señales comerciales "
                 "(reclamos, visitas de la competencia, uso digital).",
        resultado_html=resultado,
        uso_html="El ejecutivo de retención recibe cada mes la lista del 10 % de mayor riesgo, ordenada "
                 "además por saldo cuando el objetivo es defender pesos administrados, no solo cuentas. "
                 "En años de licitación de stock, esa lista se separa de la lista de sorteados: son "
                 "presupuestos y urgencias distintas.",
        mockup_html=mockup + chart_licitacion_html, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/churn_traspaso/README.md §§1, 6, 7, 13.A, 13.B",
    )


def _card03():
    resultado = (
        f'Contactando al {n("cap10")} de la cartera elegible ordenada por monto esperado se captura el '
        f'{n("p03_captura10_monto")} del ahorro voluntario de adquisición, frente al {n("p03_captura10_monto_renta")} '
        f'que se captura hoy ordenando por renta (AUC {n("p03_auc")} del modelo vs. {n("p03_auc_regla")} de la '
        f'regla de negocio). El sesgo del monto esperado para toda la población es de {n("p03_sesgo_monto")} '
        f'en el agregado del semestre (WAPE {n("p03_wape_monto")} en los aportantes).'
    )
    chart = grouped_bar_chart(
        ["Modelo (monto esperado)", "Orden por renta (hoy)"],
        {"Captura al 10 %": [num("p03_captura10_monto"), num("p03_captura10_monto_renta")]},
        {"Captura al 10 %": COL_MODELO},
        fmt=_fmt1, unit_suffix=" %", width=560, height=220, value_max=100,
        aria="Captura de ahorro voluntario al 10 % de la cartera, modelo vs. orden por renta",
    )
    mockup = mini_dashboard(
        "Campaña de ahorro voluntario · propension_apv",
        [(n("p03_captura10_monto"), "del ahorro de adquisición capturado"),
         (n("p03_captura10_aportantes"), "de los nuevos aportantes"),
         (n("p03_auc"), "AUC de test")],
        [
            ("ok", "Reactivación con empresa APVC", "aportó hace meses, empleador con match, renta alta", "prob. alta, monto alto"),
            ("warn", "Segmento VOLUNTARIO", "cotización errática, casi no aporta", "excluir de la campaña"),
            ("ok", "Marzo/abril en finanzas y minería", "bono anual coincide con el aporte esporádico", "estacionalidad"),
        ],
        chart, "Captura del 10 % de la cartera: modelo (monto esperado) vs. orden por renta.",
    )
    produccion = (
        "Se necesita el historial real de aportes voluntarios, campañas y saldo voluntario, y la renta "
        "bruta declarada al SII (la renta imponible que hoy usa el modelo está topada, y subestima el "
        "tramo de impuesto de las rentas altas). Antes de producción: confirmar con la AFP si el aporte "
        "por planilla llega sin desfase, como se asumió, o con el mismo desfase que la cotización "
        "obligatoria."
    )
    reserva = card_reserva(
        f'El modelo no ordena bien al segmento VOLUNTARIO ({n("p03_voluntario_pct")} de la cartera elegible): ahí la regla de '
        "negocio le gana. El protocolo de validación deja el entrenamiento tres a cuatro años antes del "
        "test, lo que es conservador (en producción, con reentrenamiento trimestral, la distancia real "
        "sería de doce meses), pero significa que estos números son un piso, no un techo."
    )
    return proyecto_card(
        id_="p03", numero="03", nombre="propension_apv",
        titulo="Ahorro previsional voluntario",
        problema="¿A quién llama el ejecutivo este mes para abrir o reactivar un APV, y cuánto ahorro "
                 "voluntario se puede proyectar para el semestre?",
        que_hace="Estima, para cada afiliado sin aporte reciente, la probabilidad de aportar en los "
                 "próximos seis meses y el monto esperado, combinando renta, historial de aportes, "
                 "tramo tributario y si el empleador ofrece ahorro colectivo (APVC).",
        resultado_html=resultado,
        uso_html="La asesoría previsional prioriza la lista por monto esperado cuando la meta es CLP "
                 "captados, y por probabilidad cuando la meta es número de cuentas nuevas. Cuprum es "
                 "líder de mercado en APV, así que este modelo también sirve para defender aportantes "
                 "frente al fondo mutuo o el seguro que compiten por el mismo ahorro.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/propension_apv/README.md §§1, 6, 7, 8",
    )


def _card04():
    chart = grouped_bar_chart(
        ["Cese 3 m", "Cese 6 m", "Reingreso"],
        {
            "AUC modelo": [num("p04_auc_cese3m"), num("p04_auc_cese6m"), num("p04_auc_reingreso_actual")],
        },
        {"AUC modelo": COL_MODELO},
        fmt=_fmt3, width=560, height=230, value_max=1.0,
        aria="AUC del modelo de cese y reingreso, con la corrección posterior a la ronda 3 hecha en la ronda 4",
    )
    resultado = (
        f'El modelo de cese llega a AUC {n("p04_auc_cese3m")} a tres meses y {n("p04_auc_cese6m")} a seis '
        f'meses (baseline de regla de negocio: {n("p04_auc_regla_cese3m")} a tres meses); contactando al '
        f'{n("cap10")} de mayor riesgo se captura alrededor del {n("p04_captura10_cese3m")} de los ceses '
        "reales (medido antes de la última recalibración de densidad de cotización; ver reserva). "
        f'El modelo de reingreso, en cambio, hoy da AUC {n("p04_auc_reingreso_actual")}, bajo el umbral '
        f'de {n("p04_auc_reingreso_umbral")} del brief.'
    )
    mockup = mini_dashboard(
        "Alertas de cese y reingreso · riesgo_cese_cotizacion",
        [(n("p04_auc_cese3m"), "AUC cese 3 m"), (n("p04_captura10_cese3m"), "ceses capturados al 10 %"),
         (n("p04_auc_reingreso_actual"), "AUC reingreso (bajo el umbral)")],
        [
            ("critical", "Reingreso poco predecible", "AUC actual bajo el umbral del brief", f'{n("p04_auc_reingreso_actual")} < {n("p04_auc_reingreso_umbral")}'),
            ("warn", "Cesantía en curso, sin señales de reingreso", "término AFC reciente, cobertura de cesantía activa", "prioridad de asesoría"),
            ("ok", "Cese a 3 y 6 meses", "sigue por sobre el umbral del brief tras la recalibración", f'{n("p04_auc_cese3m")} / {n("p04_auc_cese6m")}'),
        ],
        chart, f'AUC del modelo por objetivo, con la corrección posterior a la ronda 3 (ronda 4; perfil sistema, semilla {n("semilla42")}).',
    )
    produccion = (
        "Se necesita el feed real de eventos del Seguro de Cesantía (AFC) y de la planilla previsional con "
        "su desfase real de recepción. El modelo de cese está listo para un piloto de alertas de asesoría; "
        "el de reingreso necesita, primero, que el generador sintético distinga cesantía reciente de "
        "cesantía crónica antes de volver a intentarlo, y luego repetir la validación con datos reales."
    )
    reserva = card_reserva(
        f'El AUC de reingreso cayó de {n("p04_auc_reingreso_antes_densidad")} a {n("p04_auc_reingreso_tras_densidad")} '
        "(sistema) cuando se recalibró la densidad de cotización del generador contra la Ficha 165 en la "
        f'ronda 3. Una corrección posterior a la ronda 3, hecha en la ronda 4 (features del último empleo '
        f'antes del cese), lo subió a {n("p04_auc_reingreso_actual")} (+{n("p04_fix_delta_min")} en la fila '
        f'de referencia; rango +{n("p04_fix_delta_min")} a +{n("p04_fix_delta_max")}, y un caso de '
        f'-{n("p04_fix_delta_neg")}), sin recuperar el umbral de {n("p04_auc_reingreso_umbral")}. En '
        "palabras del propio informe, 'la dinámica actual del generador dejó el reingreso casi "
        "impredecible a nivel individual'. No es un problema de este modelo: es un límite del sandbox que "
        "hay que resolver en el generador antes de la próxima ronda. Nota adicional: la captura al 10 % "
        f'de {n("p04_captura10_cese3m")} / {n("p04_captura10_cese6m")} y el baseline de regla de negocio '
        f'({n("p04_auc_regla_cese3m")}) se midieron ANTES de esta recalibración de densidad y no se '
        "volvieron a medir después; se muestran igual, con esta salvedad explícita, porque el mecanismo de "
        "captura no cambió, solo la población. Decisión: la cabeza de reingreso no se usa en producción "
        "hasta ajustar el generador (probabilidad de contratación dependiente de la duración del cese) o "
        "validar con datos reales; las cabezas de cese a 3 y 6 meses sí quedan disponibles."
    )
    return proyecto_card(
        id_="p04", numero="04", nombre="riesgo_cese_cotizacion",
        titulo="Cese y reingreso de cotización",
        problema="¿Quién dejará de cotizar en tres y seis meses, y quién de los que no cotizan hoy va a "
                 "volver a hacerlo?",
        que_hace="Dos familias de modelos sobre la misma historia laboral: uno de riesgo de cese (para "
                 "asesoría proactiva y proyección de recaudación) y uno de probabilidad de reingreso (para "
                 "priorizar a quién ofrecer cotización voluntaria mientras dura la laguna).",
        resultado_html=resultado,
        uso_html="El número esperado de ceses por mes alimenta el presupuesto de recaudación esperada; el "
                 "10 % de mayor riesgo de cese recibe asesoría proactiva &mdash; estas dos cabezas ya están "
                 "disponibles. La cabeza de reingreso queda diseñada (la campaña útil no sería 'a quién va "
                 "a reingresar', sino la cola de menor probabilidad, filtrada por años a la pensión) pero "
                 "no se usa en producción todavía: ver la decisión en la reserva.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/riesgo_cese_cotizacion/README.md §§1, 6, 7, 12",
        estado_chip="Corrección aprobada (revisión ronda 4, docs/revisiones/ronda_4/cese_correccion.md)",
    )


def _card05():
    chart = grouped_bar_chart(
        ["10 % (referencia)", "20 % (doble equipo)"],
        {"Masa en riesgo capturada": [num("p05_captura10_masa"), num("p05_captura20_masa")]},
        {"Masa en riesgo capturada": COL_MODELO},
        fmt=_fmt1, unit_suffix=" %", width=560, height=220, value_max=100,
        aria="Captura de masa imponible en riesgo al 10 % y al 20 % de los empleadores gestionados",
    )
    resultado = (
        f'Gestionando al {n("cap10")} de los empleadores (unos {n("p05_empleadores")} en el universo de '
        f'referencia) se cubre el {n("p05_captura10_masa")} de la masa imponible en riesgo en esta corrida '
        f'&mdash; {n("p05_captura10_masa_promedio5")} en promedio de cinco semillas &mdash; contra '
        f'{n("p05_captura10_regla")} con la regla vigente (AUC {n("p05_auc")} vs. {n("p05_auc_regla")}). '
        f'La precisión de la lista sube de {n("p05_precision_lista_regla")} a {n("p05_precision_lista_modelo")}: '
        "una de cada cinco llamadas cae en un empleador que de verdad no va a pagar, contra una de cada "
        "nueve con la regla."
    )
    mockup = mini_dashboard(
        "Cobranza previsional · riesgo_empleador",
        [(n("p05_captura10_masa"), "masa en riesgo capturada al 10 %"),
         (n("p05_precision_lista_modelo"), "precisión de la lista"),
         (n("p05_auc"), "AUC de test")],
        [
            ("critical", "Empleador con boletín de infractores", "12 meses consecutivos de deuda", "masa esperada alta"),
            ("warn", "Segmento GRANDE", "el modelo ya gana aquí, pero con muy pocos casos de DNP", "vigilar, no reemplazar señales de respaldo"),
            ("ok", "Sector público", "nunca cae en declaración y no pago", "0 casos, fuera de la gestión"),
        ],
        chart, "Masa imponible en riesgo capturada al 10 % y al 20 % de los empleadores gestionados.",
    )
    produccion = (
        "Se necesita la planilla Previred real con su calendario de recepción y el Boletín de Infractores "
        "de la Dirección del Trabajo. El agregado mensual de masa en riesgo sigue siendo demasiado "
        "volátil para comprometerse mes a mes: escalar a más empleadores lo mejora de forma predecible "
        "(la volatilidad cae como uno sobre la raíz del número de empleadores), así que ese es el primer "
        "paso antes de prometer un número mensual al negocio."
    )
    reserva = card_reserva(
        f'La captura de masa en riesgo al {n("cap10")} queda en {n("p05_captura10_masa")} en la semilla de '
        f'referencia, bajo el umbral de aceptación de {n("p05_umbral_captura")}, aunque el promedio de cinco semillas '
        f'({n("p05_captura10_masa_promedio5")}) se acerca. En palabras del propio informe, sigue siendo '
        f'\'{n("p05_captura_rango_texto")} de la masa con el {n("cap10")} de la cartera, mejor que la regla en todos los casos\', no un '
        "cumplimiento sólido. El error del agregado mensual tampoco cumple su criterio de ±10 %: mejoró "
        "de 0 a 2 de 12 meses al triplicar los empleadores, pero sigue lejos."
    )
    return proyecto_card(
        id_="p05", numero="05", nombre="riesgo_empleador",
        titulo="Riesgo de empleador (cobranza)",
        problema="¿Qué empleadores van a declarar y no pagar, o van a pagar tarde este mes, y cuánta masa "
                 "imponible de nuestros afiliados queda en riesgo?",
        que_hace="Estima, por empleador y mes, la probabilidad de declaración-y-no-pago y la masa "
                 "imponible esperada en riesgo, a partir del historial de puntualidad, el Boletín de "
                 "Infractores y episodios de estrés financiero.",
        resultado_html=resultado,
        uso_html="Cobranza trabaja dos listas: la ordenada por probabilidad (para maximizar casos "
                 "gestionados) y la ordenada por masa esperada (para maximizar CLP protegidos), según el "
                 "canal disponible ese mes (masivo telefónico vs. ejecutivo dedicado).",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/riesgo_empleador/README.md §§1, 6.1, 6.3, 6.4, 7",
    )


def _card01():
    resultado = (
        f'A un mes de desfase, el modelo llega a WAPE {n("p01_wape_h1")} en los afiliados dependientes '
        f'(contra {n("p01_wape_h1_persistencia")} de simplemente repetir el último dato conocido), igual '
        f'en la cartera sistema que en la Cuprum. A dos meses el error sube a {n("p01_wape_h2_sistema")} en '
        f'el sistema y {n("p01_wape_h2_cuprum")} en Cuprum: pesa más la remuneración variable de finanzas '
        f'y minería, que se conoce con más rezago. El clasificador de "cotiza o no" llega a AUC '
        f'{n("p01_auc_cotiza")} a un mes y {n("p01_auc_cotiza_h2")} a dos meses.'
    )
    mockup = mini_dashboard(
        "Recaudación proyectada · rim_nowcasting",
        [(n("p01_wape_h1"), "WAPE a un mes (dependientes)"), (n("p01_wape_h2_cuprum"), "WAPE a dos meses (Cuprum)"),
         (n("p01_auc_cotiza"), "AUC cotiza/no cotiza, a un mes")],
        [
            ("ok", "Reemplazo mensual PREDICHA → REAL", "los pagos que llegan sustituyen la proyección", "ciclo automático"),
            ("warn", "Independientes", "renta anual vía SII, se redibuja cada año", "WAPE alto, modelo aparte recomendado"),
            ("ok", "Multiempleo y licencias", "un segundo pagador o un subsidio llegan con más rezago", "PARCIAL hasta completarse"),
        ],
        None, "",
    )
    produccion = (
        "El pipeline ya está diseñado para escala real: PySpark, particionado por mes de recepción, sin "
        "colecciones al driver. Falta reemplazar el generador sintético por las tablas reales de Previred, "
        "SII y el Seguro de Cesantía, y decidir la cadencia de reentrenamiento (trimestral, o disparado "
        "cuando el error de los reemplazos mensuales supere un umbral)."
    )
    reserva = card_reserva(
        f'Este resultado es de la ronda 2, con la densidad de cotización previa a la recalibración de la '
        f'ronda 3 contra la Ficha {n("f165_numero")}. La corrección de la ronda 3 que sí se hizo sobre '
        "este proyecto (código de movimiento de personal, ficha siguiente) no tocó la densidad; como 01 "
        "comparte el generador base con el resto del laboratorio, la pregunta de si el nowcasting cambia "
        "con la densidad ya recalibrada, como pasó en 02-06, sigue abierta."
    )
    return proyecto_card(
        id_="p01", numero="01", nombre="rim_nowcasting",
        titulo="Renta imponible en tiempo real (nowcasting)",
        problema="La cotización de un afiliado llega a la AFP con dos meses de desfase. ¿Cuál es su renta "
                 "imponible en esos dos meses que todavía no se conocen?",
        que_hace="Proyecta la renta imponible de los dos meses abiertos con un modelo en dos etapas "
                 "(¿cotiza o no?, y si cotiza, cuánto), y la reemplaza automáticamente por el dato real en "
                 "cuanto el empleador declara.",
        resultado_html=resultado,
        uso_html="Alimenta cualquier reporte que necesite un número de recaudación o de saldo 'hoy', sin "
                 "esperar dos meses: proyección de recaudación, saldo estimado de la cuenta obligatoria, "
                 "e insumo de historia laboral para los demás modelos del laboratorio.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/rim_nowcasting/README.md §§1, 10, 10b",
    )


def _card06():
    chart = grouped_bar_chart(
        ["Cese", "Traspaso", "APV"],
        {"Razón entre el segmento de mayor y menor tasa": [num("p06_razon_cese"), num("p06_razon_traspaso"), num("p06_razon_apv")]},
        {"Razón entre el segmento de mayor y menor tasa": COL_A2},
        fmt=_fmt1, unit_suffix="×", width=560, height=230, value_max=35,
        aria="Razón entre la tasa de negocio del segmento más propenso y el menos propenso",
    )
    resultado = (
        f'Cuatro segmentos de comportamiento (silueta {n("p06_silueta")}, estable al re-muestrear la '
        f'cartera: ARI {n("p06_ari_remuestreo")}; un clasificador los reproduce con {n("p06_clasificador_exactitud")} '
        f'de exactitud). Las tasas de cese, traspaso y aporte APV difieren {n("p06_razon_cese")}, '
        f'{n("p06_razon_traspaso")} y {n("p06_razon_apv")} entre el segmento más y menos propenso: son '
        "grupos que de verdad se comportan distinto, no una partición estadística sin sentido de negocio."
    )
    mockup = mini_dashboard(
        "Mapa de segmentos · segmentacion_afiliados",
        [(n("p06_segmentos_n"), "segmentos de comportamiento"), (n("p06_razon_apv"), "razón de aporte APV entre segmentos"),
         (n("p06_ari_temporal"), "estabilidad a 6 meses (ARI)")],
        [
            ("ok", "Segmento 1 · Ahorradores premium", "mayor renta y saldo, líder en aporte APV", "retención de valor prioritaria"),
            ("warn", "Segmento 3 · Intermitentes", "renta y densidad actuales muy bajas", "mayor tasa de cese"),
            ("ok", "Segmento 2 · Joven en acumulación", "el más joven, fondo A/B, sin APV", "educación financiera + oferta de entrada"),
        ],
        chart, "Razón de tasas de negocio entre el segmento más y menos propenso, por indicador.",
    )
    produccion = (
        "El clustering usa exclusivamente tablas ya construidas por 02, 03 y el panel laboral propio, así "
        "que el camino más corto a producción es correrlo directamente sobre esas mismas fuentes reales "
        "una vez que 02 y 03 estén en producción. La asignación de un afiliado nuevo a un segmento no "
        "requiere reclusterizar: el clasificador ya entrenado lo hace en línea."
    )
    reserva = card_reserva(
        f'La estabilidad entre re-muestreos de la misma cartera es alta ({n("p06_ari_remuestreo")}), pero la '
        f'estabilidad entre dos cortes separados por seis meses es más baja ({n("p06_ari_temporal")}): parte '
        "es deriva real del negocio (alguien deja de cotizar, empieza a aportar), no solo ruido del método, "
        "y el informe lo reporta tal cual, sin forzarlo al mismo umbral. El traspaso es, de las tres tasas "
        "de negocio, la que separa peor entre segmentos."
    )
    return proyecto_card(
        id_="p06", numero="06", nombre="segmentacion_afiliados",
        titulo="Segmentación de afiliados",
        problema="¿Qué segmentos de comportamiento existen en la cartera de Cuprum, y qué acción "
                 "corresponde a cada uno?",
        que_hace="Agrupa a toda la cartera (no solo dependientes) por renta, densidad de cotización, "
                 "saldo, fondo, ahorro voluntario y uso digital, y valida los grupos contra las tasas "
                 "reales de cese, traspaso y aporte APV de cada uno.",
        resultado_html=resultado,
        uso_html="Asignación de ejecutivos, priorización de campañas de retención y de APV, y diseño de "
                 "mensajes de asesoría previsional distintos por segmento &mdash; una decisión operativa "
                 "mensual, no un modelo de un evento puntual.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/segmentacion_afiliados/README.md §§1, 6, 7, 8",
    )


def _card07():
    chart = grouped_bar_chart(
        ["10 % de la cartera", "20 % de la cartera"],
        {
            "Separar causalidad (X-learner)": [num("p07_evitados_xlearner10"), num("p07_evitados_xlearner20")],
            "Ordenar por riesgo (02)": [num("p07_evitados_riesgo10"), num("p07_evitados_riesgo20")],
        },
        {"Separar causalidad (X-learner)": COL_A2, "Ordenar por riesgo (02)": COL_BASE},
        fmt=_fmt1, width=580, height=240,
        aria="Traspasos evitados contactando al 10 % y 20 % de la cartera, causalidad vs. riesgo",
    )
    resultado = (
        f'Separando el efecto causal del contacto (X-learner) se evitan {n("p07_evitados_xlearner10")} '
        f'traspasos contactando al {n("cap10")} de la cartera, {n("p07_razon_10")} lo que evita usar el '
        f'ranking de riesgo de churn_traspaso como si fuera uplift ({n("p07_evitados_riesgo10")} traspasos '
        f'evitados). El X-learner también contacta muchos menos "sleeping dogs" (afiliados a quienes el '
        "contacto empuja a irse) que el ranking de riesgo."
    )
    mockup = mini_dashboard(
        "A quién conviene contactar · uplift_retencion",
        [(n("p07_razon_10"), "veces más traspasos evitados que ordenar por riesgo"),
         (n("p07_evitados_xlearner10"), "traspasos evitados al 10 %"),
         (n("p07_spearman_dr"), "correlación con el efecto causal real (DR-learner)")],
        [
            ("ok", "Persuadible", "reclamo reciente ya resuelto, saldo y uso digital medios", "contactar: el contacto retiene"),
            ("critical", "Sleeping dog", "uso digital bajo, muy sensible a la comisión", "NO contactar: el contacto empuja a irse"),
            ("warn", "Validación sin sesgo (holdout aleatorio)", "el Qini del holdout no alcanza significancia estadística a esta escala", "ver reserva"),
        ],
        chart, "Traspasos evitados contactando al 10 % y al 20 % de la cartera (verdad del simulador).",
    )
    produccion = (
        "Esto exige lo que hoy no existe: un holdout aleatorio real de contacto (a quién se llama lo decide "
        "una moneda, no el ejecutivo) durante algunos meses, del tamaño suficiente para medir Qini con "
        "significancia. Sin ese piloto, la recomendación se apoya en la verdad del simulador, que solo "
        "existe en este sandbox."
    )
    reserva = card_reserva(
        "El único criterio de aceptación duro (Qini positivo en los tres cortes de test, sobre el holdout "
        "aleatorio sin sesgo) lo cumple solo el X-learner, y con un intervalo de confianza que cruza cero: "
        "no se puede afirmar con confianza estadística que le gane al azar en ese holdout. La recomendación "
        "de negocio se apoya en la correlación contra la verdad del simulador y en la métrica de negocio "
        "oracular, ambas posibles solo en este sandbox, no en el mundo real todavía."
    )
    return proyecto_card(
        id_="p07", numero="07", nombre="uplift_retencion",
        titulo="A quién conviene contactar (causalidad)",
        problema="Rankear por riesgo de traspaso mezcla a quien se queda igual, a quien se va igual, al "
                 "persuadible (el contacto lo retiene) y al sleeping dog (el contacto lo hace irse). "
                 "¿A cuál de los cuatro tipos hay que llamar?",
        que_hace="Separa el efecto causal del contacto del riesgo de traspaso con tres métodos de "
                 "inferencia causal (T-, X- y DR-learner) sobre una campaña histórica sesgada más un "
                 "pequeño holdout de contacto aleatorio.",
        resultado_html=resultado,
        uso_html="El ejecutivo propio prioriza su cupo de contacto por el ranking de uplift, no por el de "
                 "riesgo: deja de llamar automáticamente a quien tiene baja actividad digital y alta "
                 "sensibilidad a la comisión (sleeping dogs) y prioriza a quien tuvo un reclamo ya resuelto.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/uplift_retencion/README.md §§1, 6.2, 6.3, 6.4, 7, 8",
    )


def _card09():
    fuerzas_lbl = ["0×", "1×", "2×", "4×"]
    chart = line_chart(
        fuerzas_lbl,
        {
            "Thompson lineal (bandit)": [num("p09_barrido_f0_thompson"), num("p09_reward_thompson"), num("p09_barrido_f2_thompson"), num("p09_barrido_f4_thompson")],
            "Epsilon-greedy (sin contexto)": [num("p09_barrido_f0_epsgreedy"), num("p09_reward_epsgreedy"), num("p09_barrido_f2_epsgreedy"), num("p09_barrido_f4_epsgreedy")],
        },
        {"Thompson lineal (bandit)": COL_A2, "Epsilon-greedy (sin contexto)": COL_BASE},
        fmt=_fmt3, unit_suffix="", width=580, height=250,
    )
    resultado = (
        f'Con la heterogeneidad calibrada para este sandbox, Thompson lineal gana a un simple promedio por '
        f'canal (epsilon-greedy) en {n("p09_gana_epsgreedy_thompson")} semillas, no en las '
        f'{n("p09_umbral_semillas")} que exige el criterio de aceptación. Recién le gana en {n("p09_barrido_f4_thompson_gana")} '
        "semillas si la heterogeneidad real fuera cuatro veces más fuerte de lo calibrado. La política "
        f'vigente (heurística por riesgo) genera {n("p09_ganancia_vs_sincontacto")} más que no contactar a '
        f'nadie en {n("ventana24")}, pero gastando el doble que las alternativas y contactando a más sleeping dogs.'
    )
    mockup = mini_dashboard(
        "Política de contacto · bandit_contacto",
        [(n("p09_reward_heuristica"), "retención neta, política actual (mil M CLP)"),
         (n("p09_reward_thompson"), "retención neta, Thompson lineal (mil M CLP)"),
         (n("p09_dr_error_epsgreedy"), "error del mejor estimador off-policy")],
        [
            ("warn", "Bandit contextual todavía no reemplaza la heurística", "gana a eps-greedy en pocas semillas", f'{n("p09_gana_epsgreedy_thompson")} semillas (exigido: {n("p09_umbral_semillas")})'),
            ("critical", "LinUCB se apaga", "contacta a casi nadie, y peor que el azar cuando lo hace", "no desplegar tal cual"),
            ("ok", "Auditoría off-policy antes de arriesgar presupuesto", "los tres estimadores dentro del umbral de error", f'bajo {n("p09_dr_umbral")}'),
        ],
        chart, f'Retención neta acumulada ({n("ventana24")}) según qué tan fuerte sea la heterogeneidad de respuesta al contacto.',
    )
    produccion = (
        "El guardrail principal, si algún día se despliega: nunca en explotación total, mantener siempre una "
        "fracción de exploración en los cinco brazos para poder seguir auditando con evaluación off-policy, "
        "y monitorear la tasa de contactos a quienes de todas formas terminan yéndose (proxy observable de "
        "sleeping dog, porque el tipo real nunca se observa)."
    )
    reserva = card_reserva(
        "Los resultados principales que el brief esperaba (Thompson o LinUCB le ganan a lo simple en "
        "cuatro de cinco semillas; el estimador doblemente robusto es el de menor error) no se cumplen "
        "tras la corrección del generador. En palabras del propio equipo: 'el bandit contextual, con esta "
        f'calibración de señal y {n("ventana24")}, no justifica reemplazar la heurística todavía\'. Se reporta así, '
        "sin forzarlo, porque el objetivo de esta ronda era saber, no ganar."
    )
    return proyecto_card(
        id_="p09", numero="09", nombre="bandit_contacto",
        titulo="Bandit contextual de contacto",
        problema="Cada mes hay que elegir, para cada afiliado en riesgo, entre no contactar, email, "
                 "llamada, visita o una oferta de ahorro. ¿Se puede aprender esa decisión sobre la marcha, "
                 "y auditarla antes de arriesgar presupuesto real?",
        que_hace="Compara una heurística de negocio con varios algoritmos de bandit contextual "
                 "(epsilon-greedy, LinUCB, Thompson) que aprenden con la recompensa real (comisión "
                 "retenida menos costo del contacto), y audita todo con evaluación fuera de política "
                 "(IPS, SNIPS, doblemente robusto).",
        resultado_html=resultado,
        uso_html="Hoy, ninguno: la recomendación es no reemplazar la heurística vigente. El candidato para "
                 "un piloto acotado, corriendo en paralelo sin desplazar la política actual, es la variante "
                 "que aprende con LightGBM, que es la única que de verdad distingue a quién conviene o no "
                 "contactar.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/bandit_contacto/README.md §§1, 6.1, 6.3, 6.5, 7, 8",
    )


def _card_planilla():
    resultado = (
        f'La planilla sintética reproduce {n("planilla_codigos")} de movimiento de personal de la Tabla '
        "N° 7 (altas, bajas, licencias, cambios de contrato), con el mismo desfase de recepción que usan "
        "los modelos de renta y de cese. Cuando se probó como feature adicional en riesgo_cese_cotizacion, "
        "el aporte fue marginal e inconsistente en signo, y en rim_nowcasting fue chico y concentrado en "
        "la cartera Cuprum: en los dos casos buena parte de la información ya estaba, por otra vía, en los "
        "eventos del Seguro de Cesantía. El esquema de columnas quedó además documentado, columna por "
        "columna, contra la norma (Compendio de la SP y Previred) en docs/modelo_datos_regulatorio.md."
    )
    mockup_html = ""
    produccion = (
        "Es la pieza más cercana a producción: en cuanto existan credenciales de acceso a la planilla real "
        "(Previred) y al Boletín de Infractores, los proyectos 01, 04 y 05 pueden apuntar a esa fuente sin "
        "cambiar su lógica de features, porque ya están escritos contra esta misma estructura de columnas."
    )
    reserva = card_reserva(
        "El sorteo de recepción y estado de pago se hace línea por línea, no por planilla completa: dos "
        "líneas de la misma declaración pueden quedar con fecha de recepción distinta, lo que contradice "
        "el propio docstring del módulo. Es un defecto de código menor, ya identificado y documentado, "
        "pendiente de corregir; no invalida ningún resultado de los proyectos que la usan."
    )
    return proyecto_card(
        id_="planilla", numero="—", nombre="common/planilla",
        titulo="Planilla regulatoria (modelo de datos)",
        problema="Para que un modelo sirva en producción tiene que entrenar con la misma estructura de "
                 "datos que va a ver ahí: los códigos oficiales de la planilla Previred, no una "
                 "simplificación propia de cada proyecto.",
        que_hace="Genera una planilla sintética línea a línea (afiliado × empleador × mes × tipo de pago) "
                 "con los códigos de movimiento de personal, cotización, SIS y el aporte adicional de la "
                 "reforma, coherente con el panel laboral común.",
        resultado_html=resultado,
        uso_html="Es infraestructura compartida, no un modelo de decisión propio: la usan riesgo_empleador "
                 "(capa de recepción y pago del empleador) y se probó como feature en riesgo_cese_cotizacion.",
        mockup_html=mockup_html, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/common/BRIEF_PLANILLA.md; afp_lab/riesgo_empleador/README.md §2; "
                     "afp_lab/riesgo_cese_cotizacion/README.md §11.2",
    )


def _card_en_curso(*, id_, numero, nombre, titulo, pregunta, diseno):
    return f"""
<article class="proyecto encurso" id="{id_}">
  <div class="proyecto-head">
    <span class="badge">{esc(numero)}</span>
    <h3>{esc(titulo)}</h3>
    <span class="badge">{esc(nombre)}</span>
    <span class="chip chip-info">En curso</span>
  </div>
  <dl class="campos medida">
    <dt>Pregunta de negocio</dt><dd>{pregunta}</dd>
    <dt>Diseño del experimento</dt><dd>{diseno}</dd>
  </dl>
  <p class="medida" style="color:var(--muted);font-size:.9rem;">Todavía sin resultados: esta tarjeta se
  completa cuando el gerente envíe la corrida de referencia. No lleva números porque no hay ninguno
  todavía que reportar con honestidad.</p>
</article>
"""


def _card08():
    chart = grouped_bar_chart(
        ["Traspaso (02)", "Cese (04)"],
        {
            "LightGBM de producción": [num("p08_auc_lgbm_traspaso"), num("p08_auc_lgbm_cese")],
            "Red secuencial (multi-tarea)": [num("p08_auc_red_multitarea_traspaso"), num("p08_auc_red_multitarea_cese")],
        },
        {"LightGBM de producción": COL_MODELO, "Red secuencial (multi-tarea)": COL_A2},
        fmt=_fmt3, width=560, height=240, value_max=1.0,
        aria="AUC del LightGBM de producción vs. la red secuencial, traspaso y cese",
    )
    resultado = (
        f'La red (GRU de dos capas, multi-tarea) no mejora al LightGBM de producción en ninguna de las '
        f'dos tareas: AUC {n("p08_auc_red_multitarea_traspaso")} vs. {n("p08_auc_lgbm_traspaso")} en '
        f'traspaso ({n("p08_delta_auc_traspaso")} de caída) y {n("p08_auc_red_multitarea_cese")} vs. '
        f'{n("p08_auc_lgbm_cese")} en cese ({n("p08_delta_auc_cese")} de caída). Pero, con la MISMA '
        f'información que ve la red, un LightGBM sobre esa misma secuencia aplanada obtiene '
        f'{n("p08_auc_lags_traspaso")} y {n("p08_auc_lags_cese")}: prácticamente empatado con la red '
        f'(diferencias de {n("p08_delta_lags_min")} a {n("p08_delta_lags_max")} AUC). La brecha grande no es de arquitectura: es que la secuencia '
        "(el panel laboral común, sin las tablas comerciales de la AFP) lleva mucha menos información que "
        "las features completas de traspaso y cese."
    )
    mockup = mini_dashboard(
        "Comparación de arquitecturas · secuencial_afiliado",
        [(n("p08_auc_lgbm_traspaso"), "AUC LightGBM producción (traspaso)"),
         (n("p08_auc_red_multitarea_traspaso"), "AUC red secuencial (traspaso)"),
         (n("p08_delta_auc_traspaso"), "brecha de AUC")],
        [
            ("critical", "Red no aporta en ninguna cabeza", f'criterio del brief: mejora ≥ {n("p08_umbral_auc")} AUC en al menos dos cabezas', "no se cumple"),
            ("ok", "Empata con LightGBM de la misma información", "aísla arquitectura de información", f'diferencia < {n("p08_umbral_auc")} AUC'),
            ("warn", "Multi-tarea no le gana al single-task", "compartir encoder no ayudó a esta escala", "diferencia dentro del ruido"),
        ],
        chart, "AUC del LightGBM de producción vs. la red secuencial multi-tarea, mismos cortes y splits.",
    )
    produccion = (
        "No se recomienda avanzar con esta arquitectura mientras la secuencia no incorpore las mismas "
        "fuentes que ya usan los modelos de producción (contactos, reclamos, saldos, cambios de fondo, "
        "uso digital, brecha de rentabilidad). El valor futuro de una representación secuencial dependería "
        "de sumarle esas fuentes y de probarla en afiliados con historia más larga que la de este sandbox."
    )
    reserva = card_reserva(
        f'El entrenamiento se acotó a {n("p08_cortes_train")} de entrenamiento, sin calibrar las '
        "probabilidades de la red y sin una búsqueda amplia de hiperparámetros (arquitectura, tamaño del "
        "embedding, número de capas), por presupuesto de esta ronda. La conclusión &mdash;'no aporta frente "
        "al modelo de producción'&mdash; es honesta y aísla bien la causa (información, no arquitectura), "
        "pero no cierra la pregunta de si una red secuencial con más datos y más búsqueda de "
        f'hiperparámetros cambiaría el resultado. La revisión adversarial (aprobada con reserva) marca dos '
        f'puntos de alcance: el proyecto cubre dos de las cuatro cabezas del brief (traspaso y cese; '
        f'faltan APV y RIM), y el runtime nunca se corrió a la escala de {n("meta_universo_afiliados")} '
        f'afiliados que exige la convención del laboratorio, solo a {n("p08_escala")}. El revisor '
        f'reprodujo las cifras casi dígito a dígito; la única que no calzó exacto fue el LightGBM de '
        f'producción en cese ({n("p08_reprod_lgbm_cese")} reproducido vs. {n("p08_auc_lgbm_cese")} del '
        "README), una diferencia mínima que no cambia ninguna conclusión."
    )
    return proyecto_card(
        id_="p08", numero="08", nombre="secuencial_afiliado",
        titulo="Representación secuencial del afiliado",
        problema="¿Una red neuronal que lee la historia mensual completa del afiliado, en vez de un "
                 "conjunto de features tabulares, predice mejor el traspaso y el cese que los modelos de "
                 "producción de 02 y 04?",
        que_hace=f'Entrena una red GRU multi-tarea sobre {n("ventana24")} de panel laboral del afiliado y la '
                 "compara, en los mismos cortes y splits, contra el LightGBM de producción de cada "
                 "proyecto y contra un LightGBM que ve exactamente la misma secuencia aplanada (para "
                 "separar el efecto de la arquitectura del efecto de la información).",
        resultado_html=resultado,
        uso_html="Ninguno todavía: la recomendación explícita es no reemplazar los pipelines tabulares de "
                 "02 y 04 con esta arquitectura, a este presupuesto de datos y cómputo.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/secuencial_afiliado/README.md §§1, 6, 7, 8",
        estado_chip="Aprobado con reserva (revisión ronda 4, docs/revisiones/ronda_4/secuencial.md)",
    )


def _card01_correccion():
    chart = grouped_bar_chart(
        ["Sistema h=1", "Sistema h=2", "Cuprum h=1", "Cuprum h=2"],
        {
            "Antes (sin código de movimiento)": [
                num("p01c_wape_h1_sistema_antes"), num("p01c_wape_h2_sistema_antes"),
                num("p01c_wape_h1_cuprum_antes"), num("p01c_wape_h2_cuprum_antes"),
            ],
            "Después (con código Tabla N° 7)": [
                num("p01c_wape_h1_sistema_despues"), num("p01c_wape_h2_sistema_despues"),
                num("p01c_wape_h1_cuprum_despues"), num("p01c_wape_h2_cuprum_despues"),
            ],
        },
        {"Antes (sin código de movimiento)": COL_BASE, "Después (con código Tabla N° 7)": COL_A2},
        fmt=_fmt3, width=620, height=250,
        aria="WAPE antes y después de agregar el código de movimiento de personal, por perfil y horizonte",
    )
    resultado = (
        f'A escala de referencia ({n("p01c_escala")}, {n("p01c_snapshots")}), agregar el código de '
        f'movimiento de personal (Tabla N° 7) al nowcasting de RIM no cambia el WAPE de Sistema '
        f'({n("p01c_wape_h1_sistema_antes")} → {n("p01c_wape_h1_sistema_despues")} a un mes; '
        f'{n("p01c_wape_h2_sistema_antes")} → {n("p01c_wape_h2_sistema_despues")} a dos meses), y sí '
        f'mejora a Cuprum a dos meses ({n("p01c_wape_h2_cuprum_antes")} → {n("p01c_wape_h2_cuprum_despues")}, '
        f'sesgo agregado de {n("p01c_sesgo_h2_cuprum_antes")} a {n("p01c_sesgo_h2_cuprum_despues")}); a un '
        f'mes Cuprum queda igual ({n("p01c_wape_h1_cuprum_antes")} → {n("p01c_wape_h1_cuprum_despues")}). El '
        f'AUC de cotiza/no cotiza no se mueve ({n("p01c_auc_h1_sistema")} a un mes; '
        f'{n("p01c_auc_h2_sistema_antes")} → {n("p01c_auc_h2_sistema_despues")} a dos meses). El runtime de '
        f'la corrida de referencia sigue en torno a {n("p01c_runtime")}, sin aumento medible.'
    )
    mockup = mini_dashboard(
        "Nowcasting con código de movimiento · rim_nowcasting",
        [(n("p01c_wape_h2_cuprum_despues"), "WAPE Cuprum h=2 (con el código nuevo)"),
         (n("p01c_wape_h2_sistema_despues"), "WAPE Sistema h=2 (sin cambio material)"),
         (n("p01c_seg_termino_sesgo"), "sesgo en término reciente (debilidad previa)")],
        [
            ("ok", "Cuprum, segundo mes", "el código de movimiento sí aporta aquí", f'{n("p01c_wape_h2_cuprum_antes")} → {n("p01c_wape_h2_cuprum_despues")}'),
            ("warn", "Con subsidio (licencia en curso)", "mejora algo en ambos horizontes", f'{n("p01c_seg_subsidio_h1_antes")} → {n("p01c_seg_subsidio_h1_despues")}'),
            ("critical", "Término reciente (aviso AFC)", "sesgo grande en ambas versiones, no lo corrige el código nuevo", f'≈{n("p01c_seg_termino_sesgo")}'),
        ],
        chart, "WAPE antes y después de agregar el código de movimiento de personal, por perfil y horizonte.",
    )
    produccion = (
        "El diseño as-of ya está probado (siete tests nuevos, sin fuga hacia el futuro) y usa las mismas "
        "constantes de la planilla común, así que sumar esta familia de features en producción es barato "
        "una vez conectada la planilla Previred real. La recomendación es dejarla como familia de "
        "features opcional, no obligatoria: el aporte es chico y concentrado en un segmento."
    )
    reserva = card_reserva(
        f'La señal es en gran parte redundante con el aviso del Seguro de Cesantía: el código de término '
        f'coincide con el término AFC en el {n("p01c_coincidencia_termino_afc")} de los casos, lo que explica '
        f'el efecto casi nulo en Sistema. Con una sola semilla, un delta de ±{n("p01c_delta_umbral")} en un '
        f'segmento chico (independientes, '
        f'{n("p01c_seg_independiente_rango")} de WAPE) no se puede declarar significativo. El segmento de '
        f'término reciente sigue con un sesgo cercano a {n("p01c_seg_termino_sesgo")} en ambas versiones: '
        "una debilidad previa del modelo que este código no corrige."
    )
    return proyecto_card(
        id_="p01-correccion", numero="—", nombre="rim_nowcasting (ronda 3)",
        titulo="Código de movimiento de personal en el nowcasting",
        problema="¿Ayuda a proyectar mejor la renta imponible agregar a 01 el código de movimiento de "
                 "personal de la propia planilla (altas, bajas, licencias) además del aviso del Seguro de "
                 "Cesantía que ya usaba?",
        que_hace="Agrega 14 features nuevas derivadas del código de movimiento (Tabla N° 7) a las "
                 "features existentes de 01, y compara el error de proyección antes y después, en Sistema "
                 "y en Cuprum.",
        resultado_html=resultado,
        uso_html="Queda disponible como familia de features opcional: no se activa por defecto porque su "
                 "aporte es chico salvo en la cartera Cuprum a dos meses, donde sí vale la pena.",
        mockup_html=mockup, produccion_html=produccion, reserva_html=reserva,
        fuente_texto="afp_lab/rim_nowcasting/README.md §10c",
    )


def sec_casos():
    tarjetas = (
        _card02() + _card03() + _card04() + _card05() + _card01() + _card06() + _card07() + _card08() + _card09()
        + _card_planilla() + _card01_correccion()
    )
    return f"""
<section id="casos">
  <h2>Un caso de uso por proyecto</h2>
  <p class="medida">Nueve capacidades completas, la planilla regulatoria que las sostiene a todas, y la
  actualización de la ronda 3 sobre la primera. Cada tarjeta dice el problema, qué hace el modelo, el resultado en el sandbox
  traducido a negocio, cómo lo usaría el área, un mockup de dashboard, el camino a producción y la reserva
  principal &mdash; visible aquí, no escondida en el anexo.</p>
  {tarjetas}
</section>
"""


# --------------------------------------------------------------------------
# Torre de control (mockup que integra todos los casos)
# --------------------------------------------------------------------------

def sec_torre_control():
    kpis = [
        (n("p02_captura10_traspasos"), "traspasos capturados (churn, 10 %)"),
        (n("p03_captura10_monto"), "ahorro APV capturado (10 %)"),
        (n("p04_captura10_cese3m"), "ceses capturados (10 %)"),
        (n("p05_captura10_masa"), "masa en riesgo capturada (10 %)"),
        (n("p02_licitacion_saldo_riesgo"), "en riesgo, licitación 2027"),
        (n("p09_reward_heuristica"), f'retención neta, política actual ({n("ventana24")})'),
    ]
    chart = grouped_bar_chart(
        ["Traspaso (02)", "APV (03)", "Cese (04)", "Empleador (05)"],
        {
            "Modelo": [num("p02_auc_modelo"), num("p03_auc"), num("p04_auc_cese3m"), num("p05_auc")],
            "Práctica actual": [num("p02_auc_baseline"), num("p03_auc_regla"), num("p04_auc_regla_cese3m"), num("p05_auc_regla")],
        },
        {"Modelo": COL_MODELO, "Práctica actual": COL_BASE},
        fmt=_fmt3, width=680, height=280, value_max=1.0,
        aria="AUC del modelo vs. la práctica actual, cuatro capacidades de priorización",
    )
    filas5 = [
        ("Tesorería", "critical", "Licitación de stock 2027", "sorteo de afiliados no pensionados, Ley 21.735", n("p02_licitacion_saldo_pct") + " del saldo Cuprum"),
        ("Riesgo", "warn", "Reingreso tras cese", "AUC bajo el umbral del brief tras recalibrar la densidad", n("p04_auc_reingreso_actual")),
        ("Riesgo", "warn", "Masa en riesgo de empleadores", "captura al 10 % en el límite del criterio de negocio", n("p05_captura10_masa")),
        ("Comercial", "ok", "Traspaso voluntario", "el modelo gana a la regla en todos los segmentos grandes", n("p02_razon_vs_regla") + " la regla"),
        ("Comercial", "ok", "Ahorro voluntario (APV)", "líderes de mercado, captura muy por sobre ordenar por renta", n("p03_captura10_monto")),
    ]
    filas = [(sev, nom, det, val) for (_area, sev, nom, det, val) in filas5]
    caption = ("AUC del modelo vs. la práctica actual hoy, en las cuatro capacidades de priorización de cartera.")
    APP_RAW["torre_control"] = {
        "kpis": kpis,
        "filas": [{"area": a, "severidad": sv, "alerta": nm, "detalle": dt, "valor": vl} for (a, sv, nm, dt, vl) in filas5],
        "chart_svg": chart,
        "chart_caption": caption,
    }
    mockup = mini_dashboard("Torre de control · portafolio Cuprum", kpis, filas, chart, caption)
    return f"""
<section id="torre-control">
  <h2>Torre de control: los ocho casos en un solo tablero</h2>
  <p class="medida">Así se vería un tablero mensual que junta las alertas de las principales capacidades
  de priorización. No es una pantalla que exista hoy: es el mockup de a qué apunta el equipo si el negocio
  decide poner uno o más de estos modelos en producción.</p>
  {mockup}
</section>
"""


# --------------------------------------------------------------------------
# Roadmap
# --------------------------------------------------------------------------

def sec_roadmap():
    pasos = [
        ("Piloto de retención con holdout aleatorio real",
         "Reservar un grupo de contacto aleatorio (churn_traspaso y uplift_retencion) durante algunos "
         "meses, del tamaño suficiente para medir con significancia si separar causalidad mejora la "
         "retención. Es el mismo diseño que ya se simuló, pero con afiliados reales.",
         "Datos y patrocinio: capacidad de contacto del equipo comercial, y que el negocio acepte que un "
         "grupo de control no se contacte por un tiempo definido."),
        ("Datos reales de planilla, AFC y Boletín de Infractores",
         "Conectar 01, 04 y 05 a la planilla Previred real (con su calendario de recepción), a los "
         "eventos del Seguro de Cesantía y al Boletín de Infractores de la Dirección del Trabajo.",
         "Patrocinio de TI/Operaciones para el acceso, y un ciclo de validación con los mismos criterios "
         "que ya se aplicaron en el sandbox."),
        ("Corregir el generador para que el reingreso vuelva a ser predecible",
         "Antes de reintentar 04 con datos reales, separar en el generador sintético la cesantía reciente "
         "de la crónica, como ya lo propone el propio informe de la ronda 3.",
         "Solo trabajo interno del equipo; ningún dato ni patrocinio externo."),
        ("Escalar riesgo_empleador a la cartera completa de empleadores",
         "Repetir la corrida con el universo completo de empleadores de Cuprum (no solo una muestra), "
         "porque la volatilidad del agregado mensual baja de forma predecible al escalar.",
         "Patrocinio para correr a esa escala y, más adelante, datos reales de cobranza."),
        ("Presupuesto separado para la licitación de 2027",
         "Preparar, con un año de anticipación, dos campañas de retención separadas: una para el sorteo "
         "de la licitación de stock y otra para el churn voluntario de siempre, porque compiten por el "
         "mismo cupo de contacto y el sorteo lo consume casi todo.",
         "Decisión y presupuesto del negocio, con tiempo suficiente antes de agosto de 2027."),
        ("Piloto acotado del bandit de contacto, sin desplazar la heurística",
         f'Correr en paralelo, con más presupuesto de contactos y más tiempo que los {n("ventana24")} simulados, '
         "la variante que aprendió a discriminar de verdad (Thompson con LightGBM), y volver a auditar "
         "con evaluación off-policy antes de subir su participación.",
         "Patrocinio para un piloto controlado; ningún cambio a la política vigente todavía."),
    ]
    APP_RAW["roadmap"] = [{"titulo": t, "desc": d, "pide": p} for t, d, p in pasos]
    items = "".join(
        f'<li><h4>{esc(t)}</h4><p>{d}</p><p style="color:var(--muted);font-size:.86rem;"><strong>Se pide al negocio:</strong> {p}</p></li>'
        for t, d, p in pasos
    )
    return f"""
<section id="roadmap">
  <h2>Roadmap ({n("roadmap_horizonte")})</h2>
  <p class="medida">En este orden porque cada paso reduce el riesgo del siguiente: primero medir con un
  piloto real, después conectar datos reales, después escalar. Es la única sección del informe con pasos
  numerados, porque aquí sí es una secuencia.</p>
  <ol class="roadmap medida">{items}</ol>
</section>
"""


# --------------------------------------------------------------------------
# Supuestos, reservas y límites
# --------------------------------------------------------------------------

def sec_supuestos():
    intro = (
        f'Todo lo que se afirma en este informe con un número viene del sandbox sintético, salvo '
        f'cuando se cita explícitamente la Ficha {n("f165_numero")} de la Superintendencia de Pensiones o la Ley '
        '21.735. Ningún resultado se presenta "en el límite" como si cumpliera: cuando una cifra queda al borde de '
        'un umbral (masa en riesgo de riesgo_empleador) o por debajo de él (reingreso de riesgo_cese_cotizacion, '
        'el criterio de Thompson en bandit_contacto), se dice tal cual, con las mismas palabras que usó la '
        'revisión.'
    )
    datos_items = [
        f'Las seis AFP del sandbox, sus comisiones y rentabilidades son ficticias salvo Cuprum, cuya '
        f'comisión ({n("f165_cuprum_comision")}) y rentabilidad a {n("f165_rent36_meses")} (fondo C: '
        f'{n("f165_cuprum_rent36_c")}) sí son la cifra real de la Ficha {n("f165_numero")}.',
        f'La densidad de cotización del universo sintético se recalibró en la ronda 3 contra la Ficha '
        f'{n("f165_numero")}: el perfil sistema pasó de {n("dens_sistema_antes")} a {n("dens_sistema_despues")} '
        f'(objetivo: {n("f165_densidad_sistema")}) y el perfil Cuprum de {n("dens_cuprum_antes")} a '
        f'{n("dens_cuprum_despues")} (objetivo: {n("f165_densidad_cuprum_memoria")}).',
        'Los parámetros de comportamiento (probabilidades de reclamo, de visita de la competencia, de '
        'aporte voluntario, de declaración y no pago) son supuestos razonables para Chile, documentados uno por '
        'uno en cada README, no series oficiales.',
        f'La fuga bruta mensual de Cuprum (a diferencia del neto, que sí publica la Ficha '
        f'{n("f165_numero")}: {n("f165_cuprum_traspasos_netos")} traspasos netos en junio de 2026) no está '
        'disponible públicamente; se usa un supuesto de trabajo anclado al promedio del sistema.',
    ]
    modelos_items = [
        'Ninguna cifra de este informe se redondeó hacia arriba. Donde una cifra tiene rango entre semillas, '
        'se muestra el rango o la semilla de referencia, no el mejor caso.',
        'Los criterios de aceptación se fijaron antes de ver los resultados y no se relajaron para que un '
        'proyecto pasara: donde no se cumplen (04 reingreso, 05 masa en riesgo, 09 bandit), se dice.',
        f'Todos los proyectos comparten universo, pero no todos comparten escala: uplift_retencion y '
        f'bandit_contacto corrieron a {n("p07_escala")} (no {n("p02_escala")}) por restricción de '
        'tiempo de esta ronda, validado además a mayor escala en uplift_retencion.',
    ]
    APP_RAW["supuestos"] = {"intro": intro, "datos": datos_items, "modelos": modelos_items}
    datos_html = "".join(f"<li>{x}</li>" for x in datos_items)
    modelos_html = "".join(f"<li>{x}</li>" for x in modelos_items)
    return f"""
<section id="supuestos">
  <h2>Supuestos, reservas y límites</h2>
  <p class="medida">{intro}</p>
  <h3>Sobre los datos</h3>
  <ul class="medida">{datos_html}</ul>
  <h3>Sobre los modelos</h3>
  <ul class="medida">{modelos_html}</ul>
</section>
"""


# --------------------------------------------------------------------------
# Anexo técnico
# --------------------------------------------------------------------------

def sec_anexo():
    filas = [
        ("01 rim_nowcasting", n("p01_escala") + ", PySpark",
         "Hurdle: GBT clasificador + regresor", f'WAPE {n("p01_wape_h1")} (h=1)',
         n("p01c_tests_despues"), n("p01c_runtime"),
         "Ronda 3 — docs/revisiones/ronda_3/datos.md"),
        ("02 churn_traspaso", n("p02_escala"),
         "LightGBM binario + calibración", f'AUC {n("p02_auc_modelo")}', n("p02_tests"), n("p02_runtime"),
         "Rondas 1, 2 y 3 — docs/revisiones/ronda_1/churn_traspaso.md; ronda_2/churn_traspaso.md; ronda_3/churn.md"),
        ("03 propension_apv", n("p03_escala"),
         "Hurdle: LightGBM clasificador + regresor", f'AUC {n("p03_auc")}', n("p03_tests"), n("p03_runtime"),
         "Rondas 1 y 2 — docs/revisiones/ronda_1/propension_apv.md; ronda_2/propension_apv.md"),
        ("04 riesgo_cese_cotizacion", n("p04_escala"),
         "3 × LightGBM binario + calibración", f'AUC cese {n("p04_auc_cese3m")}', n("p04_tests"), n("p04_runtime"),
         "Rondas 1 y 2 — docs/revisiones/ronda_1/riesgo_cese_cotizacion.md; ronda_2/riesgo_cese_cotizacion.md. "
         "Corrección posterior a la ronda 3 (ronda 4): aprobada — docs/revisiones/ronda_4/cese_correccion.md"),
        ("05 riesgo_empleador", n("p05_escala") + f' (~{n("p05_empleadores")} empleadores)',
         "LightGBM binario por objetivo", f'AUC {n("p05_auc")}', n("p05_tests"), n("p05_runtime"),
         "Rondas 2 y 3 — docs/revisiones/ronda_2/riesgo_empleador.md; ronda_3/empleador.md"),
        ("06 segmentacion_afiliados", n("p06_escala"),
         "k-means + clasificador LightGBM", f'Silueta {n("p06_silueta")}', n("p06_tests"), n("p06_runtime"),
         "Ronda 2 — docs/revisiones/ronda_2/planilla_cese_segmentacion.md"),
        ("07 uplift_retencion", n("p07_escala"),
         "T-, X- y DR-learner sobre LightGBM", f'Qini X-learner {n("p07_qini_xlearner")}', n("p07_tests"), n("p07_runtime"),
         "Ronda 4 — docs/revisiones/ronda_4/uplift.md"),
        ("08 secuencial_afiliado", n("p08_escala"),
         "GRU multi-tarea (PyTorch) vs. LightGBM", f'AUC {n("p08_auc_red_multitarea_traspaso")} (red, traspaso)', n("p08_tests"), n("p08_runtime"),
         "Aprobado con reserva (ronda 4) — docs/revisiones/ronda_4/secuencial.md"),
        ("09 bandit_contacto", n("p09_escala"),
         "ε-greedy, LinUCB, Thompson (lineal y LightGBM)", f'DR err. {n("p09_dr_error_epsgreedy")} (ε-greedy)', n("p09_tests"), n("p09_runtime"),
         "Ronda 4 — docs/revisiones/ronda_4/bandit.md"),
    ]
    APP_RAW["anexo"] = [
        {"proyecto": p, "datos": d, "modelo": m, "metrica": me, "tests": t, "runtime": r, "revision": v}
        for p, d, m, me, t, r, v in filas
    ]
    filas_html = "".join(
        f"<tr><td>{esc(p)}</td><td>{esc(d)}</td><td>{esc(m)}</td><td class='num'>{esc(me)}</td>"
        f"<td class='num'>{esc(t)}</td><td class='num'>{esc(r)}</td><td>{esc(v)}</td></tr>"
        for p, d, m, me, t, r, v in filas
    )
    return f"""
<section id="anexo">
  <h2>Anexo técnico</h2>
  <p class="medida">Una fila por proyecto: escala de datos, tipo de modelo, métrica principal, tests,
  tiempo de ejecución de la corrida de referencia y estado de revisión. El detalle completo &mdash;
  parámetros, features, splits, baselines&mdash; vive en el README de cada proyecto.</p>
  <div class="tabla-wrap">
    <table class="tabla">
      <thead><tr><th>Proyecto</th><th>Datos</th><th>Modelo</th><th>Métrica principal</th>
      <th>Tests</th><th>Runtime</th><th>Revisión</th></tr></thead>
      <tbody>{filas_html}</tbody>
    </table>
  </div>
  <p class="fuente-cita">Suite completa del laboratorio: {n("meta_tests_total")} tests,
  {n("meta_tests_fallos")} fallos, {n("meta_tests_runtime")} (<code>python -m pytest afp_lab -q</code>,
  incluido el proyecto 01 con Spark). Fuente: README.md raíz, y el README y la carpeta
  <code>docs/revisiones/</code> de cada proyecto.</p>
</section>
"""


# --------------------------------------------------------------------------
# Glosario
# --------------------------------------------------------------------------

def sec_glosario():
    terminos = [
        ("Afiliado / cotizante", "Afiliado es cualquier persona con cuenta en la AFP; cotizante es quien "
         "pagó una cotización el mes de referencia. La razón entre ambos es la densidad de cotización."),
        ("AUC", f'Qué tan bien ordena el modelo a quién le va a pasar el evento primero. {n("glosario_auc_min")} '
         f'es no mejor que el azar; {n("glosario_auc_max")} es perfecto. No dice nada sobre el número '
         "absoluto de casos."),
        ("Baseline / regla de negocio", "La forma en que se decide hoy, sin el modelo (por ejemplo, "
         "ordenar por renta o por una regla simple). Es la vara con la que se mide si el modelo aporta."),
        ("Captura al 10 %", "Del total de casos que van a ocurrir, qué fracción cae dentro del 10 % de la "
         "cartera que el modelo marca como más riesgosa u oportuna. Es la métrica de negocio, no la "
         "técnica."),
        ("Cortes / periodo_corte", "El mes en que se toma la decisión ('hoy' para el modelo). El resultado "
         "se mide en los meses siguientes, nunca antes."),
        ("Densidad de cotización", "Fracción de afiliados que cotizó en un mes típico. Cuprum tiene la "
         "densidad más alta del sistema."),
        ("Fuga de información (leakage)", "Cuando el modelo, por error, ve información del futuro al "
         "entrenar. Cada proyecto tiene tests específicos para detectarla."),
        ("Gap (purga)", "Meses que se descartan entre el entrenamiento y la prueba de un modelo, para que "
         "ningún dato de entrenamiento roce el periodo que se está evaluando."),
        ("Latente", "Un dato que existe solo dentro del simulador para generar la dinámica (por ejemplo, "
         "la propensión verdadera a irse). Nunca puede ser una feature del modelo, porque en la vida real "
         "no se observa."),
        ("Off-policy (evaluación fuera de política)", "Estimar qué tan bien habría funcionado una "
         "política nueva usando solo los datos que ya se registraron con la política vieja, sin haberla "
         "corrido de verdad."),
        ("Qini / AUUC", "Mide si un ranking de a quién contactar identifica bien a quienes el contacto de "
         "verdad cambia de decisión, usando un grupo de control aleatorio como referencia."),
        ("Sandbox sintético", "Todos los datos de este informe, salvo donde se cita la Ficha 165, son "
         "generados por un simulador, no de afiliados reales."),
        ("Sleeping dog", "Alguien a quien contactar empeora las cosas: sin el contacto se habría quedado, "
         "y el contacto lo hace irse."),
        ("Uplift / efecto causal", "El cambio que provoca de verdad una acción (como un contacto), "
         "distinto del riesgo que ya tenía la persona antes de la acción."),
        ("WAPE", "Error relativo de una proyección de montos, robusto a los ceros (a diferencia del MAPE). "
         "0 es perfecto; 1 significa un error del mismo tamaño que el propio monto."),
    ]
    APP_RAW["glosario"] = [{"termino": t, "definicion": d} for t, d in terminos]
    dts = "".join(f"<dt>{esc(t)}</dt><dd>{esc(d)}</dd>" for t, d in terminos)
    return f"""
<section id="glosario">
  <h2>Glosario</h2>
  <dl class="glosario medida">{dts}</dl>
</section>
"""


# --------------------------------------------------------------------------
# Índice / navegación
# --------------------------------------------------------------------------

INDICE = [
    ("portada", "Resumen ejecutivo"),
    ("capacidades", "Qué ofrece el equipo"),
    ("casos", "Casos de uso"),
    ("torre-control", "Torre de control"),
    ("roadmap", "Roadmap"),
    ("supuestos", "Supuestos y límites"),
    ("anexo", "Anexo técnico"),
    ("glosario", "Glosario"),
]


# --------------------------------------------------------------------------
# Modelo de contenido para la app (mismo generador, mismos datos, otro producto)
# --------------------------------------------------------------------------

CASE_META = {
    "p01": dict(area="Tesorería", estado_nivel="ok", estado_texto="Cerrado (rondas 1 y 2)"),
    "p01-correccion": dict(area="Tesorería", estado_nivel="reserva", estado_texto="Aprobado con reserva (ronda 3)"),
    "p02": dict(area="Comercial", estado_nivel="reserva", estado_texto="Aprobado con reserva (ronda 3)"),
    "p03": dict(area="Comercial", estado_nivel="ok", estado_texto="Aprobado (ronda 2)"),
    "p04": dict(area="Riesgo", estado_nivel="reserva", estado_texto="Corrección aprobada (ronda 4)"),
    "p05": dict(area="Riesgo", estado_nivel="reserva", estado_texto="Aprobado con reserva (ronda 3)"),
    "p06": dict(area="Comercial", estado_nivel="ok", estado_texto="Aprobado (ronda 2)"),
    "p07": dict(area="Comercial", estado_nivel="reserva", estado_texto="Aprobado con reserva (ronda 4)"),
    "p08": dict(area="Operaciones", estado_nivel="reserva", estado_texto="Aprobado con reserva (ronda 4)"),
    "p09": dict(area="Comercial", estado_nivel="ok", estado_texto="Aprobado (ronda 4)"),
    "planilla": dict(area="Normativa", estado_nivel="ok", estado_texto="Aprobado (ronda 2)"),
}


def _case_metrica_principal(id_):
    return {
        "p01": f'WAPE {n("p01_wape_h1")} (h=1)',
        "p01-correccion": f'WAPE Cuprum h=2 {n("p01c_wape_h2_cuprum_despues")}',
        "p02": f'AUC {n("p02_auc_modelo")}',
        "p03": f'AUC {n("p03_auc")}',
        "p04": f'AUC cese {n("p04_auc_cese3m")}',
        "p05": f'AUC {n("p05_auc")}',
        "p06": f'Silueta {n("p06_silueta")}',
        "p07": f'Qini X-learner {n("p07_qini_xlearner")}',
        "p08": f'AUC red {n("p08_auc_red_multitarea_traspaso")}',
        "p09": f'DR err. {n("p09_dr_error_epsgreedy")}',
        "planilla": f'{n("planilla_codigos")} de la Tabla N° 7',
    }[id_]


def _case_reserva_corta(id_):
    return {
        "p01": "Ronda 2, previa a la recalibración de densidad de la ronda 3.",
        "p01-correccion": f'Redundante en {n("p01c_coincidencia_termino_afc")} con el aviso AFC; ayuda solo en Cuprum a dos meses.',
        "p02": "El saldo en riesgo esperado no se entrega esta ronda (sesgo de nivel entre semillas).",
        "p03": f'No ordena bien al segmento VOLUNTARIO ({n("p03_voluntario_pct")} de la cartera elegible).',
        "p04": f'Reingreso en {n("p04_auc_reingreso_actual")}, bajo el umbral de {n("p04_auc_reingreso_umbral")}; no se usa en producción.',
        "p05": f'Captura al 10 % en {n("p05_captura10_masa")}, en el límite del criterio de {n("p05_umbral_captura")}.',
        "p06": f'ARI temporal ({n("p06_ari_temporal")}) más bajo que el de remuestreo: hay deriva real del negocio.',
        "p07": "El Qini del holdout aleatorio no alcanza significancia estadística a esta escala.",
        "p08": f'Alcance de dos de cuatro cabezas; runtime no validado a {n("meta_universo_afiliados")}.',
        "p09": f'Gana a eps-greedy en {n("p09_gana_epsgreedy_thompson")} semillas, no en las {n("p09_umbral_semillas")} exigidas.',
        "planilla": "El sorteo de recepción es por línea, no por planilla completa (defecto menor).",
    }[id_]


def _mini_spark(values, *, unit_suffix="", width=64, height=22, color=None):
    color = color or COL_A2
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 1
    n_ = len(values)
    gap = 3
    bw = (width - gap * (n_ - 1)) / n_
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'role="img" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">']
    for i, v in enumerate(values):
        h = 3 + (height - 6) * (v - vmin) / (vmax - vmin) if vmax != vmin else height - 3
        x = i * (bw + gap)
        y = height - h
        opacity = 0.55 + 0.45 * (i / max(n_ - 1, 1))
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                     f'fill="{color}" opacity="{opacity:.2f}" rx="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def build_app_data():
    """Fuerza la construcción de todas las secciones (efecto secundario: llena
    CASE_REGISTRY y APP_RAW) y arma el modelo de contenido único para la app."""
    CASE_REGISTRY.clear()
    APP_RAW.clear()
    sec_portada()
    sec_capacidades()
    sec_casos()
    sec_torre_control()
    sec_roadmap()
    sec_supuestos()
    sec_anexo()
    sec_glosario()

    casos = []
    for c in CASE_REGISTRY:
        meta = CASE_META[c["id_"]]
        casos.append({
            "id": c["id_"],
            "numero": c["numero"],
            "nombre": c["nombre"],
            "titulo": c["titulo"],
            "area": meta["area"],
            "estado_nivel": meta["estado_nivel"],
            "estado_texto": meta["estado_texto"],
            "metrica_principal": _case_metrica_principal(c["id_"]),
            "reserva_corta": _case_reserva_corta(c["id_"]),
            "problema_html": c["problema"],
            "que_hace_html": c["que_hace"],
            "resultado_html": c["resultado_html"],
            "uso_html": c["uso_html"],
            "dashboard_html": c["mockup_html"],
            "produccion_html": c["produccion_html"],
            "reserva_html": c["reserva_html"],
            "fuente_texto": c["fuente_texto"],
        })

    kpi_sparks = [
        _mini_spark([num("p02_captura10_traspasos"), num("p02_captura20_traspasos")]),
        _mini_spark([num("p03_captura10_monto"), num("p03_captura20_monto")]),
        _mini_spark([num("p04_captura10_cese3m"), num("p04_captura10_cese6m")]),
        _mini_spark([num("p05_captura10_masa"), num("p05_captura20_masa")]),
        _mini_spark([num("p02_esc55_prioridad"), num("p02_escenario_saldo_retenido_valor"), num("p02_esc75_prioridad")]),
    ]
    kpis = APP_RAW["kpis"]
    for i, sp in enumerate(kpi_sparks):
        kpis[i]["spark"] = sp

    data = {
        "titulo": "Portafolio Analítico Cuprum",
        "areas": ["Comercial", "Operaciones", "Riesgo", "Tesorería", "Normativa"],
        "kpis": kpis,
        "mensajes": APP_RAW["mensajes"],
        "capacidades": APP_RAW["capacidades"],
        "construccion": APP_RAW["construccion"],
        "casos": casos,
        "torre_control": APP_RAW["torre_control"],
        "roadmap": APP_RAW["roadmap"],
        "supuestos": APP_RAW["supuestos"],
        "anexo": APP_RAW["anexo"],
        "glosario": APP_RAW["glosario"],
        "footer": {
            "pct_sintetico": n("meta_pct_sintetico"),
        },
    }
    return data


def sec_nav():
    items = "".join(f'<li><a href="#{sid}">{esc(label)}</a></li>' for sid, label in INDICE)
    return f"""<nav class="indice" aria-label="Índice"><h2>Índice</h2><ul>{items}</ul></nav>"""


# --------------------------------------------------------------------------
# Ensamblaje final
# --------------------------------------------------------------------------

CSS_APP = """
/* ===================== App shell (SPA) ===================== */
[hidden]{display:none!important;}
html, body{height:100%;}
body.app{margin:0; padding:0; overflow:hidden;}
.app-shell{position:fixed; inset:0; display:flex; flex-direction:column; background:var(--bg);}
.topbar{
  flex:0 0 auto; display:flex; align-items:stretch; gap:.7rem; flex-wrap:wrap;
  padding:.5rem .8rem; border-bottom:1px solid var(--border); background:var(--surface);
  min-width:0;
}
.topbar-title{font-family:var(--font-display); font-weight:700; font-size:1.05rem; white-space:nowrap; align-self:center;}
.topbar-chips{display:flex; gap:.35rem; flex-wrap:wrap; align-items:center; min-width:0;}
.topbar-chip{
  font-family:var(--font-body); font-size:.78rem; border:1px solid var(--border); background:var(--surface-2);
  color:var(--ink); border-radius:999px; padding:.25em .7em; cursor:pointer;
}
.topbar-chip[aria-pressed="true"]{background:var(--accent); color:var(--accent-ink); border-color:var(--accent);}
.topbar-search{margin-left:auto; min-width:0; flex:0 1 220px; align-self:center;}
.topbar-search input{
  width:100%; box-sizing:border-box; font-family:var(--font-body); font-size:.85rem;
  padding:.35em .7em; border-radius:8px; border:1px solid var(--border); background:var(--bg); color:var(--ink);
}
.app-body{flex:1 1 auto; min-height:0; display:flex; align-items:stretch; min-width:0;}
.navrail{
  flex:0 0 auto; width:96px; border-right:1px solid var(--border); background:var(--surface);
  display:flex; flex-direction:column; align-items:stretch; padding:.5rem .25rem; gap:.15rem; overflow-y:auto;
}
.navrail a{
  display:flex; flex-direction:column; align-items:center; gap:.15rem; padding:.5rem .2rem; border-radius:8px;
  color:var(--muted); text-decoration:none; font-size:.68rem; text-align:center; line-height:1.15;
}
.navrail a[aria-current="page"]{background:var(--surface-2); color:var(--accent); font-weight:600;}
.nav-dot{width:8px; height:8px; border-radius:50%; background:currentColor; opacity:.55; align-self:center;}
#app-main{flex:1 1 auto; min-width:0; overflow-y:auto; padding:1.1rem 1rem 1.4rem;}
.tabbar{
  display:none; flex:0 0 auto; border-top:1px solid var(--border); background:var(--surface);
  align-items:stretch;
}
.tabbar a{
  flex:1 1 0; min-width:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:.1rem; padding:.4rem .1rem; color:var(--muted); text-decoration:none; font-size:.64rem; text-align:center;
}
.tabbar a[aria-current="page"]{color:var(--accent); font-weight:600;}
@media (max-width:880px){
  .navrail{display:none;}
  .tabbar{display:flex;}
  #app-main{padding-bottom:1.2rem;}
}
.screen h1{margin-top:.2rem;}
.screen h2{margin-top:1.6rem;}
.kpi-grid.cinco{grid-template-columns:repeat(auto-fit,minmax(160px,1fr));}
.kpi-spark{margin:.3rem 0;}
.accesos-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:.8rem; margin:1rem 0;}
.acceso-card{
  display:flex; flex-direction:column; gap:.2rem; background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:.8rem .9rem; text-decoration:none; color:var(--ink); min-width:0;
  border-left:4px solid var(--accent-2);
}
.acceso-area{font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted);}
.acceso-titulo{font-weight:600; font-size:.94rem;}
.acceso-metrica{font-size:.82rem; color:var(--accent);}
.cap-sub{font-size:.86rem; color:var(--muted); margin:0 0 .3rem;}
.cap-desc{font-size:.9rem; margin:0;}
#casos-toolbar{display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; margin:.6rem 0 1rem;}
.grid-casos{display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:.9rem;}
.caso-card{
  display:flex; flex-direction:column; gap:.35rem; background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:.9rem 1rem; text-decoration:none; color:var(--ink); min-width:0;
}
.caso-card:hover, .caso-card:focus-visible{border-color:var(--accent);}
.caso-card-head{display:flex; align-items:center; justify-content:space-between; gap:.4rem; flex-wrap:wrap;}
.caso-card h3{margin:0; font-size:1rem;}
.caso-card-area{font-size:.78rem; color:var(--muted); margin:0;}
.caso-card-metrica{font-size:.86rem; color:var(--accent); margin:0;}
.caso-card-reserva{font-size:.78rem; color:var(--muted); margin:0; border-top:1px dashed var(--border); padding-top:.35rem;}
#caso-detalle{background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.9rem 1rem 1.2rem; min-width:0;}
.detalle-head{display:flex; align-items:center; gap:.5rem; flex-wrap:wrap; position:relative; padding-right:2.2rem;}
.detalle-head h3{margin:0; min-width:0;}
.btn-icon{
  position:absolute; top:-.2rem; right:-.2rem; width:2rem; height:2rem; border-radius:50%; border:1px solid var(--border);
  background:var(--surface-2); color:var(--ink); font-size:1.2rem; line-height:1; cursor:pointer;
}
.detalle-tabs{display:flex; gap:.3rem; flex-wrap:wrap; margin:.8rem 0; border-bottom:1px solid var(--border); padding-bottom:.5rem;}
.detalle-tab{
  font-family:var(--font-body); font-size:.82rem; padding:.35em .8em; border-radius:999px; border:1px solid var(--border);
  background:var(--surface-2); color:var(--ink); cursor:pointer;
}
.detalle-tab[aria-selected="true"]{background:var(--accent); color:var(--accent-ink); border-color:var(--accent);}
.detalle-body{min-height:6rem; min-width:0;}
.detalle-nav{display:flex; justify-content:space-between; gap:.6rem; margin-top:1rem; flex-wrap:wrap;}
.btn-nav{
  font-size:.82rem; text-decoration:none; color:var(--ink); background:var(--surface-2); border:1px solid var(--border);
  border-radius:8px; padding:.4em .8em; min-width:0; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.torre-filtros{display:flex; align-items:center; gap:.4rem; flex-wrap:wrap; margin:.7rem 0;}
.filtro-label{font-size:.78rem; color:var(--muted); margin-left:.4rem;}
.filtro-label:first-child{margin-left:0;}
.chip-filter{
  font-family:var(--font-body); font-size:.78rem; border:1px solid var(--border); background:var(--surface-2);
  color:var(--ink); border-radius:999px; padding:.2em .65em; cursor:pointer;
}
.chip-filter[aria-pressed="true"]{background:var(--accent-2); color:var(--accent-2-ink); border-color:var(--accent-2);}
#torre-sev{font-family:var(--font-body); font-size:.8rem; padding:.25em .5em; border-radius:6px; border:1px solid var(--border); background:var(--bg); color:var(--ink);}
.roadmap-pide{color:var(--muted); font-size:.86rem;}
table.sortable th{cursor:pointer; user-select:none; white-space:nowrap;}
table.sortable th[aria-sort]::after{content:""; display:inline-block; margin-left:.35em; width:0; height:0; vertical-align:middle;}
table.sortable th[aria-sort="ascending"]::after{border-left:4px solid transparent; border-right:4px solid transparent; border-bottom:5px solid var(--accent);}
table.sortable th[aria-sort="descending"]::after{border-left:4px solid transparent; border-right:4px solid transparent; border-top:5px solid var(--accent);}
.buscador{
  width:100%; max-width:340px; box-sizing:border-box; font-family:var(--font-body); font-size:.9rem;
  padding:.4em .8em; border-radius:8px; border:1px solid var(--border); background:var(--surface); color:var(--ink);
  margin:.6rem 0 1rem;
}
@media (prefers-reduced-motion: reduce){
  *{transition-duration:.001ms!important; animation-duration:.001ms!important;}
}
@media print{
  body.app{overflow:visible;}
  .app-shell{position:static; display:block;}
  .topbar, .navrail, .tabbar, .btn-icon, .detalle-tabs, .detalle-nav, #topbar-search, .torre-filtros, table.sortable th::after{display:none!important;}
  #app-main{overflow:visible; padding:0; height:auto;}
  .screen[hidden]{display:block!important;}
  #casos-grid, #caso-detalle{display:block!important;}
  .caso-card{break-inside:avoid; margin-bottom:.6rem;}
}
"""


JS_APP = """
(function(){
  'use strict';
  var dataEl = document.getElementById('app-data');
  var DATA = JSON.parse(dataEl.textContent);

  var SCREENS = [
    {id:'inicio', label:'Inicio'},
    {id:'casos', label:'Casos'},
    {id:'torre-control', label:'Torre'},
    {id:'roadmap', label:'Roadmap'},
    {id:'datos-limites', label:'Datos'},
    {id:'anexo', label:'Anexo'},
    {id:'glosario', label:'Glosario'}
  ];

  var state = {screen:'inicio', caseSlug:null, tab:'resumen', areaFilter:null, search:''};

  function lsGet(key, fallback){
    try { var v = localStorage.getItem(key); return v === null ? fallback : v; }
    catch(e){ return fallback; }
  }
  function lsSet(key, val){ try { localStorage.setItem(key, val); } catch(e){} }

  function esc(s){
    return String(s == null ? '' : s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function caseSlug(c){
    if (c.numero && c.numero !== '—') return c.numero;
    return c.id === 'planilla' ? 'planilla' : 'correccion-01';
  }
  function findCase(slug){
    for (var i=0;i<DATA.casos.length;i++){ if (caseSlug(DATA.casos[i]) === slug) return DATA.casos[i]; }
    return null;
  }

  function parseHash(){
    var h = location.hash.replace(/^#\/?/, '');
    var parts = h.split('/').filter(function(x){return x.length>0;});
    if (!parts.length) return null;
    var out = {screen:parts[0], caseSlug:null, tab:'resumen'};
    if (out.screen === 'casos' && parts[1]){
      out.caseSlug = decodeURIComponent(parts[1]);
      if (parts[2]) out.tab = parts[2];
    }
    return out;
  }

  function setHash(screen, slug, tab){
    var h = '#/' + screen;
    if (slug) h += '/' + encodeURIComponent(slug);
    if (slug && tab) h += '/' + tab;
    if (location.hash !== h) location.hash = h; else { syncFromHash(); render(); }
  }

  function syncFromHash(){
    var parsed = parseHash();
    if (!parsed){
      var last = lsGet('cuprum_screen', 'inicio');
      state.screen = last; state.caseSlug = null; state.tab = 'resumen';
      history.replaceState(null, '', '#/' + last);
      return;
    }
    state.screen = parsed.screen;
    state.caseSlug = parsed.caseSlug;
    state.tab = parsed.tab || 'resumen';
  }

  function chipClass(nivel){
    if (nivel === 'ok') return 'chip-ok';
    if (nivel === 'reserva') return 'chip-warn';
    if (nivel === 'critico') return 'chip-critical';
    return 'chip-info';
  }
  function chipHtml(text, nivel){
    return '<span class="chip ' + chipClass(nivel) + '"><span class="chip-dot" aria-hidden="true"></span>' + esc(text) + '</span>';
  }
  function sevLabel(sev){ return sev === 'critical' ? 'crítico' : (sev === 'warn' ? 'atención' : 'ok'); }
  function stripeHtml(sev){
    var cls = sev === 'critical' ? 'stripe-critical' : (sev === 'warn' ? 'stripe-warn' : 'stripe-ok');
    return '<span class="stripe ' + cls + '" aria-hidden="true"></span>';
  }
  function sevChip(sev){
    var cls = sev === 'critical' ? 'chip-critical' : (sev === 'warn' ? 'chip-warn' : 'chip-ok');
    return '<span class="chip ' + cls + '"><span class="chip-dot" aria-hidden="true"></span>' + sevLabel(sev) + '</span>';
  }

  function renderInicio(){
    var el = document.getElementById('screen-inicio');
    var kpisHtml = DATA.kpis.map(function(k){
      return '<div class="kpi-card"><div class="kpi-num">' + k.valor + '</div>' +
        '<div class="kpi-spark">' + (k.spark || '') + '</div>' +
        '<div class="kpi-label">' + esc(k.label) + '</div>' +
        (k.sub ? '<div class="kpi-sub">' + esc(k.sub) + '</div>' : '') + '</div>';
    }).join('');
    var msgHtml = DATA.mensajes.map(function(m){ return '<li>' + m + '</li>'; }).join('');
    var capHtml = DATA.capacidades.map(function(c){
      return '<div class="cap-card"><h4>' + esc(c.titulo) + '</h4><p class="cap-sub">' + esc(c.sub) +
        '</p><p class="cap-desc">' + c.desc + '</p></div>';
    }).join('');
    var porArea = {};
    DATA.casos.forEach(function(c){ if (!porArea[c.area]) porArea[c.area] = c; });
    var accesos = DATA.areas.map(function(a){
      var c = porArea[a];
      if (!c) return '';
      return '<a class="acceso-card" href="#/casos/' + caseSlug(c) + '">' +
        '<span class="acceso-area">' + esc(a) + '</span>' +
        '<span class="acceso-titulo">' + esc(c.titulo) + '</span>' +
        '<span class="acceso-metrica mono">' + esc(c.metrica_principal) + '</span></a>';
    }).join('');
    el.innerHTML =
      '<h1>Qué ofrece Data Analytics a AFP Cuprum</h1>' +
      '<p class="medida">Portafolio navegable por pantallas: mismo contenido y las mismas cifras que el informe completo.</p>' +
      '<h2>Cinco números</h2><div class="kpi-grid cinco">' + kpisHtml + '</div>' +
      '<h2>Cinco mensajes</h2><ul class="mensajes medida">' + msgHtml + '</ul>' +
      '<h2>Accesos directos por área</h2><div class="accesos-grid">' + accesos + '</div>' +
      '<h2>Qué ofrece el equipo</h2><div class="grid-cap">' + capHtml + '</div>' +
      '<div class="chart-wrap"><figure>' + DATA.construccion.chart_densidad +
      '<figcaption>' + esc(DATA.construccion.chart_caption) + '</figcaption></figure></div>';
  }

  function matchesFilter(c){
    if (state.areaFilter && c.area !== state.areaFilter) return false;
    if (state.search){
      var q = state.search.toLowerCase();
      var hay = (c.titulo + ' ' + c.nombre + ' ' + c.numero + ' ' + c.area).toLowerCase();
      if (hay.indexOf(q) === -1) return false;
    }
    return true;
  }
  function renderCasosGrid(){
    var grid = document.getElementById('casos-grid');
    if (!grid) return;
    var list = DATA.casos.filter(matchesFilter);
    var html = list.map(function(c){
      return '<a class="caso-card" href="#/casos/' + caseSlug(c) + '">' +
        '<div class="caso-card-head"><span class="badge badge-accent">' + esc(c.numero) + '</span>' +
        chipHtml(c.estado_texto, c.estado_nivel) + '</div>' +
        '<h3>' + esc(c.titulo) + '</h3>' +
        '<p class="caso-card-area">' + esc(c.area) + ' &middot; ' + esc(c.nombre) + '</p>' +
        '<p class="caso-card-metrica mono">' + esc(c.metrica_principal) + '</p>' +
        '<p class="caso-card-reserva">' + esc(c.reserva_corta) + '</p></a>';
    }).join('');
    grid.innerHTML = html || '<p class="medida">Ningún caso coincide con el filtro.</p>';
  }
  var TABS = [
    {id:'resumen', label:'Resumen'}, {id:'dashboard', label:'Dashboard'},
    {id:'produccion', label:'Producción'}, {id:'fuente', label:'Fuente'}
  ];
  function tabContent(c, tab){
    if (tab === 'resumen') return '<div class="medida"><h4>Problema</h4>' + c.problema_html +
      '<h4>Qué hace</h4>' + c.que_hace_html +
      '<h4>Resultado en el sandbox</h4>' + c.resultado_html +
      '<h4>Cómo lo usaría el área</h4>' + c.uso_html + '</div>' + c.reserva_html;
    if (tab === 'dashboard') return c.dashboard_html;
    if (tab === 'produccion') return '<div class="medida">' + c.produccion_html + '</div>';
    if (tab === 'fuente') return '<p class="fuente-cita">Fuente: ' + esc(c.fuente_texto) + '</p>';
    return '';
  }
  function renderDetalle(){
    var det = document.getElementById('caso-detalle');
    var grid = document.getElementById('casos-grid');
    if (!det || !grid) return;
    var c = findCase(state.caseSlug);
    if (!c){ det.hidden = true; grid.hidden = false; det.innerHTML=''; return; }
    grid.hidden = true; det.hidden = false;
    var idx = DATA.casos.indexOf(c);
    var prev = DATA.casos[(idx - 1 + DATA.casos.length) % DATA.casos.length];
    var next = DATA.casos[(idx + 1) % DATA.casos.length];
    var tabsHtml = TABS.map(function(t){
      var sel = (t.id === state.tab);
      return '<button class="detalle-tab" role="tab" aria-selected="' + sel + '" data-tab="' + t.id +
        '" id="tab-' + t.id + '">' + t.label + '</button>';
    }).join('');
    det.innerHTML =
      '<div class="detalle-head"><button class="btn-icon" id="btn-cerrar-detalle" aria-label="Cerrar y volver a la grilla">&times;</button>' +
      '<span class="badge badge-accent">' + esc(c.numero) + '</span><h3>' + esc(c.titulo) + '</h3>' +
      '<span class="badge">' + esc(c.nombre) + '</span>' + chipHtml(c.estado_texto, c.estado_nivel) + '</div>' +
      '<div class="detalle-tabs" role="tablist">' + tabsHtml + '</div>' +
      '<div class="detalle-body" id="detalle-body" role="tabpanel">' + tabContent(c, state.tab) + '</div>' +
      '<div class="detalle-nav"><a href="#/casos/' + caseSlug(prev) + '/' + state.tab + '" class="btn-nav">&larr; ' +
      esc(prev.numero) + ' ' + esc(prev.nombre) + '</a><a href="#/casos/' + caseSlug(next) + '/' + state.tab +
      '" class="btn-nav">' + esc(next.numero) + ' ' + esc(next.nombre) + ' &rarr;</a></div>';
    var tabButtons = det.querySelectorAll('.detalle-tab');
    for (var i=0;i<tabButtons.length;i++){
      tabButtons[i].addEventListener('click', (function(btn){
        return function(){ setHash('casos', state.caseSlug, btn.getAttribute('data-tab')); };
      })(tabButtons[i]));
    }
    var closeBtn = document.getElementById('btn-cerrar-detalle');
    if (closeBtn) closeBtn.addEventListener('click', function(){ setHash('casos', null, null); });
    var focusTarget = document.getElementById('tab-' + state.tab);
    if (focusTarget) focusTarget.focus({preventScroll:true});
  }
  function renderCasos(){ renderCasosGrid(); renderDetalle(); }

  var torreState = {area:null, sev:null};
  function renderTorre(){
    var el = document.getElementById('screen-torre-control');
    var t = DATA.torre_control;
    var kpisHtml = t.kpis.map(function(k){
      return '<div class="md-kpi"><div class="md-kpi-num">' + k[0] + '</div><div class="md-kpi-label">' + esc(k[1]) + '</div></div>';
    }).join('');
    var areasChips = DATA.areas.map(function(a){
      return '<button class="chip-filter" data-area="' + esc(a) + '" aria-pressed="' + (torreState.area===a) + '">' + esc(a) + '</button>';
    }).join('');
    el.innerHTML =
      '<h2>Torre de control: los casos en un solo tablero</h2>' +
      '<p class="medida">Vista integrada de las capacidades de priorización, con alertas filtrables por área y severidad.</p>' +
      '<div class="mockup"><div class="mockup-bar"><span class="mockup-dot"></span><span class="mockup-dot"></span>' +
      '<span class="mockup-dot"></span><span class="mockup-title">Torre de control &middot; portafolio Cuprum</span></div>' +
      '<div class="mockup-body"><div class="md-kpis">' + kpisHtml + '</div>' +
      '<div class="torre-filtros"><span class="filtro-label">Área</span>' +
      '<button class="chip-filter" data-area="" aria-pressed="' + (torreState.area===null) + '">Todas</button>' + areasChips +
      '<span class="filtro-label">Severidad</span><select id="torre-sev">' +
      '<option value="">Todas</option><option value="critical">Crítico</option><option value="warn">Atención</option>' +
      '<option value="ok">Ok</option></select></div>' +
      '<div class="md-table-wrap"><table class="md-table"><thead><tr><th>Área</th><th>Alerta</th><th>Detalle</th>' +
      '<th>Valor</th><th>Estado</th></tr></thead><tbody id="torre-tbody"></tbody></table></div>' +
      '<div class="md-chart"><figure>' + t.chart_svg + '<figcaption>' + esc(t.chart_caption) + '</figcaption></figure></div>' +
      '</div></div>';
    var chipButtons = el.querySelectorAll('.chip-filter');
    for (var i=0;i<chipButtons.length;i++){
      chipButtons[i].addEventListener('click', (function(b){
        return function(){
          torreState.area = b.getAttribute('data-area') || null;
          for (var j=0;j<chipButtons.length;j++){ chipButtons[j].setAttribute('aria-pressed', String(chipButtons[j]===b)); }
          renderTorreTable();
        };
      })(chipButtons[i]));
    }
    var sevSel = document.getElementById('torre-sev');
    if (sevSel) sevSel.addEventListener('change', function(e){ torreState.sev = e.target.value || null; renderTorreTable(); });
    renderTorreTable();
  }
  function renderTorreTable(){
    var t = DATA.torre_control;
    var rows = t.filas.filter(function(f){
      if (torreState.area && f.area !== torreState.area) return false;
      if (torreState.sev && f.severidad !== torreState.sev) return false;
      return true;
    });
    var body = document.getElementById('torre-tbody');
    if (!body) return;
    var html = rows.map(function(f){
      return '<tr class="sev-' + f.severidad + '"><td>' + esc(f.area) + '</td><td>' + stripeHtml(f.severidad) +
        esc(f.alerta) + '</td><td>' + esc(f.detalle) + '</td><td class="num">' + esc(f.valor) + '</td><td>' +
        sevChip(f.severidad) + '</td></tr>';
    }).join('');
    body.innerHTML = html || '<tr><td colspan="5">Sin alertas para este filtro.</td></tr>';
  }

  function renderRoadmap(){
    var el = document.getElementById('screen-roadmap');
    var items = DATA.roadmap.map(function(p){
      return '<li><h4>' + esc(p.titulo) + '</h4><p>' + p.desc + '</p><p class="roadmap-pide"><strong>Se pide al negocio:</strong> ' + p.pide + '</p></li>';
    }).join('');
    el.innerHTML = '<h2>Roadmap</h2><p class="medida">En este orden porque cada paso reduce el riesgo del siguiente. ' +
      'Es la única pantalla con pasos numerados.</p><ol class="roadmap medida">' + items + '</ol>';
  }

  function renderDatos(){
    var el = document.getElementById('screen-datos-limites');
    var s = DATA.supuestos;
    var datosHtml = s.datos.map(function(x){ return '<li>' + x + '</li>'; }).join('');
    var modelosHtml = s.modelos.map(function(x){ return '<li>' + x + '</li>'; }).join('');
    el.innerHTML = '<h2>Supuestos, reservas y límites</h2><p class="medida">' + s.intro + '</p>' +
      '<h3>Sobre los datos</h3><ul class="medida">' + datosHtml + '</ul>' +
      '<h3>Sobre los modelos</h3><ul class="medida">' + modelosHtml + '</ul>';
  }

  var anexoSort = {col:null, dir:1};
  var ANEXO_COLS = ['proyecto','datos','modelo','metrica','tests','runtime','revision'];
  var ANEXO_LABELS = {proyecto:'Proyecto', datos:'Datos', modelo:'Modelo', metrica:'Métrica principal', tests:'Tests', runtime:'Runtime', revision:'Revisión'};
  function renderAnexo(){
    var el = document.getElementById('screen-anexo');
    var head = ANEXO_COLS.map(function(c){
      var sortDir = anexoSort.col===c ? (anexoSort.dir>0 ? 'ascending' : 'descending') : 'none';
      return '<th data-col="' + c + '" aria-sort="' + sortDir + '" tabindex="0" role="button">' + ANEXO_LABELS[c] + '</th>';
    }).join('');
    el.innerHTML = '<h2>Anexo técnico</h2><p class="medida">Una fila por proyecto. Pulsa un encabezado (o Enter) para ordenar.</p>' +
      '<div class="tabla-wrap"><table class="tabla sortable"><thead><tr>' + head + '</tr></thead><tbody id="anexo-tbody"></tbody></table></div>';
    var ths = el.querySelectorAll('th');
    for (var i=0;i<ths.length;i++){
      (function(th){
        function toggle(){
          var col = th.getAttribute('data-col');
          if (anexoSort.col === col) anexoSort.dir *= -1; else { anexoSort.col = col; anexoSort.dir = 1; }
          renderAnexo();
        }
        th.addEventListener('click', toggle);
        th.addEventListener('keydown', function(e){ if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); toggle(); } });
      })(ths[i]);
    }
    renderAnexoRows();
  }
  function renderAnexoRows(){
    var rows = DATA.anexo.slice();
    if (anexoSort.col){
      rows.sort(function(a,b){
        var av = a[anexoSort.col], bv = b[anexoSort.col];
        return (av > bv ? 1 : (av < bv ? -1 : 0)) * anexoSort.dir;
      });
    }
    var body = document.getElementById('anexo-tbody');
    if (!body) return;
    body.innerHTML = rows.map(function(r){
      return '<tr><td>' + esc(r.proyecto) + '</td><td>' + esc(r.datos) + '</td><td>' + esc(r.modelo) + '</td>' +
        '<td class="num">' + esc(r.metrica) + '</td><td class="num">' + esc(r.tests) + '</td><td class="num">' +
        esc(r.runtime) + '</td><td>' + esc(r.revision) + '</td></tr>';
    }).join('');
  }

  function renderGlosario(){
    var el = document.getElementById('screen-glosario');
    el.innerHTML = '<h2>Glosario</h2>' +
      '<input type="search" id="glosario-buscador" class="buscador" placeholder="Buscar termino" aria-label="Buscar en el glosario">' +
      '<dl class="glosario medida" id="glosario-lista"></dl>';
    var input = document.getElementById('glosario-buscador');
    input.addEventListener('input', function(e){ renderGlosarioLista(e.target.value); });
    renderGlosarioLista('');
  }
  function renderGlosarioLista(q){
    q = (q || '').toLowerCase();
    var items = DATA.glosario.filter(function(g){
      return !q || g.termino.toLowerCase().indexOf(q) > -1 || g.definicion.toLowerCase().indexOf(q) > -1;
    });
    var el = document.getElementById('glosario-lista');
    if (!el) return;
    el.innerHTML = items.map(function(g){ return '<dt>' + esc(g.termino) + '</dt><dd>' + esc(g.definicion) + '</dd>'; }).join('') ||
      '<p class="medida">Sin coincidencias.</p>';
  }

  function renderChromeState(){
    var navLinks = document.querySelectorAll('[data-nav-screen]');
    for (var i=0;i<navLinks.length;i++){
      var active = navLinks[i].getAttribute('data-nav-screen') === state.screen;
      navLinks[i].setAttribute('aria-current', active ? 'page' : 'false');
    }
    var chips = document.querySelectorAll('.topbar-chip');
    for (var j=0;j<chips.length;j++){
      var area = chips[j].getAttribute('data-area') || null;
      chips[j].setAttribute('aria-pressed', String(area === state.areaFilter));
    }
  }

  var RENDERERS = {
    'inicio':renderInicio, 'casos':renderCasos, 'torre-control':renderTorre,
    'roadmap':renderRoadmap, 'datos-limites':renderDatos, 'anexo':renderAnexo, 'glosario':renderGlosario
  };
  function render(){
    var screen = RENDERERS[state.screen] ? state.screen : 'inicio';
    for (var i=0;i<SCREENS.length;i++){
      var el = document.getElementById('screen-' + SCREENS[i].id);
      if (el) el.hidden = (SCREENS[i].id !== screen);
    }
    RENDERERS[screen]();
    renderChromeState();
    lsSet('cuprum_screen', screen);
    lsSet('cuprum_area', state.areaFilter || '');
  }

  window.addEventListener('hashchange', function(){ syncFromHash(); render(); });

  document.addEventListener('click', function(e){
    var chip = e.target.closest ? e.target.closest('.topbar-chip') : null;
    if (chip){
      var area = chip.getAttribute('data-area') || null;
      state.areaFilter = (state.areaFilter === area) ? null : area;
      renderChromeState();
      if (state.screen === 'casos') renderCasosGrid();
      if (state.screen === 'inicio') renderInicio();
    }
  });

  var searchEl = document.getElementById('topbar-search');
  if (searchEl){
    searchEl.addEventListener('input', function(e){
      state.search = e.target.value;
      if (state.screen === 'casos') renderCasosGrid();
    });
  }

  document.addEventListener('keydown', function(e){
    if (state.screen === 'casos' && state.caseSlug){
      if (e.key === 'Escape'){ e.preventDefault(); setHash('casos', null, null); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft'){
        var c = findCase(state.caseSlug);
        if (!c) return;
        var idx = DATA.casos.indexOf(c);
        var dir = (e.key === 'ArrowRight') ? 1 : -1;
        var nextC = DATA.casos[(idx + dir + DATA.casos.length) % DATA.casos.length];
        e.preventDefault();
        setHash('casos', caseSlug(nextC), state.tab);
      }
    }
  });

  syncFromHash();
  state.areaFilter = lsGet('cuprum_area', '') || null;
  render();
})();
"""


def _screen_ids():
    return [
        ("inicio", "Inicio"),
        ("casos", "Casos"),
        ("torre-control", "Torre"),
        ("roadmap", "Roadmap"),
        ("datos-limites", "Datos"),
        ("anexo", "Anexo"),
        ("glosario", "Glosario"),
    ]


def build_app_html():
    data = build_app_data()
    data_json = json.dumps(data, ensure_ascii=False)
    nav_items = "".join(
        f'<a href="#/{sid}" data-nav-screen="{sid}"><span class="nav-dot" aria-hidden="true"></span>{esc(label)}</a>'
        for sid, label in _screen_ids()
    )
    area_chips = "".join(
        f'<button class="topbar-chip" data-area="{esc(a)}" aria-pressed="false">{esc(a)}</button>'
        for a in data["areas"]
    )
    screens_html = "".join(
        f'<section class="screen" id="screen-{sid}" hidden></section>'
        for sid in ["inicio", "torre-control", "roadmap", "datos-limites", "anexo", "glosario"]
    )
    casos_screen = (
        '<section class="screen" id="screen-casos" hidden>'
        '<div id="casos-toolbar"><h2 style="margin:0;">Casos de uso</h2></div>'
        '<div class="grid-casos" id="casos-grid"></div>'
        '<div id="caso-detalle" hidden></div>'
        "</section>"
    )
    fonts_link = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Bricolage+Grotesque:wght@500;700&"
        "family=Source+Sans+3:wght@400;600;700&"
        'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
    )
    html_doc = (
        "<!doctype html>\n"
        '<html lang="es">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Portafolio Analítico Cuprum</title>\n"
        '<meta name="description" content="Portafolio de Data Analytics de AFP Cuprum, navegable por '
        'pantallas: casos de uso, torre de control, roadmap y anexo.">\n'
        f"{fonts_link}\n"
        f"<style>{CSS}{CSS_APP}</style>\n"
        "</head>\n"
        '<body class="app">\n'
        '<div class="app-shell">\n'
        '  <header class="topbar">\n'
        '    <span class="topbar-title">Portafolio Analítico Cuprum</span>\n'
        f'    <div class="topbar-chips" role="group" aria-label="Filtrar por área">{area_chips}</div>\n'
        '    <div class="topbar-search"><input type="search" id="topbar-search" placeholder="Buscar caso"'
        ' aria-label="Buscar caso"></div>\n'
        "  </header>\n"
        '  <div class="app-body">\n'
        f'    <nav class="navrail" aria-label="Navegación principal">{nav_items}</nav>\n'
        f'    <main id="app-main">{screens_html}{casos_screen}</main>\n'
        "  </div>\n"
        f'  <nav class="tabbar" aria-label="Navegación principal (móvil)">{nav_items}</nav>\n'
        "</div>\n"
        f'<script type="application/json" id="app-data">{data_json}</script>\n'
        f"<script>{JS_APP}</script>\n"
        "</body>\n"
        "</html>\n"
    )
    return html_doc


def build_document():
    body = "".join([
        sec_portada(),
        sec_capacidades(),
        sec_casos(),
        sec_torre_control(),
        sec_roadmap(),
        sec_supuestos(),
        sec_anexo(),
        sec_glosario(),
    ])
    nav = sec_nav()
    fonts_link = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Bricolage+Grotesque:wght@500;700&"
        "family=Source+Sans+3:wght@400;600;700&"
        'family=IBM+Plex+Mono:wght@400;500;600&display=swap">'
    )
    html_doc = (
        "<!doctype html>\n"
        '<html lang="es">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Portafolio Analítico Cuprum</title>\n"
        '<meta name="description" content="Informe ejecutivo del portafolio de Data Analytics de AFP '
        'Cuprum: qué ofrece el equipo, evidencia en el sandbox sintético y camino a producción.">\n'
        f"{fonts_link}\n"
        f"<style>{CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<div class="wrap layout">\n'
        f"{nav}\n"
        f"<main>{body}"
        '<footer class="pie">Informe generado por <code>docs/build_reporte.py</code> a partir de '
        f'<code>docs/reporte_datos.json</code>. Datos {n("meta_pct_sintetico")}&nbsp;% sintéticos salvo donde se cita la Ficha '
        "Estadística Previsional N° 165 de la Superintendencia de Pensiones o la Ley 21.735. AFP Cuprum "
        "es la persona de este sandbox; ningún dato describe a un afiliado real.</footer>"
        "</main>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )
    return html_doc


# --------------------------------------------------------------------------
# Verificación: etiquetas balanceadas (sin dependencias externas)
# --------------------------------------------------------------------------

_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}


class _BalanceChecker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass  # etiqueta autocontenida (<rect .../>): no altera la pila

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"cierre sin apertura: </{tag}>")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            # se cerró algo fuera de orden; reportar y desenrollar hasta encontrarlo
            self.errors.append(f"cierre fuera de orden: </{tag}> (tope de pila: <{self.stack[-1]}>)")
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            if self.stack:
                self.stack.pop()
        else:
            self.errors.append(f"cierre sin apertura correspondiente: </{tag}>")


def check_balanced_tags(html_text):
    checker = _BalanceChecker()
    checker.feed(html_text)
    checker.close()
    if checker.stack:
        checker.errors.append(f"etiquetas sin cerrar al final: {checker.stack}")
    return checker.errors


def _verificar_y_escribir(doc, out_path, etiqueta):
    print(f"--- {etiqueta} ---")
    errors = check_balanced_tags(doc)
    if errors:
        print("ADVERTENCIA: posibles etiquetas HTML desbalanceadas:")
        for e in errors:
            print(" -", e)
    else:
        print("Chequeo de etiquetas: OK (balanceadas).")

    size_bytes = len(doc.encode("utf-8"))
    print(f"Tamaño del HTML: {size_bytes / 1024:.1f} KB ({size_bytes} bytes)")
    if size_bytes > 1_500_000:
        print("ADVERTENCIA: el HTML supera 1,5 MB.")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"Escrito: {out_path}")


def main():
    import sys
    modo_documento = "--documento" in sys.argv

    if modo_documento:
        doc = build_document()
        _verificar_y_escribir(doc, DOC_OUT_PATH, "Documento (una pantalla, para imprimir)")
    else:
        app = build_app_html()
        _verificar_y_escribir(app, OUT_PATH, "App (SPA, navegable por pantallas)")

    unused = sorted(set(D.keys()) - USED)
    if unused:
        print(f"Datos definidos en reporte_datos.json y NO usados en esta generación ({len(unused)}):")
        for k in unused:
            print(" -", k)
    else:
        print("Todas las entradas de reporte_datos.json se usaron en esta generación.")


if __name__ == "__main__":
    main()
