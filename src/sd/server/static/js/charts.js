// Графики на чистом SVG: столбики со штриховкой «ждут», шкала-спидометр, спарклайн, интервал F1. Без библиотек и без внешних запросов.
import { h, s } from "./dom.js";

let uid = 0;

/** days = [{label, done, pending}] → столбики: сплошные — решены, штрих — ждут. */
export function stackedBars(days, { width = 280, height = 130, label = "График" } = {}) {
  const id = `hatch${++uid}`;
  const max = Math.max(1, ...days.map((d) => d.done + d.pending));
  const n = Math.max(1, days.length), gap = 8, bw = Math.max(8, Math.min(30, (width - gap * (n + 1)) / n));
  const base = height - 18, scale = (base - 6) / max;
  const kids = [s("defs", null, s("pattern", { id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, s("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: "#0E1610", "stroke-width": 2 })))];
  days.forEach((d, i) => {
    const x = gap + i * (bw + gap);
    const hd = d.done * scale, hp = d.pending * scale;
    if (hd > 0) kids.push(s("rect", { x, y: base - hd, width: bw, height: hd, fill: "#0E1610" }));
    if (hp > 0) kids.push(s("rect", { x, y: base - hd - hp - (hd ? 2 : 0), width: bw, height: hp, fill: `url(#${id})`, stroke: "#0E1610", "stroke-width": 1 }));
    kids.push(s("text", { x: x + bw / 2, y: height - 4, "text-anchor": "middle", "font-size": 11, fill: "#45514A", "font-family": "Geologica, sans-serif" }, d.label));
    kids.push(s("title", null, `${d.label}: решено ${d.done}, ждут ${d.pending}`));
  });
  return s("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", style: { maxWidth: `${width}px` }, role: "img", "aria-label": label }, ...kids);
}

/** Полукруглая шкала из штрихов; value 0..1. */
export function gauge(value, { ticks = 48 } = {}) {
  const v = Math.max(0, Math.min(1, value || 0)), fill = Math.round(ticks * v), lines = [];
  for (let i = 0; i < ticks; i++) {
    const a = Math.PI * (1 - i / (ticks - 1)), r1 = 94, r2 = i % 4 === 0 ? 70 : 78;
    lines.push(s("line", { x1: (100 + r1 * Math.cos(a)).toFixed(1), y1: (104 - r1 * Math.sin(a)).toFixed(1), x2: (100 + r2 * Math.cos(a)).toFixed(1), y2: (104 - r2 * Math.sin(a)).toFixed(1), stroke: i < fill ? "#0B7A4B" : "#B9C2B3", "stroke-width": 2.4 }));
  }
  return s("svg", { viewBox: "0 0 200 110", width: 210, height: 116, "aria-hidden": "true" }, ...lines);
}

export function sparkline(values, { width = 60, height = 28, color = "#0E1610" } = {}) {
  const v = values.length > 1 ? values : [0, 0];
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const pts = v.map((y, i) => `${((i / (v.length - 1)) * (width - 2) + 1).toFixed(1)},${(height - 3 - ((y - lo) / span) * (height - 6)).toFixed(1)}`).join(" ");
  return s("svg", { viewBox: `0 0 ${width} ${height}`, width, height, "aria-hidden": "true" }, s("polyline", { fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "miter", points: pts }));
}

/** Интервал на шкале 0..1: линия lo–hi и точка pt. */
export function whisker(lo, hi, pt, { width = 110 } = {}) {
  const x = (v) => Math.max(0, Math.min(1, v)) * width;
  const kids = [s("path", { d: `M0 11H${width}`, stroke: "#E3E8DD", "stroke-width": 4 })];
  if (lo != null && hi != null) kids.push(s("path", { d: `M${x(lo)} 11H${x(hi)}`, stroke: "#0E1610", "stroke-width": 4 }));
  if (pt != null) kids.push(s("rect", { x: x(pt) - 5, y: 6, width: 10, height: 10, fill: "#0B7A4B", stroke: "#fff", "stroke-width": 2 }));
  return s("svg", { viewBox: `0 0 ${width} 22`, width, height: 22, role: "img", "aria-label": lo != null ? `Интервал ${lo.toFixed(2)}–${hi.toFixed(2)}` : "Нет интервала" }, ...kids);
}

/** Горизонтальные полосы: items = [{label, value, text?}], max по умолчанию — наибольшее значение. */
export function hbars(items, { max, cls = "" } = {}) {
  const top = max || Math.max(1, ...items.map((i) => i.value));
  return h("div", { class: "stack stack--sm" }, ...items.map((i) => h("div", { class: "row row--nowrap", style: { gap: "10px", fontSize: "13px" } },
    h("span", { style: { width: "112px", flex: "none" }, class: "ellipsis", title: i.label }, i.label),
    h("div", { class: ["bar grow", cls] }, h("i", { style: { width: `${Math.round((i.value / top) * 100)}%` } })),
    h("b", { style: { width: "34px", textAlign: "right" } }, i.text ?? String(i.value)))));
}
