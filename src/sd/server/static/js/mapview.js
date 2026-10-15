// Схема камер на плоскости: широта/долгота вписываются в рамку (без подложки и выдуманных улиц), камеры без координат раскладываются по районам условно.
import { s } from "./dom.js";
import { camName } from "./store.js";

export const W = 1000, H = 620, PAD = 80;

/** Позиции камер. Возвращает { mode: 'geo'|'grid', points: [{id, x, y, m}], unplaced: [m] } — m это CameraMetrics. */
export function layout(list) {
  const geo = list.filter((m) => m.camera.lat != null && m.camera.lon != null);
  if (geo.length) {
    const lat0 = geo.reduce((a, m) => a + m.camera.lat, 0) / geo.length;
    const k = Math.cos((lat0 * Math.PI) / 180);
    const pts = geo.map((m) => ({ m, gx: m.camera.lon * k, gy: -m.camera.lat }));
    const xs = pts.map((p) => p.gx), ys = pts.map((p) => p.gy);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = x1 > x0 ? (W - 2 * PAD) / (x1 - x0) : 0, sy = y1 > y0 ? (H - 2 * PAD) / (y1 - y0) : 0;
    const sc = sx && sy ? Math.min(sx, sy) : sx || sy || 1;
    const ox = (W - (x1 - x0) * sc) / 2, oy = (H - (y1 - y0) * sc) / 2;
    return { mode: "geo", points: pts.map((p) => ({ id: p.m.camera.camera_id, m: p.m, x: ox + (p.gx - x0) * sc, y: oy + (p.gy - y0) * sc })), unplaced: list.filter((m) => !geo.includes(m)) };
  }
  // координат нет ни у одной камеры: кластеры по районам, внутри — по кругу
  const groups = new Map();
  for (const m of list) { const g = m.camera.district || m.camera.city || "—"; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(m); }
  const names = [...groups.keys()].sort();
  const cols = Math.max(1, Math.ceil(Math.sqrt(names.length * (W / H))));
  const rows = Math.ceil(names.length / cols);
  const points = [];
  names.forEach((g, gi) => {
    const cx = PAD + ((gi % cols) + 0.5) * ((W - 2 * PAD) / cols), cy = PAD + (Math.floor(gi / cols) + 0.5) * ((H - 2 * PAD) / rows);
    const ms = groups.get(g);
    const r = ms.length > 1 ? Math.min(70, 26 + ms.length * 6) : 0;
    ms.forEach((m, i) => { const a = (i / ms.length) * Math.PI * 2 - Math.PI / 2; points.push({ id: m.camera.camera_id, m, x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), group: g, gx: cx, gy: cy }); });
  });
  return { mode: "grid", points, unplaced: [] };
}

function stateOf(m) {
  if (!m.camera.enabled) return { color: "#8A958D", label: "отключена" };
  if (m.pending > 0) return { color: "#D93A1B", label: `ждут решения: ${m.pending}` };
  return { color: "#0B7A4B", label: "без очереди" };
}
export const stateLabel = (m) => stateOf(m).label;

export function drawMap(lay, { selected, onSelect } = {}) {
  const kids = [];
  for (let x = 0; x <= W; x += 100) kids.push(s("line", { x1: x, y1: 0, x2: x, y2: H, stroke: "#E3E8DD", "stroke-width": 1 }));
  for (let y = 0; y <= H; y += 100) kids.push(s("line", { x1: 0, y1: y, x2: W, y2: y, stroke: "#E3E8DD", "stroke-width": 1 }));
  if (lay.mode === "grid") {
    const seen = new Set();
    for (const p of lay.points) if (!seen.has(p.group)) { seen.add(p.group); kids.push(s("text", { x: p.gx, y: p.gy - 92 < 20 ? p.gy + 100 : p.gy - 92, "text-anchor": "middle", "font-size": 15, "font-weight": 600, fill: "#5A665E", "font-family": "Geologica, sans-serif" }, p.group)); }
  }
  for (const p of lay.points) {
    const st = stateOf(p.m), on = selected === p.id, r = on ? 24 : 19;
    const g = s("g", { class: "pin", tabindex: "0", role: "button", "aria-label": `${camName(p.m.camera)}: ${st.label}`, "aria-pressed": String(on) },
      on ? s("rect", { x: p.x - 31, y: p.y - 31, width: 62, height: 62, fill: "none", stroke: "#0E1610", "stroke-width": 1.5 }) : null,
      s("circle", { cx: p.x, cy: p.y, r, fill: "#FFFFFF", stroke: st.color, "stroke-width": 3 }),
      p.m.pending > 0 ? s("text", { x: p.x, y: p.y + 5.5, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#AE2D12", "font-family": "Geologica, sans-serif" }, p.m.pending > 99 ? "99+" : String(p.m.pending)) : s("circle", { cx: p.x, cy: p.y, r: 5, fill: st.color }),
      s("text", { x: p.x, y: p.y + r + 19, "text-anchor": "middle", "font-size": 14, fill: "#0E1610", "font-family": "Geologica, sans-serif" }, p.m.camera.cam_index || p.m.camera.title || p.id),
      s("title", null, `${camName(p.m.camera)} — ${st.label}`));
    const pick = () => onSelect && onSelect(p.id);
    g.addEventListener("click", pick);
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    kids.push(g);
  }
  return s("svg", { class: "map", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet", role: "group", "aria-label": "Схема расположения камер" }, ...kids);
}
