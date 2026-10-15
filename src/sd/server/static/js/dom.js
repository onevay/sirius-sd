// Построение интерфейса без фреймворка и без innerHTML: любой текст вставляется как текст (защита от XSS), узлы собираются функцией h().
const SVG_NS = "http://www.w3.org/2000/svg";

function applyProps(el, props) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.setAttribute("class", Array.isArray(v) ? v.filter(Boolean).join(" ") : v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "value" && "value" in el) el.value = v;
    else if (k === "checked" || k === "disabled" || k === "selected" || k === "hidden" || k === "readOnly" || k === "required") { if (v) el.setAttribute(k.toLowerCase(), ""); }
    else el.setAttribute(k, v === true ? "" : String(v));
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false || c === true) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** h("div", {class: "card"}, "текст", h("b", null, "жирный")) */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  applyProps(el, props);
  append(el, children);
  return el;
}

export function s(tag, props, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyProps(el, props);
  append(el, children);
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

/** Добавляет узлы в конец, пропуская null/false/undefined (родной .append() вставил бы текст «null»). */
export function put(parent, ...nodes) {
  append(parent, nodes);
  return parent;
}

export function mount(parent, ...nodes) {
  clear(parent);
  append(parent, nodes);
  return parent;
}

// Значки: простые штриховые контуры. Добавить значок — добавить путь (viewBox 24×24).
const PATHS = {
  grid: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  bell: "M6 9a6 6 0 0 1 12 0c0 6 2 7 2 7H4s2-1 2-7M10 20a2 2 0 0 0 4 0",
  pin: "M12 21s7-6.2 7-11a7 7 0 0 0-14 0c0 4.8 7 11 7 11zM12 12a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z",
  video: "M3 7h12a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H3zM17 11l4-2.5v7L17 13",
  file: "M7 3h7l4 4v14H7zM14 3v4h4M10 12h5M10 16h5",
  chart: "M5 20V10M12 20V4M19 20v-7",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  edit: "M4 20l4-1 11-11-3-3L5 16zM14 7l3 3",
  cpu: "M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4",
  settings: "M4 7h10M18 7h2M4 17h2M10 17h10M16 4v6M8 14v6",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM21 21l-4.5-4.5",
  check: "M5 12.5l4.5 4.5L19 7",
  x: "M6 6l12 12M18 6L6 18",
  help: "M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 1-1 1.7M12 17.5v.01",
  play: "M8 5l11 7-11 7z",
  pause: "M7 5h4v14H7zM13 5h4v14h-4z",
  filter: "M4 6h16M7 12h10M10 18h4",
  plus: "M12 5v14M5 12h14",
  arrow: "M5 12h14M13 6l6 6-6 6",
  back: "M19 12H5M11 6l-6 6 6 6",
  chevl: "M15 6l-6 6 6 6",
  chevr: "M9 6l6 6-6 6",
  chevd: "M6 9l6 6 6-6",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2",
  upload: "M12 16V4M7 9l5-5 5 5M4 20h16",
  download: "M12 4v12M7 11l5 5 5-5M4 20h16",
  warn: "M12 4l9 16H3zM12 10v4M12 17.5v.01",
  star: "M12 3l2.7 5.8 6.3.7-4.7 4.3 1.3 6.2L12 17l-5.6 3 1.3-6.2L3 9.5l6.3-.7z",
  more: "M5 12h.01M12 12h.01M19 12h.01",
  users: "M9 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM2.5 20a6.5 6.5 0 0 1 13 0M16 4.5a3.5 3.5 0 0 1 0 6.5M18 14a6.5 6.5 0 0 1 3.5 6",
  logout: "M9 4H5v16h4M16 8l4 4-4 4M20 12H9",
  eye: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  screen: "M3 5h18v12H3zM8 21h8M12 17v4",
  print: "M7 9V3h10v6M7 17H4v-7h16v7h-3M7 14h10v7H7z",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
  shield: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
  flask: "M9 3h6M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3M7.5 15h9",
  tag: "M4 4h8l8 8-8 8-8-8zM8.5 8.5h.01",
  link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
};

export function icon(name, size = 18, props) {
  const d = PATHS[name];
  if (!d) throw new Error(`нет значка «${name}»`);
  return s("svg", { viewBox: "0 0 24 24", width: size, height: size, fill: "none", stroke: "currentColor", "stroke-width": 1.8, "stroke-linecap": "square", "stroke-linejoin": "miter", "aria-hidden": "true", class: "ic", ...props }, s("path", { d }));
}

export const ICON_NAMES = Object.keys(PATHS);

// ───────── форматирование ─────────
const pad = (n) => String(n).padStart(2, "0");
export const parseTs = (v) => (v ? new Date(String(v).replace(" ", "T")) : null);
export function fmtTime(v) { const d = parseTs(v); return d && !isNaN(d) ? `${pad(d.getHours())}:${pad(d.getMinutes())}` : "—"; }
export function fmtClock(v) { const d = parseTs(v); return d && !isNaN(d) ? `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}` : "—"; }
const MONTHS = ["янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
export function fmtDate(v) { const d = parseTs(v); return d && !isNaN(d) ? `${d.getDate()} ${MONTHS[d.getMonth()]}` : "—"; }
export function fmtDateTime(v) { const d = parseTs(v); return d && !isNaN(d) ? `${fmtDate(v)}, ${fmtTime(v)}` : "—"; }
export function ago(v, now = new Date()) {
  const d = parseTs(v); if (!d || isNaN(d)) return "";
  const sec = Math.round((now - d) / 1000);
  if (sec < -60) return "";
  if (sec < 60) return "только что";
  if (sec < 3600) return `${Math.round(sec / 60)} мин назад`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} ч ${Math.round((sec % 3600) / 60)} мин назад`;
  return `${Math.floor(sec / 86400)} дн назад`;
}
export const pct = (v, digits = 0) => (v === null || v === undefined || isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`);
export const num = (v, digits = 2) => (v === null || v === undefined || isNaN(v) ? "—" : Number(v).toFixed(digits).replace(".", ","));
export function dur(sec) {
  if (sec === null || sec === undefined || isNaN(sec)) return "—";
  if (sec < 90) return `${Math.round(sec)} с`;
  if (sec < 5400) return `${Math.round(sec / 60)} мин`;
  return `${(sec / 3600).toFixed(1).replace(".", ",")} ч`;
}
export function clock(sec) { const t = Math.max(0, Math.floor(sec || 0)); return `${pad(Math.floor(t / 60))}:${pad(t % 60)}`; }
export const initials = (name) => String(name || "?").split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";
export const plural = (n, one, few, many) => { const a = Math.abs(n) % 100, b = a % 10; return a > 10 && a < 20 ? many : b > 1 && b < 5 ? few : b === 1 ? one : many; };
export const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
