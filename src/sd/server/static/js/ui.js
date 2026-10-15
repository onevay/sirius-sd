// Общие компоненты: кнопки, чипы статусов, кольца уверенности, превью, уведомления, диалоги, вкладки. Новый компонент — сюда, стили — в css/app.css.
import { h, s, icon, clear } from "./dom.js";

export const STATUS = {
  new: { label: "Новая", cls: "alarm", icon: "bell" },
  confirmed: { label: "Подтверждена", cls: "ok", icon: "check" },
  false: { label: "Ложная", cls: "mute", icon: "x" },
  unsure: { label: "Не уверен", cls: "info", icon: "help" },
};
export const STATUS_ORDER = ["new", "unsure", "confirmed", "false"];
export const REASON_RU = { drink: "Питьё", phone: "Телефон", eat: "Еда", touch_face: "Касание лица", other_gesture: "Другой жест", no_person: "Нет человека", duplicate: "Дубль тревоги", other: "Другое" };

export function chip(text, cls = "", ic = null, extra = "") {
  return h("span", { class: ["chip", cls && `chip--${cls}`, extra] }, ic ? icon(ic, 13) : null, text);
}
export const statusChip = (st, ic = true) => { const m = STATUS[st] || { label: st, cls: "" }; return chip(m.label, m.cls, ic ? m.icon : null); };

export function button({ label, ic, kind = "", size = "", onClick, href, kbd, disabled, title, block, type = "button", aria } = {}) {
  const cls = ["btn", kind && `btn--${kind}`, size && `btn--${size}`, block && "btn--block", !label && "btn--icon"];
  const kids = [ic ? icon(ic, size === "sm" ? 15 : 17) : null, label, kbd ? h("span", { class: "kbd" }, kbd) : null];
  const props = { class: cls, title, "aria-label": aria || (label ? null : title) };
  if (href) return h("a", { ...props, href, "aria-disabled": disabled ? "true" : null }, ...kids);
  return h("button", { ...props, type, disabled, onClick }, ...kids);
}

export const card = (props, ...kids) => h("div", { ...props, class: ["card", props && props.class] }, ...kids);

export function pageHead(title, sub, ...actions) {
  return h("div", { class: "pagehead" }, h("div", { class: "pagehead__t" }, h("h1", null, title), sub ? h("p", null, sub) : null), actions.length ? h("div", { class: "row" }, ...actions) : null);
}

/** Кольцо уверенности: всегда с числом внутри. */
export function ring(p, { size = 46, hot = false } = {}) {
  const v = Math.max(0, Math.min(1, Number(p) || 0));
  const label = `${Math.round(v * 100)}%`;
  return s("svg", { class: "ring", viewBox: "0 0 44 44", width: size, height: size, role: "img", "aria-label": `Уверенность ${label}` },
    s("circle", { cx: 22, cy: 22, r: 18, fill: "none", stroke: hot ? "#2A382E" : "#D3D9CE", "stroke-width": 4 }),
    s("circle", { cx: 22, cy: 22, r: 18, fill: "none", stroke: "#D93A1B", "stroke-width": 4, "stroke-linecap": "butt", pathLength: 100, "stroke-dasharray": `${Math.round(v * 100)} 100`, transform: "rotate(-90 22 22)" }),
    s("text", { x: 22, y: 26, "text-anchor": "middle", "font-size": 12, "font-weight": 600, fill: hot ? "#FFFFFF" : "#0E1610", "font-family": "Geologica, sans-serif" }, label));
}

/** Схематичный кадр-заглушка (если у тревоги нет миниатюры): плоские формы без деталей реальной сцены. */
export function sceneSvg(seed = 0, box = true) {
  const x = 14 + ((Math.abs(seed) * 37) % 54);
  return s("svg", { viewBox: "0 0 96 72", preserveAspectRatio: "xMidYMid slice", role: "img", "aria-label": "Кадр недоступен" },
    s("rect", { width: 96, height: 72, fill: "#34413A" }), s("rect", { y: 46, width: 96, height: 26, fill: "#27322C" }),
    s("g", { transform: `translate(${x} 0)` }, s("circle", { cx: 20, cy: 26, r: 6, fill: "#9AA79F" }), s("rect", { x: 12, y: 33, width: 16, height: 30, rx: 6, fill: "#7B887F" }),
      box ? s("rect", { x: 9, y: 17, width: 22, height: 48, fill: "none", stroke: "#D93A1B", "stroke-width": 1.8 }) : null));
}

export function thumb(alert, w = 96, hgt = 72) {
  const box = h("div", { class: "thumb", style: { width: typeof w === "number" ? `${w}px` : w, height: typeof hgt === "number" ? `${hgt}px` : hgt } });
  if (alert.has_thumb) {
    const img = h("img", { src: `/api/v1/alerts/${alert.id}/thumb`, alt: `Кадр тревоги ${alert.id}`, loading: "lazy" });
    img.addEventListener("error", () => { clear(box); box.append(sceneSvg(alert.id)); });
    box.append(img);
  } else box.append(sceneSvg(alert.id));
  return box;
}

// ───────── уведомления ─────────
let toasts;
export function toast(text, { kind = "", action, ms = 6000 } = {}) {
  if (!toasts || !toasts.isConnected) { toasts = h("div", { class: "toasts", role: "status", "aria-live": "polite" }); document.body.append(toasts); }
  const el = h("div", { class: ["toast", kind && `toast--${kind}`] }, h("div", { class: "grow" }, text));
  const close = () => el.remove();
  if (action) el.append(h("button", { class: "btn", onClick: () => { close(); action.run(); } }, action.label));
  toasts.append(el);
  while (toasts.children.length > 4) toasts.firstChild.remove();
  if (ms) setTimeout(close, ms);
  return close;
}

// ───────── диалоги ─────────
const openDialogs = new Set();
/** Закрывает все открытые окна (при смене раздела окно не должно оставаться поверх чужого экрана). */
export function closeAllModals() { [...openDialogs].forEach((close) => close()); }

export function openModal(content, { wide = false, onClose } = {}) {
  const prevFocus = document.activeElement;
  const overlay = h("div", { class: "overlay", role: "dialog", "aria-modal": "true" });
  const close = () => {
    if (!openDialogs.delete(close)) return;
    overlay.remove();
    document.removeEventListener("keydown", onKey, true);
    if (prevFocus && prevFocus.isConnected && prevFocus.focus) prevFocus.focus();
    onClose && onClose();
  };
  openDialogs.add(close);
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
  document.addEventListener("keydown", onKey, true);
  if (wide) content.style.width = "min(820px,100%)";
  overlay.append(content);
  document.body.append(overlay);
  const first = overlay.querySelector("input,select,textarea,button");
  if (first) first.focus();
  return close;
}

export function confirmDialog(text, { ok = "Да", danger = false, title = "Подтвердите действие" } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const fin = (v) => { if (!done) { done = true; close(); resolve(v); } };
    const close = openModal(h("div", { class: "modal" }, h("h2", null, title), h("p", null, text),
      h("div", { class: "modal__foot" }, button({ label: "Отмена", onClick: () => fin(false) }), button({ label: ok, kind: danger ? "danger" : "primary", onClick: () => fin(true) }))), { onClose: () => fin(false) });
  });
}

/** Форма в окне: fields = [{name,label,type,value,options:[[value,label]],required,help,placeholder}]; onSubmit(values) может бросить ошибку — она покажется в окне. */
export function formModal({ title, fields, submit = "Сохранить", onSubmit, intro }) {
  const err = h("div", { class: "banner banner--alarm", hidden: true, role: "alert" });
  const inputs = {};
  const rows = fields.map((f) => {
    let el;
    if (f.type === "select") el = h("select", { class: "input", name: f.name }, ...f.options.map(([v, l]) => h("option", { value: v, selected: v === f.value }, l)));
    else if (f.type === "textarea") el = h("textarea", { class: "input", name: f.name, rows: 3, placeholder: f.placeholder }, f.value || "");
    else if (f.type === "checkbox") { el = h("input", { type: "checkbox", name: f.name, checked: !!f.value }); }
    else el = h("input", { class: "input", name: f.name, type: f.type || "text", value: f.value ?? "", placeholder: f.placeholder, required: f.required, autocomplete: f.type === "password" ? "new-password" : "off" });
    inputs[f.name] = el;
    if (f.type === "checkbox") return h("label", { class: "check" }, el, f.label);
    return h("label", { class: "field" }, f.label, el, f.help ? h("span", { class: "small muted", style: { fontWeight: 400 } }, f.help) : null);
  });
  return new Promise((resolve) => {
    let settled = false;
    const finish = (v) => { if (!settled) { settled = true; resolve(v); } };          // первое значение побеждает: закрытие окна не должно затирать результат
    const submitBtn = button({ label: submit, kind: "primary", type: "submit" });
    const form = h("form", { class: "modal", novalidate: true }, h("h2", null, title), intro ? h("p", { class: "muted" }, intro) : null, ...rows, err,
      h("div", { class: "modal__foot" }, button({ label: "Отмена", onClick: () => { finish(null); close(); } }), submitBtn));
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const values = {};
      for (const f of fields) {
        const el = inputs[f.name];
        values[f.name] = f.type === "checkbox" ? el.checked : el.value;
        if (f.required && !String(values[f.name]).trim()) { err.hidden = false; err.textContent = `Заполните поле «${f.label}»`; el.focus(); return; }
      }
      submitBtn.disabled = true;
      try { const r = await onSubmit(values); finish(r === undefined ? values : r); close(); }
      catch (ex) { err.hidden = false; err.textContent = ex && ex.message ? ex.message : String(ex); submitBtn.disabled = false; }
    });
    const close = openModal(form, { onClose: () => finish(null) });
  });
}

// ───────── состояния ─────────
export const empty = (title, text, ...actions) => h("div", { class: "empty" }, h("b", null, title), text ? h("span", null, text) : null, actions.length ? h("div", { class: "row", style: { justifyContent: "center" } }, ...actions) : null);
export const errorBox = (message, retry) => h("div", { class: "banner banner--alarm", role: "alert" }, icon("warn", 18), h("div", { class: "grow" }, message), retry ? button({ label: "Повторить", size: "sm", onClick: retry }) : null);
export const skeleton = (n = 3, height = 64) => h("div", { class: "stack" }, ...Array.from({ length: n }, () => h("div", { class: "skel", style: { height: `${height}px` } })));

export function seg(options, value, onChange, { label } = {}) {
  const wrap = h("div", { class: "seg", role: "group", "aria-label": label || null });
  const draw = (v) => { clear(wrap); options.forEach(([val, text]) => wrap.append(h("button", { class: ["seg__b", val === v && "is-active"], type: "button", "aria-pressed": String(val === v), onClick: () => { if (val !== v) { onChange(val); draw(val); } } }, text))); };
  draw(value);
  return wrap;
}

export function tabs(options, value, onChange) {
  const wrap = h("div", { class: "tabs", role: "tablist" });
  const draw = (v) => { clear(wrap); options.forEach(([val, text, badge]) => wrap.append(h("button", { class: ["tabs__t", val === v && "is-active"], type: "button", role: "tab", "aria-selected": String(val === v), onClick: () => { if (val !== v) { onChange(val); draw(val); } } }, text, badge ? h("span", { class: "badge badge--ink" }, badge) : null))); };
  draw(value);
  return wrap;
}

export function stubBox(title, items = [], { api } = {}) {
  return h("div", { class: "stubbox" }, h("div", { class: "stubbox__t" }, icon("layers", 16), `${title} — следующий этап`),
    items.length ? h("ul", null, ...items.map((i) => h("li", null, i))) : null,
    api ? h("div", { class: "small muted" }, "Данные: ", h("span", { class: "mono" }, api)) : null);
}

export const keyHint = (k, text) => h("span", { class: "small muted row row--nowrap", style: { gap: "6px" } }, h("span", { class: "kbd" }, k), text);
