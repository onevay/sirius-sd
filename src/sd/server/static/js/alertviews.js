// Общее для экранов тревог: строка и карточка, подписи, запись решения с отменой, запрос причины «ложной», границы периодов.
import { h, icon, fmtDate, fmtTime, ago, dur, parseTs } from "./dom.js";
import { api, errText } from "./api.js";
import { camName, reasons } from "./store.js";
import { chip, statusChip, ring, thumb, button, toast, openModal, REASON_RU, STATUS } from "./ui.js";
import { session } from "./session.js";
import { go } from "./router.js";
import { bus } from "./bus.js";

export const alertCam = (a, idx) => { const m = idx && idx.get(a.camera_id); return m ? camName(m.camera) : camName(a); };
export const alertTime = (a) => a.t_abs || a.t_raised || a.created_at;
export const whenText = (a) => { const t = alertTime(a); return t ? `${fmtDate(t)}, ${fmtTime(t)}` : "—"; };
export const lengthText = (a) => dur(Math.max(0, (a.end_sec || 0) - (a.start_sec || 0)));

const pad = (n) => String(n).padStart(2, "0");
export const isoLocal = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
/** Начало периода для фильтров: today | 7 | 30 | all. */
export function sinceFor(period) {
  const d = new Date();
  if (period === "today") { d.setHours(0, 0, 0, 0); return isoLocal(d); }
  if (period === "7" || period === "30") { d.setDate(d.getDate() - Number(period)); return isoLocal(d); }
  return undefined;
}

export const RULE_RU = { two_cycles: "Два цикла подряд", "cycle+object": "Цикл и предмет у рта", both: "Два цикла и предмет у рта" };
export const ruleText = (r) => RULE_RU[r] || r || "—";

const activate = (fn) => (e) => { if (e.target.closest("button,a,input,select,textarea") && e.target !== e.currentTarget) return; if (e.type === "click" || e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(e); } };

/** Строка списка. quick: показывать кнопки «подтвердить / ложная» (только оператору). */
export function alertRow(a, { idx, hot = false, quick = false, onQuick, onOpen } = {}) {
  const open = onOpen || (() => go(`/review/${a.id}`));
  const row = h("div", { class: ["alertrow", hot && "alertrow--hot"], role: "link", tabindex: "0", "aria-label": `Тревога ${a.id}, ${alertCam(a, idx)}` },
    thumb(a, 96, 72),
    h("div", { class: "grow" }, h("b", { class: "ellipsis", style: { display: "block" } }, alertCam(a, idx)),
      h("div", { class: "small muted" }, `${whenText(a)} · ${lengthText(a)}`), h("div", { class: "small muted" }, ago(alertTime(a)))),
    ring(a.confidence, { hot }),
    statusChip(a.status),
    quick && a.status === "new" ? h("div", { class: "alertrow__quick" },
      button({ ic: "check", kind: "accent", size: "sm", title: "Подтвердить", onClick: (e) => { e.stopPropagation(); onQuick && onQuick(a, "confirmed"); } }),
      button({ ic: "x", size: "sm", title: "Ложная", onClick: (e) => { e.stopPropagation(); onQuick && onQuick(a, "false"); } })) : null);
  row.addEventListener("click", activate(open));
  row.addEventListener("keydown", activate(open));
  return row;
}

/** Карточка доски. select: {on, toggle} — флажок выбора для пакетных решений. */
export function alertCard(a, { idx, select, onOpen } = {}) {
  const open = onOpen || (() => go(`/review/${a.id}`));
  const closed = a.status === "confirmed" || a.status === "false";
  const card = h("div", { class: ["acard", closed && "acard--closed", select && select.on && "acard--sel"], role: "link", tabindex: "0", "aria-label": `Тревога ${a.id}, ${alertCam(a, idx)}` },
    thumb(a, "100%", "130px"),
    h("div", { class: "row row--nowrap" }, h("div", { class: "grow" }, h("b", { class: "ellipsis", style: { display: "block" } }, alertCam(a, idx)), h("div", { class: "small muted" }, `${whenText(a)} · ${lengthText(a)}`)), ring(a.confidence, { size: 42 })),
    h("div", { class: "row" }, statusChip(a.status), a.reviewer ? h("span", { class: "small muted" }, a.reviewer) : null, a.demo ? chip("демо", "line") : null));
  if (select) {
    const pick = h("button", { class: "acard__pick", type: "button", role: "checkbox", "aria-checked": String(!!select.on), "aria-label": "Выбрать тревогу", onClick: (e) => { e.stopPropagation(); select.toggle(); } }, icon("check", 14));
    card.append(pick);
  }
  card.addEventListener("click", activate(open));
  card.addEventListener("keydown", activate(open));
  return card;
}

/** Выбор причины для «ложной». Возвращает ключ причины, "" (без причины) или null (отмена). */
export async function askReason(count = 1) {
  let map;
  try { map = await reasons(); } catch { map = REASON_RU; }
  return new Promise((resolve) => {
    let sel = null, done = false;
    const fin = (v) => { if (!done) { done = true; close(); resolve(v); } };
    const wrap = h("div", { class: "reasons", role: "radiogroup", "aria-label": "Причина" });
    const save = button({ label: "Сохранить", kind: "primary", disabled: true, onClick: () => fin(sel) });
    for (const [k, ru] of Object.entries(map)) {
      const b = h("button", { class: "reason", type: "button", role: "radio", "aria-checked": "false", onClick: () => { sel = k; wrap.querySelectorAll(".reason").forEach((x) => { x.classList.toggle("is-on", x === b); x.setAttribute("aria-checked", String(x === b)); }); save.disabled = false; } }, REASON_RU[k] || ru);
      wrap.append(b);
    }
    const box = h("div", { class: "modal" }, h("h2", null, count > 1 ? `Ложные тревоги: ${count}` : "Почему тревога ложная?"), h("p", { class: "muted" }, "Причина идёт в обучение модели как трудный пример. Можно пропустить."), wrap,
      h("div", { class: "modal__foot" }, button({ label: "Отмена", onClick: () => fin(null) }), button({ label: "Без причины", onClick: () => fin("") }), save));
    const close = openModal(box, { onClose: () => fin(null) });
  });
}

const ONE = { confirmed: "Тревога подтверждена", false: "Тревога отмечена ложной", unsure: "Тревога отмечена «не уверен»", new: "Тревога возвращена в очередь" };
const MANY = { confirmed: "подтверждены", false: "отмечены ложными", unsure: "отмечены «не уверен»", new: "возвращены в очередь" };

/** Записывает решение для одной или нескольких тревог. prev — прежние статусы {id: status} для отмены. Возвращает список применённых id или null при ошибке. */
export async function decide(ids, status, { reason, note = "", prev = {}, undo = true, quiet = false } = {}) {
  const list = [].concat(ids);
  if (!list.length) return [];
  if (!session.can("operator")) { toast("Решения доступны оператору и выше", { kind: "alarm" }); return null; }
  try {
    const r = await api.post("/alerts/review", { ids: list, status, note, reason: reason || null });
    if (!quiet) {
      const n = r.done.length;
      toast(`${n === 1 ? ONE[status] : `Тревог: ${n} — ${MANY[status]}`}${r.missing.length ? ` (не найдено: ${r.missing.length})` : ""}`, {
        kind: status === "confirmed" ? "ok" : "",
        action: undo && n ? { label: "Отменить", run: () => revert(r.done, prev) } : undefined,
      });
    }
    return r.done;
  } catch (e) {
    toast(errText(e), { kind: "alarm" });
    return null;
  }
}

async function revert(ids, prev) {
  const groups = {};
  for (const id of ids) (groups[prev[id] || "new"] ||= []).push(id);
  try {
    for (const [st, part] of Object.entries(groups)) await api.post("/alerts/review", { ids: part, status: st, note: "" });
    toast("Решение отменено");
    bus.emit("alerts:changed", ids);
  } catch (e) { toast(errText(e), { kind: "alarm" }); }
}

export { STATUS, parseTs };
