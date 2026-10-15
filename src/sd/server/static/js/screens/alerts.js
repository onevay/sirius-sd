// Тревоги: доска по статусам или список с фильтрами; пакетные решения с отменой; обновляется по живому потоку.
import { h, clear, debounce, plural } from "../dom.js";
import { api, errText } from "../api.js";
import { bus } from "../bus.js";
import { session } from "../session.js";
import * as store from "../store.js";
import { button, empty, errorBox, pageHead, seg, skeleton, toast, STATUS, STATUS_ORDER } from "../ui.js";
import { saveText } from "../files.js";
import { alertCard, alertRow, askReason, decide, sinceFor } from "../alertviews.js";

const PAGE = 20, LIST_PAGE = 40;
const DOT = { ok: "dot--ok", alarm: "dot--alarm", info: "dot--info" };
const COLS = { new: "Новые", unsure: "Не уверен", confirmed: "Подтверждены", false: "Ложные" };
const CONF = [["", "Любая уверенность"], ["0.5", "от 50 %"], ["0.7", "от 70 %"], ["0.85", "от 85 %"]];
const PERIOD = [["today", "Сегодня"], ["7", "7 дней"], ["30", "30 дней"], ["all", "Всё время"]];

const pref = {
  get(k, d) { try { return localStorage.getItem(`sd_${k}`) || d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`sd_${k}`, v); } catch { /* без сохранения */ } },
};

export default async function alerts(root, ctx) {
  const q = ctx.query;
  const f = { view: q.view || pref.get("alerts_view", "board"), camera: q.camera || "", min: q.min || "", period: q.period || "7", status: q.status || "" };
  const canAct = session.can("operator");
  const selected = new Set();
  let idx = new Map(), cams = [];
  try { cams = await store.cameras(); idx = new Map(cams.map((m) => [m.camera.camera_id, m])); } catch { /* камеры нужны только для подписей и фильтра */ }

  const body = h("div", null, skeleton(2, 160));
  const bulk = h("div", { class: "bulk no-print", role: "toolbar", "aria-label": "Решение для выбранных", hidden: true });
  const syncUrl = () => { ctx.setQuery({ view: f.view, camera: f.camera, min: f.min, period: f.period, status: f.status }); };

  // ───────── фильтры ─────────
  const select = (value, options, on, label) => {
    const el = h("select", { class: "input input--sm", style: { width: "auto", minWidth: "170px" }, "aria-label": label, onChange: (e) => on(e.target.value) }, ...options.map(([v, l]) => h("option", { value: v, selected: v === value }, l)));
    return el;
  };
  const filters = h("div", { class: "row no-print", style: { marginBottom: "18px" } },
    select(f.camera, [["", "Все камеры"], ...cams.map((m) => [m.camera.camera_id, store.camName(m.camera)])], (v) => { f.camera = v; changed(); }, "Камера"),
    select(f.min, CONF, (v) => { f.min = v; changed(); }, "Уверенность"),
    seg(PERIOD, f.period, (v) => { f.period = v; changed(); }, { label: "Период" }));

  const query = (extra) => ({ camera: f.camera || undefined, min_conf: f.min || undefined, since: sinceFor(f.period), ...extra });
  const changed = () => { syncUrl(); selected.clear(); paintBulk(); load(); };

  // ───────── пакетная панель ─────────
  function paintBulk() {
    bulk.hidden = !selected.size || !canAct;
    clear(bulk);
    if (bulk.hidden) return;
    const ids = [...selected];
    bulk.append(h("b", null, `Выбрано: ${ids.length}`),
      button({ label: "Подтвердить", ic: "check", kind: "accent", size: "sm", onClick: () => act(ids, "confirmed") }),
      button({ label: "Ложные", ic: "x", size: "sm", onClick: () => act(ids, "false") }),
      button({ label: "Не уверен", ic: "help", size: "sm", onClick: () => act(ids, "unsure") }),
      button({ label: "Снять выбор", kind: "ghost", size: "sm", onClick: () => { selected.clear(); paintBulk(); repaintSelection(); } }));
  }
  const repaintSelection = () => body.querySelectorAll(".acard").forEach((c) => {
    const on = selected.has(Number(c.dataset.id));
    c.classList.toggle("acard--sel", on);
    const pick = c.querySelector(".acard__pick");
    if (pick) pick.setAttribute("aria-checked", String(on));
  });
  async function act(ids, status) {
    let reason;
    if (status === "false") { reason = await askReason(ids.length); if (reason === null) return; }
    const prev = Object.fromEntries(ids.map((id) => [id, data.byId.get(id) ? data.byId.get(id).status : "new"]));
    const done = await decide(ids, status, { reason, prev });
    if (done) { selected.clear(); paintBulk(); load(true); }
  }

  // ───────── данные ─────────
  const data = { byId: new Map(), cols: {}, list: { items: [], total: 0 } };
  const remember = (items) => items.forEach((a) => data.byId.set(a.id, a));

  async function fetchCol(status, offset = 0) {
    const r = await api.get("/alerts", query({ status, limit: PAGE, offset }));
    remember(r.items);
    return r;
  }

  async function load(silent = false) {
    if (!silent) clear(body).append(skeleton(2, 160));
    try {
      if (f.view === "board") {
        const res = await Promise.all(STATUS_ORDER.map((s) => fetchCol(s)));
        STATUS_ORDER.forEach((s, i) => { data.cols[s] = { items: res[i].items, total: res[i].total, busy: false }; });
      } else {
        const r = await api.get("/alerts", query({ status: f.status || undefined, limit: LIST_PAGE }));
        remember(r.items);
        data.list = { items: r.items, total: r.total };
      }
      if (!ctx.active()) return;
      draw();
    } catch (e) {
      if (ctx.active()) clear(body).append(errorBox(errText(e), load));
    }
  }

  // ───────── отрисовка ─────────
  function card1(a) {
    const sel = canAct ? { on: selected.has(a.id), toggle: () => { selected.has(a.id) ? selected.delete(a.id) : selected.add(a.id); paintBulk(); repaintSelection(); } } : null;
    const c = alertCard(a, { idx, select: sel });
    c.dataset.id = a.id;
    return c;
  }

  function drawBoard() {
    return h("div", { class: "board" }, ...STATUS_ORDER.map((s) => {
      const col = data.cols[s];
      const list = h("div", { class: "stack" });
      const paint = () => {
        clear(list);
        if (!col.items.length) list.append(empty(s === "new" ? "Новых тревог нет" : "Пока пусто"));
        col.items.forEach((a) => list.append(card1(a)));
        if (col.items.length < col.total) list.append(button({ label: `Ещё ${Math.min(PAGE, col.total - col.items.length)}`, block: true, onClick: more }));
      };
      async function more(e) {
        if (col.busy) return;
        col.busy = true;
        const btn = e.currentTarget;
        btn.disabled = true;
        try { const r = await fetchCol(s, col.items.length); col.items.push(...r.items.filter((x) => !col.items.some((y) => y.id === x.id))); col.total = r.total; }
        catch (ex) { list.append(errorBox(errText(ex))); }
        col.busy = false;
        paint();
      }
      paint();
      return h("section", { class: "col", "aria-label": COLS[s] },
        h("div", { class: "col__head" }, h("span", { class: ["dot", DOT[STATUS[s].cls]] }), h("h2", null, COLS[s]), h("span", { class: "chip" }, col.total)), list);
    }));
  }

  function drawList() {
    const L = data.list;
    const status = h("div", { class: "row no-print", style: { marginBottom: "14px" } },
      seg([["", "Все"], ...STATUS_ORDER.map((s) => [s, COLS[s]])], f.status, (v) => { f.status = v; changed(); }, { label: "Статус" }), h("span", { class: "muted small" }, `${L.total} ${plural(L.total, "тревога", "тревоги", "тревог")}`));
    const rows = h("div", { class: "stack" });
    const paint = () => {
      clear(rows);
      if (!L.items.length) rows.append(empty("Ничего не найдено", "Измените фильтры или период."));
      L.items.forEach((a) => rows.append(alertRow(a, { idx, hot: a.status === "new" && a.confidence >= 0.85, quick: canAct, onQuick: async (al, st) => { let reason; if (st === "false") { reason = await askReason(1); if (reason === null) return; } const done = await decide(al.id, st, { reason, prev: { [al.id]: al.status } }); if (done) load(true); } })));
      if (L.items.length < L.total) rows.append(button({ label: `Показать ещё (${L.total - L.items.length})`, block: true, onClick: async (e) => { e.currentTarget.disabled = true; try { const r = await api.get("/alerts", query({ status: f.status || undefined, limit: LIST_PAGE, offset: L.items.length })); remember(r.items); L.items.push(...r.items.filter((x) => !L.items.some((y) => y.id === x.id))); L.total = r.total; } catch (ex) { rows.append(errorBox(errText(ex))); } paint(); } }));
    };
    paint();
    return h("div", null, status, rows);
  }

  function draw() {
    clear(body).append(f.view === "board" ? drawBoard() : drawList());
    repaintSelection();
  }

  const viewSwitch = seg([["board", "Доска"], ["list", "Список"]], f.view, (v) => { f.view = v; pref.set("alerts_view", v); changed(); }, { label: "Вид" });
  root.append(pageHead("Тревоги", "Решения оператора: подтвердить, отметить ложной или оставить на проверку.", viewSwitch,
    button({ ic: "download", label: "Журнал CSV", size: "sm", onClick: downloadCsv })), filters, body, bulk);

  async function downloadCsv(e) {
    const b = e.currentTarget;
    b.disabled = true;
    try { saveText("decisions.csv", await api.get("/journal.csv"), "text/csv;charset=utf-8"); }
    catch (ex) { toast(errText(ex), { kind: "alarm" }); }
    b.disabled = false;
  }

  await load();
  const reload = debounce(() => ctx.active() && load(true), 700);
  for (const ev of ["live:alert", "live:review", "alerts:changed"]) ctx.cleanup(bus.on(ev, reload));
}
