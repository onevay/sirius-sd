// Обзор: что ждёт решения прямо сейчас, итоги дня и недели, камеры с очередью, состояние системы.
import { h, icon, fmtDate, pct, dur, plural, debounce, clear, put } from "../dom.js";
import { api, errText } from "../api.js";
import { bus } from "../bus.js";
import { session } from "../session.js";
import { href } from "../router.js";
import * as store from "../store.js";
import { button, card, chip, empty, errorBox, pageHead, skeleton } from "../ui.js";
import { stackedBars, hbars } from "../charts.js";
import { alertRow, decide } from "../alertviews.js";

const WEEKDAY = ["вс", "пн", "вт", "ср", "чт", "пт", "сб"];

function kpiBand(st) {
  const today = st.today || {};
  const todayTotal = Object.values(today).reduce((a, b) => a + b, 0);
  return h("div", { class: "band" },
    h("div", { class: "kpi" }, h("div", { class: "kpi__l" }, icon("bell", 16), "Ждут решения"), h("div", { class: "kpi__v kpi__v--xl" }, st.new), h("div", { class: "kpi__n" }, st.new ? `${plural(st.new, "тревога", "тревоги", "тревог")} в очереди` : "очередь пуста")),
    h("div", { class: "kpis", style: { flex: "1 1 480px" } },
      kpi("Сегодня", todayTotal, "всего тревог"),
      kpi("Подтверждено", st.confirmed, "за всё время"),
      kpi("Ложных", st.false, "за всё время"),
      kpi("Точность", st.precision == null ? "—" : pct(st.precision), "подтверждённые из решённых"),
      kpi("Реакция", st.median_review_sec == null ? "—" : dur(st.median_review_sec), "медиана от тревоги до решения")));
}
const kpi = (label, value, note) => card({ class: "card--tight" }, h("div", { class: "kpi" }, h("div", { class: "kpi__l" }, label), h("div", { class: "kpi__v", style: { fontSize: "30px" } }, value), h("div", { class: "kpi__n" }, note)));

export default async function overview(root, ctx) {
  const canAct = session.can("operator");
  const content = h("div", { class: "stack stack--lg" }, skeleton(2, 120));
  root.append(pageHead("Обзор", `Сегодня ${fmtDate(new Date().toISOString())}`,
    canAct ? button({ label: "Проверять очередь", ic: "eye", kind: "primary", href: href("/review") }) : null,
    button({ label: "Разбор видео", ic: "video", href: href("/analyze") })), content);

  async function load() {
    const [st, queue, week, cams] = await Promise.allSettled([
      store.stats(true),
      api.get("/alerts", { status: "new", limit: 30 }),
      api.get("/reports/summary", { days: 7 }),
      store.cameras(true),
    ]);
    if (!ctx.active()) return;
    const parts = [];
    parts.push(st.status === "fulfilled" ? kpiBand(st.value) : errorBox(`Показатели недоступны: ${errText(st.reason)}`, load));

    const idx = cams.status === "fulfilled" ? new Map(cams.value.map((m) => [m.camera.camera_id, m])) : new Map();
    const left = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Нужно решить"), button({ label: "Все тревоги", ic: "arrow", kind: "ghost", size: "sm", href: href("/alerts", { status: "new" }) })));
    if (queue.status === "fulfilled") {
      const items = [...queue.value.items].sort((a, b) => b.confidence - a.confidence).slice(0, 5);
      if (!items.length) left.append(empty("Всё разобрано", "Новые тревоги появятся здесь сами."));
      else put(left, h("div", { class: "stack" }, ...items.map((a, i) => alertRow(a, { idx, hot: i === 0, quick: canAct, onQuick: async (al, status) => { const done = await decide(al.id, status, { prev: { [al.id]: al.status } }); if (done) load(); } }))),
        queue.value.total > items.length ? h("p", { class: "small muted", style: { marginTop: "12px" } }, `Показаны самые уверенные ${items.length} из ${queue.value.total}.`) : null);
    } else left.append(errorBox(errText(queue.reason), load));

    const right = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Неделя"), chip("решено / ждёт", "line")));
    if (week.status === "fulfilled") {
      const days = week.value.series.map((d) => ({ label: WEEKDAY[new Date(`${d.t}T12:00:00`).getDay()], done: d.confirmed + d.false + d.unsure, pending: d.new }));
      right.append(stackedBars(days, { label: "Тревоги по дням: решённые и ожидающие" }),
        h("div", { class: "legend", style: { marginTop: "10px" } }, h("span", null, h("i", { style: { background: "var(--ink)" } }), "решены"), h("span", null, h("i", { class: "hatch", style: { border: "1px solid var(--ink)" } }), "ждут")),
        h("p", { class: "small muted", style: { marginTop: "10px" } }, `За ${week.value.days} дн.: ${week.value.totals.total}, подтверждено ${week.value.totals.confirmed}.`));
    } else right.append(errorBox(errText(week.reason), load));

    const camCard = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Камеры с очередью")));
    if (cams.status === "fulfilled") {
      const rows = cams.value.filter((m) => m.pending > 0).sort((a, b) => b.pending - a.pending).slice(0, 6);
      camCard.append(rows.length ? hbars(rows.map((m) => ({ label: store.camName(m.camera), value: m.pending })), { cls: "bar--alarm" }) : empty("Очередей нет", cams.value.length ? "На камерах нет ожидающих тревог." : "Камеры ещё не заведены."));
    } else camCard.append(errorBox(errText(cams.reason), load));

    const health = session.health || {};
    const sys = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Система")),
      h("div", { class: "stack stack--sm small" },
        row("Сервер", `версия ${health.version || "—"}`, health.ok),
        row("База данных", health.db ? (health.schema_current ? "схема актуальна" : "нужна миграция") : "недоступна", health.db && health.schema_current),
        row("Вход", health.auth_required ? "обязателен" : "отключён (локальный режим)", true),
        st.status === "fulfilled" ? row("Камеры", `${st.value.cameras} в справочнике`, true) : null));

    const wrap = h("div", { class: "stack stack--lg" }, parts[0], h("div", { class: "cols" }, h("div", { class: "f2" }, left), h("div", { class: "f1 stack" }, right, camCard, sys)));
    clear(content).append(wrap);
  }
  const row = (k, v, ok) => h("div", { class: "row row--nowrap" }, h("span", { class: `dot ${ok ? "dot--ok" : "dot--alarm"}` }), h("span", { class: "grow" }, k), h("span", { class: "muted" }, v));

  await load();
  const reload = debounce(() => ctx.active() && load(), 600);
  for (const ev of ["live:alert", "live:review", "alerts:changed"]) ctx.cleanup(bus.on(ev, reload));
}
