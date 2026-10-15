// Эксперименты (разработчик): журнал прогонов оценки с доверительным интервалом F1, подробности и заметки, парное сравнение, фоновые задачи. Остальное — заглушки с понятным путём доработки.
import { h, clear, icon, num, fmtDateTime, debounce } from "../dom.js";
import { api, errText } from "../api.js";
import { button, card, chip, empty, errorBox, formModal, openModal, pageHead, skeleton, stubBox, tabs, toast } from "../ui.js";
import { whisker } from "../charts.js";

const STATE_CLS = { "идёт": "info", "в очереди": "", "готово": "ok", "ошибка": "alarm", "остановлена": "", "прервана": "alarm" };
const kv = (k, v) => h("div", { class: "row row--nowrap", style: { justifyContent: "space-between", padding: "5px 0", borderTop: "1px solid var(--line-2)" } }, h("span", { class: "muted" }, k), h("span", { style: { textAlign: "right", overflowWrap: "anywhere" } }, v == null || v === "" ? "—" : String(v)));

export default async function experiments(root, ctx) {
  const st = { tab: ctx.query.tab || "runs", runs: [], filter: "", pinned: false, picked: new Set(), tasks: [] };
  const body = h("div");
  const timers = new Set();
  ctx.cleanup(() => timers.forEach(clearTimeout));
  const nav = tabs([["runs", "Журнал прогонов"], ["tasks", "Задачи"], ["oof", "Честная оценка"], ["classifier", "Классификатор"]], st.tab, (v) => { st.tab = v; ctx.setQuery({ tab: v }); show(); });
  root.append(pageHead("Эксперименты", "Что пробовали, что получилось и можно ли этому верить."), nav, body);

  function show() {
    timers.forEach(clearTimeout);
    timers.clear();
    clear(body);
    ({ runs: tabRuns, tasks: tabTasks, oof: tabOof, classifier: tabClassifier }[st.tab] || tabRuns)();
  }

  // ───────── журнал ─────────
  async function tabRuns() {
    body.append(skeleton(3, 56));
    try { st.runs = await api.get("/dev/runs", { limit: 300 }); } catch (e) { clear(body).append(errorBox(errText(e), tabRuns)); return; }
    if (!ctx.active()) return;
    drawRuns();
  }

  function drawRuns() {
    clear(body);
    const list = h("div", { class: "tablewrap" });
    const search = h("input", { class: "input input--sm", style: { maxWidth: "280px" }, placeholder: "Поиск по имени, тегу, заметке", value: st.filter, "aria-label": "Поиск", onInput: debounce((e) => { st.filter = e.target.value; paint(); }, 120) });
    const pin = h("label", { class: "check", style: { alignItems: "center" } }, h("input", { type: "checkbox", checked: st.pinned, onChange: (e) => { st.pinned = e.target.checked; paint(); } }), "Только закреплённые");
    const cmp = button({ label: "Сравнить выбранные", ic: "chart", size: "sm", onClick: (e) => compare(e.currentTarget), disabled: true });
    const bar = h("div", { class: "row", style: { marginBottom: "14px" } }, search, pin, h("span", { class: "grow" }), h("span", { class: "small muted" }, "Отметьте два прогона для сравнения"), cmp);
    const paint = () => {
      const q = st.filter.trim().toLowerCase();
      const rows = st.runs.filter((r) => (!st.pinned || r.pinned) && (!q || `${r.name} ${r.id} ${r.tag || ""} ${r.note || ""}`.toLowerCase().includes(q)));
      cmp.disabled = st.picked.size !== 2;
      clear(list);
      if (!rows.length) { list.append(empty(st.runs.length ? "Ничего не найдено" : "Прогонов ещё нет", st.runs.length ? "Измените поиск." : "Прогоны появляются после оценки профиля на размеченных видео (вкладка «Задачи»).")); return; }
      list.append(h("table", { class: "table" }, h("thead", null, h("tr", null, h("th", null, ""), ...["Прогон", "Клипов", "F1 и интервал", "P", "R", "FP/час", "Профиль"].map((c, i) => h("th", { class: i > 0 && i < 6 ? "num" : "" }, c)))),
        h("tbody", null, ...rows.map((r) => {
          const pick = h("input", { type: "checkbox", checked: st.picked.has(r.id), "aria-label": `Выбрать ${r.name}`, onClick: (e) => e.stopPropagation(), onChange: (e) => { e.target.checked ? st.picked.add(r.id) : st.picked.delete(r.id); cmp.disabled = st.picked.size !== 2; } });
          const tr = h("tr", { class: "is-click", tabindex: "0", onClick: () => detail(r.id), onKeydown: (e) => { if (e.key === "Enter") detail(r.id); } },
            h("td", null, pick),
            h("td", null, h("div", { class: "row row--nowrap" }, r.pinned ? icon("star", 15, { style: { color: "var(--accent)" } }) : null, h("b", null, r.name || r.id), r.tag ? chip(r.tag, "line") : null, leak(r) ? chip(`утечка: ${leak(r)}`, "alarm") : null),
              h("div", { class: "small muted" }, `${fmtDateTime(r.created)} · ${r.mode || "—"}`)),
            h("td", { class: "num" }, r.clips ?? "—"),
            h("td", { class: "num" }, h("div", { class: "row row--nowrap", style: { justifyContent: "flex-end" } }, h("b", null, num(r.f1)), whisker(r.ci_lo, r.ci_hi, r.f1, { width: 96 }))),
            h("td", { class: "num" }, num(r.precision)), h("td", { class: "num" }, num(r.recall)), h("td", { class: "num" }, r.fp_per_hour == null ? "—" : num(r.fp_per_hour, 1)),
            h("td", { class: "small" }, [r.pose, r.fps ? `${r.fps} к/с` : null, r.imgsz, r.cycle_model].filter(Boolean).join(" · ")));
          return tr;
        }))));
    };
    body.append(bar, list);
    paint();
  }
  const leak = (r) => (Number.isFinite(r.leak_clips) && r.leak_clips > 0 ? r.leak_clips : 0);

  async function detail(id) {
    let d;
    try { d = await api.get(`/dev/runs/${encodeURIComponent(id)}`); } catch (e) { toast(errText(e), { kind: "alarm" }); return; }
    const m = d.meta || {}, rep = d.report || {}, mt = rep.metrics || {}, ci = rep.ci || {}, pl = rep.plateau || {}, bd = rep.budget || {}, notes = d.notes || {};
    const box = h("div", { class: "modal", style: { width: "min(860px,100%)" } },
      h("div", { class: "row row--nowrap" }, h("div", { class: "grow" }, h("h2", null, m.name || id), h("span", { class: "small muted mono" }, id)), button({ ic: "x", kind: "ghost", title: "Закрыть", onClick: () => close() })),
      h("div", { class: "kpis" }, ...[["F1", num(mt.f1), ci.f1 ? `интервал ${num(ci.f1[0])}–${num(ci.f1[1])}` : "интервала нет"], ["Точность", num(mt.precision), `TP ${mt.tp} · FP ${mt.fp}`], ["Полнота", num(mt.recall), `FN ${mt.fn}`], ["Порог", num(mt.threshold), pl.lo != null ? `плато ${num(pl.lo)}–${num(pl.hi)}` : ""]]
        .map(([l, v, n]) => card({ class: "card--tight" }, h("div", { class: "kpi" }, h("div", { class: "kpi__l" }, l), h("div", { class: "kpi__v", style: { fontSize: "26px" } }, v), h("div", { class: "kpi__n" }, n))))),
      m.leak && m.leak.clips && m.leak.clips.length ? h("div", { class: "banner banner--alarm" }, icon("warn", 18), `Клипы обучения попали в оценку (${m.leak.clips.length}): F1 завышена, честная оценка — вкладка «Честная оценка».`) : null,
      h("div", { class: "cols" },
        h("div", { class: "f1" }, h("h3", { style: { marginBottom: "6px" } }, "Конфигурация"), ...Object.entries(m.describe || {}).map(([k, v]) => kv(k, v)), kv("Отпечаток", m.fingerprint), kv("Отпечаток разметки", m.gt_fingerprint), kv("Строк разметки", m.gt_rows), kv("Режим", m.mode)),
        h("div", { class: "f1" }, h("h3", { style: { marginBottom: "6px" } }, "Бюджет ошибок"), kv("Цель F1", bd.target), kv("Достигнута", bd.reached == null ? "—" : bd.reached ? "да" : "нет"), kv("Допустимо ошибок", bd.allowed), kv("Ошибок сейчас", bd.errors), kv("Медианная задержка", mt.median_latency == null ? "—" : `${num(mt.median_latency, 1)} с`), kv("FP на час негативов", mt.fp_per_hour == null ? "—" : num(mt.fp_per_hour, 1)))),
      m.clips && m.clips.length ? h("div", null, h("h3", { style: { marginBottom: "6px" } }, `Клипы (${m.clips.length})`), h("div", { class: "tablewrap", style: { maxHeight: "220px", overflow: "auto" } }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Клип", "Длина", "Циклов", "Людей", "Состояние"].map((c) => h("th", null, c)))),
        h("tbody", null, ...m.clips.map((c) => h("tr", null, h("td", { class: "ellipsis" }, c.clip_id), h("td", null, c.duration ? `${Math.round(c.duration)} с` : "—"), h("td", null, c.cycles ?? "—"), h("td", null, c.people ?? "—"), h("td", null, c.error ? chip("ошибка", "alarm") : chip("ок", "ok")))))))) : null,
      h("div", { class: "modal__foot" }, button({ label: "Заметка и тег", ic: "edit", onClick: async () => { const r = await editNotes(id, notes); if (r) { close(); drawRuns(); tabRuns(); } } }), button({ label: "Закрыть", kind: "primary", onClick: () => close() })));
    const close = openModal(box);
  }

  async function editNotes(id, notes) {
    return formModal({
      title: "Заметка к прогону", fields: [{ name: "tag", label: "Тег подхода", value: notes.tag || "", help: "Например: fps, imgsz, классификатор" }, { name: "note", label: "Заметка", type: "textarea", value: notes.note || "" }, { name: "pinned", label: "Закрепить в журнале", type: "checkbox", value: !!notes.pinned }],
      onSubmit: async (v) => { await api.patch(`/dev/runs/${encodeURIComponent(id)}/notes`, { tag: v.tag.trim(), note: v.note.trim(), pinned: v.pinned }); toast("Заметка сохранена", { kind: "ok" }); },
    });
  }

  async function compare(btn) {
    const [a, b] = [...st.picked];
    let r;
    btn.disabled = true;
    toast("Считаю сравнение по клипам, это занимает несколько секунд…", { ms: 4000 });
    try { r = await api.get(`/dev/runs/${encodeURIComponent(a)}/compare/${encodeURIComponent(b)}`, { n: 300 }); } catch (e) { toast(errText(e), { kind: "alarm" }); btn.disabled = false; return; }
    btn.disabled = st.picked.size !== 2;
    const ids = [a, b];
    const name = (id) => { const x = st.runs.find((y) => y.id === id); return x ? `${x.name} (${fmtDateTime(x.created)})` : id; };
    const box = h("div", { class: "modal", style: { width: "min(760px,100%)" } }, h("h2", null, "Парное сравнение"),
      h("p", { class: "muted small" }, "Разница F1 в процентных пунктах на общих клипах с 95 % интервалом (bootstrap по клипам). Знак «≈» — различие в пределах шума, улучшением его считать нельзя."),
      h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, h("th", null, "строка − столбец"), ...ids.map((i) => h("th", null, name(i))))),
        h("tbody", null, ...ids.map((row, i) => h("tr", null, h("td", null, h("b", null, name(row))), ...ids.map((col) => h("td", null, row === col ? "—" : String((r[i] || {})[col] ?? "")))))))),
      h("div", { class: "modal__foot" }, button({ label: "Закрыть", kind: "primary", onClick: () => close() })));
    const close = openModal(box);
  }

  // ───────── задачи ─────────
  async function tabTasks() {
    body.append(skeleton(3, 56));
    const load = async () => {
      try { st.tasks = await api.get("/dev/tasks", { limit: 30 }); } catch (e) { if (ctx.active() && st.tab === "tasks") clear(body).append(errorBox(errText(e), load)); return; }
      if (!ctx.active() || st.tab !== "tasks") return;
      drawTasks();
      if (st.tasks.some((t) => t.state === "идёт" || t.state === "в очереди")) timers.add(setTimeout(load, 3000));
    };
    await load();
  }

  function drawTasks() {
    clear(body);
    const head = h("div", { class: "row", style: { marginBottom: "14px" } }, h("span", { class: "grow muted" }, "Тяжёлые операции работают в фоне и не мешают интерфейсу."), button({ label: "Новая задача", ic: "plus", kind: "primary", onClick: newTask }));
    if (!st.tasks.length) { body.append(head, empty("Задач пока нет", "Запустите оценку профиля или обучение классификатора.")); return; }
    body.append(head, card(null, h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Задача", "Состояние", "Прогресс", "Время", ""].map((c) => h("th", null, c)))),
      h("tbody", null, ...st.tasks.map((t) => h("tr", { class: "is-click", onClick: () => taskLog(t.id) }, h("td", null, h("b", null, t.title || t.task), h("div", { class: "small muted mono" }, t.id)), h("td", null, chip(t.state, STATE_CLS[t.state] || "")),
        h("td", { style: { minWidth: "160px" } }, t.progress ? h("div", null, h("div", { class: "bar bar--accent" }, h("i", { style: { width: `${Math.round(t.progress.frac * 100)}%` } })), h("div", { class: "small muted ellipsis" }, t.progress.message)) : h("span", { class: "muted" }, "—")),
        h("td", { class: "small" }, t.seconds ? `${Math.round(t.seconds)} с` : "—"),
        h("td", null, t.state === "идёт" || t.state === "в очереди" ? button({ label: "Остановить", size: "sm", onClick: async (e) => { e.stopPropagation(); try { await api.post(`/dev/tasks/${t.id}/stop`); toast("Остановка запрошена"); tabTasks(); } catch (ex) { toast(errText(ex), { kind: "alarm" }); } } }) : null))))))));
  }

  async function taskLog(id) {
    let d;
    try { d = await api.get(`/dev/tasks/${id}`, { lines: 200 }); } catch (e) { toast(errText(e), { kind: "alarm" }); return; }
    const pre = h("pre", { class: "mono small", style: { background: "var(--dark)", color: "#D9E2DB", padding: "14px", borderRadius: "10px", maxHeight: "50vh", overflow: "auto", whiteSpace: "pre-wrap", margin: 0 } }, d.log || "Журнал пока пуст");
    const close = openModal(h("div", { class: "modal", style: { width: "min(820px,100%)" } }, h("h2", null, d.title || id), h("div", { class: "row" }, chip(d.state, STATE_CLS[d.state] || ""), h("span", { class: "small muted" }, `${Math.round(d.seconds || 0)} с`)), pre,
      h("div", { class: "modal__foot" }, button({ label: "Закрыть", kind: "primary", onClick: () => close() }))));
    pre.scrollTop = pre.scrollHeight;
  }

  async function newTask() {
    let cat;
    try { cat = await api.get("/dev/tasks/catalog"); } catch (e) { toast(errText(e), { kind: "alarm" }); return; }
    const pick = await formModal({ title: "Новая задача", fields: [{ name: "task", label: "Что запустить", type: "select", value: cat[0] && cat[0].id, options: cat.map((t) => [t.id, t.title]) }], submit: "Дальше", onSubmit: async () => {} });
    if (!pick) return;
    const t = cat.find((x) => x.id === pick.task);
    const fields = t.fields.map((f) => ({
      name: f.key, label: f.label, help: f.help, value: f.default,
      type: f.kind === "bool" ? "checkbox" : f.kind === "lines" ? "textarea" : f.kind === "choice" ? "select" : f.kind === "int" || f.kind === "float" ? "text" : "text",
      options: f.kind === "choice" ? f.choices.map((c) => [c, c]) : undefined,
    }));
    const r = await formModal({
      title: t.title, intro: `${t.description}${t.needs_net ? " Задаче нужна сеть." : ""}`, fields, submit: "Запустить",
      onSubmit: async (v) => {
        const values = {};
        t.fields.forEach((f) => { let x = v[f.key]; if ((f.kind === "int" || f.kind === "float") && String(x).trim() !== "") { x = Number(String(x).replace(",", ".")); if (!Number.isFinite(x)) throw new Error(`Поле «${f.label}» должно быть числом`); } values[f.key] = x; });
        await api.post("/dev/tasks", { task: t.id, values });
      },
    });
    if (r) { toast("Задача запущена", { kind: "ok" }); tabTasks(); }
  }

  // ───────── заглушки ─────────
  function tabOof() {
    body.append(card(null, h("h2", null, "Честная оценка"), h("p", { class: "muted", style: { margin: "6px 0 14px" } }, "F1 на тех же клипах, на которых обучались, завышена. Честная оценка считает качество на клипах, которых модель не видела (вложенная перекрёстная проверка)."),
      stubBox("Вложенная оценка (OOF)", ["Таблица сравнения «на обучении» и «на отложенных» с интервалами", "Кривая обучения и проверка перестановкой меток", "Сейчас запускается задачей «Честная оценка наборов признаков» (вкладка «Задачи»), итог — в журнале прогонов"], { api: "POST /dev/tasks  ·  GET /dev/runs" })));
  }
  async function tabClassifier() {
    body.append(skeleton(2, 80));
    let names;
    try { names = await api.get("/dev/classifiers/specs"); } catch (e) { clear(body).append(errorBox(errText(e), tabClassifier)); return; }
    if (!ctx.active() || st.tab !== "classifier") return;
    clear(body).append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Классификатор циклов"), chip(`спецификаций: ${names.length}`, "line")),
      h("p", { class: "muted", style: { marginBottom: "14px" } }, "Спецификация описывает, из каких признаков и как обучается модель, решающая, настоящий ли жест затяжки перед нами. Обучение запускается задачей."),
      names.length ? h("div", { class: "vlist" }, ...names.map((n) => h("button", { class: "vitem", type: "button", onClick: () => spec(n) }, icon("layers", 18), h("b", { class: "grow" }, n), icon("chevr", 16)))) : empty("Спецификаций нет", "Они появляются после настройки классификатора в Dev Studio."),
      h("div", { style: { marginTop: "16px" } }, stubBox("Диагностика переобучения", ["Разрыв «обучение — отложенные», кривая обучения, проверка перестановкой меток", "Конструктор признаков и калибровка"], { api: "sd classifier-*  ·  Dev Studio, вкладка «Лаборатория»" }))));
  }

  async function spec(name) {
    try {
      const d = await api.get(`/dev/classifiers/specs/${encodeURIComponent(name)}`);
      const close = openModal(h("div", { class: "modal", style: { width: "min(760px,100%)" } }, h("h2", null, name),
        h("pre", { class: "mono small", style: { background: "var(--panel-2)", border: "1px solid var(--line)", padding: "14px", borderRadius: "10px", maxHeight: "55vh", overflow: "auto", margin: 0, whiteSpace: "pre-wrap" } }, JSON.stringify(d, null, 2)),
        h("div", { class: "modal__foot" }, button({ label: "Закрыть", kind: "primary", onClick: () => close() }))));
    } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  show();
}
