// Разметка (разработчик): версии файлов разметки с возможностью зафиксировать текущее состояние. Ввод меток и очередь неясных циклов — следующий этап.
import { h, clear, fmtDateTime } from "../dom.js";
import { api, errText } from "../api.js";
import { button, card, chip, empty, errorBox, formModal, pageHead, skeleton, stubBox, tabs, toast } from "../ui.js";

const KINDS = { events: ["События", "Эталон тревог: где в клипе человек курит, а где нет"], gestures: ["Жесты", "Метки отдельных жестов: затяжка, питьё, телефон, касание лица"], cycles: ["Циклы", "Метки циклов «рука ко рту» для обучения классификатора"] };

export default async function labeling(root, ctx) {
  const st = { kind: ctx.query.kind && KINDS[ctx.query.kind] ? ctx.query.kind : "events" };
  const body = h("div");
  const nav = tabs(Object.entries(KINDS).map(([k, [l]]) => [k, l]), st.kind, (v) => { st.kind = v; ctx.setQuery({ kind: v }); load(); });
  root.append(pageHead("Разметка", "Версии эталона: что менялось, кто и когда. Метрики сравнимы только на одной версии."), nav, body);

  async function load() {
    clear(body).append(skeleton(2, 90));
    let rows;
    try { rows = await api.get(`/dev/labels/${st.kind}/versions`); } catch (e) { if (ctx.active()) clear(body).append(errorBox(errText(e), load)); return; }
    if (!ctx.active()) return;
    const [title, hint] = KINDS[st.kind];
    const last = rows[0];
    clear(body).append(
      h("div", { class: "cols" },
        h("div", { class: "f2 stack" }, card(null, h("div", { class: "card__head" }, h("div", { class: "grow" }, h("h2", null, title), h("span", { class: "small muted" }, hint)), button({ label: "Зафиксировать версию", ic: "plus", kind: "primary", size: "sm", onClick: snapshot })),
          rows.length ? h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Когда", "Кто", "Строк", "Откуда", "Заметка"].map((c) => h("th", null, c)))),
            h("tbody", null, ...rows.map((r) => h("tr", null, h("td", null, fmtDateTime(r.ts)), h("td", null, r.author || "—"), h("td", null, r.rows ?? "—"), h("td", null, chip(r.action || "—", "line")), h("td", { class: "small" }, r.note || ""))))))
            : empty("Версий нет", "Зафиксируйте текущее состояние файла, чтобы отслеживать правки.")),
          stubBox("Просмотр клипа и ввод меток", ["Клип с интервалом, кнопки «затяжка / нет жеста / неясно», горячие клавиши", "Очередь неясных циклов (активное обучение) и согласие двух разметчиков", "Сейчас метки правятся в Dev Studio, вкладка «Данные»"], { api: "POST /dev/labels/{вид}/snapshot" })),
        h("div", { class: "f1 stack" }, card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Текущая версия")),
          last ? h("div", { class: "stack stack--sm" }, h("div", { class: "kpi__v", style: { fontSize: "32px" } }, last.rows ?? "—"), h("span", { class: "muted small" }, "строк разметки"), h("span", { class: "small" }, `Зафиксировано ${fmtDateTime(last.ts)}`), last.gt_fp ? h("span", { class: "small muted mono" }, `отпечаток ${String(last.gt_fp).slice(0, 12)}`) : null)
            : h("p", { class: "muted" }, "Файл разметки ещё не зафиксирован."))))
    );
  }

  async function snapshot() {
    const r = await formModal({ title: "Зафиксировать версию", intro: `Будет сохранён снимок файла «${KINDS[st.kind][0].toLowerCase()}».`, fields: [{ name: "note", label: "Что изменилось", placeholder: "Например: добавлены 12 клипов с дальних камер" }], submit: "Зафиксировать",
      onSubmit: async (v) => { const res = await api.post(`/dev/labels/${st.kind}/snapshot`, undefined, { note: v.note.trim() }); if (res && res.note === "файла разметки нет") throw new Error("Файла разметки пока нет — фиксировать нечего"); } });
    if (r) { toast("Версия зафиксирована", { kind: "ok" }); load(); }
  }

  await load();
}
