// Отчёты: показатели за период, динамика, причины ложных, разрез по камерам, суточный профиль, наблюдения; выгрузка CSV и печать в PDF.
import { h, clear, num, pct, plural, fmtDate } from "../dom.js";
import { api, errText } from "../api.js";
import * as store from "../store.js";
import { button, card, chip, empty, errorBox, pageHead, seg, skeleton, toast, REASON_RU } from "../ui.js";
import { stackedBars, hbars } from "../charts.js";
import { saveText, toCsv } from "../files.js";

const PERIODS = [["1", "Сутки"], ["3", "3 дня"], ["7", "7 дней"], ["14", "14 дней"], ["30", "30 дней"]];

const dayLabel = (t) => `${t.slice(8, 10)}.${t.slice(5, 7)}`;

export default async function reports(root, ctx) {
  const f = { days: ctx.query.days || "7", camera: ctx.query.camera || "", bucket: ctx.query.bucket || "day" };
  let cams = [];
  try { cams = await store.cameras(); } catch { /* фильтр по камере необязателен */ }
  let last = null;

  const body = h("div", { class: "stack stack--lg" }, skeleton(2, 120));
  const periodSeg = seg(PERIODS, f.days, (v) => { f.days = v; if (Number(v) > 3) f.bucket = "day"; sync(); load(); }, { label: "Период" });
  const camSel = h("select", { class: "input input--sm", style: { width: "auto", minWidth: "180px" }, "aria-label": "Камера", onChange: (e) => { f.camera = e.target.value; sync(); load(); } },
    h("option", { value: "" }, "Все камеры"), ...cams.map((m) => h("option", { value: m.camera.camera_id, selected: m.camera.camera_id === f.camera }, store.camName(m.camera))));
  const bucketHolder = h("span");
  const sync = () => { ctx.setQuery({ days: f.days, camera: f.camera, bucket: f.bucket }); drawBucket(); };
  const drawBucket = () => {
    clear(bucketHolder);
    if (Number(f.days) <= 3) bucketHolder.append(seg([["day", "По дням"], ["hour", "По часам"]], f.bucket, (v) => { f.bucket = v; sync(); load(); }, { label: "Разбивка" }));
  };
  drawBucket();

  root.append(pageHead("Отчёты", "Сводка по тревогам и решениям операторов за выбранный период.",
    button({ label: "Скачать CSV", ic: "download", size: "sm", onClick: exportCsv }), button({ label: "Печать / PDF", ic: "print", size: "sm", onClick: () => window.print() })),
    h("div", { class: "row no-print", style: { marginBottom: "18px" } }, periodSeg, bucketHolder, camSel), body);

  async function load() {
    clear(body).append(skeleton(2, 120));
    try {
      last = await api.get("/reports/summary", { days: f.days, bucket: Number(f.days) <= 3 ? f.bucket : "day", camera: f.camera || undefined });
      if (ctx.active()) draw();
    } catch (e) { if (ctx.active()) clear(body).append(errorBox(errText(e), load)); }
  }

  function observations(r) {
    const t = r.totals, out = [];
    const decided = t.confirmed + t.false;
    if (!t.total) return [];
    if (t.new) out.push(`Не разобрано ${t.new} ${plural(t.new, "тревога", "тревоги", "тревог")} (${pct(t.new / t.total)} от всех за период).`);
    if (decided >= 10 && t.precision != null && t.precision < 0.6) out.push(`Ложных больше, чем хотелось бы: подтверждено ${pct(t.precision)} из ${decided} решённых. Стоит посмотреть причины и камеры ниже.`);
    if (decided >= 10 && t.precision != null && t.precision >= 0.85) out.push(`Точность высокая: ${pct(t.precision)} решённых тревог подтверждены.`);
    const rs = r.reasons.filter((x) => x.reason !== "—");
    if (rs.length && t.false >= 5) out.push(`Чаще всего ложные из-за «${(REASON_RU[rs[0].reason] || rs[0].reason).toLowerCase()}»: ${rs[0].n} из ${t.false}.`);
    const worst = r.cameras.filter((c) => c.confirmed + c.false >= 5).map((c) => ({ ...c, share: c.false / (c.confirmed + c.false) })).sort((a, b) => b.share - a.share)[0];
    if (worst && worst.share >= 0.6) out.push(`Больше всего ложных на камере «${store.camName(worst)}»: ${pct(worst.share)} решённых.`);
    const peak = r.hours.indexOf(Math.max(...r.hours));
    if (Math.max(...r.hours) >= 3) out.push(`Пик тревог — около ${String(peak).padStart(2, "0")}:00 (${r.hours[peak]} за период).`);
    if (r.truncated) out.push("Тревог за период слишком много, учтены не все: сузьте период.");
    return out;
  }

  function draw() {
    const r = last, t = r.totals;
    clear(body);
    body.append(h("p", { class: "print-only" }, `Период: ${fmtDate(r.since)} — ${fmtDate(r.until)}${f.camera ? `, камера ${f.camera}` : ""}`));
    if (!t.total) { body.append(empty("За период тревог нет", "Выберите больший период или другую камеру.")); return; }
    const kpi = (label, value, note) => card({ class: "card--tight" }, h("div", { class: "kpi" }, h("div", { class: "kpi__l" }, label), h("div", { class: "kpi__v", style: { fontSize: "30px" } }, value), h("div", { class: "kpi__n" }, note)));
    body.append(h("div", { class: "kpis" },
      kpi("Всего", t.total, `за ${r.days} ${plural(r.days, "день", "дня", "дней")}, уверенность ${t.mean_confidence == null ? "—" : pct(t.mean_confidence)}`), kpi("Подтверждено", t.confirmed, "курение"), kpi("Ложных", t.false, "отмечены оператором"), kpi("Не уверен", t.unsure, "ждут второго мнения"),
      kpi("Ждут решения", t.new, "в очереди"), kpi("Точность", t.precision == null ? "—" : pct(t.precision), "подтверждённые из решённых")));

    const series = r.series.map((d) => ({ label: r.bucket === "day" ? dayLabel(d.t) : d.t.slice(11, 13), done: d.confirmed + d.false + d.unsure, pending: d.new }));
    const dyn = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Динамика"), chip(r.bucket === "day" ? "по дням" : "по часам", "line")),
      h("div", { style: { overflowX: "auto" } }, stackedBars(series, { width: Math.max(560, series.length * 18 + 16), height: 170, label: "Тревоги по периодам: решённые и ожидающие" })),
      h("div", { class: "legend", style: { marginTop: "10px" } }, h("span", null, h("i", { style: { background: "var(--ink)" } }), "решены"), h("span", null, h("i", { class: "hatch", style: { border: "1px solid var(--ink)" } }), "ждут")));

    const reasons = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Почему ложные")),
      r.reasons.length ? hbars(r.reasons.map((x) => ({ label: REASON_RU[x.reason] || (x.reason === "—" ? "Без причины" : x.reason), value: x.n })), { cls: "" }) : h("p", { class: "muted small" }, "Ложных тревог с решением пока нет."));

    const hour = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Время суток")),
      stackedBars(r.hours.map((n, i) => ({ label: i % 3 === 0 ? String(i).padStart(2, "0") : "", done: n, pending: 0 })), { width: 520, height: 130, label: "Тревоги по часам суток" }));

    const obs = observations(r);
    const notes = card({ class: "card--flat" }, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Наблюдения"), chip("считаются по правилам, без модели", "line")),
      obs.length ? h("ul", { style: { margin: 0, paddingLeft: "18px" } }, ...obs.map((x) => h("li", { style: { marginBottom: "6px" } }, x))) : h("p", { class: "muted small" }, "Заметных отклонений нет."));

    const table = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Камеры")),
      h("div", { class: "tablewrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, ...["Камера", "Всего", "Подтв.", "Ложных", "Не увер.", "Ждут", "Подтверждение"].map((c, i) => h("th", { class: i && i < 6 ? "num" : "" }, c)))),
        h("tbody", null, ...r.cameras.map((c) => {
          const dec = c.confirmed + c.false;
          return h("tr", null, h("td", null, h("b", null, store.camName(c))), h("td", { class: "num" }, c.total), h("td", { class: "num" }, c.confirmed), h("td", { class: "num" }, c.false), h("td", { class: "num" }, c.unsure), h("td", { class: "num" }, c.pending),
            h("td", { style: { minWidth: "140px" } }, dec ? h("div", { class: "row row--nowrap" }, h("div", { class: "bar bar--accent grow" }, h("i", { style: { width: `${Math.round((c.confirmed / dec) * 100)}%` } })), h("span", { class: "small" }, pct(c.confirmed / dec))) : h("span", { class: "muted small" }, "нет решений")));
        })))));

    body.append(h("div", { class: "cols" }, h("div", { class: "f2 stack" }, dyn, table), h("div", { class: "f1 stack" }, notes, reasons, hour)));
  }

  async function exportCsv() {
    if (!last) return;
    const r = last;
    const rows = [["Период", r.since, r.until], [], ["Камера", "Район", "Всего", "Подтверждено", "Ложных", "Не уверен", "Ждут", "Средняя уверенность"],
      ...r.cameras.map((c) => [c.camera_id, c.district || "", c.total, c.confirmed, c.false, c.unsure, c.pending, c.mean_confidence == null ? "" : num(c.mean_confidence, 3)]), [],
      ["Период", "Всего", "Подтверждено", "Ложных", "Не уверен", "Ждут"], ...r.series.map((d) => [d.t, d.total, d.confirmed, d.false, d.unsure, d.new])];
    saveText(`report_${r.since.slice(0, 10)}_${r.until.slice(0, 10)}.csv`, toCsv(rows), "text/csv;charset=utf-8");
    toast("Отчёт сохранён в CSV", { kind: "ok" });
  }

  await load();
}
