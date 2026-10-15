// Карта камер: схема по координатам, список слева, карточка выбранной камеры справа. Администратор задаёт координаты прямо отсюда.
import { h, clear, ago, pct, dur, debounce } from "../dom.js";
import { api, errText } from "../api.js";
import { bus } from "../bus.js";
import { session } from "../session.js";
import { href } from "../router.js";
import * as store from "../store.js";
import { button, chip, empty, errorBox, formModal, pageHead, seg, skeleton, toast } from "../ui.js";
import { drawMap, layout, stateLabel } from "../mapview.js";

const PERIODS = [["24", "24 часа"], ["168", "7 дней"], ["0", "Всё время"]];
const STATUS_RU = { idle: "простаивает", running: "обрабатывает", error: "ошибка", offline: "нет связи" };

export default async function map(root, ctx) {
  const state = { hours: ctx.query.hours ?? "168", sel: ctx.query.camera || null, filter: "", list: [] };
  const box = h("div", { class: "mapbox" }, skeleton(1, 520));
  const hint = h("div", { class: "banner banner--info", hidden: true, style: { marginTop: "12px" } }, "Координаты камер не заданы: они расставлены по районам условно.");
  const periodSeg = seg(PERIODS, state.hours, (v) => { state.hours = v; ctx.setQuery({ hours: v }); load(); }, { label: "Период показателей" });
  root.append(pageHead("Карта камер", "Где сейчас очередь и какие камеры дают больше всего тревог.", periodSeg), box,
    h("div", { class: "legend", style: { marginTop: "12px" } }, h("span", null, h("i", { style: { background: "#D93A1B" } }), "есть ждущие тревоги"), h("span", null, h("i", { style: { background: "#0B7A4B" } }), "очереди нет"), h("span", null, h("i", { style: { background: "#8A958D" } }), "камера отключена")), hint);

  async function load(silent = false) {
    if (!silent) clear(box).append(skeleton(1, 520));
    try {
      state.list = await api.get("/cameras", { hours: Number(state.hours) || undefined });
      if (!ctx.active()) return;
      draw();
    } catch (e) { if (ctx.active()) clear(box).append(errorBox(errText(e), load)); }
  }

  function select(id) { state.sel = id; ctx.setQuery({ camera: id }); draw(); }

  function listPanel(lay) {
    const q = state.filter.toLowerCase();
    const match = (m) => !q || `${store.camName(m.camera)} ${m.camera.street} ${m.camera.camera_id}`.toLowerCase().includes(q);
    const search = h("input", { class: "input input--sm", placeholder: "Поиск камеры", value: state.filter, "aria-label": "Поиск камеры" });
    const scroll = h("div", { class: "scroll" });
    const paint = () => {
      clear(scroll);
      const shown = [...state.list].filter(match).sort((a, b) => b.pending - a.pending || store.camName(a.camera).localeCompare(store.camName(b.camera), "ru"));
      if (!shown.length) scroll.append(h("p", { class: "muted small" }, "Ничего не найдено"));
      for (const m of shown) {
        const dotCls = !m.camera.enabled ? "" : m.pending ? "dot--alarm" : "dot--ok";
        scroll.append(h("button", { class: ["camrow", state.sel === m.camera.camera_id && "is-active"], type: "button", onClick: () => select(m.camera.camera_id) },
          h("span", { class: ["dot", dotCls] }), h("div", { class: "grow" }, h("b", { class: "ellipsis", style: { display: "block" } }, store.camName(m.camera)), h("span", { class: "small muted ellipsis", style: { display: "block" } }, m.camera.street || m.camera.camera_id)),
          m.pending ? h("span", { class: "badge" }, m.pending) : h("span", { class: "small muted" }, m.alerts)));
      }
    };
    search.addEventListener("input", debounce(() => { state.filter = search.value; paint(); }, 120));
    paint();
    return h("div", { class: "mapbox__list" }, h("div", { class: "row row--nowrap" }, h("h2", { class: "grow" }, "Камеры"), chip(String(state.list.length), "line")), search, scroll,
      lay.unplaced.length ? h("p", { class: "small muted" }, `Без координат: ${lay.unplaced.length}. ${session.can("admin") ? "Укажите их в карточке камеры." : "Координаты задаёт администратор."}`) : null);
  }

  function infoCard(m) {
    const c = m.camera;
    const line = (k, v) => h("div", { class: "row row--nowrap", style: { justifyContent: "space-between" } }, h("span", { class: "muted" }, k), h("span", { style: { textAlign: "right" } }, v));
    return h("div", { class: "mapbox__card" },
      h("div", { class: "row row--nowrap" }, h("div", { class: "grow" }, h("h2", null, store.camName(c)), h("span", { class: "small muted" }, [c.city, c.street].filter(Boolean).join(", ") || c.camera_id)), button({ ic: "x", kind: "ghost", size: "sm", title: "Закрыть", onClick: () => select(null) })),
      chip(stateLabel(m), m.pending ? "alarm" : c.enabled ? "ok" : "", null, "chip--lg"),
      h("div", { class: "stack stack--sm small" }, line("Тревог за период", m.alerts), line("Ждут решения", m.pending), line("Средняя уверенность", m.mean_confidence == null ? "—" : pct(m.mean_confidence)),
        line("Состояние", STATUS_RU[c.status] || c.status), line("На связи", c.last_seen ? ago(c.last_seen) : "—"), line("Обработано", dur(c.processed_sec))),
      h("div", { class: "row" }, button({ label: "Тревоги камеры", kind: "accent", size: "sm", href: href("/alerts", { camera: c.camera_id }) }),
        session.can("admin") ? button({ label: "Координаты", ic: "pin", size: "sm", onClick: () => editCoords(c) }) : null));
  }

  async function editCoords(c) {
    const r = await formModal({
      title: `Координаты: ${store.camName(c)}`, intro: "Широта и долгота в градусах, например 59.5634 и 30.1286. Пустые поля уберут камеру с карты.",
      fields: [{ name: "lat", label: "Широта", value: c.lat ?? "", type: "text", placeholder: "59.5634" }, { name: "lon", label: "Долгота", value: c.lon ?? "", type: "text", placeholder: "30.1286" }],
      onSubmit: async (v) => {
        const parse = (x, name, lo, hi) => { const t = String(x).trim().replace(",", "."); if (!t) return null; const n = Number(t); if (!Number.isFinite(n) || n < lo || n > hi) throw new Error(`${name}: число от ${lo} до ${hi}`); return n; };
        const lat = parse(v.lat, "Широта", -90, 90), lon = parse(v.lon, "Долгота", -180, 180);
        if ((lat === null) !== (lon === null)) throw new Error("Заполните оба поля или оставьте оба пустыми");
        await api.put(`/cameras/${encodeURIComponent(c.camera_id)}`, { lat, lon });
      },
    });
    if (r) { store.cameras.reset(); toast("Координаты сохранены", { kind: "ok" }); load(true); }
  }

  function draw() {
    clear(box);
    hint.hidden = true;
    if (!state.list.length) { box.append(empty("Камер пока нет", session.can("admin") ? "Добавьте камеры в разделе «Администрирование»." : "Камеры заводит администратор.", session.can("admin") ? button({ label: "Администрирование", href: href("/admin"), kind: "primary" }) : null)); box.style.minHeight = "0"; return; }
    box.style.minHeight = "";
    const lay = layout(state.list);
    const selM = state.list.find((m) => m.camera.camera_id === state.sel);
    const stage = h("div", { class: "mapbox__stage" }, drawMap(lay, { selected: state.sel, onSelect: select }));
    if (selM) stage.append(infoCard(selM));
    box.append(listPanel(lay), stage);
    hint.hidden = lay.mode !== "grid";
  }

  await load();
  const reload = debounce(() => ctx.active() && load(true), 1000);
  for (const ev of ["live:alert", "live:review", "alerts:changed"]) ctx.cleanup(bus.on(ev, reload));
}
