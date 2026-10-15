// Конструктор (разработчик): профили конфигурации — схема конвейера, основные параметры, отличия от умолчаний, правка отдельного параметра, копия, выгрузка YAML.
// Визуальный редактор конвейера (перетаскивание этапов) — следующий этап; его данные уже отдаёт /dev/profiles/{имя}.
import { h, clear, put, icon, debounce } from "../dom.js";
import { api, errText } from "../api.js";
import { href, replace } from "../router.js";
import { button, card, chip, confirmDialog, empty, errorBox, formModal, pageHead, skeleton, stubBox, toast } from "../ui.js";
import { saveText } from "../files.js";

const NODES = [["Видео", (d) => d.fps && `${d.fps} к/с`], ["Поза", (d) => [d.pose, d.imgsz].filter(Boolean).join(" · ")], ["Уточнение", (d) => d.refine], ["Трекер", (d) => d.tracker], ["Циклы", (d) => d.cycle_model],
  ["Предмет", (d) => d.objects], ["Фото", (d) => d.photo], ["Языковая модель", (d) => d.vlm], ["Правила", (d) => [d.threshold != null ? `порог ${d.threshold}` : null, d.fusion].filter((x) => x && x !== "—").join(" · ")]];

const show = (v) => (v === null || v === undefined || v === "" ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v));

/** Вложенная запись a.b.c = value в объект config. */
function setPath(obj, path, value) {
  const keys = path.split(".");
  let cur = obj;
  keys.slice(0, -1).forEach((k) => { if (typeof cur[k] !== "object" || cur[k] === null) cur[k] = {}; cur = cur[k]; });
  cur[keys[keys.length - 1]] = value;
}

export default async function designer(root, ctx) {
  const st = { names: [], name: ctx.params.name || "", p: null, filter: "" };
  const left = h("div", { class: "stack" }, skeleton(3, 40));
  const right = h("div", { class: "stack" });
  root.append(pageHead("Конструктор", "Профили конвейера: что включено, с какими параметрами и чем отличается от умолчаний."), h("div", { class: "cols" }, h("div", { class: "f1" }, left), h("div", { class: "f2" }, right)));

  async function init() {
    try { st.names = await api.get("/dev/profiles"); } catch (e) { clear(left).append(errorBox(errText(e), init)); return; }
    if (!ctx.active()) return;
    if (!st.name && st.names.length) st.name = st.names.includes("mvp") ? "mvp" : st.names[0];
    drawList();
    if (st.name) await open(st.name, false);
    else clear(right).append(empty("Профилей нет", "Профиль можно создать копированием из встроенного."));
  }

  function drawList() {
    clear(left).append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Профили"), chip(String(st.names.length), "line")),
      h("div", { class: "vlist", role: "listbox" }, ...st.names.map((n) => h("button", { class: ["vitem", n === st.name && "is-active"], type: "button", role: "option", "aria-selected": String(n === st.name), onClick: () => open(n) }, icon("layers", 18), h("b", { class: "grow ellipsis" }, n))))));
  }

  async function open(name, push = true) {
    st.name = name;
    if (push) replace(`/designer/${encodeURIComponent(name)}`);
    drawList();
    clear(right).append(skeleton(3, 90));
    try { st.p = await api.get(`/dev/profiles/${encodeURIComponent(name)}`); } catch (e) { if (ctx.active() && st.name === name) clear(right).append(errorBox(errText(e), () => open(name))); return; }
    if (ctx.active() && st.name === name) drawProfile();
  }

  function drawProfile() {
    const p = st.p, d = p.describe || {};
    clear(right);
    const errors = (p.issues || []).filter((i) => i.level === "error"), warns = (p.issues || []).filter((i) => i.level !== "error");
    right.append(card(null, h("div", { class: "card__head" }, h("div", { class: "grow" }, h("h2", null, p.name), h("span", { class: "small muted" }, [p.description, p.base ? `основан на «${p.base}»` : null].filter(Boolean).join(" · ") || "без описания")),
      button({ label: "Копировать", ic: "layers", size: "sm", onClick: clone }), button({ label: "YAML", ic: "download", size: "sm", onClick: yaml }), button({ label: "Изменить параметр", ic: "edit", size: "sm", kind: "primary", onClick: editKey }),
      button({ ic: "trash", size: "sm", title: "Удалить профиль", onClick: remove })),
      h("div", { class: "row small" }, chip(errors.length ? "не готов к запуску" : "готов к запуску", errors.length ? "alarm" : "ok"), p.fingerprint ? h("span", { class: "muted mono" }, `отпечаток ${String(p.fingerprint).slice(0, 12)}`) : null),
      ...[...errors, ...warns].map((i) => h("div", { class: ["banner", i.level === "error" ? "banner--alarm" : ""], style: { marginTop: "10px" } }, icon("warn", 16), i.text))));

    right.append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Конвейер")),
      h("div", { class: "pipe" }, ...NODES.flatMap(([label, fn], i) => {
        const v = fn(d), off = !v || v === "—";
        const node = h("div", { class: ["pipe__n", off && "is-off"] }, h("span", { class: "label" }, label), h("b", { style: { overflowWrap: "anywhere" } }, off ? "выключено" : v));
        return i < NODES.length - 1 ? [node] : [node];
      }))));

    const diff = Object.keys(p.effective || {}).filter((k) => JSON.stringify(p.effective[k]) !== JSON.stringify((p.default || {})[k])).sort();
    right.append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Основные параметры")),
      h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("tbody", null, ...Object.entries(p.main || {}).map(([k, v]) => h("tr", null, h("td", { class: "mono" }, k), h("td", null, show(v)), h("td", { class: "small muted" }, JSON.stringify(v) !== JSON.stringify((p.default || {})[k]) ? chip("не по умолчанию", "info") : "")))))))
    );
    right.append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Отличия от умолчаний"), chip(String(diff.length), "line")),
      diff.length ? h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Параметр", "Значение", "По умолчанию"].map((c) => h("th", null, c)))),
        h("tbody", null, ...diff.map((k) => h("tr", null, h("td", { class: "mono" }, k), h("td", null, h("b", null, show(p.effective[k]))), h("td", { class: "muted" }, show((p.default || {})[k]))))))) : h("p", { class: "muted" }, "Профиль использует значения по умолчанию.")));

    const all = h("div"), q = h("input", { class: "input input--sm", placeholder: "Найти параметр", "aria-label": "Поиск параметра", value: st.filter, onInput: debounce((e) => { st.filter = e.target.value; paintAll(); }, 120) });
    const paintAll = () => {
      clear(all);
      const f = st.filter.trim().toLowerCase();
      const keys = Object.keys(p.effective || {}).sort().filter((k) => !f || k.toLowerCase().includes(f)).slice(0, 200);
      put(all, h("div", { class: "tablewrap", style: { maxHeight: "360px", overflow: "auto" } }, h("table", { class: "table" }, h("tbody", null, ...keys.map((k) => h("tr", null, h("td", { class: "mono" }, k), h("td", null, show(p.effective[k]))))))), keys.length >= 200 ? h("p", { class: "small muted" }, "Показаны первые 200. Уточните поиск.") : null);
    };
    right.append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, `Все параметры (${Object.keys(p.effective || {}).length})`), q), all));
    paintAll();

    right.append(stubBox("Визуальный редактор конвейера", ["Включение и порядок этапов мышью, подсказки по цене пересчёта", "Серия вариантов и сравнение влияния параметров", "Сейчас: правка отдельных параметров, копирование и YAML — выше"], { api: "PUT /dev/profiles/{имя}" }));
    right.append(h("div", { class: "row" }, button({ label: "Разобрать видео этим профилем", ic: "video", href: href("/analyze"), onClick: () => { try { localStorage.setItem("sd_profile", p.name); } catch { /* без сохранения */ } } })));
  }

  async function clone() {
    const r = await formModal({ title: `Копия профиля «${st.name}»`, fields: [{ name: "to", label: "Имя копии", required: true, value: `${st.name}_copy` }], submit: "Создать",
      onSubmit: async (v) => { const n = v.to.trim(); if (!/^[\w.\-а-яА-ЯёЁ]{1,64}$/.test(n)) throw new Error("Имя: буквы, цифры, точка, дефис, подчёркивание"); await api.post(`/dev/profiles/${encodeURIComponent(st.name)}/clone`, { to: n }); return n; } });
    if (r) { st.names = await api.get("/dev/profiles").catch(() => st.names); toast("Копия создана", { kind: "ok" }); open(r); }
  }

  async function yaml() {
    try { saveText(`${st.name}.yaml`, await api.get(`/dev/profiles/${encodeURIComponent(st.name)}/yaml`), "text/yaml;charset=utf-8"); } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  async function editKey() {
    const p = st.p;
    const r = await formModal({
      title: "Изменить параметр", intro: `Значение запишется в профиль «${p.name}». Ключ и значение проверяются до записи.`,
      fields: [{ name: "key", label: "Параметр", required: true, placeholder: "например, video.process_fps", help: "Путь через точки, как в таблице параметров" }, { name: "value", label: "Новое значение", required: true, help: "Число, true/false, текст в кавычках или null" }],
      onSubmit: async (v) => {
        const key = v.key.trim();
        if (!(key in (p.effective || {}))) throw new Error(`Нет параметра «${key}». Скопируйте имя из таблицы «Все параметры».`);
        let val;
        try { val = JSON.parse(v.value.trim()); } catch { val = v.value.trim(); }
        const config = JSON.parse(JSON.stringify(p.config || {}));
        setPath(config, key, val);
        await api.put(`/dev/profiles/${encodeURIComponent(p.name)}`, { description: p.description || "", base: p.base || null, config, options: p.options || {} });
      },
    });
    if (r) { toast("Параметр сохранён", { kind: "ok" }); open(p.name, false); }
  }

  async function remove() {
    if (!(await confirmDialog(`Профиль «${st.name}» будет удалён без возможности восстановления.`, { ok: "Удалить", danger: true, title: "Удалить профиль?" }))) return;
    try { await api.del(`/dev/profiles/${encodeURIComponent(st.name)}`); } catch (e) { toast(errText(e), { kind: "alarm" }); return; }
    toast("Профиль удалён");
    st.names = st.names.filter((n) => n !== st.name);
    st.name = st.names[0] || "";
    drawList();
    st.name ? open(st.name) : clear(right).append(empty("Профилей нет"));
  }

  await init();
}
