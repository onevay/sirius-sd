// Администрирование: пользователи и роли, камеры, журнал действий, источники видео и демо-данные, состояние системы.
import { h, clear, icon, fmtDateTime, ago, debounce, plural } from "../dom.js";
import { api, errText } from "../api.js";
import { session, ROLES, ROLE_RU } from "../session.js";
import { button, card, chip, confirmDialog, empty, errorBox, formModal, pageHead, skeleton, tabs, toast } from "../ui.js";
import * as store from "../store.js";

const PAGE = 50;
const ROLE_OPTIONS = ROLES.map((r) => [r, ROLE_RU[r]]);
const num = (x, name, lo, hi) => { const t = String(x ?? "").trim().replace(",", "."); if (!t) return null; const n = Number(t); if (!Number.isFinite(n) || n < lo || n > hi) throw new Error(`${name}: число от ${lo} до ${hi}`); return n; };

export default async function admin(root, ctx) {
  const st = { tab: ctx.query.tab || "users" };
  const body = h("div");
  const nav = tabs([["users", "Пользователи"], ["cameras", "Камеры"], ["audit", "Журнал действий"], ["sources", "Источники и демо"], ["system", "Система"]], st.tab, (v) => { st.tab = v; ctx.setQuery({ tab: v }); show(); });
  root.append(pageHead("Администрирование", "Кто и с какими правами работает, какие камеры подключены, что происходило в системе."), nav, body);

  const show = () => { clear(body); ({ users, cameras, audit, sources, system }[st.tab] || users)(); };

  // ───────── пользователи ─────────
  async function users() {
    body.append(skeleton(3, 56));
    let list;
    try { list = await api.get("/users"); } catch (e) { clear(body).append(errorBox(errText(e), users)); return; }
    if (!ctx.active() || st.tab !== "users") return;
    clear(body).append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Пользователи"), chip(String(list.length), "line"), button({ label: "Новый пользователь", ic: "plus", kind: "primary", size: "sm", onClick: create })),
      h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Имя", "Роль", "Состояние", "Последний вход", ""].map((c) => h("th", null, c)))),
        h("tbody", null, ...list.map((u) => h("tr", null, h("td", null, h("b", null, u.display_name || u.username), h("div", { class: "small muted" }, u.username)), h("td", null, chip(ROLE_RU[u.role] || u.role, u.role === "admin" ? "ink" : "line")),
          h("td", null, chip(u.is_active ? "активен" : "заблокирован", u.is_active ? "ok" : "")), h("td", { class: "small" }, u.last_login_at ? `${fmtDateTime(u.last_login_at)} (${ago(u.last_login_at)})` : "не входил"),
          h("td", null, h("div", { class: "row row--nowrap", style: { justifyContent: "flex-end" } }, button({ ic: "edit", size: "sm", title: "Изменить", onClick: () => edit(u) }), button({ ic: "trash", size: "sm", title: "Удалить", onClick: () => remove(u) }))))))))));
  }

  async function create() {
    const r = await formModal({ title: "Новый пользователь", fields: [{ name: "username", label: "Имя для входа", required: true }, { name: "display_name", label: "Отображаемое имя" }, { name: "password", label: "Пароль", type: "password", required: true, help: "Не короче 8 символов" }, { name: "role", label: "Роль", type: "select", value: "operator", options: ROLE_OPTIONS }], submit: "Создать",
      onSubmit: (v) => api.post("/users", { username: v.username.trim(), display_name: v.display_name.trim(), password: v.password, role: v.role }) });
    if (r) { toast("Пользователь создан", { kind: "ok" }); users(); }
  }
  async function edit(u) {
    const self = session.me && session.me.id != null && session.me.id === u.id;
    const r = await formModal({ title: `Изменить: ${u.username}`, intro: self ? "Это вы: роль и блокировку себе менять нельзя." : undefined,
      fields: [{ name: "display_name", label: "Отображаемое имя", value: u.display_name }, ...(self ? [] : [{ name: "role", label: "Роль", type: "select", value: u.role, options: ROLE_OPTIONS }, { name: "is_active", label: "Может входить", type: "checkbox", value: u.is_active }]),
        { name: "password", label: "Новый пароль", type: "password", help: "Оставьте пустым, чтобы не менять. Прежние сессии пользователя закроются." }], submit: "Сохранить",
      onSubmit: (v) => { const b = { display_name: v.display_name.trim() }; if (!self) { b.role = v.role; b.is_active = v.is_active; } if (v.password) b.password = v.password; return api.patch(`/users/${u.id}`, b); } });
    if (r) { toast("Сохранено", { kind: "ok" }); users(); }
  }
  async function remove(u) {
    if (!(await confirmDialog(`Пользователь «${u.username}» будет удалён. История его решений по тревогам сохранится с его именем.`, { ok: "Удалить", danger: true, title: "Удалить пользователя?" }))) return;
    try { await api.del(`/users/${u.id}`); toast("Пользователь удалён"); users(); } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  // ───────── камеры ─────────
  const camFields = (c = {}) => [
    { name: "title", label: "Название", value: c.title }, { name: "city", label: "Город", value: c.city }, { name: "district", label: "Район", value: c.district }, { name: "cam_index", label: "Номер камеры", value: c.cam_index }, { name: "street", label: "Улица", value: c.street },
    { name: "lat", label: "Широта", value: c.lat ?? "" }, { name: "lon", label: "Долгота", value: c.lon ?? "" }, { name: "folder", label: "Папка с записями", value: c.folder, help: "Для имитации потока из файлов" }, { name: "source_url", label: "Адрес потока (RTSP/HTTP)", value: c.source_url },
    { name: "enabled", label: "Камера включена", type: "checkbox", value: c.enabled ?? true }];
  const camBody = (v) => { const lat = num(v.lat, "Широта", -90, 90), lon = num(v.lon, "Долгота", -180, 180); if ((lat === null) !== (lon === null)) throw new Error("Заполните оба поля координат или оставьте оба пустыми"); return { title: v.title.trim(), city: v.city.trim(), district: v.district.trim() || null, cam_index: v.cam_index.trim() || null, street: v.street.trim(), lat, lon, folder: v.folder.trim() || null, source_url: v.source_url.trim() || null, enabled: v.enabled }; };

  async function cameras() {
    body.append(skeleton(3, 56));
    let list;
    try { list = await api.get("/cameras"); } catch (e) { clear(body).append(errorBox(errText(e), cameras)); return; }
    if (!ctx.active() || st.tab !== "cameras") return;
    clear(body).append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Камеры"), chip(String(list.length), "line"), button({ label: "Новая камера", ic: "plus", kind: "primary", size: "sm", onClick: add })),
      list.length ? h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Камера", "Расположение", "Координаты", "Тревог", "Состояние", ""].map((c) => h("th", null, c)))),
        h("tbody", null, ...list.map((m) => { const c = m.camera; return h("tr", null, h("td", null, h("b", null, store.camName(c)), h("div", { class: "small muted mono" }, c.camera_id)), h("td", { class: "small" }, [c.city, c.street].filter(Boolean).join(", ") || "—"),
          h("td", { class: "small" }, c.lat != null ? `${c.lat.toFixed(4)}, ${c.lon.toFixed(4)}` : chip("не заданы", "line")), h("td", null, `${m.alerts}${m.pending ? ` (ждут ${m.pending})` : ""}`), h("td", null, chip(c.enabled ? "включена" : "отключена", c.enabled ? "ok" : "")),
          h("td", null, h("div", { class: "row row--nowrap", style: { justifyContent: "flex-end" } }, button({ ic: "edit", size: "sm", title: "Изменить", onClick: () => editCam(c) }), button({ ic: "trash", size: "sm", title: "Удалить", onClick: () => delCam(c) })))); })))) : empty("Камер нет", "Добавьте камеру или запустите монитор потоков: он заведёт камеры сам.")));
  }
  async function add() {
    const r = await formModal({ title: "Новая камера", fields: [{ name: "camera_id", label: "Идентификатор", required: true, help: "Латиница, цифры, дефис: например gatchina-01" }, ...camFields()], submit: "Создать",
      onSubmit: (v) => { const id = v.camera_id.trim(); if (!/^[\w.\-]{1,128}$/.test(id)) throw new Error("Идентификатор: латиница, цифры, точка, дефис, подчёркивание"); return api.put(`/cameras/${encodeURIComponent(id)}`, camBody(v)); } });
    if (r) { store.cameras.reset(); toast("Камера создана", { kind: "ok" }); cameras(); }
  }
  async function editCam(c) {
    const r = await formModal({ title: `Камера ${c.camera_id}`, fields: camFields(c), submit: "Сохранить", onSubmit: (v) => api.put(`/cameras/${encodeURIComponent(c.camera_id)}`, camBody(v)) });
    if (r) { store.cameras.reset(); toast("Сохранено", { kind: "ok" }); cameras(); }
  }
  async function delCam(c) {
    const r = await formModal({ title: `Удалить камеру ${c.camera_id}?`, intro: "Камера пропадёт из справочника. Тревоги по умолчанию остаются в истории.", fields: [{ name: "with_alerts", label: "Удалить и все тревоги этой камеры", type: "checkbox", value: false }], submit: "Удалить",
      onSubmit: (v) => api.del(`/cameras/${encodeURIComponent(c.camera_id)}`, { with_alerts: v.with_alerts ? "true" : undefined }) });
    if (r) { store.cameras.reset(); toast("Камера удалена"); cameras(); }
  }

  // ───────── журнал действий ─────────
  async function audit() {
    const f = { action: "", username: "", offset: 0 };
    const out = h("div");
    const act = h("input", { class: "input input--sm", style: { maxWidth: "220px" }, placeholder: "Действие, например alert.review", "aria-label": "Действие", onInput: debounce((e) => { f.action = e.target.value.trim(); f.offset = 0; load(); }, 300) });
    const usr = h("input", { class: "input input--sm", style: { maxWidth: "200px" }, placeholder: "Пользователь", "aria-label": "Пользователь", onInput: debounce((e) => { f.username = e.target.value.trim(); f.offset = 0; load(); }, 300) });
    body.append(h("div", { class: "row", style: { marginBottom: "14px" } }, act, usr), out);
    async function load() {
      clear(out).append(skeleton(3, 40));
      let r;
      try { r = await api.get("/audit", { action: f.action || undefined, username: f.username || undefined, limit: PAGE, offset: f.offset }); } catch (e) { clear(out).append(errorBox(errText(e), load)); return; }
      if (!ctx.active() || st.tab !== "audit") return;
      clear(out).append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Журнал действий"), chip(`${r.total} ${plural(r.total, "запись", "записи", "записей")}`, "line")),
        r.items.length ? h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Время", "Кто", "Действие", "Объект", "Подробности", "Адрес"].map((c) => h("th", null, c)))),
          h("tbody", null, ...r.items.map((a) => h("tr", null, h("td", { class: "small" }, fmtDateTime(a.ts)), h("td", null, a.username || "—"), h("td", { class: "mono small" }, a.action), h("td", { class: "small" }, [a.entity, a.entity_id].filter(Boolean).join(" ")),
            h("td", { class: "small muted", style: { maxWidth: "320px", overflowWrap: "anywhere" } }, a.details ? JSON.stringify(a.details) : ""), h("td", { class: "small muted" }, a.ip)))))) : empty("Записей нет", "Измените фильтры."),
        h("div", { class: "row", style: { marginTop: "14px", justifyContent: "space-between" } }, button({ label: "Новее", ic: "chevl", size: "sm", disabled: f.offset === 0, onClick: () => { f.offset = Math.max(0, f.offset - PAGE); load(); } }), h("span", { class: "small muted" }, r.total ? `${f.offset + 1}–${Math.min(r.total, f.offset + PAGE)} из ${r.total}` : ""),
          button({ label: "Старше", ic: "chevr", size: "sm", disabled: f.offset + PAGE >= r.total, onClick: () => { f.offset += PAGE; load(); } }))));
    }
    await load();
  }

  // ───────── источники и демо ─────────
  async function sources() {
    body.append(skeleton(3, 56));
    let folders;
    try { folders = await api.get("/sources/folders"); } catch (e) { clear(body).append(errorBox(errText(e), sources)); return; }
    if (!ctx.active() || st.tab !== "sources") return;
    const demoN = h("input", { class: "input input--sm", type: "number", min: 1, max: 200, value: 14, style: { width: "90px" }, "aria-label": "Сколько демо-тревог" });
    clear(body).append(h("div", { class: "cols" },
      h("div", { class: "f15" }, card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Папки с видео"), button({ label: "Подключить папку", ic: "plus", kind: "primary", size: "sm", onClick: addRoot })),
        folders.length ? h("div", { class: "divide" }, ...folders.map((f) => h("div", { class: "row row--nowrap", style: { padding: "10px 0" } }, icon("video", 18), h("div", { class: "grow" }, h("b", { class: "ellipsis", style: { display: "block" } }, f.name), h("span", { class: "small muted" }, `${f.kind === "camera" ? "камера" : "данные"} · файлов: ${f.n}`)),
          f.removable ? button({ ic: "trash", size: "sm", title: "Отключить", onClick: () => dropRoot(f) }) : null))) : empty("Источников нет", "Подключите папку с записями на сервере.")),
        h("p", { class: "small muted", style: { marginTop: "10px" } }, "Подключение папок и загрузка файлов работают, когда сервер запущен на локальном адресе; в сети они отключены.")),
      h("div", { class: "f1" }, card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Демо-данные")), h("p", { class: "muted small", style: { marginBottom: "12px" } }, "Тестовые тревоги помечены «демо» и не попадают в точность и время реакции."),
        h("div", { class: "row" }, demoN, button({ label: "Добавить", ic: "plus", size: "sm", kind: "primary", onClick: () => demo(false) }), button({ label: "Удалить демо", ic: "trash", size: "sm", onClick: () => demo(true) }))))));
    async function demo(clearAll) {
      if (clearAll && !(await confirmDialog("Все тревоги с пометкой «демо» будут удалены.", { ok: "Удалить", danger: true, title: "Удалить демо-данные?" }))) return;
      try { const n = Math.max(1, Math.min(200, Number(demoN.value) || 14)); const r = await api.post("/demo", { clear: clearAll, n }); store.cameras.reset(); store.stats.reset(); toast(clearAll ? `Удалено демо-тревог: ${r.n}` : `Добавлено демо-тревог: ${r.n}`, { kind: "ok" }); } catch (e) { toast(errText(e), { kind: "alarm" }); }
    }
  }
  async function addRoot() {
    const r = await formModal({ title: "Подключить папку", intro: "Путь к папке на компьютере, где работает сервер.", fields: [{ name: "path", label: "Путь", required: true, placeholder: "D:\\\\video\\\\cam01" }], submit: "Подключить", onSubmit: (v) => api.post("/sources/roots", { path: v.path.trim() }) });
    if (r) { toast("Папка подключена", { kind: "ok" }); sources(); }
  }
  async function dropRoot(f) {
    if (!(await confirmDialog(`Папка «${f.name}» будет отключена от системы. Файлы на диске не удаляются.`, { ok: "Отключить", title: "Отключить папку?" }))) return;
    try { await api.del(`/sources/roots/${encodeURIComponent(f.root)}`); toast("Папка отключена"); sources(); } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  // ───────── система ─────────
  async function system() {
    let hl;
    try { hl = await api.get("/health"); } catch (e) { body.append(errorBox(errText(e), system)); return; }
    const line = (k, v, ok) => h("div", { class: "row row--nowrap", style: { padding: "9px 0", borderTop: "1px solid var(--line-2)" } }, h("span", { class: `dot ${ok === undefined ? "" : ok ? "dot--ok" : "dot--alarm"}` }), h("span", { class: "grow" }, k), h("span", { class: "muted" }, v));
    body.append(h("div", { class: "cols" }, h("div", { class: "f1" }, card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Состояние")),
      line("Сервер", `версия ${hl.version}`, hl.ok), line("База данных", hl.db ? "подключена" : "недоступна", hl.db), line("Схема базы", hl.schema_current ? "актуальна" : "нужна миграция (sd db upgrade)", hl.schema_current), line("Ревизия схемы", (hl.schema_revision || []).join(", ") || "—"), line("Вход", hl.auth_required ? "обязателен" : "отключён"))),
      h("div", { class: "f1" }, card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Для разработчиков")), h("p", { class: "muted small", style: { marginBottom: "12px" } }, "Интерфейс работает на том же API, что доступно сторонним системам."),
        button({ label: "Описание API (OpenAPI)", ic: "link", href: "/docs", size: "sm" })))));
  }

  show();
}
