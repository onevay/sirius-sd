// Каркас приложения: боковая панель (на телефоне — нижняя), поиск-палитра Ctrl+K, живые уведомления, загрузка экранов по маршруту с защитой от гонок и ошибок.
import { h, clear, icon, initials, plural, pct } from "./dom.js";
import { bus } from "./bus.js";
import { session, ROLE_RU } from "./session.js";
import { live } from "./live.js";
import { SCREENS, GROUPS, BOTTOM } from "./registry.js";
import { go, href, listen, match, parse, replace } from "./router.js";
import { openModal, closeAllModals, toast, empty, errorBox, skeleton, button } from "./ui.js";
import { errText } from "./api.js";
import * as store from "./store.js";

const TITLE = "Детектор курения";

export function startShell(root) {
  const allowed = () => SCREENS.filter((s) => session.can(s.role));
  let cleanups = [];
  let token = 0;
  let lastPath = null;
  const offs = [];

  // ───────── боковая панель ─────────
  const badgeEl = h("span", { class: "badge", hidden: true });
  const links = new Map();
  const navGroup = (gid) => {
    const items = allowed().filter((s) => s.nav && s.group === gid);
    if (!items.length) return null;
    return h("div", null, h("div", { class: "rail__group label" }, GROUPS[gid]), ...items.map((s) => {
      const a = h("a", { class: "rail__link", href: href(s.navPath || s.path) }, icon(s.icon, 18), h("span", { class: "grow" }, s.title));
      if (s.badge === "new") a.append(badgeEl);
      links.set(s.id, a);
      return a;
    }));
  };
  const dot = h("span", { class: "dot" });
  const liveText = h("span", null, "Нет связи");
  const me = session.me;
  const rail = h("aside", { class: "rail", "aria-label": "Разделы" },
    h("a", { class: "brand", href: href("/"), "aria-label": "На главную" }, h("div", { class: "brand__mark" }, icon("shield", 18, { style: { color: "#fff" } })), h("div", { class: "brand__t" }, h("b", null, TITLE), h("span", null, "Videofor × Сириус"))),
    ...["work", "dev", "system"].map(navGroup).filter(Boolean),
    h("div", { class: "rail__foot" },
      h("div", { class: "live", role: "status" }, dot, liveText),
      h("a", { class: "me", href: href("/account"), title: "Мой профиль" }, h("div", { class: "me__av" }, initials(me.display_name || me.username)),
        h("div", { class: "me__t grow" }, h("b", { class: "ellipsis" }, me.display_name || me.username), h("span", { class: "small muted" }, ROLE_RU[me.role] || me.role))),
      session.health && session.health.auth_required ? button({ label: "Выйти", ic: "logout", kind: "ghost", size: "sm", onClick: () => session.logout() }) : h("div", { class: "small muted", style: { padding: "0 4px" } }, "Вход отключён: локальный режим")));

  // ───────── верхняя строка и область экрана ─────────
  const searchBtn = h("button", { class: "search", type: "button", onClick: openPalette, "aria-label": "Поиск и переходы" }, icon("search", 17), h("span", { class: "grow" }, "Раздел, камера или номер тревоги"), h("span", { class: "kbd" }, "Ctrl K"));
  const view = h("div", { id: "view", tabindex: "-1" });
  const main = h("main", { class: "main" }, h("div", { class: "page" }, h("div", { class: "topbar no-print" }, searchBtn), view));

  // ───────── нижняя панель телефона ─────────
  const bottomLinks = new Map();
  const bottom = h("nav", { class: "bottombar", "aria-label": "Разделы" },
    ...BOTTOM.map((id) => SCREENS.find((s) => s.id === id)).filter((s) => s && session.can(s.role)).map((s) => {
      const a = h("a", { href: href(s.navPath || s.path) }, icon(s.icon, 20), s.title);
      if (s.badge === "new") a.append(h("span", { class: "badge", hidden: true, "data-new": "1" }));
      bottomLinks.set(s.id, a);
      return a;
    }),
    h("button", { type: "button", onClick: openMore }, icon("more", 20), "Ещё"));

  clear(root);
  root.append(h("div", { class: "app" }, rail, main), bottom);

  function openMore() {
    const rest = allowed().filter((s) => s.nav && !BOTTOM.includes(s.id));
    const box = h("div", { class: "modal" }, h("h2", null, "Разделы"),
      h("div", { class: "stack stack--sm" }, ...rest.map((s) => h("a", { class: "rail__link", href: href(s.navPath || s.path), onClick: () => close() }, icon(s.icon, 18), s.title)),
        h("a", { class: "rail__link", href: href("/account"), onClick: () => close() }, icon("settings", 18), "Мой профиль")),
      session.health && session.health.auth_required ? button({ label: "Выйти", ic: "logout", onClick: () => { close(); session.logout(); } }) : null);
    const close = openModal(box);
  }

  // ───────── число ждущих тревог ─────────
  let badgeTimer = 0;
  async function refreshBadge() {
    try {
      const st = await store.stats(true);
      const n = st.new;
      badgeEl.hidden = !n;
      badgeEl.textContent = n > 99 ? "99+" : String(n);
      bottom.querySelectorAll("[data-new]").forEach((b) => { b.hidden = !n; b.textContent = n > 99 ? "99+" : String(n); });
      document.title = `${n ? `(${n}) ` : ""}${currentTitle}`;
    } catch { /* значок не критичен */ }
  }
  const soon = () => { clearTimeout(badgeTimer); badgeTimer = setTimeout(refreshBadge, 400); };

  // ───────── загрузка экранов ─────────
  let currentTitle = TITLE;
  function setActive(id) {
    for (const [k, a] of links) { a.classList.toggle("is-active", k === id); k === id ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"); }
    for (const [k, a] of bottomLinks) a.classList.toggle("is-active", k === id);
  }

  async function show(loc) {
    const my = ++token;
    closeAllModals();
    for (const fn of cleanups) { try { fn(); } catch (e) { console.error(e); } }
    cleanups = [];
    const m = match(SCREENS, loc);
    if (!m) { setActive(null); clear(view).append(empty("Раздел не найден", `Адрес «${loc.path}» не существует.`, button({ label: "На главную", kind: "primary", href: href("/") }))); return; }
    const sc = m.screen;
    currentTitle = `${sc.title} · ${TITLE}`;
    document.title = currentTitle;
    setActive(sc.id);
    if (!session.can(sc.role)) { clear(view).append(empty("Недостаточно прав", `Раздел «${sc.title}» доступен роли «${ROLE_RU[sc.role]}» и выше.`, button({ label: "На главную", kind: "primary", href: href("/") }))); return; }
    clear(view).append(skeleton(3, 88));
    if (lastPath !== loc.path) window.scrollTo(0, 0);
    lastPath = loc.path;
    try {
      const mod = await sc.load();
      if (my !== token) return;
      const holder = h("div", { class: "screen", "data-screen": sc.id });
      const ctx = {
        params: m.params, query: m.query, screen: sc, go, user: session.me,
        cleanup(fn) { if (my === token) cleanups.push(fn); else try { fn(); } catch (e) { console.error(e); } },
        active: () => my === token,
        setQuery(q) { replace(loc.path, { ...m.query, ...q }); },
      };
      clear(view).append(holder);
      await mod.default(holder, ctx);
    } catch (e) {
      console.error(`экран «${sc.id}»`, e);
      if (my === token) clear(view).append(errorBox(`Не удалось открыть «${sc.title}»: ${errText(e)}`, () => show(parse())));
    }
  }

  // ───────── палитра ─────────
  function openPalette() {
    const input = h("input", { class: "palette__in", placeholder: "Раздел, камера или номер тревоги…", "aria-label": "Поиск", autocomplete: "off" });
    const list = h("div", { class: "palette__list", role: "listbox" });
    const box = h("div", { class: "palette" }, input, list);
    let sel = 0, shown = [], camItems = [];
    const base = allowed().filter((s) => s.nav || s.id === "account").map((s) => ({ label: s.title, hint: "раздел", ic: s.icon, run: () => go(s.navPath || s.path) }));
    store.cameras().then((cams) => { camItems = cams.map((m) => ({ label: `Камера ${m.camera.title || m.camera.camera_id}`, hint: `${m.alerts} ${plural(m.alerts, "тревога", "тревоги", "тревог")}`, ic: "video", run: () => go("/alerts", { camera: m.camera.camera_id }), key: `${m.camera.camera_id} ${m.camera.district || ""} ${m.camera.cam_index || ""}` })); draw(); }).catch(() => {});
    const draw = () => {
      const q = input.value.trim().toLowerCase();
      const items = [...base, ...camItems];
      const id = /^#?(\d{1,9})$/.exec(q);
      shown = items.filter((i) => !q || `${i.label} ${i.key || ""}`.toLowerCase().includes(q)).slice(0, 12);
      if (id) shown.unshift({ label: `Открыть тревогу №${id[1]}`, hint: "проверка", ic: "bell", run: () => go(`/review/${id[1]}`) });
      sel = Math.min(sel, Math.max(0, shown.length - 1));
      clear(list);
      if (!shown.length) list.append(h("div", { class: "empty" }, "Ничего не найдено"));
      shown.forEach((it, i) => list.append(h("button", { class: ["palette__it", i === sel && "is-active"], type: "button", role: "option", "aria-selected": String(i === sel), onClick: () => pick(it), onMouseenter: () => { sel = i; mark(); } }, icon(it.ic, 18), h("span", { class: "grow" }, it.label), h("span", { class: "small muted" }, it.hint))));
    };
    const mark = () => [...list.children].forEach((c, i) => c.classList.toggle("is-active", i === sel));
    const pick = (it) => { close(); it.run(); };
    input.addEventListener("input", () => { sel = 0; draw(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(sel + 1, shown.length - 1); mark(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(sel - 1, 0); mark(); }
      else if (e.key === "Enter" && shown[sel]) { e.preventDefault(); pick(shown[sel]); }
    });
    const close = openModal(box);
    draw();
  }
  const onKey = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); if (!document.querySelector(".overlay")) openPalette(); }
  };
  document.addEventListener("keydown", onKey);

  // ───────── живые события ─────────
  const liveView = (state) => { dot.className = `dot ${state === "on" ? "dot--ok" : ""}`; liveText.textContent = state === "on" ? "Онлайн, обновляется само" : "Нет связи, переподключаюсь"; };
  liveView(live.state);
  offs.push(bus.on("live:state", liveView));
  offs.push(bus.on("live:ready", soon));
  offs.push(bus.on("live:review", soon));
  offs.push(bus.on("live:alert", (a) => {
    soon();
    if (parse().path.startsWith("/review")) return;           // на экране проверки новые тревоги видны в очереди — всплывающее окно не нужно
    toast(`Новая тревога · камера ${a.camera_id} · уверенность ${pct(a.confidence)}`, { kind: "alarm", action: { label: session.can("operator") ? "Проверить" : "Открыть", run: () => go(`/review/${a.id}`) } });
  }));

  const stopListen = listen(show);
  show(parse());
  refreshBadge();

  return {
    destroy() {
      token++;
      clearTimeout(badgeTimer);
      for (const fn of cleanups) { try { fn(); } catch { /* уходим */ } }
      cleanups = [];
      offs.forEach((f) => f());
      stopListen();
      document.removeEventListener("keydown", onKey);
      document.title = TITLE;
    },
  };
}
