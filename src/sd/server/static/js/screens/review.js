// Проверка тревоги: клип, почему сработало, решение одной клавишей (Y — подтвердить, N — ложная, U — не уверен, J/K — соседние), история решений и очередь.
import { h, clear, icon, pct, fmtDateTime, fmtTime, ago } from "../dom.js";
import { api, errText } from "../api.js";
import { bus } from "../bus.js";
import { session } from "../session.js";
import { href, replace } from "../router.js";
import * as store from "../store.js";
import { button, card, chip, empty, errorBox, keyHint, pageHead, ring, sceneSvg, skeleton, statusChip, REASON_RU } from "../ui.js";
import { alertCam, decide, lengthText, ruleText, whenText } from "../alertviews.js";
import { waitForMedia } from "../media.js";

const QUEUE_LIMIT = 100;

export default async function review(root, ctx) {
  const canAct = session.can("operator");
  const holder = h("div", null, skeleton(2, 200));
  root.append(holder);

  let queue = [];            // ждущие тревоги, старые сверху
  let idx = new Map();
  let reasons = REASON_RU;
  let cur = null, hist = [];
  let player = null;         // AbortController подготовки видео
  let busy = false;
  let queueTotal = 0;
  let nav = 0;               // номер открытия: поздний ответ по прежней тревоге не должен перекрыть новую

  try { idx = await store.cameraIndex(); } catch { /* подписи камер не критичны */ }
  try { reasons = { ...REASON_RU, ...(await store.reasons()) }; } catch { /* справочник по умолчанию */ }
  ctx.cleanup(() => player && player.abort());

  async function loadQueue() {
    const r = await api.get("/alerts", { status: "new", oldest_first: true, limit: QUEUE_LIMIT });
    queue = r.items;
    queueTotal = r.total;
  }

  async function open(id, { updateUrl = true } = {}) {
    const my = ++nav;
    clear(holder).append(skeleton(2, 200));
    try {
      const [a, hs] = await Promise.all([api.get(`/alerts/${id}`), api.get(`/alerts/${id}/reviews`)]);
      if (!ctx.active() || my !== nav) return;
      cur = a;
      hist = hs;
      if (updateUrl) replace(`/review/${id}`);
      paint();
    } catch (e) {
      if (ctx.active() && my === nav) clear(holder).append(errorBox(e.status === 404 ? `Тревога №${id} не найдена` : errText(e), () => open(id)), h("div", { style: { marginTop: "14px" } }, button({ label: "К списку тревог", href: href("/alerts") })));
    }
  }

  const pos = () => queue.findIndex((a) => a.id === cur.id);
  const neighbours = () => {
    const i = pos();
    return { prev: i > 0 ? queue[i - 1] : null, next: i >= 0 ? queue[i + 1] || null : queue[0] || null };
  };

  async function startFresh() {
    try {
      await loadQueue();
      const want = Number(ctx.params.id);
      if (ctx.params.id && Number.isFinite(want)) return open(want, { updateUrl: false });
      if (!queue.length) { cur = null; return paint(); }
      return open(queue[0].id);
    } catch (e) { clear(holder).append(errorBox(errText(e), startFresh)); }
  }

  // ───────── решение ─────────
  async function submit(status, reason) {
    if (busy || !cur || !canAct) return;
    busy = true;
    const note = holder.querySelector("#note") ? holder.querySelector("#note").value.trim() : "";
    const prev = { [cur.id]: cur.status };
    const wasNew = cur.status === "new";
    const { next } = neighbours();
    const done = await decide(cur.id, status, { reason, note, prev, undo: true });
    busy = false;
    if (!done || !ctx.active()) return;
    if (wasNew) queue = queue.filter((a) => a.id !== cur.id);
    bus.emit("alerts:changed", [cur.id]);
    const target = next && next.id !== cur.id ? next : queue[0];
    if (target) open(target.id); else { await loadQueue().catch(() => {}); cur = null; replace("/review"); paint(); }
  }

  const jump = (a) => { if (a) open(a.id); };

  // ───────── отрисовка ─────────
  function mediaBlock(a) {
    const box = h("div", { class: "media" });
    const tag = h("div", { class: "media__tag" }, statusChip(a.status), chip(`уверенность ${pct(a.confidence)}`), a.demo ? chip("демо") : null);
    if (player) player.abort();
    player = new AbortController();
    if (a.has_clip) {
      const status = h("div", { class: "banner banner--info", style: { margin: "12px" } }, "Готовлю видео для просмотра…");
      const video = h("video", { controls: true, loop: true, playsinline: true, preload: "metadata", poster: a.has_thumb ? `/api/v1/alerts/${a.id}/thumb` : null, "aria-label": `Клип тревоги ${a.id}` });
      video.muted = true;
      box.append(video, tag, status);
      const mine = a.id;
      waitForMedia("alert", a.id, { signal: player.signal, onProgress: (p) => { status.textContent = `Готовлю видео для просмотра… ${Math.round(p * 100)} %`; } })
        .then((url) => { if (cur && cur.id === mine) { video.src = url; status.remove(); video.play().catch(() => {}); } })
        .catch((e) => { if (e.name !== "AbortError") { status.className = "banner banner--alarm"; status.textContent = errText(e); } });
      video.addEventListener("error", () => { if (video.src) { status.className = "banner banner--alarm"; status.textContent = "Браузер не смог воспроизвести клип."; box.append(status); } });
    } else {
      const fig = a.has_thumb ? h("img", { src: `/api/v1/alerts/${a.id}/thumb`, alt: "Кадр тревоги" }) : sceneSvg(a.id);
      if (a.has_thumb) fig.addEventListener("error", () => fig.replaceWith(sceneSvg(a.id)));
      box.append(fig, tag, h("div", { class: "banner", style: { margin: "12px" } }, a.has_thumb ? "Клипа для этой тревоги нет, показан кадр." : "Для этой тревоги нет ни клипа, ни кадра."));
    }
    return box;
  }

  function decisionBlock(a) {
    if (!canAct) return card(null, h("p", { class: "muted" }, "Решения принимает оператор. У вас права на просмотр."));
    let reasonRow = null;
    const note = h("textarea", { class: "input", id: "note", rows: 2, placeholder: "Заметка для истории (необязательно)", maxlength: 1000 }, a.note || "");
    const decision = (d, label, ic, key) => h("button", { class: ["decision", a.status === d && "is-on"], "data-d": d, type: "button", onClick: () => d === "false" ? showReasons() : submit(d) }, icon(ic, 20), h("span", { class: "grow" }, label), h("span", { class: "kbd" }, key));
    const box = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Ваше решение"), a.status !== "new" ? button({ label: "Вернуть в очередь", size: "sm", onClick: () => submit("new") }) : null),
      h("div", { class: "stack" }, decision("confirmed", "Курение подтверждено", "check", "Y"), decision("false", "Ложная тревога", "x", "N"), decision("unsure", "Не уверен", "help", "U"), note,
        h("div", { class: "row" }, keyHint("Y / N / U", "решение"), keyHint("J / K", "предыдущая / следующая"))));
    function showReasons() {
      if (reasonRow) return reasonRow.querySelector("button")?.focus();
      reasonRow = h("div", { class: "stack stack--sm" }, h("div", { class: "label" }, "Причина"), h("div", { class: "reasons" },
        ...Object.keys(reasons).map((k) => h("button", { class: "reason", type: "button", onClick: () => submit("false", k) }, REASON_RU[k] || reasons[k])),
        h("button", { class: "reason", type: "button", onClick: () => submit("false", "") }, "Без причины")));
      box.querySelector(".stack").insertBefore(reasonRow, note);
      reasonRow.querySelector("button").focus();
    }
    box._showReasons = showReasons;
    return box;
  }

  function whyBlock(a) {
    return card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Почему сработало")),
      h("div", { class: "why" },
        h("div", null, h("span", { class: "small muted" }, "Уверенность"), h("b", null, pct(a.confidence))),
        h("div", null, h("span", { class: "small muted" }, "Правило"), h("b", { style: { fontSize: "16px", paddingTop: "6px" } }, ruleText(a.rule))),
        h("div", null, h("span", { class: "small muted" }, "Циклов жеста"), h("b", null, a.n_cycles ?? "—")),
        h("div", null, h("span", { class: "small muted" }, "Длительность"), h("b", null, lengthText(a)))),
      a.explain ? h("p", { class: "small muted", style: { marginTop: "12px" } }, a.explain) : null);
  }

  function detailsBlock(a) {
    const line = (k, v) => h("div", { class: "row row--nowrap", style: { justifyContent: "space-between", padding: "6px 0", borderTop: "1px solid var(--line-2)" } }, h("span", { class: "muted" }, k), h("span", { style: { textAlign: "right" } }, v));
    return card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Детали")), line("Камера", alertCam(a, idx)), line("Время", whenText(a)), line("Человек в кадре", a.tid == null ? "—" : `ID ${a.tid}`),
      line("Тревога поднята", a.t_raised ? fmtTime(a.t_raised) : "—"), line("Решил", a.reviewer ? `${a.reviewer}${a.reviewed_at ? `, ${ago(a.reviewed_at)}` : ""}` : "—"));
  }

  function historyBlock() {
    const c = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "История решений")));
    if (!hist.length) { c.append(h("p", { class: "muted small" }, "Решений пока нет.")); return c; }
    [...hist].reverse().forEach((r) => c.append(h("div", { class: "hist" }, statusChip(r.status, false), h("div", { class: "grow small" }, h("b", null, r.reviewer_name || "—"), r.reason ? ` · ${REASON_RU[r.reason] || r.reason}` : "", r.note ? h("div", { class: "muted" }, r.note) : null), h("span", { class: "small muted" }, fmtDateTime(r.created_at)))));
    return c;
  }

  function queueBlock() {
    const i = pos();
    const upcoming = (i >= 0 ? queue.slice(i + 1) : queue).slice(0, 5);
    const c = card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Дальше в очереди"), chip(String(queueTotal), "line")));
    if (!upcoming.length) c.append(h("p", { class: "muted small" }, "Больше ждущих тревог нет."));
    upcoming.forEach((a) => c.append(h("a", { class: "camrow", href: href(`/review/${a.id}`), onClick: (e) => { e.preventDefault(); open(a.id); } }, h("div", { class: "grow" }, h("b", { class: "ellipsis", style: { display: "block" } }, alertCam(a, idx)), h("span", { class: "small muted" }, whenText(a))), ring(a.confidence, { size: 36 }))));
    return c;
  }

  function paint() {
    clear(holder);
    if (!cur) {
      holder.append(pageHead("Проверка", "Очередь тревог на решение."), empty("Очередь пуста", "Все тревоги разобраны. Новые появятся здесь сами.", button({ label: "Все тревоги", href: href("/alerts"), kind: "primary" })));
      return;
    }
    const { prev, next } = neighbours();
    const i = pos();
    const place = i >= 0 ? `${i + 1} из ${queue.length}${queueTotal > queue.length ? "+" : ""}` : "вне очереди";
    const dec = decisionBlock(cur);
    holder._dec = dec;
    const qb = queueBlock();
    holder._queue = qb;
    holder.append(
      pageHead(`Тревога №${cur.id}`, `${alertCam(cur, idx)} · ${whenText(cur)}`,
        h("span", { class: "muted small" }, place),
        button({ ic: "chevl", title: "Предыдущая (K)", disabled: !prev, onClick: () => jump(prev) }), button({ ic: "chevr", title: "Следующая (J)", disabled: !next, onClick: () => jump(next) })),
      h("div", { class: "cols" },
        h("div", { class: "f15 stack" }, mediaBlock(cur), dec),
        h("div", { class: "f1 stack" }, whyBlock(cur), detailsBlock(cur), qb, historyBlock())));
  }

  // ───────── клавиши и живые события ─────────
  const onKey = (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || document.querySelector(".overlay")) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
    const map = { KeyY: () => submit("confirmed"), KeyU: () => submit("unsure"), KeyN: () => holder._dec && holder._dec._showReasons && holder._dec._showReasons(), KeyJ: () => jump(neighbours().next), KeyK: () => jump(neighbours().prev) };
    const fn = map[e.code];
    if (!fn || !cur || (!canAct && /Key[YUN]/.test(e.code))) return;
    e.preventDefault();
    fn();
  };
  document.addEventListener("keydown", onKey);
  ctx.cleanup(() => document.removeEventListener("keydown", onKey));

  ctx.cleanup(bus.on("live:alert", async () => {
    try {
      const empty_ = !cur;
      await loadQueue();
      if (!ctx.active()) return;
      if (empty_ && queue.length) open(queue[0].id);
      else if (holder._queue && holder._queue.isConnected) { const fresh = queueBlock(); holder._queue.replaceWith(fresh); holder._queue = fresh; }
    } catch { /* следующее событие обновит */ }
  }));
  const myName = session.me.display_name || session.me.username;
  ctx.cleanup(bus.on("live:review", async (r) => {
    if (!cur || r.alert_id !== cur.id || busy || r.reviewer === myName) return;        // свои решения экран уже учёл
    const tag = nav, id = cur.id;
    try {
      const [a, hs] = await Promise.all([api.get(`/alerts/${id}`), api.get(`/alerts/${id}/reviews`)]);
      if (!ctx.active() || nav !== tag || !cur || cur.id !== id) return;
      cur = a;
      hist = hs;
      paint();
    } catch { /* следующее событие обновит */ }
  }));

  await startFresh();
}
