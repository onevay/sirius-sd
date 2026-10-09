"use strict";
/* Общее: утилиты, состояние, иконки, вывод видео (любой формат), таймлайн, кольцевые индикаторы */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const view = $("#view");
const S = { cams: [], stats: null, sel: new Set(), maxId: null, hours: 0, timers: [], cleanup: [] };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const J = async (u, o) => { const r = await fetch(u, o); const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.error || r.statusText); return d; };
const POST = (u, b) => J(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
const fmtT = iso => { const d = new Date(iso); return iso && !isNaN(d) ? d.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—"; };
const mmss = t => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; };
const dur = s => s == null ? "—" : s >= 3600 ? `${(s / 3600).toFixed(1)} ч` : s >= 60 ? `${Math.round(s / 60)} мин` : `${Math.round(s)} с`;
const pct = x => x == null ? "—" : Math.round(x * 100) + "%";
const mem = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { } } };
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 3800); }
const operator = $("#operator"); operator.value = mem.get("op") || ""; operator.addEventListener("input", () => mem.set("op", operator.value));
const every = (ms, fn) => S.timers.push(setInterval(fn, ms));
const onKey = fn => { addEventListener("keydown", fn); S.cleanup.push(() => removeEventListener("keydown", fn)); };
const typing = e => /INPUT|TEXTAREA|SELECT/.test(e.target.tagName);
function leave() { S.timers.forEach(clearInterval); S.timers = []; S.cleanup.forEach(f => { try { f(); } catch { } }); S.cleanup = []; }
const camById = id => S.cams.find(c => c.camera_id === id);

/* иконки (stroke 1.8) */
const IC = {
  monitor: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4M6 11h3l2-3 2 5 2-2h3"/></svg>',
  play: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M10 9l5 3-5 3z"/></svg>',
  models: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/><path d="M17.5 3v7M14 6.5h7"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 3L3 10.5l7 2.5 2.5 7z"/><path d="M10 13l5-5"/></svg>',
  pin: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21s7-6.2 7-11.5A7 7 0 005 9.5C5 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/></svg>',
  bell: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M6 17V11a6 6 0 0112 0v6l1.5 2h-15z"/><path d="M10 21a2 2 0 004 0"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></svg>',
  chev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" width="18" height="18"><path d="M9 5l7 7-7 7"/></svg>',
  filter: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16l-6 8v6l-4-2v-4z"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" width="20" height="20"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  pack: '<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><rect x="12" y="14" width="22" height="30" rx="3" fill="#e8e6e0" stroke="#bbb"/><path d="M12 22h22" stroke="#e55"/><rect x="16" y="6" width="3" height="10" fill="#e8e6e0" stroke="#bbb"/><rect x="22" y="4" width="3" height="12" fill="#e8e6e0" stroke="#bbb"/><rect x="28" y="7" width="3" height="9" fill="#e8e6e0" stroke="#bbb"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" width="26" height="26"><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3 2"/></svg>',
  warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 4.5-3.4 7.8-8 9-4.6-1.2-8-4.5-8-9V6z"/><path d="M12 8v5M12 16v.5"/></svg>',
  bolt: '<svg viewBox="0 0 24 24" fill="currentColor" width="26" height="26"><path d="M13 2L5 14h6l-1 8 8-12h-6z"/></svg>',
};
document.querySelectorAll("[data-i]").forEach(e => { e.innerHTML = IC[e.dataset.i] || ""; });

/* ---------------------------------------------------------------- данные */
async function refreshBase() {
  [S.cams, S.stats] = await Promise.all([J(`/api/cameras?hours=${S.hours}`), J("/api/stats")]);
  const b = $("#pendingBadge"); b.textContent = S.stats.new; b.classList.toggle("hidden", !S.stats.new);
  if (S.maxId !== null && S.stats.max_id > S.maxId) toast(`Новых тревог: ${S.stats.max_id - S.maxId}`);
  S.maxId = S.stats.max_id;
  const ds = [...new Set(S.cams.map(c => c.district))].map(d => d.charAt(0).toUpperCase() + d.slice(1));
  $("#subArea").textContent = ds.length ? (ds.length > 3 ? `${ds.slice(0, 3).join(", ")} и ещё ${ds.length - 3}` : ds.join(", ")) : "камер пока нет";
  $("#subDate").textContent = new Date().toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric" });
}
const toggleSel = id => S.sel.has(id) ? S.sel.delete(id) : (S.sel.size < 4 ? S.sel.add(id) : toast("В мультипросмотре не больше 4 камер"));

/* ---------------------------------------------------------------- вывод видео: формат выбирает браузер, источник — файл или клип тревоги, сбой не молчит */
const FMT = (() => { const v = document.createElement("video"); return v.canPlayType('video/mp4; codecs="avc1.42E01E"') ? "h264" : (v.canPlayType('video/webm; codecs="vp9"') ? "vp9" : "h264"); })();
const srcQuery = s => s.alert ? `alert=${s.alert}` : `id=${s.id}`;
const srcUrl = (s, f) => s.alert ? `/media/clip/${s.alert}?fmt=${f}` : `/media/video/${s.id}?fmt=${f}`;
async function loadVideo(video, src, note, fmt = FMT, t0 = 0) {
  if (typeof src === "string") src = { id: src };
  const say = (txt, bad) => { if (note) { note.textContent = txt; note.classList.toggle("err", !!bad); } };
  video.removeAttribute("src"); video.load(); video.onerror = null;
  for (let i = 0; i < 3600; i++) {
    if (!document.body.contains(video)) return false;
    let st; try { st = await J(`/api/media-status?${srcQuery(src)}&fmt=${fmt}`); } catch (e) { say("Видео недоступно: " + e.message, true); return false; }
    if (st.state === "direct" || st.state === "ready") break;
    if (st.state === "error") { say("Не удалось подготовить видео: " + st.error, true); return false; }
    say(`Подготовка видео для браузера… ${Math.round(st.pct * 100)}%`);
    await new Promise(r => setTimeout(r, 700));
  }
  return new Promise(res => {
    video.onloadedmetadata = () => { say(""); if (t0) video.currentTime = t0; res(true); };
    video.onerror = async () => {      // кодек не принят браузером: один раз пробуем второй формат
      video.onerror = null;
      if (!video.dataset.retried) { video.dataset.retried = "1"; say("Пробую другой формат…"); res(await loadVideo(video, src, note, fmt === "h264" ? "vp9" : "h264", t0)); }
      else { say("Браузер не воспроизводит это видео ни в одном формате (код " + (video.error ? video.error.code : "?") + ")", true); res(false); }
    };
    video.src = srcUrl(src, fmt);
  });
}
function timeline(el, total, items, onSeek) {
  el.innerHTML = items.map(it => `<i class="${it.cls}" style="left:${(it.a / total) * 100}%;width:${Math.max(0.5, ((it.b - it.a) / total) * 100)}%" title="${esc(it.tip || "")}"></i>`).join("") + `<div class="cursor"></div>`;
  el.onclick = e => { const r = el.getBoundingClientRect(); onSeek(Math.max(0, Math.min(total, ((e.clientX - r.left) / r.width) * total))); };
  return t => { const c = el.querySelector(".cursor"); if (c) c.style.left = Math.min(100, (t / total) * 100) + "%"; };
}

/* кольцевой индикатор (как в референсе): дуга 270°, значение 0..1 */
function gaugeSvg(v, color = "#f6b44a") {
  const R = 78, C = 2 * Math.PI * R, arc = C * 0.75, val = Math.max(0, Math.min(1, v || 0));
  return `<svg viewBox="0 0 190 190"><defs><linearGradient id="gg${Math.round(val * 100)}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${color}"/><stop offset="1" stop-color="#7a5418"/></linearGradient></defs>
    <circle cx="95" cy="95" r="${R}" fill="none" stroke="#3a3a3d" stroke-width="12" stroke-linecap="round" stroke-dasharray="${arc} ${C}" transform="rotate(135 95 95)"/>
    <circle cx="95" cy="95" r="${R}" fill="none" stroke="url(#gg${Math.round(val * 100)})" stroke-width="12" stroke-linecap="round" stroke-dasharray="${arc * val} ${C}" transform="rotate(135 95 95)"/></svg>`;
}
