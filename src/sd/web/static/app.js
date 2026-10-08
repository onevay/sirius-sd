"use strict";
/* Мониторинг камер. Три экрана: #/map (карта и камеры), #/alerts (тревоги и журнал решений), #/analysis (разбор видео моделью). Мультипросмотр: #/multi?cams=a,b */
const $ = (s, r = document) => r.querySelector(s);
const view = $("#view");
const S = { cams: [], stats: null, sel: new Set(), maxId: null, timers: [], cleanup: [] };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const J = async (u, o) => { const r = await fetch(u, o); const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.error || r.statusText); return d; };
const POST = (u, b) => J(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
const fmtT = iso => { const d = new Date(iso); return iso && !isNaN(d) ? d.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—"; };
const mmss = t => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; };
const mem = { get: k => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { } } };
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 3500); }
const operator = $("#operator"); operator.value = mem.get("op") || ""; operator.addEventListener("input", () => mem.set("op", operator.value));

/* ---------------------------------------------------------------- общее */
async function refreshBase() {
  [S.cams, S.stats] = await Promise.all([J("/api/cameras"), J("/api/stats")]);
  const b = $("#pendingBadge"); b.textContent = S.stats.new; b.classList.toggle("hidden", !S.stats.new);
  if (S.maxId !== null && S.stats.max_id > S.maxId) toast(`Новых тревог: ${S.stats.max_id - S.maxId}`);
  S.maxId = S.stats.max_id;
}
const camById = id => S.cams.find(c => c.camera_id === id);
const every = (ms, fn) => S.timers.push(setInterval(fn, ms));
const onKey = fn => { addEventListener("keydown", fn); S.cleanup.push(() => removeEventListener("keydown", fn)); };
const typing = e => /INPUT|TEXTAREA|SELECT/.test(e.target.tagName);
function leave() { S.timers.forEach(clearInterval); S.timers = []; S.cleanup.forEach(f => { try { f(); } catch { } }); S.cleanup = []; }

const routes = { map: vMap, multi: vMulti, alerts: vAlerts, analysis: vAnalysis };
async function route() {
  leave();
  let [path, qs] = (location.hash.slice(2) || "map").split("?");
  if (path === "video") path = "analysis";
  const name = routes[path] ? path : "map";
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", a.dataset.r === name || (name === "multi" && a.dataset.r === "map")));
  try { await refreshBase(); await routes[name](Object.fromEntries(new URLSearchParams(qs || ""))); }
  catch (e) { view.innerHTML = `<div class="card pad err">Ошибка: ${esc(e.message)}</div>`; }
}
addEventListener("hashchange", route);

/* ---------------------------------------------------------------- вывод видео: формат выбирает браузер, сбой не молчит */
const FMT = (() => { const v = document.createElement("video"); return v.canPlayType('video/mp4; codecs="avc1.42E01E"') ? "h264" : (v.canPlayType('video/webm; codecs="vp9"') ? "vp9" : "h264"); })();
async function loadVideo(video, vid, note, fmt = FMT, t0 = 0) {
  const say = (txt, bad) => { if (note) { note.textContent = txt; note.classList.toggle("err", !!bad); } };
  video.removeAttribute("src"); video.load(); video.onerror = null;
  for (let i = 0; i < 1800; i++) {
    if (!document.body.contains(video)) return false;
    let st; try { st = await J(`/api/media-status?id=${vid}&fmt=${fmt}`); } catch (e) { say("Видео недоступно: " + e.message, true); return false; }
    if (st.state === "direct" || st.state === "ready") break;
    if (st.state === "error") { say("Не удалось подготовить видео: " + st.error, true); return false; }
    say(`Подготовка видео для браузера… ${Math.round(st.pct * 100)}%`);
    await new Promise(r => setTimeout(r, 700));
  }
  return new Promise(res => {
    video.onloadedmetadata = () => { say(""); if (t0) video.currentTime = t0; res(true); };
    video.onerror = async () => {      // кодек не поддержан браузером: пробуем второй формат один раз
      const other = fmt === "h264" ? "vp9" : "h264"; video.onerror = null;
      if (!video.dataset.retried) { video.dataset.retried = "1"; say("Пробую другой формат…"); res(await loadVideo(video, vid, note, other, t0)); }
      else { say("Браузер не воспроизводит это видео ни в одном формате (код " + (video.error ? video.error.code : "?") + ")", true); res(false); }
    };
    video.src = `/media/video/${vid}?fmt=${fmt}`;
  });
}
function timeline(el, dur, items, onSeek) {
  el.innerHTML = items.map(it => `<i class="${it.cls}" style="left:${(it.a / dur) * 100}%;width:${Math.max(0.5, ((it.b - it.a) / dur) * 100)}%" title="${esc(it.tip || "")}"></i>`).join("") + `<div class="cursor"></div>`;
  el.onclick = e => { const r = el.getBoundingClientRect(); onSeek(Math.max(0, Math.min(dur, ((e.clientX - r.left) / r.width) * dur))); };
  return t => { const c = el.querySelector(".cursor"); if (c) c.style.left = Math.min(100, (t / dur) * 100) + "%"; };
}

/* ---------------------------------------------------------------- карта и камеры */
const ledClass = c => c.pending ? "alert" : (c.n_chunks || c.alerts ? "ok" : "");
function loadLeaflet() {
  return new Promise(res => {
    if (window.L) return res(true);
    const css = document.createElement("link"); css.rel = "stylesheet"; css.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"; document.head.appendChild(css);
    const s = document.createElement("script"); s.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
    const to = setTimeout(() => res(false), 2500);
    s.onload = () => { clearTimeout(to); res(!!window.L); }; s.onerror = () => { clearTimeout(to); res(false); };
    document.head.appendChild(s);
  });
}
const toggleSel = id => S.sel.has(id) ? S.sel.delete(id) : (S.sel.size < 4 ? S.sel.add(id) : toast("В мультипросмотре не больше 4 камер"));
async function vMap() {
  const k = S.stats;
  view.innerHTML = `<div class="kpis">${[["Ждут решения", k.new], ["Подтверждено", k.confirmed], ["Ложных", k.false], ["Камер", S.cams.length], ["Точность тревог", k.precision == null ? "—" : Math.round(k.precision * 100) + "%"]]
    .map(([a, b]) => `<div class="card kpi"><span class="mut">${a}</span><b>${b}</b></div>`).join("")}</div>
  <div class="mapwrap"><div class="card mapbox"><div id="map"></div><div class="note" id="mapnote"></div></div><div class="card side" id="side"></div></div>
  <div class="card selbar"><span id="selinfo" class="mut"></span><div class="grow"></div><button id="clear">Сбросить</button><button class="primary" id="openMulti">Смотреть выбранные</button></div>`;
  const draw = () => {
    $("#side").innerHTML = (S.cams.length ? "" : `<div class="mut pad">Камер нет. Положите папки <code>район-индекс-время</code> с видео в каталог потоков (SD_STREAMS).</div>`) +
      S.cams.map(c => `<div class="camrow ${S.sel.has(c.camera_id) ? "sel" : ""}" data-id="${esc(c.camera_id)}"><span class="led ${ledClass(c)}"></span>
      <div class="grow"><b>${esc(c.street)}</b><div class="mut">${esc(c.district)} · камера ${esc(c.index)} · ${c.n_chunks} фрагм.${c.placed ? "" : " · схема"}</div></div>${c.pending ? `<span class="badge">${c.pending}</span>` : ""}</div>`).join("");
    document.querySelectorAll(".camrow").forEach(r => r.onclick = () => { toggleSel(r.dataset.id); draw(); markSel(); });
    $("#selinfo").textContent = S.sel.size ? `Выбрано: ${[...S.sel].map(i => camById(i)?.street || i).join(", ")}` : "Выберите до 4 камер на карте или в списке для одновременного просмотра";
    $("#openMulti").disabled = !S.sel.size;
  };
  let markSel = () => { };
  $("#clear").onclick = () => { S.sel.clear(); draw(); markSel(); };
  $("#openMulti").onclick = () => { location.hash = "#/multi?cams=" + [...S.sel].map(encodeURIComponent).join(","); };
  const ok = S.cams.length ? await loadLeaflet() : false;
  markSel = ok ? leafletMap(draw) : svgMap(draw, S.cams.length ? "Схема: нет доступа к плиткам OpenStreetMap" : "");
  draw(); markSel();
  every(5000, async () => { try { await refreshBase(); draw(); markSel(); } catch { } });
}
function leafletMap(redraw) {
  const m = L.map("map"); L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(m);
  const markers = {};
  const sync = () => S.cams.forEach(c => {
    const icon = L.divIcon({ html: `<div class="beacon ${c.pending ? "alert" : (c.n_chunks || c.alerts ? "" : "none")} ${S.sel.has(c.camera_id) ? "sel" : ""}"></div>`, className: "", iconSize: [18, 18] });
    if (!markers[c.camera_id]) markers[c.camera_id] = L.marker([c.lat, c.lon], { icon }).addTo(m).on("click", () => { toggleSel(c.camera_id); redraw(); sync(); });
    else markers[c.camera_id].setIcon(icon);
    markers[c.camera_id].bindTooltip(`<b>${esc(c.street)}</b><br>${esc(c.title)}<br>${c.pending ? `Ждут решения: ${c.pending}` : "Тревог нет"}`);
  });
  m.fitBounds(S.cams.map(c => [c.lat, c.lon]), { padding: [40, 40], maxZoom: 16 });
  S.cleanup.push(() => m.remove()); sync(); return sync;
}
function svgMap(redraw, note) {
  $("#mapnote").textContent = note;
  const el = $("#map"), W = () => el.clientWidth || 800, H = () => el.clientHeight || 500;
  const lats = S.cams.map(c => c.lat), lons = S.cams.map(c => c.lon);
  const [a0, a1] = [Math.min(...lats, 59.5), Math.max(...lats, 59.51)], [o0, o1] = [Math.min(...lons, 30.1), Math.max(...lons, 30.11)];
  let pos = {};
  const layout = () => {      // проекция + раздвижка: маячки не слипаются
    const w = W(), h = H(), p = 50, P = S.cams.map(c => [p + (c.lon - o0) / ((o1 - o0) || 1) * (w - 2 * p), h - p - (c.lat - a0) / ((a1 - a0) || 1) * (h - 2 * p)]);
    for (let it = 0; it < 60; it++) for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      let dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], d = Math.hypot(dx, dy); if (d === 0) { dx = 1; dy = 0; d = 1; }
      if (d < 54) { const k = (54 - d) / 2 / d; P[i][0] -= dx * k; P[i][1] -= dy * k; P[j][0] += dx * k; P[j][1] += dy * k; }
    }
    pos = {}; S.cams.forEach((c, i) => { pos[c.camera_id] = [Math.min(w - 20, Math.max(20, P[i][0])), Math.min(h - 20, Math.max(20, P[i][1]))]; });
  };
  const sync = () => {
    layout(); let g = "";
    S.cams.forEach((c, i) => { const [x, y] = pos[c.camera_id], hor = i % 2 === 0, [x1, y1, x2, y2] = hor ? [x - 70, y, x + 70, y] : [x, y - 50, x, y + 50];
      g += `<line class="street" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><line class="street2" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><text x="${hor ? x - 66 : x + 12}" y="${hor ? y - 12 : y - 38}">${esc(c.street)}</text>`; });
    S.cams.forEach(c => { const [x, y] = pos[c.camera_id], col = c.pending ? "var(--red)" : (c.n_chunks || c.alerts ? "var(--green)" : "#8a94a3");
      g += `<g class="pin" data-id="${esc(c.camera_id)}"><title>${esc(c.title)}</title>${c.pending ? `<circle cx="${x}" cy="${y}" r="16" fill="var(--red)" opacity=".25"><animate attributeName="r" values="10;22;10" dur="1.8s" repeatCount="indefinite"/></circle>` : ""}
      <circle cx="${x}" cy="${y}" r="9" fill="${col}" stroke="#fff" stroke-width="3"/>${S.sel.has(c.camera_id) ? `<circle cx="${x}" cy="${y}" r="14" fill="none" stroke="var(--brand)" stroke-width="3"/>` : ""}</g>`; });
    el.innerHTML = `<svg class="svgmap" width="${W()}" height="${H()}">${g}</svg>`;
    el.querySelectorAll(".pin").forEach(p => p.onclick = () => { toggleSel(p.dataset.id); redraw(); sync(); });
  };
  const ro = new ResizeObserver(sync); ro.observe(el); S.cleanup.push(() => ro.disconnect()); sync(); return sync;
}

/* ---------------------------------------------------------------- мультипросмотр: непрерывное воспроизведение фрагментов камеры */
async function vMulti(q) {
  const ids = (q.cams || "").split(",").filter(Boolean).map(decodeURIComponent).slice(0, 4);
  if (!ids.length) { location.hash = "#/map"; return; }
  const cams = await Promise.all(ids.map(i => J("/api/camera/" + encodeURIComponent(i))));
  view.innerHTML = `<div class="row" style="margin-bottom:10px"><a href="#/map"><button>← К карте</button></a><h2 style="margin:0">Камеры: ${cams.length}</h2><div class="grow"></div><button id="playAll">▶ Все</button><button id="pauseAll">⏸ Все</button></div><div class="grid g${cams.length}" id="grid"></div>`;
  const vids = [];
  cams.forEach((c, ci) => {
    const t = document.createElement("div"); t.className = "card tile";
    t.innerHTML = `<header><span class="led ${ledClass(c)}"></span><b>${esc(c.street)}</b><span class="mut">${esc(c.district)} ${esc(c.index)}</span><div class="grow"></div>
      ${c.chunks.length ? `<select>${c.chunks.map((ch, i) => `<option value="${i}">${esc(ch.name)}${ch.duration ? " · " + mmss(ch.duration) : ""}</option>`).join("")}</select>` : ""}${c.pending ? `<span class="badge">${c.pending}</span>` : ""}</header>
      <div class="vwrap">${c.chunks.length ? `<video controls muted playsinline preload="metadata"></video>` : `<div class="pad mut" style="line-height:1.4">Видео нет</div>`}</div><div class="timeline"></div>
      <div class="pad row"><span class="mut grow" data-st></span>${c.chunks.length ? `<a data-an href="#/analysis"><button>Разобрать моделью</button></a>` : ""}</div>`;
    $("#grid").appendChild(t);
    if (!c.chunks.length) return;
    const v = $("video", t), sel = $("select", t), note = $("[data-st]", t); vids.push(v);
    const load = async (i, t0 = 0, play = false) => {
      const ch = c.chunks[i]; sel.value = i; $("[data-an]", t).href = `#/analysis?vid=${ch.id}&auto=1`;
      const off = c.chunks.slice(0, i).reduce((s, x) => s + (x.duration || 0), 0), dur = ch.duration || 1;
      const al = await J(`/api/alerts?camera=${encodeURIComponent(c.camera_id)}&limit=200`);
      const upd = timeline($(".timeline", t), dur, al.filter(a => a.start_sec >= off && a.start_sec <= off + dur).map(a => ({ cls: "alert", a: a.start_sec - off, b: Math.max(a.end_sec, a.start_sec + 1) - off, tip: `${a.explain} ${Math.round(a.confidence * 100)}%` })), s => { v.currentTime = s; });
      v.ontimeupdate = () => upd(v.currentTime);
      v.onended = () => { if (i + 1 < c.chunks.length) load(i + 1, 0, true); };
      v.dataset.retried = ""; if (await loadVideo(v, ch.id, note, FMT, t0) && play) v.play().catch(() => { });
    };
    sel.onchange = () => load(+sel.value);
    const first = ci === 0 && q.chunk ? Math.max(0, c.chunks.findIndex(x => x.name === q.chunk)) : 0;
    load(first, ci === 0 && q.t ? +q.t : 0);
  });
  $("#playAll").onclick = () => vids.forEach(v => v.play().catch(() => { }));
  $("#pauseAll").onclick = () => vids.forEach(v => v.pause());
  onKey(e => { if (typing(e) || e.code !== "Space") return; e.preventDefault(); const any = vids.some(v => !v.paused); vids.forEach(v => any ? v.pause() : v.play().catch(() => { })); });
  every(5000, async () => { try { await refreshBase(); } catch { } });
}

/* ---------------------------------------------------------------- тревоги и журнал решений */
const STATUS = { new: "Ждут решения", confirmed: "Подтверждены", false: "Ложные", unsure: "Не уверен" };
async function vAlerts(q) {
  let status = q.status ? q.status.split(",") : ["new"], cur = q.id ? +q.id : null, items = [];
  view.innerHTML = `<div class="row" style="margin-bottom:10px"><div id="chips" class="row"></div><div class="grow"></div><select id="fcam"><option value="">Все камеры</option></select><a href="/api/journal.csv"><button>Журнал CSV</button></a></div>
    <div class="split"><div class="card" id="list" style="max-height:calc(100vh - 150px);overflow:auto"></div><div id="detail"></div></div>`;
  $("#fcam").innerHTML += S.cams.map(c => `<option value="${esc(c.camera_id)}">${esc(c.street)} (${esc(c.camera_id)})</option>`).join("");
  const load = async () => {
    const cam = $("#fcam").value;
    items = await J(`/api/alerts?limit=100&status=${status.join(",")}${cam ? "&camera=" + encodeURIComponent(cam) : ""}`);
    $("#chips").innerHTML = Object.entries(STATUS).map(([k, v]) => `<button data-s="${k}" class="${status.includes(k) ? "primary" : ""}">${v}</button>`).join("");
    $("#chips").querySelectorAll("button").forEach(b => b.onclick = () => { const s = b.dataset.s; status = status.includes(s) ? status.filter(x => x !== s) : [...status, s]; load(); });
    $("#list").innerHTML = items.length ? items.map(a => `<div class="acard ${a.id === cur ? "on" : ""}" data-id="${a.id}">${a.has_thumb ? `<img loading="lazy" src="/media/alert/${a.id}/thumb">` : `<img alt="">`}
      <div class="grow"><b>${esc(camById(a.camera_id)?.street || a.district)}</b> <span class="pill ${a.status}">${a.status_ru}</span>${a.demo ? ' <span class="pill">ДЕМО</span>' : ""}
      <div class="mut">${fmtT(a.t_abs)} · ${esc(a.explain)}</div><div class="bar"><i style="width:${Math.round(a.confidence * 100)}%"></i></div></div></div>`).join("")
      : `<div class="pad mut">Тревог по выбранным фильтрам нет.${S.stats.total ? "" : `<br><br>Тревоги появятся, когда работает <code>sd monitor</code>. Для проверки интерфейса: <button id="demo">создать демо-тревоги</button>`}</div>`;
    const d = $("#demo"); if (d) d.onclick = async () => { toast(`Создано: ${(await POST("/api/demo")).n}`); route(); };
    $("#list").querySelectorAll(".acard").forEach(e => e.onclick = () => { cur = +e.dataset.id; show(); $("#list").querySelectorAll(".acard").forEach(x => x.classList.toggle("on", x === e)); });
  };
  const decide = async (id, st) => {
    await POST(`/api/alert/${id}/review`, { status: st, reviewer: operator.value, note: ($("#note") || {}).value || "" });
    await refreshBase(); await load();
    const nxt = items.find(a => a.status === "new"); cur = nxt ? nxt.id : null; show();
  };
  const show = async () => {
    const d = $("#detail"); if (!cur) { d.innerHTML = `<div class="card pad mut">Выберите тревогу слева.</div>`; return; }
    const a = await J("/api/alert/" + cur), c = camById(a.camera_id);
    const t0 = Math.max(0, a.start_sec - (a.chunk_offset || 0) - 3);
    d.innerHTML = `<div class="card pad"><div class="row"><h2 style="margin:0">Тревога №${a.id}</h2><span class="pill ${a.status}">${a.status_ru}</span>${a.demo ? '<span class="pill">ДЕМО</span>' : ""}</div>
      <p class="mut">${esc(c?.street || a.district)} · ${esc(a.district)}, камера ${esc(a.cam_index)} · ${fmtT(a.t_abs)}</p>
      <div class="vwrap">${a.has_clip ? `<video controls autoplay muted loop src="/media/alert/${a.id}/clip" style="width:100%"></video>` : (a.has_thumb ? `<img src="/media/alert/${a.id}/thumb" style="width:100%">` : `<div class="pad mut">Клип недоступен</div>`)}</div>
      <div class="row" style="margin:10px 0"><div><span class="mut">Уверенность</span><br><b style="font-size:22px">${Math.round(a.confidence * 100)}%</b></div><div><span class="mut">Что сработало</span><br>${esc(a.explain)}</div><div><span class="mut">ID трека</span><br>${a.tid}</div></div>
      <textarea id="note" rows="2" placeholder="Заметка (необязательно)" style="width:100%">${esc(a.note || "")}</textarea>
      <div class="row" style="margin-top:10px"><button class="ok" data-d="confirmed">Подтвердить (Y)</button><button class="bad" data-d="false">Ложная (N)</button><button data-d="unsure">Не уверен (U)</button><div class="grow"></div>
      ${c && c.folder && a.chunk ? `<a href="#/multi?cams=${encodeURIComponent(a.camera_id)}&chunk=${encodeURIComponent(a.chunk)}&t=${t0}"><button>Открыть в камере</button></a>` : ""}</div>
      ${a.reviewed_at ? `<p class="mut">Решение: ${a.status_ru} · ${esc(a.reviewer || "оператор")} · ${fmtT(a.reviewed_at)}</p>` : ""}</div>`;
    d.querySelectorAll("[data-d]").forEach(b => b.onclick = () => decide(a.id, b.dataset.d));
  };
  onKey(e => { if (typing(e) || !cur) return; const m = { y: "confirmed", n: "false", u: "unsure" }[e.key.toLowerCase()]; if (m) decide(cur, m); });
  $("#fcam").onchange = load; await load(); await show();
  every(5000, async () => { try { await refreshBase(); if (!document.activeElement || document.activeElement.id !== "note") await load(); } catch { } });
}

/* ---------------------------------------------------------------- разбор видео моделью */
async function vAnalysis(q) {
  const [folders, profiles] = await Promise.all([J("/api/folders"), J("/api/profiles")]);
  if (!folders.length) { view.innerHTML = `<div class="card pad mut">Видео нет. Положите папки с видео в каталог потоков (SD_STREAMS) или данных (SD_DATA).</div>`; return; }
  const ready = profiles.filter(p => p.ready), remembered = mem.get("an_profile");
  view.innerHTML = `<div class="an">
    <aside class="card pad"><select id="folder">${folders.map(f => `<option value="${f.id}">${esc(f.name)} (${f.n})</option>`).join("")}</select>
      <div id="vlist" class="vlist"></div>
      <select id="prof">${profiles.map(p => `<option value="${esc(p.name)}">${p.ready ? "" : "⚠ "}${esc(p.name)}${p.kind === "solver" ? " · решатель" : ""}</option>`).join("") || "<option value=''>нет профилей</option>"}</select>
      <div id="pinfo" class="mut small"></div>
      <label class="row small" style="gap:6px"><input type="checkbox" id="live" checked> В реальном времени: видео идёт вместе с моделью, рамки и тревоги сразу</label>
      <button class="primary" id="run">▶ Смотреть с анализом</button><button id="runAll" title="Быстрый анализ всех видео папки без просмотра">Всю папку (быстро)</button><div class="prog hidden" id="prog"><i></i></div><div id="jstat" class="mut small"></div></aside>
    <section><div class="card"><div class="vwrap" id="pw"><video id="pv" controls muted playsinline></video><canvas id="ov"></canvas></div><div class="timeline" id="tl"></div>
      <div class="toolbar"><button id="prevA" title="[">◀ тревога</button><button id="nextA" title="]">тревога ▶</button>
        <select id="rate"><option value="0.5">×0.5</option><option value="1" selected>×1</option><option value="2">×2</option><option value="4">×4</option></select>
        <label><input type="checkbox" id="tBox" checked> рамки</label><label><input type="checkbox" id="tGt" checked> разметка</label><span class="grow mut small" id="pst"></span></div></div>
      <div class="card pad hidden" id="res" style="margin-top:12px"></div><div class="card pad hidden" id="batch" style="margin-top:12px"></div></section></div>`;
  let cur = null, trace = null, meta = null, dur = 1, upd = () => { }, vs = [], liveJob = null;
  S.cleanup.push(() => { liveJob = null; });
  const pv = $("#pv"), note = $("#pst");
  // профиль: ранее выбранный, иначе первый готовый
  const pick0 = profiles.find(p => p.name === remembered && p.ready) || ready[0] || profiles[0]; if (pick0) $("#prof").value = pick0.name;
  const pinfo = () => { const p = profiles.find(x => x.name === $("#prof").value); mem.set("an_profile", $("#prof").value);
    $("#pinfo").innerHTML = !p ? "" : p.ready ? esc(Object.entries(p.describe).filter(([k]) => ["pose", "imgsz", "cycle_model", "fps"].includes(k)).map(([k, v]) => `${k}: ${v}`).join(" · ")) : `<span class="err">${p.problems.map(esc).join("<br>")}</span>`;
    $("#run").disabled = !p || !p.ready || !cur; $("#runAll").disabled = !p || !p.ready; };
  $("#prof").onchange = pinfo;
  const loadList = async () => {
    const fid = $("#folder").value; mem.set("an_folder", fid); vs = await J("/api/videos?folder=" + fid);
    $("#vlist").innerHTML = vs.map(v => { const a = v.analysis; return `<div class="camrow ${cur && cur.id === v.id ? "sel" : ""}" data-id="${v.id}"><div class="grow"><b>${esc(v.name)}</b>
      <div class="mut small">${v.duration ? mmss(v.duration) : "—"}${a ? ` · <span class="${a.alerts ? "err" : ""}">тревог: ${a.alerts ?? "?"}</span>${a.f1 != null ? ` · F1 ${a.f1.toFixed(2)}` : ""}` : ""}</div></div></div>`; }).join("") || `<div class="mut">Видео нет</div>`;
    $("#vlist").querySelectorAll(".camrow").forEach(r => r.onclick = () => select(vs.find(v => v.id === r.dataset.id)));
  };
  const select = async v => {
    liveJob = null; cur = v; trace = null; meta = null; $("#res").classList.add("hidden"); $("#jstat").textContent = ""; dur = v.duration || 1; history.replaceState(null, "", `#/analysis?vid=${v.id}`);
    $("#vlist").querySelectorAll(".camrow").forEach(r => r.classList.toggle("sel", r.dataset.id === v.id));
    upd = timeline($("#tl"), dur, [], s => { pv.currentTime = s; }); note.textContent = ""; pinfo();
    pv.dataset.retried = ""; await loadVideo(pv, v.id, note);
    const a = await J("/api/analysis?video=" + v.id); if (a.trace) setAnalysis(a.trace, a.meta); else note.textContent = note.textContent || "Разбора ещё нет — нажмите «Анализировать».";
  };
  const drawTimeline = t => { upd = timeline($("#tl"), dur, [...(t.gt || []).map(g => ({ cls: "gt", a: g.start, b: g.end, tip: "разметка: курение" })), ...(t.alerts || []).map(a => ({ cls: "alert", a: a.start, b: Math.max(a.end, a.start + 1), tip: `${a.explain} ${Math.round(a.confidence * 100)}%` }))], s => { pv.currentTime = s; }); };
  const setAnalysis = (t, m) => {
    trace = t; meta = m; drawTimeline(t);
    const s = (m && m.stats) || {}, mt = (m && m.metrics) || null, al = t.alerts || [];
    note.textContent = `профиль ${m ? m.profile : "—"}`;
    const kp = [["Тревог", al.length], ["Скорость", s.rt_factor != null ? `×${s.rt_factor}` : "—"], ...(mt ? [["F1", mt.f1.toFixed(2)], ["TP / FP / FN", `${mt.tp}/${mt.fp}/${mt.fn}`], ["Задержка", mt.alert_delay_median == null ? "—" : mt.alert_delay_median.toFixed(1) + " с"]] : [])];
    $("#res").classList.remove("hidden");
    $("#res").innerHTML = `<div class="row">${kp.map(([a, b]) => `<div><span class="mut small">${a}</span><br><b style="font-size:20px">${b}</b></div>`).join("")}<div class="grow"></div><span class="${s.rt_factor >= 1 ? "mut" : "err"} small">${esc(s.verdict || "")}</span></div>
      ${al.length ? `<table><tr><th>Тревога в</th><th>Начало</th><th>ID</th><th>Причина</th><th>Увер.</th></tr>${al.map(a => `<tr class="jump" data-t="${a.start}"><td>${mmss(a.t_open)}</td><td>${mmss(a.start)}</td><td>${a.tid}</td><td>${esc(a.explain)}</td><td>${Math.round(a.confidence * 100)}%</td></tr>`).join("")}</table>` : `<p class="mut">Тревог нет.</p>`}`;
    $("#res").querySelectorAll("tr.jump").forEach(r => r.onclick = () => seekAlert(+r.dataset.t));
  };
  const seekAlert = t => { pv.currentTime = Math.max(0, t - 1); pv.play().catch(() => { }); };
  const jump = dir => { const al = (trace && trace.alerts) || []; if (!al.length) return toast("Тревог нет"); const c = pv.currentTime;      // переход ставит просмотр за 1 с до начала тревоги
    const nx = dir > 0 ? al.find(a => a.start - 1 > c + 0.3) : [...al].reverse().find(a => a.start - 1 < c - 1.0);
    if (nx) seekAlert(nx.start); else toast(dir > 0 ? "Больше тревог нет" : "Раньше тревог нет"); };
  $("#nextA").onclick = () => jump(1); $("#prevA").onclick = () => jump(-1);
  $("#rate").onchange = () => { pv.playbackRate = +$("#rate").value; };
  onKey(e => { if (typing(e)) return; if (e.code === "Space") { e.preventDefault(); pv.paused ? pv.play().catch(() => { }) : pv.pause(); }
    else if (e.key === "ArrowRight") pv.currentTime += 5; else if (e.key === "ArrowLeft") pv.currentTime -= 5; else if (e.key === "]") jump(1); else if (e.key === "[") jump(-1); else if (e.key.toLowerCase() === "b") { $("#tBox").click(); } });
  /* оверлей: рамки людей по времени кадра; у тех, по кому есть тревога, красные */
  const cv = $("#ov");
  const paint = () => {
    const w = pv.clientWidth, h = pv.clientHeight; if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d"); g.clearRect(0, 0, w, h); upd(pv.currentTime);
    if (liveJob && liveJob.started) {        // видео не убегает от модели: ждём, пока она посчитает до текущей секунды
      const lag = pv.currentTime - liveJob.horizon;
      if (!pv.paused && lag > 0.25) { pv.pause(); liveJob.waiting = true; }
      else if (liveJob.waiting && liveJob.horizon > pv.currentTime + 1.0) { liveJob.waiting = false; pv.play().catch(() => { }); }
      if (liveJob.waiting) { g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(0, h / 2 - 16, w, 32); g.fillStyle = "#fff"; g.font = "14px system-ui"; g.fillText(`Модель отстаёт на ${Math.max(0, lag).toFixed(1)} с — ждём…`, 12, h / 2 + 5); }
    }
    if (!trace || !trace.frames.length) return;
    const t = pv.currentTime, live = (trace.alerts || []).filter(a => t >= a.start && t <= a.end + 1);
    if (live.some(a => t >= a.t_open - 1e-3)) { g.fillStyle = "rgba(217,45,32,.92)"; g.fillRect(0, 0, w, 26); g.fillStyle = "#fff"; g.font = "bold 14px system-ui"; g.fillText("ТРЕВОГА: курение", 10, 18); }
    if (($("#tGt").checked) && (trace.gt || []).some(x => t >= x.start && t <= x.end)) { g.fillStyle = "rgba(18,128,60,.9)"; g.fillRect(0, h - 22, 150, 22); g.fillStyle = "#fff"; g.font = "12px system-ui"; g.fillText("разметка: курение", 8, h - 7); }
    if (!$("#tBox").checked) return;
    const fr = trace.frames; let lo = 0, hi = fr.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; fr[m][0] < t ? lo = m + 1 : hi = m; }
    let k = lo; if (k > 0 && Math.abs(fr[k - 1][0] - t) < Math.abs(fr[k][0] - t)) k--;
    if (Math.abs(fr[k][0] - t) > 2.5 / (trace.fps || 5)) return;
    const [fh, fw] = trace.frame_hw || [pv.videoHeight, pv.videoWidth], sx = w / (fw || pv.videoWidth || 1), sy = h / (fh || pv.videoHeight || 1);
    for (const [tid, x1, y1, x2, y2] of fr[k][1]) {
      const al = live.find(a => a.tid === tid), hot = !!al;
      g.lineWidth = hot ? 3 : 1.5; g.strokeStyle = hot ? "#ff3b30" : "rgba(255,255,255,.85)"; g.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      const lbl = hot ? `ID ${tid} · ${Math.round(al.confidence * 100)}%` : `ID ${tid}`; g.font = "12px system-ui"; const tw = g.measureText(lbl).width + 8;
      g.fillStyle = hot ? "#ff3b30" : "rgba(0,0,0,.55)"; g.fillRect(x1 * sx, Math.max(0, y1 * sy - 18), tw, 18); g.fillStyle = "#fff"; g.fillText(lbl, x1 * sx + 4, Math.max(13, y1 * sy - 5));
    }
  };
  let raf = 0; const loop = () => { paint(); raf = requestAnimationFrame(loop); }; loop(); S.cleanup.push(() => cancelAnimationFrame(raf));
  /* запуск анализа */
  const bar = (p) => { $("#prog").classList.remove("hidden"); $("#prog i").style.width = Math.round(p * 100) + "%"; };
  const waitJob = async j => { while (j.state === "queued" || j.state === "running") { await new Promise(r => setTimeout(r, 800)); j = await J("/api/job/" + j.id); if (!document.body.contains(pv)) return null;
    bar(j.progress); $("#jstat").textContent = `${j.state === "queued" ? "в очереди" : "анализ"}: ${Math.round(j.progress * 100)}% · тревог ${j.alerts.length}`; } return j; };
  const runLive = async () => {
    $("#run").disabled = $("#runAll").disabled = true; $("#res").classList.add("hidden"); pv.pause(); pv.currentTime = 0; note.textContent = "анализ в реальном времени"; $("#jstat").textContent = "Модель загружается…"; bar(0);
    let j; try { j = await POST("/api/analyze", { video: cur.id, profile: $("#prof").value, live: true }); } catch (e) { $("#jstat").innerHTML = `<span class="err">${esc(e.message)}</span>`; $("#prog").classList.add("hidden"); pinfo(); return; }
    trace = { frames: [], alerts: [], gt: [], fps: 10, frame_hw: null }; drawTimeline(trace);
    const me = liveJob = { id: j.id, since: 0, horizon: 0, started: false, waiting: false }; let tick = 0;
    while (liveJob === me && document.body.contains(pv)) {
      let r; try { r = await J(`/api/job/${me.id}/stream?since=${me.since}`); } catch (e) { $("#jstat").innerHTML = `<span class="err">${esc(e.message)}</span>`; break; }
      if (liveJob !== me) return;
      trace.frames.push(...r.frames); me.since = r.next; me.horizon = r.horizon; trace.fps = r.fps; trace.gt = r.gt; if (r.frame_hw) trace.frame_hw = r.frame_hw;
      if (r.alerts.length !== trace.alerts.length) { trace.alerts = r.alerts; drawTimeline(trace); } else trace.alerts = r.alerts;
      bar(r.progress); $("#jstat").textContent = r.state === "queued" ? "в очереди" : (me.started ? `идёт анализ · тревог ${r.alerts.length}` : "модель загружается…");
      if (!me.started && r.frames.length && r.horizon > 0.5) { me.started = true; $("#rate").value = "1"; pv.playbackRate = 1; pv.play().catch(() => { }); }
      if (r.state === "error") { $("#jstat").innerHTML = `<span class="err">Ошибка: ${esc(r.error)}</span>`; liveJob = null; break; }
      if (r.state === "done") { liveJob = null; $("#jstat").textContent = "Готово"; const a = await J(`/api/analysis?video=${cur.id}&profile=${encodeURIComponent($("#prof").value)}`);
        if (a.trace) { const t0 = pv.currentTime, was = !pv.paused; setAnalysis(a.trace, a.meta); pv.currentTime = t0; if (was) pv.play().catch(() => { }); } loadList(); break; }
      await new Promise(r2 => setTimeout(r2, 300)); tick++;
    }
    $("#prog").classList.add("hidden"); pinfo();
  };
  $("#live").checked = mem.get("an_live") !== "0";
  $("#live").onchange = () => { mem.set("an_live", $("#live").checked ? "1" : "0"); $("#run").textContent = $("#live").checked ? "▶ Смотреть с анализом" : "Анализировать"; };
  $("#run").textContent = $("#live").checked ? "▶ Смотреть с анализом" : "Анализировать";
  $("#run").onclick = async () => {
    if ($("#live").checked) return runLive();
    $("#run").disabled = $("#runAll").disabled = true; $("#jstat").textContent = "Модель запускается…"; bar(0);
    try { const j = await waitJob(await POST("/api/analyze", { video: cur.id, profile: $("#prof").value })); if (!j) return;
      if (j.state === "error") $("#jstat").innerHTML = `<span class="err">Ошибка: ${esc(j.error)}</span>`;
      else { $("#jstat").textContent = "Готово"; const a = await J(`/api/analysis?video=${cur.id}&profile=${encodeURIComponent($("#prof").value)}`); if (a.trace) setAnalysis(a.trace, a.meta); loadList(); }
    } catch (e) { $("#jstat").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("#prog").classList.add("hidden"); pinfo();
  };
  $("#runAll").onclick = async () => {
    $("#run").disabled = $("#runAll").disabled = true; const profile = $("#prof").value;
    try { const js = await POST("/api/analyze", { folder: $("#folder").value, profile }); $("#batch").classList.remove("hidden");
      for (;;) { const now = await Promise.all(js.map(j => J("/api/job/" + j.id))); if (!document.body.contains(pv)) return;
        $("#batch").innerHTML = `<h3 style="margin:0 0 8px">Вся папка · ${esc(profile)}</h3><table><tr><th>Видео</th><th>Состояние</th><th>Тревог</th></tr>${now.map(j => `<tr><td>${esc(j.name)}</td><td class="${j.state === "error" ? "err" : ""}">${j.state === "error" ? esc(j.error) : { queued: "в очереди", running: Math.round(j.progress * 100) + "%", done: "готово" }[j.state]}</td><td>${j.state === "done" ? j.alerts.length : ""}</td></tr>`).join("")}</table>`;
        if (now.every(j => j.state === "done" || j.state === "error")) break; await new Promise(r => setTimeout(r, 1200)); }
      await loadList(); if (cur) { const a = await J(`/api/analysis?video=${cur.id}`); if (a.trace) setAnalysis(a.trace, a.meta); }
    } catch (e) { $("#batch").classList.remove("hidden"); $("#batch").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    pinfo();
  };
  // стартовое состояние: папка из ссылки / запомненная, видео из ссылки или первое
  let f0 = mem.get("an_folder");
  if (q.vid) { try { f0 = (await J("/api/folder-of?video=" + q.vid)).folder; } catch { } }
  if (f0 && [...$("#folder").options].some(o => o.value === f0)) $("#folder").value = f0;
  $("#folder").onchange = async () => { cur = null; await loadList(); if (vs[0]) select(vs[0]); };
  await loadList(); pinfo();
  const v0 = (q.vid && vs.find(v => v.id === q.vid)) || vs[0]; if (v0) await select(v0);
  if (q.auto && cur && !trace && !$("#run").disabled) { history.replaceState(null, "", `#/analysis?vid=${cur.id}`); $("#run").click(); }
}
route();
