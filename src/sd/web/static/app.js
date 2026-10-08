"use strict";
/* Приложение мониторинга: без фреймворков. Маршруты: #/map #/multi #/alerts #/video #/cameras #/journal */
const $ = (s, r = document) => r.querySelector(s);
const view = $("#view");
const S = { cams: [], stats: null, sel: new Set(), maxId: null, timers: [], cleanup: [] };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const J = async (u, o) => { const r = await fetch(u, o); const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.error || r.statusText); return d; };
const POST = (u, b) => J(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
const fmtT = iso => { try { const d = new Date(iso); return isNaN(d) ? "—" : d.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" }); } catch { return "—"; } };
const mmss = t => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; };
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden"); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 3500); }
const operator = $("#operator");
try { operator.value = localStorage.getItem("op") || ""; } catch { }
operator.addEventListener("input", () => { try { localStorage.setItem("op", operator.value); } catch { } });

/* ---------------------------------------------------------------- данные и опрос */
async function refreshBase() {
  [S.cams, S.stats] = await Promise.all([J("/api/cameras"), J("/api/stats")]);
  const b = $("#pendingBadge"); b.textContent = S.stats.new; b.classList.toggle("hidden", !S.stats.new);
  if (S.maxId !== null && S.stats.max_id > S.maxId) toast(`Новых тревог: ${S.stats.max_id - S.maxId}`);
  S.maxId = S.stats.max_id;
}
const camName = c => `${c.street} · ${c.district} ${c.index}`;
const camById = id => S.cams.find(c => c.camera_id === id);
function every(ms, fn) { S.timers.push(setInterval(fn, ms)); }
function leave() { S.timers.forEach(clearInterval); S.timers = []; S.cleanup.forEach(f => { try { f(); } catch { } }); S.cleanup = []; }

/* ---------------------------------------------------------------- маршрутизация */
const routes = { map: vMap, multi: vMulti, alerts: vAlerts, video: vVideo, cameras: vCameras, journal: vJournal };
async function route() {
  leave();
  const [path, qs] = (location.hash.slice(2) || "map").split("?");
  const name = routes[path] ? path : "map";
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", a.dataset.r === name || (name === "multi" && a.dataset.r === "map")));
  const q = Object.fromEntries(new URLSearchParams(qs || ""));
  try { await refreshBase(); await routes[name](q); }
  catch (e) { view.innerHTML = `<div class="card pad err">Ошибка: ${esc(e.message)}</div>`; }
}
addEventListener("hashchange", route);

/* ---------------------------------------------------------------- карта */
function ledClass(c) { return c.pending ? "alert" : (c.n_chunks || c.alerts ? "ok" : ""); }
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
function toggleSel(id) { S.sel.has(id) ? S.sel.delete(id) : (S.sel.size < 4 ? S.sel.add(id) : toast("В мультипросмотре не больше 4 камер")); }
async function vMap() {
  view.innerHTML = `<div class="kpis" id="kpis"></div>
  <div class="mapwrap"><div class="card mapbox"><div id="map"></div><div class="note" id="mapnote"></div></div>
  <div class="card side" id="side"></div></div>
  <div class="card selbar"><span id="selinfo" class="mut"></span><div class="grow"></div><button id="clear">Сбросить</button><button class="primary" id="openMulti">Смотреть выбранные</button></div>`;
  const k = S.stats;
  $("#kpis").innerHTML = [["Ждут решения", k.new], ["Подтверждено", k.confirmed], ["Ложных", k.false], ["Камер", S.cams.length],
    ["Точность тревог", k.precision == null ? "—" : Math.round(k.precision * 100) + "%"]].map(([a, b]) => `<div class="card kpi"><span class="mut">${a}</span><b>${b}</b></div>`).join("");
  const draw = () => {
    $("#side").innerHTML = (S.cams.length ? "" : `<div class="mut pad">Камер нет. Положите папки <code>район-индекс-время</code> с видео в каталог потоков (SD_STREAMS).</div>`) +
      S.cams.map(c => `<div class="camrow ${S.sel.has(c.camera_id) ? "sel" : ""}" data-id="${esc(c.camera_id)}"><span class="led ${ledClass(c)}"></span>
      <div class="grow"><b>${esc(c.street)}</b><div class="mut">${esc(c.district)} · камера ${esc(c.index)} · ${c.n_chunks} фрагм.</div></div>${c.pending ? `<span class="badge">${c.pending}</span>` : ""}</div>`).join("");
    document.querySelectorAll(".camrow").forEach(r => r.onclick = () => { toggleSel(r.dataset.id); draw(); markSel(); });
    $("#selinfo").textContent = S.sel.size ? `Выбрано камер: ${S.sel.size} — ${[...S.sel].map(i => camById(i)?.street || i).join(", ")}` : "Нажмите на маячок или строку, чтобы выбрать камеры (до 4) для одновременного просмотра";
    $("#openMulti").disabled = !S.sel.size;
  };
  $("#clear").onclick = () => { S.sel.clear(); draw(); markSel(); };
  $("#openMulti").onclick = () => { location.hash = "#/multi?cams=" + [...S.sel].map(encodeURIComponent).join(","); };
  let markSel = () => { };
  const ok = await loadLeaflet();
  if (ok && S.cams.length) markSel = leafletMap(draw); else markSel = svgMap(draw, ok ? "" : "Карта-схема (нет доступа к плиткам OpenStreetMap)");
  draw(); markSel();
  every(5000, async () => { try { await refreshBase(); draw(); markSel(); } catch { } });
}
function popupHtml(c) { return `<b>${esc(c.street)}</b><br>${esc(c.title)}<br>${c.pending ? `<span class="err">Ждут решения: ${c.pending}</span>` : "Тревог нет"}`; }
function leafletMap(redraw) {
  const m = L.map("map", { zoomControl: true }); L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap" }).addTo(m);
  const markers = {};
  const sync = () => S.cams.forEach(c => {
    const html = `<div class="beacon ${c.pending ? "alert" : (c.n_chunks || c.alerts ? "" : "none")} ${S.sel.has(c.camera_id) ? "sel" : ""}"></div>`;
    if (!markers[c.camera_id]) {
      markers[c.camera_id] = L.marker([c.lat, c.lon], { icon: L.divIcon({ html, className: "", iconSize: [18, 18] }) }).addTo(m).on("click", () => { toggleSel(c.camera_id); redraw(); sync(); });
    } else markers[c.camera_id].setIcon(L.divIcon({ html, className: "", iconSize: [18, 18] }));
    markers[c.camera_id].bindTooltip(popupHtml(c));
  });
  const pts = S.cams.map(c => [c.lat, c.lon]); m.fitBounds(pts, { padding: [40, 40], maxZoom: 16 });
  S.cleanup.push(() => m.remove()); sync(); return sync;
}
function svgMap(redraw, note) {
  $("#mapnote").textContent = note;
  const el = $("#map"), W = () => el.clientWidth || 800, H = () => el.clientHeight || 500;
  const lats = S.cams.map(c => c.lat), lons = S.cams.map(c => c.lon);
  const [a0, a1] = [Math.min(...lats, 59.5), Math.max(...lats, 59.51)], [o0, o1] = [Math.min(...lons, 30.1), Math.max(...lons, 30.11)];
  let pos = {};
  const layout = () => {      // проекция + раздвижка: маячки одного района не должны слипаться на схеме
    const w = W(), h = H(), p = 50, P = S.cams.map(c => [p + (c.lon - o0) / ((o1 - o0) || 1) * (w - 2 * p), h - p - (c.lat - a0) / ((a1 - a0) || 1) * (h - 2 * p)]);
    for (let it = 0; it < 60; it++) for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      const dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], d = Math.hypot(dx, dy) || 0.01;
      if (d < 54) { const k = (54 - d) / 2 / d, ux = dx === 0 && dy === 0 ? 1 : dx, uy = dx === 0 && dy === 0 ? 0 : dy; P[i][0] -= ux * k; P[i][1] -= uy * k; P[j][0] += ux * k; P[j][1] += uy * k; }
    }
    pos = {}; S.cams.forEach((c, i) => { pos[c.camera_id] = [Math.min(w - 20, Math.max(20, P[i][0])), Math.min(h - 20, Math.max(20, P[i][1]))]; });
  };
  const proj = c => pos[c.camera_id] || [0, 0];
  const sync = () => {
    layout(); const w = W(), h = H(); let g = "";
    S.cams.forEach((c, i) => { const [x, y] = proj(c), hor = i % 2 === 0;
      const [x1, y1, x2, y2] = hor ? [x - 70, y, x + 70, y] : [x, y - 50, x, y + 50];
      g += `<line class="street" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/><line class="street2" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
      g += `<text x="${hor ? x - 66 : x + 12}" y="${hor ? y - 12 : y - 38}">${esc(c.street)}</text>`; });
    S.cams.forEach(c => { const [x, y] = proj(c); const col = c.pending ? "var(--red)" : (c.n_chunks || c.alerts ? "var(--green)" : "#8a94a3");
      g += `<g class="pin" data-id="${esc(c.camera_id)}"><title>${esc(c.title)}</title>${c.pending ? `<circle cx="${x}" cy="${y}" r="16" fill="var(--red)" opacity=".25"><animate attributeName="r" values="10;22;10" dur="1.8s" repeatCount="indefinite"/></circle>` : ""}
      <circle cx="${x}" cy="${y}" r="9" fill="${col}" stroke="#fff" stroke-width="3"/>${S.sel.has(c.camera_id) ? `<circle cx="${x}" cy="${y}" r="14" fill="none" stroke="var(--brand)" stroke-width="3"/>` : ""}</g>`; });
    el.innerHTML = `<svg class="svgmap" width="${w}" height="${h}">${g}</svg>`;
    el.querySelectorAll(".pin").forEach(p => p.onclick = () => { toggleSel(p.dataset.id); redraw(); sync(); });
  };
  const ro = new ResizeObserver(sync); ro.observe(el); S.cleanup.push(() => ro.disconnect()); sync(); return sync;
}

/* ---------------------------------------------------------------- видео-плитки и общие элементы */
async function setSrc(video, vid, statusEl) {
  for (let i = 0; i < 600; i++) {
    const st = await J("/api/media-status?id=" + vid);
    if (st.state === "direct" || st.state === "ready") { video.src = "/media/video/" + vid; if (statusEl) statusEl.textContent = ""; return true; }
    if (st.state === "error") { if (statusEl) statusEl.textContent = "Не удалось подготовить видео: " + st.error; return false; }
    if (statusEl) statusEl.textContent = "Подготовка видео для браузера (перекодирование)…";
    await new Promise(r => setTimeout(r, 1000));
    if (!document.body.contains(video)) return false;
  }
  return false;
}
function timeline(el, dur, items, onSeek) {
  el.innerHTML = items.map(it => `<i class="${it.cls}" style="left:${(it.a / dur) * 100}%;width:${Math.max(0.4, ((it.b - it.a) / dur) * 100)}%" title="${esc(it.tip || "")}"></i>`).join("") + `<div class="cursor" style="left:0"></div>`;
  el.onclick = e => { const r = el.getBoundingClientRect(); onSeek(((e.clientX - r.left) / r.width) * dur); };
  return t => { el.querySelector(".cursor").style.left = Math.min(100, (t / dur) * 100) + "%"; };
}

async function vMulti(q) {
  const ids = (q.cams || "").split(",").filter(Boolean).map(decodeURIComponent).slice(0, 4);
  if (!ids.length) { location.hash = "#/map"; return; }
  const cams = await Promise.all(ids.map(i => J("/api/camera/" + encodeURIComponent(i))));
  view.innerHTML = `<div class="row" style="margin-bottom:10px"><a href="#/map"><button>← К карте</button></a><h2 style="margin:0">Камеры: ${cams.length}</h2><div class="grow"></div>
    <button id="playAll">▶ Все</button><button id="pauseAll">⏸ Все</button></div><div class="grid g${cams.length}" id="grid"></div>`;
  const vids = [];
  for (const c of cams) {
    const t = document.createElement("div"); t.className = "card tile";
    const opts = c.chunks.map((ch, i) => `<option value="${i}">${esc(ch.name)}${ch.duration ? " · " + mmss(ch.duration) : ""}</option>`).join("");
    t.innerHTML = `<header><span class="led ${ledClass(c)}"></span><b>${esc(c.street)}</b><span class="mut">${esc(c.district)} ${esc(c.index)}</span><div class="grow"></div>
      ${c.chunks.length ? `<select>${opts}</select>` : ""}${c.pending ? `<span class="badge">${c.pending}</span>` : ""}</header>
      <div class="vwrap">${c.chunks.length ? `<video controls muted playsinline preload="metadata"></video>` : `<div class="pad mut" style="line-height:1.4">Видео нет</div>`}</div>
      <div class="timeline"></div><div class="pad mut small" data-st></div>`;
    $("#grid").appendChild(t);
    if (!c.chunks.length) continue;
    const v = $("video", t), sel = $("select", t), st = $("[data-st]", t); vids.push(v);
    const load = async i => {
      const ch = c.chunks[i]; await setSrc(v, ch.id, st);
      const al = await J(`/api/alerts?camera=${encodeURIComponent(c.camera_id)}&limit=200`);
      const off = (c.chunks.slice(0, i).reduce((s, x) => s + (x.duration || 0), 0));
      const items = al.filter(a => a.start_sec >= off && a.start_sec <= off + (ch.duration || 1e9)).map(a => ({ cls: "alert", a: a.start_sec - off, b: a.end_sec - off, tip: `${a.explain} ${Math.round(a.confidence * 100)}%` }));
      const dur = ch.duration || 1; const upd = timeline($(".timeline", t), dur, items, s => { v.currentTime = s; });
      v.ontimeupdate = () => upd(v.currentTime);
    };
    sel.onchange = () => load(+sel.value); load(0);
  }
  $("#playAll").onclick = () => vids.forEach(v => v.play().catch(() => { }));
  $("#pauseAll").onclick = () => vids.forEach(v => v.pause());
  every(5000, async () => { try { await refreshBase(); } catch { } });
}

/* ---------------------------------------------------------------- тревоги */
const STATUS = { new: "ждёт решения", confirmed: "подтверждена", false: "ложная", unsure: "не уверен" };
async function vAlerts(q) {
  let status = q.status ? q.status.split(",") : ["new"], cur = q.id ? +q.id : null, items = [];
  view.innerHTML = `<div class="row" style="margin-bottom:10px"><div id="chips" class="row"></div><div class="grow"></div><label class="mut">Камера <select id="fcam"><option value="">все</option></select></label></div>
    <div class="split"><div class="card" id="list" style="max-height:calc(100vh - 160px);overflow:auto"></div><div id="detail"></div></div>`;
  $("#fcam").innerHTML += S.cams.map(c => `<option value="${esc(c.camera_id)}">${esc(c.street)} (${esc(c.camera_id)})</option>`).join("");
  const load = async () => {
    const cam = $("#fcam").value;
    items = await J(`/api/alerts?limit=100&status=${status.join(",")}${cam ? "&camera=" + encodeURIComponent(cam) : ""}`);
    $("#chips").innerHTML = Object.entries(STATUS).map(([k, v]) => `<button data-s="${k}" class="${status.includes(k) ? "primary" : ""}">${v}</button>`).join("");
    $("#chips").querySelectorAll("button").forEach(b => b.onclick = () => { const s = b.dataset.s; status = status.includes(s) ? status.filter(x => x !== s) : [...status, s]; load(); });
    $("#list").innerHTML = items.length ? items.map(a => `<div class="acard ${a.id === cur ? "on" : ""}" data-id="${a.id}">${a.has_thumb ? `<img loading="lazy" src="/media/alert/${a.id}/thumb">` : `<img alt="">`}
      <div class="grow"><b>${esc(camById(a.camera_id)?.street || a.district)}</b> <span class="pill ${a.status}">${a.status_ru}</span>${a.demo ? ' <span class="pill">ДЕМО</span>' : ""}
      <div class="mut">${fmtT(a.t_abs)} · ${esc(a.explain)}</div><div class="bar"><i style="width:${Math.round(a.confidence * 100)}%"></i></div></div></div>`).join("") : `<div class="pad mut">Тревог по выбранным фильтрам нет.</div>`;
    $("#list").querySelectorAll(".acard").forEach(e => e.onclick = () => { cur = +e.dataset.id; show(); $("#list").querySelectorAll(".acard").forEach(x => x.classList.toggle("on", x === e)); });
  };
  const decide = async (id, st) => {
    await POST(`/api/alert/${id}/review`, { status: st, reviewer: operator.value, note: ($("#note") || {}).value || "" });
    await refreshBase(); await load();
    const nxt = items.find(a => a.status === "new"); cur = st === "new" ? id : (nxt ? nxt.id : null); show();
  };
  const show = async () => {
    const d = $("#detail"); if (!cur) { d.innerHTML = `<div class="card pad mut">Выберите тревогу слева.</div>`; return; }
    const a = await J("/api/alert/" + cur), c = camById(a.camera_id);
    d.innerHTML = `<div class="card pad"><div class="row"><h2 style="margin:0">Тревога №${a.id}</h2><span class="pill ${a.status}">${a.status_ru}</span>${a.demo ? '<span class="pill">ДЕМО</span>' : ""}</div>
      <p class="mut">${esc(c?.street || a.district)} · ${esc(a.district)}, камера ${esc(a.cam_index)} · ${fmtT(a.t_abs)}</p>
      <div class="vwrap">${a.has_clip ? `<video controls autoplay muted loop src="/media/alert/${a.id}/clip" style="width:100%"></video>` : (a.has_thumb ? `<img src="/media/alert/${a.id}/thumb" style="width:100%">` : `<div class="pad mut">Клип недоступен</div>`)}</div>
      <div class="row" style="margin:10px 0"><div><span class="mut">Уверенность</span><br><b style="font-size:22px">${Math.round(a.confidence * 100)}%</b></div><div><span class="mut">Что сработало</span><br>${esc(a.explain)}</div>
      <div><span class="mut">Правило</span><br>${esc(a.rule)}</div><div><span class="mut">ID трека</span><br>${a.tid}</div></div>
      <textarea id="note" rows="2" placeholder="Заметка (необязательно)" style="width:100%">${esc(a.note || "")}</textarea>
      <div class="row" style="margin-top:10px"><button class="ok" data-d="confirmed">Подтвердить (Y)</button><button class="bad" data-d="false">Ложная (N)</button><button data-d="unsure">Не уверен (U)</button>
      <div class="grow"></div>${a.chunk ? `<a href="#/map"><button>К камере</button></a>` : ""}</div>
      ${a.reviewed_at ? `<p class="mut">Решение: ${a.status_ru} · ${esc(a.reviewer || "оператор")} · ${fmtT(a.reviewed_at)}</p>` : ""}
      <details><summary class="mut">Для разбора разработчиком</summary><pre class="mut">фрагмент: ${esc(a.chunk)}\nначало в фрагменте: ${Math.max(0, a.start_sec - (a.chunk_offset || 0)).toFixed(1)} с\nключ: ${esc(a.key)}\nпрофиль: ${esc(a.fingerprint)}</pre></details></div>`;
    d.querySelectorAll("[data-d]").forEach(b => b.onclick = () => decide(a.id, b.dataset.d));
  };
  const keys = e => { if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || !cur) return; const m = { y: "confirmed", n: "false", u: "unsure" }[e.key.toLowerCase()]; if (m) decide(cur, m); };
  addEventListener("keydown", keys); S.cleanup.push(() => removeEventListener("keydown", keys));
  $("#fcam").onchange = load; await load(); await show();
  every(5000, async () => { try { await refreshBase(); if (!document.activeElement || document.activeElement.id !== "note") await load(); } catch { } });
}

/* ---------------------------------------------------------------- просмотр моделью */
async function vVideo(q) {
  const [folders, profiles] = await Promise.all([J("/api/folders"), J("/api/profiles")]);
  view.innerHTML = `<div class="split" style="grid-template-columns:minmax(280px,360px) 1fr">
    <div class="card pad"><h3>Видео из папки</h3><select id="folder" style="width:100%">${folders.map(f => `<option value="${f.id}">${esc(f.name)} (${f.n})</option>`).join("") || "<option value=''>папок нет</option>"}</select>
      <div id="vlist" style="margin:10px 0;max-height:34vh;overflow:auto"></div><h3>Решатель</h3>
      <select id="prof" style="width:100%">${profiles.map(p => `<option value="${esc(p.name)}" ${p.ready ? "" : "data-bad"}>${p.ready ? "" : "⚠ "}${esc(p.name)}${p.kind === "solver" ? " (решатель)" : ""}</option>`).join("") || "<option value=''>нет профилей</option>"}</select>
      <div id="pinfo" class="mut" style="margin:6px 0"></div><button class="primary" id="run" style="width:100%">Анализировать моделью</button><div class="prog hidden" id="prog" style="margin-top:8px"><i style="width:0"></i></div><div id="jstat" class="mut" style="margin-top:6px"></div></div>
    <div><div class="card"><div class="vwrap" id="pw"><video id="pv" controls muted playsinline></video><canvas id="ov"></canvas></div><div class="timeline" id="tl"></div>
      <div class="pad mut" id="pst">Выберите видео слева.</div></div><div class="card pad" style="margin-top:12px" id="res"></div></div></div>`;
  let cur = null, trace = null, meta = null, dur = 1, upd = () => { };
  const pinfo = () => { const p = profiles.find(x => x.name === $("#prof").value); $("#pinfo").innerHTML = p ? (p.ready ? `${esc(Object.entries(p.describe).filter(([k]) => ["pose", "imgsz", "cycle_model", "fps"].includes(k)).map(([k, v]) => k + ": " + v).join(" · "))}` : `<span class="err">${p.problems.map(esc).join("<br>")}</span>`) : ""; $("#run").disabled = !p || !p.ready || !cur; };
  $("#prof").onchange = pinfo;
  const loadList = async () => {
    const fid = $("#folder").value; if (!fid) return;
    const vs = await J("/api/videos?folder=" + fid);
    $("#vlist").innerHTML = vs.map(v => `<div class="camrow ${cur && cur.id === v.id ? "sel" : ""}" data-id="${v.id}"><div class="grow"><b>${esc(v.name)}</b><div class="mut">${v.duration ? mmss(v.duration) : "—"} · ${v.size_mb} МБ ${v.analyzed ? "· есть разбор" : ""}</div></div></div>`).join("") || `<div class="mut">Видео нет</div>`;
    $("#vlist").querySelectorAll(".camrow").forEach(r => r.onclick = () => pick(vs.find(v => v.id === r.dataset.id)));
    if (q.vid && !cur) { const v = vs.find(x => x.id === q.vid); if (v) pick(v); }
  };
  const pick = async v => {
    cur = v; trace = null; meta = null; $("#res").innerHTML = ""; $("#jstat").textContent = "";
    $("#vlist").querySelectorAll(".camrow").forEach(r => r.classList.toggle("sel", r.dataset.id === v.id));
    dur = v.duration || 1; await setSrc($("#pv"), v.id, $("#pst")); pinfo();
    const a = await J("/api/analysis?video=" + v.id); if (a.trace) setAnalysis(a.trace, a.meta);
    else { upd = timeline($("#tl"), dur, [], s => { $("#pv").currentTime = s; }); $("#pst").textContent = "Разбора моделью ещё нет — выберите решатель и нажмите «Анализировать»."; }
  };
  const setAnalysis = (t, m) => {
    trace = t; meta = m; const items = [...(t.gt || []).map(g => ({ cls: "gt", a: g.start, b: g.end, tip: "разметка: курение" })), ...(t.alerts || []).map(a => ({ cls: "alert", a: a.start, b: a.end, tip: `${a.explain} ${Math.round(a.confidence * 100)}%` }))];
    upd = timeline($("#tl"), dur, items, s => { $("#pv").currentTime = s; });
    const s = (m && m.stats) || {}, mt = (m && m.metrics) || null;
    $("#pst").textContent = `Профиль: ${m ? m.profile : "—"} · кадров ${t.frames.length} · тревог ${(t.alerts || []).length}`;
    $("#res").innerHTML = `<h3>Результат модели</h3><div class="row">${[["Тревог", (t.alerts || []).length], ["× реального времени", s.rt_factor ?? "—"], ["мс/кадр p95", s.ms_p95 ?? "—"], ["Циклов", s.cycles ?? "—"],
      ...(mt ? [["F1", mt.f1.toFixed(3)], ["TP/FP/FN", `${mt.tp}/${mt.fp}/${mt.fn}`], ["Задержка, с", mt.alert_delay_median == null ? "—" : mt.alert_delay_median.toFixed(1)]] : [])].map(([a, b]) => `<div><span class="mut">${a}</span><br><b style="font-size:20px">${b}</b></div>`).join("")}</div>
      <p class="${s.rt_factor >= 1 ? "" : "err"}">${esc(s.verdict || "")}</p>
      <table><tr><th>Время потока</th><th>Начало</th><th>ID</th><th>Причина</th><th>Увер.</th><th>Задержка</th></tr>${(t.alerts || []).map(a => `<tr style="cursor:pointer" data-t="${a.start}"><td>${mmss(a.t_open)}</td><td>${mmss(a.start)}</td><td>${a.tid}</td><td>${esc(a.explain)}</td><td>${Math.round(a.confidence * 100)}%</td><td>${a.delay.toFixed(1)} с</td></tr>`).join("") || "<tr><td colspan=6 class='mut'>тревог нет</td></tr>"}</table>`;
    $("#res").querySelectorAll("tr[data-t]").forEach(r => r.onclick = () => { $("#pv").currentTime = Math.max(0, +r.dataset.t - 1); $("#pv").play().catch(() => { }); });
  };
  /* оверлей: рамки людей по времени кадра, красные у тех, по кому есть тревога */
  const cv = $("#ov"), v = $("#pv");
  const paint = () => {
    const w = v.clientWidth, h = v.clientHeight; if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d"); g.clearRect(0, 0, w, h); upd(v.currentTime);
    if (!trace || !trace.frames.length) return;
    const fr = trace.frames, t = v.currentTime; let lo = 0, hi = fr.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; fr[m][0] < t ? lo = m + 1 : hi = m; }
    let k = lo; if (k > 0 && Math.abs(fr[k - 1][0] - t) < Math.abs(fr[k][0] - t)) k--;
    if (Math.abs(fr[k][0] - t) > 2.5 / (trace.fps || 5)) return;
    const [fh, fw] = trace.frame_hw || [v.videoHeight, v.videoWidth]; const sx = w / (fw || v.videoWidth || 1), sy = h / (fh || v.videoHeight || 1);
    const live = (trace.alerts || []).filter(a => t >= a.start && t <= a.end + 1);
    for (const [tid, x1, y1, x2, y2] of fr[k][1]) {
      const al = live.find(a => a.tid === tid), hot = !!al;
      g.lineWidth = hot ? 3 : 1.5; g.strokeStyle = hot ? "#ff3b30" : "rgba(255,255,255,.85)"; g.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      g.fillStyle = hot ? "#ff3b30" : "rgba(0,0,0,.55)"; const lbl = hot ? `ТРЕВОГА ID ${tid} · ${Math.round(al.confidence * 100)}%` : `ID ${tid}`; g.font = "12px system-ui"; const tw = g.measureText(lbl).width + 8;
      g.fillRect(x1 * sx, Math.max(0, y1 * sy - 18), tw, 18); g.fillStyle = "#fff"; g.fillText(lbl, x1 * sx + 4, Math.max(13, y1 * sy - 5));
    }
    if (live.some(a => t >= a.t_open - 1e-3)) { g.fillStyle = "rgba(217,45,32,.9)"; g.fillRect(0, 0, w, 26); g.fillStyle = "#fff"; g.font = "bold 14px system-ui"; g.fillText("ТРЕВОГА: курение", 10, 18); }
  };
  let raf = 0; const loop = () => { paint(); raf = requestAnimationFrame(loop); }; loop(); S.cleanup.push(() => cancelAnimationFrame(raf));
  $("#run").onclick = async () => {
    $("#run").disabled = true; $("#prog").classList.remove("hidden"); $("#jstat").textContent = "Модель запускается…";
    try {
      let j = await POST("/api/analyze", { video: cur.id, profile: $("#prof").value });
      while (j.state === "queued" || j.state === "running") {
        await new Promise(r => setTimeout(r, 800)); j = await J("/api/job/" + j.id); if (!document.body.contains($("#prog"))) return;
        $("#prog i").style.width = Math.round(j.progress * 100) + "%"; $("#jstat").textContent = `${j.state === "queued" ? "в очереди" : "анализ"}: ${Math.round(j.progress * 100)}% · тревог ${j.alerts.length}`;
      }
      if (j.state === "error") { $("#jstat").innerHTML = `<span class="err">Ошибка: ${esc(j.error)}</span>`; }
      else { $("#jstat").textContent = "Готово"; const a = await J(`/api/analysis?video=${cur.id}&profile=${encodeURIComponent($("#prof").value)}`); if (a.trace) setAnalysis(a.trace, a.meta); }
    } catch (e) { $("#jstat").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("#prog").classList.add("hidden"); pinfo();
  };
  if (q.vid) { try { const f = (await J("/api/folder-of?video=" + q.vid)).folder; if ([...$("#folder").options].some(o => o.value === f)) $("#folder").value = f; } catch { } }
  $("#folder").onchange = loadList; await loadList(); pinfo();
}

/* ---------------------------------------------------------------- камеры и журнал */
async function vCameras() {
  view.innerHTML = `<div class="card"><table><tr><th>Улица</th><th>Район</th><th>Камера</th><th>Начало записи</th><th>Фрагментов</th><th>Обработано</th><th>Тревог</th><th>Ждут</th><th></th></tr>
    ${S.cams.map(c => `<tr><td><span class="led ${ledClass(c)}" style="display:inline-block"></span> <b>${esc(c.street)}</b>${c.placed ? "" : ' <span class="mut" title="координаты не заданы в cameras.yaml">·схема</span>'}</td><td>${esc(c.district)}</td><td>${esc(c.index)}</td><td>${fmtT(c.start)}</td><td>${c.n_chunks}</td>
    <td>${(c.processed_sec / 60).toFixed(1)} мин</td><td>${c.alerts}</td><td>${c.pending}</td><td><a href="#/multi?cams=${encodeURIComponent(c.camera_id)}"><button>Смотреть</button></a></td></tr>`).join("") || `<tr><td colspan="9" class="mut">Камер нет</td></tr>`}</table></div>
    <div class="row" style="margin-top:12px"><button id="demo">Создать демо-тревоги</button><button id="undemo">Удалить демо-тревоги</button><span class="mut">Демо — синтетика для проверки интерфейса, не данные распознавания.</span></div>`;
  $("#demo").onclick = async () => { const r = await POST("/api/demo"); toast(`Создано: ${r.n}`); route(); };
  $("#undemo").onclick = async () => { const r = await POST("/api/demo", { clear: true }); toast(`Удалено: ${r.n}`); route(); };
}
async function vJournal() {
  const al = await J("/api/alerts?limit=500&status=confirmed,false,unsure");
  view.innerHTML = `<div class="row" style="margin-bottom:10px"><h2 style="margin:0">Решения оператора</h2><div class="grow"></div><a href="/api/journal.csv"><button>Скачать CSV</button></a></div>
  <div class="card"><table><tr><th>№</th><th>Время события</th><th>Улица</th><th>Увер.</th><th>Решение</th><th>Оператор</th><th>Принято</th><th>Заметка</th></tr>${al.map(a => `<tr><td>${a.id}</td><td>${fmtT(a.t_abs)}</td>
  <td>${esc(camById(a.camera_id)?.street || a.district)}</td><td>${Math.round(a.confidence * 100)}%</td><td><span class="pill ${a.status}">${a.status_ru}</span></td><td>${esc(a.reviewer)}</td><td>${fmtT(a.reviewed_at)}</td><td>${esc(a.note)}</td></tr>`).join("") || `<tr><td colspan="8" class="mut">Решений пока нет</td></tr>`}</table></div>`;
}
route();
