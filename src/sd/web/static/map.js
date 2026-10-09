"use strict";
/* Экраны: карта (вкладки, период, карточка улицы), мультипросмотр, тревоги */
const ledClass = c => c.m.pending ? "alert" : (c.n_chunks || c.m.alerts || c.alerts ? "ok" : "");
const glowClass = c => (c.m.pending ? "a" : (c.n_chunks || c.alerts ? "n" : "z")) + (S.sel.has(c.camera_id) ? " s" : "");
const isActive = c => c.m.pending > 0, isClosed = c => c.m.alerts > 0 && c.m.pending === 0;
const PERIODS = [[1, "Последний час"], [3, "Последние 3 часа"], [24, "Последние сутки"], [168, "Последние 7 дней"], [0, "За всё время"]];

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

function streetCard(c, onClose) {
  const m = c.m, chip = m.pending ? `<span class="chip warn">Требует разбора: ${m.pending}</span>` : (m.alerts ? `<span class="chip ok">Разборов не требует</span>` : `<span class="chip">Тревог нет</span>`);
  const per = PERIODS.find(p => p[0] === S.hours);
  return `<div class="scard" id="scard">
    <div class="scard-head"><div class="pack">${IC.pack}</div><div><b>${esc(c.street)}</b><div class="mut small">${esc(c.district)} · камера ${esc(c.index)} · ${c.n_chunks} фрагм.${c.placed ? "" : " · положение условное"}</div></div>${chip}<button class="x" id="scardX" title="Закрыть">${IC.x}</button></div>
    <div class="block"><div class="line big"><span>Активность курения, относительно всех камер</span><span><b>${m.per_hour} / ч</b> &nbsp;<span class="mut">${Math.round(m.activity_rel * 100)} %</span></span></div>
      <div class="pbar"><i style="width:${Math.round(m.activity_rel * 100)}%"></i></div>
      <div class="line"><span class="row" style="gap:10px">${IC.clock}<b>Общее время курения</b></span><span><b>${dur(m.smoke_sec)}</b> <span class="mut">${esc(per ? per[1].toLowerCase() : "")}</span></span></div></div>
    <div class="gauges">
      <div class="gauge">${gaugeSvg(m.avg_conf)}<div class="in"><span>уверенность</span><b>${m.avg_conf == null ? "—" : Math.round(m.avg_conf * 100) + "%"}</b><span>средняя</span></div></div>
      <div class="gauge">${gaugeSvg(m.reaction_sec == null ? 0 : Math.max(0.05, 1 - Math.min(m.reaction_sec, 600) / 600), "#d99a3a")}<div class="in"><span style="color:var(--amber)">${IC.bolt}</span><b>${m.reaction_sec == null ? "—" : (m.reaction_sec >= 60 ? Math.round(m.reaction_sec / 60) + " м" : Math.round(m.reaction_sec) + " с")}</b><span>время реагирования</span></div></div></div>
    ${c.notice ? `<div class="notice">${IC.warn}<div><b>${esc(c.notice.title)}</b> &nbsp;<span>причина: ${esc(c.notice.reason)}</span></div></div>` : ""}
    <div class="row actions"><button class="primary" id="scView" ${c.n_chunks ? "" : "disabled"}>Смотреть камеру</button><button id="scAdd">${S.sel.has(c.camera_id) ? "Убрать из мультипросмотра" : "В мультипросмотр"}</button>
      <a class="btn" href="#/alerts?camera=${encodeURIComponent(c.camera_id)}&status=new,confirmed,false,unsure">Тревоги камеры</a></div></div>`;
}

async function vMap(q) {
  S.hours = +(mem.get("hours") ?? S.hours) || 0;
  let tab = mem.get("map_tab") || "all", showClosed = mem.get("show_closed") !== "0", open = q.cam || null, district = "";
  view.innerHTML = `<div class="toolrow"><div class="tabs" id="tabs"></div><button class="circle" id="fbtn" title="Фильтр по району">${IC.filter}</button><select class="pill hidden" id="fdist"></select><div class="grow"></div>
      <select class="pill" id="period">${PERIODS.map(([h, l]) => `<option value="${h}" ${h === S.hours ? "selected" : ""}>${l}</option>`).join("")}</select></div>
    <div class="toolrow" style="justify-content:flex-end"><label class="switch">Показывать закрытые<input type="checkbox" id="swc" ${showClosed ? "checked" : ""}><i></i></label></div>
    <div class="mapwrap"><div id="map"></div><div class="compass"><div><b></b></div></div><div class="mapnote" id="mapnote"></div><div id="cardhost"></div>
      <div class="sidelist" id="side"></div></div>
    <div class="row" id="selbar" style="margin-top:14px"></div>`;
  const shown = () => S.cams.filter(c => (!district || c.district === district) && (showClosed || !isClosed(c)) && (tab === "all" || (tab === "active" ? isActive(c) : isClosed(c))));
  const draw = () => {
    const all = S.cams.filter(c => !district || c.district === district);
    $("#tabs").innerHTML = [["all", "Все", all.length], ["active", "Активные", all.filter(isActive).length], ["closed", "Закрытые", all.filter(isClosed).length]]
      .map(([k, l, n]) => `<button class="tab ${tab === k ? "on" : ""}" data-t="${k}">${l}<span class="cnt">${n}</span></button>`).join("");
    $$("#tabs .tab").forEach(b => b.onclick = () => { tab = b.dataset.t; mem.set("map_tab", tab); draw(); markers(); });
    const ds = [...new Set(S.cams.map(c => c.district))].sort(); $("#fdist").innerHTML = `<option value="">Все районы</option>` + ds.map(d => `<option ${d === district ? "selected" : ""}>${esc(d)}</option>`).join("");
    const list = shown();
    $("#side").innerHTML = list.length ? list.map(c => `<div class="camrow ${S.sel.has(c.camera_id) ? "sel" : ""}" data-id="${esc(c.camera_id)}"><span class="led ${ledClass(c)}"></span><div class="grow"><b>${esc(c.street)}</b>
      <div class="mut small">${esc(c.district)} · ${esc(c.index)}</div></div>${c.m.pending ? `<span class="badge">${c.m.pending}</span>` : ""}</div>`).join("") : `<div class="mut pad" style="padding:12px">${S.cams.length ? "Нет камер по фильтру" : "Камер нет. Добавьте папки с видео на экране «Источники» (рейка слева, значок карты внизу)."}</div>`;
    $$(".camrow").forEach(r => r.onclick = () => select(r.dataset.id));
    const c = open && camById(open); $("#cardhost").innerHTML = c ? streetCard(c) : "";
    if (c) { $("#scardX").onclick = () => { open = null; draw(); markers(); }; $("#scView").onclick = () => { const ids = S.sel.size ? [...new Set([...S.sel, c.camera_id])].slice(0, 4) : [c.camera_id]; location.hash = "#/multi?cams=" + ids.map(encodeURIComponent).join(","); };
      $("#scAdd").onclick = () => { toggleSel(c.camera_id); draw(); markers(); }; }
    $("#selbar").innerHTML = S.sel.size ? `<span class="mut">Выбрано для мультипросмотра: ${[...S.sel].map(i => esc(camById(i)?.street || i)).join(", ")}</span><button class="primary" id="openMulti">Смотреть выбранные (${S.sel.size})</button><button id="clearSel">Сбросить</button>` : "";
    if (S.sel.size) { $("#openMulti").onclick = () => { location.hash = "#/multi?cams=" + [...S.sel].map(encodeURIComponent).join(","); }; $("#clearSel").onclick = () => { S.sel.clear(); draw(); markers(); }; }
  };
  const select = id => { open = id; draw(); focus(id); markers(); };
  $("#period").onchange = async () => { S.hours = +$("#period").value; mem.set("hours", S.hours); await refreshBase(); draw(); markers(); };
  $("#swc").onchange = () => { showClosed = $("#swc").checked; mem.set("show_closed", showClosed ? "1" : "0"); draw(); markers(); };
  $("#fbtn").onclick = () => $("#fdist").classList.toggle("hidden"); $("#fdist").onchange = () => { district = $("#fdist").value; draw(); markers(); fit(); };
  let markers = () => { }, focus = () => { }, fit = () => { };
  const ok = S.cams.length ? await loadLeaflet() : false;
  const api = ok ? leafletMap(shown, select) : svgMap(shown, select, S.cams.length ? "Схема: нет доступа к плиткам карты (нужен интернет)" : "");
  markers = api.markers; focus = api.focus; fit = api.fit; window.__mapFocus = id => select(id);
  draw(); markers(); fit();
  every(5000, async () => { try { await refreshBase(); draw(); markers(); } catch { } });
}
function leafletMap(shown, select) {
  const m = L.map("map", { zoomControl: false, attributionControl: true }); L.control.zoom({ position: "bottomleft" }).addTo(m);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { subdomains: "abcd", maxZoom: 19, attribution: "© OpenStreetMap, © CARTO" }).addTo(m);
  const mk = {}; let first = true;
  const markers = () => { const ids = new Set(shown().map(c => c.camera_id));
    Object.keys(mk).forEach(k => { if (!ids.has(k)) { mk[k].remove(); delete mk[k]; } });
    shown().forEach(c => { const icon = L.divIcon({ html: `<div class="glow ${glowClass(c)}"></div>`, className: "", iconSize: [64, 64], iconAnchor: [32, 32] });
      if (!mk[c.camera_id]) mk[c.camera_id] = L.marker([c.lat, c.lon], { icon }).addTo(m).on("click", () => select(c.camera_id)); else mk[c.camera_id].setIcon(icon); }); };
  const fit = () => { const p = shown().map(c => [c.lat, c.lon]); if (p.length) m.fitBounds(p, { padding: [120, 120], maxZoom: 16 }); };
  const focus = id => { const c = camById(id); if (c) m.setView([c.lat, c.lon], Math.max(m.getZoom(), 15), { animate: true }); };
  S.cleanup.push(() => m.remove()); return { markers, fit, focus };
}
function svgMap(shown, select, note) {
  $("#mapnote").textContent = note;
  const el = $("#map"), W = () => el.clientWidth || 800, H = () => el.clientHeight || 500;
  let pos = {};
  const layout = list => {
    const lats = list.map(c => c.lat), lons = list.map(c => c.lon), a0 = Math.min(...lats, 59.5), a1 = Math.max(...lats, 59.51), o0 = Math.min(...lons, 30.1), o1 = Math.max(...lons, 30.11);
    const w = W(), h = H(), p = 90, P = list.map(c => [p + (c.lon - o0) / ((o1 - o0) || 1) * (w - 2 * p), h - p - (c.lat - a0) / ((a1 - a0) || 1) * (h - 2 * p)]);
    for (let it = 0; it < 60; it++) for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      let dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], d = Math.hypot(dx, dy); if (d === 0) { dx = 1; dy = 0; d = 1; }
      if (d < 80) { const k = (80 - d) / 2 / d; P[i][0] -= dx * k; P[i][1] -= dy * k; P[j][0] += dx * k; P[j][1] += dy * k; } }
    pos = {}; list.forEach((c, i) => { pos[c.camera_id] = [Math.min(w - 40, Math.max(40, P[i][0])), Math.min(h - 40, Math.max(40, P[i][1]))]; });
  };
  const markers = () => {
    const list = shown(); layout(list); let g = "";
    const w = W(), h = H(); for (let x = 0; x < w; x += 70) g += `<line x1="${x}" y1="0" x2="${x + 40}" y2="${h}" style="stroke:var(--line)" stroke-width="1.5" opacity=".55"/>`; for (let y = 0; y < h; y += 70) g += `<line x1="0" y1="${y}" x2="${w}" y2="${y - 30}" style="stroke:var(--line)" stroke-width="1.5" opacity=".55"/>`;
    list.forEach((c, i) => { const [x, y] = pos[c.camera_id], hor = i % 2 === 0, [x1, y1, x2, y2] = hor ? [x - 120, y, x + 120, y] : [x, y - 90, x, y + 90];
      g += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:var(--card2)" stroke-width="9" stroke-linecap="round"/><line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:var(--panel)" stroke-width="5" stroke-linecap="round"/>
      <text x="${hor ? x - 110 : x + 14}" y="${hor ? y - 12 : y - 70}" style="fill:var(--mut)" font-size="12">${esc(c.street)}</text>`; });
    list.forEach(c => { const [x, y] = pos[c.camera_id], col = c.m.pending ? "#ff5a5f" : (c.n_chunks || c.alerts ? "#c9e04a" : "#7b7b80");
      g += `<g class="pin" data-id="${esc(c.camera_id)}" style="cursor:pointer"><title>${esc(c.title)}</title><circle cx="${x}" cy="${y}" r="30" fill="${col}" opacity=".22">${c.m.pending ? '<animate attributeName="r" values="22;38;22" dur="1.8s" repeatCount="indefinite"/>' : ""}</circle>
      <circle cx="${x}" cy="${y}" r="11" fill="${col}"/>${S.sel.has(c.camera_id) ? `<circle cx="${x}" cy="${y}" r="17" fill="none" style="stroke:var(--ink)" stroke-width="3"/>` : ""}</g>`; });
    el.innerHTML = `<svg width="${w}" height="${h}">${g}</svg>`; $$(".pin", el).forEach(p => p.onclick = () => select(p.dataset.id));
  };
  const ro = new ResizeObserver(markers); ro.observe(el); S.cleanup.push(() => ro.disconnect());
  return { markers, fit() { }, focus() { } };
}

/* ---------------------------------------------------------------- мультипросмотр */
async function vMulti(q) {
  const ids = (q.cams || "").split(",").filter(Boolean).map(decodeURIComponent).slice(0, 4);
  if (!ids.length) { location.hash = "#/map"; return; }
  const cams = await Promise.all(ids.map(i => J("/api/camera/" + encodeURIComponent(i))));
  view.innerHTML = `<div class="toolrow"><a class="btn" href="#/map">← К карте</a><h2 style="margin:0">Камеры: ${cams.length}</h2><div class="grow"></div><button id="playAll">▶ Все</button><button id="pauseAll">⏸ Все</button></div><div class="grid g${cams.length}" id="grid"></div>`;
  const vids = [];
  cams.forEach((c, ci) => {
    const t = document.createElement("div"); t.className = "tile";
    t.innerHTML = `<header><span class="led ${c.pending ? "alert" : "ok"}"></span><b>${esc(c.street)}</b><span class="mut">${esc(c.district)} ${esc(c.index)}</span><div class="grow"></div>
      ${c.chunks.length ? `<select class="pill">${c.chunks.map((ch, i) => `<option value="${i}">${esc(ch.name)}${ch.duration ? " · " + mmss(ch.duration) : ""}</option>`).join("")}</select>` : ""}${c.pending ? `<span class="badge">${c.pending}</span>` : ""}</header>
      <div class="vwrap">${c.chunks.length ? `<video controls muted playsinline preload="metadata"></video>` : `<div class="mut" style="padding:24px;line-height:1.4">Видео нет</div>`}</div><div class="timeline"></div>
      <div class="row" style="padding:10px 16px"><span class="mut grow small" data-st></span>${c.chunks.length ? `<a class="btn" data-an href="#/analysis">Разобрать моделью</a>` : ""}</div>`;
    $("#grid").appendChild(t);
    if (!c.chunks.length) return;
    const v = $("video", t), sel = $("select", t), note = $("[data-st]", t); vids.push(v);
    const load = async (i, t0 = 0, play = false) => {
      const ch = c.chunks[i]; sel.value = i; $("[data-an]", t).href = `#/analysis?vid=${ch.id}&auto=1`;
      const off = c.chunks.slice(0, i).reduce((s, x) => s + (x.duration || 0), 0), d = ch.duration || 1;
      const al = await J(`/api/alerts?camera=${encodeURIComponent(c.camera_id)}&limit=200`);
      const upd = timeline($(".timeline", t), d, al.filter(a => a.start_sec >= off && a.start_sec <= off + d).map(a => ({ cls: "alert", a: a.start_sec - off, b: Math.max(a.end_sec, a.start_sec + 1) - off, tip: `${a.explain} ${pct(a.confidence)}` })), s => { v.currentTime = s; });
      v.ontimeupdate = () => upd(v.currentTime); v.onended = () => { if (i + 1 < c.chunks.length) load(i + 1, 0, true); };
      v.dataset.retried = ""; if (await loadVideo(v, ch.id, note, FMT, t0) && play) v.play().catch(() => { });
    };
    sel.onchange = () => load(+sel.value);
    load(ci === 0 && q.chunk ? Math.max(0, c.chunks.findIndex(x => x.name === q.chunk)) : 0, ci === 0 && q.t ? +q.t : 0);
  });
  $("#playAll").onclick = () => vids.forEach(v => v.play().catch(() => { })); $("#pauseAll").onclick = () => vids.forEach(v => v.pause());
  onKey(e => { if (typing(e) || e.code !== "Space") return; e.preventDefault(); const any = vids.some(v => !v.paused); vids.forEach(v => any ? v.pause() : v.play().catch(() => { })); });
  every(5000, async () => { try { await refreshBase(); } catch { } });
}

/* ---------------------------------------------------------------- тревоги */
const STATUS = { new: "Ждут решения", confirmed: "Подтверждены", false: "Ложные", unsure: "Не уверен" };
const STATUS_CLS = { new: "bad", confirmed: "ok", false: "", unsure: "warn" };
async function vAlerts(q) {
  let status = q.status ? q.status.split(",") : ["new"], cur = q.id ? +q.id : null, items = [], camSel = q.camera || "";
  view.innerHTML = `<div class="toolrow"><div class="tabs" id="chips"></div><div class="grow"></div><select class="pill" id="fcam"><option value="">Все камеры</option></select><a class="btn" href="/api/journal.csv">Журнал CSV</a></div>
    <div class="split"><div class="panel" id="list" style="max-height:calc(100vh - 250px);overflow:auto;padding:8px"></div><div id="detail"></div></div>`;
  $("#fcam").innerHTML += S.cams.map(c => `<option value="${esc(c.camera_id)}" ${c.camera_id === camSel ? "selected" : ""}>${esc(c.street)} (${esc(c.camera_id)})</option>`).join("");
  const load = async () => {
    camSel = $("#fcam").value;
    items = await J(`/api/alerts?limit=100&status=${status.join(",")}${camSel ? "&camera=" + encodeURIComponent(camSel) : ""}`);
    $("#chips").innerHTML = Object.entries(STATUS).map(([k, v]) => `<button class="tab ${status.includes(k) ? "on" : ""}" data-s="${k}" style="font-size:15px;padding:8px 16px">${v}</button>`).join("");
    $$("#chips .tab").forEach(b => b.onclick = () => { const s = b.dataset.s; status = status.includes(s) ? status.filter(x => x !== s) : [...status, s]; if (!status.length) status = ["new"]; load(); });
    $("#list").innerHTML = items.length ? items.map(a => `<div class="acard ${a.id === cur ? "on" : ""}" data-id="${a.id}">${a.has_thumb ? `<img loading="lazy" src="/media/alert/${a.id}/thumb">` : `<img alt="">`}
      <div class="grow"><b>${esc(camById(a.camera_id)?.street || a.district)}</b> <span class="chip ${STATUS_CLS[a.status]}">${a.status_ru}</span>${a.demo ? ' <span class="chip">ДЕМО</span>' : ""}
      <div class="mut small">${fmtT(a.t_abs)} · ${esc(a.explain)}</div><div class="bar" style="margin-top:6px"><i style="width:${Math.round(a.confidence * 100)}%"></i></div></div></div>`).join("")
      : `<div class="mut" style="padding:16px">Тревог по выбранным фильтрам нет.${S.stats.total ? "" : `<br><br>Тревоги появятся, когда работает <code>sd monitor</code>. Для проверки интерфейса: <button id="demo">создать демо-тревоги</button>`}</div>`;
    const d = $("#demo"); if (d) d.onclick = async () => { toast(`Создано: ${(await POST("/api/demo")).n}`); route(); };
    $$("#list .acard").forEach(e => e.onclick = () => { cur = +e.dataset.id; show(); $$("#list .acard").forEach(x => x.classList.toggle("on", x === e)); });
  };
  const decide = async (id, st) => {
    await POST(`/api/alert/${id}/review`, { status: st, reviewer: operator.value, note: ($("#note") || {}).value || "" });
    await refreshBase(); await load(); const nxt = items.find(a => a.status === "new"); cur = nxt ? nxt.id : null; show();
  };
  const show = async () => {
    const d = $("#detail"); if (!cur) { d.innerHTML = `<div class="panel mut">Выберите тревогу слева.</div>`; return; }
    const a = await J("/api/alert/" + cur), c = camById(a.camera_id), t0 = Math.max(0, a.start_sec - (a.chunk_offset || 0) - 3);
    d.innerHTML = `<div class="panel"><div class="row"><h2 style="margin:0">Тревога №${a.id}</h2><span class="chip ${STATUS_CLS[a.status]}">${a.status_ru}</span>${a.demo ? '<span class="chip">ДЕМО</span>' : ""}</div>
      <p class="mut">${esc(c?.street || a.district)} · ${esc(a.district)}, камера ${esc(a.cam_index)} · ${fmtT(a.t_abs)}</p>
      <div class="vwrap" style="border-radius:14px;overflow:hidden">${a.has_clip ? `<video id="aclip" controls autoplay muted loop playsinline style="width:100%"></video>` : (a.has_thumb ? `<img src="/media/alert/${a.id}/thumb" style="width:100%">` : `<div class="mut" style="padding:24px">Клип недоступен</div>`)}</div>
      <div class="mut small" id="anote" style="margin-top:6px"></div>
      <div class="row" style="margin:12px 0;gap:28px"><div><span class="mut small">Уверенность</span><br><b style="font-size:26px">${pct(a.confidence)}</b></div><div><span class="mut small">Что сработало</span><br>${esc(a.explain)}</div><div><span class="mut small">Правило</span><br>${esc(a.rule || "—")}</div><div><span class="mut small">ID трека</span><br>${a.tid}</div></div>
      <textarea id="note" rows="2" placeholder="Заметка (необязательно)" style="width:100%">${esc(a.note || "")}</textarea>
      <div class="row" style="margin-top:12px"><button class="ok" data-d="confirmed">Подтвердить (Y)</button><button class="bad" data-d="false">Ложная (N)</button><button data-d="unsure">Не уверен (U)</button><div class="grow"></div>
      ${c && c.folder && a.chunk ? `<a class="btn" href="#/multi?cams=${encodeURIComponent(a.camera_id)}&chunk=${encodeURIComponent(a.chunk)}&t=${t0}">Открыть в камере</a>` : ""}</div>
      ${a.reviewed_at ? `<p class="mut">Решение: ${a.status_ru} · ${esc(a.reviewer || "оператор")} · ${fmtT(a.reviewed_at)}</p>` : ""}</div>`;
    $$("[data-d]", d).forEach(b => b.onclick = () => decide(a.id, b.dataset.d));
    const v = $("#aclip"); if (v) { v.dataset.retried = ""; loadVideo(v, { alert: a.id }, $("#anote")).then(ok => { if (ok) v.play().catch(() => { }); }); }
  };
  onKey(e => { if (typing(e) || !cur) return; const m = { y: "confirmed", n: "false", u: "unsure" }[e.key.toLowerCase()]; if (m) decide(cur, m); });
  $("#fcam").onchange = load; await load(); await show();
  every(5000, async () => { try { await refreshBase(); if (!document.activeElement || document.activeElement.id !== "note") await load(); } catch { } });
}
