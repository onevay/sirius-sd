"use strict";
/* Анализ видео: источники (папки, загрузка), живой разбор моделью, итог, триггеры, уверенность, люди */
const RULES = { two_cycles: "два цикла подряд в окне регламента", "cycle+object": "цикл + предмет у рта", cycle_object: "цикл + предмет у рта", cycle_plus_object: "цикл + предмет у рта" };
const ruleText = r => RULES[r] || r || "—";
const sigChips = c => {
  const s = [`<span class="chip">пауза у рта ${(c.hold ?? 0).toFixed(1)} с</span>`];
  if (c.object != null) s.push(`<span class="chip teal">предмет ${pct(c.object)}</span>`);
  if (c.photo != null) s.push(`<span class="chip teal">фото-модель ${pct(c.photo)}</span>`);
  if (c.vlm != null) s.push(`<span class="chip teal">VLM ${pct(c.vlm)}</span>`);
  if (c.fusion) s.push(`<span class="chip ${c.fusion > 0 ? "ok" : "warn"}">поправка ${c.fusion > 0 ? "+" : ""}${c.fusion.toFixed(2)}</span>`);
  if (c.base != null && c.fusion) s.push(`<span class="chip">до поправки ${pct(c.base)}</span>`);
  if (s.length === 1 && c.object == null && c.photo == null && c.vlm == null) s.push(`<span class="chip">только поза и ритм</span>`);
  return s.join("");
};
const scoreBar = (v, th) => `<div class="bar" style="min-width:90px"><i style="width:${Math.round((v || 0) * 100)}%"></i>${th != null ? `<b style="left:${Math.round(th * 100)}%"></b>` : ""}</div>`;

function uploadOne(file, onProgress) {
  return new Promise((res, rej) => {
    const x = new XMLHttpRequest(); x.open("POST", `/api/upload?name=${encodeURIComponent(file.name)}`);
    x.upload.onprogress = e => e.lengthComputable && onProgress(e.loaded / e.total);
    x.onload = () => { let d = {}; try { d = JSON.parse(x.responseText); } catch { } x.status === 200 ? res(d) : rej(new Error(d.error || `HTTP ${x.status}`)); };
    x.onerror = () => rej(new Error("сеть недоступна")); x.send(file);
  });
}

const qTxt = q => q ? `F1 ${q.f1 == null ? "—" : q.f1.toFixed(2)}${q.lo != null ? ` [${q.lo.toFixed(2)}–${q.hi.toFixed(2)}]` : ""} · P ${q.precision == null ? "—" : q.precision.toFixed(2)} / R ${q.recall == null ? "—" : q.recall.toFixed(2)} · ${q.clips} клип.` : "качество не измерялось";
/* выбор конфигурации: профили, решатели и конфигурации, по которым уже считалась оценка (журнал), — каждая с качеством на размеченных клипах */
const profOptions = ps => {
  const o = p => `<option value="${esc(p.name)}">${p.ready ? "" : "⚠ "}${esc(p.label || p.name)}${p.kind === "solver" ? " · решатель" : ""} — ${qTxt(p.quality)}</option>`;
  const g = (t, a) => a.length ? `<optgroup label="${t}">${a.map(o).join("")}</optgroup>` : "";
  return (g("Профили и решатели", ps.filter(p => p.kind !== "journal")) + g("Уже прогонялись (из журнала оценок)", ps.filter(p => p.kind === "journal"))) || "<option value=''>нет профилей</option>";
};
async function vAnalysis(q) {
  let [folders, profiles] = await Promise.all([J("/api/folders"), J("/api/profiles")]);
  const ready = profiles.filter(p => p.ready);
  view.innerHTML = `<div class="an">
    <aside class="panel" style="display:flex;flex-direction:column;gap:12px">
      <select id="folder"></select>
      <div id="vlist" class="vlist"></div>
      <div class="dz" id="dz">Перетащите видео сюда или нажмите, чтобы выбрать<br><span class="small">любые форматы: mp4, avi, mkv, mov, wmv, webm, ts…</span><input type="file" id="file" accept="video/*,.mkv,.avi,.wmv,.ts,.mov,.m4v,.webm" multiple class="hidden"></div>
      <div id="upl" class="small"></div>
      <div class="row" style="flex-wrap:nowrap"><input id="rootPath" type="text" placeholder="Путь к папке с видео" style="flex:1;min-width:0"><button id="addRoot" title="Подключить папку">+</button></div>
      <select id="prof">${profOptions(profiles)}</select>
      <div id="pinfo" class="mut small"></div>
      <label class="row small" style="gap:8px;flex-wrap:nowrap"><input type="checkbox" id="live" checked><span>В реальном времени: видео идёт вместе с моделью, рамки и тревоги сразу</span></label>
      <button class="primary" id="run">▶ Смотреть с анализом</button><button id="runAll" title="Быстрый анализ всех видео папки без просмотра">Всю папку (быстро)</button>
      <div class="prog hidden" id="prog"><i></i></div><div id="jstat" class="mut small"></div></aside>
    <section><div class="panel" style="padding:0;overflow:hidden"><div class="vwrap" id="pw"><video id="pv" controls muted playsinline></video><canvas id="ov"></canvas></div><div class="timeline" id="tl"></div>
      <div class="toolbar"><button id="prevA" title="[">◀ тревога</button><button id="nextA" title="]">тревога ▶</button><select class="pill" id="rate"><option value="0.5">×0.5</option><option value="1" selected>×1</option><option value="2">×2</option><option value="4">×4</option></select>
        <label><input type="checkbox" id="tBox" checked> рамки</label><label><input type="checkbox" id="tGt" checked> разметка</label><span class="grow mut small" id="pst"></span></div></div>
      <div class="panel hidden" id="res" style="margin-top:14px"></div><div class="panel hidden" id="batch" style="margin-top:14px"></div></section></div>`;
  let cur = null, trace = null, meta = null, total = 1, upd = () => { }, vs = [], liveJob = null, tabSel = mem.get("an_tab") || "summary", lastRender = 0;
  const pv = $("#pv"), note = $("#pst");
  S.cleanup.push(() => { liveJob = null; });
  const fillFolders = (sel) => { $("#folder").innerHTML = folders.map(f => `<option value="${f.id}">${f.kind === "camera" ? "камера · " : ""}${esc(f.name)} (${f.n})${f.removable ? " ✕" : ""}</option>`).join("") || "<option value=''>папок нет</option>"; if (sel) $("#folder").value = sel; };
  const refreshFolders = async sel => { folders = await J("/api/folders"); fillFolders(sel); };
  const p0 = profiles.find(p => p.name === mem.get("an_profile") && p.ready) || ready[0] || profiles[0]; if (p0) $("#prof").value = p0.name;
  const pinfo = () => { const p = profiles.find(x => x.name === $("#prof").value); mem.set("an_profile", $("#prof").value);
    $("#pinfo").innerHTML = !p ? "" : (p.ready ? esc(Object.entries(p.describe).filter(([k]) => ["pose", "imgsz", "cycle_model", "fps"].includes(k)).map(([k, v]) => `${k}: ${v}`).join(" · ")) : `<span class="err">${p.problems.map(esc).join("<br>")}</span> ${p.kind === "profile" ? '<a href="#/models">Исправить в «Моделях»</a>' : ""}`)
      + `<div><b>${esc(qTxt(p.quality))}</b>${p.quality ? ` · порог ${p.quality.threshold ?? "—"} · TP/FP/FN ${p.quality.tp}/${p.quality.fp}/${p.quality.fn}${p.quality.failed ? ` · ошибок клипов: ${p.quality.failed}` : ""}<br>оценка ${esc(String(p.quality.created || "").slice(0, 16))} на размеченных клипах` : ""}</div>`;
    $("#run").disabled = !p || !p.ready || !cur; $("#runAll").disabled = !p || !p.ready; };
  $("#prof").onchange = pinfo;
  const loadList = async () => {
    const fid = $("#folder").value; if (!fid) { $("#vlist").innerHTML = `<div class="mut" style="padding:10px">Папок с видео нет. Загрузите файл или подключите папку.</div>`; return; }
    mem.set("an_folder", fid); vs = await J("/api/videos?folder=" + fid);
    $("#vlist").innerHTML = vs.map(v => { const a = v.analysis; return `<div class="camrow ${cur && cur.id === v.id ? "sel" : ""}" data-id="${v.id}"><div class="grow"><b>${esc(v.name)}</b>
      <div class="mut small">${v.duration ? mmss(v.duration) : "—"}${a ? ` · <span class="${a.alerts ? "err" : ""}">тревог: ${a.alerts ?? "?"}</span>${a.f1 != null ? ` · F1 ${a.f1.toFixed(2)}` : ""}` : ""}</div></div></div>`; }).join("") || `<div class="mut" style="padding:10px">Видео нет</div>`;
    $$("#vlist .camrow").forEach(r => r.onclick = () => select(vs.find(v => v.id === r.dataset.id)));
  };
  /* ---------- загрузка и подключение источников */
  const upload = async files => {
    const list = [...files]; if (!list.length) return; let last = null;
    $("#upl").innerHTML = list.map((f, i) => `<div>${esc(f.name)}<div class="prog"><i id="u${i}"></i></div></div>`).join("");
    for (const [i, f] of list.entries()) {
      try { last = await uploadOne(f, p => { const b = $("#u" + i); if (b) b.style.width = Math.round(p * 100) + "%"; }); $("#u" + i).style.width = "100%"; }
      catch (e) { $("#upl").innerHTML += `<div class="err">${esc(f.name)}: ${esc(e.message)}</div>`; }
    }
    if (last) { toast(`Загружено файлов: ${list.length}`); await refreshFolders(last.folder); await loadList(); const v = vs.find(x => x.id === last.id); if (v) select(v); }
  };
  const dz = $("#dz"); dz.onclick = () => $("#file").click(); $("#file").onchange = e => upload(e.target.files);
  ["dragover", "dragenter"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("over"); })); ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", e => upload(e.dataTransfer.files));
  $("#addRoot").onclick = async () => { try { const r = await POST("/api/root", { path: $("#rootPath").value }); toast(`Папка подключена: видео ${r.n}`); $("#rootPath").value = ""; await refreshFolders(); const f = folders.find(x => x.root === r.key); if (f) { $("#folder").value = f.id; } cur = null; await loadList(); if (vs[0]) select(vs[0]); } catch (e) { toast(e.message); } };
  /* ---------- выбор видео и результаты */
  const select = async v => {
    liveJob = null; cur = v; trace = null; meta = null; $("#res").classList.add("hidden"); $("#jstat").textContent = ""; total = v.duration || 1; history.replaceState(null, "", `#/analysis?vid=${v.id}`);
    $$("#vlist .camrow").forEach(r => r.classList.toggle("sel", r.dataset.id === v.id));
    upd = timeline($("#tl"), total, [], s => { pv.currentTime = s; }); note.textContent = ""; pinfo();
    pv.dataset.retried = ""; await loadVideo(pv, { id: v.id }, note);
    const a = await J("/api/analysis?video=" + v.id); if (a.trace) setAnalysis(a.trace, a.meta); else note.textContent = note.textContent || "Разбора ещё нет — нажмите «Смотреть с анализом».";
  };
  const drawTimeline = t => { const th = (t.thresholds || {}).cycle ?? 0.5;
    upd = timeline($("#tl"), total, [...(t.gt || []).map(g => ({ cls: "gt", a: g.start, b: g.end, tip: "разметка: курение" })),
      ...(t.cycles || []).map(c => ({ cls: "cyc" + (c.score >= th ? "" : " lo"), a: c.peak, b: c.peak + 0.2, tip: `цикл ID ${c.tid}: ${pct(c.score)}` })),
      ...(t.alerts || []).map(a => ({ cls: "alert", a: a.start, b: Math.max(a.end, a.start + 1), tip: `${a.explain} ${pct(a.confidence)}` }))], s => { pv.currentTime = s; }); };
  const seek = t => { pv.currentTime = Math.max(0, t - 1); pv.play().catch(() => { }); };
  const setAnalysis = (t, m) => { trace = t; meta = m; drawTimeline(t); note.textContent = `профиль ${m ? m.profile : "—"}`; renderResult(true); };
  const renderResult = force => {
    if (!trace) return; const now = performance.now(); if (!force && now - lastRender < 1000) return; lastRender = now;
    const box = $("#res"); box.classList.remove("hidden"); const st = (meta && meta.stats) || {}, mt = (meta && meta.metrics) || null, al = trace.alerts || [], cy = trace.cycles || [], th = trace.thresholds || {};
    const tabs = [["summary", "Итог"], ["triggers", "Триггеры"], ["conf", "Уверенность"], ["people", "Люди"]];
    let body = "";
    if (tabSel === "summary") {
      const avg = al.length ? al.reduce((s, a) => s + a.confidence, 0) / al.length : null, delay = al.length ? [...al.map(a => a.delay)].sort((a, b) => a - b)[Math.floor(al.length / 2)] : null;
      body = `<div class="gauges" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">
        <div class="gauge">${gaugeSvg(avg)}<div class="in"><span>уверенность</span><b>${pct(avg)}</b><span>средняя по тревогам</span></div></div>
        <div class="gauge">${gaugeSvg(delay == null ? 0 : Math.max(0.05, 1 - Math.min(delay, 30) / 30), "#d99a3a")}<div class="in"><span style="color:var(--amber)">${IC.bolt}</span><b>${delay == null ? "—" : delay.toFixed(1) + " с"}</b><span>от начала до тревоги</span></div></div>
        <div class="gauge">${gaugeSvg(st.rt_factor ? Math.min(st.rt_factor / 4, 1) : 0, "#5ed3c0")}<div class="in"><span>скорость</span><b>${st.rt_factor ? "×" + st.rt_factor : "—"}</b><span>${st.ms_p95 ? "p95 " + Math.round(st.ms_p95) + " мс/кадр" : "от реального времени"}</span></div></div></div>
        <div class="kpis"><div class="kpi"><span class="mut small">Тревог</span><b>${al.length}</b></div><div class="kpi"><span class="mut small">Циклов жеста</span><b>${cy.length}</b><span class="mut small">прошли порог: ${cy.filter(c => c.passed).length}</span></div>
          ${mt ? `<div class="kpi"><span class="mut small">F1 по разметке</span><b>${mt.f1.toFixed(2)}</b><span class="mut small">TP ${mt.tp} · FP ${mt.fp} · FN ${mt.fn}</span></div>` : `<div class="kpi"><span class="mut small">Разметка</span><b>—</b><span class="mut small">для клипа нет эталона</span></div>`}</div>
        <p class="${(st.rt_factor || 9) >= 1 ? "mut" : "err"} small">${esc(st.verdict || "")}</p>
        ${al.length ? `<table><tr><th>Тревога в</th><th>Начало</th><th>ID</th><th>Правило</th><th>Уверенность</th></tr>${al.map(a => `<tr class="jump" data-t="${a.start}"><td>${mmss(a.t_open)}</td><td>${mmss(a.start)}</td><td>${a.tid}</td><td>${esc(ruleText(a.rule))}${a.rule ? "" : esc(a.explain)}</td><td>${scoreBar(a.confidence, th.event)}<span class="small">${pct(a.confidence)}</span></td></tr>`).join("")}</table>` : `<p class="mut">Тревог нет.</p>`}`;
    } else if (tabSel === "triggers") {
      body = `<h3>Что сработало</h3>` + (al.length ? al.map(a => { const mine = cy.filter(c => c.tid === a.tid && c.start >= a.start - 0.5 && c.end <= a.end + 1.5);
        return `<div class="card jump" data-t="${a.start}" style="margin-bottom:10px;cursor:pointer"><div class="row"><b>ID ${a.tid} · ${mmss(a.start)}–${mmss(a.end)}</b><span class="chip bad">${pct(a.confidence)}</span><span class="chip">${esc(ruleText(a.rule))}</span><span class="chip">циклов: ${a.n_cycles ?? mine.length}</span><span class="grow"></span><span class="mut small">тревога на ${mmss(a.t_open)} (через ${a.delay.toFixed(1)} с)</span></div>
        ${mine.map(c => `<div class="row" style="margin-top:8px;flex-wrap:nowrap"><span class="mut small" style="width:90px">цикл ${mmss(c.peak)}</span><div style="width:120px">${scoreBar(c.score, th.cycle)}</div><b class="small" style="width:44px">${pct(c.score)}</b><div>${sigChips(c)}</div></div>`).join("")}</div>`; }).join("") : `<p class="mut">Тревог нет — ни одно правило регламента не выполнилось.</p>`) +
        `<h3 style="margin-top:16px">Все циклы жеста</h3><p class="mut small">Цикл — подъём руки ко рту, пауза, возврат. Классификатор оценивает каждый цикл; белая отметка — порог цикла (${th.cycle != null ? pct(th.cycle) : "—"}).</p>` +
        (cy.length ? `<table><tr><th>Время</th><th>ID</th><th>Оценка цикла</th><th>Решение</th><th>Сигналы</th></tr>${cy.map(c => `<tr class="jump" data-t="${c.start}"><td>${mmss(c.peak)}</td><td>${c.tid}</td><td style="min-width:150px">${scoreBar(c.score, th.cycle)}<span class="small">${pct(c.score)}</span></td><td><span class="chip ${c.passed ? "ok" : ""}">${c.passed ? "курение" : "ниже порога"}</span></td><td>${sigChips(c)}</td></tr>`).join("")}</table>` : `<p class="mut">Циклов жеста не найдено.</p>`);
    } else if (tabSel === "conf") {
      body = `<p class="mut small">Точки — циклы жеста (янтарные прошли порог цикла), красные отрезки — тревоги на уровне их уверенности, зелёная полоса — разметка. Пунктир: порог цикла и порог события.</p><svg class="chart" id="chart" viewBox="0 0 1000 230" preserveAspectRatio="none"></svg>`;
    } else {
      const by = {}; (trace.frames || []).forEach(f => f[1].forEach(d => { const o = by[d[0]] || (by[d[0]] = { tid: d[0], a: f[0], b: f[0], n: 0 }); o.b = f[0]; o.n++; }));
      const rows = Object.values(by).sort((a, b) => a.a - b.a);
      body = rows.length ? `<table><tr><th>ID</th><th>В кадре</th><th>Кадров</th><th>Циклов</th><th>Лучшая оценка</th><th>Тревог</th></tr>${rows.map(r => { const c = cy.filter(x => x.tid === r.tid); const mx = c.length ? Math.max(...c.map(x => x.score)) : null;
        return `<tr class="jump" data-t="${r.a}"><td>${r.tid}</td><td>${mmss(r.a)}–${mmss(r.b)}</td><td>${r.n}</td><td>${c.length}</td><td>${mx == null ? "—" : scoreBar(mx, th.cycle) + `<span class="small">${pct(mx)}</span>`}</td><td>${al.filter(a => a.tid === r.tid).length}</td></tr>`; }).join("")}</table>` : `<p class="mut">Люди в кадре не найдены.</p>`;
    }
    box.innerHTML = `<div class="subtabs">${tabs.map(([k, l]) => `<button data-k="${k}" class="${tabSel === k ? "on" : ""}">${l}</button>`).join("")}</div>${body}`;
    $$("#res .subtabs button").forEach(b => b.onclick = () => { tabSel = b.dataset.k; mem.set("an_tab", tabSel); renderResult(true); });
    $$("#res .jump").forEach(r => r.onclick = () => seek(+r.dataset.t));
    if (tabSel === "conf") drawChart();
  };
  const drawChart = () => {
    const el = $("#chart"); if (!el || !trace) return; const W = 1000, H = 230, L = 36, B = 22, T = 10, X = t => L + (t / total) * (W - L - 8), Y = v => T + (1 - v) * (H - T - B), th = trace.thresholds || {};
    let g = [0, .25, .5, .75, 1].map(v => `<line x1="${L}" x2="${W - 8}" y1="${Y(v)}" y2="${Y(v)}" stroke="#333" stroke-width="1" vector-effect="non-scaling-stroke"/><text x="2" y="${Y(v) + 4}">${Math.round(v * 100)}%</text>`).join("");
    g += (trace.gt || []).map(x => `<rect x="${X(x.start)}" y="${H - B - 6}" width="${Math.max(2, X(x.end) - X(x.start))}" height="6" fill="#7be0a0" opacity=".8"><title>разметка: курение</title></rect>`).join("");
    if (th.cycle != null) g += `<line x1="${L}" x2="${W - 8}" y1="${Y(th.cycle)}" y2="${Y(th.cycle)}" stroke="#f6b44a" stroke-dasharray="6 5" vector-effect="non-scaling-stroke"/>`;
    if (th.event != null) g += `<line x1="${L}" x2="${W - 8}" y1="${Y(th.event)}" y2="${Y(th.event)}" stroke="#ff5a5f" stroke-dasharray="3 5" vector-effect="non-scaling-stroke"/>`;
    g += (trace.alerts || []).map(a => `<line x1="${X(a.start)}" x2="${X(Math.max(a.end, a.start + 1))}" y1="${Y(a.confidence)}" y2="${Y(a.confidence)}" stroke="#ff5a5f" stroke-width="5" stroke-linecap="round" vector-effect="non-scaling-stroke"><title>тревога ${pct(a.confidence)}</title></line>`).join("");
    g += (trace.cycles || []).map(c => `<line x1="${X(c.peak)}" x2="${X(c.peak)}" y1="${Y(0)}" y2="${Y(c.score)}" stroke="${c.passed ? "#f6b44a" : "#77777c"}" stroke-width="1.5" vector-effect="non-scaling-stroke"/><circle cx="${X(c.peak)}" cy="${Y(c.score)}" r="6" fill="${c.passed ? "#f6b44a" : "#77777c"}"><title>цикл ID ${c.tid}: ${pct(c.score)}</title></circle>`).join("");
    for (let t = 0; t <= total; t += Math.max(5, Math.ceil(total / 10 / 5) * 5)) g += `<text x="${X(t)}" y="${H - 4}">${mmss(t)}</text>`;
    el.innerHTML = g + `<line id="ccur" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="#fff" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
    el.onclick = e => { const r = el.getBoundingClientRect(); pv.currentTime = Math.max(0, ((e.clientX - r.left) / r.width * W - L) / (W - L - 8) * total); };
    el._x = X;
  };
  /* ---------- переходы по тревогам, скорость, горячие клавиши */
  const jump = dir => { const al = (trace && trace.alerts) || []; if (!al.length) return toast("Тревог нет"); const c = pv.currentTime;
    const nx = dir > 0 ? al.find(a => a.start - 1 > c + 0.3) : [...al].reverse().find(a => a.start - 1 < c - 1.0); if (nx) seek(nx.start); else toast(dir > 0 ? "Больше тревог нет" : "Раньше тревог нет"); };
  $("#nextA").onclick = () => jump(1); $("#prevA").onclick = () => jump(-1); $("#rate").onchange = () => { pv.playbackRate = +$("#rate").value; };
  onKey(e => { if (typing(e)) return; if (e.code === "Space") { e.preventDefault(); pv.paused ? pv.play().catch(() => { }) : pv.pause(); } else if (e.key === "ArrowRight") pv.currentTime += 5; else if (e.key === "ArrowLeft") pv.currentTime -= 5;
    else if (e.key === "]") jump(1); else if (e.key === "[") jump(-1); else if (e.key.toLowerCase() === "b") $("#tBox").click(); });
  /* ---------- оверлей: рамки, баннер тревоги, текущая уверенность */
  const cv = $("#ov");
  const paint = () => {
    const w = pv.clientWidth, h = pv.clientHeight; if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const g = cv.getContext("2d"); g.clearRect(0, 0, w, h); const t = pv.currentTime; upd(t);
    // кадр вписан в <video> с полями (object-fit: contain): рамки считаются от реального прямоугольника кадра, а не от всего элемента
    const vw = pv.videoWidth || (trace && trace.frame_hw ? trace.frame_hw[1] : 16), vh = pv.videoHeight || (trace && trace.frame_hw ? trace.frame_hw[0] : 9);
    const kk = Math.min(w / vw, h / vh), cw = vw * kk, ch = vh * kk, ox = (w - cw) / 2, oy = (h - ch) / 2; cv.style.left = pv.offsetLeft + "px"; cv.style.top = pv.offsetTop + "px";
    const cc = $("#ccur"), chartEl = $("#chart"); if (cc && chartEl && chartEl._x) { const x = chartEl._x(t); cc.setAttribute("x1", x); cc.setAttribute("x2", x); }
    if (liveJob && liveJob.started) {        // видео не убегает от модели
      const lag = t - liveJob.horizon;
      if (!pv.paused && lag > 0.25) { pv.pause(); liveJob.waiting = true; } else if (liveJob.waiting && liveJob.horizon > t + 1.0) { liveJob.waiting = false; pv.play().catch(() => { }); }
      if (liveJob.waiting) { g.fillStyle = "rgba(0,0,0,.65)"; g.fillRect(0, h / 2 - 18, w, 36); g.fillStyle = "#fff"; g.font = "14px system-ui"; g.fillText(`Модель отстаёт на ${Math.max(0, lag).toFixed(1)} с — ждём…`, 14, h / 2 + 5); }
    }
    if (!trace || !trace.frames.length) return;
    const live = (trace.alerts || []).filter(a => t >= a.start && t <= a.end + 1), th = trace.thresholds || {};
    const chips = []; const lastC = [...(trace.cycles || [])].reverse().find(c => c.end <= t + 0.2 && t - c.end < 4);
    if (lastC) chips.push([`цикл ID ${lastC.tid}: ${pct(lastC.score)}`, lastC.score >= (th.cycle ?? .5) ? "#f6b44a" : "#77777c"]);
    const hot = live.filter(a => t >= a.t_open - 1e-3); if (hot.length) { const a = hot.reduce((m, x) => x.confidence > m.confidence ? x : m); chips.push([`тревога ${pct(a.confidence)}`, "#ff5a5f"]); }
    if ($("#tGt").checked && (trace.gt || []).some(x => t >= x.start && t <= x.end)) chips.push(["разметка: курение", "#3aa66a"]);
    g.font = "bold 13px system-ui"; let x = ox + cw - 10; chips.forEach(([txt, col]) => { const tw = g.measureText(txt).width + 18; x -= tw; g.fillStyle = col; g.beginPath(); g.roundRect(x, oy + 10, tw, 26, 13); g.fill(); g.fillStyle = "#111"; g.fillText(txt, x + 9, oy + 28); x -= 8; });
    if (!$("#tBox").checked) return;
    const fr = trace.frames; let lo = 0, hi = fr.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; fr[m][0] < t ? lo = m + 1 : hi = m; }
    let k = lo; if (k > 0 && Math.abs(fr[k - 1][0] - t) < Math.abs(fr[k][0] - t)) k--; if (Math.abs(fr[k][0] - t) > 2.5 / (trace.fps || 5)) return;
    const [fh, fw] = trace.frame_hw || [pv.videoHeight, pv.videoWidth], sx = cw / (fw || pv.videoWidth || 1), sy = ch / (fh || pv.videoHeight || 1);
    for (const [tid, x1, y1, x2, y2] of fr[k][1]) {
      const al = live.find(a => a.tid === tid), isHot = !!al; g.lineWidth = isHot ? 3 : 1.5; g.strokeStyle = isHot ? "#ff5a5f" : "rgba(255,255,255,.85)"; g.strokeRect(ox + x1 * sx, oy + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      const lbl = isHot ? `ID ${tid} · ${pct(al.confidence)}` : `ID ${tid}`; g.font = "12px system-ui"; const tw = g.measureText(lbl).width + 8; g.fillStyle = isHot ? "#ff5a5f" : "rgba(0,0,0,.6)"; g.fillRect(ox + x1 * sx, Math.max(0, oy + y1 * sy - 18), tw, 18); g.fillStyle = "#fff"; g.fillText(lbl, ox + x1 * sx + 4, Math.max(13, oy + y1 * sy - 5));
    }
  };
  let raf = 0; const loop = () => { paint(); raf = requestAnimationFrame(loop); }; loop(); S.cleanup.push(() => cancelAnimationFrame(raf));
  /* ---------- запуск */
  const bar = p => { $("#prog").classList.remove("hidden"); $("#prog i").style.width = Math.round(p * 100) + "%"; };
  const finish = async () => { const a = await J(`/api/analysis?video=${cur.id}&profile=${encodeURIComponent($("#prof").value)}`); if (a.trace) { const t0 = pv.currentTime, was = !pv.paused; setAnalysis(a.trace, a.meta); pv.currentTime = t0; if (was) pv.play().catch(() => { }); } loadList(); };
  const runLive = async () => {
    $("#run").disabled = $("#runAll").disabled = true; $("#res").classList.add("hidden"); pv.pause(); pv.currentTime = 0; note.textContent = "анализ в реальном времени"; $("#jstat").textContent = "Модель загружается…"; bar(0);
    let j; try { j = await POST("/api/analyze", { video: cur.id, profile: $("#prof").value, live: true }); } catch (e) { $("#jstat").innerHTML = `<span class="err">${esc(e.message)}</span>`; $("#prog").classList.add("hidden"); pinfo(); return; }
    trace = { frames: [], alerts: [], gt: [], cycles: [], thresholds: {}, fps: 10, frame_hw: null }; drawTimeline(trace);
    const me = liveJob = { id: j.id, since: 0, horizon: 0, started: false, waiting: false };
    while (liveJob === me && document.body.contains(pv)) {
      let r; try { r = await J(`/api/job/${me.id}/stream?since=${me.since}`); } catch (e) { $("#jstat").innerHTML = `<span class="err">${esc(e.message)}</span>`; break; }
      if (liveJob !== me) return;
      trace.frames.push(...r.frames); me.since = r.next; me.horizon = r.horizon; trace.fps = r.fps; trace.gt = r.gt; trace.thresholds = r.thresholds; if (r.frame_hw) trace.frame_hw = r.frame_hw;
      const changed = r.alerts.length !== trace.alerts.length || r.cycles.length !== trace.cycles.length; trace.alerts = r.alerts; trace.cycles = r.cycles; if (changed) drawTimeline(trace);
      renderResult(false); bar(r.progress); $("#jstat").textContent = r.state === "queued" ? "в очереди" : (me.started ? `идёт анализ · циклов ${r.cycles.length} · тревог ${r.alerts.length}` : "модель загружается…");
      if (!me.started && r.frames.length && r.horizon > 0.5) { me.started = true; $("#rate").value = "1"; pv.playbackRate = 1; pv.play().catch(() => { }); }
      if (r.state === "error") { $("#jstat").innerHTML = `<span class="err">Ошибка: ${esc(r.error)}</span>`; liveJob = null; break; }
      if (r.state === "done") { liveJob = null; $("#jstat").textContent = "Готово"; await finish(); break; }
      await new Promise(r2 => setTimeout(r2, 300));
    }
    $("#prog").classList.add("hidden"); pinfo();
  };
  $("#live").checked = mem.get("an_live") !== "0";
  const lab = () => { $("#run").textContent = $("#live").checked ? "▶ Смотреть с анализом" : "Анализировать"; }; lab();
  $("#live").onchange = () => { mem.set("an_live", $("#live").checked ? "1" : "0"); lab(); };
  $("#run").onclick = async () => {
    if ($("#live").checked) return runLive();
    $("#run").disabled = $("#runAll").disabled = true; $("#jstat").textContent = "Модель запускается…"; bar(0);
    try { let j = await POST("/api/analyze", { video: cur.id, profile: $("#prof").value }); while (j.state === "queued" || j.state === "running") { await new Promise(r => setTimeout(r, 800)); j = await J("/api/job/" + j.id); if (!document.body.contains(pv)) return; bar(j.progress); $("#jstat").textContent = `${j.state === "queued" ? "в очереди" : "анализ"}: ${Math.round(j.progress * 100)}%`; }
      if (j.state === "error") $("#jstat").innerHTML = `<span class="err">Ошибка: ${esc(j.error)}</span>`; else { $("#jstat").textContent = "Готово"; await finish(); } }
    catch (e) { $("#jstat").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $("#prog").classList.add("hidden"); pinfo();
  };
  $("#runAll").onclick = async () => {
    $("#run").disabled = $("#runAll").disabled = true; const profile = $("#prof").value;
    try { const js = await POST("/api/analyze", { folder: $("#folder").value, profile }); $("#batch").classList.remove("hidden");
      for (;;) { const now = await Promise.all(js.map(j => J("/api/job/" + j.id))); if (!document.body.contains(pv)) return;
        $("#batch").innerHTML = `<h3>Вся папка · ${esc(profile)}</h3><table><tr><th>Видео</th><th>Состояние</th><th>Тревог</th></tr>${now.map(j => `<tr><td>${esc(j.name)}</td><td class="${j.state === "error" ? "err" : ""}">${j.state === "error" ? esc(j.error) : { queued: "в очереди", running: Math.round(j.progress * 100) + "%", done: "готово" }[j.state]}</td><td>${j.state === "done" ? j.alerts.length : ""}</td></tr>`).join("")}</table>`;
        if (now.every(j => j.state === "done" || j.state === "error")) break; await new Promise(r => setTimeout(r, 1200)); }
      await loadList(); if (cur) { const a = await J(`/api/analysis?video=${cur.id}`); if (a.trace) setAnalysis(a.trace, a.meta); }
    } catch (e) { $("#batch").classList.remove("hidden"); $("#batch").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    pinfo();
  };
  /* ---------- старт */
  let f0 = mem.get("an_folder"); if (q.vid) { try { f0 = (await J("/api/folder-of?video=" + q.vid)).folder; } catch { } }
  fillFolders(f0 && folders.some(f => f.id === f0) ? f0 : null);
  $("#folder").onchange = async () => { cur = null; await loadList(); if (vs[0]) select(vs[0]); };
  await loadList(); pinfo();
  const v0 = (q.vid && vs.find(v => v.id === q.vid)) || vs[0]; if (v0) await select(v0);
  if (q.auto && cur && !trace && !$("#run").disabled) { history.replaceState(null, "", `#/analysis?vid=${cur.id}`); $("#run").click(); }
}
