// Разбор видео: выбрать файл и профиль, запустить, смотреть результат поверх видео (рамки людей, скелет, тревоги), таймлайн, циклы жеста, график уверенности.
import { h, s, clear, put, icon, clock, pct, num, plural, dur, fmtDateTime } from "../dom.js";
import { api, errText } from "../api.js";
import { session } from "../session.js";
import { button, card, chip, empty, errorBox, pageHead, skeleton, tabs, toast, stubBox } from "../ui.js";
import { waitForMedia } from "../media.js";

const SKEL = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]];
const KP_MIN = 25;
const kpColor = (c) => (c >= 70 ? "#9BE3BC" : c >= 40 ? "#F2C46D" : "#F08A7A");
const JOB_RU = { queued: "в очереди", running: "считается", done: "готово", error: "ошибка" };
const pref = {
  get(k) { try { return localStorage.getItem(`sd_${k}`) || ""; } catch { return ""; } },
  set(k, v) { try { localStorage.setItem(`sd_${k}`, v); } catch { /* не критично */ } },
};

const ruleText = (r) => ({ two_cycles: "Два цикла подряд", "cycle+object": "Цикл и предмет у рта", both: "Два цикла и предмет у рта" }[r] || r || "—");

export default async function analyze(root, ctx) {
  const canRun = session.can("operator");
  const st = { folders: [], folder: ctx.query.folder || "", videos: [], video: ctx.query.video || "", profiles: [], profile: pref.get("profile"), trace: null, meta: null, tab: "summary", jobs: [], job: null, duration: 0 };
  const aborts = new Set();
  ctx.cleanup(() => aborts.forEach((a) => a.abort()));
  const timers = new Set();
  ctx.cleanup(() => timers.forEach(clearTimeout));
  let raf = 0;
  ctx.cleanup(() => cancelAnimationFrame(raf));

  const left = h("div", { class: "stack" }, skeleton(2, 100));
  const right = h("div", { class: "stack" });
  root.append(pageHead("Разбор видео", "Модель просматривает запись и показывает, где и почему она увидела курение."), h("div", { class: "cols" }, h("div", { class: "f1" }, left), h("div", { class: "f2" }, right)));

  const profileOf = (name) => st.profiles.find((p) => p.name === name);

  // ───────── загрузка справочников ─────────
  async function init() {
    // профили считаются несколько секунд (проверка устройств): список файлов показываем сразу, профили догружаем рядом
    st.profLoading = true;
    const profiles = api.get("/analysis/profiles").then((p) => {
      st.profiles = p;
      const ready = p.filter((x) => x.ready);
      if (!profileOf(st.profile) || !profileOf(st.profile).ready) st.profile = ready.length ? ready[0].name : "";
    }).catch((e) => { toast(`Профили недоступны: ${errText(e)}`, { kind: "alarm" }); }).finally(() => { st.profLoading = false; });
    try { st.folders = await api.get("/sources/folders"); } catch (e) { clear(left).append(errorBox(errText(e), init)); return; }
    if (!ctx.active()) return;
    if (st.video && !st.folder) { try { st.folder = (await api.get("/sources/folder-of", { video: st.video })).folder; } catch { st.video = ""; } }
    if (!st.folder && st.folders.length) st.folder = st.folders[0].id;
    await loadVideos();
    if (!ctx.active()) return;
    drawLeft();
    drawRight();
    refreshJobs();
    if (st.video) selectVideo(st.video, false);
    await profiles;
    if (!ctx.active()) return;
    drawLeft();
    drawRight();
    if (st.video) loadResult({ any: true });
  }

  async function loadVideos() {
    st.videos = [];
    if (!st.folder) return;
    try { st.videos = await api.get("/sources/videos", { folder: st.folder }); } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  // ───────── левая колонка ─────────
  function drawLeft() {
    clear(left);
    if (!st.folders.length) {
      left.append(card(null, empty("Источников видео нет", session.can("operator") ? "Загрузите файл ниже или подключите папку в разделе «Администрирование»." : "Источники подключает администратор.")), uploadBox());
      return;
    }
    const sel = h("select", { class: "input", "aria-label": "Папка с видео", onChange: async (e) => { st.folder = e.target.value; st.video = ""; await loadVideos(); drawLeft(); drawRight(); } },
      ...st.folders.map((f) => h("option", { value: f.id, selected: f.id === st.folder }, `${f.name} (${f.n})`)));
    const list = h("div", { class: "vlist", role: "listbox", "aria-label": "Видео" });
    if (!st.videos.length) list.append(h("p", { class: "muted small", style: { padding: "8px" } }, "В папке нет видео."));
    st.videos.forEach((v) => list.append(h("button", { class: ["vitem", v.id === st.video && "is-active"], type: "button", role: "option", "aria-selected": String(v.id === st.video), onClick: () => selectVideo(v.id) },
      icon("video", 18), h("div", { class: "grow" }, h("b", { class: "ellipsis", style: { display: "block" } }, v.name), h("span", { class: "small muted" }, [v.duration ? dur(v.duration) : null, v.size_mb ? `${v.size_mb} МБ` : null].filter(Boolean).join(" · "))),
      v.analysis ? chip(`${v.analysis.alerts ?? 0} ${plural(v.analysis.alerts ?? 0, "тревога", "тревоги", "тревог")}`, v.analysis.alerts ? "alarm" : "ok") : null)));
    left.append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Видео")), h("div", { class: "stack" }, sel, list,
      canRun && st.videos.length ? button({ label: "Разобрать всю папку", ic: "play", block: true, onClick: runFolder, disabled: !st.profile }) : null)), uploadBox(), jobsBox());
  }

  function uploadBox() {
    if (!canRun) return h("div");
    const input = h("input", { type: "file", accept: "video/*,.mkv,.avi,.mov,.mp4", hidden: true });
    const dz = h("div", { class: "dz", role: "button", tabindex: "0", "aria-label": "Загрузить видео" }, icon("upload", 22), h("div", null, h("b", null, "Загрузить видео"), h("div", { class: "small muted" }, "Перетащите файл сюда или нажмите")), input);
    const pick = () => input.click();
    dz.addEventListener("click", (e) => { if (e.target !== input) pick(); });
    dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-over"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-over"); }));
    dz.addEventListener("drop", (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) upload(f, dz); });
    input.addEventListener("change", () => { if (input.files[0]) upload(input.files[0], dz); input.value = ""; });
    return dz;
  }

  async function upload(file, dz) {
    dz.style.opacity = ".6";
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await api.upload("/sources/upload", fd);
      toast(`Загружено: ${r.name}`, { kind: "ok" });
      st.folders = await api.get("/sources/folders");
      st.folder = r.folder;
      await loadVideos();
      drawLeft();
      selectVideo(r.id);
    } catch (e) { toast(errText(e), { kind: "alarm" }); dz.style.opacity = ""; }
  }

  // ───────── очередь разбора ─────────
  const jobsHolder = h("div");
  function jobsBox() {
    paintJobs();
    return jobsHolder;
  }
  function paintJobs() {
    clear(jobsHolder);
    if (!st.jobs.length) return;
    jobsHolder.append(card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Очередь разбора")),
      h("div", { class: "stack stack--sm" }, ...st.jobs.slice(0, 8).map((j) => h("button", { class: "vitem", type: "button", onClick: () => openJob(j), title: j.error || "" },
        h("div", { class: "grow" }, h("b", { class: "ellipsis", style: { display: "block", fontSize: "13px" } }, j.name), h("span", { class: "small muted" }, `${j.profile}${j.state === "running" ? ` · ${Math.round(j.progress * 100)} %` : ""}`)),
        chip(JOB_RU[j.state] || j.state, j.state === "done" ? "ok" : j.state === "error" ? "alarm" : j.state === "running" ? "info" : ""))))));
  }
  async function refreshJobs() {
    try { st.jobs = await api.get("/analysis/jobs", { limit: 10 }); } catch { return; }
    if (!ctx.active()) return;
    paintJobs();
    const cur = st.job && st.jobs.find((j) => j.id === st.job.id);
    if (cur) { const was = st.job.state; st.job = cur; paintJobStatus(); if (was !== "done" && cur.state === "done") { st.profile = cur.profile; loadResult(); } }
    if (st.jobs.some((j) => j.state === "queued" || j.state === "running")) timers.add(setTimeout(refreshJobs, 1500));
  }
  async function openJob(j) {
    if (j.state !== "done") { toast(j.state === "error" ? (j.error || "Разбор завершился ошибкой") : "Разбор ещё не закончен", { kind: j.state === "error" ? "alarm" : "" }); return; }
    st.profile = j.profile;
    await selectVideo(j.video);
  }

  async function runFolder() {
    try {
      const jobs = await api.post("/analysis/jobs", { folder: st.folder, profile: st.profile });
      toast(`В очередь поставлено: ${jobs.length}`, { kind: "ok" });
      refreshJobs();
    } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  async function run() {
    try {
      st.job = await api.post("/analysis/jobs", { video: st.video, profile: st.profile });
      paintJobStatus();
      refreshJobs();
    } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  // ───────── выбор видео и результат ─────────
  async function selectVideo(id, push = true) {
    st.video = id;
    st.trace = st.meta = null;
    st.job = null;
    if (push) ctx.setQuery({ video: id, folder: st.folder });
    if (!st.videos.some((v) => v.id === id)) {
      try { st.folder = (await api.get("/sources/folder-of", { video: id })).folder; await loadVideos(); } catch { /* список останется прежним */ }
    }
    drawLeft();
    drawRight();
    loadResult({ any: true });
  }

  /** any — искать самый свежий разбор файла любым профилем (при выборе видео); иначе строго выбранным профилем. */
  async function loadResult({ any = false } = {}) {
    if (!st.video) return;
    const my = st.video;
    try {
      const r = await api.get("/analysis/result", { video: st.video, profile: any ? undefined : st.profile || undefined });
      if (!ctx.active() || my !== st.video) return;
      st.trace = r.trace;
      st.meta = r.meta;
      if (any && r.meta && r.meta.profile && r.meta.profile !== st.profile && profileOf(r.meta.profile)) {
        st.profile = r.meta.profile;
        const sel = right.querySelector('select[aria-label="Профиль"]');
        if (sel) sel.value = st.profile;
      }
      drawResults();
      drawTimeline();
    } catch (e) { if (ctx.active() && my === st.video) resultsBox.replaceChildren(errorBox(errText(e), () => loadResult({ any }))); }
  }

  // ───────── правая колонка ─────────
  const statusLine = h("div");
  const resultsBox = h("div", { class: "stack" });
  const timelineBox = h("div");
  let video, canvas, playerBox, flags = { box: true, kp: true, gt: true };

  function drawRight() {
    aborts.forEach((a) => a.abort());
    aborts.clear();
    cancelAnimationFrame(raf);
    clear(right);
    if (!st.video) { right.append(card(null, empty("Выберите видео", "Слева список файлов. Результат прошлого разбора откроется сразу."))); return; }
    const v = st.videos.find((x) => x.id === st.video);
    const profSel = h("select", { class: "input input--sm", style: { width: "auto", minWidth: "240px" }, "aria-label": "Профиль", onChange: (e) => { st.profile = e.target.value; pref.set("profile", st.profile); st.trace = st.meta = null; drawResults(); drawTimeline(); drawLeft(); loadResult(); } },
      ...(st.profiles.length ? st.profiles : [{ name: "", label: st.profLoading ? "загрузка профилей…" : "профилей нет", ready: false }]).map((p) => h("option", { value: p.name, disabled: !p.ready, selected: p.name === st.profile }, `${p.label}${p.quality && p.quality.f1 != null ? ` · F1 ${num(p.quality.f1)}` : ""}${p.ready ? "" : " — не готов"}`)));
    const desc = (profileOf(st.profile) || {}).describe || {};
    const head = card(null, h("div", { class: "card__head" }, h("div", { class: "grow" }, h("h2", { class: "ellipsis" }, v ? v.name : "Видео"), h("span", { class: "small muted" }, [desc.fps ? `${desc.fps} к/с` : null, desc.cycle_model, desc.pose].filter(Boolean).join(" · "))), profSel,
      canRun ? button({ label: "Запустить разбор", ic: "play", kind: "accent", disabled: !st.profile, onClick: run }) : null), statusLine);
    paintJobStatus();

    playerBox = h("div", { class: "player" });
    video = h("video", { controls: true, playsinline: true, preload: "metadata", "aria-label": "Видео для разбора" });
    video.muted = true;
    canvas = h("canvas", { "aria-hidden": "true" });
    const loading = h("div", { class: "banner banner--info", style: { margin: "12px" } }, "Готовлю видео для просмотра…");
    playerBox.append(video, canvas, loading);
    const ac = new AbortController();
    aborts.add(ac);
    const mine = st.video;
    waitForMedia("video", st.video, { signal: ac.signal, onProgress: (p) => { loading.textContent = `Готовлю видео для просмотра… ${Math.round(p * 100)} %`; } })
      .then((url) => { if (mine === st.video) { video.src = url; loading.remove(); } })
      .catch((e) => { if (e.name !== "AbortError") { loading.className = "banner banner--alarm"; loading.textContent = errText(e); } });
    video.addEventListener("loadedmetadata", () => { st.duration = video.duration || 0; drawTimeline(); drawResults(); });
    video.addEventListener("error", () => { if (video.src) { loading.className = "banner banner--alarm"; loading.textContent = "Браузер не смог воспроизвести это видео."; playerBox.append(loading); } });

    const toggles = h("div", { class: "row" },
      toggle("Рамки людей", "box"), toggle("Скелет", "kp"), toggle("Разметка", "gt"),
      h("label", { class: "row row--nowrap small", style: { gap: "6px" } }, "Скорость", h("select", { class: "input input--sm", style: { width: "auto" }, "aria-label": "Скорость", onChange: (e) => { video.playbackRate = Number(e.target.value); } }, ...[0.25, 0.5, 1, 2].map((r) => h("option", { value: r, selected: r === 1 }, `×${r}`)))),
      h("span", { class: "grow" }), button({ ic: "chevl", label: "Прошлая тревога", size: "sm", onClick: () => jumpAlert(-1) }), button({ ic: "chevr", label: "Следующая тревога", size: "sm", onClick: () => jumpAlert(1) }));

    right.append(head, card(null, playerBox, timelineBox, h("div", { style: { marginTop: "12px" } }, toggles)), resultsBox);
    drawTimeline();
    drawResults();
    loop();
  }

  const toggle = (label, key) => h("label", { class: "check", style: { alignItems: "center" } }, h("input", { type: "checkbox", checked: flags[key], onChange: (e) => { flags[key] = e.target.checked; paint(); } }), label);

  function paintJobStatus() {
    clear(statusLine);
    const j = st.job;
    if (!j) return;
    if (j.state === "error") statusLine.append(h("div", { class: "banner banner--alarm", style: { marginTop: "12px" } }, j.error || "Разбор завершился ошибкой"));
    else if (j.state !== "done") statusLine.append(h("div", { style: { marginTop: "12px" } }, h("div", { class: "small muted" }, `${JOB_RU[j.state]}… ${Math.round(j.progress * 100)} %`), h("div", { class: "bar bar--accent", style: { marginTop: "6px" } }, h("i", { style: { width: `${Math.round(j.progress * 100)}%` } }))));
  }

  // ───────── переходы по тревогам ─────────
  function jumpAlert(dir) {
    const al = (st.trace && st.trace.alerts) || [];
    if (!al.length) { toast("В этом разборе тревог нет"); return; }
    const t = video.currentTime;
    const starts = al.map((a) => a.start).sort((a, b) => a - b);
    const target = dir > 0 ? starts.find((x) => x > t + 0.5) : [...starts].reverse().find((x) => x < t - 1.5);
    if (target === undefined) { toast(dir > 0 ? "Дальше тревог нет" : "Раньше тревог нет"); return; }
    seek(target);
  }
  function seek(t) { if (!video) return; video.currentTime = Math.max(0, t - 1); video.play().catch(() => {}); }

  // ───────── таймлайн ─────────
  const total = () => st.duration || (st.trace && st.trace.frames && st.trace.frames.length ? st.trace.frames[st.trace.frames.length - 1][0] : 0) || Math.max(1, ...((st.trace && st.trace.alerts) || []).map((a) => a.end));
  let head;
  function drawTimeline() {
    if (!timelineBox) return;
    clear(timelineBox);
    const tr = st.trace;
    const T = total();
    const wrap = h("div", { class: "timeline", role: "slider", "aria-label": "Положение в видео", tabindex: "0" }, h("div", { class: "timeline__rail" }));
    const pos = (t) => `${Math.max(0, Math.min(100, (t / T) * 100))}%`;
    if (tr && T) {
      (tr.gt || []).forEach((g) => wrap.append(h("div", { class: "timeline__gt", style: { left: pos(g.start), width: `calc(${pos(g.end)} - ${pos(g.start)})` }, title: "разметка: курение" })));
      (tr.alerts || []).forEach((a) => wrap.append(h("div", { class: "timeline__seg", style: { left: pos(a.start), width: `max(6px, calc(${pos(a.end)} - ${pos(a.start)}))` }, title: `Тревога ${pct(a.confidence)}` })));
    }
    head = h("div", { class: "timeline__head" });
    wrap.append(head);
    wrap.addEventListener("click", (e) => { const r = wrap.getBoundingClientRect(); if (video && T) { video.currentTime = ((e.clientX - r.left) / r.width) * T; } });
    wrap.addEventListener("keydown", (e) => { if (!video) return; if (e.key === "ArrowRight") video.currentTime += 2; if (e.key === "ArrowLeft") video.currentTime -= 2; });
    timelineBox.append(wrap, h("div", { class: "row small muted", style: { justifyContent: "space-between" } }, h("span", null, clock(0)), tr ? h("span", { class: "row", style: { gap: "14px" } }, legendItem("#F0B6A8", "тревога"), (tr.gt || []).length ? legendItem("var(--accent)", "разметка") : null) : h("span", null, "результата ещё нет"), h("span", null, clock(T))));
  }
  const legendItem = (color, text) => h("span", null, h("i", { style: { display: "inline-block", width: "10px", height: "10px", background: color, marginRight: "6px" } }), text);

  // ───────── наложение на видео ─────────
  function nearestFrame(frames, t) {
    let lo = 0, hi = frames.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; frames[m][0] < t ? (lo = m + 1) : (hi = m); }
    if (lo > 0 && Math.abs(frames[lo - 1][0] - t) < Math.abs(frames[lo][0] - t)) lo--;
    return frames[lo];
  }

  function paint() {
    if (!video || !canvas || !canvas.isConnected) return;
    const w = video.clientWidth, hgt = video.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hgt * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(hgt * dpr); canvas.style.width = `${w}px`; canvas.style.height = `${hgt}px`; }
    canvas.style.left = `${video.offsetLeft}px`;
    canvas.style.top = `${video.offsetTop}px`;
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, hgt);
    const t = video.currentTime;
    const T = total();
    if (head && T) head.style.left = `${Math.max(0, Math.min(100, (t / T) * 100))}%`;
    const tr = st.trace;
    if (!tr || !tr.frames || !tr.frames.length) return;
    const [fh, fw] = tr.frame_hw || [video.videoHeight, video.videoWidth];
    const vw = video.videoWidth || fw || 16, vh = video.videoHeight || fh || 9;
    const k = Math.min(w / vw, hgt / vh), cw = vw * k, ch = vh * k, ox = (w - cw) / 2, oy = (hgt - ch) / 2;
    const sx = cw / (fw || vw), sy = ch / (fh || vh);
    const th = tr.thresholds || {};
    const live = (tr.alerts || []).filter((a) => t >= a.start && t <= a.end + 1);
    const chips = [];
    const lastC = [...(tr.cycles || [])].reverse().find((c) => c.end <= t + 0.2 && t - c.end < 4);
    if (lastC) chips.push([`цикл ID ${lastC.tid}: ${pct(lastC.score)}`, lastC.score >= (th.cycle ?? 0.5) ? "#F2C46D" : "#8A958D"]);
    const hot = live.filter((a) => t >= (a.t_open ?? a.start) - 1e-3);
    if (hot.length) { const a = hot.reduce((m, x) => (x.confidence > m.confidence ? x : m)); chips.push([`тревога ${pct(a.confidence)}`, "#F0705A"]); }
    if (flags.gt && (tr.gt || []).some((x) => t >= x.start && t <= x.end)) chips.push(["разметка: курение", "#5FD39B"]);
    g.font = '600 13px Geologica, "Segoe UI", sans-serif';
    let x = ox + cw - 10;
    for (const [txt, col] of chips) { const tw = g.measureText(txt).width + 16; x -= tw; g.fillStyle = col; g.fillRect(x, oy + 10, tw, 24); g.fillStyle = "#0E1610"; g.fillText(txt, x + 8, oy + 27); x -= 6; }
    if (!flags.box) return;
    const fr = nearestFrame(tr.frames, t);
    if (!fr || Math.abs(fr[0] - t) > 2.5 / (tr.fps || 5)) return;
    for (const [tid, x1, y1, x2, y2, kp] of fr[1]) {
      const al = live.find((a) => a.tid === tid), isHot = !!al;
      g.lineWidth = isHot ? 3 : 1.5;
      g.strokeStyle = isHot ? "#F0705A" : "rgba(255,255,255,.8)";
      g.strokeRect(ox + x1 * sx, oy + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      const label = isHot ? `ID ${tid} · ${pct(al.confidence)}` : `ID ${tid}`;
      g.font = '500 12px Geologica, "Segoe UI", sans-serif';
      const tw = g.measureText(label).width + 10, ly = Math.max(0, oy + y1 * sy - 19);
      g.fillStyle = isHot ? "#F0705A" : "rgba(14,22,16,.75)";
      g.fillRect(ox + x1 * sx, ly, tw, 19);
      g.fillStyle = isHot ? "#0E1610" : "#fff";
      g.fillText(label, ox + x1 * sx + 5, ly + 14);
      if (flags.kp && kp && kp.length >= 51) {
        const P = (i) => [ox + kp[i * 3] * sx, oy + kp[i * 3 + 1] * sy, kp[i * 3 + 2]];
        g.lineWidth = 2; g.lineCap = "round";
        for (const [a, b] of SKEL) { const p = P(a), q = P(b); if (p[2] < KP_MIN || q[2] < KP_MIN) continue; g.strokeStyle = isHot ? "rgba(240,112,90,.9)" : "rgba(95,211,155,.8)"; g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); }
        const nose = P(0);
        if (isHot) for (const wi of [9, 10]) { const p = P(wi); if (p[2] >= KP_MIN && nose[2] >= KP_MIN) { g.setLineDash([4, 4]); g.strokeStyle = "rgba(242,196,109,.95)"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(nose[0], nose[1]); g.stroke(); g.setLineDash([]); } }
        for (let i = 0; i < 17; i++) { const p = P(i); if (p[2] < KP_MIN) continue; const big = i === 0 || i === 9 || i === 10; g.fillStyle = kpColor(p[2]); g.beginPath(); g.arc(p[0], p[1], big ? 4.5 : 3, 0, Math.PI * 2); g.fill(); }
      }
    }
  }
  function loop() { paint(); raf = requestAnimationFrame(loop); }

  // ───────── результат ─────────
  function drawResults() {
    clear(resultsBox);
    if (!st.trace) {
      resultsBox.append(card({ class: "card--flat" }, h("p", { class: "muted" }, st.job && st.job.state !== "done" ? "Идёт разбор — результат появится здесь." : "Для этого видео и профиля разбора ещё нет."),
        canRun ? button({ label: "Запустить разбор", ic: "play", kind: "accent", size: "sm", disabled: !st.profile, onClick: run }) : null));
      return;
    }
    const body = h("div");
    const paintTab = () => {
      clear(body);
      ({ summary: tabSummary, cycles: tabCycles, chart: tabChart, people: tabPeople }[st.tab] || tabSummary)(body);
    };
    const m0 = st.meta || {};
    resultsBox.append(h("p", { class: "small muted" }, `Результат: профиль «${m0.profile || "—"}»${m0.created ? `, ${fmtDateTime(m0.created)}` : ""}`), card(null, tabs([["summary", "Итог"], ["cycles", "Циклы жеста"], ["chart", "График"], ["people", "Люди"]], st.tab, (v) => { st.tab = v; paintTab(); }), body),
      stubBox("Просмотр во время расчёта", ["Видео и рамки появятся по мере работы модели, не дожидаясь конца разбора", "Данные уже отдаёт сервер: поток кадров и тревог задания"], { api: "GET /analysis/jobs/{id}/stream" }));
    paintTab();
  }

  const seekRow = (t, cells) => h("tr", { class: "is-click", tabindex: "0", onClick: () => seek(t), onKeydown: (e) => { if (e.key === "Enter") seek(t); } }, ...cells);

  function tabSummary(body) {
    const tr = st.trace, m = st.meta || {}, stt = m.stats || {}, mt = m.metrics, al = tr.alerts || [], cy = tr.cycles || [], th = tr.thresholds || {};
    const avg = al.length ? al.reduce((a, x) => a + x.confidence, 0) / al.length : null;
    const kp = (l, v, n) => card({ class: "card--tight" }, h("div", { class: "kpi" }, h("div", { class: "kpi__l" }, l), h("div", { class: "kpi__v", style: { fontSize: "28px" } }, v), h("div", { class: "kpi__n" }, n)));
    put(body, h("div", { class: "kpis", style: { marginBottom: "16px" } },
      kp("Тревог", al.length, avg == null ? "тревог нет" : `средняя уверенность ${pct(avg)}`), kp("Циклов жеста", cy.length, `прошли порог: ${cy.filter((c) => c.passed).length}`),
      kp("F1 по разметке", mt ? num(mt.f1) : "—", mt ? `TP ${mt.tp} · FP ${mt.fp} · FN ${mt.fn}` : "для клипа нет эталона"), kp("Скорость", stt.rt_factor ? `×${stt.rt_factor}` : "—", stt.ms_p95 ? `p95 ${Math.round(stt.ms_p95)} мс/кадр` : "от реального времени")),
      stt.verdict ? h("p", { class: "small muted", style: { marginBottom: "12px" } }, stt.verdict) : null,
      al.length ? h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Тревога в", "Начало", "ID", "Правило", "Уверенность"].map((c) => h("th", null, c)))),
        h("tbody", null, ...al.map((a) => seekRow(a.start, [h("td", null, clock(a.t_open ?? a.start)), h("td", null, clock(a.start)), h("td", null, a.tid), h("td", null, a.rule ? ruleText(a.rule) : a.explain || "—"),
          h("td", { style: { minWidth: "150px" } }, h("div", { class: "row row--nowrap" }, h("div", { class: ["bar grow", "bar--alarm"] }, h("i", { style: { width: `${Math.round(a.confidence * 100)}%` } })), h("b", { class: "small" }, pct(a.confidence))))]))))) : h("p", { class: "muted" }, "Тревог нет: ни одно правило не выполнилось."));
  }

  function tabCycles(body) {
    const cy = st.trace.cycles || [], th = (st.trace.thresholds || {}).cycle;
    body.append(h("p", { class: "small muted", style: { marginBottom: "10px" } }, `Цикл — подъём руки ко рту, пауза, возврат. Классификатор оценивает каждый цикл; порог — ${th == null ? "—" : pct(th)}.`),
      cy.length ? h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Время", "ID", "Оценка", "Решение"].map((c) => h("th", null, c)))),
        h("tbody", null, ...cy.map((c) => seekRow(c.start ?? c.peak, [h("td", null, clock(c.peak)), h("td", null, c.tid), h("td", { style: { minWidth: "150px" } }, h("div", { class: "row row--nowrap" }, h("div", { class: "bar grow" }, h("i", { style: { width: `${Math.round(c.score * 100)}%` } })), h("b", { class: "small" }, pct(c.score)))), h("td", null, chip(c.passed ? "курение" : "ниже порога", c.passed ? "alarm" : ""))]))))) : h("p", { class: "muted" }, "Циклов жеста не найдено."));
  }

  function tabChart(body) {
    const tr = st.trace, T = total() || 1, W = 1000, H = 240, L = 40, B = 24, Tp = 10, th = tr.thresholds || {};
    const X = (t) => L + (t / T) * (W - L - 8), Y = (v) => Tp + (1 - v) * (H - Tp - B);
    const kids = [];
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => kids.push(s("line", { x1: L, x2: W - 8, y1: Y(v), y2: Y(v), stroke: "#E3E8DD" }), s("text", { x: 4, y: Y(v) + 4, "font-size": 11, fill: "#5A665E" }, `${Math.round(v * 100)}`)));
    const tick = Math.max(1, Math.ceil(T / 8 / 5) * 5);
    for (let t = 0; t <= T; t += tick) kids.push(s("text", { x: X(t), y: H - 6, "font-size": 11, "text-anchor": "middle", fill: "#5A665E" }, clock(t)));
    if (th.cycle != null) kids.push(s("line", { x1: L, x2: W - 8, y1: Y(th.cycle), y2: Y(th.cycle), stroke: "#0E1610", "stroke-dasharray": "5 4" }));
    if (th.event != null) kids.push(s("line", { x1: L, x2: W - 8, y1: Y(th.event), y2: Y(th.event), stroke: "#D93A1B", "stroke-dasharray": "2 4" }));
    (tr.gt || []).forEach((g) => kids.push(s("rect", { x: X(g.start), y: H - B - 7, width: Math.max(2, X(g.end) - X(g.start)), height: 7, fill: "#0B7A4B" }, s("title", null, "разметка: курение"))));
    (tr.alerts || []).forEach((a) => kids.push(s("line", { x1: X(a.start), x2: X(Math.max(a.end, a.start + 1)), y1: Y(a.confidence), y2: Y(a.confidence), stroke: "#D93A1B", "stroke-width": 5 }, s("title", null, `тревога ${pct(a.confidence)}`))));
    (tr.cycles || []).forEach((c) => {
      const col = c.passed ? "#0E1610" : "#8A958D";
      kids.push(s("line", { x1: X(c.peak), x2: X(c.peak), y1: Y(0), y2: Y(c.score), stroke: col, "stroke-width": 1.5 }), s("rect", { x: X(c.peak) - 5, y: Y(c.score) - 5, width: 10, height: 10, fill: col }, s("title", null, `цикл ID ${c.tid}: ${pct(c.score)}`)));
    });
    body.append(h("p", { class: "small muted", style: { marginBottom: "8px" } }, "Квадраты — циклы жеста (тёмные прошли порог цикла), красные отрезки — тревоги на уровне их уверенности, зелёная полоса — разметка. Пунктир: порог цикла (чёрный) и события (красный)."),
      s("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Уверенность по времени" }, ...kids));
  }

  function tabPeople(body) {
    const tr = st.trace, by = new Map();
    (tr.frames || []).forEach((f) => f[1].forEach((d) => { const o = by.get(d[0]) || { tid: d[0], a: f[0], b: f[0], n: 0 }; o.b = f[0]; o.n++; by.set(d[0], o); }));
    const rows = [...by.values()].sort((a, b) => a.a - b.a);
    if (!rows.length) { body.append(h("p", { class: "muted" }, "Люди в кадре не найдены.")); return; }
    body.append(h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["ID", "В кадре", "Кадров", "Циклов", "Лучшая оценка", "Тревог"].map((c) => h("th", null, c)))),
      h("tbody", null, ...rows.map((r) => {
        const c = (tr.cycles || []).filter((x) => x.tid === r.tid), best = c.length ? Math.max(...c.map((x) => x.score)) : null;
        return seekRow(r.a, [h("td", null, r.tid), h("td", null, `${clock(r.a)}–${clock(r.b)}`), h("td", null, r.n), h("td", null, c.length), h("td", null, best == null ? "—" : pct(best)), h("td", null, (tr.alerts || []).filter((a) => a.tid === r.tid).length)]);
      })))));
  }

  await init();
}
