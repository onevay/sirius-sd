"use strict";
/* Модели (профили, реестр, доработка), Передача (решатель, конфигурации, пакеты), Источники (папки, загрузка, камеры) */
const field = (label, inner, cls = "") => `<label class="fld ${cls}"><span>${label}</span>${inner}</label>`;
const opt = (items, cur, fmt) => items.map(x => { const v = typeof x === "object" ? x.id : x; const l = fmt ? fmt(x) : (typeof x === "object" ? x.label : x); return `<option value="${esc(v)}" ${String(v) === String(cur) ? "selected" : ""}>${esc(l)}</option>`; }).join("");
const issueHtml = i => `<div class="notice" style="margin-bottom:6px;${i.level === "error" ? "background:var(--badbg)" : ""}">${IC.warn}<div><b>${i.level === "error" ? "Ошибка" : "Предупреждение"}</b> &nbsp;<span>${esc(i.text)}</span></div></div>`;
const rawPost = async (url, file, params = {}) => { const u = url + (url.includes("?") ? "&" : "?") + new URLSearchParams(params); const r = await fetch(u, { method: "POST", body: file }); const d = await r.json().catch(() => ({})); if (!r.ok) throw new Error(d.error || r.statusText); return d; };
function dropzone(el, text, accept, onFiles, multi = false) {
  el.classList.add("dz"); el.innerHTML = `${text}<input type="file" accept="${accept}" class="hidden" ${multi ? "multiple" : ""}>`; const inp = $("input", el);
  el.onclick = e => { if (e.target !== inp) inp.click(); }; inp.onchange = () => inp.files.length && onFiles([...inp.files]);
  ["dragover", "dragenter"].forEach(ev => el.addEventListener(ev, e => { e.preventDefault(); el.classList.add("over"); })); ["dragleave", "drop"].forEach(ev => el.addEventListener(ev, e => { e.preventDefault(); el.classList.remove("over"); }));
  el.addEventListener("drop", e => e.dataTransfer.files.length && onFiles([...e.dataTransfer.files]));
}
async function uploadMany(files, out) {
  out.innerHTML = ""; let ok = 0;
  for (const f of files) { const row = document.createElement("div"); row.innerHTML = `${esc(f.name)}<div class="prog"><i></i></div>`; out.appendChild(row);
    try { await uploadOne(f, p => { $("i", row).style.width = Math.round(p * 100) + "%"; }); row.innerHTML = `${esc(f.name)} <span class="ok">загружено</span>`; ok++; } catch (e) { row.innerHTML = `${esc(f.name)} <span class="err">${esc(e.message)}</span>`; } }
  if (ok) toast(`Загружено файлов: ${ok}`);
}

/* ================================================================= МОДЕЛИ */
const MTABS = [["profiles", "Профили"], ["classifier", "Классификатор"], ["validate", "Проверка на данных"], ["registry", "Модели и задачи"]];
const modelTabs = tab => `<div class="pagetabs">${MTABS.map(([k, l]) => `<a href="#/models?tab=${k}" class="${k === tab ? "on" : ""}">${l}</a>`).join("")}</div>`;
async function vModels(q) {
  const tab = MTABS.some(t => t[0] === q.tab) ? q.tab : "profiles";
  if (tab === "classifier") return vClassifier(q);
  if (tab === "validate") return vValidate(q);
  if (tab === "registry") { view.innerHTML = modelTabs(tab) + `<div class="panel" id="registry"></div><div class="panel" style="margin-top:16px" id="tasks"></div>`; const cat = await J("/api/catalog"); registryPanel(cat); tasksPanel(); return; }
  const [cat, all0] = await Promise.all([J("/api/catalog"), J("/api/profiles")]); const names0 = all0.filter(p => p.kind === "profile");
  let names = names0, cur = q.profile || mem.get("md_profile") || (names[0] && names[0].name), P = null, over = {}, opts = {}, orig = {};
  view.innerHTML = modelTabs("profiles") + `<div class="split" style="grid-template-columns:minmax(240px,300px) 1fr"><div class="panel"><div class="row" style="margin-bottom:10px"><h3 style="margin:0" class="grow">Профили</h3><button id="newP" class="primary" title="Новый профиль — копия выбранного">+</button></div><div class="plist" id="plist"></div>
      <div id="importYaml" style="margin-top:12px"></div></div><div id="editor"></div></div>`;
  const listP = () => { $("#plist").innerHTML = names.map(p => `<div class="pitem ${p.name === cur ? "on" : ""}" data-n="${esc(p.name)}"><span class="led ${p.ready ? "ok" : "alert"}"></span><div class="grow"><b>${esc(p.name)}</b><div class="mut small">${p.kind === "solver" ? "решатель" : esc(p.describe.pose || "")}</div></div></div>`).join("") || `<div class="mut">Профилей нет</div>`;
    $$("#plist .pitem").forEach(e => e.onclick = () => { cur = e.dataset.n; mem.set("md_profile", cur); listP(); load(); }); };
  const reloadNames = async () => { names = (await J("/api/profiles")).filter(p => p.kind === "profile"); listP(); };
  const setCfg = (k, v) => { if (v === orig[k] && !(k in P.config)) delete over[k]; else over[k] = v; };
  const load = async () => {
    if (!cur) { $("#editor").innerHTML = `<div class="panel mut">Создайте профиль кнопкой «+».</div>`; return; }
    try { P = await J("/api/profile/" + encodeURIComponent(cur)); } catch (e) { $("#editor").innerHTML = `<div class="panel err">${esc(e.message)}</div>`; return; }
    over = { ...P.config }; opts = { ...P.options }; orig = P.effective; const E = P.effective, o = opts, dev = (cat.runtimes[E["pose.runtime"]] || []);
    const imgs = [...new Set([...cat.imgsz, E["pose.imgsz"]])].sort((a, b) => a - b), fz = (o.fusion && o.fusion.weights) || {};
    $("#editor").innerHTML = `<div class="panel"><div class="row" style="margin-bottom:12px"><h2 style="margin:0">${esc(P.name)}</h2>${P.issues.some(i => i.level === "error") ? '<span class="chip bad">не запустится на этом компьютере</span>' : '<span class="chip ok">готов к запуску</span>'}<span class="mut small">отпечаток ${esc(P.fingerprint)}</span></div>
      ${field("Описание", `<input type="text" id="pdesc" value="${esc(P.description)}">`)}
      <div class="sect"><h3 style="margin-top:14px">Поза и трекинг</h3><div class="cols">
        ${field("Модель позы", `<select id="f_pose">${opt(cat.pose, E["pose.weights"], c => c.label + (c.ready ? "" : " (нет весов)"))}${cat.pose.some(c => c.id === E["pose.weights"]) ? "" : `<option selected>${esc(E["pose.weights"])}</option>`}</select>`)}
        ${field("Рантайм", `<select id="f_rt">${opt(Object.keys(cat.runtimes), E["pose.runtime"])}</select>`)}
        ${field("Устройство", `<select id="f_dev">${opt(dev, E["pose.device"])}</select>`)}
        ${field("Размер входа, px (длинная сторона)", `<select id="f_img">${opt(imgs, E["pose.imgsz"])}</select>`)}
        ${field("Уточнение точек", `<select id="f_ref"><option value="">выкл</option>${opt(cat.refine, E["pose.refine.enabled"] ? E["pose.refine.method"] : "")}</select>`)}
        ${field("Трекер", `<select id="f_trk">${opt(cat.trackers, E["tracking.tracker"])}</select>`)}
        ${field("Кадров обработки в секунду", `<input type="number" id="f_fps" min="2" max="30" step="1" value="${E["video.process_fps"]}">`)}</div></div>
      <div class="sect"><h3>Классификатор цикла (обязателен) и признаки</h3><div class="cols">
        ${field("Пакет классификатора", `<select id="f_cb"><option value="">— не выбран —</option>${opt(cat.cycle, o.cycle_bundle, c => c.label)}${o.cycle_bundle && !cat.cycle.some(c => c.id === o.cycle_bundle) ? `<option selected>${esc(o.cycle_bundle)}</option>` : ""}</select>`)}
        ${field("Фото-модель", `<select id="f_pb"><option value="">— нет —</option>${opt(cat.photo, o.photo_bundle)}</select>`)}
        ${field("VLM (Ollama)", `<select id="f_vlm"><option value="">— нет —</option>${opt(cat.vlm, o.vlm_model)}${o.vlm_model && !cat.vlm.some(c => c.id === o.vlm_model) ? `<option selected>${esc(o.vlm_model)}</option>` : ""}</select>`)}
        ${field("Режим VLM", `<select id="f_vm">${opt(cat.vlm_modes, o.vlm_mode)}</select>`)}</div>
        <div class="fld" style="margin-top:10px"><span>Детекторы предмета на кропах кисть–рот</span><div class="row" id="f_obj">${cat.detectors.map(d => `<label class="row small" style="gap:6px"><input type="checkbox" value="${esc(d.id)}" ${(o.objects || []).includes(d.id) ? "checked" : ""}>${esc(d.label)}${d.ready ? "" : " (нет весов)"}</label>`).join("") || '<span class="mut small">детекторов в реестре нет</span>'}</div></div>
        <div class="fld" style="margin-top:10px"><span>Слияние сигналов с оценкой классификатора (вес 0 — выключено; вектор признаков классификатора не меняется)</span><div class="row">
          ${[["vlm", "VLM"], ["object", "предмет"], ["photo", "фото-модель"]].map(([k, l]) => `<label class="row small" style="gap:6px">${l}<input type="number" step="0.1" min="0" max="5" id="fz_${k}" value="${fz[k] ?? 0}" style="width:80px"></label>`).join("")}
          <label class="row small" style="gap:6px">макс. поправка<input type="number" step="0.5" min="0.5" max="10" id="fz_clip" value="${(o.fusion && o.fusion.clip_logit) ?? 2}" style="width:80px"></label></div></div></div>
      <div class="sect"><h3>Пороги</h3><div class="cols">
        ${field("Порог оценки цикла (events.cycle_th)", `<input type="number" step="0.05" min="0" max="1" id="f_cth" value="${E["events.cycle_th"]}">`)}
        ${field("Порог события (confidence) — фиксируется до скрытого набора", `<input type="number" step="0.01" min="0" max="1" id="f_th" value="${E["events.confidence_threshold"]}">`)}
        ${field("th_in — вход ко рту", `<input type="number" step="0.05" id="f_in" value="${E["cycles.th_in"]}">`)}${field("th_out — выход", `<input type="number" step="0.05" id="f_out" value="${E["cycles.th_out"]}">`)}</div></div>
      <details class="sect"><summary class="mut" style="cursor:pointer">Все параметры конфигурации (${Object.keys(E).length})</summary><input type="text" id="pfilter" placeholder="Фильтр по имени, например cycles. или evidence." style="width:100%;margin:10px 0">
        <div style="max-height:340px;overflow:auto"><table id="ptable"></table></div></details>
      <div class="sect" id="issues">${P.issues.map(issueHtml).join("") || '<div class="chip ok">Проверка пройдена: профиль соберёт рабочий пайплайн на этом компьютере</div>'}</div>
      <div class="row"><button class="primary" id="save">Сохранить</button><button id="clone">Копия…</button><button id="del" class="bad">Удалить</button><a class="btn" href="/export/profile/${encodeURIComponent(P.name)}.yaml">Экспорт YAML</a><a class="btn" href="#/transfer?profile=${encodeURIComponent(P.name)}">Упаковать для передачи →</a><span class="grow"></span><span id="smsg" class="small"></span></div></div>`;
    const rows = () => { const f = ($("#pfilter").value || "").toLowerCase(); $("#ptable").innerHTML = `<tr><th>Параметр</th><th>Значение</th><th>По умолчанию</th></tr>` + Object.keys(E).filter(k => k.toLowerCase().includes(f)).slice(0, 400).map(k => {
      const v = k in over ? over[k] : E[k]; const ch = k in over && over[k] !== P.default[k];
      return `<tr><td class="small">${esc(k)}</td><td><input type="${typeof E[k] === "boolean" ? "checkbox" : typeof E[k] === "number" ? "number" : "text"}" data-k="${esc(k)}" ${typeof E[k] === "boolean" ? (v ? "checked" : "") : `value="${esc(v ?? "")}"`} ${typeof E[k] === "number" ? 'step="any"' : ""} style="${ch ? "border-color:var(--amber)" : ""};width:100%"></td><td class="mut small">${esc(P.default[k] ?? "")}</td></tr>`; }).join("");
      $$("#ptable input").forEach(i => i.onchange = () => { const k = i.dataset.k, t = typeof E[k]; setCfg(k, t === "boolean" ? i.checked : t === "number" ? Number(i.value) : i.value); }); };
    rows(); $("#pfilter").oninput = rows;
    const num = (id, key) => { $(id).onchange = () => setCfg(key, Number($(id).value)); };
    $("#f_pose").onchange = () => { const v = $("#f_pose").value; setCfg("pose.weights", v); setCfg("pose.backend", v.startsWith("rtmlib") ? "rtmlib" : "ultralytics"); };
    $("#f_rt").onchange = () => { const r = $("#f_rt").value; $("#f_dev").innerHTML = opt(cat.runtimes[r], cat.runtimes[r][0]); setCfg("pose.runtime", r); setCfg("pose.device", cat.runtimes[r][0]); };
    $("#f_dev").onchange = () => setCfg("pose.device", $("#f_dev").value); $("#f_img").onchange = () => setCfg("pose.imgsz", Number($("#f_img").value));
    $("#f_ref").onchange = () => { const v = $("#f_ref").value; setCfg("pose.refine.enabled", !!v); if (v) setCfg("pose.refine.method", v); }; $("#f_trk").onchange = () => setCfg("tracking.tracker", $("#f_trk").value);
    num("#f_fps", "video.process_fps"); num("#f_cth", "events.cycle_th"); num("#f_th", "events.confidence_threshold"); num("#f_in", "cycles.th_in"); num("#f_out", "cycles.th_out");
    const collect = () => { opts.cycle_bundle = $("#f_cb").value || null; opts.photo_bundle = $("#f_pb").value || null; opts.vlm_model = $("#f_vlm").value || null; opts.vlm_mode = opts.vlm_model ? $("#f_vm").value : "off";
      opts.objects = $$("#f_obj input:checked").map(i => i.value); const w = {}; [["vlm"], ["object"], ["photo"]].forEach(([k]) => { const v = Number($("#fz_" + k).value); if (v > 0) w[k] = v; });
      opts.fusion = Object.keys(w).length ? { weights: w, clip_logit: Number($("#fz_clip").value) || 2 } : null; };
    $("#save").onclick = async () => { collect(); try { P = await POST("/api/profile/" + encodeURIComponent(cur), { description: $("#pdesc").value, base: P.base, config: over, options: opts }); $("#smsg").innerHTML = '<span class="ok">Сохранено</span>'; toast("Профиль сохранён"); await reloadNames(); load(); } catch (e) { $("#smsg").innerHTML = `<span class="err">${esc(e.message)}</span>`; } };
    $("#clone").onclick = async () => { const n = prompt("Имя копии (буквы, цифры, - _ .)", P.name + "_copy"); if (!n) return; try { await POST(`/api/profile/${encodeURIComponent(cur)}/clone`, { to: n }); cur = n; mem.set("md_profile", cur); await reloadNames(); load(); } catch (e) { toast(e.message); } };
    $("#del").onclick = async () => { if (!confirm(`Удалить профиль «${cur}»?`)) return; await POST(`/api/profile/${encodeURIComponent(cur)}/delete`); cur = null; await reloadNames(); cur = names[0] && names[0].name; listP(); load(); };
  };
  $("#newP").onclick = async () => { if (!cur) return toast("Нет профиля-образца"); const n = prompt("Имя нового профиля (копия выбранного)", cur + "_new"); if (!n) return; try { await POST(`/api/profile/${encodeURIComponent(cur)}/clone`, { to: n }); cur = n; await reloadNames(); load(); } catch (e) { toast(e.message); } };
  dropzone($("#importYaml"), "Импорт профиля: перетащите YAML", ".yaml,.yml", async ([f]) => { try { const p = await rawPost("/api/config/import", f, { name: f.name.replace(/\.ya?ml$/i, "") }); toast("Профиль импортирован: " + p.name); cur = p.name; await reloadNames(); load(); } catch (e) { toast(e.message); } });
  listP(); await load();
}

async function registryPanel(cat) {
  const box = $("#registry"); if (!box) return;
  const draw = async () => {
    const R = await J("/api/registry");
    box.innerHTML = `<h3>Реестр моделей и пакеты</h3><div class="cols" style="grid-template-columns:2fr 1fr;align-items:start"><div style="max-height:320px;overflow:auto"><table><tr><th>Модель</th><th>Тип</th><th>Доверие</th><th>Веса</th><th>МБ</th><th></th></tr>
      ${R.models.map(m => `<tr><td><b>${esc(m.id)}</b><div class="mut small">${esc(m.license)} ${esc(m.note)}</div></td><td>${esc(m.kind)}</td><td><span class="chip ${m.trust === "official" ? "teal" : "warn"}">${esc(m.trust)}</span></td><td>${m.present ? '<span class="chip ok">есть</span>' : '<span class="chip bad">нет</span>'}</td><td>${m.size_mb || ""}</td>
      <td>${R.user.includes(m.id) ? `<button class="ghost" data-rm="${esc(m.id)}">убрать</button>` : ""}</td></tr>`).join("")}</table></div>
      <div><h3>Добавить свою модель</h3>${field("id (буквы, цифры, - _ .)", '<input type="text" id="am_id" value="my-detector">')}${field("Тип", '<select id="am_kind"><option>detector</option><option>pose</option><option>photo</option><option>vlm</option><option>tube</option></select>')}${field("Лицензия", '<input type="text" id="am_lic" value="?">')}
        <div id="am_dz" style="margin:8px 0"></div>${field("…или репозиторий Hugging Face", '<input type="text" id="am_hf" placeholder="user/name">')}
        <label class="row small" style="gap:6px;margin:8px 0"><input type="checkbox" id="am_unk"> разрешить «unknown» после проверки безопасности (осознанно)</label><button class="primary" id="am_go">Подключить</button><div id="am_msg" class="small" style="margin-top:6px"></div></div></div>
      <h3 style="margin-top:14px">Пакеты классификаторов и фото-моделей</h3><table><tr><th>Тип</th><th>Имя</th><th>Признаков</th><th>Создан</th><th>Заметка</th></tr>${R.bundles.map(b => `<tr><td>${esc(b.kind)}</td><td><b>${esc(b.name)}</b></td><td>${b.features ?? "—"}</td><td class="mut small">${esc(b.created)}</td><td class="mut small">${esc(b.note)}</td></tr>`).join("") || '<tr><td colspan="5" class="mut">Пакетов нет: обучите классификатор задачей ниже или установите решатель на экране «Передача».</td></tr>'}</table>`;
    let file = null; dropzone($("#am_dz"), "Файл весов: .pt .pth .onnx .safetensors", ".pt,.pth,.onnx,.safetensors,.bin", ([f]) => { file = f; $("#am_dz").firstChild.textContent = f.name; });
    $$("[data-rm]").forEach(b => b.onclick = async () => { await POST("/api/model/remove", { id: b.dataset.rm }); draw(); });
    $("#am_go").onclick = async () => { const p = { id: $("#am_id").value, kind: $("#am_kind").value, license: $("#am_lic").value, name: file ? file.name : "", allow_unknown: $("#am_unk").checked ? "1" : "0" }; try {
        const r = file ? await rawPost("/api/model/add", file, p) : await rawPost("/api/model/add", new Blob([]), { ...p, hf: $("#am_hf").value });
        $("#am_msg").innerHTML = `<span class="ok">Подключено${r.file ? ": " + esc(r.file) : ""}</span> ${r.scan ? "· проверка: " + esc(r.scan.verdict) : esc(r.note || "")}`; draw(); } catch (e) { $("#am_msg").innerHTML = `<span class="err">${esc(e.message)}</span>`; } };
  };
  draw();
}

async function tasksPanel() {
  const box = $("#tasks"); if (!box) return; const cat = await J("/api/tasks/catalog"); let tid = null;
  box.innerHTML = `<h3>Доработка модели: запуск задач</h3><p class="mut small">Белый список команд проекта: обучение классификатора, признаки цикла, честная оценка, лестница позы. Результат — в журнале и в «Моделях».</p>
    <div class="cols" style="grid-template-columns:minmax(260px,380px) 1fr;align-items:start"><div><select id="tk" style="width:100%">${cat.map(t => `<option value="${t.id}">${esc(t.title)}</option>`).join("")}</select><p class="mut small" id="tdesc"></p><div id="tform"></div><button class="primary" id="trun" style="margin-top:8px">Запустить</button></div>
    <div><div class="row"><b id="tname" class="grow">Журнал задачи</b><button id="tstop" class="ghost hidden">Остановить</button></div><pre class="log" id="tlog">Выберите или запустите задачу.</pre><div id="tlist" class="small"></div></div></div>`;
  const form = () => { const t = cat.find(x => x.id === $("#tk").value); $("#tdesc").textContent = t.description; $("#tform").innerHTML = t.fields.map(f => f.kind === "bool" ? `<label class="row small" style="gap:8px;margin:6px 0"><input type="checkbox" data-f="${f.key}" ${f.default ? "checked" : ""}>${esc(f.label)}</label>`
      : f.kind === "choice" ? field(esc(f.label), `<select data-f="${f.key}">${opt(f.choices, f.default)}</select>`) : f.kind === "lines" ? field(esc(f.label), `<textarea data-f="${f.key}" rows="3">${esc(f.default)}</textarea>`)
      : field(esc(f.label), `<input type="${f.kind === "int" || f.kind === "float" ? "number" : "text"}" step="any" data-f="${f.key}" value="${esc(f.default)}">`)).join(""); };
  $("#tk").onchange = form; form();
  const poll = async () => { if (!tid) return; try { const r = await J(`/api/task/${tid}`); $("#tname").textContent = `${r.title} · ${r.state} · ${r.seconds} с`; $("#tlog").textContent = r.log || "(пока пусто)"; $("#tlog").scrollTop = 1e9; $("#tstop").classList.toggle("hidden", r.state !== "идёт"); } catch { } };
  const list = async () => { const l = await J("/api/tasks"); $("#tlist").innerHTML = l.slice(0, 8).map(t => `<div class="camrow" data-t="${t.id}"><span class="chip ${t.state === "готово" ? "ok" : t.state === "идёт" ? "teal" : "bad"}">${esc(t.state)}</span><div class="grow">${esc(t.title)}</div><span class="mut">${Math.round(t.seconds)} с</span></div>`).join("");
    $$("#tlist .camrow").forEach(r => r.onclick = () => { tid = r.dataset.t; poll(); }); };
  $("#trun").onclick = async () => { const values = {}; $$("#tform [data-f]").forEach(e => { values[e.dataset.f] = e.type === "checkbox" ? e.checked : e.type === "number" ? (e.value === "" ? "" : Number(e.value)) : e.value; });
    try { tid = (await POST("/api/task/start", { task: $("#tk").value, values })).id; toast("Задача запущена"); poll(); list(); } catch (e) { toast(e.message); } };
  $("#tstop").onclick = async () => { await POST(`/api/task/${tid}/stop`); poll(); };
  every(2000, () => { poll(); list(); }); list();
}

/* ================================================================= ПЕРЕДАЧА */
async function vTransfer(q) {
  const [allp, cat, R] = await Promise.all([J("/api/profiles"), J("/api/catalog"), J("/api/registry")]); const profiles = allp.filter(p => p.kind === "profile");
  view.innerHTML = `<div class="cols" style="grid-template-columns:repeat(auto-fit,minmax(380px,1fr));align-items:start">
    <div class="panel"><h2>Упаковать решатель</h2><p class="mut small">Профиль целиком одним архивом: классификатор, метаданные архитектуры, по желанию веса и разметка. На другом компьютере — «Установить».</p>
      ${field("Профиль", `<select id="pk_prof">${opt(profiles.map(p => p.name), q.profile || mem.get("md_profile"))}</select>`)}${field("Классификатор (если нужно заменить)", `<select id="pk_cls"><option value="">как в профиле</option>${opt(cat.cycle, "")}</select>`)}${field("Имя решателя", `<input type="text" id="pk_name">`)}
      <label class="row small" style="gap:8px;margin:8px 0"><input type="checkbox" id="pk_w"> включить веса моделей (если лежат на диске)</label><label class="row small" style="gap:8px;margin-bottom:10px"><input type="checkbox" id="pk_d"> включить набор разметки (events_gt, cycle_labels)</label>
      <button class="primary" id="pk_go">Упаковать</button><div id="pk_out" style="margin-top:12px"></div></div>
    <div class="panel"><h2>Установить решатель</h2><div id="in_dz"></div><div id="in_out" style="margin-top:12px"></div>
      <h3 style="margin-top:16px">Установленные решатели</h3><table><tr><th>Имя</th><th>Режим</th><th>F1</th><th>Создан</th></tr>${R.solvers.map(s => `<tr><td><b>${esc(s.name)}</b></td><td class="small">${esc(s.mode)}</td><td>${s.f1 != null ? s.f1.toFixed(2) : "—"}</td><td class="mut small">${esc(s.created || "")}</td></tr>`).join("") || '<tr><td colspan="4" class="mut">нет</td></tr>'}</table></div></div>
    <div class="cols" style="grid-template-columns:repeat(auto-fit,minmax(380px,1fr));align-items:start;margin-top:16px"><div class="panel"><h2>Конфигурации профилей</h2><p class="mut small">Один профиль — один YAML-файл: можно хранить в репозитории и переносить.</p>
      <table>${profiles.map(p => `<tr><td><b>${esc(p.name)}</b><div class="mut small">${esc(p.describe.pose || "")} ${esc(p.describe.cycle_model || "")}</div></td><td style="text-align:right"><a class="btn" href="/export/profile/${encodeURIComponent(p.name)}.yaml">YAML</a></td></tr>`).join("")}</table><div id="cf_dz" style="margin-top:12px"></div></div>
    <div class="panel"><h2>Пакеты моделей</h2><p class="mut small">Классификаторы и фото-модели (только безопасные форматы, sha256 для каждого файла).</p>
      ${R.bundles.map(b => `<label class="row" style="gap:8px;margin:4px 0"><input type="checkbox" data-b="${esc(b.kind)}/${esc(b.name)}"><b>${esc(b.name)}</b><span class="chip">${esc(b.kind)}</span><span class="mut small">${b.features ?? ""} признаков</span></label>`).join("") || '<div class="mut">Пакетов нет</div>'}
      <button class="primary" id="bd_go" style="margin-top:10px">Собрать zip</button><span id="bd_out" style="margin-left:10px"></span><div id="bd_dz" style="margin-top:12px"></div>
      <label class="row small" style="gap:6px;margin-top:8px"><input type="checkbox" id="bd_ow"> перезаписать существующие при импорте</label></div></div>`;
  const nm = () => { $("#pk_name").value = $("#pk_prof").value; }; $("#pk_prof").onchange = nm; nm();
  const card = r => `<div class="card"><div class="row"><b>${esc(r.id)}</b><span class="chip ${r.ok === false ? "bad" : "ok"}">${r.ok === false ? "нарушена целостность" : "целостность проверена"}</span><span class="chip teal">${esc(r.mode)}</span></div>
    <div class="row small" style="margin:8px 0">${(r.stages || []).map(s => `<span class="chip">${esc(s)}</span>`).join("")}</div>
    <div class="row small">${Object.entries(r.streaming || {}).map(([k, v]) => `<span class="chip ${v ? "ok" : "bad"}">${k}: ${v ? "да" : "нет"}</span>`).join("")}${r.resources ? `<span class="chip">память: пик ${r.resources.peak_gb} ГБ · последовательно ≈ ${r.resources.sequential_gb} ГБ</span>` : ""}${r.evaluation ? `<span class="chip">F1 ${r.evaluation.f1} · порог ${r.evaluation.threshold}</span>` : ""}</div>
    ${(r.problems || []).map(p => `<div class="err small">${esc(p)}</div>`).join("")}${(r.notes || []).map(n => `<div class="mut small">• ${esc(n)}</div>`).join("")}</div>`;
  $("#pk_go").onclick = async () => { $("#pk_out").textContent = "Упаковываю…"; try { const r = await POST("/api/solver/pack", { profile: $("#pk_prof").value, name: $("#pk_name").value, classifier: $("#pk_cls").value || null, with_weights: $("#pk_w").checked, with_dataset: $("#pk_d").checked });
      const m = r.meta, a = m.architecture; $("#pk_out").innerHTML = card({ id: m.id, mode: a.mode, stages: a.stages.map(s => s.id), streaming: a.streaming, resources: a.resources_estimate, evaluation: (m.evaluation || {}).metrics, notes: m.notes }) +
        `<div class="row" style="margin-top:10px"><a class="btn primary" href="/export/${encodeURIComponent(r.file)}" style="background:var(--teal);color:#06231e">Скачать ${esc(r.file)} (${r.size_mb} МБ)</a></div>`; } catch (e) { $("#pk_out").innerHTML = `<span class="err">${esc(e.message)}</span>`; } };
  dropzone($("#in_dz"), "Перетащите архив .sdsolver.zip", ".zip", async ([f]) => { $("#in_out").textContent = "Проверяю…";
    try { const r = await rawPost("/api/solver/inspect", f); $("#in_out").innerHTML = card(r) + `<div class="row" style="margin-top:10px"><label class="row small" style="gap:6px"><input type="checkbox" id="in_ow"> заменить, если уже установлен</label><button class="primary" id="in_go" ${r.ok ? "" : "disabled"}>Установить</button></div><div id="in_res"></div>`;
      $("#in_go").onclick = async () => { try { const x = await rawPost("/api/solver/install", f, { overwrite: $("#in_ow").checked ? "1" : "0" }); $("#in_res").innerHTML = `<p class="ok">Установлен: ${esc(x.name)} (${esc(x.installed.join(", "))})</p>` + (x.issues || []).map(issueHtml).join("") + `<a class="btn" href="#/analysis">Запустить на видео →</a>`; } catch (e) { $("#in_res").innerHTML = `<span class="err">${esc(e.message)}</span>`; } };
    } catch (e) { $("#in_out").innerHTML = `<span class="err">Это не решатель: ${esc(e.message)}</span>`; } });
  dropzone($("#cf_dz"), "Импорт профиля: перетащите YAML", ".yaml,.yml", async ([f]) => { try { const p = await rawPost("/api/config/import", f, { name: f.name.replace(/\.ya?ml$/i, "") }); toast("Импортирован профиль " + p.name); } catch (e) { toast(e.message); } });
  $("#bd_go").onclick = async () => { const items = $$("[data-b]:checked").map(i => i.dataset.b); if (!items.length) return toast("Отметьте пакеты"); try { const r = await POST("/api/bundles/export", { items }); $("#bd_out").innerHTML = `<a href="/export/${encodeURIComponent(r.file)}">Скачать ${esc(r.file)}</a> (${r.files} файлов)`; } catch (e) { toast(e.message); } };
  dropzone($("#bd_dz"), "Импорт пакетов: перетащите zip", ".zip", async ([f]) => { try { const r = await rawPost("/api/bundles/import", f, { overwrite: $("#bd_ow").checked ? "1" : "0" }); toast("Импортировано: " + r.imported.join(", ")); } catch (e) { toast(e.message); } });
}

/* ================================================================= ИСТОЧНИКИ */
async function vSources() {
  const [folders, health] = await Promise.all([J("/api/folders"), J("/api/health")]);
  view.innerHTML = `<div class="cols" style="grid-template-columns:repeat(auto-fit,minmax(420px,1fr));align-items:start"><div class="panel"><h2>Источники видео</h2>
      <table><tr><th>Папка</th><th>Тип</th><th>Видео</th><th></th></tr>${folders.map(f => `<tr><td><b>${esc(f.name)}</b></td><td>${f.kind === "camera" ? '<span class="chip teal">камера</span>' : '<span class="chip">файлы</span>'}</td><td>${f.n}</td><td>${f.removable ? `<button class="ghost" data-rm="${esc(f.root)}">убрать</button>` : ""}</td></tr>`).join("") || '<tr><td colspan="4" class="mut">Источников нет</td></tr>'}</table>
      <h3 style="margin-top:16px">Подключить папку с видео</h3><div class="row" style="flex-wrap:nowrap"><input type="text" id="rp" placeholder="C:\\Видео\\улица или /data/video" style="flex:1"><button class="primary" id="rp_go">Подключить</button></div><p class="mut small">Видео ищутся рекурсивно, формат любой (mp4, avi, mkv, mov, wmv, webm, ts…). Папка запоминается.</p>
      <h3>Загрузить видео</h3><div id="up_dz"></div><div id="up_out" class="small" style="margin-top:8px"></div></div>
    <div class="panel"><h2>Камеры на карте</h2><p class="mut small">Камера — папка с видео-фрагментами в каталоге потоков: <code>${esc(health.roots.streams)}</code></p>
      <pre class="log">gatchina-03-20261009T143000/   part1.mp4  part2.mp4\npavlovsk-01-20261009T100000/   a.mp4\ntsarskoe-selo-12-1430/         b.mp4</pre><p class="mut small">Имя папки: <b>район-индекс-время начала</b>. Положение и название улицы — файл <code>cameras.yaml</code> рядом с папками:</p>
      <pre class="log">gatchina-03: {lat: 59.5762, lon: 30.1291, street: "ул. Соборная", title: "Перекрёсток у вокзала"}</pre><p class="mut small">Без файла маячки раскладываются условно, на карте это помечено «положение условное».</p>
      <div class="row"><button id="dm">Создать демо-тревоги</button><button id="dm0" class="ghost">Удалить демо-тревоги</button></div><p class="mut small">Демо — синтетика для проверки интерфейса, не данные распознавания.</p>
      <h3>Каталоги</h3><table>${Object.entries(health.roots).map(([k, v]) => `<tr><td class="small">${esc(k)}</td><td class="mut small">${esc(v)}</td></tr>`).join("")}</table></div></div>`;
  $$("[data-rm]").forEach(b => b.onclick = async () => { await POST("/api/root/remove", { key: b.dataset.rm }); route(); });
  $("#rp_go").onclick = async () => { try { const r = await POST("/api/root", { path: $("#rp").value }); toast(`Папка подключена: видео ${r.n}`); route(); } catch (e) { toast(e.message); } };
  dropzone($("#up_dz"), "Перетащите видео или нажмите, чтобы выбрать (можно несколько файлов)", "video/*,.mkv,.avi,.wmv,.ts,.mov,.m4v,.webm", fs => uploadMany(fs, $("#up_out")), true);
  $("#dm").onclick = async () => { toast(`Создано: ${(await POST("/api/demo")).n}`); route(); }; $("#dm0").onclick = async () => { toast(`Удалено: ${(await POST("/api/demo", { clear: true })).n}`); route(); };
}
