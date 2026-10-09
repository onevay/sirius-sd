"use strict";
/* Модели → «Классификатор» (порог цикла по OOF, подгонка, диагностика, обучение) и «Проверка на данных» (оценка профиля на папках, кривая порога, ошибки) */
/* опрос фоновой операции; если пользователь ушёл с экрана — прекращаем (null), чтобы не писать в уже убранные элементы */
const pollOp = async (id, onTick) => { const seq = window.__routeSeq; for (;;) { const r = await J("/api/op/" + id); if (seq !== window.__routeSeq) return null; onTick && onTick(r); if (r.state !== "running") return r; await new Promise(res => setTimeout(res, 600)); } };
const opStatus = (el, r) => { el.innerHTML = r.state === "running" ? `<div class="prog"><i style="width:${Math.round(r.progress * 100)}%"></i></div><div class="mut small" style="margin-top:4px">${esc(r.title)} · ${esc(r.msg)} · ${r.seconds} с</div>` : r.state === "error" ? `<span class="err">${esc(r.error)}</span>` : ""; };
const fix = (v, d = 2) => v == null ? "—" : Number(v).toFixed(d);

/* ================================================================= КЛАССИФИКАТОР */
async function vClassifier(q) {
  const [R, profs, T] = await Promise.all([J("/api/registry"), J("/api/profiles"), J("/api/tune/table")]);
  const bundles = R.bundles.filter(b => b.kind === "cycle"), profiles = profs.filter(p => p.kind === "profile");
  let cur = q.bundle || mem.get("tn_bundle") || (bundles[0] && bundles[0].name), A = null, cand = null, spec = null;
  view.innerHTML = modelTabs("classifier") + `<div class="panel"><h2>Порог цикла и качество пакета</h2><p class="lead">Классификатор оценивает каждый цикл жеста («подъём руки ко рту»). Порог цикла решает, какие циклы считать затяжками. Кривые построены по OOF-оценкам: каждую оценку выдала модель, не видевшая этот клип.</p>
      <div class="row" style="margin-bottom:12px"><select id="bsel" style="min-width:240px">${bundles.map(b => `<option ${b.name === cur ? "selected" : ""}>${esc(b.name)}</option>`).join("") || "<option value=''>пакетов нет</option>"}</select>
        <select id="mem" class="hidden"></select><span class="mut small">профиль для сравнения и применения:</span><select id="bprof">${profiles.map(p => `<option>${esc(p.name)}</option>`).join("")}</select></div><div id="ba"></div></div>
    <div class="panel" style="margin-top:16px"><h2>Подгонка классификатора</h2><div id="fit"></div></div>`;
  const drawA = async () => {
    const box = $("#ba"); if (!cur) { box.innerHTML = `<div class="empty">Нет пакетов классификатора. Обучите первый в блоке ниже или установите решатель на экране «Передача».</div>`; return; }
    box.innerHTML = `<div class="mut">Считаю…</div>`; mem.set("tn_bundle", cur);
    try { A = await J(`/api/tune/bundle/${encodeURIComponent(cur)}?member=${encodeURIComponent($("#mem").value || "oof_ensemble")}`); } catch (e) { box.innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
    const i = A.info, pc = profiles.length ? await J("/api/profile/" + encodeURIComponent($("#bprof").value)) : null, inProf = pc ? pc.effective["events.cycle_th"] : null;
    cand = cand ?? (A.best ? A.best.suggested : null);
    const chips = `<div class="row" style="margin-bottom:10px"><span class="chip teal">${i.n_features} признаков</span>${Object.entries(i.groups || {}).map(([k, v]) => `<span class="chip">${esc(k)}: ${v}</span>`).join("")}<span class="chip">члены: ${esc(Object.keys(i.members || {}).join(", ") || "—")}</span>${i.auc_oof != null ? `<span class="chip ok">AUC OOF ${fix(i.auc_oof)}</span>` : ""}<span class="chip">циклов ${i.n ?? "—"}, затяжек ${i.n_pos ?? "—"}</span></div>`;
    if (!A.thresholds.length) { box.innerHTML = chips + `<div class="notice">${IC.warn}<div>${esc(A.note || "")}</div></div>`; return; }
    box.innerHTML = chips + `<div class="cols" style="grid-template-columns:2fr 1fr;align-items:start"><div><svg class="chart" id="thc"></svg>${legend([["var(--blue)", "точность (precision)"], ["var(--green)", "полнота (recall)"], ["var(--amber)", "F1 цикла"], ["var(--teal)", "рекомендуется"], ["var(--red)", "порог в профиле"]])}
      <div class="help">Клик по графику выбирает порог. Рекомендация — середина плато: порогов, у которых F1 не хуже лучшего на 0,02 (так цифра не держится на одном пике).</div></div>
      <div><h3>Выбранный порог</h3><div class="row"><input type="number" id="cth" step="0.05" min="0" max="1" value="${cand}" style="width:90px"><span class="mut small">в профиле «${esc($("#bprof").value)}»: <b>${inProf ?? "—"}</b></span></div><div id="crow" style="margin:10px 0"></div>
        <button class="primary" id="apply">Записать порог цикла в профиль</button><div id="amsg" class="small" style="margin-top:6px"></div></div></div>
      <div class="cols" style="margin-top:14px;grid-template-columns:1fr 1fr;align-items:start"><div><h3>Распределение оценок</h3><svg class="chart" id="hist" style="height:170px"></svg>${legend([["var(--blue)", "не затяжки"], ["var(--amber)", "затяжки"]])}</div>
      <div><h3>Члены ансамбля (AUC)</h3><table>${A.members.map(m => `<tr><td>${esc(m.name)}</td><td><div class="bar"><i style="width:${Math.round(m.auc * 100)}%"></i></div></td><td>${fix(m.auc)}</td></tr>`).join("")}</table></div></div>`;
    const t = A.thresholds, near = th => t.reduce((a, b) => Math.abs(b.threshold - th) < Math.abs(a.threshold - th) ? b : a);
    const paint = () => { chartSvg($("#thc"), { xmin: .05, xmax: .95, xticks: [.1, .2, .3, .4, .5, .6, .7, .8, .9], xfmt: v => v.toFixed(1),
        series: [{ pts: t.map(r => [r.threshold, r.precision]), color: "var(--blue)", name: "precision" }, { pts: t.map(r => [r.threshold, r.recall]), color: "var(--green)", name: "recall" }, { pts: t.map(r => [r.threshold, r.f1]), color: "var(--amber)", width: 3.2, dots: 3, name: "F1" }],
        bands: A.best ? [{ x0: A.best.plateau[0], x1: A.best.plateau[1], color: "#74d3c3", tip: "плато" }] : [], vlines: [{ x: A.best.suggested, color: "var(--teal)", label: "рек." }, ...(inProf != null ? [{ x: inProf, color: "var(--red)", dash: "5 4", label: "профиль" }] : []), { x: cand, color: "var(--ink)", dash: "2 3" }],
        onClick: x => { cand = Math.round(Math.max(.05, Math.min(.95, x)) * 20) / 20; $("#cth").value = cand; paint(); row(); } });
      const h = A.hist, mx = Math.max(1, ...h.pos, ...h.neg), bars = []; h.neg.forEach((v, k) => bars.push({ x: (h.bins[k] + h.bins[k + 1]) / 2 - .015, y: v, n: 20, color: "var(--blue)", tip: `${h.bins[k].toFixed(1)}–${h.bins[k + 1].toFixed(1)}: не затяжек ${v}` }));
      h.pos.forEach((v, k) => bars.push({ x: (h.bins[k] + h.bins[k + 1]) / 2 + .015, y: v, n: 20, color: "var(--amber)", tip: `${h.bins[k].toFixed(1)}–${h.bins[k + 1].toFixed(1)}: затяжек ${v}` }));
      chartSvg($("#hist"), { h: 170, xmin: 0, xmax: 1, ymin: 0, ymax: mx, yticks: [0, mx / 2, mx], yfmt: v => Math.round(v), xticks: [0, .25, .5, .75, 1], xfmt: v => v.toFixed(2), bars }); };
    const row = () => { const r = near(+$("#cth").value); $("#crow").innerHTML = `<table><tr><td class="mut">precision</td><td><b>${fix(r.precision)}</b></td><td class="mut">recall</td><td><b>${fix(r.recall)}</b></td></tr><tr><td class="mut">F1</td><td><b>${fix(r.f1)}</b></td><td class="mut">TP/FP/FN</td><td>${r.tp}/${r.fp}/${r.fn}</td></tr></table>`; };
    paint(); row(); $("#cth").onchange = () => { cand = +$("#cth").value; paint(); row(); };
    $("#apply").onclick = async () => { const name = $("#bprof").value; if (!name) return toast("Нет профиля"); try { const p = await J("/api/profile/" + encodeURIComponent(name)); const cfg = { ...p.config, "events.cycle_th": +$("#cth").value };
        await POST("/api/profile/" + encodeURIComponent(name), { description: p.description, base: p.base, config: cfg, options: p.options }); $("#amsg").innerHTML = `<span class="ok">Порог цикла ${$("#cth").value} записан в «${esc(name)}»</span> · проверьте на вкладке «Проверка на данных»`; } catch (e) { $("#amsg").innerHTML = `<span class="err">${esc(e.message)}</span>`; } };
  };
  $("#bsel").onchange = () => { cur = $("#bsel").value; cand = null; drawA(); }; $("#bprof").onchange = () => { cand = cand; drawA(); };
  await drawA(); fitPanel(T, profiles, spec, cur, name => { cur = name; mem.set("tn_bundle", name); route(); });
}

function fitPanel(T, profiles, _s, curBundle, onTrained) {
  const box = $("#fit"); let spec = null;
  if (!T.ok) { box.innerHTML = `<div class="notice">${IC.warn}<div><b>Таблица циклов не собрана.</b> <span>${esc(T.error || "")}<br>Нужны полные запуски распознавания, метки жестов и (для признаков предмета/фото) задача «Признаки цикла» на вкладке «Модели и задачи».</span></div></div>`; return; }
  const thin = T.n_labeled < 60 || !T.n_pos || !T.n_neg;
  box.innerHTML = `<div class="kpis"><div class="kpi"><span class="mut small">Размеченных циклов</span><b>${T.n_labeled}</b></div><div class="kpi"><span class="mut small">Затяжек</span><b>${T.n_pos}</b></div><div class="kpi"><span class="mut small">Не затяжек</span><b>${T.n_neg}</b></div><div class="kpi"><span class="mut small">Видео с метками</span><b>${T.n_videos}</b></div></div>
    ${thin ? `<div class="notice" style="margin-bottom:12px">${IC.warn}<div>Мало размеченных циклов: оценка будет шумной. <span>Разметьте больше жестов в Streamlit-интерфейсе (страница «Анализ»).</span></div></div>` : ""}
    <div class="row" style="margin-bottom:12px"><div class="fld"><span>Набор признаков</span><select id="fset">${Object.keys(T.sets).map(s => `<option ${s === "obj_hold_zsd" ? "selected" : ""}>${esc(s)}</option>`).join("")}</select></div>
      <div class="fld"><span>Члены ансамбля</span><div class="row">${[["lr", "логрегрессия"], ["gb", "LightGBM"], ["nn", "малая сеть"]].map(([k, l]) => `<label class="row small" style="gap:5px"><input type="checkbox" data-k="${k}" checked>${l}</label>`).join("")}</div></div>
      <button id="mk" class="primary" style="align-self:flex-end">Создать спецификацию</button>
      ${T.saved.length ? `<div class="fld"><span>или сохранённая</span><select id="saved"><option value="">—</option>${T.saved.map(s => `<option>${esc(s)}</option>`).join("")}</select></div>` : ""}</div><div id="spec"></div><div id="fres"></div>`;
  const draw = () => {
    const el = $("#spec"); if (!spec) { el.innerHTML = ""; return; }
    const filled = Object.fromEntries(T.columns.map(c => [c.name, c.filled]));
    el.innerHTML = `<h3>Спецификация</h3><div class="cols" style="align-items:start">${Object.entries(spec.members).map(([n, m]) => `<div class="card" data-m="${esc(n)}"><div class="row" style="flex-wrap:nowrap"><b class="grow">${esc(n)}</b><select data-f="kind" style="max-width:62%">${Object.entries(T.kinds).map(([k, l]) => `<option value="${k}" ${k === m.kind ? "selected" : ""}>${esc(l.split(" (")[0])}</option>`).join("")}</select><button class="ghost" data-del="${esc(n)}" title="убрать член" style="padding:6px 10px">✕</button></div>
      <div class="row small" style="margin:8px 0">${Object.entries(T.params[m.kind] || {}).map(([k, d]) => `<label class="fld" style="width:96px"><span>${esc(k)}</span><input type="number" step="any" data-p="${esc(k)}" value="${(m.params || {})[k] ?? d}"></label>`).join("")}</div>
      <details><summary class="mut small" style="cursor:pointer">признаки (${m.features.length})</summary><div style="max-height:180px;overflow:auto;margin-top:6px">${[...new Set([...T.columns.map(c => c.name), ...m.features])].map(c => `<label class="row small" style="gap:6px"><input type="checkbox" data-c="${esc(c)}" ${m.features.includes(c) ? "checked" : ""}>${esc(c)} <span class="mut">${filled[c] != null ? Math.round(filled[c] * 100) + "%" : "нет в таблице"}</span></label>`).join("")}</div></details></div>`).join("")}</div>
      <div class="row" style="margin:12px 0"><label class="fld" style="width:90px"><span>Фолдов</span><input type="number" id="ns" min="2" max="10" value="${spec.cv.n_splits}"></label><label class="fld" style="width:90px"><span>Повторов</span><input type="number" id="rp" min="1" max="10" value="${spec.cv.repeats}"></label>
        <label class="fld" style="width:90px"><span>Seed</span><input type="number" id="sd" value="${spec.cv.seed}"></label><label class="row small" style="gap:6px"><input type="checkbox" id="bsc" ${spec.cv.by_scene ? "checked" : ""}>фолды по сценам</label><label class="row small" style="gap:6px"><input type="checkbox" id="cal" ${spec.calibrate ? "checked" : ""}>изотоническая калибровка</label>
        <label class="fld" style="min-width:180px"><span>Имя спецификации</span><input type="text" id="sname" value="${esc(spec.name || "")}"></label></div><div id="sprob"></div>
      <div class="row"><button id="vd">Проверить</button><button id="sv">Сохранить спецификацию</button><button id="dg" class="primary">Диагностика переобучения</button><span class="mut small">перестановок меток <input type="number" id="np" value="3" min="0" max="20" style="width:64px"></span><label class="row small" style="gap:5px"><input type="checkbox" id="imp" checked>важность признаков (долго)</label></div>
      <div class="row" style="margin-top:12px;padding-top:12px;border-top:1px solid var(--line)"><label class="fld"><span>Имя нового пакета</span><input type="text" id="bname" placeholder="cycle_new" value="${esc(spec.name ? "cycle_" + spec.name : "")}"></label>
        <label class="fld"><span>Подключить к профилю</span><select id="attach"><option value="">— не подключать —</option>${profiles.map(p => `<option>${esc(p.name)}</option>`).join("")}</select></label><button id="tr" class="primary" style="align-self:flex-end">Обучить и сохранить пакет</button></div>
      <div class="help" style="margin-top:6px">Пакет не содержит pickle: числа и текстовые деревья, переносится копированием. Подключение к профилю меняет его отпечаток (кэш оценок пересчитается).</div>`;
    $$("[data-m]", el).forEach(card => { const n = card.dataset.m, m = spec.members[n];
      $("[data-f=kind]", card).onchange = e => { m.kind = e.target.value; m.params = {}; draw(); };
      $$("[data-p]", card).forEach(i => i.onchange = () => { m.params = { ...(m.params || {}), [i.dataset.p]: Number(i.value) }; });
      $$("[data-c]", card).forEach(i => i.onchange = () => { m.features = i.checked ? [...new Set([...m.features, i.dataset.c])] : m.features.filter(x => x !== i.dataset.c); });
      $("[data-del]", card).onclick = () => { delete spec.members[n]; draw(); }; });
    const sync = () => { spec.cv = { n_splits: +$("#ns").value, repeats: +$("#rp").value, seed: +$("#sd").value, by_scene: $("#bsc").checked }; spec.calibrate = $("#cal").checked; spec.name = $("#sname").value.trim(); spec.features = [...new Set(Object.values(spec.members).flatMap(m => m.features))]; };
    const out = $("#fres");
    $("#vd").onclick = async () => { sync(); const r = await POST("/api/tune/spec/validate", { spec }); $("#sprob").innerHTML = r.problems.length ? r.problems.map(p => `<div class="err small">• ${esc(p)}</div>`).join("") : `<div class="ok small">Спецификация корректна</div>`; };
    $("#sv").onclick = async () => { sync(); try { const r = await POST("/api/tune/spec/save", { spec }); toast("Сохранено: " + r.file); } catch (e) { toast(e.message); } };
    $("#dg").onclick = async () => { sync(); const pr = await POST("/api/tune/spec/validate", { spec }); if (pr.problems.length) { $("#sprob").innerHTML = pr.problems.map(p => `<div class="err small">• ${esc(p)}</div>`).join(""); return; }
      $("#sprob").innerHTML = ""; const op = await POST("/api/tune/diagnose", { spec, n_perm: +$("#np").value, importance: $("#imp").checked }); const r = await pollOp(op.id, x => opStatus(out, x)); if (!r) return; if (r.state === "done") renderDiag(out, r.result); else opStatus(out, r); };
    $("#tr").onclick = async () => { sync(); const nm = $("#bname").value.trim(); if (!nm) return toast("Введите имя пакета"); try { const op = await POST("/api/tune/train", { spec, name: nm, attach: $("#attach").value || null }); const r = await pollOp(op.id, x => opStatus(out, x)); if (!r) return;
        if (r.state === "done") { out.innerHTML = `<div class="notice" style="background:var(--greenbg)"><div><b>Пакет ${esc(r.result.bundle)} обучен.</b> AUC OOF ${fix(r.result.auc)}, циклов ${r.result.n}, затяжек ${r.result.n_pos}${r.result.attached ? `. Подключён к профилю «${esc(r.result.attached)}».` : ""}</div></div>`; setTimeout(() => onTrained(nm), 1500); } else opStatus(out, r); } catch (e) { toast(e.message); } };
  };
  $("#mk").onclick = async () => { const kinds = $$("[data-k]:checked").map(i => i.dataset.k); spec = await POST("/api/tune/spec/default", { set: $("#fset").value, kinds }); spec.cv = spec.cv || T.cv; draw(); };
  const sv = $("#saved"); if (sv) sv.onchange = async () => { if (!sv.value) return; spec = await J("/api/tune/spec/" + encodeURIComponent(sv.value)); spec.cv = { ...T.cv, ...(spec.cv || {}) }; draw(); };
}
function renderDiag(el, r) {
  const mb = r.members || [], cols = mb.length ? Object.keys(mb[0]) : [];
  el.innerHTML = `<h3 style="margin-top:14px">Диагностика</h3>${(r.verdict || []).map(v => `<div class="notice" style="margin-bottom:6px;${/велик|утечк/.test(v) ? "" : "background:var(--card)"}">${IC.warn}<div>${esc(v)}</div></div>`).join("")}
    <table style="margin:10px 0"><tr>${cols.map(c => `<th>${esc(c)}</th>`).join("")}</tr>${mb.map(m => `<tr>${cols.map(c => `<td>${typeof m[c] === "number" ? fix(m[c], 3) : esc(m[c])}</td>`).join("")}</tr>`).join("")}</table>
    <div class="help">циклов ${r.n}, затяжек ${r.n_pos}, групп ${r.n_groups}. «Разрыв» = AUC на обучающих данных минус OOF: большой разрыв — запоминание. ${r.perm_mean != null ? `AUC при перемешанных метках: <b>${fix(r.perm_mean)}</b> (должен быть ≈ 0,5).` : ""}</div>
    <div class="cols" style="margin-top:12px;grid-template-columns:1fr 1fr;align-items:start"><div><h3>Кривая обучения</h3><svg class="chart" id="lc" style="height:190px"></svg></div><div><h3>Важность признаков</h3><div id="imp"></div></div></div>`;
  const cv = r.curve || []; if (cv.length) { const xs = cv.map(p => p["доля_видео"]), ys = cv.map(p => p["AUC_oof"]); chartSvg($("#lc"), { h: 190, xmin: Math.min(...xs), xmax: Math.max(...xs), ymin: Math.max(0, Math.min(...ys) - .1), ymax: 1, yticks: [.5, .6, .7, .8, .9, 1], yfmt: v => v.toFixed(1), xticks: xs, xfmt: v => Math.round(v * 100) + "%", series: [{ pts: xs.map((x, i) => [x, ys[i]]), color: "var(--teal)", dots: 4, name: "AUC OOF" }] }); } else $("#lc").outerHTML = `<div class="mut small">нет данных</div>`;
  const im = (r.importance || []).slice(0, 14), mx = Math.max(1e-9, ...im.map(x => Math.abs(x["падение_AUC"] || 0)));
  $("#imp").innerHTML = im.length ? im.map(x => `<div class="row" style="flex-wrap:nowrap;margin:3px 0"><span class="small" style="width:150px;overflow:hidden;text-overflow:ellipsis" title="${esc(x["признак"])}">${esc(x["признак"])}</span><div class="bar grow"><i style="width:${Math.max(2, Math.abs(x["падение_AUC"]) / mx * 100)}%"></i></div><span class="small" style="width:46px;text-align:right">${fix(x["падение_AUC"], 3)}</span></div>`).join("") : `<div class="mut small">не считалась</div>`;
}

/* ================================================================= ПРОВЕРКА НА ДАННЫХ */
async function vValidate(q) {
  const [profs, folders, cat, runs] = await Promise.all([J("/api/profiles"), J("/api/folders"), J("/api/catalog"), J("/api/tune/experiments")]);
  const profiles = profs.filter(p => p.kind === "profile");
  view.innerHTML = modelTabs("validate") + `<div class="panel"><h2>Проверка профиля на размеченных видео</h2><p class="lead">Прогон профиля по выбранным папкам и сверка с эталоном событий по протоколу организаторов (Event F1). Результат попадает в журнал и в выбор конфигураций на экране анализа.</p>
    <div class="cols" style="align-items:start"><div class="fld"><span>Профиль</span><select id="vp">${profiles.map(p => `<option ${p.name === (q.profile || mem.get("md_profile")) ? "selected" : ""}>${p.ready ? "" : "⚠ "}${esc(p.name)}</option>`).join("")}</select></div>
      <div class="fld"><span>Классификатор (заменить выбранный в профиле)</span><select id="vc"><option value="">как в профиле</option>${cat.cycle.map(c => `<option value="${esc(c.id)}">${esc(c.label)}</option>`).join("")}</select></div>
      <div class="fld"><span>Режим</span><select id="vm"><option value="events">по эталону событий (Event F1)</option><option value="clips">по папкам: «курение» / «лжекурение» (слабая оценка)</option></select></div>
      <div class="fld"><span>Порог</span><select id="vpol"><option value="fixed">как в профиле (честно)</option><option value="plateau">подобрать по плато (оптимистично)</option></select></div>
      <div class="fld"><span>Только первые N секунд клипа (0 — целиком)</span><input type="number" id="vms" min="0" value="0"></div>
      <label class="row small" style="gap:8px;align-self:flex-end"><input type="checkbox" id="vlab" checked>только размеченные клипы</label></div>
    <div class="fld" style="margin-top:12px"><span>Папки с видео</span><div class="row" id="vf">${folders.map(f => `<label class="row small" style="gap:6px"><input type="checkbox" value="${f.id}" ${/курение|лжекур/i.test(f.name) ? "checked" : ""}>${esc(f.name)} (${f.n})</label>`).join("") || '<span class="mut">Папок нет: подключите на экране «Источники»</span>'}</div></div>
    <div class="row" style="margin-top:14px"><button class="primary" id="vgo" ${profiles.length ? "" : "disabled"}>Проверить</button><span id="vst" class="grow"></span></div></div>
    <div class="panel hidden" id="vres" style="margin-top:16px"></div>
    <div class="panel" style="margin-top:16px"><h2>Журнал оценок</h2><div id="vj"></div></div>`;
  const journal = list => { $("#vj").innerHTML = list.length ? `<table><tr><th>Когда</th><th>Профиль</th><th>Режим</th><th>Клипов</th><th>F1</th><th>P / R</th><th>Порог</th><th></th></tr>${list.map(r => `<tr class="jump" data-id="${esc(r.id)}"><td class="small">${esc(String(r.created || "").slice(0, 16))}</td><td><b>${esc(r.name)}</b></td><td>${esc(r.mode)}</td><td>${r.clips ?? "—"}${r.failed ? ` <span class="chip bad">ошибок ${r.failed}</span>` : ""}</td>
      <td><b>${fix(r.f1)}</b> <span class="mut small">[${fix(r.ci_lo)}–${fix(r.ci_hi)}]</span></td><td class="small">${fix(r.precision)} / ${fix(r.recall)}</td><td>${r.threshold ?? "—"}</td><td>${r.reached ? '<span class="chip ok">цель</span>' : ""}</td></tr>`).join("")}</table>` : `<div class="empty">Оценок пока нет.</div>`;
    $$("#vj .jump").forEach(tr => tr.onclick = async () => { renderRun(await J("/api/tune/experiment/" + tr.dataset.id)); $("#vres").scrollIntoView({ behavior: "smooth" }); }); };
  journal(runs);
  const renderRun = d => {
    const box = $("#vres"); box.classList.remove("hidden"); const m = d.metrics, ci = d.ci || {}, f1ci = ci.f1 || [null, null], b = d.budget || {}, pl = d.plateau || {};
    let cand = m.threshold, prof = $("#vp").value.replace("⚠ ", "");
    box.innerHTML = `<div class="row" style="margin-bottom:10px"><h2 style="margin:0">${esc(d.name || "прогон")}</h2><span class="chip">${esc(d.mode === "clips" ? "слабая оценка по папкам" : "Event F1 по эталону")}</span><span class="mut small">${esc(String(d.created || "").slice(0, 16))} · ${esc((d.dirs || []).join(", "))} · отпечаток ${esc(d.fingerprint || "")}</span></div>
      <div class="kpis"><div class="kpi"><span class="mut small">F1</span><b>${fix(m.f1, 3)}</b><span class="mut small">${f1ci[0] == null ? "интервал недоступен (мало событий)" : "95%: " + fix(f1ci[0]) + "–" + fix(f1ci[1])}</span></div><div class="kpi"><span class="mut small">Precision / Recall</span><b>${fix(m.precision)} / ${fix(m.recall)}</b></div><div class="kpi"><span class="mut small">TP / FP / FN</span><b>${m.tp} / ${m.fp} / ${m.fn}</b></div>
        <div class="kpi"><span class="mut small">Ложных тревог в час</span><b>${m.fp_per_hour == null ? "—" : fix(m.fp_per_hour, 1)}</b></div><div class="kpi"><span class="mut small">Порог события</span><b>${fix(m.threshold)}</b><span class="mut small">плато ${fix(pl.lo)}–${fix(pl.hi)}</span></div>
        <div class="kpi"><span class="mut small">К цели ${fix(b.target)}</span><b style="color:var(${b.reached ? "--green" : "--amber"})">${b.f1 != null && b.target != null ? (b.f1 - b.target >= 0 ? "+" : "") + fix(b.f1 - b.target, 3) : "—"}</b></div></div>
      ${(d.notes || []).map(n => `<div class="notice" style="margin-bottom:6px">${IC.warn}<div>${esc(n)}</div></div>`).join("")}
      <div class="cols" style="grid-template-columns:2fr 1fr;align-items:start;margin-top:10px"><div><svg class="chart" id="vc2"></svg>${legend([["var(--amber)", "F1"], ["var(--blue)", "precision"], ["var(--green)", "recall"], ["var(--red)", "порог в прогоне"]])}</div>
        <div><h3>Порог события</h3><div class="row"><input type="number" id="vth" step="0.025" min="0" max="1" value="${cand}" style="width:90px"></div><div id="vrow" style="margin:8px 0"></div><div class="fld"><span>Записать в профиль</span><select id="vwp">${profiles.map(p => `<option ${p.name === prof ? "selected" : ""}>${esc(p.name)}</option>`).join("")}</select></div>
          <button id="vwr" style="margin-top:8px">Записать порог</button><div id="vwm" class="help" style="margin-top:6px">Порог подбирается по валидации и потому оптимистичен; перед скрытым набором его нужно зафиксировать и больше не менять.</div></div></div>
      <h3 style="margin-top:16px">Ошибки (${(d.errors || []).length})</h3>${(d.errors || []).length ? `<table><tr><th>Тип</th><th>Клип</th><th>Время</th><th>Увер.</th><th>Причина</th><th></th></tr>${d.errors.map(e => `<tr><td><span class="chip ${e.kind === "FP" ? "bad" : "warn"}">${esc(e.kind)}</span></td><td class="small">${esc(e.clip_id)}</td><td>${mmss(e.start)}–${mmss(e.end)}</td><td>${e.confidence == null ? "—" : pct(e.confidence)}</td><td class="small">${esc(e.cause_ru || e.cause || "")}</td><td>${e.video ? `<a class="btn" href="#/analysis?vid=${esc(e.video)}" style="padding:4px 12px">Разобрать</a>` : ""}</td></tr>`).join("")}</table>` : `<div class="empty">Ошибок нет.</div>`}
      ${(d.clips || []).some(c => c.error) ? `<h3 style="margin-top:16px">Клипы с ошибкой обработки</h3>${d.clips.filter(c => c.error).map(c => `<div class="err small">${esc(c.clip_id)}: ${esc(c.error)}</div>`).join("")}` : ""}`;
    const cv = d.curve || [], near = th => cv.length ? cv.reduce((a, c) => Math.abs(c.threshold - th) < Math.abs(a.threshold - th) ? c : a) : null;
    const paint = () => { if (!cv.length) { $("#vc2").outerHTML = `<div class="mut">Кривой нет</div>`; return; } const xs = cv.map(c => c.threshold);
      chartSvg($("#vc2"), { xmin: Math.min(...xs), xmax: Math.max(...xs), xticks: [.1, .2, .3, .4, .5, .6, .7, .8, .9].filter(v => v >= Math.min(...xs) && v <= Math.max(...xs)), xfmt: v => v.toFixed(1),
        series: [{ pts: cv.map(c => [c.threshold, c.precision]), color: "var(--blue)" }, { pts: cv.map(c => [c.threshold, c.recall]), color: "var(--green)" }, { pts: cv.map(c => [c.threshold, c.f1]), color: "var(--amber)", width: 3.2 }],
        bands: pl.lo != null ? [{ x0: pl.lo, x1: pl.hi, color: "#74d3c3", tip: "плато" }] : [], vlines: [{ x: m.threshold, color: "var(--red)", label: "прогон" }, { x: cand, color: "var(--ink)", dash: "2 3" }], onClick: x => { cand = Math.round(x * 40) / 40; $("#vth").value = cand; paint(); row(); } }); };
    const row = () => { const r = near(+$("#vth").value); $("#vrow").innerHTML = r ? `<table><tr><td class="mut">F1</td><td><b>${fix(r.f1)}</b></td><td class="mut">P / R</td><td>${fix(r.precision)} / ${fix(r.recall)}</td></tr><tr><td class="mut">TP/FP/FN</td><td>${r.tp}/${r.fp}/${r.fn}</td><td class="mut">ложных/ч</td><td>${r.fp_per_hour == null ? "—" : fix(r.fp_per_hour, 1)}</td></tr></table>` : ""; };
    paint(); row(); $("#vth").onchange = () => { cand = +$("#vth").value; paint(); row(); };
    $("#vwr").onclick = async () => { const name = $("#vwp").value; try { const p = await J("/api/profile/" + encodeURIComponent(name)); await POST("/api/profile/" + encodeURIComponent(name), { description: p.description, base: p.base, config: { ...p.config, "events.confidence_threshold": +$("#vth").value }, options: p.options }); $("#vwm").innerHTML = `<span class="ok">Порог ${$("#vth").value} записан в «${esc(name)}»</span>`; } catch (e) { $("#vwm").innerHTML = `<span class="err">${esc(e.message)}</span>`; } };
  };
  $("#vgo").onclick = async () => {
    const folders_ = $$("#vf input:checked").map(i => i.value); if (!folders_.length) return toast("Выберите хотя бы одну папку");
    $("#vgo").disabled = true; mem.set("md_profile", $("#vp").value.replace("⚠ ", ""));
    try { const op = await POST("/api/tune/eval", { profile: $("#vp").value.replace("⚠ ", ""), folders: folders_, classifier: $("#vc").value || null, mode: $("#vm").value, policy: $("#vpol").value, only_labeled: $("#vlab").checked, max_sec: +$("#vms").value || null, n_boot: 200 });
      const r = await pollOp(op.id, x => opStatus($("#vst"), x)); if (!r) return; if (r.state === "done") { $("#vst").innerHTML = '<span class="ok">Готово</span>'; renderRun(r.result); journal(await J("/api/tune/experiments")); } else opStatus($("#vst"), r);
    } catch (e) { const el = $("#vst"); if (el) el.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    const go = $("#vgo"); if (go) go.disabled = false;
  };
}
