"use strict";
/* Рейка навигации, маршрутизация, поиск места */
const NAV = [["map", "Карта", "monitor"], ["analysis", "Анализ видео", "play"], ["models", "Модели и профили", "models"], ["transfer", "Передача и конфигурации", "send"]];
const BOTTOM = ["sources", "Источники видео", "pin"];
const SHORT = { map: "Карта", analysis: "Анализ", models: "Модели", transfer: "Передача", sources: "Источники" };
const mk = ([r, t, i]) => `<a class="ric" href="#/${r}" data-r="${r}" title="${t}"><i>${IC[i]}</i>${SHORT[r]}</a>`;
$("#nav").innerHTML = NAV.map(mk).join(""); $("#navBottom").innerHTML = mk(BOTTOM);
const routes = { map: vMap, multi: vMulti, alerts: vAlerts, analysis: vAnalysis, models: vModels, transfer: vTransfer, sources: vSources };
async function route() {
  window.__routeSeq = (window.__routeSeq || 0) + 1; leave();
  let [path, qs] = (location.hash.slice(2) || "map").split("?");
  if (path === "video") path = "analysis";
  const name = routes[path] ? path : "map";
  $$(".ric").forEach(a => a.classList.toggle("on", a.dataset.r === name || (name === "multi" && a.dataset.r === "map")));
  $("#bell").classList.toggle("on", name === "alerts");
  try { await refreshBase(); await routes[name](Object.fromEntries(new URLSearchParams(qs || ""))); }
  catch (e) { view.innerHTML = `<div class="panel err">Ошибка: ${esc(e.message)}</div>`; }
}
addEventListener("hashchange", route);
/* поиск места: улица, район, номер камеры */
const sug = $("#suggest"), sin = $("#search");
const goCam = id => { sug.classList.add("hidden"); sin.value = ""; if (location.hash.startsWith("#/map") && window.__mapFocus) window.__mapFocus(id); else location.hash = `#/map?cam=${encodeURIComponent(id)}`; };
const found = () => { const q = sin.value.trim().toLowerCase(); return q ? S.cams.filter(c => `${c.street} ${c.district} ${c.index} ${c.title} ${c.camera_id}`.toLowerCase().includes(q)).slice(0, 7) : []; };
sin.oninput = () => { const f = found(); sug.innerHTML = f.map(c => `<div data-id="${esc(c.camera_id)}"><b>${esc(c.street)}</b> <span class="mut small">${esc(c.district)} · камера ${esc(c.index)}</span></div>`).join(""); sug.classList.toggle("hidden", !f.length); $$("div", sug).forEach(d => d.onclick = () => goCam(d.dataset.id)); };
sin.onkeydown = e => { if (e.key === "Enter") { const f = found(); if (f[0]) goCam(f[0].camera_id); else toast("Место не найдено"); } if (e.key === "Escape") sug.classList.add("hidden"); };
$("#searchGo").onclick = () => { const f = found(); if (f[0]) goCam(f[0].camera_id); else toast("Введите улицу, район или номер камеры"); };
document.addEventListener("click", e => { if (!e.target.closest(".search")) sug.classList.add("hidden"); });
route();
