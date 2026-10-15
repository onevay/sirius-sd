// Точка входа: узнаём состояние сервера → вход (если нужен) → каркас. Выход и истёкший токен возвращают на экран входа без перезагрузки страницы.
import { h, clear, icon } from "./dom.js";
import { bus } from "./bus.js";
import { session } from "./session.js";
import { live } from "./live.js";
import { token, errText } from "./api.js";
import { button } from "./ui.js";
import { startShell } from "./shell.js";

const root = document.getElementById("root");
let shell = null;
let booting = false;

function stopApp() {
  live.stop();
  if (shell) { shell.destroy(); shell = null; }
}

function startApp() {
  stopApp();
  live.start();
  shell = startShell(root);
}

async function showLogin(message) {
  stopApp();
  clear(root);
  try {
    const m = await import("./screens/login.js");
    m.default(root, { message, onDone: startApp });
  } catch (e) { fatal(e); }
}

function fatal(e) {
  stopApp();
  clear(root).append(h("div", { class: "stage" }, h("div", { class: "stage__body", style: { alignItems: "center", justifyContent: "center" } },
    h("div", { class: "card", style: { maxWidth: "460px" } }, h("div", { class: "stack" }, icon("warn", 28), h("h2", null, "Сервер недоступен"), h("p", { class: "muted" }, errText(e)),
      button({ label: "Повторить", kind: "primary", onClick: boot }))))));
}

async function boot() {
  if (booting) return;
  booting = true;
  clear(root).append(h("div", { class: "stage" }, h("div", { class: "stage__body" }, h("div", { class: "skel", style: { flex: 1, minHeight: "200px" } }))));
  try {
    await session.init();
    if (session.me) startApp(); else await showLogin();
  } catch (e) { console.error(e); fatal(e); }
  finally { booting = false; }
}

bus.on("auth:lost", () => { if (!session.me) return; session.me = null; token.set(""); showLogin("Сессия закончилась — войдите снова."); });
bus.on("auth:out", () => showLogin());

window.addEventListener("unhandledrejection", (e) => console.error("необработанная ошибка", e.reason));
boot();
