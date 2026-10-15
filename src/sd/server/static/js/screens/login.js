// Экран входа: слева — о системе и состояние сервера, справа — форма. Ошибки показываются в форме, поле пароля очищается.
import { h, icon, clear } from "../dom.js";
import { session } from "../session.js";
import { button } from "../ui.js";
import { errText } from "../api.js";

export default function login(root, { message, onDone }) {
  const user = h("input", { class: "input", name: "username", autocomplete: "username", required: true, autocapitalize: "none", spellcheck: "false" });
  const pass = h("input", { class: "input", name: "password", type: "password", autocomplete: "current-password", required: true });
  const err = h("div", { class: "banner banner--alarm", role: "alert", hidden: !message }, message || "");
  const submit = button({ label: "Войти", kind: "accent", size: "lg", block: true, type: "submit" });

  const form = h("form", { novalidate: true },
    h("div", null, h("h1", null, "Вход"), h("p", { class: "muted", style: { marginTop: "6px" } }, "Введите имя и пароль, выданные администратором.")),
    err,
    h("label", { class: "field" }, "Имя пользователя", user),
    h("label", { class: "field" }, "Пароль", pass),
    submit);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!user.value.trim() || !pass.value) { err.hidden = false; err.textContent = "Введите имя и пароль"; (user.value.trim() ? pass : user).focus(); return; }
    submit.disabled = true;
    err.hidden = true;
    try {
      await session.login(user.value.trim(), pass.value);
      onDone();
    } catch (ex) {
      err.hidden = false;
      err.textContent = errText(ex);
      pass.value = "";
      pass.focus();
      submit.disabled = false;
    }
  });

  const status = h("div", { class: "small", style: { color: "#B6C2B9" } }, session.health ? `Сервер ${session.health.version} · база данных ${session.health.db ? "работает" : "недоступна"}` : "");
  clear(root).append(h("div", { class: "login" },
    h("div", { class: "login__art" },
      h("div", { class: "brand", style: { padding: 0 } }, h("div", { class: "brand__mark", style: { background: "var(--accent)" } }, icon("shield", 18, { style: { color: "#fff" } })),
        h("div", { class: "brand__t" }, h("b", { style: { color: "#fff" } }, "Детектор курения"), h("span", { style: { color: "#B6C2B9" } }, "Videofor × Сириус"))),
      h("div", { class: "stack", style: { marginTop: "auto", maxWidth: "460px" } },
        h("h1", { style: { color: "#fff", fontSize: "40px" } }, "Тревога — решение оператора, а не догадка системы"),
        h("p", { style: { color: "#B6C2B9", fontSize: "15px" } }, "Камеры анализируются локально. Каждое событие проходит проверку человеком; решения копятся и улучшают качество модели.")),
      status),
    h("div", { class: "login__form" }, form)));
  user.focus();
}
