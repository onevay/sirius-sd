// Мой профиль: смена пароля, выход со всех устройств, личные API-токены для скриптов и интеграций (значение токена показывается один раз).
import { h, clear, icon, fmtDateTime, ago } from "../dom.js";
import { api, errText } from "../api.js";
import { session, ROLE_RU } from "../session.js";
import { button, card, chip, confirmDialog, empty, errorBox, formModal, openModal, pageHead, skeleton, toast } from "../ui.js";

export default async function account(root, ctx) {
  const me = session.me;
  const tokensBox = h("div", null, skeleton(1, 80));
  const local = me.id == null;

  root.append(pageHead("Мой профиль", `${me.display_name || me.username} · ${ROLE_RU[me.role] || me.role}`),
    h("div", { class: "cols" },
      h("div", { class: "f1 stack" },
        card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "Учётная запись")),
          kv("Имя для входа", me.username), kv("Роль", ROLE_RU[me.role] || me.role), kv("Последний вход", me.last_login_at ? `${fmtDateTime(me.last_login_at)} (${ago(me.last_login_at)})` : "—"),
          local ? h("p", { class: "small muted", style: { marginTop: "10px" } }, "Вход отключён: вы работаете как локальный администратор. Пароль и токены не нужны.") : h("div", { class: "row", style: { marginTop: "14px" } },
            button({ label: "Сменить пароль", ic: "edit", size: "sm", onClick: changePassword }), button({ label: "Выйти на всех устройствах", ic: "logout", size: "sm", onClick: logoutAll }))),
        session.health && session.health.auth_required ? button({ label: "Выйти", ic: "logout", onClick: () => session.logout() }) : null),
      h("div", { class: "f15" }, card(null, h("div", { class: "card__head" }, h("h2", { class: "grow" }, "API-токены"), local ? null : button({ label: "Создать токен", ic: "plus", size: "sm", kind: "primary", onClick: createToken })),
        h("p", { class: "muted small", style: { marginBottom: "12px" } }, "Токен нужен скриптам и внешним системам: передаётся в заголовке Authorization: Bearer. Права токена совпадают с вашими."), tokensBox))));

  function kv(k, v) { return h("div", { class: "row row--nowrap", style: { justifyContent: "space-between", padding: "8px 0", borderTop: "1px solid var(--line-2)" } }, h("span", { class: "muted" }, k), h("span", null, v)); }

  async function loadTokens() {
    if (local) { clear(tokensBox).append(empty("Токены не нужны", "Вход отключён.")); return; }
    clear(tokensBox).append(skeleton(1, 80));
    let list;
    try { list = await api.get("/auth/tokens"); } catch (e) { clear(tokensBox).append(errorBox(errText(e), loadTokens)); return; }
    if (!ctx.active()) return;
    clear(tokensBox);
    if (!list.length) { tokensBox.append(empty("Токенов нет", "Создайте токен для скрипта или интеграции.")); return; }
    tokensBox.append(h("div", { class: "tablewrap" }, h("table", { class: "table" }, h("thead", null, h("tr", null, ...["Название", "Начало", "Срок", "Использован", ""].map((c) => h("th", null, c)))),
      h("tbody", null, ...list.map((t) => h("tr", null, h("td", null, h("b", null, t.name), t.revoked_at ? h("span", { style: { marginLeft: "8px" } }, chip("отозван", "")) : null), h("td", { class: "mono small" }, `${t.prefix}…`), h("td", { class: "small" }, t.expires_at ? fmtDateTime(t.expires_at) : "бессрочный"),
        h("td", { class: "small" }, t.last_used_at ? ago(t.last_used_at) : "не использовался"), h("td", null, t.revoked_at ? null : button({ label: "Отозвать", size: "sm", onClick: () => revoke(t) }))))))));
  }

  async function createToken() {
    let created;
    const r = await formModal({ title: "Новый API-токен", fields: [{ name: "name", label: "Название", required: true, placeholder: "Например: выгрузка отчётов" }, { name: "ttl", label: "Срок жизни, дней", help: "Пусто — без срока" }], submit: "Создать",
      onSubmit: async (v) => { const ttl = String(v.ttl).trim(); if (ttl && !(Number.isInteger(Number(ttl)) && Number(ttl) >= 1 && Number(ttl) <= 3650)) throw new Error("Срок: целое число дней от 1 до 3650"); created = await api.post("/auth/tokens", { name: v.name.trim(), ttl_days: ttl ? Number(ttl) : null }); } });
    if (!r || !created) return;
    const field = h("input", { class: "input mono", readonly: true, value: created.token, "aria-label": "Значение токена" });
    const close = openModal(h("div", { class: "modal" }, h("h2", null, "Токен создан"), h("div", { class: "banner banner--alarm" }, icon("warn", 18), "Скопируйте токен сейчас: потом его показать нельзя."), field,
      h("div", { class: "modal__foot" }, button({ label: "Скопировать", ic: "link", onClick: async () => { try { await navigator.clipboard.writeText(created.token); toast("Скопировано", { kind: "ok" }); } catch { field.select(); toast("Выделено: нажмите Ctrl+C"); } } }), button({ label: "Готово", kind: "primary", onClick: () => close() }))));
    field.select();
    loadTokens();
  }

  async function revoke(t) {
    if (!(await confirmDialog(`Токен «${t.name}» перестанет работать сразу.`, { ok: "Отозвать", danger: true, title: "Отозвать токен?" }))) return;
    try { await api.del(`/auth/tokens/${t.id}`); toast("Токен отозван"); loadTokens(); } catch (e) { toast(errText(e), { kind: "alarm" }); }
  }

  async function changePassword() {
    const r = await formModal({ title: "Смена пароля", intro: "После смены прежние сессии закроются и нужно войти заново.",
      fields: [{ name: "old", label: "Текущий пароль", type: "password", required: true }, { name: "new1", label: "Новый пароль", type: "password", required: true, help: "Не короче 8 символов" }, { name: "new2", label: "Повторите новый пароль", type: "password", required: true }], submit: "Сменить",
      onSubmit: async (v) => { if (v.new1 !== v.new2) throw new Error("Новые пароли не совпадают"); await api.post("/auth/password", { old_password: v.old, new_password: v.new1 }); } });
    if (r) { toast("Пароль изменён, войдите заново", { kind: "ok" }); session.logout(); }
  }

  async function logoutAll() {
    if (!(await confirmDialog("Все выданные вам токены доступа перестанут действовать, включая текущий. Нужно будет войти заново.", { ok: "Выйти везде", title: "Выйти на всех устройствах?" }))) return;
    try { await api.post("/auth/logout-all"); } catch (e) { toast(errText(e), { kind: "alarm" }); return; }
    session.logout();
  }

  await loadTokens();
}
