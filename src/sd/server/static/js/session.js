// Сессия: кто вошёл, нужен ли вход, права. Роли идут по возрастанию прав (как на сервере).
import { api, token } from "./api.js";
import { bus } from "./bus.js";

export const ROLES = ["viewer", "operator", "developer", "admin"];
export const ROLE_RU = { viewer: "наблюдатель", operator: "оператор", developer: "разработчик", admin: "администратор" };
export const rank = (r) => ROLES.indexOf(r);

export const session = {
  me: null,
  health: null,
  can(role) { return !!this.me && rank(this.me.role) >= rank(role); },

  /** Состояние сервера и текущий пользователь (если вход не нужен или токен ещё действует). */
  async init() {
    this.health = await api.get("/health");
    this.me = null;
    if (!this.health.auth_required || token.get()) {
      try { this.me = await api.get("/auth/me"); } catch (e) { if (e.status !== 401) throw e; token.set(""); }
    }
    return this.me;
  },

  async login(username, password) {
    const r = await api.post("/auth/login", { username, password, cookie: true });
    token.set(r.access_token);
    this.me = r.user;
    return r.user;
  },

  async logout() {
    try { await api.post("/auth/logout"); } catch { /* cookie удалится и без ответа */ }
    token.set("");
    this.me = null;
    bus.emit("auth:out");
  },
};
