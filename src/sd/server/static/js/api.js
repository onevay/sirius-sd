// Клиент API v1: заголовок Authorization, разбор ошибок, повторяемые параметры (?status=a&status=b). Единственное место, где интерфейс ходит в сеть.
import { bus } from "./bus.js";

export const BASE = "/api/v1";
const KEY = "sd_token";

export class ApiError extends Error {
  constructor(status, detail, path) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.path = path;
  }
}

const store = {
  get() { try { return sessionStorage.getItem(KEY) || ""; } catch { return ""; } },
  set(v) { try { v ? sessionStorage.setItem(KEY, v) : sessionStorage.removeItem(KEY); } catch { /* приватный режим: токен живёт до перезагрузки */ } },
};
let memory = store.get();

export const token = { get: () => memory, set: (v) => { memory = v || ""; store.set(memory); } };

export function url(path, query) {
  const u = new URL(BASE + path, location.origin);
  for (const [k, v] of Object.entries(query || {})) {
    if (v === null || v === undefined || v === "") continue;
    for (const x of Array.isArray(v) ? v : [v]) u.searchParams.append(k, String(x));
  }
  return u.pathname + u.search;
}

function describe(body, status) {
  const d = body && body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => `${(e.loc || []).slice(1).join(".") || "поле"}: ${e.msg}`).join("; ");
  return { 401: "Требуется вход", 403: "Недостаточно прав", 404: "Не найдено", 409: "Конфликт данных", 422: "Неверные данные", 429: "Слишком много попыток, подождите" }[status] || `Ошибка сервера (${status})`;
}

export async function request(method, path, { query, body, form, raw } = {}) {
  const headers = {};
  if (memory) headers.Authorization = `Bearer ${memory}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  let res;
  try {
    res = await fetch(url(path, query), { method, headers, body: payload, credentials: "same-origin" });
  } catch {
    throw new ApiError(0, "Нет связи с сервером", path);
  }
  if (raw) return res;
  const ct = res.headers.get("content-type") || "";
  let data = null;
  if (res.status !== 204) data = ct.includes("json") ? await res.json().catch(() => null) : await res.text().catch(() => "");
  if (!res.ok) {
    if (res.status === 401 && !path.startsWith("/auth/login")) bus.emit("auth:lost", path);
    throw new ApiError(res.status, describe(data, res.status), path);
  }
  return data;
}

export const api = {
  get: (path, query) => request("GET", path, { query }),
  post: (path, body, query) => request("POST", path, { body, query }),
  put: (path, body) => request("PUT", path, { body }),
  patch: (path, body) => request("PATCH", path, { body }),
  del: (path, query) => request("DELETE", path, { query }),
  upload: (path, form) => request("POST", path, { form }),
};

/** Сообщение об ошибке для пользователя. */
export const errText = (e) => (e instanceof ApiError ? e.detail : e && e.message ? e.message : String(e));
