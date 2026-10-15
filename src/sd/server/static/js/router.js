// Маршрутизатор на location.hash: «#/review/12?x=1». Страница открывается по ссылке, работает «Назад», нет запросов к серверу за страницами.
import { bus } from "./bus.js";

const compile = (pattern) => {
  const keys = [];
  const src = pattern.replace(/\/:(\w+)(\?)?/g, (_, k, opt) => { keys.push(k); return opt ? "(?:/([^/]+))?" : "/([^/]+)"; });
  return { re: new RegExp(`^${src === "" ? "/?" : src}/?$`), keys };
};

export function parse(hash = location.hash) {
  const raw = hash.replace(/^#/, "") || "/";
  const [path, qs = ""] = raw.split("?");
  const query = {};
  for (const [k, v] of new URLSearchParams(qs)) query[k] = v;
  return { path: path.startsWith("/") ? path : `/${path}`, query };
}

export function href(path, query) {
  const qs = new URLSearchParams(Object.entries(query || {}).filter(([, v]) => v !== undefined && v !== null && v !== "")).toString();
  return `#${path}${qs ? `?${qs}` : ""}`;
}

export function go(path, query) { location.hash = href(path, query); }
export function replace(path, query) { history.replaceState(null, "", href(path, query)); bus.emit("route:replaced", parse()); }

/** Сопоставляет путь с таблицей экранов: возвращает {screen, params, query} или null. */
export function match(screens, loc = parse()) {
  for (const sc of screens) {
    const { re, keys } = compile(sc.path);
    const m = re.exec(loc.path);
    if (m) return { screen: sc, params: Object.fromEntries(keys.map((k, i) => [k, m[i + 1] ? decodeURIComponent(m[i + 1]) : undefined])), query: loc.query };
  }
  return null;
}

export function listen(fn) {
  const on = () => fn(parse());
  window.addEventListener("hashchange", on);
  return () => window.removeEventListener("hashchange", on);
}
