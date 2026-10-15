// Поток обновлений (SSE): новые тревоги и решения. EventSource сам переподключается; после каждого переподключения экраны обновляют данные (события, пришедшие за разрыв, не теряются).
import { BASE } from "./api.js";
import { bus } from "./bus.js";

let es = null;
let state = "off";

export const live = {
  get state() { return state; },
  start() {
    if (es || typeof EventSource === "undefined") return;
    es = new EventSource(`${BASE}/events?timeout=300`);
    es.addEventListener("ready", () => { state = "on"; bus.emit("live:state", state); bus.emit("live:ready"); });
    es.addEventListener("alert.new", (e) => bus.emit("live:alert", JSON.parse(e.data)));
    es.addEventListener("alert.review", (e) => bus.emit("live:review", JSON.parse(e.data)));
    es.onerror = () => { if (state !== "off") { state = "off"; bus.emit("live:state", state); } };
  },
  stop() {
    if (es) es.close();
    es = null;
    state = "off";
    bus.emit("live:state", state);
  },
};
