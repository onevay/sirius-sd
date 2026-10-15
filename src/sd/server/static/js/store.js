// Справочники, которые нужны многим экранам: камеры, причины «ложных», статистика. Кэшируются на короткое время; ошибки не кэшируются.
import { api } from "./api.js";
import { bus } from "./bus.js";

function cached(loader, ttl) {
  let value, at = 0, pending = null;
  const get = (force = false) => {
    if (!force && value !== undefined && Date.now() - at < ttl) return Promise.resolve(value);
    if (!pending) pending = loader().then((v) => { value = v; at = Date.now(); return v; }).finally(() => { pending = null; });
    return pending;
  };
  get.reset = () => { value = undefined; at = 0; };
  return get;
}

export const cameras = cached(() => api.get("/cameras"), 30000);
export const reasons = cached(() => api.get("/review-reasons"), 600000);
export const stats = cached(() => api.get("/stats"), 3000);

bus.on("auth:out", () => { cameras.reset(); stats.reset(); });

/** Подпись камеры: «Район · 05»; если районов нет — идентификатор. */
export function camName(x) {
  if (!x) return "—";
  const district = x.district, idx = x.cam_index;
  if (district && idx) return `${district} · ${idx}`;
  if (x.title) return x.title;
  return x.camera_id || "—";
}

/** Словарь camera_id → CameraMetrics для быстрых подписей в списках. */
export async function cameraIndex() {
  const list = await cameras();
  return new Map(list.map((m) => [m.camera.camera_id, m]));
}
