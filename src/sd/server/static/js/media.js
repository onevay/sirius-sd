// Видео для браузера: файл может быть не в H.264 — сервер готовит копию в фоне. Здесь ждём готовности (с процентами) и только потом подставляем адрес в <video>.
import { api, BASE } from "./api.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** kind: 'alert' | 'video'. Возвращает адрес видео. onProgress(0..1); signal — AbortSignal для выхода с экрана. */
export async function waitForMedia(kind, id, { onProgress, signal } = {}) {
  const query = kind === "alert" ? { alert: id } : { id };
  for (let i = 0; i < 600; i++) {
    if (signal && signal.aborted) throw new DOMException("отменено", "AbortError");
    const r = await api.get("/media/status", query);
    if (r.state === "direct" || r.state === "ready") return kind === "alert" ? `${BASE}/media/alert/${id}/clip` : `${BASE}/media/video/${encodeURIComponent(id)}`;
    if (r.state === "error") throw new Error(`Не удалось подготовить видео: ${r.error || "ошибка перекодирования"}`);
    onProgress && onProgress(r.pct || 0);
    await sleep(i < 5 ? 600 : 1500);
  }
  throw new Error("Видео готовится слишком долго");
}

