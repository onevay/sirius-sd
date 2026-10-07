"""Видео и кадры с отметками промежуточных результатов (руководство, шаг 9): точки, проценты, состояния, таймлайн.

Что рисуется (всё включается/выключается флагами):
  * рамка человека: ID, уверенность детектора, высота в px; < min_person_height_px — серая «IGNORE»;
  * 17 ключевых точек: цвет = уверенность (красный -> жёлтый -> зелёный), у носа и запястий подписан процент;
  * оценка рта (крест) и отрезок запястье–рот, цвет = состояние автомата REST/APPROACH/AT_MOUTH/RETRACT, подпись d_t;
  * счётчик циклов человека, плашка «SMOKING-LIKE 0.87» на время события;
  * детекции предмета на кропе (extras["objects"]) и проценты видео-моделей X-CLIP/VideoMAE (extras["tube"]);
  * таймлайн под видео: состояния каждого трека, циклы, события, курсор времени.
Текст только ASCII (шрифты OpenCV не поддерживают кириллицу).
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .cycles import STATE_ID
from .features import COCO_SKELETON, LWRI, NOSE, RWRI
from .tracks import Tracks
from .video_io import FFmpegWriter, iter_frames, probe

STATE_NAMES = {v: k for k, v in STATE_ID.items()}
STATE_COLORS = {0: (140, 140, 140), 1: (0, 215, 255), 2: (60, 60, 255), 3: (255, 160, 30), 4: (200, 0, 200)}
PALETTE = [(255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 165, 0), (186, 85, 211), (0, 206, 209), (220, 20, 60), (154, 205, 50)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def conf_color(c: float) -> tuple[int, int, int]:
    """BGR: 0 -> красный, 0.5 -> жёлтый, 1 -> зелёный."""
    c = float(np.clip(c, 0, 1))
    return (0, int(255 * min(1.0, 2 * c)), int(255 * min(1.0, 2 * (1 - c))))


def tid_color(tid: int) -> tuple[int, int, int]:
    r, g, b = PALETTE[tid % len(PALETTE)]
    return (b, g, r)


def put_text(img, text, org, scale=0.5, color=(255, 255, 255), thick=1, bg=(0, 0, 0)):
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), FONT, scale, bg, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


class Renderer:
    def __init__(self, tr: Tracks, cfg: dict, series: pd.DataFrame | None = None, cycles: pd.DataFrame | None = None,
                 states: pd.DataFrame | None = None, events: pd.DataFrame | None = None, extras: dict | None = None,
                 out_width: int | None = None, show_timeline: bool | None = None):
        self.tr, self.cfg = tr, cfg
        self.rc = cfg["render"]
        self.out_w = out_width or cfg["video"]["render_max_width"]
        self.show_timeline = self.rc["show_timeline"] if show_timeline is None else show_timeline
        self.min_h = cfg["video"]["min_person_height_px"]
        self.series = series
        self.cycles = cycles if cycles is not None else pd.DataFrame(columns=["tid", "start", "end"])
        self.events = events if events is not None else pd.DataFrame(columns=["person_track_id", "start_sec", "end_sec", "confidence"])
        self.extras = extras or {}
        self.pos_by_frame = tr.df.groupby("frame").indices           # frame -> позиции строк
        self.series_idx = {}
        if series is not None and len(series):
            self.series_idx = {tid: g.set_index("frame") for tid, g in series.groupby("tid")}
        self.states_by_tid = {}
        if states is not None and len(states):
            self.states_by_tid = {tid: g.reset_index(drop=True) for tid, g in states.groupby("tid")}
        ft = tr.frame_t
        self.t0, self.t1 = (float(ft.t.min()), float(ft.t.max())) if len(ft) else (0.0, 1.0)
        self.small = set(tr.summary(self.min_h).query("ignore_small").tid.tolist()) if len(tr.df) else set()

    # ------------------------------------------------------------------ pieces
    def _person(self, img, f, row, kp, frame, t):
        tid = int(row.tid)
        small = tid in self.small or row.h < self.min_h
        col = (150, 150, 150) if small else tid_color(tid)
        x1, y1, x2, y2 = [int(v * f) for v in (row.x1, row.y1, row.x2, row.y2)]
        th = max(1, int(round(2 * f * 1.5)))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
        head = f"ID{tid} {row.score:.0%} h{row.h:.0f}px" + (" IGNORE" if small else "")
        put_text(img, head, (x1, max(12, y1 - 5)), 0.45, col)
        r = max(2, int(round(self.rc["point_radius"] * f * 1.3)))
        if self.rc["show_skeleton"]:
            for a, b in COCO_SKELETON:
                if kp[a, 2] > 0.2 and kp[b, 2] > 0.2:
                    cv2.line(img, (int(kp[a, 0] * f), int(kp[a, 1] * f)), (int(kp[b, 0] * f), int(kp[b, 1] * f)), (200, 200, 200), 1, cv2.LINE_AA)
        if self.rc["show_keypoints"]:
            for k in range(min(len(kp), 17)):
                if kp[k, 2] < 0.05:
                    continue
                p = (int(kp[k, 0] * f), int(kp[k, 1] * f))
                cv2.circle(img, p, r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, p, r, conf_color(float(kp[k, 2])), -1, cv2.LINE_AA)
                if self.rc["show_percent"] and k in (NOSE, LWRI, RWRI):
                    put_text(img, f"{kp[k, 2]:.0%}", (p[0] + r + 2, p[1] + 4), 0.38, conf_color(float(kp[k, 2])))
        # признаки жеста
        s = self.series_idx.get(tid)
        if s is not None and frame in s.index:
            sr = s.loc[frame]
            st = 0
            sb = self.states_by_tid.get(tid)
            if sb is not None and "frame" in sb:
                m = sb.frame.values == frame
                if m.any():
                    st = int(sb.state.values[m][0])
            if np.isfinite(sr.mouth_x):
                mp = (int(sr.mouth_x * f), int(sr.mouth_y * f))
                cv2.drawMarker(img, mp, (255, 255, 255), cv2.MARKER_CROSS, max(6, int(10 * f)), 1, cv2.LINE_AA)
                if sr.hand in (0, 1) and np.isfinite(sr.d):
                    wx, wy = (sr.handL_x, sr.handL_y) if sr.hand == 0 else (sr.handR_x, sr.handR_y)
                    if np.isfinite(wx):
                        wp = (int(wx * f), int(wy * f))
                        cv2.line(img, wp, mp, STATE_COLORS[st], max(1, int(2 * f * 1.5)), cv2.LINE_AA)
                        put_text(img, f"d={sr.d:.2f} {STATE_NAMES[st]}", (wp[0] + 6, wp[1] - 6), 0.42, STATE_COLORS[st])
            n_cyc = int(((self.cycles.tid == tid) & (self.cycles.end <= t)).sum()) if len(self.cycles) else 0
            put_text(img, f"cycles:{n_cyc}", (x1, y2 + int(14 * max(f, 0.7))), 0.45, col)
        cs = self.extras.get("cycle_scores")
        if cs is not None and len(cs):
            m = cs[(cs.tid == tid) & (cs.start <= t) & (cs.end + 1.5 >= t)]    # оценка цикла видна во время цикла и ещё 1.5 с после
            if len(m):
                r = m.iloc[-1]
                parts = [f"cycle P={r.score:.2f}"]
                for col, nm in (("vlm_yesno", "vlm"), ("photo_p_mean", "photo")):
                    if col in m.columns and np.isfinite(r[col]):
                        parts.append(f"{nm} {r[col]:.2f}")
                put_text(img, "  ".join(parts), (x1, y2 + int(32 * max(f, 0.7))), 0.5, conf_color(1.0 - float(np.clip(r.score, 0, 1))))   # высокая оценка — красный
        ev = self.events[(self.events.person_track_id == tid) & (self.events.start_sec <= t) & (self.events.end_sec >= t)]
        if len(ev):
            e = ev.iloc[0]
            cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 0, 255), th + 2)
            put_text(img, f"SMOKING-LIKE {e.confidence:.2f}", (x1, max(26, y1 - 22)), 0.65, (255, 255, 255), 2, bg=(0, 0, 200))

    def _objects(self, img, f, frame):
        o = self.extras.get("objects")
        if o is None or not len(o):
            return
        for r in o[o.frame == frame].itertuples():
            p1, p2 = (int(r.x1 * f), int(r.y1 * f)), (int(r.x2 * f), int(r.y2 * f))
            cv2.rectangle(img, p1, p2, (0, 255, 255), 2)
            put_text(img, f"{r.cls} {r.conf:.0%}", (p1[0], max(10, p1[1] - 4)), 0.45, (0, 255, 255))

    def _tube_bars(self, img, f, t):
        tb = self.extras.get("tube")
        if tb is None or not len(tb):
            return
        w = tb[(tb.t0 <= t) & (tb.t1 >= t)]
        if not len(w):
            return
        r = w.iloc[0]
        prompts = [c for c in tb.columns if c.startswith("p_")]
        order = sorted(prompts, key=lambda c: -r[c])[:4]
        x0, y0 = img.shape[1] - 330, 40
        cv2.rectangle(img, (x0 - 8, y0 - 22), (img.shape[1] - 6, y0 + 22 * len(order) + 4), (0, 0, 0), -1)
        put_text(img, f"X-CLIP ID{int(r.tid)} [{r.t0:.1f}-{r.t1:.1f}s]", (x0, y0 - 6), 0.45, (255, 255, 255), bg=(0, 0, 0))
        for i, c in enumerate(order):
            y = y0 + 8 + 22 * i
            cv2.rectangle(img, (x0, y), (x0 + int(200 * r[c]), y + 12), (0, 200, 255), -1)
            put_text(img, f"{c[2:][:24]} {r[c]:.0%}", (x0 + 4, y + 10), 0.38, (255, 255, 255))

    def _hud(self, img, t, frame):
        put_text(img, f"t={t:7.2f}s  frame {frame}", (8, 18), 0.55, (255, 255, 255))
        x = 8
        for i in range(5):
            cv2.rectangle(img, (x, 26), (x + 12, 38), STATE_COLORS[i], -1)
            put_text(img, STATE_NAMES[i], (x + 15, 37), 0.34, (255, 255, 255))
            x += 15 + 10 * len(STATE_NAMES[i]) + 8
        for i, c in enumerate((0.1, 0.5, 0.9)):
            cv2.circle(img, (14 + i * 70, 56), 5, conf_color(c), -1)
            put_text(img, f"conf {c:.0%}", (22 + i * 70, 60), 0.34, (255, 255, 255))

    def _timeline(self, width: int, t: float) -> np.ndarray:
        tids = self.tr.tids[:8]
        row_h, pad = 16, 4
        h = pad * 2 + row_h * max(1, len(tids)) + 14
        strip = np.full((h, width, 3), 28, np.uint8)
        x_of = lambda tt: int(np.clip((tt - self.t0) / max(self.t1 - self.t0, 1e-6), 0, 1) * (width - 60)) + 52
        for i, tid in enumerate(tids):
            y = pad + i * row_h
            put_text(strip, f"ID{tid}", (4, y + 12), 0.4, tid_color(tid), bg=(28, 28, 28))
            sb = self.states_by_tid.get(tid)
            if sb is not None:
                xs = np.array([x_of(v) for v in sb.t.values])
                for j in range(len(sb) - 1):
                    s = int(sb.state.values[j])
                    cv2.rectangle(strip, (xs[j], y + 2), (max(xs[j + 1], xs[j] + 1), y + row_h - 3), STATE_COLORS[s] if s else (70, 70, 70), -1)
            else:
                g = self.tr.df[self.tr.df.tid == tid]
                cv2.rectangle(strip, (x_of(g.t.min()), y + 2), (x_of(g.t.max()), y + row_h - 3), (90, 90, 90), -1)
            if len(self.cycles):
                for c in self.cycles[self.cycles.tid == tid].itertuples():
                    cv2.rectangle(strip, (x_of(c.start), y), (x_of(c.end), y + row_h - 1), (255, 255, 255), 1)
            if len(self.events):
                for e in self.events[self.events.person_track_id == tid].itertuples():
                    cv2.rectangle(strip, (x_of(e.start_sec), y + row_h - 5), (x_of(e.end_sec), y + row_h - 2), (0, 0, 255), -1)
        cx = x_of(t)
        cv2.line(strip, (cx, 0), (cx, h), (0, 255, 255), 1)
        put_text(strip, f"{self.t0:.0f}s", (50, h - 3), 0.33, (180, 180, 180), bg=(28, 28, 28))
        put_text(strip, f"{self.t1:.0f}s", (width - 36, h - 3), 0.33, (180, 180, 180), bg=(28, 28, 28))
        return strip

    # ------------------------------------------------------------------ main
    def render(self, frame: int, t: float, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        f = self.out_w / W if W > self.out_w else 1.0
        out = cv2.resize(img, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA) if f != 1.0 else img.copy()
        pos = self.pos_by_frame.get(frame, [])
        for p in pos:
            self._person(out, f, self.tr.df.iloc[p], self.tr.kp[p], frame, t)
        self._objects(out, f, frame)
        self._tube_bars(out, f, t)
        self._hud(out, t, frame)
        if self.show_timeline and len(self.tr.tids):
            out = np.vstack([out, self._timeline(out.shape[1], t)])
        return out


def render_video(video: str | Path, tr: Tracks, cfg: dict, out_path: str | Path, renderer: Renderer | None = None,
                 start: float | None = None, end: float | None = None, progress=None, **kw) -> Path:
    """Видео с разметкой. Кадры берутся с тем же шагом, что и при обработке позы, поэтому отметки точно совпадают с детекциями."""
    info = probe(video)
    r = renderer or Renderer(tr, cfg, **kw)
    stride = int(tr.meta.get("stride", 1))
    s0 = float(tr.meta.get("start", 0.0)) if start is None else start
    e0 = tr.meta.get("end") if end is None else end
    n = max(1, len(tr.frame_t))
    out_fps = info.fps / stride
    w = None
    try:
        for i, fr in enumerate(iter_frames(video, s0, e0, stride)):
            img = r.render(fr.idx, fr.t, fr.img)
            if w is None:
                w = FFmpegWriter(out_path, img.shape[1], img.shape[0], out_fps)
            w.write(img)
            if progress:
                progress(i + 1, n)
    finally:
        if w:
            w.close()
    return Path(out_path)


def render_frame_img(video: str | Path, tr: Tracks, cfg: dict, t: float, renderer: Renderer | None = None, **kw) -> tuple[np.ndarray, float]:
    """Размеченный кадр (BGR) на момент ближайшего обработанного кадра + его точное время. Для UI и скриншотов."""
    r = renderer or Renderer(tr, cfg, **kw)
    ft = tr.frame_t
    j = int(np.abs(ft.t.values - t).argmin())
    frame, tt = int(ft.frame.values[j]), float(ft.t.values[j])
    for fr in iter_frames(video, max(0.0, tt - 0.001), None, 1, max_frames=1):
        return r.render(frame, tt, fr.img), tt
    raise RuntimeError("не удалось прочитать кадр")


def render_frame(video: str | Path, tr: Tracks, cfg: dict, t: float, out_png: str | Path, renderer: Renderer | None = None, **kw) -> Path:
    """Один размеченный кадр (PNG) — скриншот для проверки глазами."""
    img, _ = render_frame_img(video, tr, cfg, t, renderer, **kw)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return Path(out_png)
