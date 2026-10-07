"""Прямой признак: предмет (сигарета/вейп) на кропе кисть–рот (руководство §5.5, шаг 6).

Сигарета на кадре — 2–20 px; детектор на полном кадре её не увидит. Вырезаем квадрат вокруг активной кисти и рта, увеличиваем до
`evidence.imgsz` и гоняем детектор на кропе (идея SAHI, но с умным выбором фрагмента). «Устойчиво виден» = детекция в >= k из n
соседних кадров у той же кисти.
"""
from __future__ import annotations

from . import _env  # noqa: F401

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .models import load_registry, local_weights
from .tracks import Tracks
from .video_io import choose_stride, iter_frames


def hand_mouth_crop(img: np.ndarray, mouth: tuple[float, float], wrist: tuple[float, float], s_px: float, cfg: dict
                    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Квадратный кроп вокруг середины отрезка запястье–рот. Возвращает (кроп, (x1,y1,x2,y2) в координатах кадра)."""
    ev = cfg["evidence"]
    H, W = img.shape[:2]
    cx, cy = (mouth[0] + wrist[0]) / 2, (mouth[1] + wrist[1]) / 2
    seg = float(np.hypot(mouth[0] - wrist[0], mouth[1] - wrist[1]))
    side = int(max(ev["crop_scale"] * s_px, 1.6 * seg, ev["crop_min_px"]))
    side = min(side, min(H, W))
    x1 = int(np.clip(cx - side / 2, 0, W - side))
    y1 = int(np.clip(cy - side / 2, 0, H - side))
    return img[y1:y1 + side, x1:x1 + side], (x1, y1, x1 + side, y1 + side)


class ObjectDetector:
    """YOLO-детектор предмета из реестра весов (community-.pt перед загрузкой проходят сканирование)."""

    def __init__(self, detector_id: str, cfg: dict):
        from ultralytics import YOLO, settings

        settings.update({"sync": False})
        self.id = detector_id
        self.cfg = cfg
        self.model = YOLO(str(local_weights(detector_id)))
        self.names = self.model.names

    def detect(self, crop: np.ndarray) -> list[dict]:
        ev = self.cfg["evidence"]
        r = self.model.predict(crop, imgsz=ev["imgsz"], conf=ev["conf"], verbose=False, device="cpu")[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                out.append(dict(cls=str(self.names[int(k)]), conf=float(c), box=b.tolist()))
        return out


def cycle_windows(cycles: pd.DataFrame, pad: float) -> list[tuple[int, float, float, float]]:
    """[(индекс цикла, tid, t0, t1)] — окно AT_MOUTH ± pad."""
    return [(i, int(r.tid), float(r.mouth_in - pad), float(r.mouth_out + pad)) for i, r in enumerate(cycles.itertuples())]


def run_evidence(video: str | Path, tr: Tracks, series: pd.DataFrame, cycles: pd.DataFrame, cfg: dict, detector_ids: list[str] | None = None,
                 windows: list[tuple[int, int, float, float]] | None = None, progress=None, save_crops: Path | None = None,
                 max_crops: int = 40) -> pd.DataFrame:
    """Детекции предмета на кропах кисть–рот для кадров вокруг циклов (или произвольных окон `windows`).

    Возвращает DataFrame: detector, cycle, tid, frame, t, cls, conf, x1,y1,x2,y2 (в координатах кадра), dist_mouth_s, dist_wrist_s,
    а также строки с cls=None для кадров, где детекций нет (нужны для подсчёта k из n).
    """
    ev = cfg["evidence"]
    ids = detector_ids or ev["detectors"]
    dets = [ObjectDetector(i, cfg) for i in ids]
    stride = int(tr.meta.get("stride", 1))
    wins = windows if windows is not None else cycle_windows(cycles, ev["crop_pad_frames_sec"])
    S = {tid: g.set_index("frame") for tid, g in series.groupby("tid")}
    rows, saved = [], 0
    for wi, (ci, tid, t0, t1) in enumerate(wins):
        s = S.get(tid)
        if s is None:
            continue
        for fr in iter_frames(video, max(0.0, t0), t1, only_idx=set(s.index)):   # именно кадры, обработанные позой: сетка не совпадает с шагом от начала окна
            sr = s.loc[fr.idx]
            if not np.isfinite(sr.mouth_x) or sr.hand not in (0, 1):
                continue
            wx, wy = (sr.handL_x, sr.handL_y) if sr.hand == 0 else (sr.handR_x, sr.handR_y)
            if not np.isfinite(wx):
                continue
            crop, (cx1, cy1, cx2, cy2) = hand_mouth_crop(fr.img, (sr.mouth_x, sr.mouth_y), (wx, wy), float(sr.s), cfg)
            if crop.size == 0:
                continue
            vis = cv2.resize(crop, (ev["imgsz"], ev["imgsz"]), interpolation=cv2.INTER_CUBIC) if save_crops is not None else None
            vscale = ev["imgsz"] / crop.shape[0]
            any_found = False
            for d in dets:
                found = d.detect(crop)
                any_found |= bool(found)
                if vis is not None:
                    for f in found:
                        b = f["box"]
                        p1, p2 = (int(b[0] * vscale), int(b[1] * vscale)), (int(b[2] * vscale), int(b[3] * vscale))
                        cv2.rectangle(vis, p1, p2, (0, 255, 255), 2)
                        cv2.putText(vis, f"{f['cls']} {f['conf']:.0%}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                if found:
                    for f in found:
                        b = f["box"]  # Ultralytics отдаёт рамки в пикселях исходного входа (кропа) -> прибавляем смещение кропа
                        bx1, by1, bx2, by2 = cx1 + b[0], cy1 + b[1], cx1 + b[2], cy1 + b[3]
                        bc = ((bx1 + bx2) / 2, (by1 + by2) / 2)
                        rows.append(dict(detector=d.id, cycle=ci, tid=tid, frame=fr.idx, t=fr.t, cls=f["cls"], conf=f["conf"],
                                         x1=bx1, y1=by1, x2=bx2, y2=by2,
                                         dist_mouth_s=float(np.hypot(bc[0] - sr.mouth_x, bc[1] - sr.mouth_y) / sr.s),
                                         dist_wrist_s=float(np.hypot(bc[0] - wx, bc[1] - wy) / sr.s),
                                         crop_x1=cx1, crop_y1=cy1, crop_x2=cx2, crop_y2=cy2))
                else:
                    rows.append(dict(detector=d.id, cycle=ci, tid=tid, frame=fr.idx, t=fr.t, cls=None, conf=0.0,
                                     crop_x1=cx1, crop_y1=cy1, crop_x2=cx2, crop_y2=cy2))
            if vis is not None and saved < max_crops and (any_found or fr.idx % (stride * 3) == 0):
                save_crops.mkdir(parents=True, exist_ok=True)
                cv2.putText(vis, f"c{ci} ID{tid} t={fr.t:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(save_crops / f"c{ci:03d}_tid{tid}_f{fr.idx}{'_HIT' if any_found else ''}.jpg"), vis)
                saved += 1
        if progress:
            progress(wi + 1, len(wins))
    return pd.DataFrame(rows)


def cycle_object_evidence(det: pd.DataFrame, cfg: dict, conf_th: float | None = None, max_dist_s: float = 0.8) -> pd.DataFrame:
    """Сводка по циклам: предмет «устойчиво виден» = детекция (conf >= порога, рядом с ртом/кистью) в >= k из n соседних кадров."""
    ev = cfg["events"]
    k, n = ev["object_k"], ev["object_n"]
    th = cfg["evidence"]["conf"] if conf_th is None else conf_th
    out = []
    if det.empty:
        return pd.DataFrame(columns=["detector", "cycle", "frames", "hit_frames", "max_conf", "has_object"])
    det = det.copy()
    for col in ("dist_mouth_s", "dist_wrist_s"):   # если ни один кадр окна не дал детекций, этих колонок нет (падало на реальных данных)
        if col not in det.columns:
            det[col] = np.nan
    for (dn, ci), g in det.groupby(["detector", "cycle"]):
        by_t = g.sort_values("t").groupby("frame", sort=True)
        flags, confs = [], []
        for _, gg in by_t:
            ok = gg[(gg.cls.notna()) & (gg.conf >= th) & ((gg.dist_mouth_s <= max_dist_s) | (gg.dist_wrist_s <= max_dist_s))]
            flags.append(len(ok) > 0)
            confs.append(float(ok.conf.max()) if len(ok) else 0.0)
        f = np.asarray(flags, int)
        has = bool(len(f) >= 1 and (np.convolve(f, np.ones(min(n, len(f)), int), "valid") >= min(k, len(f))).any()) if len(f) else False
        out.append(dict(detector=dn, cycle=int(ci), frames=len(f), hit_frames=int(f.sum()), max_conf=max(confs) if confs else 0.0, has_object=has))
    return pd.DataFrame(out)


def mouth_frames(video: str | Path, tr: Tracks, series: pd.DataFrame, tid: int, t0: float, t1: float, cfg: dict, n: int = 6, size: int = 224,
                 scale: float = 2.5) -> tuple[list[np.ndarray], list[float]]:
    """n кадров RGB size×size вокруг рта и активной кисти на равномерных моментах окна [t0, t1] — крупнее, чем «трубка» верха тела, где сигарета 5–10 px.

    Кроп — квадрат вокруг середины отрезка рот–кисть со стороной не меньше `scale` ширин плеч (лицо + рука), как `hand_mouth_crop` для детектора предмета.
    Если активная кисть в кадре неизвестна, кроп центрируется на рту. Пустой результат, если у трека в окне нет положения рта.
    """
    import copy

    cfg_v = copy.deepcopy(cfg)
    cfg_v["evidence"]["crop_scale"], cfg_v["evidence"]["crop_min_px"] = scale, 64
    s = series[series.tid == tid].drop_duplicates("frame").set_index("frame")
    if s.empty:
        return [], []
    frames = {fr.idx: (fr.t, fr.img) for fr in iter_frames(video, max(0.0, t0 - 0.05), t1 + 0.05, only_idx=set(s.index))}
    idxs = [i for i in sorted(frames) if np.isfinite(s.at[i, "mouth_x"]) and np.isfinite(s.at[i, "s"])]
    if not idxs:
        return [], []
    times = np.array([frames[i][0] for i in idxs])
    out, ts = [], []
    for t in np.linspace(t0, t1, n):
        i = idxs[int(np.abs(times - t).argmin())]
        sr = s.loc[i]
        wx, wy = ((sr.handL_x, sr.handL_y) if sr.hand == 0 else (sr.handR_x, sr.handR_y)) if sr.hand in (0, 1) else (np.nan, np.nan)
        if not np.isfinite(wx):
            wx, wy = sr.mouth_x, sr.mouth_y
        crop, _ = hand_mouth_crop(frames[i][1], (sr.mouth_x, sr.mouth_y), (wx, wy), float(sr.s), cfg_v)
        if crop.size == 0:
            return [], []
        out.append(cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
        ts.append(frames[i][0])
    return out, ts
