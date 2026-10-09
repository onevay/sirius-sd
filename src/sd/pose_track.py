"""Поза + трекинг людей: потоковый `PoseTracker.step(frame, t) -> list[Det]` (состояние хранится внутри).

Бэкенды:
  ultralytics — YOLO-pose + BoT-SORT/ByteTrack (`model.track(persist=True)`); рантайм torch (CPU) или OpenVINO (intel:cpu / intel:gpu = встроенная графика);
  rtmlib      — YOLOX + RTMPose через ONNX Runtime (Apache-2.0) + ByteTrack из Ultralytics (отдельным вызовом).
Опционально `pose.refine`: второй проход позы по кропу мелкого человека (top-down) — на кадрах 1080p+ люди 60–150 px,
и ключевые точки кистей/носа на полном кадре теряются.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .models import load_registry, local_weights
from .paths import MODELS, OUTPUTS


@dataclass
class Det:
    tid: int
    box: np.ndarray    # (4,) x1,y1,x2,y2 в пикселях исходного кадра
    score: float
    kp: np.ndarray     # (K, 3) x, y, conf


def enable_openvino_cache() -> None:
    """Кэш скомпилированных моделей OpenVINO (CACHE_DIR): без него ядра iGPU компилируются заново при каждом запуске (~20 с).

    Ultralytics создаёт свой `ov.Core()` и не передаёт CACHE_DIR, поэтому подмешиваем его в `compile_model` на уровне класса.
    """
    import openvino as ov

    if getattr(ov.Core, "_sd_cache_patched", False):
        return
    orig = ov.Core.compile_model
    cache = str(MODELS / "ov_cache")

    def patched(self, *args, **kw):
        cfg = dict(kw.get("config") or {})
        cfg.setdefault("CACHE_DIR", cache)
        kw["config"] = cfg
        return orig(self, *args, **kw)

    ov.Core.compile_model = patched
    ov.Core._sd_cache_patched = True


def imgsz_hw(long_side: int, frame_hw: tuple[int, int]) -> tuple[int, int]:
    """(h, w) кратные 32 для фиксированной формы экспорта; длинная сторона = long_side."""
    h, w = frame_hw
    if w >= h:
        return int(np.ceil(long_side * h / w / 32) * 32), int(long_side)
    return int(long_side), int(np.ceil(long_side * w / h / 32) * 32)


def _tracker_yaml(cfg: dict, proc_fps: float) -> Path:
    import ultralytics

    tr = cfg["tracking"]
    name = "botsort.yaml" if tr["tracker"] == "botsort" else "bytetrack.yaml"
    base = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "trackers" / name).read_text(encoding="utf-8"))
    base.update(track_buffer=max(5, int(round(tr["track_buffer_sec"] * proc_fps))), match_thresh=tr["match_thresh"],
                new_track_thresh=tr["new_track_thresh"], track_low_thresh=0.1, track_high_thresh=min(cfg["pose"]["conf"], 0.25))
    if tr["tracker"] == "botsort":
        base.update(gmc_method=tr["gmc_method"], with_reid=False)
    d = OUTPUTS / ".tmp"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"tracker_{hashlib.md5(yaml.dump(base).encode()).hexdigest()[:8]}.yaml"
    p.write_text(yaml.dump(base), encoding="utf-8")
    return p


def ensure_openvino(weights: Path, hw: tuple[int, int]) -> Path:
    """Экспорт YOLO .pt -> OpenVINO IR фиксированной формы (один раз), возвращает каталог модели."""
    # Ultralytics определяет формат по имени каталога: оно должно оканчиваться на `_openvino_model`
    dst = MODELS / "ov" / f"{weights.stem}_{hw[0]}x{hw[1]}_openvino_model"
    if list(dst.glob("*.xml")):
        return dst
    from ultralytics import YOLO

    out = YOLO(str(weights), task="pose").export(format="openvino", imgsz=list(hw), half=False, dynamic=False, batch=1, verbose=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(out), str(dst))
    return dst


class _Dets:
    """Минимальный аналог `Boxes` для подачи детекций в трекер Ultralytics отдельным вызовом (rtmlib-бэкенд)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy, self.conf = xyxy.astype(np.float32), conf.astype(np.float32)
        self.cls = np.zeros(len(conf), np.float32)
        w, h = self.xyxy[:, 2] - self.xyxy[:, 0], self.xyxy[:, 3] - self.xyxy[:, 1]
        self.xywh = np.stack([self.xyxy[:, 0] + w / 2, self.xyxy[:, 1] + h / 2, w, h], 1) if len(conf) else np.zeros((0, 4), np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Dets":
        return _Dets(self.xyxy[idx], self.conf[idx])


def _rtm_backend(refine: dict) -> tuple[str, str]:
    """(backend, device) для RTMPose в rtmlib: onnxruntime (cpu | cuda, если есть CUDAExecutionProvider) или openvino (cpu | GPU)."""
    if str(refine.get("backend", "onnxruntime")).lower() == "openvino":
        dev = str(refine.get("device", "cpu"))
        return "openvino", "GPU" if dev.lower() in ("gpu", "intel:gpu") else "cpu"
    return "onnxruntime", _ort_device(refine)


def _ort_device(refine: dict) -> str:
    """Устройство onnxruntime для RTMPose: cuda только если в сборке есть CUDAExecutionProvider, иначе cpu."""
    want = str(refine.get("device", "cpu")).lower()
    if want.startswith("cuda"):
        import onnxruntime as ort

        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    return "cpu"


def _hand_near_face(kp: np.ndarray, box: np.ndarray, gate: float) -> bool:
    """По сырым точкам YOLO: ближайшая к носу кисть ближе `gate` ширин плеч (нет плеч — четверть высоты рамки). Нет видимых носа/кистей — считаем «рядом» (не рискуем пропустить)."""
    c = 0.2
    if kp[0, 2] < c or max(kp[9, 2], kp[10, 2]) < c:
        return True
    s = float(np.hypot(*(kp[5, :2] - kp[6, :2]))) if min(kp[5, 2], kp[6, 2]) >= c else float(box[3] - box[1]) / 4.0
    s = max(s, 1.0)
    dist = min(float(np.hypot(*(kp[w, :2] - kp[0, :2]))) for w in (9, 10) if kp[w, 2] >= c)
    return dist / s < gate


class PoseTracker:
    def __init__(self, cfg: dict, proc_fps: float, frame_hw: tuple[int, int]):
        self.cfg, self.proc_fps, self.frame_hw = cfg, proc_fps, frame_hw
        pc = cfg["pose"]
        self.backend = pc["backend"]
        self.timing: dict[str, list[float]] = {"infer": [], "refine": [], "total": []}
        self._ref_n, self._ref_until = 0, {}          # счётчик кадров и «уточнять до кадра N» по человеку (gate_s)
        self.refine_model = None
        self.rtm = None   # RTMPose (rtmlib) как уточнитель ключевых точек по рамкам YOLO
        self.last_raw = np.zeros((0, 5), np.float32)   # сырые детекции последнего кадра [x1,y1,x2,y2,conf] — до трекера
        if self.backend == "ultralytics":
            self._init_ultralytics()
        elif self.backend == "rtmlib":
            self._init_rtmlib()
        else:
            raise ValueError(f"pose.backend: {self.backend}")

    # ------------------------------------------------------------------ ultralytics
    def _load_yolo(self, weights_id: str, imgsz_long: int):
        from ultralytics import YOLO

        pc = self.cfg["pose"]
        reg = load_registry()
        w = local_weights(weights_id)
        if pc["runtime"] == "openvino":
            hw = imgsz_hw(imgsz_long, self.frame_hw)
            path = ensure_openvino(w, hw)
            return YOLO(str(path), task="pose"), list(hw), pc["device"] if pc["device"].startswith("intel") else "intel:cpu"
        return YOLO(str(w), task="pose"), imgsz_long, pc["device"]

    def _init_ultralytics(self) -> None:
        from ultralytics import settings

        settings.update({"sync": False})  # телеметрия выключена: обработка локальная
        pc = self.cfg["pose"]
        if pc["runtime"] == "openvino":
            enable_openvino_cache()
        self.model, self.imgsz, self.device = self._load_yolo(pc["weights"], pc["imgsz"])
        self.tracker_yaml = _tracker_yaml(self.cfg, self.proc_fps)
        self.tracker = self._make_tracker()
        if pc["refine"]["enabled"]:
            method = str(pc["refine"].get("method", "yolo"))
            if method.startswith("rtmpose"):
                from rtmlib import Body

                rb, rdev = _rtm_backend(pc["refine"])
                self.rtm = Body(mode="lightweight" if method.endswith("s") else "balanced", backend=rb, device=rdev).pose_model
            else:
                self.refine_model, self.refine_imgsz, self.refine_device = self._load_yolo_crop(pc["weights"], pc["refine"]["imgsz"])

    def _load_yolo_crop(self, weights_id: str, imgsz: int):
        from ultralytics import YOLO

        pc = self.cfg["pose"]
        w = local_weights(weights_id)
        if pc["runtime"] == "openvino":
            path = ensure_openvino(w, (imgsz, imgsz))
            return YOLO(str(path), task="pose"), [imgsz, imgsz], pc["device"] if pc["device"].startswith("intel") else "intel:cpu"
        return YOLO(str(w), task="pose"), imgsz, pc["device"]

    def _make_tracker(self):
        """Трекер Ultralytics (BoT-SORT / ByteTrack) отдельным объектом: сырые детекции остаются доступными и их можно переиграть."""
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace

        cfg = IterableSimpleNamespace(**YAML.load(str(self.tracker_yaml)))
        cfg.device = "cpu"
        return TRACKER_MAP[cfg.tracker_type](args=cfg)

    def _step_ultralytics(self, img: np.ndarray) -> list[Det]:
        pc = self.cfg["pose"]
        # детектор отдаёт и слабые детекции (conf >= conf_low): ByteTrack использует их во втором проходе ассоциации
        r = self.model.predict(img, imgsz=self.imgsz, conf=pc.get("conf_low", 0.1), iou=pc["iou"], device=self.device, classes=[0],
                               verbose=False, max_det=pc["max_det"])[0]
        self.timing["infer"].append(float(r.speed.get("inference", 0.0)))
        if r.boxes is None or len(r.boxes) == 0:
            self.last_raw = np.zeros((0, 5), np.float32)
            self.tracker.update(_Dets(np.zeros((0, 4)), np.zeros(0)), img)
            return []
        xyxy, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        self.last_raw = np.concatenate([xyxy, conf[:, None]], 1).astype(np.float32)
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2], np.float32)
        tracks = self.tracker.update(r.boxes.cpu().numpy(), img)   # строки [x1,y1,x2,y2,id,score,cls,idx]
        return [Det(int(t[4]), t[:4].astype(np.float32), float(t[5]), np.concatenate([kxy[int(t[-1])], kc[int(t[-1])][:, None]], 1))
                for t in tracks]

    def _refine(self, img: np.ndarray, dets: list[Det]) -> list[Det]:
        """Уточнение ключевых точек. rtmpose-*: RTMPose (top-down) по рамкам YOLO для всех людей >= 20 px;
        yolo: второй проход YOLO-pose по кропу мелкого человека (рамка + pad -> pose-модель -> координаты обратно в кадр)."""
        if self.rtm is not None:
            min_h = max(20.0, float(self.cfg["pose"]["refine"].get("min_height_px", 20)))
            keep = [i for i, d in enumerate(dets) if d.box[3] - d.box[1] >= min_h]
            gate = float(self.cfg["pose"]["refine"].get("gate_s", 0.0))
            if gate > 0:
                self._ref_n += 1
                hold = int(np.ceil(float(self.cfg["pose"]["refine"].get("gate_hold_sec", 1.0)) * self.proc_fps))
                for i in keep:
                    if _hand_near_face(dets[i].kp, dets[i].box, gate):
                        self._ref_until[dets[i].tid] = self._ref_n + hold
                keep = [i for i in keep if self._ref_until.get(dets[i].tid, -1) >= self._ref_n]
            if not keep:
                return dets
            k, s = self.rtm(img, bboxes=np.stack([dets[i].box for i in keep]))
            out = list(dets)
            for j, i in enumerate(keep):
                out[i] = Det(dets[i].tid, dets[i].box, dets[i].score, np.concatenate([k[j], s[j][:, None]], 1).astype(np.float32))
            return out
        rc = self.cfg["pose"]["refine"]
        H, W = img.shape[:2]
        out = []
        for d in dets:
            x1, y1, x2, y2 = d.box
            h = y2 - y1
            if h > rc["max_height_px"] or h < 20:
                out.append(d)
                continue
            pad = rc["pad"]
            cx1, cy1 = max(0, int(x1 - pad * (x2 - x1))), max(0, int(y1 - pad * h))
            cx2, cy2 = min(W, int(x2 + pad * (x2 - x1))), min(H, int(y2 + pad * h))
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                out.append(d)
                continue
            r = self.refine_model.predict(crop, imgsz=self.refine_imgsz, conf=0.1, device=self.refine_device, classes=[0],
                                          verbose=False, max_det=5)[0]
            if r.boxes is None or len(r.boxes) == 0:
                out.append(d)
                continue
            bx = r.boxes.xyxy.cpu().numpy()
            centre = np.array([(x1 + x2) / 2 - cx1, (y1 + y2) / 2 - cy1])
            j = int(np.argmin(np.hypot((bx[:, 0] + bx[:, 2]) / 2 - centre[0], (bx[:, 1] + bx[:, 3]) / 2 - centre[1])))
            kxy = r.keypoints.xy.cpu().numpy()[j] + np.array([cx1, cy1], np.float32)
            kc = r.keypoints.conf.cpu().numpy()[j] if r.keypoints.conf is not None else np.ones(len(kxy), np.float32)
            out.append(Det(d.tid, d.box, d.score, np.concatenate([kxy, kc[:, None]], 1)))
        return out

    # ------------------------------------------------------------------ rtmlib
    def _init_rtmlib(self) -> None:
        from rtmlib import Body
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace as NS

        pc, tr = self.cfg["pose"], self.cfg["tracking"]
        spec = load_registry()[pc["weights"]]
        mode = (spec.rtmlib or {}).get("mode", "balanced")
        self.body = Body(mode=mode, backend="onnxruntime", device="cpu")
        args = NS(track_high_thresh=min(pc["conf"], 0.25), track_low_thresh=0.1, new_track_thresh=tr["new_track_thresh"],
                  track_buffer=max(5, int(round(tr["track_buffer_sec"] * self.proc_fps))), match_thresh=tr["match_thresh"], fuse_score=True)
        self._tracker_cls, self._tracker_args = BYTETracker, args
        self.byte = BYTETracker(args)

    def _step_rtmlib(self, img: np.ndarray) -> list[Det]:
        t0 = time.perf_counter()
        bboxes = self.body.det_model(img)                      # (N,4) люди
        if len(bboxes) == 0:
            self.byte.update(_Dets(np.zeros((0, 4)), np.zeros(0)), img)
            return []
        kps, sc = self.body.pose_model(img, bboxes=bboxes)     # (N,K,2), (N,K)
        self.timing["infer"].append((time.perf_counter() - t0) * 1000)
        conf = np.full(len(bboxes), 0.9, np.float32)           # YOLOX в rtmlib не отдаёт score — фиксируем
        self.last_raw = np.concatenate([np.asarray(bboxes)[:, :4], conf[:, None]], 1).astype(np.float32)
        tracks = self.byte.update(_Dets(np.asarray(bboxes), conf), img)
        out = []
        for row in tracks:
            j = int(row[-1])
            out.append(Det(int(row[4]), row[:4].astype(np.float32), float(row[5]), np.concatenate([kps[j], sc[j][:, None]], 1)))
        return out

    # ------------------------------------------------------------------ api
    def step(self, img: np.ndarray, t: float) -> list[Det]:
        t0 = time.perf_counter()
        dets = self._step_ultralytics(img) if self.backend == "ultralytics" else self._step_rtmlib(img)
        if (self.refine_model is not None or self.rtm is not None) and dets:
            t1 = time.perf_counter()
            dets = self._refine(img, dets)
            self.timing["refine"].append((time.perf_counter() - t1) * 1000)
        self.timing["total"].append((time.perf_counter() - t0) * 1000)
        return dets

    def describe(self) -> dict:
        pc = self.cfg["pose"]
        return dict(backend=self.backend, weights=pc["weights"], runtime=pc["runtime"], device=pc["device"], imgsz=pc["imgsz"],
                    tracker=self.cfg["tracking"]["tracker"], refine=pc["refine"]["enabled"], refine_method=pc["refine"].get("method", "yolo"))
