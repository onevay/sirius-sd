"""«Лестница вычислений»: что даёт более тяжёлая поза на тех же клипах — для оценки выигрыша от более мощного ПК.

Для каждой конфигурации позы (модель, размер входа, уточнение точек) на клипах с метками затяжек считаются циклы рабочего автомата и сравниваются с метками жестов
(`docs/review/gesture_gt.csv`, затяжка = `smoke`): какая доля затяжек находится циклом (recall), сколько циклов приходится на не-затяжки и пустое место, насколько уверенно
видны запястья в момент затяжки, сколько времени на кадр. Поза каждой конфигурации кэшируется, как обычно (`outputs/runs`), и повторный запуск мгновенный.
Не меняет конвейер и не обучает ничего: это замер «потолка по позе»; классификатор и события здесь не участвуют.
"""
from __future__ import annotations

from . import _env  # noqa: F401

import time
from typing import Sequence

import numpy as np
import pandas as pd

from . import stages
from .config import load_config
from .paths import resolve_video, video_id

CONFIGS = {
    "n960": ["pose.weights=yolo26n-pose", "pose.imgsz=960", "pose.refine.method=rtmpose-s"],
    "s960": ["pose.weights=yolo26s-pose", "pose.imgsz=960", "pose.refine.method=rtmpose-s"],
    "m1280": ["pose.weights=yolo26m-pose", "pose.imgsz=1280", "pose.refine.method=rtmpose-s"],
    "n1280": ["pose.weights=yolo26n-pose", "pose.imgsz=1280", "pose.refine.method=rtmpose-s"],
    "n1600": ["pose.weights=yolo26n-pose", "pose.imgsz=1600", "pose.refine.method=rtmpose-s"],
    "n1920": ["pose.weights=yolo26n-pose", "pose.imgsz=1920", "pose.refine.method=rtmpose-s"],
    "s1920": ["pose.weights=yolo26s-pose", "pose.imgsz=1920", "pose.refine.method=rtmpose-s"],
    "s1600": ["pose.weights=yolo26s-pose", "pose.imgsz=1600", "pose.refine.method=rtmpose-s"],
    "m1600": ["pose.weights=yolo26m-pose", "pose.imgsz=1600", "pose.refine.method=rtmpose-s"],
    # NVIDIA (нужна сборка с CUDA): то же на видеокарте, RTMPose-m
    "gpu_s1600": ["pose.weights=yolo26s-pose", "pose.runtime=torch", "pose.device=cuda:0", "pose.imgsz=1600", "pose.refine.method=rtmpose-m", "pose.refine.device=cuda"],
    "gpu_m1600": ["pose.weights=yolo26m-pose", "pose.runtime=torch", "pose.device=cuda:0", "pose.imgsz=1600", "pose.refine.method=rtmpose-m", "pose.refine.device=cuda"],
    "gpu_m1920": ["pose.weights=yolo26m-pose", "pose.runtime=torch", "pose.device=cuda:0", "pose.imgsz=1920", "pose.refine.method=rtmpose-m", "pose.refine.device=cuda"],
    "x1280": ["pose.weights=yolo26x-pose", "pose.imgsz=1280", "pose.refine.method=rtmpose-s"],
    "x1280_rtmm": ["pose.weights=yolo26x-pose", "pose.imgsz=1280", "pose.refine.method=rtmpose-m"],
}


def match_cycles(cycles: pd.DataFrame, smoke_peaks: Sequence[float], tol: float = 1.5) -> dict:
    """Сопоставление циклов с затяжками по времени пика (любой трек): найденные затяжки, лишние циклы."""
    peaks = np.asarray(sorted(smoke_peaks), float)
    cp = cycles.peak_t.to_numpy(float) if len(cycles) else np.array([])
    found = int(sum(1 for t in peaks if len(cp) and np.min(np.abs(cp - t)) <= tol))
    extra = int(sum(1 for t in cp if not len(peaks) or np.min(np.abs(peaks - t)) > tol))
    return dict(smoke=int(len(peaks)), found=found, extra=extra, cycles=int(len(cp)))


def scaled_copy(video: Path, scale: float, out_dir: Path | None = None, pad: bool = True) -> Path:
    """Копия клипа, где содержимое кадра уменьшено в `scale` раз, а размер кадра сохранён (серые поля) — имитация более далёкой камеры: человек мельче, но кадр прежнего разрешения, как у
    настоящей дальней камеры (сеть получает тот же вход, что и на скрытом наборе). `pad=False` — просто уменьшенный файл (кадр мал, сеть его увеличивает: имитация оптимистична).
    Кодек H.264, crf 18, без звука; повторный вызов берёт готовый файл."""
    import subprocess

    import imageio_ffmpeg

    from .paths import OUTPUTS
    from .video_io import probe

    out_dir = out_dir or OUTPUTS / "ladder_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video.stem}_{'z' if pad else 'x'}{int(round(scale * 1000)):04d}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    info = probe(video)
    w, h = int(info.width) // 2 * 2, int(info.height) // 2 * 2
    sw, sh = max(int(w * scale) // 2 * 2, 2), max(int(h * scale) // 2 * 2, 2)
    vf = f"scale={sw}:{sh}" + (f",pad={w}:{h}:{(w - sw) // 2}:{(h - sh) // 2}:color=gray" if pad else "")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-vf", vf, "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg не смог уменьшить {video.name}: {r.stderr[:200]}")
    return out


def run(clips: Sequence[tuple], gestures: pd.DataFrame, names: Sequence[str], progress=None, scale: float = 1.0) -> pd.DataFrame:
    """clips: (video_id, начало, конец|None[, масштаб клипа]); names — ключи CONFIGS. Строка на (конфигурация, клип); ошибка конфигурации записывается в `error`, остальные идут дальше.
    `scale` < 1 — прогон на уменьшенных копиях (имитация дальней камеры); метки жестов и их время те же, высота людей — в `h_med`."""
    rows = []
    for ci, name in enumerate(names):
        cfg = load_config(None, CONFIGS[name])
        for vi, spec in enumerate(clips):
            vid, a, b = spec[:3]
            sc = float(spec[3]) if len(spec) > 3 and spec[3] else scale
            row = dict(config=name, clip=vid, scale=sc, window=f"{a:g}-{'' if b is None else format(b, 'g')}", error="")
            t0 = time.perf_counter()
            try:
                video = resolve_video(vid)
                src_id = video_id(video)
                if sc != 1.0:
                    video = scaled_copy(video, sc)
                tr, rd = stages.stage_pose(video, cfg, a, b)
                ser = stages.stage_features(tr, cfg, rd)
                cyc, _, _ = stages.stage_cycles(ser, cfg, rd)
                s = tr.summary(cfg["video"]["min_person_height_px"])
                ok = s[~s.ignore_small]
                cyc = cyc[cyc.tid.isin(ok.tid)]
                g = gestures[(gestures.video == src_id) & (gestures.label == "smoke") & (gestures.peak_t >= a) & ((gestures.peak_t <= b) if b is not None else True)]
                row.update(match_cycles(cyc, g.peak_t.tolist()), tracks=int(len(ok)), h_med=float(ok.h_med.median()) if len(ok) else np.nan, wrist_conf=float(ser.wrist_conf.mean()) if len(ser) else np.nan,
                           ms_per_frame=float(tr.meta["wall_sec"]) / max(int(tr.meta["frames_processed"]), 1) * 1000, frames=int(tr.meta["frames_processed"]))
            except Exception as e:   # noqa: BLE001 — тяжёлая модель может не поместиться в память: фиксируем и идём дальше
                row["error"] = f"{type(e).__name__}: {str(e)[:140]}"
            row["sec"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            if progress:
                progress(ci * len(clips) + vi + 1, len(names) * len(clips), f"{name} {vid}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """По конфигурациям: найдено затяжек из всех, лишних циклов, мс на кадр, достоверность запястий."""
    ok = df[df.error == ""]
    g = ok.groupby("config").agg(clips=("clip", "size"), smoke=("smoke", "sum"), found=("found", "sum"), extra=("extra", "sum"), cycles=("cycles", "sum"), tracks=("tracks", "sum"),
                                 wrist_conf=("wrist_conf", "mean"), ms_per_frame=("ms_per_frame", "mean"), frames=("frames", "sum"))
    g = g.reindex(list(dict.fromkeys(df.config)))          # конфигурации, упавшие целиком, остаются в таблице (с пропусками и числом ошибок)
    g.index.name = "config"
    g["recall"] = g.found / g.smoke.clip(lower=1)
    g["errors"] = df.groupby("config").error.apply(lambda s: int((s != "").sum()))
    return g.reset_index()
