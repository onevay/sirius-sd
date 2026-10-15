"""Видео для браузера. Формат выбирает КЛИЕНТ (`canPlayType`): H.264/mp4 везде, где есть кодек, иначе VP9/webm (Chromium без проприетарных кодеков, Linux-сборки).
Подходящий исходник отдаётся как есть; остальное один раз перекодируется в `outputs/web_cache` (ffmpeg из imageio-ffmpeg) в фоне, с процентом готовности."""
from __future__ import annotations

import hashlib
import subprocess
import threading
from pathlib import Path

from ..paths import OUTPUTS

CACHE = OUTPUTS / "web_cache"
FORMATS = {"h264": dict(ext=".mp4", mime="video/mp4", codecs={"avc1", "h264", "x264"}, direct_ext={".mp4", ".m4v", ".mov"},
                        args=["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
           "vp9": dict(ext=".webm", mime="video/webm", codecs={"vp80", "vp90", "vp8", "vp9", "av01"}, direct_ext={".webm"},
                       args=["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-pix_fmt", "yuv420p"])}
_lock = threading.Lock()
_jobs: dict[str, dict] = {}           # ключ → {pct, error}


def codec_of(path: Path) -> str:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            v = int(cap.get(cv2.CAP_PROP_FOURCC))
            return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()
        finally:
            cap.release()
    except Exception:
        return ""


def norm_fmt(fmt: str | None) -> str:
    return fmt if fmt in FORMATS else "h264"


def is_direct(path: Path, fmt: str) -> bool:
    f = FORMATS[fmt]
    return path.suffix.lower() in f["direct_ext"] and codec_of(path) in f["codecs"]


def cache_path(path: Path, fmt: str) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{fmt}".encode()).hexdigest()[:16]
    return CACHE / f"{key}{FORMATS[fmt]['ext']}"


def _transcode(src: Path, dst: Path, key: str, fmt: str, duration: float) -> None:
    job = _jobs[key]
    try:
        import imageio_ffmpeg

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale='min(960,iw)':-2", *FORMATS[fmt]["args"], "-an", "-progress", "pipe:1", str(tmp)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        for line in p.stdout:
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    job["pct"] = min(0.99, float(line.split("=")[1]) / 1e6 / duration) if duration else 0.0
                except ValueError:
                    pass
        err = p.stderr.read()
        if p.wait() != 0 or not tmp.exists():
            raise RuntimeError(err[-300:] or "ffmpeg завершился с ошибкой")
        tmp.replace(dst)
        job["pct"] = 1.0
    except Exception as e:
        job["error"] = str(e)


def playable(path: Path, fmt: str | None = None, duration: float = 0.0) -> dict:
    """{state: direct|ready|preparing|error, path, mime, pct, error}. Перекодирование стартует в фоне при первом обращении."""
    fmt = norm_fmt(fmt)
    f = FORMATS[fmt]
    if is_direct(path, fmt):
        return dict(state="direct", path=path, mime=f["mime"], pct=1.0, error=None, fmt=fmt)
    dst = cache_path(path, fmt)
    if dst.exists():
        return dict(state="ready", path=dst, mime=f["mime"], pct=1.0, error=None, fmt=fmt)
    key = dst.stem
    with _lock:
        job = _jobs.get(key)
        if job is None:
            job = _jobs[key] = dict(pct=0.0, error=None)
            threading.Thread(target=_transcode, args=(path, dst, key, fmt, duration), daemon=True).start()
        elif job["error"]:
            return dict(state="error", path=None, mime=None, pct=0.0, error=job["error"], fmt=fmt)
    return dict(state="preparing", path=None, mime=None, pct=job["pct"], error=None, fmt=fmt)
