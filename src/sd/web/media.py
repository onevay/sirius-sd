"""Видео для браузера. Браузер играет H.264 в mp4/webm; wmv, avi, mkv, H.265 и т. п. перекодируются в кэш (`outputs/web_cache`) через ffmpeg из imageio-ffmpeg — один раз на файл."""
from __future__ import annotations

import hashlib
import subprocess
import threading
from pathlib import Path

from ..paths import OUTPUTS

CACHE = OUTPUTS / "web_cache"
DIRECT_EXT = {".mp4", ".m4v", ".webm"}
DIRECT_CODECS = {"avc1", "h264", "x264", "vp80", "vp90", "vp8", "vp9", "av01"}
_lock = threading.Lock()
_running: dict[str, str] = {}          # ключ → "running" | "error: …"


def _codec(path: Path) -> str:
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


def is_direct(path: Path) -> bool:
    return path.suffix.lower() in DIRECT_EXT and _codec(path) in DIRECT_CODECS


def cache_path(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    return CACHE / f"{key}.mp4"


def _transcode(src: Path, dst: Path, key: str) -> None:
    try:
        import imageio_ffmpeg

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part.mp4")
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not tmp.exists():
            raise RuntimeError((r.stderr or b"").decode("utf-8", "replace")[-300:] or "ffmpeg завершился с ошибкой")
        tmp.replace(dst)
        _running.pop(key, None)
    except Exception as e:
        _running[key] = f"error: {e}"


def playable(path: Path) -> tuple[str, Path | None, str | None]:
    """('direct', path) | ('ready', кэш) | ('preparing', None) | ('error', None, текст). Перекодирование запускается в фоне при первом обращении."""
    if is_direct(path):
        return "direct", path, None
    dst = cache_path(path)
    if dst.exists():
        return "ready", dst, None
    key = dst.stem
    with _lock:
        st = _running.get(key)
        if st and st.startswith("error"):
            return "error", None, st[7:]
        if st is None:
            _running[key] = "running"
            threading.Thread(target=_transcode, args=(path, dst, key), daemon=True).start()
    return "preparing", None, None
