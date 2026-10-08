"""Лёгкая копия видео для просмотра и разметки в браузере: меньший размер, постоянная частота кадров, браузерный кодек.

Исходники 1080p весят десятки мегабайт на минуту и не вшиваются в страницу; копия 480p/12 к/с — около 1 МБ на минуту. Время сохраняется (фильтр `fps` перестраивает
метки по исходным), поэтому секунды в копии = секунды исходного видео и все интервалы/события совпадают. Копия лежит в `outputs/proxy/` и пересоздаётся при изменении исходника.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from .paths import OUTPUTS, portable, video_id

PROXY_DIR = OUTPUTS / "proxy"


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise FileNotFoundError("ffmpeg не найден: установите imageio-ffmpeg (pip install imageio-ffmpeg) или ffmpeg в PATH")
        return exe


def proxy_path(video: str | Path, height: int = 480, fps: float = 12.0, codec: str = "h264", root: Path | None = None) -> Path:
    v = Path(video)
    st = v.stat()
    key = hashlib.md5(f"{portable(v)}|{st.st_size}|{st.st_mtime_ns}|{height}|{fps}|{codec}".encode()).hexdigest()[:8]
    return (root or PROXY_DIR) / f"{video_id(v)}_{key}.{'webm' if codec == 'vp8' else 'mp4'}"


def ensure_proxy(video: str | Path, height: int = 480, fps: float = 12.0, codec: str = "h264", root: Path | None = None, timeout: float = 1800.0) -> Path:
    """Путь к копии (создаётся при первом обращении). `codec`: h264 (по умолчанию) | vp8 (webm: для браузеров без H.264)."""
    out = proxy_path(video, height, fps, codec, root)
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part" + out.suffix)
    vf = f"fps={fps:g},scale=-2:{int(height)}"
    enc = ["-c:v", "libvpx", "-b:v", "600k", "-deadline", "realtime", "-cpu-used", "8"] if codec == "vp8" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
                                                                                                           "-movflags", "+faststart"]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-an", "-vf", vf, *enc, str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg не смог сделать копию {Path(video).name}: {r.stderr.strip()[-300:]}")
    tmp.replace(out)
    return out
