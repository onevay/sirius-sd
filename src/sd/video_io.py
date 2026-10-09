"""Чтение видео с таймкодами контейнера и запись H.264 с разметкой.

Время считаем по меткам контейнера (CAP_PROP_POS_MSEC), а не как «кадр / 25»: у архивов видеорегистраторов
fps бывает переменным и есть пропуски кадров (руководство, шаг 1).
"""
from __future__ import annotations

from . import _env  # noqa: F401

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import cv2
import numpy as np


class Frame(NamedTuple):
    idx: int          # номер кадра в исходном видео
    t: float          # секунды от начала клипа (по меткам контейнера)
    img: np.ndarray   # BGR, исходное разрешение


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    codec: str
    size_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def _fourcc(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")


def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Не удалось открыть видео: {path}")
    return cap


def probe(path: str | Path) -> VideoInfo:
    p = Path(path)
    cap = open_video(p)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0 or n > 10_000_000:          # wmv/ts/некоторые mkv не сообщают число кадров (0, -1 или мусор): считаем проходом без декодирования
            n = 0
            while cap.grab():
                n += 1
        return VideoInfo(
            path=str(p),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=round(fps, 3),
            n_frames=n,
            duration=round(n / fps, 2) if fps else 0.0,
            codec=_fourcc(cap),
            size_mb=round(p.stat().st_size / 1e6, 1),
        )
    finally:
        cap.release()


def choose_stride(src_fps: float, target_fps: float) -> int:
    """Шаг по кадрам, чтобы получить ~target_fps обработанных кадров/с."""
    return max(1, int(round(src_fps / max(target_fps, 1e-6))))


def iter_frames(path: str | Path, start: float = 0.0, end: float | None = None, stride: int = 1,
                max_frames: int | None = None, only_idx: set[int] | None = None) -> Iterator[Frame]:
    """Потоковое чтение. Пропускаемые кадры только декодируются (`grab`), без преобразования цвета.

    `stride` отсчитывается от начала окна, а сетка кадров, обработанных позой, привязана к началу запуска: у окна с произвольным началом они не совпадают
    (0 общих кадров из 27 — так терялись кадры детектора предмета). Чтобы взять именно обработанные кадры, передайте их номера в `only_idx` (stride тогда не нужен).
    """
    cap = open_video(path)
    try:
        if start and start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        idx0 = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        i, produced, last_t = 0, 0, -1.0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        while True:
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= last_t or (i == 0 and t == 0 and idx0 > 0):  # метка не вернулась / не монотонна -> по fps
                t = (idx0 + i) / fps
            last_t = t
            if end is not None and t > end:
                break
            if ((idx0 + i) in only_idx) if only_idx is not None else (i % stride == 0):
                ok, img = cap.retrieve()
                if not ok:
                    break
                yield Frame(idx0 + i, t, img)
                produced += 1
                if max_frames and produced >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def window_frame_index(path: str | Path, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """(номера кадров файла, их время) в окне [t0, t1] — только декодирование без преобразования цвета, по тем же правилам нумерации и монотонности времени, что у `iter_frames`
    (в mkv/wmv метки времени неточны, поэтому `round(t * fps)` не годится). Стоит доли секунды на окно, а не проход по всему файлу."""
    cap = open_video(path)
    idx, ts = [], []
    try:
        if t0 and t0 > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
        idx0 = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        i, last = 0, -1.0
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= last or (i == 0 and t == 0 and idx0 > 0):
                t = (idx0 + i) / fps
            last = t
            if t > t1:
                break
            idx.append(idx0 + i)
            ts.append(t)
            i += 1
    finally:
        cap.release()
    return np.asarray(idx, int), np.asarray(ts, float)


def prefetch(it: Iterator, size: int = 6) -> Iterator:
    """Декодирование в отдельном потоке: пока модель считает кадр, следующие уже читаются (cv2 отпускает GIL на grab/retrieve). Даёт ≈ 20–25 % времени на кадр при тех же результатах;
    исключение источника пробрасывается потребителю, ранний выход из цикла останавливает поток."""
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=size)
    stop = threading.Event()
    end = object()

    def work() -> None:
        try:
            for x in it:
                while not stop.is_set():
                    try:
                        q.put(x, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(end)
        except BaseException as e:   # noqa: BLE001 — передаём потребителю
            q.put(e)

    t = threading.Thread(target=work, daemon=True, name="frame-prefetch")
    t.start()
    try:
        while True:
            x = q.get()
            if x is end:
                return
            if isinstance(x, BaseException):
                raise x
            yield x
    finally:
        stop.set()


def read_frame_at(path: str | Path, t: float) -> Frame | None:
    for fr in iter_frames(path, start=max(0.0, t), stride=1, max_frames=1):
        return fr
    return None


def timestamp_stats(path: str | Path, n: int = 400) -> dict:
    """Проверка равномерности меток времени на первых n кадрах: признак VFR / пропусков кадров."""
    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        ts = []
        for _ in range(n):
            if not cap.grab():
                break
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    if len(ts) < 3:
        return dict(n=len(ts), fps_nominal=fps)
    d = np.diff(np.asarray(ts))
    return dict(
        n=len(ts), fps_nominal=round(fps, 3), fps_measured=round(float(1.0 / np.median(d)), 3) if np.median(d) > 0 else None,
        dt_median_ms=round(float(np.median(d)) * 1000, 2), dt_max_ms=round(float(d.max()) * 1000, 2),
        dt_std_ms=round(float(d.std()) * 1000, 2), gaps_gt_2x=int((d > 2.2 * np.median(d)).sum()),
        monotonic=bool((d > 0).all()),
    )


def resize_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / w
    return cv2.resize(img, (max_w, int(round(h * s))), interpolation=cv2.INTER_AREA)


class FFmpegWriter:
    """Пишет BGR-кадры в H.264 mp4 (yuv420p, +faststart) — проигрывается в браузере/Streamlit.

    `cv2.VideoWriter('mp4v')` даёт файлы, которые браузер не воспроизводит, поэтому кодируем через ffmpeg (imageio-ffmpeg).
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 24, preset: str = "veryfast"):
        import imageio_ffmpeg

        self.width, self.height = width - width % 2, height - height % 2
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}", "-r", f"{fps:.4f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]
        self.path = str(path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img: np.ndarray) -> None:
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self) -> "FFmpegWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
