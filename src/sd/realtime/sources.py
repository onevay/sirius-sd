"""Источники потока. Для MVP камера = папка с видео-фрагментами; имя папки задаёт район, номер камеры и время начала записи:

    <район>-<индекс>-<время начала>       gatchina-03-20261009T143000      gatchina_03_2026-10-09_14-30-00      tsarskoe-selo-12-1430 (только время — дата берётся из `base_date`)

Фрагменты внутри папки идут по имени (естественная сортировка: `chunk2` < `chunk10`), абсолютное время кадра = время начала + сумма длительностей предыдущих фрагментов + время в фрагменте.
Папку можно пополнять новыми фрагментами в любой момент (режим `--watch` подхватит), так имитируется живая камера. Настоящие потоки (RTSP) подключаются классом `CaptureSource`.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..paths import VIDEO_EXT

_START = r"\d{4}-?\d{2}-?\d{2}[T_ -]?\d{2}[-_:]?\d{2}(?:[-_:]?\d{2})?|\d{2}[-_:]\d{2}(?:[-_:]\d{2})?|\d{4}(?:\d{2})?"
STREAM_RE = re.compile(rf"^(?P<district>.+?)[-_](?P<index>\d{{1,4}})[-_](?P<start>{_START})$")


@dataclass(frozen=True)
class StreamDir:
    path: Path
    district: str
    index: str
    start: datetime
    parsed: bool = True            # False — имя папки не соответствует шаблону (район = имя папки, время = mtime первого файла)

    @property
    def camera_id(self) -> str:
        return f"{self.district}-{self.index}"


def parse_start(text: str, base_date: date | None = None) -> datetime | None:
    digits = re.sub(r"\D", "", text)
    base = base_date or date.today()
    try:
        if len(digits) in (12, 14):
            return datetime.strptime(digits, "%Y%m%d%H%M%S" if len(digits) == 14 else "%Y%m%d%H%M")
        if len(digits) in (4, 6):
            hh, mm, ss = int(digits[:2]), int(digits[2:4]), int(digits[4:6] or 0)
            return datetime(base.year, base.month, base.day, hh, mm, ss)
    except ValueError:
        return None
    return None


def parse_stream_dir(path: str | Path, base_date: date | None = None) -> StreamDir:
    p = Path(path)
    m = STREAM_RE.match(p.name)
    start = parse_start(m.group("start"), base_date) if m else None
    if m and start:
        return StreamDir(p, m.group("district"), m.group("index"), start, True)
    first = next(iter(list_chunks(p)), None)
    ts = datetime.fromtimestamp(first.stat().st_mtime) if first else datetime.now()
    return StreamDir(p, p.name, "0", ts.replace(microsecond=0), False)


def _natural(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_chunks(path: str | Path) -> list[Path]:
    p = Path(path)
    return sorted((f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT), key=lambda f: _natural(f.name)) if p.is_dir() else []


def scan_streams(root: str | Path, base_date: date | None = None) -> list[StreamDir]:
    """Подпапки `root`, в которых есть видео, как камеры (отсортированы по району, номеру, времени)."""
    r = Path(root)
    out = [parse_stream_dir(d, base_date) for d in sorted(r.iterdir()) if d.is_dir() and list_chunks(d)] if r.is_dir() else []
    return sorted(out, key=lambda s: (s.district, int(s.index) if s.index.isdigit() else 0, s.start))


def abs_time(stream: StreamDir, t_stream: float) -> datetime:
    return stream.start + timedelta(seconds=float(t_stream))


def iter_chunk_frames(path: str | Path, offset: float, process_fps: float = 10.0) -> Iterator[tuple[np.ndarray, float]]:
    """(кадр, секунды потока) одного фрагмента с шагом под `process_fps` (время по меткам контейнера)."""
    from ..video_io import choose_stride, iter_frames, probe

    info = probe(path)
    stride = choose_stride(info.fps, process_fps)
    for fr in iter_frames(path, 0.0, None, stride):
        yield fr.img, offset + fr.t


class Pacer:
    """Выдерживает скорость «как в жизни»: speed=1 — реальное время, 4 — в 4 раза быстрее, 0 — без пауз (как можно быстрее)."""

    def __init__(self, speed: float = 0.0, clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep):
        self.speed, self.clock, self.sleep = speed, clock, sleep
        self.t0: float | None = None
        self.s0 = 0.0

    def wait(self, t_stream: float) -> None:
        if self.speed <= 0:
            return
        if self.t0 is None:
            self.t0, self.s0 = self.clock(), t_stream
            return
        delay = (t_stream - self.s0) / self.speed - (self.clock() - self.t0)
        if delay > 0:
            self.sleep(delay)


class CaptureSource:
    """Живой источник (RTSP/веб-камера) через OpenCV с очередью из 2–4 кадров: когда обработка не успевает, старые кадры выбрасываются, а не копятся (задержка не растёт, §8.3)."""

    def __init__(self, url: str | int, process_fps: float = 10.0, maxsize: int = 3):
        import cv2

        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise OSError(f"не удалось открыть поток: {url}")
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self.stop_ev = threading.Event()
        self.step = 1.0 / process_fps
        self.t0 = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        last = -1e9
        while not self.stop_ev.is_set():
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            now = time.time() - self.t0
            if now - last < self.step:
                continue
            last = now
            if self.q.full():
                try:
                    self.q.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            self.q.put((img, now))

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        while not self.stop_ev.is_set():
            try:
                yield self.q.get(timeout=2.0)
            except queue.Empty:
                continue

    def close(self) -> None:
        self.stop_ev.set()
        self.cap.release()
