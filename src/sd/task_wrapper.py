"""Процесс-обёртка фоновой задачи: `python -m sd.task_wrapper <каталог задачи> <аргументы sd…>`.

Запускает `python -m sd <аргументы>` с выводом в log.txt, пишет state.json (pid дочернего процесса, время окончания, код возврата) и по окончании продвигает очередь
(`tasks.pump`). task.json не трогает — его пишет только интерфейс (см. sd.tasks).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    from sd import tasks as T

    d, args = Path(argv[0]), argv[1:]
    with open(d / "log.txt", "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "sd", *args], stdout=log, stderr=subprocess.STDOUT, cwd=str(T.ROOT))
        T._write_json(d / "state.json", dict(child=p.pid, finished=None, returncode=None))
        rc = p.wait()
    state = T._read_json(d / "state.json")
    if state.get("returncode") != T.STOPPED_RC:          # «остановлена» уже записана интерфейсом — не затираем
        T._write_json(d / "state.json", dict(child=p.pid, finished=time.time(), returncode=rc))
    T.pump(d.parent)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
