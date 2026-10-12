"""Фоновые задачи интерфейсов (Dev Studio и веб): длинная операция проекта — отдельный процесс `python -m sd <команда>` с журналом на диске и очередью.

Зачем: тяжёлое (оценка, признаки, обучение, диагностика, лестница) не должно выполняться в потоке Streamlit — закрытая вкладка или перезапуск интерфейса
теряли работу, а сессия блокировалась. Процесс не привязан к вкладке, состояние читается с диска (`outputs/tasks/<id>/`):

    task.json   — что запущено: тип, заголовок, аргументы, значения формы, время постановки, «тяжёлая» ли, pid обёртки. Пишет ТОЛЬКО интерфейс (под замком очереди).
    state.json  — чем закончилось: pid дочернего `sd`, время окончания, код возврата. Пишет ТОЛЬКО процесс-обёртка (раньше оба писали task.json — гонка, статус терялся).
    log.txt     — вывод (stdout+stderr).

Очередь. Тяжёлые задачи (поза, признаки, обучение) на ноутбуке с 8 ГБ ОЗУ нельзя запускать параллельно: вторая встаёт «в очередь» и стартует, когда закончится
первая. Отдельного демона нет: очередь продвигает `pump()`, который вызывают обёртка по окончании задачи и любой просмотр списка задач (под файловым замком).

Прогресс. Команды печатают строки вида `[i/n] сообщение` — последняя такая строка журнала даёт долю и подпись (`progress()`).

Безопасность. Только типы из `CATALOG` (белый список), аргументы собираются списком по описанию полей (shell=False); остановить можно лишь задачу из этого каталога
задач (pid берётся из её файлов).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import OUTPUTS, ROOT

TASKS_DIR = OUTPUTS / "tasks"
STATES = ("в очереди", "идёт", "готово", "ошибка", "остановлена", "прервана")
STOPPED_RC = -15


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str = "text"          # text | int | float | bool | choice | lines
    default: Any = ""
    flag: str | None = None     # None — позиционный аргумент
    choices: tuple = ()
    help: str = ""


@dataclass(frozen=True)
class TaskType:
    id: str
    title: str
    command: tuple[str, ...]
    description: str
    fields: tuple[Field, ...] = field(default_factory=tuple)
    needs_net: bool = False
    heavy: bool = True          # тяжёлые задачи выполняются по одной (очередь)
    group: str = "прочее"       # раздел интерфейса: оценка | данные | признаки | обучение | выпуск | окружение


RUN_FIELD = Field("run", "Таблица циклов из прогона оценки (id из журнала; пусто — последние запуски)", "text", "", "--run",
                  help="циклы тех клипов и той конфигурации, что в выбранном прогоне `sd eval` — воспроизводимо, в отличие от «последнего, что считалось»")

CATALOG: dict[str, TaskType] = {t.id: t for t in (
    TaskType("doctor", "Проверка окружения", ("doctor",), "Версии, устройства (CPU / iGPU / CUDA), веса, данные, свободная память и место. Первое, что запускают на новой машине.",
             heavy=False, group="окружение"),
    TaskType("models_get", "Скачать веса моделей", ("models", "get"), "Скачивает веса по id из реестра (нужна сеть). Community-.pt сканируются перед загрузкой.", (
        Field("ids", "Идентификаторы через пробел", "text", "yolo26n-pose yolo26s-pose smoking_yolo11m_beehzod smoking_yolo26s_basant18", None, help="`sd models list` — список"),),
        needs_net=True, heavy=False, group="окружение"),
    TaskType("gt_from_gestures", "Эталон событий из меток жестов", ("gt-from-gestures",),
             "Перестраивает labels/events_gt.csv по меткам жестов. Строки разметчика `assistant` заменяются; прежний файл сохраняется версией (labels/history).", (
                 Field("labeler", "Разметчик", "text", "assistant", "--labeler"),), group="данные"),
    TaskType("enrich", "Признаки циклов: предмет, фото, VLM", ("enrich",),
             "Считает и кэширует (outputs/enrich) признаки предмета, фото-модели и VLM для циклов. Повтор берёт готовое из кэша; прерванный прогон продолжается.", (
                 RUN_FIELD,
                 Field("objects", "Детекторы предмета (через запятую; пусто — не считать)", "text", "smoking_yolo11m_beehzod", "--objects"),
                 Field("photo_bundle", "Фото-модель (zero-shot CLIP и photo_*)", "text", "models/photo/photo_v2", "--photo-bundle"),
                 Field("vlm", "VLM (тег Ollama; пусто — без VLM, ≈ 11 с на цикл)", "text", "", "--vlm"),
                 Field("clips", "Только клипы (video_id через запятую; пусто — все)", "text", "", "--clips")), group="признаки"),
    TaskType("oof_eval", "Честная оценка наборов признаков", ("oof-eval",),
             "Оценка циклов моделью, не видевшей видео; F1 события по регламенту; вложенный подбор порогов. Результат — outputs/analysis/oof_study<метка>.csv.", (
                 RUN_FIELD,
                 Field("sets", "Наборы признаков (через запятую)", "text", "obj_hold_zsd", "--sets"),
                 Field("kinds", "Члены ансамбля", "text", "lr,gb", "--kinds"),
                 Field("repeats", "Перемешиваний", "int", 3, "--repeats"),
                 Field("by_scene", "Фолды по сценам", "bool", False, "--by-scene"),
                 Field("tag", "Метка результата", "text", "_ui", "--tag")), group="обучение"),
    TaskType("clf_diagnose", "Диагностика классификатора по спецификации", ("clf", "diagnose"),
             "oof против train, контроль перестановкой меток, кривая обучения, важность признаков. Результат — outputs/classifier/<время>_<спецификация>/.", (
                 Field("spec", "Спецификация (configs/classifiers/<имя>)", "text", "", "--spec"),
                 RUN_FIELD,
                 Field("perm", "Перестановок меток", "int", 3, "--perm"),
                 Field("importance", "Важность признаков (долго)", "bool", True, "--importance")), group="обучение"),
    TaskType("clf_train", "Обучить пакет по спецификации", ("clf", "train"),
             "Пакет без pickle в models/cycle/<имя> с манифестом: спецификация, источник таблицы, видео обучения (для защиты от утечки), отпечаток меток.", (
                 Field("spec", "Спецификация (configs/classifiers/<имя>)", "text", "", "--spec"),
                 Field("bundle", "Имя пакета", "text", "cycle_new", "--bundle"),
                 RUN_FIELD,
                 Field("overwrite", "Перезаписать, если пакет есть", "bool", False, "--overwrite"),
                 Field("attach", "Подключить к профилю (имя; пусто — не подключать)", "text", "", "--attach")), group="обучение"),
    TaskType("eval", "Оценка конфигурации на папках (sd eval)", ("eval",), "Event F1 по эталону; результат — в журнале экспериментов и на вкладке «Валидация».", (
        Field("dirs", "Папки с видео (по одной на строку)", "lines", "data/курение\ndata/лжекурение", "--dir"),
        Field("profile", "Профиль", "text", "mvp", "--profile"),
        Field("classifier", "Классификатор цикла (пусто — как в профиле)", "text", "", "--classifier"),
        Field("sets", "Правки профиля ключ=значение (по одной на строку)", "lines", "", "--set"),
        Field("mode", "Эталон", "choice", "events", "--mode", choices=("events", "clips")),
        Field("role", "Набор", "choice", "validation", "--role", choices=("validation", "hidden")),
        Field("policy", "Порог", "choice", "fixed", "--policy", choices=("fixed", "plateau", "best")),
        Field("target_f1", "Целевая F1", "float", 0.8, "--target-f1"),
        Field("max_sec", "Только первые N секунд клипа (0 — целиком)", "float", 0.0, "--max-sec"),
        Field("clips", "Точечно: клипы (подстроки id через запятую; пусто — все)", "text", "", "--clips"),
        Field("no_cache", "Пересчитать без кэша", "bool", False, "--no-cache")), group="оценка"),
    TaskType("variants", "Варианты профиля «что если»", ("variants",), "Серия оценок: базовый профиль и его варианты (строки «имя / параметр / значение») на одних папках.", (
        Field("dirs", "Папки с видео (по одной на строку)", "lines", "data/курение\ndata/лжекурение", "--dir"),
        Field("base", "Базовый профиль", "text", "mvp", "--base"),
        Field("rows", "Файл вариантов (JSON)", "text", "", "--rows"),
        Field("policy", "Порог", "choice", "fixed", "--policy", choices=("fixed", "plateau"))), group="оценка"),
    TaskType("report", "PDF-отчёт прогона оценки", ("report",), "Конфигурация и отпечатки, данные, метрики с интервалом, порог, ошибки, клипы, производительность, сравнение с базовым.", (
        Field("run", "Прогон (id из журнала)", "text", "", None),
        Field("baseline", "Базовый прогон для сравнения (необязательно)", "text", "", "--baseline")), heavy=False, group="оценка"),
    TaskType("ladder", "Лестница вычислений (поза)", ("ladder",), "Сколько затяжек находит автомат при разных моделях и размерах входа; масштаб < 1 имитирует более далёкую камеру.", (
        Field("configs", "Конфигурации (через запятую)", "text", "n960,n1280", "--configs", help="n960 s960 n1280 n1600 n1920 s1600 m1600 m1280 x1280 gpu_s1600 gpu_m1600 gpu_m1920"),
        Field("clips", "Клипы video_id[:a-b][@масштаб] через запятую (пусто — по умолчанию)", "text", "", "--clips"),
        Field("scale", "Масштаб содержимого кадра (1 — как есть)", "float", 1.0, "--scale"),
        Field("tag", "Метка результата", "text", "_ui", "--tag")), group="окружение"),
    TaskType("release", "Выпуск: ворота качества → решатель", ("release",),
             "Проверяет ворота (прогон без сбоев, F1 не ниже порога, нет утечки, профиль совпадает с оценённым) и упаковывает решатель с манифестом выпуска.", (
                 Field("profile", "Профиль", "text", "mvp", "--profile"),
                 Field("run", "Прогон оценки, подтверждающий качество", "text", "", "--run"),
                 Field("name", "Имя выпуска", "text", "", "--name"),
                 Field("min_f1", "Минимальная F1", "float", 0.7, "--min-f1"),
                 Field("weights", "Положить веса", "bool", False, "--with-weights"),
                 Field("force", "Выпустить, даже если ворота не пройдены (записывается в манифест)", "bool", False, "--force")), heavy=False, group="выпуск"),
    TaskType("train_bundle", "Обучить пакет по набору (sd train-bundle, старый путь)", ("train-bundle",),
             "Командная версия без спецификации и диагностики. Для новых моделей используйте «Обучить пакет по спецификации».", (
                 Field("name", "Имя пакета", "text", "cycle_new", None),
                 Field("set", "Набор признаков", "text", "obj_hold_zsd", "--set"),
                 Field("enriched", "С признаками enrich", "bool", True, "--enriched"),
                 Field("calibrate", "Изотоническая калибровка", "bool", False, "--calibrate"),
                 Field("kinds", "Члены ансамбля", "text", "lr,gb,nn", "--kinds")), group="обучение"),
)}

BOOL_PAIRS = {"--calibrate": "--no-calibrate", "--importance": "--no-importance"}   # флаги вида --x/--no-x: передаём явно и при False
KEEP_EMPTY = {"objects", "vlm"}                                                       # пустое значение осмысленно («не считать») и передаётся явно


def _args(t: TaskType, values: dict[str, Any]) -> list[str]:
    """Аргументы командной строки из значений формы по описанию полей (ничего, кроме описанных полей, не принимается)."""
    out = list(t.command)
    pos: list[str] = []
    for f in t.fields:
        v = values.get(f.key, f.default)
        if f.kind == "bool":
            if f.flag in BOOL_PAIRS:
                out.append(f.flag if v else BOOL_PAIRS[f.flag])
            elif f.flag and v:
                out.append(f.flag)
            continue
        if f.kind in ("int", "float") and f.key in ("max_sec",) and not v:
            continue
        if v is None or str(v).strip() == "":
            if f.flag and f.key in KEEP_EMPTY:
                out += [f.flag, ""]
            continue
        if f.kind == "lines":
            for line in str(v).splitlines():
                if line.strip():
                    out += [f.flag, line.strip()]
        elif f.flag:
            out += [f.flag, str(v).strip()]
        else:
            pos += str(v).split()
    return out + pos


def command(task_id: str, values: dict[str, Any]) -> list[str]:
    if task_id not in CATALOG:
        raise KeyError(f"неизвестная задача {task_id!r}")
    return _args(CATALOG[task_id], values)


# ---------------------------------------------------------------------------------------------- файлы задачи
def _write_json(f: Path, data: dict) -> None:
    """Атомарная запись (временный файл + замена): читатель никогда не видит наполовину записанный JSON."""
    fd, tmp = tempfile.mkstemp(dir=str(f.parent), suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        for attempt in range(8):
            try:
                os.replace(tmp, f)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_json(f: Path) -> dict:
    for _ in range(5):
        try:
            return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05)
    return {}


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            # tasklist пишет в кодовой странице консоли (cp866/cp1251): text=True ломает чтение потока → читаем байты и декодируем с заменой
            r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"], capture_output=True)
            out = (r.stdout or b"").decode("utf-8", "replace")
            return any(len(p) > 1 and p[1].strip('"') == str(int(pid)) for p in (ln.split('","') for ln in out.splitlines()))
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


@contextmanager
def _queue_lock(root: Path, timeout: float = 10.0):
    """Замок очереди (O_EXCL-файл): два интерфейса или обёртка не запустят одну задачу дважды. Замок старше 30 с считается брошенным."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".queue.lock"
    t0 = time.time()
    while True:
        try:
            os.close(os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 30:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout:
                raise TimeoutError("очередь задач занята другим процессом")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _launch(d: Path, meta: dict) -> None:
    """Запуск процесса-обёртки (sd.task_wrapper): он запускает `python -m sd <args>`, пишет state.json и продвигает очередь."""
    # Windows: скрытая консоль (CREATE_NO_WINDOW), а не DETACHED_PROCESS. Процесс без консоли запускает консольных потомков (`python -m sd` → перезапуск с окружением)
    # с новой консолью каждый, и рабочий процесс зависал на выходе: результат записан, а задача «идёт» вечно. Отдельная группа — Ctrl+C интерфейса её не трогает.
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([sys.executable, "-m", "sd.task_wrapper", str(d), *meta["args"]], cwd=str(ROOT), env=env, creationflags=flags, start_new_session=(os.name != "nt"),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    meta.update(pid=p.pid, started=time.time(), queued=False)
    _write_json(d / "task.json", meta)


def start(task_id: str, values: dict[str, Any], root: Path | None = None, queue: bool = True) -> str:
    """Ставит задачу и возвращает её id. Лёгкая задача стартует сразу; тяжёлая — сразу, если не идёт другая тяжёлая, иначе ждёт в очереди (`queue=False` — без очереди)."""
    args = command(task_id, values)
    root = root or TASKS_DIR
    root.mkdir(parents=True, exist_ok=True)
    t = CATALOG[task_id]
    base = f"{time.strftime('%Y%m%d_%H%M%S')}_{task_id}"
    tid, k = base, 1
    while (root / tid).exists():          # две задачи в одну секунду
        k += 1
        tid = f"{base}_{k}"
    d = root / tid
    d.mkdir(parents=True)
    meta = dict(id=tid, task=task_id, title=t.title, args=args, values={k: v for k, v in values.items() if k in {f.key for f in t.fields}}, created=time.time(), started=None,
                heavy=t.heavy, pid=None, queued=True)
    with _queue_lock(root):
        _write_json(d / "task.json", meta)
        busy = t.heavy and queue and any(s["heavy"] and s["state"] in ("идёт", "в очереди") and s["id"] != tid for s in _scan(root))
        if not busy:
            _launch(d, meta)
    return tid


def pump(root: Path | None = None) -> str | None:
    """Продвинуть очередь: если ни одна тяжёлая задача не идёт — запустить самую раннюю из ожидающих. Возвращает id запущенной."""
    root = root or TASKS_DIR
    if not root.exists():
        return None
    try:
        with _queue_lock(root, timeout=2.0):
            st = _scan(root)
            if any(s["heavy"] and s["state"] == "идёт" for s in st):
                return None
            waiting = sorted((s for s in st if s["state"] == "в очереди"), key=lambda s: s["created"])
            if not waiting:
                return None
            d = root / waiting[0]["id"]
            _launch(d, _read_json(d / "task.json"))
            return waiting[0]["id"]
    except TimeoutError:
        return None


def status(tid: str, root: Path | None = None) -> dict:
    d = (root or TASKS_DIR) / tid
    meta = _read_json(d / "task.json")
    if not meta:
        raise FileNotFoundError(f"задача {tid} не найдена")
    state = _read_json(d / "state.json")
    meta = {**meta, **{k: v for k, v in state.items() if k in ("child", "finished", "returncode")}}
    meta.setdefault("created", meta.get("started") or time.time())
    meta.setdefault("heavy", CATALOG.get(meta.get("task"), TaskType("", "", (), "")).heavy)
    if meta.get("finished"):
        rc = meta.get("returncode")
        meta["state"] = "готово" if rc == 0 else ("остановлена" if rc == STOPPED_RC else "ошибка")
    elif meta.get("queued") and not meta.get("pid"):
        meta["state"] = "в очереди"
    elif _alive(meta.get("pid")) or _alive(meta.get("child")):
        meta["state"] = "идёт"
    else:
        meta["state"] = "прервана"
    t0 = meta.get("started") or meta["created"]
    meta["seconds"] = max(0.0, (meta.get("finished") or time.time()) - t0) if meta.get("started") else 0.0
    return meta


def _scan(root: Path, limit: int | None = None) -> list[dict]:
    out = []
    dirs = sorted((x for x in root.iterdir() if (x / "task.json").exists()), reverse=True) if root.exists() else []
    for d in dirs[:limit] if limit else dirs:
        try:
            out.append(status(d.name, root))
        except Exception:      # одна битая запись не должна ронять список
            continue
    return out


def list_tasks(root: Path | None = None, limit: int = 30) -> list[dict]:
    """Задачи (новые сверху) с состоянием и прогрессом. Заодно продвигает очередь (если обёртка упала, не успев это сделать)."""
    r = root or TASKS_DIR
    if not r.exists():
        return []
    pump(r)
    rows = _scan(r, limit)
    for m in rows:
        m["progress"] = progress(m["id"], r) if m["state"] == "идёт" else None
    return rows


PROGRESS_RE = re.compile(r"\[(\d+)\s*/\s*(\d+)\]\s*(.*)")


def progress(tid: str, root: Path | None = None) -> dict | None:
    """Последняя строка вида `[i/n] сообщение` журнала: {'done', 'total', 'frac', 'message'} или None."""
    f = (root or TASKS_DIR) / tid / "log.txt"
    if not f.exists():
        return None
    data = f.read_bytes()[-20000:].decode("utf-8", "replace").replace("\r", "\n")
    for line in reversed(data.splitlines()):
        m = PROGRESS_RE.search(line)
        if m and int(m.group(2)) > 0:
            i, n = int(m.group(1)), int(m.group(2))
            return dict(done=i, total=n, frac=min(1.0, i / n), message=m.group(3).strip()[:160])
    return None


def tail(tid: str, lines: int = 80, root: Path | None = None) -> str:
    f = (root or TASKS_DIR) / tid / "log.txt"
    if not f.exists():
        return ""
    data = f.read_bytes()[-60000:].decode("utf-8", "replace").replace("\r", "\n")
    return "\n".join(data.splitlines()[-lines:])


def stop(tid: str, root: Path | None = None) -> bool:
    """Останавливает задачу этого каталога: ожидающую — снимает с очереди, идущую — завершает обёртку и её дочерний `sd` (pid только из файлов задачи)."""
    r = root or TASKS_DIR
    st = status(tid, r)
    if st["state"] not in ("идёт", "в очереди"):
        return False
    d = r / tid
    if st["state"] == "идёт":
        for pid in (st.get("child"), st.get("pid")):
            if pid and _alive(pid):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True)
                else:
                    try:
                        os.kill(int(pid), 15)
                    except OSError:
                        pass
    _write_json(d / "state.json", {**_read_json(d / "state.json"), "finished": time.time(), "returncode": STOPPED_RC})
    pump(r)
    return True


def rerun(tid: str, root: Path | None = None) -> str:
    """Повторить задачу с теми же значениями формы."""
    m = status(tid, root)
    if m.get("task") not in CATALOG:
        raise KeyError(f"тип задачи {m.get('task')!r} больше не поддерживается")
    return start(m["task"], m.get("values") or {}, root)


def inputs_dir(root: Path | None = None) -> Path:
    """Каталог входных файлов задач (например, JSON вариантов профиля)."""
    d = (root or TASKS_DIR) / "_inputs"
    d.mkdir(parents=True, exist_ok=True)
    return d
