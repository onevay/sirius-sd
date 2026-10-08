"""Фоновые задачи из веб-интерфейса: любая длинная операция проекта запускается как отдельный процесс `python -m sd <команда>` с журналом на диске.

Зачем: на новой машине всё должно делаться из браузера, без консоли — разметить, обогатить циклы, оценить, обучить, замерить «лестницу вычислений», скачать веса. Процесс не привязан к вкладке
браузера: закрытие страницы или перезапуск интерфейса его не останавливает, состояние читается с диска (`outputs/tasks/<id>/`):
  task.json — команда, время старта, pid, завершение, код возврата;  log.txt — вывод (stdout+stderr).
Допускаются только команды из `CATALOG` (белый список): произвольный текст в оболочку не передаётся, аргументы собираются списком (shell=False).
Останавливать можно только задачу, запущенную из этого интерфейса (по pid из её task.json, пока процесс жив и это тот же процесс).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import OUTPUTS, ROOT

TASKS_DIR = OUTPUTS / "tasks"


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str = "text"          # text | int | float | bool | choice | multi
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


CATALOG: dict[str, TaskType] = {t.id: t for t in (
    TaskType("doctor", "Проверка окружения", ("doctor",), "Версии, устройства (CPU / iGPU / CUDA), веса, данные, свободная память и место. Первое, что запускают на новой машине."),
    TaskType("models_get", "Скачать веса моделей", ("models", "get"), "Скачивает веса по id из реестра (нужна сеть). Community-.pt сканируются перед загрузкой.", (
        Field("ids", "Идентификаторы через пробел", "text", "yolo26n-pose yolo26s-pose smoking_yolo11m_beehzod smoking_yolo26s_basant18", None, help="`sd models list` — список"),), needs_net=True),
    TaskType("gt_from_gestures", "Эталон событий из меток жестов", ("gt-from-gestures",), "Перестраивает labels/events_gt.csv по меткам жестов (страница «Анализ»).", (
        Field("labeler", "Разметчик", "text", "assistant", "--labeler"),)),
    TaskType("enrich", "Признаки циклов: предмет, фото, VLM", ("enrich",), "Считает и кэширует (outputs/enrich) признаки предмета, фото-модели и VLM для циклов полных запусков. Повтор берёт готовое из кэша.", (
        Field("objects", "Детекторы предмета (через запятую; пусто — не считать)", "text", "smoking_yolo11m_beehzod,smoking_yolo26s_basant18", "--objects"),
        Field("vlm", "VLM (тег Ollama; пусто — без VLM)", "text", "", "--vlm"),
        Field("clips", "Только клипы (video_id через запятую; пусто — все)", "text", "", "--clips"),
        Field("photo_bundle", "Фото-модель", "text", "models/photo/photo_v2", "--photo-bundle"))),
    TaskType("oof_eval", "Честная оценка набора признаков", ("oof-eval",), "Оценка циклов моделью, не видевшей видео; F1 события по регламенту; вложенный подбор порогов.", (
        Field("sets", "Наборы признаков (через запятую)", "text", "obj_hold_zsd", "--sets"),
        Field("kinds", "Члены ансамбля", "text", "lr,gb,nn", "--kinds"),
        Field("repeats", "Перемешиваний", "int", 3, "--repeats"),
        Field("by_scene", "Фолды по сценам", "bool", False, "--by-scene"),
        Field("tag", "Метка результата", "text", "ui", "--tag"))),
    TaskType("ladder", "Лестница вычислений (поза)", ("ladder",), "Сколько затяжек находит автомат при разных моделях и размерах входа; масштаб < 1 имитирует более далёкую камеру. На новой машине запустите конфигурации gpu_*.", (
        Field("configs", "Конфигурации (через запятую)", "text", "n960,n1280", "--configs", help="n960 s960 n1280 n1600 n1920 s1600 m1600 m1280 x1280 gpu_s1600 gpu_m1600 gpu_m1920"),
        Field("clips", "Клипы video_id[:a-b][@масштаб] через запятую", "text", "", "--clips"),
        Field("scale", "Масштаб содержимого кадра (1 — как есть)", "float", 1.0, "--scale"),
        Field("tag", "Метка результата", "text", "ui", "--tag"))),
    TaskType("eval", "Оценка профиля на папках (sd eval)", ("eval",), "Event F1 по эталону для профиля из configs/experiments; результат — в журнале страницы «Оценка».", (
        Field("dirs", "Папки с видео (по одной на строку)", "lines", "data/курение\ndata/лжекурение", "--dir"),
        Field("profile", "Профиль", "text", "laptop_cpu", "--profile"),
        Field("policy", "Порог", "choice", "fixed", "--policy", choices=("fixed", "plateau", "best")))),
    TaskType("train_bundle", "Обучить пакет классификатора (sd train-bundle)", ("train-bundle",), "Командная версия страницы «Классификатор» (без диагностики).", (
        Field("name", "Имя пакета", "text", "cycle_new", None),
        Field("set", "Набор признаков", "text", "obj_hold_zsd", "--set"),
        Field("enriched", "С признаками enrich", "bool", True, "--enriched"),
        Field("calibrate", "Изотоническая калибровка", "bool", False, "--calibrate"),
        Field("kinds", "Члены ансамбля", "text", "lr,gb,nn", "--kinds"))),
)}


def _args(t: TaskType, values: dict[str, Any]) -> list[str]:
    """Собирает аргументы командной строки из значений формы по описанию полей (ничего, кроме описанных полей, не принимается)."""
    out = list(t.command)
    pos: list[str] = []
    for f in t.fields:
        v = values.get(f.key, f.default)
        if f.kind == "bool":
            if f.flag and (f.flag in ("--calibrate",)):
                out.append("--calibrate" if v else "--no-calibrate")
            elif f.flag and v:
                out.append(f.flag)
            continue
        if v is None or str(v).strip() == "":
            if f.flag and f.key in ("objects", "vlm"):       # пустое значение осмысленно: «не считать»
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


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"], capture_output=True, text=True)
            return any(len(p) > 1 and p[1].strip('"') == str(int(pid)) for p in (ln.split('","') for ln in r.stdout.splitlines()))
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def start(task_id: str, values: dict[str, Any], root: Path | None = None) -> str:
    """Запускает задачу отдельным процессом и возвращает её id. Вывод — в `<root>/<id>/log.txt`."""
    args = command(task_id, values)
    root = root or TASKS_DIR
    tid = f"{time.strftime('%Y%m%d_%H%M%S')}_{task_id}"
    d = root / tid
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(id=tid, task=task_id, title=CATALOG[task_id].title, args=args, started=time.time(), pid=None, finished=None, returncode=None)
    (d / "task.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    # обёртка пишет код возврата и время окончания даже если интерфейс уже перезапущен
    runner = ("import json,subprocess,sys,time;d=sys.argv[1];a=sys.argv[2:];"
              "f=open(d+'/log.txt','ab');"
              "p=subprocess.Popen([sys.executable,'-m','sd']+a,stdout=f,stderr=subprocess.STDOUT,cwd=%r);"
              "m=json.load(open(d+'/task.json',encoding='utf-8'));m['child']=p.pid;json.dump(m,open(d+'/task.json','w',encoding='utf-8'),ensure_ascii=False);"
              "r=p.wait();m['finished']=time.time();m['returncode']=r;json.dump(m,open(d+'/task.json','w',encoding='utf-8'),ensure_ascii=False)") % str(ROOT)
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen([sys.executable, "-c", runner, str(d), *args], cwd=str(ROOT), env=env, creationflags=flags, start_new_session=(os.name != "nt"),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    meta["pid"] = p.pid
    cur = json.loads((d / "task.json").read_text(encoding="utf-8"))      # обёртка могла уже дописать child
    cur["pid"] = p.pid
    (d / "task.json").write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
    return tid


def status(tid: str, root: Path | None = None) -> dict:
    d = (root or TASKS_DIR) / tid
    meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
    if meta.get("finished"):
        meta["state"] = "готово" if meta.get("returncode") == 0 else "ошибка"
    elif _alive(meta.get("pid")):
        meta["state"] = "идёт"
    else:
        meta["state"] = "прервана"
    meta["seconds"] = (meta.get("finished") or time.time()) - meta["started"]
    return meta


def tail(tid: str, lines: int = 80, root: Path | None = None) -> str:
    f = (root or TASKS_DIR) / tid / "log.txt"
    if not f.exists():
        return ""
    data = f.read_bytes()[-60000:].decode("utf-8", "replace").replace("\r", "\n")
    return "\n".join(data.splitlines()[-lines:])


def list_tasks(root: Path | None = None, limit: int = 30) -> list[dict]:
    r = root or TASKS_DIR
    if not r.exists():
        return []
    out = []
    for d in sorted((x for x in r.iterdir() if (x / "task.json").exists()), reverse=True)[:limit]:
        try:
            out.append(status(d.name, r))
        except (OSError, ValueError):
            continue
    return out


def stop(tid: str, root: Path | None = None) -> bool:
    """Останавливает задачу, запущенную этим интерфейсом (процесс-обёртка и её дочерний `sd`). Чужие процессы не затрагиваются: pid берётся только из task.json."""
    st = status(tid, root)
    if st["state"] != "идёт":
        return False
    for pid in (st.get("child"), st.get("pid")):
        if pid and _alive(pid):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True)
            else:
                try:
                    os.kill(int(pid), 15)
                except OSError:
                    pass
    d = (root or TASKS_DIR) / tid
    m = json.loads((d / "task.json").read_text(encoding="utf-8"))
    m["finished"], m["returncode"] = time.time(), -15
    (d / "task.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return True
