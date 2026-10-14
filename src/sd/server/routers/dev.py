"""Интерфейс разработчика через API (роль developer): профили, журнал прогонов и их пометки, задачи доработки, версии разметки, спецификации классификатора, выпуски.

Это те же операции, что в Dev Studio (`sd ui`): логика остаётся в `sd.web.admin`, `sd.experiments`, `sd.compare`, `sd.tasks`, `sd.labels_history`, `sd.classifier`, `sd.release` —
здесь они получают вход, роль и запись в журнал действий, чтобы веб-версия могла делать то же без Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from ... import classifier as CL
from ... import compare as CP
from ... import experiments as XP
from ... import labels_history as LH
from ... import profiles as PR
from ... import release as RL
from ... import tasks as TK
from ...web import admin
from .. import deps
from ..jsonutil import clean
from ..models import User
from ..schemas import CloneIn, Message, ProfileBody, RunNotes, TaskStart

router = APIRouter(prefix="/dev", tags=["разработчик"], dependencies=[Depends(deps.developer)])


# ---------------------------------------------------------------------------------------------- каталоги
@router.get("/catalog", summary="Доступные модели по типам (поза, детекторы, классификаторы, фото, VLM) и допустимые значения")
def catalog():
    return clean(admin.catalog())


@router.get("/registry", summary="Реестр весов, пакеты, установленные решатели")
def registry():
    return clean(admin.registry())


# ---------------------------------------------------------------------------------------------- профили
@router.get("/profiles", summary="Имена профилей конфигурации")
def profiles() -> list[str]:
    return PR.list_profiles()


@router.get("/profiles/{name}", summary="Профиль: настройки, действующие значения, проверка устройств")
def profile_get(name: str):
    return clean(admin.profile_get(name))


@router.put("/profiles/{name}", summary="Создать или изменить профиль (ключи проверяются до записи)")
def profile_put(name: str, body: ProfileBody, request: Request, me: User = Depends(deps.current_user), s=Depends(deps.get_session)):
    r = admin.profile_save(name, body.model_dump())
    deps.audit(s, request, me, "profile.save", "profile", name, keys=sorted(body.config))
    return clean(r)


@router.post("/profiles/{name}/clone", summary="Скопировать профиль")
def profile_clone(name: str, body: CloneIn, request: Request, me: User = Depends(deps.current_user), s=Depends(deps.get_session)):
    r = admin.profile_clone(name, body.to)
    deps.audit(s, request, me, "profile.clone", "profile", body.to, source=name)
    return clean(r)


@router.delete("/profiles/{name}", response_model=Message, summary="Удалить профиль")
def profile_delete(name: str, request: Request, me: User = Depends(deps.current_user), s=Depends(deps.get_session)) -> Message:
    if not PR.delete(name):
        raise HTTPException(404, "профиль не найден")
    deps.audit(s, request, me, "profile.delete", "profile", name)
    return Message()


@router.get("/profiles/{name}/yaml", response_class=PlainTextResponse, summary="Профиль в YAML (для переноса)")
def profile_yaml(name: str) -> PlainTextResponse:
    return PlainTextResponse(admin.profile_export_yaml(name), media_type="text/yaml; charset=utf-8")


# ---------------------------------------------------------------------------------------------- журнал прогонов
@router.get("/runs", summary="Журнал прогонов оценки с действующими параметрами (новые сверху)")
def runs(limit: int = Query(200, ge=1, le=2000)):
    df = CP.catalog()
    return clean(df.head(limit)) if len(df) else []


@router.get("/runs/{run_id}", summary="Прогон: метаданные и отчёт")
def run_detail(run_id: str):
    d = XP.EXP_DIR / Path(run_id).name
    if not (d / "meta.json").exists():
        raise HTTPException(404, "прогон не найден")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    report = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else None
    return clean(dict(meta=meta, report=report, notes=XP.read_notes(d.name)))


@router.patch("/runs/{run_id}/notes", summary="Тег подхода, заметка и закрепление прогона")
def run_notes(run_id: str, body: RunNotes, request: Request, me: User = Depends(deps.current_user), s=Depends(deps.get_session)):
    try:
        r = XP.set_notes(Path(run_id).name, **body.model_dump(exclude_unset=True))
    except FileNotFoundError:
        raise HTTPException(404, "прогон не найден") from None
    deps.audit(s, request, me, "run.notes", "run", run_id, **body.model_dump(exclude_unset=True))
    return clean(r)


@router.get("/runs/{run_id}/compare/{other_id}", summary="Парное сравнение двух прогонов на общих клипах (bootstrap)")
def run_compare(run_id: str, other_id: str, n: int = Query(300, ge=50, le=2000)):
    try:
        return clean(CP.pairwise([Path(run_id).name, Path(other_id).name], n=n))
    except FileNotFoundError:
        raise HTTPException(404, "прогон не найден") from None


# ---------------------------------------------------------------------------------------------- задачи
@router.get("/tasks/catalog", summary="Какие задачи можно запускать (белый список) и их поля")
def tasks_catalog():
    return clean(admin.tasks_catalog())


@router.get("/tasks", summary="Последние задачи")
def tasks(limit: int = Query(30, ge=1, le=200)):
    return clean(TK.list_tasks(limit=limit))


@router.post("/tasks", status_code=202, summary="Запустить задачу (тяжёлые встают в очередь)")
def task_start(body: TaskStart, request: Request, me: User = Depends(deps.current_user), s=Depends(deps.get_session)):
    if body.task not in TK.CATALOG:
        raise HTTPException(422, f"неизвестная задача {body.task!r}")
    tid = TK.start(body.task, body.values)
    deps.audit(s, request, me, "task.start", "task", tid, task=body.task)
    return dict(id=tid, **clean({k: v for k, v in TK.status(tid).items() if k in ("state", "title")}))


@router.get("/tasks/{task_id}", summary="Состояние задачи и конец журнала")
def task_get(task_id: str, lines: int = Query(120, ge=1, le=2000)):
    try:
        return clean(admin.task_log(task_id, lines) | dict(progress=TK.progress(task_id)))
    except FileNotFoundError:
        raise HTTPException(404, "задача не найдена") from None


@router.post("/tasks/{task_id}/stop", response_model=Message, summary="Остановить задачу")
def task_stop(task_id: str, request: Request, me: User = Depends(deps.current_user), s=Depends(deps.get_session)) -> Message:
    ok = TK.stop(task_id)
    deps.audit(s, request, me, "task.stop", "task", task_id, ok=ok)
    return Message(ok=bool(ok))


# ---------------------------------------------------------------------------------------------- разметка и классификаторы
@router.get("/labels/{kind}/versions", summary="Версии файла разметки (events | gestures | cycles)")
def label_versions(kind: str):
    if kind not in LH.files():
        raise HTTPException(404, f"kind ∈ {', '.join(LH.files())}")
    return clean(LH.versions(kind))


@router.post("/labels/{kind}/snapshot", summary="Зафиксировать текущую версию файла разметки")
def label_snapshot(kind: str, request: Request, note: str = Query("", max_length=300), me: User = Depends(deps.current_user), s=Depends(deps.get_session)):
    if kind not in LH.files():
        raise HTTPException(404, f"kind ∈ {', '.join(LH.files())}")
    r = LH.snapshot(kind, me.username, note or "через API", "api", force=True)
    deps.audit(s, request, me, "labels.snapshot", "labels", kind, note=note)
    return clean(r) or dict(note="файла разметки нет")


@router.get("/classifiers/specs", summary="Сохранённые спецификации классификатора")
def specs() -> list[str]:
    return CL.list_specs()


@router.get("/classifiers/specs/{name}", summary="Спецификация классификатора")
def spec_get(name: str):
    try:
        return clean(CL.load_spec(name))
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "спецификация не найдена") from None


# ---------------------------------------------------------------------------------------------- выпуски
@router.get("/releases", summary="Выпуски (ворота качества, метрики, подпись)")
def releases():
    return clean(RL.list_releases())


@router.get("/releases/gates", summary="Проверка ворот качества для профиля и прогона (без выпуска)")
def release_gates(profile: str, run_id: str, min_f1: float = Query(0.7, ge=0, le=1)):
    try:
        gs = RL.gates(profile, run_id, min_f1)
    except (FileNotFoundError, KeyError) as e:
        raise HTTPException(404, str(e)) from None
    return dict(passed=RL.passed(gs), gates=clean(gs))
