"""Источники видео, воспроизведение и разбор видео моделью (очередь заданий). Логика — `services.videos.VideoService`; здесь она получает вход, роли и аудит."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .. import deps
from ..jsonutil import clean
from ..models import User
from ..schemas import AnalyzeIn, ExportIn, Message, RootIn

router = APIRouter(tags=["разбор видео"])


def app_of(ctx: deps.Ctx):
    return ctx.videos


# ---------------------------------------------------------------------------------------------- источники
@router.get("/sources/folders", summary="Папки с видео: камеры-папки и подключённые источники")
def folders(ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return clean(app_of(ctx).folders())


@router.get("/sources/videos", summary="Видео папки (с итогом прошлого разбора)")
def videos(folder: str = Query(..., description="идентификатор из /sources/folders"), ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return clean(app_of(ctx).videos(folder))


@router.get("/sources/folder-of", summary="Папка, в которой лежит видео")
def folder_of(video: str, ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return dict(folder=app_of(ctx).folder_of(video))


@router.post("/sources/roots", summary="Подключить папку с видео на сервере (только при локальной привязке)", status_code=201)
def add_root(body: RootIn, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.admin), s=Depends(deps.get_session)):
    r = app_of(ctx).add_root(body.path)
    deps.audit(s, request, me, "source.add", "root", r["key"], path=r["path"])
    return clean(r)


@router.delete("/sources/roots/{key}", response_model=Message, summary="Отключить подключённую папку")
def remove_root(key: str, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.admin), s=Depends(deps.get_session)) -> Message:
    if not app_of(ctx).remove_root(key):
        raise HTTPException(404, "источник не найден")
    deps.audit(s, request, me, "source.remove", "root", key)
    return Message()


@router.post("/sources/upload", summary="Загрузить видео (multipart; только при локальной привязке)", status_code=201)
def upload(request: Request, file: UploadFile = File(...), ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.operator), s=Depends(deps.get_session)):
    size = file.size
    if size is None:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
    r = app_of(ctx).save_upload(file.filename or "", file.file, int(size))
    deps.audit(s, request, me, "source.upload", "video", r["id"], name=r["name"], size=size)
    return clean(r)


# ---------------------------------------------------------------------------------------------- воспроизведение
def _playable(r: dict):
    if r["path"] is None:
        code = 202 if r["state"] == "preparing" else 500
        return JSONResponse(dict(state=r["state"], error=r["error"], pct=round(r["pct"], 3)), status_code=code)
    return FileResponse(r["path"], media_type=r["mime"])


@router.get("/media/video/{fid}", summary="Видео для браузера (H.264/VP9, Range); 202 — копия готовится")
def media_video(fid: str, fmt: str | None = Query(None, pattern="^(h264|vp9)$"), ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return _playable(app_of(ctx).media_for(fid, fmt))


@router.get("/media/status", summary="Готовность копии видео (по файлу `id` или по тревоге `alert`)")
def media_status(id: str | None = None, alert: int | None = None, fmt: str | None = Query(None, pattern="^(h264|vp9)$"), ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    if id is None and alert is None:
        raise HTTPException(422, "нужен id или alert")
    r = app_of(ctx).media_for_alert(alert, fmt) if alert is not None else app_of(ctx).media_for(id, fmt)
    return dict(state=r["state"], error=r["error"], pct=round(r["pct"], 3))


@router.get("/media/alert/{alert_id}/clip", summary="Клип тревоги в формате для браузера")
def media_alert_clip(alert_id: int, fmt: str | None = Query(None, pattern="^(h264|vp9)$"), ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return _playable(app_of(ctx).media_for_alert(alert_id, fmt))


@router.get("/media/export/{name}", summary="Готовое видео с разметкой (скачивание)", responses={404: {"description": "файла нет"}})
def media_export(name: str, ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return FileResponse(app_of(ctx).export_path(name), media_type="video/mp4", filename=name)


# ---------------------------------------------------------------------------------------------- разбор
@router.get("/analysis/profiles", summary="Профили, пригодные для разбора (с качеством по журналу оценок)")
def profiles(ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return clean(app_of(ctx).profiles())


@router.post("/analysis/jobs", status_code=202, summary="Поставить разбор видео или всей папки в очередь")
def submit(body: AnalyzeIn, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.operator), s=Depends(deps.get_session)):
    app = app_of(ctx)
    if body.folder:
        jobs = [j.to_dict() for j in app.submit_folder(body.folder, body.profile)]
        deps.audit(s, request, me, "analysis.submit_folder", "folder", body.folder, profile=body.profile, n=len(jobs))
        return clean(jobs)
    if not body.video:
        raise HTTPException(422, "нужен video или folder")
    job = app.submit(body.video, body.profile, body.live)
    deps.audit(s, request, me, "analysis.submit", "video", body.video, profile=body.profile, live=body.live)
    return clean(job.to_dict())


@router.post("/analysis/export", summary="Видео с разметкой из сохранённого разбора: рамки людей, тревоги, эталон (модель заново не запускается)")
def export(body: ExportIn, request: Request, ctx: deps.Ctx = Depends(deps.get_ctx), me: User = Depends(deps.operator), s=Depends(deps.get_session)):
    r = app_of(ctx).export_video(body.video, body.profile)
    deps.audit(s, request, me, "analysis.export", "video", body.video, profile=body.profile, file=r["name"])
    return clean(r)


@router.get("/analysis/jobs", summary="Последние задания разбора")
def jobs(limit: int = Query(50, ge=1, le=200), ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return clean(app_of(ctx).jobs_list(limit))


@router.get("/analysis/jobs/{job_id}", summary="Состояние задания")
def job(job_id: str, ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return clean(app_of(ctx).job(job_id).to_dict())


@router.get("/analysis/jobs/{job_id}/stream", summary="Кадры и тревоги живого разбора после курсора `since`")
def job_stream(job_id: str, since: int = Query(0, ge=0), ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return clean(app_of(ctx).stream(job_id, since))


@router.get("/analysis/result", summary="Последний готовый разбор видео (трасса, тревоги, итог)")
def result(video: str, profile: str | None = None, ctx: deps.Ctx = Depends(deps.get_ctx), _: User = Depends(deps.viewer)):
    return clean(app_of(ctx).latest_analysis(video, profile) or dict(trace=None, meta=None))
