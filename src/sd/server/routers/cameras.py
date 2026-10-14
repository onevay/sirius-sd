"""Камеры: справочник, показатели по тревогам, создание и правка (admin)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from .. import deps
from ..models import User
from ..schemas import CameraIn, CameraMetrics, CameraOut, DistrictOut, Message
from ..services import cameras as CS

router = APIRouter(tags=["камеры"])


@router.get("/cameras", response_model=list[CameraMetrics], summary="Камеры с числом тревог")
def list_cameras(city: str | None = None, district: str | None = None, hours: float = Query(0.0, ge=0, description="период для показателей, часов; 0 — всё время"), enabled: bool | None = None,
                 _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> list[dict]:
    return CS.list_with_metrics(s, city=city, district=district, hours=hours, enabled=enabled)


@router.get("/districts", response_model=list[DistrictOut], summary="Города и районы с числом камер")
def districts(_: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)) -> list[dict]:
    return CS.districts(s)


@router.get("/cameras/{camera_id}", response_model=CameraOut, summary="Камера")
def get_camera(camera_id: str, _: User = Depends(deps.viewer), s: Session = Depends(deps.get_session)):
    cam = CS.get(s, camera_id)
    if cam is None:
        raise HTTPException(404, "камера не найдена")
    return cam


@router.put("/cameras/{camera_id}", response_model=CameraOut, summary="Создать или изменить камеру (передаются только изменяемые поля)")
def put_camera(camera_id: str, body: CameraIn, request: Request, me: User = Depends(deps.admin), s: Session = Depends(deps.get_session)):
    cam, created = CS.upsert(s, camera_id, body.model_dump(exclude_unset=True))
    deps.audit(s, request, me, "camera.create" if created else "camera.update", "camera", camera_id, **body.model_dump(exclude_unset=True))
    return cam


@router.delete("/cameras/{camera_id}", response_model=Message, summary="Удалить камеру (тревоги остаются; `with_alerts=true` удаляет и их)")
def delete_camera(camera_id: str, request: Request, with_alerts: bool = False, me: User = Depends(deps.admin), s: Session = Depends(deps.get_session)) -> Message:
    if not CS.delete(s, camera_id, with_alerts=with_alerts):
        raise HTTPException(404, "камера не найдена")
    deps.audit(s, request, me, "camera.delete", "camera", camera_id, with_alerts=with_alerts)
    return Message()
