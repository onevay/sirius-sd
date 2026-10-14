"""Камеры: справочник с показателями по тревогам, создание и правка (поля воркера — статус, обработано секунд — правит только он)."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import clock
from ..models import CAMERA_STATUSES, Alert, Camera

EDITABLE = ("city", "district", "cam_index", "title", "street", "lat", "lon", "folder", "source_url", "enabled")


class CameraError(ValueError):
    pass


def _check(data: dict[str, Any]) -> None:
    lat, lon = data.get("lat"), data.get("lon")
    if lat is not None and not -90 <= float(lat) <= 90:
        raise CameraError("lat ∈ [−90, 90]")
    if lon is not None and not -180 <= float(lon) <= 180:
        raise CameraError("lon ∈ [−180, 180]")
    for k in ("camera_id",):
        if k in data and not str(data[k]).strip():
            raise CameraError("пустой идентификатор камеры")


def get(s: Session, camera_id: str) -> Camera | None:
    return s.get(Camera, camera_id)


def upsert(s: Session, camera_id: str, data: dict[str, Any]) -> tuple[Camera, bool]:
    """Создаёт камеру или обновляет редактируемые поля. Возвращает (камера, создана)."""
    cid = camera_id.strip()
    _check({**data, "camera_id": cid})
    cam = s.get(Camera, cid)
    created = cam is None
    if created:
        cam = Camera(camera_id=cid, city="", title="", street="", status="idle", enabled=True, processed_sec=0.0, created_at=clock.now())
        s.add(cam)
    for k in EDITABLE:
        if k in data:
            setattr(cam, k, data[k])
    s.flush()
    return cam, created


def delete(s: Session, camera_id: str, *, with_alerts: bool = False) -> bool:
    """Удаляет камеру и её фрагменты; тревоги остаются (история), если не задано `with_alerts`."""
    cam = s.get(Camera, camera_id)
    if cam is None:
        return False
    if with_alerts:
        for a in s.scalars(select(Alert).where(Alert.camera_id == camera_id)):
            s.delete(a)
    s.delete(cam)
    return True


def list_with_metrics(s: Session, *, city: str | None = None, district: str | None = None, hours: float = 0.0, enabled: bool | None = None) -> list[dict]:
    """Камеры с числом тревог (всего или за последние `hours` часов) и ждущих решения."""
    q = select(Camera).order_by(Camera.city, Camera.district, Camera.cam_index, Camera.camera_id)
    if city:
        q = q.where(Camera.city == city)
    if district:
        q = q.where(Camera.district == district)
    if enabled is not None:
        q = q.where(Camera.enabled.is_(enabled))
    cams = list(s.scalars(q))
    cond = []
    if hours and hours > 0:
        cond = [Alert.t_abs >= clock.now() - timedelta(hours=float(hours))]
    tot = dict(s.execute(select(Alert.camera_id, func.count()).where(*cond).group_by(Alert.camera_id)).all())
    pend = dict(s.execute(select(Alert.camera_id, func.count()).where(Alert.status == "new", *cond).group_by(Alert.camera_id)).all())
    conf = dict(s.execute(select(Alert.camera_id, func.avg(Alert.confidence)).where(*cond).group_by(Alert.camera_id)).all())
    return [dict(camera=c, alerts=int(tot.get(c.camera_id, 0)), pending=int(pend.get(c.camera_id, 0)), mean_confidence=float(conf[c.camera_id]) if conf.get(c.camera_id) is not None else None)
            for c in cams]


def districts(s: Session) -> list[dict]:
    """Районы (и города) с числом камер — для фильтров интерфейса."""
    rows = s.execute(select(Camera.city, Camera.district, func.count()).group_by(Camera.city, Camera.district).order_by(Camera.city, Camera.district)).all()
    return [dict(city=c or "", district=d or "", cameras=int(n)) for c, d, n in rows]


__all__ = ["CAMERA_STATUSES", "CameraError", "EDITABLE", "get", "upsert", "delete", "list_with_metrics", "districts"]
