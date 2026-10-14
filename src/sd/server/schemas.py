"""Схемы запросов и ответов API (pydantic v2). Они же дают OpenAPI-описание: клиенты (веб, десктоп, интеграции) генерируются из `/openapi.json`."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")
Role = Literal["viewer", "operator", "developer", "admin"]
AlertStatus = Literal["new", "confirmed", "false", "unsure"]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class Message(BaseModel):
    ok: bool = True
    detail: str = ""


class ErrorOut(BaseModel):
    detail: str


# ---------------------------------------------------------------------------------------------- система
class HealthOut(BaseModel):
    ok: bool
    version: str
    db: bool
    schema_current: bool = Field(description="версия схемы БД совпадает с последней миграцией")
    schema_revision: list[str]
    auth_required: bool


# ---------------------------------------------------------------------------------------------- пользователи
class UserOut(ORM):
    id: int | None
    username: str
    display_name: str
    role: Role
    is_active: bool
    created_at: datetime | None = None
    last_login_at: datetime | None = None


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=63)
    password: str = Field(min_length=1, max_length=256)
    role: Role = "viewer"
    display_name: str = Field(default="", max_length=120)


class UserUpdate(BaseModel):
    role: Role | None = None
    display_name: str | None = Field(default=None, max_length=120)
    is_active: bool | None = None
    password: str | None = Field(default=None, max_length=256)


class LoginIn(BaseModel):
    username: str
    password: str
    cookie: bool = Field(default=False, description="дополнительно выдать HttpOnly-cookie (для <video>/<img> и SSE в браузере); изменяющие запросы по cookie не принимаются")


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class PasswordChange(BaseModel):
    old_password: str
    new_password: str = Field(max_length=256)


class ApiTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    ttl_days: int | None = Field(default=None, ge=1, le=3650, description="срок жизни, дней; пусто — бессрочный")


class ApiTokenOut(ORM):
    id: int
    name: str
    prefix: str
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiTokenCreated(ApiTokenOut):
    token: str = Field(description="показывается один раз")


class AuditOut(ORM):
    id: int
    ts: datetime
    username: str
    action: str
    entity: str
    entity_id: str
    details: dict[str, Any] | None
    ip: str


# ---------------------------------------------------------------------------------------------- камеры
class CameraIn(BaseModel):
    city: str | None = Field(default=None, max_length=80)
    district: str | None = Field(default=None, max_length=80)
    cam_index: str | None = Field(default=None, max_length=16)
    title: str | None = Field(default=None, max_length=160)
    street: str | None = Field(default=None, max_length=160)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    folder: str | None = Field(default=None, max_length=512)
    source_url: str | None = Field(default=None, max_length=512)
    enabled: bool | None = None


class CameraOut(ORM):
    camera_id: str
    city: str
    district: str | None
    cam_index: str | None
    title: str
    street: str
    lat: float | None
    lon: float | None
    folder: str | None
    source_url: str | None
    start_at: datetime | None
    status: str
    enabled: bool
    last_seen: datetime | None
    processed_sec: float
    created_at: datetime


class CameraMetrics(BaseModel):
    camera: CameraOut
    alerts: int
    pending: int
    mean_confidence: float | None


class DistrictOut(BaseModel):
    city: str
    district: str
    cameras: int


# ---------------------------------------------------------------------------------------------- тревоги
class AlertOut(ORM):
    id: int
    key: str
    camera_id: str
    district: str | None
    cam_index: str | None
    start_sec: float
    end_sec: float
    peak_sec: float
    t_abs: datetime | None
    t_raised: datetime | None
    confidence: float
    tid: int | None
    box: list[float] | None
    rule: str | None
    explain: str | None
    n_cycles: int | None
    state: str
    status: AlertStatus
    reviewer: str | None
    reviewed_at: datetime | None
    note: str | None
    demo: bool
    fingerprint: str | None
    created_at: datetime
    updated_at: datetime
    has_thumb: bool = False
    has_clip: bool = False


class ReviewIn(BaseModel):
    status: AlertStatus
    note: str = Field(default="", max_length=1000)
    reason: str | None = Field(default=None, max_length=64, description="для «false»: причина из справочника `/review-reasons`")


class ReviewOut(ORM):
    id: int
    alert_id: int
    status: AlertStatus
    reason: str | None
    note: str | None
    reviewer_name: str
    created_at: datetime


class BulkReviewIn(ReviewIn):
    ids: list[int] = Field(min_length=1, max_length=500)


class BulkReviewOut(BaseModel):
    done: list[int]
    missing: list[int]


class StatsOut(BaseModel):
    new: int
    confirmed: int
    false: int
    unsure: int
    total: int
    today: dict[str, int]
    cameras: int
    precision: float | None
    median_review_sec: float | None
    max_id: int


# ---------------------------------------------------------------------------------------------- анализ видео
class AnalyzeIn(BaseModel):
    video: str | None = Field(default=None, description="непрозрачный идентификатор файла из `/sources/videos`")
    folder: str | None = Field(default=None, description="идентификатор папки из `/sources/folders`: разобрать все видео")
    profile: str
    live: bool = False


class RootIn(BaseModel):
    path: str = Field(max_length=1024)


class TaskStart(BaseModel):
    task: str
    values: dict[str, Any] = Field(default_factory=dict)


class RunNotes(BaseModel):
    tag: str | None = Field(default=None, max_length=80)
    note: str | None = Field(default=None, max_length=2000)
    pinned: bool | None = None


class ProfileBody(BaseModel):
    description: str = ""
    base: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class CloneIn(BaseModel):
    to: str = Field(min_length=1, max_length=64)


class DemoIn(BaseModel):
    clear: bool = False
    n: int = Field(default=14, ge=1, le=200)
