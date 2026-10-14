"""Схема БД платформы. Единственный источник правды о таблицах — этот файл + миграции в `migrations/versions` (тест `test_migrations_match_models` следит, чтобы они не расходились).

Идентичность и аудит (миграция 0001): users, api_tokens, audit_log.
Мониторинг (миграция 0002): cameras, chunks, alerts, alert_reviews.
Все времена — локальные наивные (см. `clock.py`). Решения оператора по тревоге не перезаписываются: каждое — строка `alert_reviews`, а в `alerts.status` лежит последнее.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

ROLES = ("viewer", "operator", "developer", "admin")        # по возрастанию прав: каждая следующая включает предыдущие
ALERT_STATUSES = ("new", "confirmed", "false", "unsure")
ALERT_STATES = ("open", "closed")
CAMERA_STATUSES = ("idle", "processing", "offline")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="role"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)          # в нижнем регистре
    display_name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    token_version: Mapped[int] = mapped_column(Integer, default=1)          # увеличение отзывает все выданные токены доступа (смена пароля, блокировка)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    tokens: Mapped[list["ApiToken"]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)


class ApiToken(Base):
    """Долгоживущий токен для интеграций и скриптов. Хранится только sha256 токена; сам токен показывается один раз при создании."""

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    prefix: Mapped[str] = mapped_column(String(16))                         # начало токена — чтобы узнавать его в списке
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="tokens")


class AuditLog(Base):
    """Журнал действий (только добавление): вход, смена прав, решения по тревогам, запуск задач, правки профилей."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_log_entity", "entity", "entity_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    username: Mapped[str] = mapped_column(String(64), default="")           # копия имени: журнал читается и после удаления пользователя
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity: Mapped[str] = mapped_column(String(32), default="")
    entity_id: Mapped[str] = mapped_column(String(128), default="")
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ip: Mapped[str] = mapped_column(String(64), default="")


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (CheckConstraint(_in("status", CAMERA_STATUSES), name="status"),)

    camera_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    city: Mapped[str] = mapped_column(String(80), default="", index=True)
    district: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    cam_index: Mapped[str | None] = mapped_column(String(16), nullable=True)
    title: Mapped[str] = mapped_column(String(160), default="")
    street: Mapped[str] = mapped_column(String(160), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    folder: Mapped[str | None] = mapped_column(String(512), nullable=True)   # источник-папка (имитация потока)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)   # RTSP/HTTP-поток (резерв: подключение к монитору — этап E3)
    start_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="idle")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    processed_sec: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="camera", cascade="all, delete-orphan", passive_deletes=True)


class Chunk(Base):
    __tablename__ = "chunks"

    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.camera_id", ondelete="CASCADE"), primary_key=True)
    name: Mapped[str] = mapped_column(String(512), primary_key=True)
    duration: Mapped[float] = mapped_column(Float)
    offset_sec: Mapped[float] = mapped_column(Float)
    done_at: Mapped[datetime] = mapped_column(DateTime)

    camera: Mapped[Camera] = relationship(back_populates="chunks")


class Alert(Base):
    """Тревога (событие курения). `camera_id` — мягкая ссылка без внешнего ключа: воркер пишет тревоги потока независимо от порядка заведения камер."""

    __tablename__ = "alerts"
    __table_args__ = (CheckConstraint(_in("status", ALERT_STATUSES), name="status"), CheckConstraint(_in("state", ALERT_STATES), name="state"),
                      Index("ix_alerts_status_t_abs", "status", "t_abs"), Index("ix_alerts_camera_id_t_abs", "camera_id", "t_abs"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(255), unique=True)              # <камера>:<ID человека>:<начало> — защита от дублей при перезапуске воркера
    camera_id: Mapped[str] = mapped_column(String(128))
    district: Mapped[str | None] = mapped_column(String(80), nullable=True)
    cam_index: Mapped[str | None] = mapped_column(String(16), nullable=True)
    chunk: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chunk_offset: Mapped[float] = mapped_column(Float, default=0.0)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    peak_sec: Mapped[float] = mapped_column(Float)
    t_abs: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # абсолютное время события (по часам камеры)
    t_raised: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)   # когда тревога поднята
    confidence: Mapped[float] = mapped_column(Float)
    tid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    box: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    rule: Mapped[str | None] = mapped_column(String(64), nullable=True)
    explain: Mapped[str | None] = mapped_column(Text, nullable=True)
    n_cycles: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state: Mapped[str] = mapped_column(String(8), default="open")
    status: Mapped[str] = mapped_column(String(12), default="new")
    reviewer: Mapped[str | None] = mapped_column(String(120), nullable=True)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    thumb: Mapped[str | None] = mapped_column(String(512), nullable=True)
    clip: Mapped[str | None] = mapped_column(String(512), nullable=True)
    demo: Mapped[bool] = mapped_column(Boolean, default=False)
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    reviews: Mapped[list["AlertReview"]] = relationship(back_populates="alert", cascade="all, delete-orphan", passive_deletes=True, order_by="AlertReview.id")


class AlertReview(Base):
    """История решений оператора по тревоге (неизменяемая): кто, когда, что решил и почему."""

    __tablename__ = "alert_reviews"
    __table_args__ = (CheckConstraint(_in("status", ALERT_STATUSES), name="status"), Index("ix_alert_reviews_alert_id_id", "alert_id", "id"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(12))
    reason: Mapped[str | None] = mapped_column(String(64), nullable=True)   # причина «ложная» из справочника (питьё, телефон, …)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewer_name: Mapped[str] = mapped_column(String(120), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)

    alert: Mapped[Alert] = relationship(back_populates="reviews")


__all__ = ["Base", "User", "ApiToken", "AuditLog", "Camera", "Chunk", "Alert", "AlertReview", "ROLES", "ALERT_STATUSES"]
