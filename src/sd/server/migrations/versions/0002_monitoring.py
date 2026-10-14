"""Мониторинг: камеры, обработанные фрагменты, тревоги и история решений оператора.

Revision ID: 0002_monitoring
Revises: 0001_identity
Create Date: 2026-10-14
"""
import sqlalchemy as sa
from alembic import op

revision = "0002_monitoring"
down_revision = "0001_identity"
branch_labels = None
depends_on = None

STATUSES = "'new', 'confirmed', 'false', 'unsure'"


def upgrade() -> None:
    op.create_table(
        "cameras",
        sa.Column("camera_id", sa.String(128), nullable=False),
        sa.Column("city", sa.String(80), nullable=False),
        sa.Column("district", sa.String(80), nullable=True),
        sa.Column("cam_index", sa.String(16), nullable=True),
        sa.Column("title", sa.String(160), nullable=False),
        sa.Column("street", sa.String(160), nullable=False),
        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lon", sa.Float(), nullable=True),
        sa.Column("folder", sa.String(512), nullable=True),
        sa.Column("source_url", sa.String(512), nullable=True),
        sa.Column("start_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_seen", sa.DateTime(), nullable=True),
        sa.Column("processed_sec", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("status IN ('idle', 'processing', 'offline')", name=op.f("ck_cameras_status")),
        sa.PrimaryKeyConstraint("camera_id", name=op.f("pk_cameras")),
    )
    op.create_index(op.f("ix_cameras_city"), "cameras", ["city"])
    op.create_index(op.f("ix_cameras_district"), "cameras", ["district"])
    op.create_table(
        "chunks",
        sa.Column("camera_id", sa.String(128), nullable=False),
        sa.Column("name", sa.String(512), nullable=False),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column("offset_sec", sa.Float(), nullable=False),
        sa.Column("done_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.camera_id"], name=op.f("fk_chunks_camera_id_cameras"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("camera_id", "name", name=op.f("pk_chunks")),
    )
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("camera_id", sa.String(128), nullable=False),
        sa.Column("district", sa.String(80), nullable=True),
        sa.Column("cam_index", sa.String(16), nullable=True),
        sa.Column("chunk", sa.String(512), nullable=True),
        sa.Column("chunk_offset", sa.Float(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("peak_sec", sa.Float(), nullable=False),
        sa.Column("t_abs", sa.DateTime(), nullable=True),
        sa.Column("t_raised", sa.DateTime(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("tid", sa.Integer(), nullable=True),
        sa.Column("box", sa.JSON(), nullable=True),
        sa.Column("rule", sa.String(64), nullable=True),
        sa.Column("explain", sa.Text(), nullable=True),
        sa.Column("n_cycles", sa.Integer(), nullable=True),
        sa.Column("state", sa.String(8), nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("reviewer", sa.String(120), nullable=True),
        sa.Column("reviewer_id", sa.Integer(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("thumb", sa.String(512), nullable=True),
        sa.Column("clip", sa.String(512), nullable=True),
        sa.Column("demo", sa.Boolean(), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(f"status IN ({STATUSES})", name=op.f("ck_alerts_status")),
        sa.CheckConstraint("state IN ('open', 'closed')", name=op.f("ck_alerts_state")),
        sa.ForeignKeyConstraint(["reviewer_id"], ["users.id"], name=op.f("fk_alerts_reviewer_id_users"), ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
        sa.UniqueConstraint("key", name=op.f("uq_alerts_key")),
    )
    op.create_index("ix_alerts_status_t_abs", "alerts", ["status", "t_abs"])
    op.create_index("ix_alerts_camera_id_t_abs", "alerts", ["camera_id", "t_abs"])
    op.create_table(
        "alert_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("alert_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("reason", sa.String(64), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("reviewer_id", sa.Integer(), nullable=True),
        sa.Column("reviewer_name", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(f"status IN ({STATUSES})", name=op.f("ck_alert_reviews_status")),
        sa.ForeignKeyConstraint(["alert_id"], ["alerts.id"], name=op.f("fk_alert_reviews_alert_id_alerts"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reviewer_id"], ["users.id"], name=op.f("fk_alert_reviews_reviewer_id_users"), ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alert_reviews")),
    )
    op.create_index("ix_alert_reviews_alert_id_id", "alert_reviews", ["alert_id", "id"])


def downgrade() -> None:
    op.drop_index("ix_alert_reviews_alert_id_id", table_name="alert_reviews")
    op.drop_table("alert_reviews")
    op.drop_index("ix_alerts_camera_id_t_abs", table_name="alerts")
    op.drop_index("ix_alerts_status_t_abs", table_name="alerts")
    op.drop_table("alerts")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_cameras_district"), table_name="cameras")
    op.drop_index(op.f("ix_cameras_city"), table_name="cameras")
    op.drop_table("cameras")
