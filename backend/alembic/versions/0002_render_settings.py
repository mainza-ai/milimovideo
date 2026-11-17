"""Persist render quality settings on the project.

Revision ID: 0002_render_settings
Revises: 0001_baseline
Create Date: 2026-11-17

Fresh databases already get these columns from SQLModel.metadata.create_all() in
the baseline, so this revision checks for the columns before adding them. Legacy
databases repaired by ``_add_missing_columns`` are stamped at head.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0002_render_settings"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

COLUMNS = {
    "render_crf": sa.Column("render_crf", sa.Integer(), nullable=False, server_default="18"),
    "render_speed": sa.Column("render_speed", sa.String(), nullable=False, server_default="fast"),
    "render_audio_bitrate": sa.Column(
        "render_audio_bitrate", sa.String(), nullable=False, server_default="192k"
    ),
    "render_audio_normalize": sa.Column(
        "render_audio_normalize", sa.Boolean(), nullable=False, server_default=sa.false()
    ),
}


def upgrade() -> None:
    existing = {column["name"] for column in inspect(op.get_bind()).get_columns("project")}
    for name, column in COLUMNS.items():
        if name not in existing:
            op.add_column("project", column)


def downgrade() -> None:
    existing = {column["name"] for column in inspect(op.get_bind()).get_columns("project")}
    for name in reversed(list(COLUMNS)):
        if name in existing:
            op.drop_column("project", name)
