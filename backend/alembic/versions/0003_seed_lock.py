"""Persist the per-shot seed lock.

Revision ID: 0003_seed_lock
Revises: 0002_render_settings
Create Date: 2026-11-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0003_seed_lock"
down_revision = "0002_render_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("shot")}
    if "seed_locked" not in columns:
        op.add_column(
            "shot",
            sa.Column("seed_locked", sa.Boolean(), nullable=False, server_default=sa.true()),
        )


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("shot")}
    if "seed_locked" in columns:
        op.drop_column("shot", "seed_locked")
