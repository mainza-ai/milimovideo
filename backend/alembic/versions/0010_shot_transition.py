"""Add per-shot V1 crossfade duration.

Revision ID: 0010_shot_transition
Revises: 0009_project_titles
Create Date: 2026-11-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0010_shot_transition"
down_revision = "0009_project_titles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("shot")}
    if "transition_in" not in columns:
        op.add_column("shot", sa.Column("transition_in", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("shot")}
    if "transition_in" in columns:
        op.drop_column("shot", "transition_in")
