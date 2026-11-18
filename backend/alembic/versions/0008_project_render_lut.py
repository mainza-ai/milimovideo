"""Add the project render LUT path.

Revision ID: 0008_project_render_lut
Revises: 0007_shot_priority_steps
Create Date: 2026-11-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0008_project_render_lut"
down_revision = "0007_shot_priority_steps"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("project")}
    if "render_lut" not in columns:
        op.add_column("project", sa.Column("render_lut", sa.Text(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("project")}
    if "render_lut" in columns:
        op.drop_column("project", "render_lut")
