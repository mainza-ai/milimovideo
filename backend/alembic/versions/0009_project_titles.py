"""Add project title overlays.

Revision ID: 0009_project_titles
Revises: 0008_project_render_lut
Create Date: 2026-11-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0009_project_titles"
down_revision = "0008_project_render_lut"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("project")}
    if "titles" not in columns:
        op.add_column("project", sa.Column("titles", sa.Text(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("project")}
    if "titles" in columns:
        op.drop_column("project", "titles")
