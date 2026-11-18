"""Add named ComfyUI workflows.

Revision ID: 0011_comfy_workflows
Revises: 0010_shot_transition
Create Date: 2026-11-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0011_comfy_workflows"
down_revision = "0010_shot_transition"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = inspect(op.get_bind()).get_table_names()
    if "comfyworkflow" in tables:
        return
    op.create_table(
        "comfyworkflow",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("project_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("graph_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("bindings_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_comfyworkflow_project_id", "comfyworkflow", ["project_id"])


def downgrade() -> None:
    op.drop_table("comfyworkflow")
