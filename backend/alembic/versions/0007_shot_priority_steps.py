"""Add per-shot generation steps and queue priority.

Revision ID: 0007_shot_priority_steps
Revises: 0006_job_progress
Create Date: 2026-11-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0007_shot_priority_steps"
down_revision = "0006_job_progress"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("shot")}
    if "num_inference_steps" not in columns:
        op.add_column("shot", sa.Column("num_inference_steps", sa.Integer(), nullable=False, server_default="40"))
    if "priority" not in columns:
        op.add_column("shot", sa.Column("priority", sa.Integer(), nullable=False, server_default="50"))


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("shot")}
    if "priority" in columns:
        op.drop_column("shot", "priority")
    if "num_inference_steps" in columns:
        op.drop_column("shot", "num_inference_steps")
