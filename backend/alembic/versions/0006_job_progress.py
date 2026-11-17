"""Add job stage, start time, and memory trace.

Revision ID: 0006_job_progress
Revises: 0005_app_settings
Create Date: 2026-11-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0006_job_progress"
down_revision = "0005_app_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("job")}
    if "stage" not in columns:
        op.add_column("job", sa.Column("stage", sa.String(), nullable=True))
    if "started_at" not in columns:
        op.add_column("job", sa.Column("started_at", sa.DateTime(), nullable=True))
    if "memory_json" not in columns:
        op.add_column("job", sa.Column("memory_json", sa.String(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("job")}
    if "memory_json" in columns:
        op.drop_column("job", "memory_json")
    if "started_at" in columns:
        op.drop_column("job", "started_at")
    if "stage" in columns:
        op.drop_column("job", "stage")
