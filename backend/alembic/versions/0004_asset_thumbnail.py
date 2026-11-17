"""Add uploaded-media thumbnail URLs.

Revision ID: 0004_asset_thumbnail
Revises: 0003_seed_lock
Create Date: 2026-11-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0004_asset_thumbnail"
down_revision = "0003_seed_lock"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("asset")}
    if "thumbnail_url" not in columns:
        op.add_column("asset", sa.Column("thumbnail_url", sa.String(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in inspect(op.get_bind()).get_columns("asset")}
    if "thumbnail_url" in columns:
        op.drop_column("asset", "thumbnail_url")
