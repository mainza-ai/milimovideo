"""Add versioned application settings.

Revision ID: 0005_app_settings
Revises: 0004_asset_thumbnail
Create Date: 2026-11-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0005_app_settings"
down_revision = "0004_asset_thumbnail"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "app_setting" in set(inspect(op.get_bind()).get_table_names()):
        return
    op.create_table(
        "app_setting",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("value_json", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    if "app_setting" in set(inspect(op.get_bind()).get_table_names()):
        op.drop_table("app_setting")
