"""Baseline: current Milimo Video schema.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-17

The baseline uses SQLModel metadata so it exactly matches the models at the time
Alembic was adopted. Future changes must be separate revisions; do not edit this
baseline after it ships.
"""

from alembic import op
from sqlmodel import SQLModel

# Importing registers every table on the metadata.
import database  # noqa: F401

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    SQLModel.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    SQLModel.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
