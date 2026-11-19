"""Agentic memory store with optional FTS5 index.

Revision ID: 0012_agent_memory
Revises: 0011_comfy_workflows
Create Date: 2026-11-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0012_agent_memory"
down_revision = "0011_comfy_workflows"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = inspect(op.get_bind()).get_table_names()
    if "agentmemory" not in tables:
        op.create_table(
            "agentmemory",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("project_id", sa.String(), nullable=False),
            sa.Column("scope", sa.String(), nullable=False, server_default="project"),
            sa.Column("shot_id", sa.String(), nullable=True),
            sa.Column("kind", sa.String(), nullable=False, server_default="semantic"),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("metadata_json", sa.Text(), nullable=True),
            sa.Column("provenance", sa.String(), nullable=False, server_default="user"),
            sa.Column("trust", sa.String(), nullable=False, server_default="user"),
            sa.Column("content_hash", sa.String(), nullable=False, server_default=""),
            sa.Column("pinned", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("quarantined", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("last_used_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_agentmemory_project_id", "agentmemory", ["project_id"])
        op.create_index("ix_agentmemory_shot_id", "agentmemory", ["shot_id"])
        op.create_index(
            "ix_agentmemory_project_hash",
            "agentmemory",
            ["project_id", "content_hash"],
        )

    # FTS5 is compiled into the bundled SQLite and into macOS Python builds. The
    # virtual table is an accelerator only: retrieval falls back to LIKE when it
    # is missing, so a build without FTS5 still works.
    try:
        op.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS agentmemory_fts "
            "USING fts5(text, content='agentmemory', content_rowid='rowid')"
        )
        op.execute(
            "CREATE TRIGGER IF NOT EXISTS agentmemory_ai AFTER INSERT ON agentmemory BEGIN "
            "INSERT INTO agentmemory_fts(rowid, text) VALUES (new.rowid, new.text); END"
        )
        op.execute(
            "CREATE TRIGGER IF NOT EXISTS agentmemory_ad AFTER DELETE ON agentmemory BEGIN "
            "INSERT INTO agentmemory_fts(agentmemory_fts, rowid, text) "
            "VALUES ('delete', old.rowid, old.text); END"
        )
        op.execute(
            "CREATE TRIGGER IF NOT EXISTS agentmemory_au AFTER UPDATE ON agentmemory BEGIN "
            "INSERT INTO agentmemory_fts(agentmemory_fts, rowid, text) "
            "VALUES ('delete', old.rowid, old.text); "
            "INSERT INTO agentmemory_fts(rowid, text) VALUES (new.rowid, new.text); END"
        )
    except Exception:
        pass


def downgrade() -> None:
    for name in ("agentmemory_ai", "agentmemory_ad", "agentmemory_au"):
        op.execute(f"DROP TRIGGER IF EXISTS {name}")
    op.execute("DROP TABLE IF EXISTS agentmemory_fts")
    op.drop_table("agentmemory")
