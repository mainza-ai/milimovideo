"""Agentic memory: project-scoped notes, decisions, feedback, and preferences.

This is deliberately not a vector database. The studio's memory needs are small,
structured, and auditable: a fact, who said it, when, which shot it belongs to,
and why it was retrieved. Retrieval is lexical (FTS5 when available, token LIKE
as the fallback) with recency, pinning, trust, and scope signals, packed into a
token budget.

Security model (OWASP ASI06): every entry carries provenance and trust. Agent and
imported text is scanned for instruction-injection patterns and quarantined
instead of trusted; quarantined entries never enter a context pack until a human
approves them.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import text
from sqlmodel import Session, select

from database import AgentMemory, engine

SCOPES = {"project", "shot", "session", "agent"}
KINDS = {"working", "episodic", "semantic", "procedural"}
TRUST_LEVELS = {"user", "verified", "agent", "imported"}
PROVENANCE = {"user", "agent", "imported", "system"}

MAX_TEXT_CHARS = 8_000
DEFAULT_BUDGET_TOKENS = 900

# Instruction-injection signatures. These do not prove malice, but agent or
# imported text matching them must be reviewed before it can steer a prompt.
_POISON_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all )?(previous|prior|above) instructions",
        r"disregard (the )?(system|developer) (prompt|message)",
        r"you are now (a|an|the) ",
        r"reveal (the )?(system|developer) prompt",
        r"print (the )?api[ _-]?key",
        r"exfiltrate|send .* to https?://",
        r"act as (root|admin|developer mode)",
    )
]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _hash(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def _clean(value: str) -> str:
    value = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(ch for ch in value if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return value.strip()


def _looks_injected(value: str) -> bool:
    return any(pattern.search(value) for pattern in _POISON_PATTERNS)


# Function words carry no retrieval signal and otherwise outrank real nouns
# (every fact contains "the"/"is"). Dropping them is the cheapest precision win.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "how", "why", "where", "when",
    "of", "for", "to", "in", "on", "at", "by", "as", "from", "than", "with",
    "and", "or", "but", "do", "does", "did", "this", "that", "these", "those",
    "it", "its", "we", "our", "you", "your", "i", "me", "my", "use", "used",
    "about", "into", "over", "under", "any", "all", "there", "here",
}

# Studio vocabulary: the question uses a verb the note does not. Expansion is
# additive (the original token still scores) and stays small on purpose.
_SYNONYMS = {
    "dislike": ["reject"],
    "disliked": ["reject"],
    "reviewer": ["reject", "review"],
    "rejected": ["reject"],
    "rule": ["never", "no more than", "always"],
    "wearing": ["wear", "wears"],
    "costume": ["wear", "wears", "raincoat"],
    "approved": ["approve", "favorite"],
    "favorite": ["approve"],
    "duration": ["frames"],
}


def _tokens(query: str) -> list[str]:
    words = [
        token
        for token in re.findall(r"[a-z0-9_@.-]{2,}", (query or "").lower())
        if token not in _STOPWORDS
    ]
    expanded: list[str] = []
    for token in words:
        expanded.append(token)
        expanded.extend(_SYNONYMS.get(token, []))
    # De-duplicate while preserving order, then cap the FTS/scan size.
    seen: set[str] = set()
    unique = [token for token in expanded if not (token in seen or seen.add(token))]
    return unique[:16]


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)


def _has_fts(db: Session) -> bool:
    try:
        row = db.exec(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='agentmemory_fts'")
        ).first()
        return bool(row)
    except Exception:
        return False


def _json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _row(entry: AgentMemory, why: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": entry.id,
        "project_id": entry.project_id,
        "scope": entry.scope,
        "shot_id": entry.shot_id,
        "kind": entry.kind,
        "text": entry.text,
        "metadata": _json(entry.metadata_json),
        "provenance": entry.provenance,
        "trust": entry.trust,
        "pinned": bool(entry.pinned),
        "quarantined": bool(entry.quarantined),
        "use_count": entry.use_count,
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
        "last_used_at": entry.last_used_at.isoformat() if entry.last_used_at else None,
        "tokens": _estimate_tokens(entry.text),
        "why": why or [],
    }


def remember(
    db: Session,
    project_id: str,
    text_value: str,
    *,
    scope: str = "project",
    kind: str = "semantic",
    shot_id: str | None = None,
    provenance: str = "user",
    trust: str = "user",
    metadata: dict[str, Any] | None = None,
    pinned: bool = False,
) -> dict[str, Any]:
    cleaned = _clean(text_value)
    if not cleaned:
        raise ValueError("memory text is empty")
    if len(cleaned) > MAX_TEXT_CHARS:
        raise ValueError(f"memory text exceeds {MAX_TEXT_CHARS} characters")
    scope = scope if scope in SCOPES else "project"
    kind = kind if kind in KINDS else "semantic"
    provenance = provenance if provenance in PROVENANCE else "user"
    trust = trust if trust in TRUST_LEVELS else "agent"

    digest = _hash(cleaned)
    existing = db.exec(
        select(AgentMemory)
        .where(AgentMemory.project_id == project_id)
        .where(AgentMemory.content_hash == digest)
    ).first()
    if existing is not None:
        existing.updated_at = _now()
        existing.use_count += 1
        if pinned:
            existing.pinned = True
        if metadata:
            merged = json.loads(existing.metadata_json) if existing.metadata_json else {}
            merged.update(metadata)
            existing.metadata_json = json.dumps(merged)
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return _row(existing, ["duplicate"])

    quarantined = provenance in {"agent", "imported"} and (
        trust == "imported" or _looks_injected(cleaned)
    )
    entry = AgentMemory(
        project_id=project_id,
        scope=scope,
        shot_id=shot_id if scope == "shot" else None,
        kind=kind,
        text=cleaned,
        metadata_json=json.dumps(metadata or {}) if metadata else None,
        provenance=provenance,
        trust=trust,
        content_hash=digest,
        pinned=bool(pinned),
        quarantined=quarantined,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return _row(
        entry,
        ["quarantined:instruction-injection"] if quarantined else ["created"],
    )


def _fts_ids(db: Session, query: str, limit: int) -> list[str]:
    tokens = _tokens(query)
    if not tokens:
        return []
    match = " OR ".join(f'"{token}"' for token in tokens)
    rows = db.exec(
        text(
            "SELECT a.id FROM agentmemory a JOIN agentmemory_fts f ON f.rowid = a.rowid "
            "WHERE f MATCH :match LIMIT :limit"
        ),
        params={"match": match, "limit": limit},
    ).all()
    return [row[0] for row in rows]


def _score(
    entry: AgentMemory,
    tokens: Iterable[str],
    *,
    shot_id: str | None,
    now: datetime,
) -> tuple[float, list[str]]:
    why: list[str] = []
    score = 1.0
    lowered = entry.text.lower()
    for token in tokens:
        hits = lowered.count(token)
        if hits:
            score += 2.0 + min(3, hits)
            why.append(f"match:{token}")
    if entry.pinned:
        score += 5.0
        why.append("pinned")
    if entry.trust in {"user", "verified"}:
        score += 1.0
        why.append(f"trust:{entry.trust}")
    if shot_id and entry.shot_id == shot_id:
        score += 2.0
        why.append("same-shot")
    updated = entry.updated_at or entry.created_at or now
    age_days = max(0.0, (now - updated).total_seconds() / 86_400)
    recency = max(0.0, 1.0 - age_days / 30.0) * 2.0
    if recency > 0:
        score += recency
        why.append("recent")
    return score, why


def search(
    db: Session,
    project_id: str,
    query: str = "",
    *,
    kind: str | None = None,
    scope: str | None = None,
    shot_id: str | None = None,
    include_quarantined: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    statement = select(AgentMemory).where(AgentMemory.project_id == project_id)
    if kind:
        statement = statement.where(AgentMemory.kind == kind)
    if scope:
        statement = statement.where(AgentMemory.scope == scope)
    if shot_id:
        statement = statement.where(AgentMemory.shot_id == shot_id)
    if not include_quarantined:
        statement = statement.where(AgentMemory.quarantined == False)  # noqa: E712
    candidates = list(db.exec(statement).all())

    tokens = _tokens(query)
    now = _now()
    if query and tokens:
        matched: set[str] = set()
        if _has_fts(db):
            try:
                matched = set(_fts_ids(db, query, max(limit * 5, 50)))
            except Exception:
                matched = set()
        scored = []
        for entry in candidates:
            if matched and entry.id not in matched:
                # FTS found better candidates; drop non-matches only when the
                # query produced hits, otherwise LIKE/token scoring below wins.
                continue
            score, why = _score(entry, tokens, shot_id=shot_id, now=now)
            if score > 1.5:
                scored.append((score, entry, why))
        if not scored:
            for entry in candidates:
                score, why = _score(entry, tokens, shot_id=shot_id, now=now)
                if score > 1.5:
                    scored.append((score, entry, why))
        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [_row(entry, why) for _score, entry, why in scored[:limit]]

    candidates.sort(
        key=lambda entry: (entry.pinned, entry.updated_at or entry.created_at),
        reverse=True,
    )
    return [_row(entry) for entry in candidates[:limit]]


def context_pack(
    db: Session,
    project_id: str,
    query: str = "",
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    shot_id: str | None = None,
    include_quarantined: bool = False,
) -> dict[str, Any]:
    budget = max(64, min(int(budget_tokens or DEFAULT_BUDGET_TOKENS), 8000))
    rows = search(
        db,
        project_id,
        query,
        shot_id=shot_id,
        include_quarantined=include_quarantined,
        limit=80,
    )
    packed: list[dict[str, Any]] = []
    used = 0
    dropped = 0
    for row in rows:
        cost = row["tokens"]
        if used + cost > budget:
            dropped += 1
            continue
        packed.append(row)
        used += cost
        if used >= budget:
            break
    if packed:
        now = _now()
        for row in packed:
            entry = db.get(AgentMemory, row["id"])
            if entry is None:
                continue
            entry.use_count += 1
            entry.last_used_at = now
            db.add(entry)
        db.commit()
    return {
        "project_id": project_id,
        "query": query,
        "budget_tokens": budget,
        "used_tokens": used,
        "entries": packed,
        "dropped": dropped,
    }


def update(
    db: Session,
    project_id: str,
    memory_id: str,
    *,
    text_value: str | None = None,
    pinned: bool | None = None,
    trust: str | None = None,
    kind: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    entry = db.get(AgentMemory, memory_id)
    if entry is None or entry.project_id != project_id:
        return None
    if text_value is not None:
        cleaned = _clean(text_value)
        if not cleaned:
            raise ValueError("memory text is empty")
        if len(cleaned) > MAX_TEXT_CHARS:
            raise ValueError(f"memory text exceeds {MAX_TEXT_CHARS} characters")
        entry.text = cleaned
        entry.content_hash = _hash(cleaned)
    if pinned is not None:
        entry.pinned = bool(pinned)
    if trust is not None:
        entry.trust = trust if trust in TRUST_LEVELS else entry.trust
    if kind is not None:
        entry.kind = kind if kind in KINDS else entry.kind
    if metadata is not None:
        entry.metadata_json = json.dumps(metadata)
    entry.updated_at = _now()
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return _row(entry, ["updated"])


def approve(db: Session, project_id: str, memory_id: str) -> dict[str, Any] | None:
    entry = db.get(AgentMemory, memory_id)
    if entry is None or entry.project_id != project_id:
        return None
    entry.quarantined = False
    entry.trust = "verified"
    entry.updated_at = _now()
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return _row(entry, ["approved"])


def forget(db: Session, project_id: str, memory_id: str) -> bool:
    entry = db.get(AgentMemory, memory_id)
    if entry is None or entry.project_id != project_id:
        return False
    db.delete(entry)
    db.commit()
    return True


def clear_project(db: Session, project_id: str) -> int:
    rows = list(
        db.exec(select(AgentMemory).where(AgentMemory.project_id == project_id)).all()
    )
    for entry in rows:
        db.delete(entry)
    db.commit()
    return len(rows)


def export_project(db: Session, project_id: str) -> dict[str, Any]:
    rows = db.exec(
        select(AgentMemory).where(AgentMemory.project_id == project_id)
    ).all()
    return {
        "project_id": project_id,
        "exported_at": _now().isoformat(),
        "entries": [_row(entry) for entry in rows],
    }


def prune(
    db: Session,
    *,
    older_than_days: int = 90,
    kinds: tuple[str, ...] = ("working", "episodic"),
) -> int:
    """TTL/decay pass: drop unpinned working/episodic notes that went stale."""
    cutoff = _now() - timedelta(days=max(1, older_than_days))
    rows = list(
        db.exec(
            select(AgentMemory)
            .where(AgentMemory.kind.in_(kinds))
            .where(AgentMemory.pinned == False)  # noqa: E712
            .where(AgentMemory.quarantined == False)  # noqa: E712
            .where(AgentMemory.updated_at < cutoff)
        ).all()
    )
    for entry in rows:
        db.delete(entry)
    if rows:
        db.commit()
    return len(rows)
