"""Model capability classification shared by catalogs, discovery, and defaults."""

from __future__ import annotations

import re
from typing import Any, Iterable

# Families that cannot serve a chat role. Kept deliberately explicit: a false
# "chat" costs a bad role assignment, a false "non-chat" hides a usable model.
# Short tokens use separators so "fable5-composer" is not read as "e5-".
EMBEDDING_PATTERNS = (
    r"embed", r"nomic-bert", r"minilm",
    r"(^|[-_:/])bge([-_:/]|$)", r"(^|[-_:/])gte([-_:/]|$)", r"(^|[-_:/])e5([-_:/]|$)",
)
NON_CHAT_PATTERNS = (
    r"whisper", r"\btts\b", r"music", r"audio-gen", r"rerank", r"markitdown",
    r"clip-vit", r"stable-diffusion", r"controlnet", r"upscal", r"real-esrgan",
    r"depth-anything", r"(^|[-_:/])sam([-_:/]|$)", r"segment",
    r"(^|[-_:/])sdxl([-_:/]|$)", r"(^|[-_:/])sd3([-_:/]|$)", r"flux-",
)
VISION_PATTERNS = (
    r"vision", r"llava", r"mllama", r"pixtral", r"(^|[-_:/])vl([-_:/]|$)",
    r"qwen2-vl", r"qwen2\.5-vl", r"qwen3-vl", r"gemma3", r"minicpm-v",
    r"moondream", r"internvl", r"mmproj", r"multimodal",
)


def _matches(blob: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, blob) for pattern in patterns)

CONTEXT_KEYS = (
    "context_length", "max_model_len", "max_context_length",
    "context_window", "max_context", "n_ctx", "num_ctx",
)


def _blob(name: str, family: str, families: Iterable[str], extra: Iterable[str]) -> str:
    return " ".join([name or "", family or "", *[f or "" for f in families], *[e or "" for e in extra]]).lower()


def classify_model(
    name: str,
    *,
    family: str = "",
    families: Iterable[str] = (),
    extra: Iterable[str] = (),
    declared_vision: bool = False,
) -> dict[str, bool]:
    blob = _blob(name, family, families, extra)
    embedding = _matches(blob, EMBEDDING_PATTERNS)
    non_chat = _matches(blob, NON_CHAT_PATTERNS)
    vision = bool(declared_vision) or _matches(blob, VISION_PATTERNS)
    return {
        "embedding_only": embedding,
        "vision": vision and not embedding,
        "chat": not embedding and not non_chat,
    }


_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")
_UNWANTED = ("uncensored", "abliterated", "roleplay", "erp")


def parameter_billions(name: str) -> float | None:
    matches = _PARAM_RE.findall(name or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def pick_default_model(models: Iterable[dict[str, Any]]) -> str:
    """Choose a sensible chat default for a provider's catalog.

    Prefers a mid-size instruct/chat model and penalises uncensored variants and
    non-chat entries. Returns "" when nothing is chat-capable.
    """
    best: tuple[float, str] | None = None
    for index, model in enumerate(models):
        model_id = str(model.get("id") or "")
        if not model_id or not model.get("chat", True):
            continue
        blob = model_id.lower()
        penalty = 100.0 if any(word in blob for word in _UNWANTED) else 0.0
        size = parameter_billions(model_id)
        if size is None:
            size = model.get("parameter_size")
            if isinstance(size, str):
                size = parameter_billions(size)
        size = float(size) if isinstance(size, (int, float)) and size > 0 else 13.0
        tuned = (
            "instruct" in blob
            or "chat" in blob
            or "aligned" in blob
            or bool(re.search(r"(^|[-_:/])(it|instruct)([-_:/]|$)", blob))
        )
        instruct = 0.0 if tuned else 1.5
        score = penalty + abs(size - 8.0) + instruct + index * 0.01
        candidate = (score, model_id)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1] if best else ""


def extract_context(row: dict[str, Any], meta: dict[str, Any] | None = None) -> int | None:
    """Read a context window from the many shapes local engines use."""
    meta = meta or {}
    for source in (row, meta):
        for key in CONTEXT_KEYS:
            value = source.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
    return None
