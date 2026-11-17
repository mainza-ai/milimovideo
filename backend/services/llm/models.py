"""Typed LLM settings, providers, and film-production roles."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


PROVIDER_TYPES = {
    "gemma", "ollama", "lmstudio", "openai_compatible", "openai", "anthropic",
}
WIRE_TYPES = {
    "gemma_native", "ollama_native", "openai_chat", "anthropic_messages",
}
ROLE_IDS = {
    "prompt_video", "prompt_image", "script_parse", "agent_plan",
    "review_critic", "continuity",
}


class LLMProviderConfig(BaseModel):
    type: str
    name: str
    base_url: str = ""
    default_model: str = ""
    wire: str = "openai_chat"
    enabled: bool = True
    is_local: bool = True
    timeout_sec: int = 120
    keep_alive: str = "0"

    # Input only; never serialized in a public snapshot.
    api_key: Optional[str] = None
    clear_key: bool = False

    # Public status, recomputed from the credential store.
    has_key: bool = False
    secret_source: str = "none"
    key_status: str = "none"
    last_success_at: Optional[str] = None
    last_fault: Optional[str] = None


class LLMModelTarget(BaseModel):
    provider: str
    model: str


class LLMRoleAssignment(BaseModel):
    role: str
    provider: str
    model: str
    fallback: List[LLMModelTarget] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None


class LLMSettings(BaseModel):
    schema_version: int = 1
    revision: int = 0
    active_provider: str = "gemma"
    active_model: str = "ltx-gemma3"
    air_gapped: bool = True
    temperature: float = 0.7
    max_tokens: int = 400
    providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict)
    roles: Dict[str, LLMRoleAssignment] = Field(default_factory=dict)


class LLMTestRequest(BaseModel):
    role: str = "prompt_video"
    prompt: str = "Confirm you are active and describe the current shot."
    provider: Optional[str] = None
    model: Optional[str] = None


class LLMVerifyResult(BaseModel):
    ok: bool
    provider: str
    wire: str = ""
    latency_ms: int = 0
    models: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    fault: Optional[str] = None
