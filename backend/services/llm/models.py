"""Typed LLM settings, providers, capabilities, and film-production roles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


SCHEMA_VERSION = 2

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
    # `id` mirrors the providers-dict key; it is the stable instance identity.
    id: str = ""
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
    # Empty provider means the role is explicitly unassigned.
    provider: str = ""
    model: str = ""
    fallback: List[LLMModelTarget] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None

    @property
    def assigned(self) -> bool:
        return bool(self.provider and self.model)


class LLMBudgetConfig(BaseModel):
    # 0 disables cloud spend entirely; local calls are never budgeted.
    daily_usd: float = 0.0
    # When true, enabling or assigning a cloud model requires a passing eval.
    eval_gate: bool = False


class LLMEvalSummary(BaseModel):
    completed_at: str
    revision: int
    pass_rate: float
    p50_ms: int
    cases: int = 0
    roles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LLMSettings(BaseModel):
    schema_version: int = SCHEMA_VERSION
    revision: int = 0
    active_provider: str = "gemma"
    active_model: str = "ltx-gemma3"
    air_gapped: bool = True
    temperature: float = 0.7
    max_tokens: int = 400
    providers: Dict[str, LLMProviderConfig] = Field(default_factory=dict)
    roles: Dict[str, LLMRoleAssignment] = Field(default_factory=dict)
    budget: LLMBudgetConfig = Field(default_factory=LLMBudgetConfig)
    last_eval: Optional[LLMEvalSummary] = None


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
    needs_key: bool = False


class LLMDiscoverAttach(BaseModel):
    provider: str
    base_url: str
    name: str = ""
    wire: str = "openai_chat"
    models: List[str] = Field(default_factory=list)


class LLMEvalRequest(BaseModel):
    roles: List[str] = Field(default_factory=list)
    limit: int = 2
