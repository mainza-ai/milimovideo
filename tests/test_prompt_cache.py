"""Prompt enhancement cache: same prompt/seed/settings reuses one rewrite."""

from __future__ import annotations

import pytest

import llm


class _Settings:
    def __init__(self, revision: int):
        self.revision = revision


class _Result:
    provider = "ollama"
    model = "gemma3"
    latency_ms = 12

    def __init__(self, text: str):
        self.text = text

    def provenance(self) -> dict:
        return {"provider": self.provider, "model": self.model, "role": "prompt_video"}


@pytest.fixture(autouse=True)
def _cache_reset(monkeypatch):
    llm.clear_prompt_cache()
    monkeypatch.setattr(llm.llm_settings, "get", lambda: _Settings(1))
    yield
    llm.clear_prompt_cache()


def test_second_identical_enhancement_hits_the_cache(monkeypatch):
    calls: list[str] = []

    def fake_enhance(role, prompt, **kwargs):
        calls.append(prompt)
        return f"enhanced: {prompt}", False, _Result("ok")

    monkeypatch.setattr(llm.runtime, "enhance", fake_enhance)
    first = llm.enhance_prompt("a rooftop chase", seed=7)
    second = llm.enhance_prompt("a rooftop chase", seed=7)
    assert first == second == ("enhanced: a rooftop chase", False)
    assert len(calls) == 1
    assert llm.last_enhancement().get("cached") is True


def test_seed_and_settings_revision_invalidate(monkeypatch):
    calls: list[str] = []

    def fake_enhance(role, prompt, **kwargs):
        calls.append(prompt)
        return f"enhanced: {prompt}", False, _Result("ok")

    monkeypatch.setattr(llm.runtime, "enhance", fake_enhance)
    llm.enhance_prompt("a rooftop chase", seed=7)
    llm.enhance_prompt("a rooftop chase", seed=8)
    assert len(calls) == 2

    monkeypatch.setattr(llm.llm_settings, "get", lambda: _Settings(2))
    llm.enhance_prompt("a rooftop chase", seed=7)
    assert len(calls) == 3


def test_failed_enhancement_is_not_cached(monkeypatch):
    calls: list[str] = []

    def fake_enhance(role, prompt, **kwargs):
        calls.append(prompt)
        return prompt, False, None

    monkeypatch.setattr(llm.runtime, "enhance", fake_enhance)
    assert llm.enhance_prompt("a rooftop chase") == ("a rooftop chase", False)
    assert llm.enhance_prompt("a rooftop chase") == ("a rooftop chase", False)
    assert len(calls) == 2
    assert llm.last_enhancement().get("failed") is True
