"""LLM configuration plane: persistence, revision CAS, redaction, routing."""

import copy

import pytest

from services.llm import providers, runtime
from services.llm.defaults import default_settings
from services.llm.store import llm_settings


@pytest.fixture()
def fresh_settings(app_client):
    """Reset the singleton so each test loads from the test database."""
    llm_settings._settings = None
    yield
    llm_settings._settings = None


def _snapshot(client):
    response = client.get("/settings/llm")
    assert response.status_code == 200, response.text
    return response.json()


def _put(client, body, expected_revision=None):
    params = {} if expected_revision is None else {"expected_revision": expected_revision}
    return client.put("/settings/llm", params=params, json=body)


def test_settings_are_persisted_across_reload(app_client, fresh_settings):
    body = _snapshot(app_client)
    body["providers"]["ollama"]["default_model"] = "qwen3:8b"
    body["roles"]["script_parse"]["model"] = "qwen3:8b"
    body["active_model"] = "qwen3:8b"
    saved = _put(app_client, body)
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == body["revision"] + 1

    # Simulate a process restart: drop the in-memory copy, re-read the row.
    llm_settings._settings = None
    reloaded = _snapshot(app_client)
    assert reloaded["providers"]["ollama"]["default_model"] == "qwen3:8b"
    assert reloaded["roles"]["script_parse"]["model"] == "qwen3:8b"
    assert reloaded["revision"] == body["revision"] + 1


def test_stale_write_returns_409_with_current_settings(app_client, fresh_settings):
    body = _snapshot(app_client)
    first = _put(app_client, body, expected_revision=body["revision"])
    assert first.status_code == 200, first.text

    stale = _put(app_client, body, expected_revision=body["revision"])
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert detail["current_revision"] if "current_revision" in detail else detail["revision"] == body["revision"] + 1
    assert detail["settings"]["revision"] == body["revision"] + 1


def test_unknown_provider_role_and_model_are_rejected(app_client, fresh_settings):
    body = _snapshot(app_client)

    bad_provider = copy.deepcopy(body)
    bad_provider["active_provider"] = "not-a-provider"
    response = _put(app_client, bad_provider)
    assert response.status_code == 400
    assert "active provider" in response.json()["detail"]

    bad_role = copy.deepcopy(body)
    bad_role["roles"]["prompt_video"]["role"] = "director"
    response = _put(app_client, bad_role)
    assert response.status_code == 400
    assert "role" in response.json()["detail"]

    bad_model = copy.deepcopy(body)
    bad_model["roles"]["prompt_video"]["model"] = "  "
    response = _put(app_client, bad_model)
    assert response.status_code == 400
    assert "model" in response.json()["detail"]


def test_remote_plaintext_http_is_refused(app_client, fresh_settings):
    body = _snapshot(app_client)
    body["providers"]["openai_compatible"]["base_url"] = "http://example.com/v1"
    response = _put(app_client, body)
    assert response.status_code == 400
    assert "plaintext" in response.json()["detail"]


def test_api_keys_are_redacted_from_reads(app_client, fresh_settings):
    body = _snapshot(app_client)
    body["providers"]["ollama"]["api_key"] = "local-secret"
    saved = _put(app_client, body)
    assert saved.status_code == 200, saved.text
    assert saved.json()["providers"]["ollama"]["api_key"] is None

    public = _snapshot(app_client)
    assert public["providers"]["ollama"]["api_key"] is None
    assert public["providers"]["ollama"]["has_key"] is True
    assert public["providers"]["ollama"]["secret_source"] in {"plaintext_db", "sealed_db"}
    assert "local-secret" not in str(public)

    cleared = app_client.delete("/settings/llm/providers/ollama/key")
    assert cleared.status_code == 200
    assert cleared.json()["providers"]["ollama"]["has_key"] is False


def test_air_gap_patch_is_field_level_only(app_client, fresh_settings):
    body = _snapshot(app_client)
    response = app_client.patch("/settings/llm", json={"air_gapped": False})
    assert response.status_code == 200
    assert response.json()["air_gapped"] is False
    assert response.json()["revision"] == body["revision"] + 1

    rejected = app_client.patch("/settings/llm", json={"active_provider": "ollama"})
    assert rejected.status_code == 400


def test_cloud_provider_is_refused_when_not_enabled(app_client, fresh_settings, monkeypatch):
    monkeypatch.delenv("MILIMO_LLM_CLOUD", raising=False)
    body = _snapshot(app_client)
    body["air_gapped"] = False
    body["providers"]["openai"]["enabled"] = True
    body["providers"]["openai"]["default_model"] = "gpt-x"
    body["providers"]["openai"]["api_key"] = "cloud-secret"
    body["roles"]["agent_plan"]["provider"] = "openai"
    body["roles"]["agent_plan"]["model"] = "gpt-x"
    saved = _put(app_client, body)
    assert saved.status_code == 200, saved.text

    plan = runtime.resolve("agent_plan")
    assert plan["usable"] is False
    assert "cloud is disabled" in plan["reason"]


def test_air_gap_blocks_cloud_even_when_cloud_is_enabled(app_client, fresh_settings, monkeypatch):
    monkeypatch.setenv("MILIMO_LLM_CLOUD", "1")
    body = _snapshot(app_client)
    body["air_gapped"] = True
    body["providers"]["openai"]["enabled"] = True
    body["providers"]["openai"]["default_model"] = "gpt-x"
    body["roles"]["agent_plan"]["provider"] = "openai"
    body["roles"]["agent_plan"]["model"] = "gpt-x"
    assert _put(app_client, body).status_code == 200

    plan = runtime.resolve("agent_plan")
    assert plan["usable"] is False
    assert "air-gap" in plan["reason"]


def test_default_settings_follow_legacy_environment(monkeypatch):
    monkeypatch.setenv("MILIMO_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("MILIMO_OLLAMA_MODEL", "gemma3:12b")
    monkeypatch.setenv("MILIMO_OLLAMA_URL", "http://127.0.0.1:11434")
    settings = default_settings()
    assert settings.active_provider == "ollama"
    assert settings.active_model == "gemma3:12b"
    assert settings.providers["ollama"].default_model == "gemma3:12b"
    assert settings.roles["script_parse"].provider == "ollama"


def test_discovery_returns_live_servers(app_client, monkeypatch):
    from services.llm import discovery

    monkeypatch.setattr(
        discovery,
        "discover_local_servers",
        lambda: [{
            "provider": "ollama",
            "name": "Ollama",
            "wire": "ollama_native",
            "base_url": "http://127.0.0.1:11434",
            "port": 11434,
            "models": ["qwen3:8b"],
            "source": "loopback",
        }],
    )
    response = app_client.post("/settings/llm/discover")
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["servers"][0]["port"] == 11434


def test_verify_provider_reports_models_and_latency(app_client, fresh_settings, monkeypatch):
    monkeypatch.setattr(
        providers,
        "list_models",
        lambda **kwargs: [{
            "id": "qwen3:8b", "name": "qwen3:8b", "size": 1, "modified_at": None,
            "parameter_size": "8B", "quantization": "Q4", "family": "qwen3",
            "vision": False, "context_length": None,
        }],
    )
    response = app_client.post("/settings/llm/providers/ollama/verify")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["models"] == ["qwen3:8b"]


def test_role_fallback_uses_the_next_hop(app_client, fresh_settings, monkeypatch):
    body = _snapshot(app_client)
    body["roles"]["script_parse"]["provider"] = "ollama"
    body["roles"]["script_parse"]["model"] = "primary"
    body["roles"]["script_parse"]["fallback"] = [{"provider": "lmstudio", "model": "backup"}]
    body["providers"]["lmstudio"]["default_model"] = "backup"
    body["providers"]["lmstudio"]["enabled"] = True
    assert _put(app_client, body).status_code == 200

    def fake_complete(**kwargs):
        if kwargs["provider"] == "ollama":
            raise providers.ProviderError("ollama", "unreachable", "connection refused")
        return providers.Completion(
            text="fallback reply", provider=kwargs["provider"], model=kwargs["model"], latency_ms=7
        )

    monkeypatch.setattr(providers, "complete", fake_complete)
    result = runtime.complete("script_parse", "hello")
    assert result.provider == "lmstudio"
    assert result.model == "backup"
    assert result.text == "fallback reply"


def test_enhance_prompt_returns_original_when_every_hop_fails(app_client, fresh_settings, monkeypatch):
    import llm

    monkeypatch.setattr(providers, "complete", lambda **kwargs: (_ for _ in ()).throw(
        providers.ProviderError(kwargs["provider"], "unreachable", "down")
    ))
    enhanced, is_reference = llm.enhance_prompt("A slow dolly in on the hero.", is_video=True)
    assert enhanced == "A slow dolly in on the hero."
    assert is_reference is False
    assert llm.last_enhancement()["failed"] is True


def test_ai_parse_runs_without_a_text_encoder(app_client, fresh_settings, monkeypatch):
    from services.ai_storyboard import ai_parse_script

    reply = (
        '[{"name": "Scene 1", "shots": ['
        '{"action": "A filmmaker edits at night", "shot_type": "wide"}]}]'
    )

    def fake_complete(role, prompt, **kwargs):
        assert role == "script_parse"
        assert kwargs.get("text_encoder") is None
        return runtime.LLMResult(
            text=reply, provider="ollama", model="qwen3:8b", role=role, latency_ms=12
        )

    monkeypatch.setattr(runtime, "complete", fake_complete)
    scenes = ai_parse_script("INT. STUDIO - NIGHT\n\nA filmmaker edits.", text_encoder=None)
    assert scenes[0]["name"] == "Scene 1"
    assert scenes[0]["shots"][0]["action"] == "A filmmaker edits at night"


def test_gemma_has_no_standalone_test_endpoint(app_client, fresh_settings):
    response = app_client.post("/settings/llm/test", json={
        "role": "prompt_video", "provider": "gemma", "prompt": "ping",
    })
    assert response.status_code == 400
    assert "standalone endpoint" in response.json()["detail"]


def test_enhancement_records_provider_provenance(app_client, fresh_settings, monkeypatch):
    import llm

    monkeypatch.setattr(
        providers,
        "complete_gemma",
        lambda *args, **kwargs: '{"enhanced_prompt": "A steadier push-in.", "is_reference_only": false}',
    )
    enhanced, is_reference = llm.enhance_prompt("push in", is_video=True, text_encoder=object())
    assert enhanced == "A steadier push-in."
    assert is_reference is False

    provenance = llm.last_enhancement()
    assert provenance["role"] == "prompt_video"
    assert provenance["provider"] == "gemma"
    assert provenance["model"] == "ltx-gemma3"


def test_job_params_capture_llm_provenance(app_client, fresh_settings, monkeypatch):
    import json

    import llm
    import job_utils
    from database import Job, engine
    from sqlmodel import Session

    monkeypatch.setattr(llm, "last_enhancement", lambda: {
        "role": "prompt_image", "provider": "ollama", "model": "qwen3:8b", "latency_ms": 42,
    })

    with Session(engine) as session:
        session.add(Job(id="job-llm-provenance", type="image", status="processing", params_json="{}"))
        session.commit()

    job_utils.update_job_db("job-llm-provenance", "processing", enhanced_prompt="A neon alley.")

    with Session(engine) as session:
        job = session.get(Job, "job-llm-provenance")
        params = json.loads(job.params_json)
    assert params["llm"]["provider"] == "ollama"
    assert params["llm"]["model"] == "qwen3:8b"
    assert params["llm"]["role"] == "prompt_image"


def test_ollama_payload_uses_native_json_and_credentials_stay_in_headers():
    messages = [
        providers.ChatMessage(role="system", content="system text"),
        providers.ChatMessage(role="user", content="user text"),
    ]
    payload = providers._ollama_payload("qwen3:8b", messages, {"temperature": 0.2}, "0", True)
    assert payload["format"] == "json"
    assert payload["keep_alive"] == "0"
    assert payload["system"] == "system text"
    assert payload["prompt"] == "user text"
    assert payload["options"] == {"temperature": 0.2}

    assert providers._openai_headers("secret")["Authorization"] == "Bearer secret"
    assert providers._anthropic_headers("secret")["x-api-key"] == "secret"


def test_prompt_video_requests_json_mode_from_ollama(app_client, fresh_settings, monkeypatch):
    import llm

    body = _snapshot(app_client)
    body["roles"]["prompt_video"]["provider"] = "ollama"
    body["roles"]["prompt_video"]["model"] = "qwen3:8b"
    assert _put(app_client, body).status_code == 200

    captured: dict = {}

    def fake_complete(**kwargs):
        captured.update(kwargs)
        return providers.Completion(
            text='{"enhanced_prompt": "A slow dolly in.", "is_reference_only": false}',
            provider=kwargs["provider"], model=kwargs["model"], latency_ms=5,
        )

    monkeypatch.setattr(providers, "complete", fake_complete)
    enhanced, _ = llm.enhance_prompt("dolly in", is_video=True)
    assert enhanced == "A slow dolly in."
    assert captured["provider"] == "ollama"
    assert captured["json_mode"] is True
    assert captured["keep_alive"] == "0"
    assert captured["base_url"] == "http://127.0.0.1:11434"

    # A caller-supplied Director prompt asks for prose, not the JSON contract.
    llm.enhance_prompt("continue the scene", is_video=True, system_prompt="Output only the prompt.")
    assert captured["json_mode"] is False


def test_ai_parse_accepts_a_wrapped_scene_array(app_client, fresh_settings):
    from services.ai_storyboard import _extract_json

    assert _extract_json('{"scenes": [{"name": "Scene 1"}]}') == [{"name": "Scene 1"}]
    assert _extract_json('[{"name": "Scene 2"}]') == [{"name": "Scene 2"}]
