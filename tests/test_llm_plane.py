"""L5-L9: instances, capabilities, preflight, eval, budget, profiles, stream."""

import asyncio
import json

import pytest

from services.llm import budget, capabilities, eval as eval_engine, providers, runtime
from services.llm.defaults import build_defaults
from services.llm.store import llm_settings


@pytest.fixture(autouse=True)
def plane(app_client, monkeypatch):
    from sqlmodel import Session

    from database import AppSetting, engine
    from services.llm import discovery as discovery_module

    monkeypatch.setattr(discovery_module, "discover_local_servers", lambda **kwargs: [])
    with Session(engine) as session:
        row = session.get(AppSetting, "llm")
        if row:
            session.delete(row)
            session.commit()
    llm_settings._settings = None
    runtime.clear_catalog()
    yield
    llm_settings._settings = None
    runtime.clear_catalog()


def doc(client):
    response = client.get("/settings/llm")
    assert response.status_code == 200, response.text
    return response.json()


def put(client, body, revision=None):
    params = {} if revision is None else {"expected_revision": revision}
    return client.put("/settings/llm", params=params, json=body)


def attach(client, base_url="http://127.0.0.1:8787/v1", models=("qwen3:8b",), provider="openai_compatible"):
    return client.post("/settings/llm/attach", json={
        "provider": provider,
        "base_url": base_url,
        "name": "Local engine",
        "wire": "openai_chat",
        "models": list(models),
    })


# ── L5: instances ───────────────────────────────────────────────────


def test_two_openai_compatible_instances_coexist(app_client, monkeypatch):
    assert attach(app_client, "http://127.0.0.1:8787/v1", ["omlx-model"]).status_code == 200
    assert attach(app_client, "http://127.0.0.1:47831/v1", ["strata-model"]).status_code == 200

    body = doc(app_client)
    omlx = "openai_compatible@127.0.0.1:8787"
    strata = "openai_compatible@127.0.0.1:47831"
    assert omlx in body["providers"]
    assert strata in body["providers"]
    assert body["providers"][omlx]["base_url"] == "http://127.0.0.1:8787/v1"
    assert body["providers"][strata]["base_url"] == "http://127.0.0.1:47831/v1"

    body["roles"]["script_parse"] = {"role": "script_parse", "provider": omlx, "model": "omlx-model", "fallback": []}
    body["roles"]["agent_plan"] = {"role": "agent_plan", "provider": strata, "model": "strata-model", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200

    assert runtime.resolve("script_parse")["plan"][0]["provider"] == omlx
    assert runtime.resolve("agent_plan")["plan"][0]["provider"] == strata


def test_attach_is_idempotent_by_url(app_client):
    assert attach(app_client).status_code == 200
    first = doc(app_client)
    assert attach(app_client).status_code == 200
    second = doc(app_client)
    assert set(first["providers"]) == set(second["providers"])
    assert len(second["providers"]) == len(first["providers"])


def test_remove_instance_refuses_when_in_use(app_client):
    instance = "openai_compatible@127.0.0.1:8787"
    assert attach(app_client).status_code == 200
    body = doc(app_client)
    body["roles"]["script_parse"] = {"role": "script_parse", "provider": instance, "model": "qwen3:8b", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200

    response = app_client.delete(f"/settings/llm/providers/{instance}")
    assert response.status_code == 400
    assert "unassign" in response.json()["detail"]


def test_v1_document_migrates_to_v2(app_client):
    from sqlmodel import Session

    from database import AppSetting, engine

    legacy = {
        "schema_version": 1,
        "revision": 3,
        "active_provider": "gemma",
        "active_model": "ltx-gemma3",
        "air_gapped": True,
        "temperature": 0.7,
        "max_tokens": 400,
        "providers": {
            "gemma": {"type": "gemma", "name": "Gemma", "wire": "gemma_native", "is_local": True,
                      "default_model": "ltx-gemma3"},
            "ollama": {"type": "ollama", "name": "Ollama", "base_url": "http://127.0.0.1:11434",
                       "wire": "ollama_native", "is_local": True, "default_model": "qwen3:8b"},
        },
        "roles": {
            "prompt_video": {"role": "prompt_video", "provider": "gemma", "model": "ltx-gemma3"},
            "script_parse": {"role": "script_parse", "provider": "ollama", "model": "qwen3:8b"},
        },
    }
    with Session(engine) as session:
        session.add(AppSetting(key="llm", value_json=json.dumps(legacy), updated_at=__import__("datetime").datetime.utcnow()))
        session.commit()
    llm_settings._settings = None

    body = doc(app_client)
    assert body["schema_version"] == 2
    assert body["providers"]["ollama"]["id"] == "ollama"
    assert body["budget"]["daily_usd"] == 0.0
    assert body["revision"] == 3


# ── L5: discovery shape ─────────────────────────────────────────────


def test_discovery_requires_a_catalog_shape():
    from services.llm import discovery

    assert discovery._catalog_shape("openai_chat", {"object": "list", "data": [{"id": "m"}]}) == ["m"]
    assert discovery._catalog_shape("ollama_native", {"models": [{"name": "m"}]}) == ["m"]
    assert discovery._catalog_shape("openai_chat", "<html>hello</html>") is None
    assert discovery._catalog_shape("openai_chat", {"detail": "Not Found"}) is None


def test_probe_rejects_an_unrelated_service():
    import http.server
    import socketserver
    import threading

    from services.llm import discovery

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = b"<html>not an llm</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert discovery._probe(port, "openai_compatible", "Fake", "openai_chat") is None
        finally:
            server.shutdown()


# ── L5: capabilities and preflight ──────────────────────────────────


def test_capability_classification_and_context_shapes():
    assert capabilities.extract_context({"max_model_len": 262144}) == 262144
    assert capabilities.extract_context({}, {"context_length": "8192"}) == 8192
    assert capabilities.extract_context({"detail": "none"}) is None

    assert capabilities.classify_model("nomic-embed-text-v2-moe:latest")["embedding_only"] is True
    assert capabilities.classify_model("MarkItDown")["chat"] is False
    assert capabilities.classify_model("mlx-community--MiniMax-Music3-bf16")["chat"] is False
    assert capabilities.classify_model("Qwen3.8-27B-MLX-8bit")["chat"] is True
    assert capabilities.classify_model("gemma-3-27b-it", declared_vision=True)["vision"] is True


def test_context_preflight_refuses_before_calling_the_provider(app_client, monkeypatch):
    body = doc(app_client)
    body["roles"]["script_parse"] = {"role": "script_parse", "provider": "ollama", "model": "tiny", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200
    runtime.cache_catalog("ollama", [{
        "id": "tiny", "name": "tiny", "chat": True, "embedding_only": False,
        "vision": False, "context_length": 100,
    }])

    def explode(**kwargs):
        raise AssertionError("provider should not be called")

    monkeypatch.setattr(providers, "complete", explode)
    with pytest.raises(Exception) as error:
        runtime.complete("script_parse", "x" * 2000)
    assert "allows 100" in str(error.value)


def test_non_chat_model_is_refused_for_a_text_role(app_client):
    body = doc(app_client)
    body["roles"]["script_parse"] = {"role": "script_parse", "provider": "ollama", "model": "embed", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200
    runtime.cache_catalog("ollama", [{
        "id": "embed", "name": "embed", "chat": False, "embedding_only": True,
        "vision": False, "context_length": 8192,
    }])
    with pytest.raises(Exception) as error:
        runtime.complete("script_parse", "hello")
    assert "not a chat model" in str(error.value)


def test_defaults_seed_from_discovery_and_leave_unassigned_roles():
    discovered = [{
        "provider": "openai_compatible",
        "name": "oMLX",
        "wire": "openai_chat",
        "base_url": "http://127.0.0.1:8787/v1",
        "instance_id": "openai_compatible@127.0.0.1:8787",
        "models": ["Qwen3.8-27B-MLX-8bit", "MarkItDown"],
    }]
    catalog = {
        "openai_compatible@127.0.0.1:8787": [
            {"id": "Qwen3.8-27B-MLX-8bit", "chat": True},
            {"id": "MarkItDown", "chat": False},
        ],
    }
    settings = build_defaults(discovered=discovered, catalog=catalog)
    assert "openai_compatible@127.0.0.1:8787" in settings.providers
    assert settings.providers["openai_compatible@127.0.0.1:8787"].default_model == "Qwen3.8-27B-MLX-8bit"
    assert settings.roles["script_parse"].model == "Qwen3.8-27B-MLX-8bit"
    assert settings.roles["prompt_video"].provider == "gemma"

    empty = build_defaults(discovered=[], catalog={})
    assert empty.roles["script_parse"].provider == ""
    assert empty.roles["script_parse"].model == ""


# ── L6: enhancement parsing, streaming, warm ────────────────────────


def test_parse_enhancement_unwraps_common_keys():
    assert runtime.parse_enhancement('{"prompt": "Neon rain."}', "raw") == ("Neon rain.", False)
    assert runtime.parse_enhancement('{"text": "A slow push-in."}', "raw")[0] == "A slow push-in."
    assert runtime.parse_enhancement('{"is_reference_only": true, "enhanced_prompt": "Sheet."}', "raw") == ("Sheet.", True)
    assert runtime.parse_enhancement('{"foo": "bar"}', "raw") == ("raw", False)
    assert runtime.parse_enhancement("```\nA plain prompt.\n```", "raw")[0] == "A plain prompt."


def test_streaming_falls_back_when_the_engine_never_sends(app_client, monkeypatch):
    body = doc(app_client)
    body["roles"]["script_parse"] = {"role": "script_parse", "provider": "ollama", "model": "m", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200

    async def silent(**kwargs):
        await asyncio.sleep(30)
        yield "never"

    monkeypatch.setattr(providers, "astream", silent)

    async def fallback(**kwargs):
        return providers.Completion(
            text="fallback text", provider=kwargs["provider"], model=kwargs["model"], latency_ms=5,
        )

    monkeypatch.setattr(providers, "acomplete", fallback)
    monkeypatch.setattr(runtime, "STREAM_FIRST_BYTE_SEC", 0.05)

    async def collect():
        return [chunk async for chunk in runtime.astream("script_parse", "hello")]

    assert asyncio.run(collect()) == ["fallback text"]


def test_warm_records_load_time(app_client, monkeypatch):
    body = doc(app_client)
    body["providers"]["ollama"]["default_model"] = "qwen3:8b"
    assert put(app_client, body, body["revision"]).status_code == 200
    monkeypatch.setattr(providers, "warm", lambda **kwargs: {"ok": True, "load_ms": 1234})
    response = app_client.post("/settings/llm/providers/ollama/warm")
    assert response.status_code == 200
    assert response.json()["load_ms"] == 1234
    health = app_client.get("/settings/llm/health").json()
    assert health["providers"]["ollama"]["telemetry"]["last_load_ms"] == 1234


# ── L8: vision payloads ─────────────────────────────────────────────


def test_vision_payloads_carry_the_image():
    from services.llm.providers import ChatMessage

    messages = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="look")]
    image = ("image/png", "QUJD")

    ollama = providers._ollama_payload("m", messages, {}, "0", image=image)
    assert ollama["images"] == ["QUJD"]

    openai = providers._openai_payload("m", messages, {}, image=image)
    parts = openai["messages"][-1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,QUJD"

    anthropic = providers._anthropic_payload("m", messages, {"max_tokens": 10}, image=image)
    assert anthropic["messages"][-1]["content"][0]["source"]["data"] == "QUJD"


# ── L7: eval ────────────────────────────────────────────────────────


def test_eval_records_results_and_recommends_a_plan(app_client, monkeypatch):
    body = doc(app_client)
    body["roles"]["prompt_video"] = {"role": "prompt_video", "provider": "gemma", "model": "ltx-gemma3", "fallback": []}
    body["roles"]["prompt_image"] = {"role": "prompt_image", "provider": "ollama", "model": "m", "fallback": []}
    body["roles"]["script_parse"] = {"role": "script_parse", "provider": "ollama", "model": "m", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200

    def fake_enhance(role, prompt, **kwargs):
        text = "Style: noir. The camera drifts as rain falls." if role == "prompt_video" else "A detailed portrait in rain."
        result = runtime.LLMResult(text=text, provider="ollama", model="m", role=role, latency_ms=5)
        return text, False, result

    def fake_complete(role, prompt, **kwargs):
        return runtime.LLMResult(
            text='[{"name": "Scene 1", "shots": [{"action": "Mara edits", "shot_type": "wide"}]}]',
            provider="ollama", model="m", role=role, latency_ms=5,
        )

    monkeypatch.setattr(runtime, "enhance", fake_enhance)
    monkeypatch.setattr(runtime, "complete", fake_complete)

    response = app_client.post("/settings/llm/eval", json={"roles": ["prompt_video", "prompt_image", "script_parse"], "limit": 1})
    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["pass_rate"] == 1.0
    assert summary["cases"] == 3
    assert set(summary["roles"]) == {"prompt_video", "prompt_image", "script_parse"}

    body = doc(app_client)
    assert body["last_eval"]["pass_rate"] == 1.0

    applied = app_client.post(f"/settings/llm/eval/apply?expected_revision={body['revision']}")
    assert applied.status_code == 200, applied.text
    assert applied.json()["roles"]["script_parse"]["provider"] == "ollama"


# ── L9: budget, eval gate, profiles ─────────────────────────────────


def test_budget_precheck_fails_closed():
    settings = llm_settings.get()
    with pytest.raises(ValueError) as zero:
        budget.precheck(settings, provider="openai", is_local=False, model="gpt-6", prompt_tokens=10, max_tokens=10)
    assert "budget is 0" in str(zero.value)

    settings.budget.daily_usd = 5.0
    with pytest.raises(ValueError) as unknown:
        budget.precheck(settings, provider="openai", is_local=False, model="mystery-model", prompt_tokens=10, max_tokens=10)
    assert "no price" in str(unknown.value)

    with pytest.raises(ValueError) as too_much:
        budget.precheck(settings, provider="openai", is_local=False, model="gpt-6", prompt_tokens=1_000_000, max_tokens=1_000_000)
    assert "would be exceeded" in str(too_much.value)

    assert budget.precheck(settings, provider="ollama", is_local=True, model="m", prompt_tokens=1, max_tokens=1) == 0.0


def test_eval_gate_blocks_cloud_until_an_eval_passes(app_client, monkeypatch):
    monkeypatch.setenv("MILIMO_LLM_CLOUD", "1")
    body = doc(app_client)
    body["air_gapped"] = False
    body["budget"] = {"daily_usd": 5.0, "eval_gate": True}
    body["providers"]["openai"]["enabled"] = True
    body["providers"]["openai"]["default_model"] = "gpt-6"
    body["providers"]["openai"]["api_key"] = "sk-test"
    body["roles"]["agent_plan"] = {"role": "agent_plan", "provider": "openai", "model": "gpt-6", "fallback": []}
    blocked = put(app_client, body, body["revision"])
    assert blocked.status_code == 400
    assert "eval gate" in blocked.json()["detail"]

    llm_settings.set_last_eval({
        "completed_at": "2026-11-17T00:00:00+00:00",
        "revision": body["revision"],
        "pass_rate": 0.9,
        "p50_ms": 100,
        "cases": 3,
        "roles": {"agent_plan": {"provider": "openai", "model": "gpt-6", "pass_rate": 0.9, "p50_ms": 100}},
    })
    body = doc(app_client)
    body["air_gapped"] = False
    body["budget"] = {"daily_usd": 5.0, "eval_gate": True}
    body["providers"]["openai"]["enabled"] = True
    body["providers"]["openai"]["default_model"] = "gpt-6"
    body["providers"]["openai"]["api_key"] = "sk-test"
    body["roles"]["agent_plan"] = {"role": "agent_plan", "provider": "openai", "model": "gpt-6", "fallback": []}
    allowed = put(app_client, body, body["revision"])
    assert allowed.status_code == 200, allowed.text


def test_sovereign_profile_moves_cloud_roles_to_local(app_client, monkeypatch):
    monkeypatch.setenv("MILIMO_LLM_CLOUD", "1")
    body = doc(app_client)
    body["air_gapped"] = False
    body["budget"] = {"daily_usd": 5.0, "eval_gate": False}
    body["providers"]["ollama"]["default_model"] = "qwen3:8b"
    body["providers"]["openai"]["enabled"] = True
    body["providers"]["openai"]["default_model"] = "gpt-6"
    body["providers"]["openai"]["api_key"] = "sk-test"
    body["roles"]["agent_plan"] = {"role": "agent_plan", "provider": "openai", "model": "gpt-6", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200

    response = app_client.post(f"/settings/llm/profiles/sovereign?expected_revision={doc(app_client)['revision']}")
    assert response.status_code == 200, response.text
    snapshot = response.json()
    assert snapshot["air_gapped"] is True
    assert snapshot["budget"]["daily_usd"] == 0.0
    assert snapshot["providers"]["openai"]["enabled"] is False
    assert snapshot["roles"]["agent_plan"]["provider"] == "ollama"


def test_hybrid_profile_requires_key_budget_and_eval(app_client):
    response = app_client.post("/settings/llm/profiles/hybrid")
    assert response.status_code == 400
    assert "cloud provider" in response.json()["detail"]


def test_classification_is_separator_aware_and_picks_sane_defaults():
    assert capabilities.classify_model("gemma-4-12b-coder-fable5-composer2.5-4bit")["chat"] is True
    assert capabilities.classify_model("bge-m3")["embedding_only"] is True
    assert capabilities.classify_model("text-embedding-3-small")["embedding_only"] is True
    assert capabilities.classify_model("Qwen3.8-27B-Uncensored-OrcaRouter-MLX-8bit")["chat"] is True

    picked = capabilities.pick_default_model([
        {"id": "Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-mlx-8bit"},
        {"id": "Llama-3.2-3B-Instruct-bf16"},
        {"id": "Qwen3.8-27B-MLX-8bit"},
        {"id": "MarkItDown", "chat": False},
    ])
    assert picked == "Llama-3.2-3B-Instruct-bf16"


def test_release_unloads_every_role_model_not_just_the_default(app_client, monkeypatch):
    body = doc(app_client)
    body["providers"]["ollama"]["default_model"] = "default-model"
    body["roles"]["script_parse"] = {"role": "script_parse", "provider": "ollama", "model": "role-model", "fallback": []}
    assert put(app_client, body, body["revision"]).status_code == 200

    calls = []

    def fake_unload(**kwargs):
        calls.append(kwargs["model"])
        return True

    monkeypatch.setattr(providers, "unload", fake_unload)
    runtime.release_local()
    assert set(calls) == {"default-model", "role-model"}
