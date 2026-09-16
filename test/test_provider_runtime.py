import asyncio
from dataclasses import FrozenInstanceError
from copy import deepcopy

import httpx
import pytest

from duolaAgent.agent.model_runtime import ModelRuntimeResolver
from duolaAgent.config.schema import Config
from duolaAgent.llm.providers.base import GenerationSettings, LLMResponse, classify_error
from duolaAgent.llm.providers.fallback_provider import FallbackProvider, ProviderCandidate
from duolaAgent.llm.providers.factory import build_provider_snapshot
from duolaAgent.llm.providers.openai_compat_provider import OpenAICompatibleProvider
from duolaAgent.llm.providers.messages import to_anthropic_messages
from duolaAgent.session.manager import Session
from test.runtime_helpers import ScriptedProvider


def candidate(provider, model="model"):
    return ProviderCandidate(provider, model, GenerationSettings(max_tokens=10), 12000)


async def test_snapshot_preset_override_refresh_and_minimum_window(monkeypatch):
    config = Config.model_validate({"agents": {"defaults": {"model": "default", "fallbacks": ["small"]}},
        "modelPresets": {"small": {"provider": "deepseek", "model": "small", "contextWindowTokens": 8000},
                         "large": {"model": "large", "contextWindowTokens": 40000, "temperature": 0.1}}})
    monkeypatch.setattr("duolaAgent.llm.providers.factory.create_provider", lambda *_: ScriptedProvider())
    snapshot = build_provider_snapshot(config)
    assert snapshot.runtime.context_window_tokens == 8000
    with pytest.raises(FrozenInstanceError):
        snapshot.runtime.model = "changed"
    resolver = ModelRuntimeResolver(config=config, config_loader=lambda: config)
    session = Session("a", metadata={"model_preset": "large"})
    admitted = resolver.admit(session)
    override = resolver.admit(session, model="one-run")
    assert override.model == "one-run" and admitted.model == "large"
    assert resolver.admit(session).model == "large"
    assert session.metadata == {"model_preset": "large"}
    config.model_presets["large"].model = "new"
    assert resolver.admit(session).model == "large"
    resolver.invalidate()
    assert resolver.admit(session).model == "new"
    assert admitted.model == "large"
    await resolver.aclose()


async def test_fallback_error_classification_circuit_and_recovery(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("duolaAgent.llm.providers.fallback_provider.time.monotonic", lambda: now[0])
    primary = ScriptedProvider([LLMResponse("fail", finish_reason="error", error_type="rate_limit")] * 3)
    fallback = ScriptedProvider()
    wrapper = FallbackProvider(candidate(primary, "primary"), [candidate(fallback, "backup")],
                               failure_threshold=2, cooldown_s=60)
    kw = {"messages": [{"role": "user", "content": "hello"}], "model": "primary"}
    await wrapper.chat_with_retry(**kw)
    await wrapper.chat_with_retry(**kw)
    await wrapper.chat_with_retry(**kw)
    assert len(primary.calls) == 2 and len(fallback.calls) == 3
    now[0] += 61
    primary.responses = [LLMResponse("recovered")]
    assert (await wrapper.chat_with_retry(**kw)).content == "recovered"
    assert wrapper._primary_failures == 0
    primary.responses = [LLMResponse("bad request", finish_reason="error", error_type="invalid_request")]
    assert (await wrapper.chat_with_retry(**kw)).finish_reason == "error"
    assert len(fallback.calls) == 3


@pytest.mark.parametrize("error,should_fallback", [(TimeoutError(), True), (ConnectionError(), False)])
async def test_partial_stream_only_timeout_can_continue(error, should_fallback):
    class Streaming(ScriptedProvider):
        async def chat_stream(self, *, on_content_delta=None, **kw):
            await on_content_delta("first ")
            raise error

    first = Streaming()
    second = ScriptedProvider([LLMResponse("second")])
    wrapper = FallbackProvider(candidate(first), [candidate(second)])
    visible = []

    async def emit(text):
        visible.append(text)

    result = await wrapper.chat_stream_with_retry(messages=[{"role": "user", "content": "x"}],
                                                   model="m", on_content_delta=emit)
    if should_fallback:
        assert result.content == "first second"
        assert visible == ["first ", "second"]
        assert second.calls[0]["messages"][-2]["content"] == "first "
    else:
        assert result.finish_reason == "error"
        assert not second.calls
        assert visible == ["first "]


async def test_openai_sse_assembles_calls_and_canonical_to_anthropic():
    async def handler(request):
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello ","tool_calls":[{"index":0,"id":"t","function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}\n\ndata: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"a\\"}"}}]},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n')
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(client=client)
    output = []

    async def emit(text):
        output.append(text)

    response = await provider.chat_stream(messages=[{"role": "user", "content": "read"}],
                                         model="m", on_content_delta=emit)
    assert response.tool_calls[0].arguments == {"path": "a"}
    assert output == ["hello "]
    canonical = [{"role": "assistant", "content": "checking", "tool_calls": [
        response.tool_calls[0].to_openai_tool_call()]},
        {"role": "tool", "tool_call_id": "t", "content": "file", "is_error": False}]
    original = deepcopy(canonical)
    wire = to_anthropic_messages(canonical)
    assert wire[0]["content"][1]["type"] == "tool_use"
    assert wire[1]["content"][0]["tool_use_id"] == "t"
    assert OpenAICompatibleProvider._format_messages(canonical)[0]["tool_calls"][0]["id"] == "t"
    assert canonical == original
    await provider.aclose()


async def test_openai_sse_accepts_null_optional_collections():
    async def handler(request):
        return httpx.Response(200, text=(
            'data: {"choices":null,"usage":{"prompt_tokens":2}}\n\n'
            'data: {"choices":[{"delta":{"content":"你好","tool_calls":null}}]}\n\n'
            'data: {"choices":[{"delta":null,"finish_reason":"stop"}]}\n\n'
            'data: [DONE]\n\n'
        ))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(client=client)

    response = await provider.chat_stream(
        messages=[{"role": "user", "content": "你好"}], model="m"
    )

    assert response.content == "你好"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 2}
    await provider.aclose()


@pytest.mark.parametrize("status,kind", [(401, "authentication"), (429, "rate_limit"), (500, "server_error"), (400, "invalid_request")])
def test_http_error_types(status, kind):
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status, request=request)
    assert classify_error(httpx.HTTPStatusError("error", request=request, response=response)) == kind


async def test_same_provider_stream_timeout_recovery():
    class Provider(ScriptedProvider):
        attempts = 0

        async def chat_stream(self, *, on_content_delta, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                await on_content_delta("part1 ")
                raise TimeoutError()
            assert kwargs["messages"][-2]["content"] == "part1 "
            await on_content_delta("part2")
            return LLMResponse("part2")

    provider = Provider()
    provider.retry_attempts = 2
    visible = []

    async def emit(text):
        visible.append(text)

    result = await provider.chat_stream_with_retry(messages=[{"role": "user", "content": "go"}],
                                                   model="test", on_content_delta=emit)
    assert visible == ["part1 ", "part2"]
    assert result.content == "part1 part2"


def test_cli_model_and_credentials_override_selected_preset(tmp_path, monkeypatch):
    import json
    from duolaAgent.config.loader import load_runtime_config
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agents": {"defaults": {"modelPreset": "chosen"}},
        "modelPresets": {"chosen": {"provider": "openai-compatible", "model": "original"}}}))
    monkeypatch.setattr("duolaAgent.config.loader.load_dotenv", lambda **kw: None)
    for name in ("DUOLA_PROVIDER", "DUOLA_MODEL", "DUOLA_API_KEY", "ANTHROPIC_MODEL_ID", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = load_runtime_config(config_path=path, workspace=tmp_path, model="override", api_key="test-only")
    assert config.model_presets["chosen"].model == "override"
    assert config.providers.openai_compatible.api_key == "test-only"
