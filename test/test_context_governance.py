from copy import deepcopy

from duolaAgent.agent.context_governance import ContextGovernor, ContextGovernanceConfig
from duolaAgent.agent.memory import Consolidator, MemoryStore
from duolaAgent.agent.tools.registry import ToolRegistry
from duolaAgent.llm.llm_runtime import LLMRuntime
from duolaAgent.llm.providers.base import GenerationSettings, LLMResponse
from duolaAgent.session.manager import SessionManager
from duolaAgent.utils.helpers import normalize_messages, is_user_turn
from test.runtime_helpers import ScriptedProvider


def call(cid="a", name="exec"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}


def config(tmp_path, **kwargs):
    return ContextGovernanceConfig(ScriptedProvider(), "test", ToolRegistry(), tmp_path, "one",
                                   max_tool_result_chars=5000, **kwargs)


def assert_protocol(messages):
    expected = set()
    for message in messages:
        if message["role"] == "tool":
            assert message["tool_call_id"] in expected
            expected.remove(message["tool_call_id"])
        else:
            assert not expected
            expected = {c["id"] for c in message.get("tool_calls", [])}
    assert not expected


def test_repair_is_on_model_copy_and_handles_legacy_history(tmp_path):
    original = [
        {"role": "tool", "tool_call_id": "orphan", "content": "lost"},
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "", "tool_calls": [call(), call("bad", None)]},
        {"role": "user", "content": "next"},
    ]
    saved = deepcopy(original)
    result = ContextGovernor().prepare_for_model(config(tmp_path), original, set())
    assert original == saved
    assert_protocol(result)
    assert any("interrupted" in str(m.get("content")) for m in result)
    legacy = [{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "exec", "input": {}}]},
              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]}]
    assert_protocol(normalize_messages(legacy))
    assert legacy[0]["content"][0]["type"] == "tool_use"


def test_offload_compaction_and_snipping_keep_protocol(tmp_path):
    governor = ContextGovernor()
    raw = "x" * 25000
    normalized = governor.normalize_tool_result(config(tmp_path), "id", "exec", raw)
    paths = list((tmp_path / ".duola" / "tool-results").rglob("*.txt"))
    assert paths[0].read_text() == raw
    assert str(paths[0]) in normalized
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "old" * 3000},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "run"}, {"role": "assistant", "tool_calls": [call()], "content": ""},
                {"role": "tool", "tool_call_id": "a", "name": "exec", "content": "y" * 4500}]
    saved = deepcopy(messages)
    compacted = set()
    result = governor.prepare_for_model(config(tmp_path, context_window_tokens=2500, max_tokens=100,
                                                inflight_start_index=3), messages, compacted)
    assert "a" in compacted
    assert_protocol(result)
    assert messages == saved
    assert not any(m.get("content") == "old" * 3000 for m in result)


async def test_token_consolidation_raw_fallback_cursor_and_user_suffix(tmp_path):
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("one")
    for i in range(10):
        session.add_message("user", f"question {i} " + "x" * 3000)
        session.add_message("assistant", "response")
        session.add_message("assistant", "", tool_calls=[call(str(i))])
        session.add_message("tool", "ok", tool_call_id=str(i))
    sessions.save(session)
    provider = ScriptedProvider([LLMResponse("error", finish_reason="error")] * 5)
    runtime = LLMRuntime(provider, "test", GenerationSettings(max_tokens=100), 5000)
    store = MemoryStore(tmp_path)
    consolidator = Consolidator(store, sessions)
    original = deepcopy(session.messages)
    await consolidator.maybe_consolidate_by_tokens(session, runtime=runtime,
                                                   build_messages=session.get_history)
    assert session.last_consolidated > 0
    assert session.messages == original
    assert is_user_turn(session.messages[session.last_consolidated])
    assert store.history_file.exists()
    assert is_user_turn(session.get_history()[0])
    assert_protocol(session.get_history())
    restored = SessionManager(tmp_path).get_or_create("one")
    assert restored.last_consolidated == session.last_consolidated
    assert len(restored.get_history()) < len(restored.messages)


def test_live_session_identity_survives_lru_eviction(tmp_path):
    manager = SessionManager(tmp_path)
    manager._max_cached_sessions = 1
    first = manager.get_or_create("a")
    manager.get_or_create("b")
    assert manager.get_or_create("a") is first
