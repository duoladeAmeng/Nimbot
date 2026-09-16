import asyncio
from copy import deepcopy

import pytest

from duolaAgent.agent.loop import AgentLoop
from duolaAgent.agent.tools.registry import ToolRegistry
from duolaAgent.llm.providers.base import LLMResponse, ToolCallRequest
from duolaAgent.session.manager import SessionManager
from test.runtime_helpers import ScriptedProvider, FunctionTool, inbound, wait_until


async def test_stages_input_persistence_and_final_checkpoint(tmp_path):
    provider = ScriptedProvider([LLMResponse("answer")])
    loop = AgentLoop(None, provider, "test", workspace=tmp_path, tools=ToolRegistry(), stream=False)
    stages = []
    original = loop._run_turn_stage

    async def stage(ctx, name, handler):
        stages.append(name)
        if name == "run":
            persisted = SessionManager(tmp_path).get_or_create(ctx.session_key)
            assert persisted.messages[-1]["content"] == "hello"
        if name == "save":
            checkpoint = SessionManager(tmp_path).get_or_create(ctx.session_key).metadata["runtime_checkpoint"]
            assert checkpoint["stage"] == "final_response"
            assert checkpoint["messages"][-1]["content"] == "answer"
        return await original(ctx, name, handler)

    loop._run_turn_stage = stage
    result = await loop._process_message(inbound())
    assert result.content == "answer"
    assert stages == ["restore", "compact", "command", "build", "run", "save", "respond"]
    session = SessionManager(tmp_path).get_or_create("cli:one")
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    assert "runtime_checkpoint" not in session.metadata
    await loop.aclose()


async def test_same_session_injection_and_global_admission_limit(tmp_path):
    from duolaAgent.bus.queue import MessageBus
    entered = asyncio.Event()
    release = asyncio.Event()
    active = peak = 0

    async def handler(kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        await release.wait()
        active -= 1
        return LLMResponse("answer")

    provider = ScriptedProvider(handler=handler)
    loop = AgentLoop(MessageBus(), provider, "test", workspace=tmp_path, tools=ToolRegistry(),
                     stream=False, max_concurrent_sessions=2, max_tool_iterations=3)
    first = asyncio.create_task(loop._dispatch(inbound()))
    await entered.wait()
    await loop._dispatch(inbound("followup"))
    second = asyncio.create_task(loop._dispatch(inbound("other", key="two")))
    third = asyncio.create_task(loop._dispatch(inbound("third", key="three")))
    await wait_until(lambda: active == 2)
    assert len(provider.calls) == 2
    release.set()
    await asyncio.gather(first, second, third)
    assert peak == 2
    calls = [c for c in provider.calls if any(m.get("content") == "followup" for m in c["messages"])]
    assert len(calls) == 1
    assert any(m.get("content") == "answer" for m in calls[0]["messages"])
    history = loop.sessions.get_or_create("cli:one").messages
    assert sum(m.get("content") == "followup" for m in history) == 1
    assert not loop._active_tasks and not loop._pending_queues
    await loop.aclose()


async def test_cancel_partial_batch_recovers_completed_and_interrupted_results(tmp_path):
    started, finished = asyncio.Event(), asyncio.Event()
    calls = []

    async def fast():
        calls.append("fast")
        finished.set()
        return "already wrote file"

    async def slow():
        calls.append("slow")
        started.set()
        await asyncio.Event().wait()

    registry = ToolRegistry()
    registry.register(FunctionTool("fast", fast, safe=True))
    registry.register(FunctionTool("slow", slow, safe=True))
    provider = ScriptedProvider([LLMResponse(None, [
        ToolCallRequest("a", "fast", {}), ToolCallRequest("b", "slow", {})])])
    from duolaAgent.bus.queue import MessageBus
    loop = AgentLoop(MessageBus(), provider, "test", workspace=tmp_path, tools=registry,
                     concurrent_tools=True, stream=False)
    task = asyncio.create_task(loop._dispatch(inbound()))
    await started.wait()
    await finished.wait()
    await wait_until(lambda: len(loop.sessions.get_or_create("cli:one").metadata["runtime_checkpoint"]["messages"]) == 2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    restored = SessionManager(tmp_path).get_or_create("cli:one")
    outputs = {m["tool_call_id"]: m["content"] for m in restored.messages if m["role"] == "tool"}
    assert outputs["a"] == "already wrote file"
    assert "interrupted" in outputs["b"]
    assert calls == ["fast", "slow"]
    assert not restored.metadata.get("runtime_checkpoint")
    await loop._process_message(inbound("continue"))
    assert calls == ["fast", "slow"]
    assert sum(m.get("content") == "hello" for m in restored.messages) == 1
    await loop.aclose()


@pytest.mark.parametrize("stage", ["awaiting_tools", "tools_completed", "final_response"])
async def test_restart_restores_checkpoint_once(tmp_path, stage):
    sessions = SessionManager(tmp_path)
    session = sessions.get_or_create("cli:one")
    session.add_message("user", "old")
    recorded = [{"role": "assistant", "content": "complete"}] if stage == "final_response" else [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "a", "function": {"name": "write_file", "arguments": "{}"}, "type": "function"}]}]
    if stage == "tools_completed":
        recorded.append({"role": "tool", "tool_call_id": "a", "content": "saved"})
    session.metadata.update(pending_user_turn={"persisted": True},
                            runtime_checkpoint={"version": 1, "stage": stage, "messages": recorded})
    sessions.save(session)
    provider = ScriptedProvider()
    loop = AgentLoop(None, provider, "test", workspace=tmp_path, tools=ToolRegistry(), stream=False)
    await loop._process_message(inbound("new"))
    history = loop.sessions.get_or_create("cli:one").messages
    before = deepcopy(history)
    loop._restore_interrupted_turn(loop.sessions.get_or_create("cli:one"))
    assert history == before
    assert sum(m.get("content") == "old" for m in history) == 1
    if stage != "final_response":
        assert sum(m.get("tool_call_id") == "a" for m in history) == 1
    await loop.aclose()


async def test_goal_slices_bounded_and_can_complete(tmp_path):
    from duolaAgent.bus.queue import MessageBus
    count = 0

    async def handler(kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return LLMResponse(None, [ToolCallRequest("g", "create_goal", {"objective": "finish task"})])
        if count == 2:
            return LLMResponse(None, [ToolCallRequest("u", "update_goal", {"action": "complete"})])
        return LLMResponse("Verified done")

    loop = AgentLoop(MessageBus(), ScriptedProvider(handler=handler), "test", workspace=tmp_path,
                     stream=False, max_tool_iterations=1, max_continuation_turns=2, plugins=False)
    await loop._dispatch(inbound("/goal finish task"))
    session = loop.sessions.get_or_create("cli:one")
    assert session.metadata["goal_state"]["status"] == "completed"
    assert count == 2
    assert sum(m.get("content") == "/goal finish task" for m in session.messages) == 1
    await loop.aclose()

    loop = AgentLoop(MessageBus(), ScriptedProvider(), "test", workspace=tmp_path / "cap",
                     stream=False, max_tool_iterations=1, max_continuation_turns=2, plugins=False)
    session = loop.sessions.get_or_create("cli:one")
    session.metadata["goal_state"] = {"status": "active", "objective": "ongoing"}
    loop.sessions.save(session)
    await loop._dispatch(inbound())
    assert session.metadata["goal_continuation_rounds"] == 2
    assert len(loop.provider.calls) == 3
    assert session.metadata["goal_state"]["status"] == "active"
    await loop.aclose()


async def test_goal_without_queue_does_not_schedule_and_creation_is_explicit(tmp_path):
    provider = ScriptedProvider([LLMResponse(None, [ToolCallRequest("g", "create_goal", {"objective": "unasked"})])])
    loop = AgentLoop(None, provider, "test", workspace=tmp_path, stream=False, plugins=False)
    await loop._process_message(inbound())
    assert "goal_state" not in loop.sessions.get_or_create("cli:one").metadata
    assert any(m.get("is_error") for m in loop.sessions.get_or_create("cli:one").messages)
    await loop.aclose()
