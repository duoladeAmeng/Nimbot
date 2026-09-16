import asyncio
import sys

import pytest

from duolaAgent.agent.mcp import MCPProvider, _safe_mcp_http_client
from duolaAgent.agent.subagent import SubagentManager
from duolaAgent.agent.tools.context import current_request_context
from duolaAgent.agent.tools.registry import ToolRegistry
from duolaAgent.bus.queue import MessageBus
from duolaAgent.llm.llm_runtime import LLMRuntime
from duolaAgent.security.workspace_access import default_workspace_scope, current_workspace_scope
from test.runtime_helpers import ScriptedProvider, wait_until
from duolaAgent.llm.providers.base import LLMResponse


async def test_real_stdio_mcp_connect_call_reload_and_owner_task_close(tmp_path):
    server = tmp_path / "server.py"
    server.write_text("from mcp.server.mcpserver import MCPServer\n"
                      "mcp = MCPServer('test', log_level='ERROR')\n"
                      "@mcp.tool()\ndef add(a: int, b: int) -> int:\n    return a + b\n"
                      "mcp.run()\n")
    registry = ToolRegistry()
    mcp = MCPProvider({"local": {"command": sys.executable, "args": [str(server)]}}, registry)
    await mcp.connect()
    assert mcp.connected_server_names == {"local"}
    assert "5" in await registry.execute("mcp_local_add", {"a": "2", "b": 3})
    old = list(mcp._connections.values())
    status = await mcp.reload()
    assert status["ok"]
    assert all(c._task.done() for c in old)
    current = list(mcp._connections.values())
    await mcp.aclose()
    assert not registry.tool_names
    assert all(c._task.done() for c in current)


async def test_mcp_http_ssrf_transport_and_config_rejection():
    import httpx2
    async with _safe_mcp_http_client() as client:
        with pytest.raises(httpx2.RequestError):
            await client.get("http://127.0.0.1:8/mcp")
    mcp = MCPProvider({"unsafe": {"url": "http://169.254.169.254/mcp"}}, ToolRegistry())
    await mcp.connect()
    assert mcp.runtime_status()["unsafe"] == "failed"
    await mcp.aclose()


async def test_subagent_inline_background_scope_and_parent_delivery(tmp_path):
    bus = MessageBus()
    contexts = []

    async def handler(kwargs):
        ctx = current_request_context()
        contexts.append(ctx)
        assert current_workspace_scope().project_path == tmp_path
        assert ctx.attributes["parent_session_key"] == "cli:parent"
        return LLMResponse("child answer")

    provider = ScriptedProvider(handler=handler)
    runtime = LLMRuntime(provider, "test")
    manager = SubagentManager(tmp_path, bus, max_concurrent_subagents=2, plugins=False)
    kwargs = dict(task="work", runtime=runtime, session_key="cli:parent",
                  origin_channel="cli", origin_chat_id="parent",
                  workspace_scope=default_workspace_scope(tmp_path, True))
    assert await manager.run_inline(**kwargs) == "child answer"
    await manager.spawn(**kwargs)
    message = await asyncio.wait_for(bus.consume_inbound(), 3)
    assert message.session_key == "cli:parent"
    assert message.metadata["subagent_result"]
    assert "child answer" in message.content
    assert len({ctx.session_key for ctx in contexts}) == 2
    assert contexts[0] is not contexts[1]
    assert current_request_context() is None
    await manager.aclose()


async def test_subagent_capacity_and_parent_cancellation(tmp_path):
    started = asyncio.Event()

    async def handler(kwargs):
        started.set()
        await asyncio.Event().wait()

    manager = SubagentManager(tmp_path, MessageBus(), max_concurrent_subagents=1, plugins=False)
    runtime = LLMRuntime(ScriptedProvider(handler=handler), "test")
    kwargs = dict(task="work", runtime=runtime, session_key="cli:parent",
                  origin_channel="cli", origin_chat_id="parent",
                  workspace_scope=default_workspace_scope(tmp_path, True))
    await manager.spawn(**kwargs)
    await started.wait()
    with pytest.raises(ValueError, match="limit"):
        await manager.spawn(**kwargs)
    await manager.cancel_by_session("cli:parent")
    assert manager.get_running_count() == 0
    assert list(manager._statuses.values())[0]["status"] == "cancelled"
    await manager.aclose()


async def test_subagent_cancel_before_first_scheduled_step(tmp_path):
    manager = SubagentManager(tmp_path, MessageBus(), plugins=False)
    await manager.spawn(task="work", runtime=LLMRuntime(ScriptedProvider(), "test"),
                        session_key="cli:parent", origin_channel="cli", origin_chat_id="parent",
                        workspace_scope=default_workspace_scope(tmp_path, True))
    await manager.cancel_by_session("cli:parent")
    assert manager.get_running_count() == 0
    assert list(manager._statuses.values())[0]["status"] == "cancelled"
    await manager.aclose()
