import asyncio
import os
import re
import sys
from copy import deepcopy

import httpx
import pytest

from duolaAgent.agent.runner import AgentRunner, AgentRunSpec
from duolaAgent.agent.tools.context import RequestContext, request_context
from duolaAgent.agent.tools.exec_session import ExecSessionManager
from duolaAgent.agent.tools.file_state import FileStates, bind_file_states, reset_file_states
from duolaAgent.agent.tools.registry import ToolRegistry
from duolaAgent.agent.tools.shell import ExecTool
from duolaAgent.agent.tools.tool_impl.filesystem import ReadFileTool, WriteFileTool
from duolaAgent.llm.llm_runtime import LLMRuntime
from duolaAgent.llm.providers.base import LLMResponse, ToolCallRequest
from duolaAgent.security.network import PinnedDNSAsyncTransport, UnsafeURLRequestError
from duolaAgent.security.workspace_access import default_workspace_scope, bind_workspace_scope, reset_workspace_scope
from test.runtime_helpers import FunctionTool, ScriptedProvider


async def test_schema_recursive_cast_validation_errors_and_cancellation():
    received = []

    async def execute(**kwargs):
        received.append(kwargs)
        return "ok"

    tool = FunctionTool("typed", execute, schema={"type": "object", "required": ["n"],
        "properties": {"n": {"type": "integer", "minimum": 1}, "flag": {"type": "boolean"},
                       "items": {"type": "array", "items": {"type": "integer"}}},
        "additionalProperties": False})
    registry = ToolRegistry()
    registry.register(tool)
    assert await registry.execute("typed", '{"n":"2","flag":"false","items":["3"]}') == "ok"
    assert received == [{"n": 2, "flag": False, "items": [3]}]
    for args in ({"n": True}, {"n": 0}, {"n": 1, "extra": 2}, {"n": 1, "items": ["bad"]}):
        result = await registry.execute("typed", args)
        assert result.is_error
    assert len(received) == 1
    assert (await registry.execute("missing", {})).is_error


async def test_concurrency_safe_batches_preserve_exclusive_barriers(tmp_path):
    active, events = set(), []
    both = asyncio.Event()

    async def safe(name):
        active.add(name)
        events.append(name)
        if len(active) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 1)
        active.remove(name)
        return name

    async def exclusive():
        assert not active
        assert set(events) == {"a", "b"}
        events.append("exclusive")
        return "ok"

    registry = ToolRegistry()
    registry.register(FunctionTool("a", lambda: safe("a"), safe=True))
    registry.register(FunctionTool("b", lambda: safe("b"), safe=True))
    registry.register(FunctionTool("exclusive", exclusive))
    provider = ScriptedProvider([LLMResponse(None, [ToolCallRequest(n, n, {}) for n in ("a", "b", "exclusive")])])
    result = await AgentRunner().run(AgentRunSpec(
        [{"role": "user", "content": "run"}], LLMRuntime(provider, "test"), registry,
        concurrent_tools=True, workspace=tmp_path))
    assert events[-1] == "exclusive"
    assert result.stop_reason == "completed"


async def test_workspace_context_and_file_state_are_isolated(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    scoped = root / "child"
    scoped.mkdir()
    (scoped / "a.txt").write_text("child")
    read = ReadFileTool(root, restrict_to_workspace=True)
    states = FileStates()
    token = bind_file_states(states)
    scope_token = bind_workspace_scope(default_workspace_scope(scoped, True))
    try:
        assert "child" in await read.execute("a.txt")
        assert states.check_read(str(scoped / "a.txt")) is None
        assert "not been read" in FileStates().check_read(str(scoped / "a.txt"))
        assert (await read.execute(str(outside))).is_error
        assert (await WriteFileTool(root, restrict_to_workspace=True).execute("../escape.txt", "x")).is_error
    finally:
        reset_workspace_scope(scope_token)
        reset_file_states(token)
    assert not (root / "escape.txt").exists()


@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd"])
async def test_direct_web_ssrf_rejected_before_transport(url):
    touched = []
    inner = httpx.MockTransport(lambda request: touched.append(request) or httpx.Response(200))
    async with httpx.AsyncClient(transport=PinnedDNSAsyncTransport(inner=inner)) as client:
        with pytest.raises((UnsafeURLRequestError, httpx.UnsupportedProtocol)):
            await client.get(url)
    assert not touched


async def test_redirect_revalidated(monkeypatch):
    import socket
    original = socket.getaddrinfo
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *args, **kw:
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        if host == "public.example" else original(host, *args, **kw))
    touched = []

    async def handle(request):
        touched.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    async with httpx.AsyncClient(transport=PinnedDNSAsyncTransport(inner=httpx.MockTransport(handle)),
                                 follow_redirects=True) as client:
        with pytest.raises(UnsafeURLRequestError):
            await client.get("http://public.example")
    assert touched == ["http://public.example"]


async def test_exec_session_owner_and_cancellation_cleanup(tmp_path):
    manager = ExecSessionManager()
    tool = ExecTool(working_dir=str(tmp_path), session_manager=manager)
    command = "Write-Output ready; Start-Sleep -Seconds 30" if os.name == "nt" else "echo ready; sleep 30"
    with request_context(RequestContext("cli", "one", session_key="cli:one")):
        result = await tool.execute(command=command, yield_time_ms=100)
    match = re.search(r"session_id: ([a-f0-9]+)", result)
    assert match, result
    sid = match[1]
    infos = await manager.list(owner_session_key="cli:one")
    assert len(infos) == 1
    with pytest.raises(KeyError):
        await manager.write(session_id=sid, chars=None, close_stdin=False, terminate=True,
                            yield_time_ms=0, max_output_chars=1000, owner_session_key="cli:other")
    processes = [s.process for s in manager._sessions.values()]
    assert await manager.terminate_by_owner("cli:one") == 1
    assert all(p.returncode is not None for p in processes)
    assert await manager.list(owner_session_key="cli:one") == []
    await manager.close_all()


async def test_exec_timeout_and_task_cancellation(tmp_path):
    tool = ExecTool(working_dir=str(tmp_path))
    command = "Start-Sleep -Seconds 30" if os.name == "nt" else "sleep 30"
    result = await tool.execute(command=command, timeout=1)
    assert "timed out" in result
    processes = []
    original = tool._spawn

    async def spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process

    # Use the real one-shot subprocess path and verify its owned root is reaped.
    tool._spawn = spawn
    task = asyncio.create_task(tool.execute(command=command))
    from test.runtime_helpers import wait_until
    await wait_until(lambda: bool(processes))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None


async def test_entrypoint_plugins_construct_independent_instances(monkeypatch, tmp_path):
    from duolaAgent.agent.tools.loader import load_tool_plugins
    from duolaAgent.agent.tools.base import Tool

    class Plugin(Tool):
        name, description = "plugin", "test"
        parameters = {"type": "object", "properties": {}}

        async def execute(self):
            return "ok"

    class Entry:
        name = "plugin"

        def load(self):
            return Plugin

    monkeypatch.setattr("duolaAgent.agent.tools.loader.entry_points",
                        lambda group: [Entry()] if group == "duolaAgent.tools" else [])
    first, second = ToolRegistry(), ToolRegistry()
    load_tool_plugins(first, workspace=tmp_path)
    load_tool_plugins(second, workspace=tmp_path)
    assert first.get("plugin") is not second.get("plugin")
    assert await first.execute("plugin", {}) == "ok"


async def test_unpolled_exec_session_still_obeys_deadline(tmp_path):
    manager = ExecSessionManager()
    tool = ExecTool(working_dir=str(tmp_path), session_manager=manager)
    command = "Start-Sleep -Seconds 30" if os.name == "nt" else "sleep 30"
    result = await tool.execute(command=command, timeout=1, yield_time_ms=0)
    sid = re.search(r"session_id: ([a-f0-9]+)", result)[1]
    process = manager._sessions[sid].process
    from test.runtime_helpers import wait_until
    await wait_until(lambda: process.returncode is not None)
    assert manager._sessions[sid]._timed_out
    await manager.close_all()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")
async def test_cancellation_terminates_grandchild_process(tmp_path):
    import ctypes
    from ctypes import wintypes
    from test.runtime_helpers import wait_until
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "tree.py"
    script.write_text("import subprocess, sys, time\nfrom pathlib import Path\n"
                      "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                      f"Path({str(pidfile)!r}).write_text(str(child.pid))\n"
                      "time.sleep(60)\n")
    tool = ExecTool(working_dir=str(tmp_path))
    task = asyncio.create_task(tool.execute(command=f"& '{sys.executable}' '{script}'"))
    try:
        await wait_until(pidfile.exists, timeout=5)
        pid = int(pidfile.read_text())
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, pid)
        assert handle
        try:
            assert kernel.WaitForSingleObject(handle, 0) == 258
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            assert kernel.WaitForSingleObject(handle, 1000) == 0
        finally:
            kernel.CloseHandle(handle)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
