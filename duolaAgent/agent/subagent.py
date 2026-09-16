"""Isolated child runners with bounded admission and parent result delivery."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import replace

from duolaAgent.agent.context import ContextBuilder
from duolaAgent.agent.runner import AgentRunner, AgentRunSpec
from duolaAgent.agent.tools.context import RequestContext, request_context
from duolaAgent.agent.tools.exec_session import ExecSessionManager
from duolaAgent.agent.tools.file_state import FileStates, bind_file_states, reset_file_states
from duolaAgent.agent.tools.registry import create_default_tool_registry
from duolaAgent.bus.message import InboundMessage
from duolaAgent.security.workspace_access import bind_workspace_scope, reset_workspace_scope


class SubagentManager:
    def __init__(self, workspace, bus, *, max_concurrent_subagents=4, max_iterations=20,
                 plugins=True, mcp_configs=None, tool_config=None):
        self.workspace, self.bus = workspace, bus
        self.max_concurrent_subagents = max_concurrent_subagents
        self.max_iterations = max_iterations
        self.plugins = plugins
        self.mcp_configs = mcp_configs or {}
        self.tool_config = tool_config
        self._running_tasks = {}
        self._statuses = {}

    def get_running_count(self):
        return sum(not task.done() for task in self._running_tasks.values())

    def get_status(self, task_id):
        return dict(self._statuses.get(task_id, {}))

    async def spawn(self, **kwargs):
        task_id, _task = self._start(**kwargs, background=True)
        return f"Subagent started: {task_id}"

    async def run_inline(self, **kwargs):
        _id, task = self._start(**kwargs, background=False)
        return await task

    def _start(self, *, task, runtime, session_key, origin_channel, origin_chat_id,
               workspace_scope, label=None, temperature=None, origin_message_id=None, background):
        if self.get_running_count() >= self.max_concurrent_subagents:
            raise ValueError("Subagent concurrency limit reached")
        task_id = uuid.uuid4().hex[:12]
        self._statuses[task_id] = {"task_id": task_id, "parent_session_key": session_key,
                                  "status": "running", "label": label or task[:80], "started_at": time.time()}
        child = asyncio.create_task(self._run(
            task_id, task, runtime, session_key, origin_channel, origin_chat_id,
            workspace_scope, temperature, background))
        self._running_tasks[task_id] = child
        def finished(future):
            self._running_tasks.pop(task_id, None)
            if future.cancelled():
                self._statuses[task_id]["status"] = "cancelled"
            elif future.exception() is not None:
                self._statuses[task_id].update(status="failed", result=str(future.exception()))
        child.add_done_callback(finished)
        # Keep bounded completed status history, never evict running entries.
        for key in list(self._statuses):
            if len(self._statuses) <= 128:
                break
            if self._statuses[key]["status"] != "running":
                del self._statuses[key]
        return task_id, child

    async def _run(self, task_id, task, runtime, parent, channel, chat_id, scope, temperature, background):
        from duolaAgent.agent.mcp import MCPProvider
        workspace = scope.project_path if scope else self.workspace
        manager = ExecSessionManager()
        mcp = None
        state_token = bind_file_states(FileStates())
        scope_token = bind_workspace_scope(scope) if scope else None
        if temperature is not None:
            runtime = replace(runtime, generation=replace(runtime.generation, temperature=temperature))
        ctx = RequestContext(channel=channel, chat_id=chat_id, session_key=f"subagent:{task_id}",
                             runtime=runtime, workspace=workspace, attributes={"parent_session_key": parent})
        result = ""
        try:
            with request_context(ctx):
                tools = create_default_tool_registry(workspace, restrict_to_workspace=scope.restrict_to_workspace if scope else True,
                                                     exec_session_manager=manager, plugins=self.plugins,
                                                     plugin_scope="subagent", tool_config=self.tool_config)
                mcp = MCPProvider(self.mcp_configs, tools)
                await mcp.connect()
                messages = ContextBuilder(workspace).build_messages([], task)
                run = await AgentRunner().run(AgentRunSpec(
                    initial_messages=messages, runtime=runtime, tools=tools,
                    max_tool_iterations=self.max_iterations, concurrent_tools=True,
                    workspace=workspace, session_key=ctx.session_key))
                result = run.final_content or f"Subagent stopped: {run.stop_reason}"
                self._statuses[task_id].update(status="failed" if run.stop_reason == "error" else "completed",
                                               stop_reason=run.stop_reason, result=result)
        except asyncio.CancelledError:
            self._statuses[task_id]["status"] = "cancelled"
            raise
        except Exception as exc:
            result = f"Subagent failed: {type(exc).__name__}: {exc}"
            self._statuses[task_id].update(status="failed", result=result)
        finally:
            try:
                cleanup = [manager.close_all()]
                if mcp is not None:
                    cleanup.append(mcp.aclose())
                await asyncio.gather(*cleanup)
            finally:
                reset_file_states(state_token)
                if scope_token:
                    reset_workspace_scope(scope_token)
        if background:
            await self.bus.publish_inbound(InboundMessage(
                channel=channel, sender_id="system", chat_id=chat_id,
                content=f"Subagent {task_id} result:\n{result}", session_key_override=parent,
                metadata={"subagent_result": True, "subagent_task_id": task_id},
                require_existing_session=True))
        return result

    async def cancel_by_session(self, session_key):
        tasks = [task for key, task in self._running_tasks.items()
                 if self._statuses[key]["parent_session_key"] == session_key]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self):
        tasks = list(self._running_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
