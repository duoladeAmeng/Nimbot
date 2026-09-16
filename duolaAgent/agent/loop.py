"""Seven-stage turn runtime, session admission, cancellation and durable recovery."""
from __future__ import annotations

import asyncio
import time
import weakref
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from loguru import logger

from duolaAgent.agent.context import ContextBuilder
from duolaAgent.agent.context_governance import ContextGovernor
from duolaAgent.agent.memory import Consolidator
from duolaAgent.agent.model_runtime import ModelRuntimeResolver
from duolaAgent.agent.runner import AgentRunner, AgentRunSpec
from duolaAgent.agent.tools.context import RequestContext, request_context
from duolaAgent.agent.tools.exec_session import ExecSessionManager
from duolaAgent.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from duolaAgent.agent.tools.registry import create_default_tool_registry
from duolaAgent.bus.message import InboundMessage, OutboundMessage
from duolaAgent.llm.llm_runtime import GenerationSettings, LLMRuntime
from duolaAgent.security.workspace_access import (
    WorkspaceScopeResolver,
    bind_workspace_scope,
    reset_workspace_scope,
)
from duolaAgent.session.goal_state import goal_state_runtime_lines, sustained_goal_active
from duolaAgent.session.manager import Session, SessionManager
from duolaAgent.session.turn_continuation import _goal_continuation_prompt, continuation_message
from duolaAgent.utils.helpers import normalize_messages


class TurnKind(Enum):
    USER = auto()
    SYSTEM = auto()


@dataclass
class TurnContext:
    msg: InboundMessage
    session_key: str
    turn_id: str
    kind: TurnKind
    runtime: LLMRuntime | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    session: Session | None = None
    save_skip: int = 0
    history_start: int = 0
    final_content: str | None = None
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    outbound: OutboundMessage | None = None
    pending_queue: asyncio.Queue | None = None
    continuation: InboundMessage | None = None
    context: ContextBuilder | None = None

    def require_session(self) -> Session:
        if self.session is None:
            raise RuntimeError("RESTORE must initialize the session")
        return self.session


class AgentLoop:
    def __init__(self, bus, provider=None, model="", workspace=None, tools=None,
                 session_manager=None, max_tool_iterations=3, max_context_tokens=12000,
                 compact_target_tokens=8000, max_tool_result_chars=20000,
                 restrict_to_workspace=True, concurrent_tools=False, stream=True,
                 max_concurrent_sessions=4, max_pending_messages=64, max_subagents=4,
                 max_continuation_turns=12, runtime_resolver=None, plugins=True, mcp_configs=None, tool_config=None):
        from duolaAgent.agent.mcp import MCPProvider
        from duolaAgent.agent.subagent import SubagentManager
        from duolaAgent.agent.tools.long_task import CreateGoalTool, GetGoalTool, UpdateGoalTool
        from duolaAgent.agent.tools.spawn import SpawnTool, SubagentStatusTool
        if max_concurrent_sessions < 1 or max_pending_messages < 1:
            raise ValueError("Concurrency and pending queue limits must be positive")
        self.bus, self.provider, self.model = bus, provider, model
        self.workspace = Path(workspace or ".").expanduser().resolve()
        self.runner = AgentRunner()
        self.context = ContextBuilder(self.workspace)
        self.sessions = session_manager or SessionManager(self.workspace)
        self.consolidator = Consolidator(self.context.memory, self.sessions)
        self.exec_sessions = ExecSessionManager()
        self.file_states = FileStateStore()
        self.subagents = SubagentManager(self.workspace, bus, max_concurrent_subagents=max_subagents,
                                         max_iterations=max_tool_iterations, plugins=plugins,
                                         mcp_configs=mcp_configs, tool_config=tool_config)
        self.tools = tools if tools is not None else create_default_tool_registry(
            self.workspace, restrict_to_workspace=restrict_to_workspace,
            exec_session_manager=self.exec_sessions, plugins=plugins, tool_config=tool_config)
        if tools is None:
            for tool in (CreateGoalTool(self.sessions), UpdateGoalTool(self.sessions),
                         GetGoalTool(self.sessions), SpawnTool(self.subagents), SubagentStatusTool(self.subagents)):
                self.tools.register(tool)
        self.mcp = MCPProvider(mcp_configs or {}, self.tools)
        self.max_tool_iterations = max_tool_iterations
        self.max_context_tokens = max_context_tokens
        self.compact_target_tokens = compact_target_tokens
        self.max_tool_result_chars = max_tool_result_chars
        self.concurrent_tools, self.stream = concurrent_tools, stream
        self.max_pending_messages = max_pending_messages
        self.max_continuation_turns = max_continuation_turns
        self.model_runtime = runtime_resolver or ModelRuntimeResolver(
            runtime=LLMRuntime(provider, model, GenerationSettings(), max_context_tokens))
        self.scope_resolver = WorkspaceScopeResolver(self.workspace, restrict_to_workspace)
        self._running = False
        self._closed = False
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._session_locks = weakref.WeakValueDictionary()
        self._pending_queues = {}
        self._deferred_messages = {}
        self._active_tasks = {}
        self._dispatch_tasks = set()
        self._concurrency_gate = asyncio.Semaphore(max_concurrent_sessions)

    @property
    def runtime(self):
        return self.model_runtime.current

    async def initialize(self):
        async with self._initialize_lock:
            if not self._initialized:
                await self.mcp.connect()
                self._initialized = True

    async def run(self):
        self._running = True
        try:
            await self.initialize()
            while self._running:
                msg = await self.bus.consume_inbound()
                task = asyncio.create_task(self._dispatch(msg))
                self._dispatch_tasks.add(task)
                task.add_done_callback(self._dispatch_tasks.discard)
        finally:
            await self.aclose()

    async def _dispatch(self, msg):
        key = msg.session_key
        if msg.content.strip() == "/stop" and not msg.metadata.get("subagent_result"):
            await self.cancel_session(key)
            await self.bus.publish_outbound(self._assemble_outbound(msg, "Session stopped.", "cancelled"))
            return
        owner = self._active_tasks.get(key)
        if owner is not None and not owner.done():
            queue = self._pending_queues[key]
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                await self.bus.publish_outbound(self._assemble_outbound(msg, "Pending message queue is full.", "queue_full"))
            return
        queue = asyncio.Queue(maxsize=self.max_pending_messages)
        self._pending_queues[key] = queue
        self._active_tasks[key] = asyncio.current_task()
        lock = self._session_locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                current = msg
                while current is not None:
                    try:
                        async with self._concurrency_gate:
                            outbound = await self._process_message(current, session_key=key)
                            if outbound is not None:
                                await self.bus.publish_outbound(outbound)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.exception("Turn failed for {}", key)
                        await self.bus.publish_outbound(self._assemble_outbound(
                            current, f"Turn failed: {type(exc).__name__}: {exc}", "error"))
                    current = self._deferred_messages.pop(key, None)
                    if current is None:
                        try:
                            current = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
        finally:
            self._active_tasks.pop(key, None)
            self._pending_queues.pop(key, None)
            self._deferred_messages.pop(key, None)

    async def _process_message(self, msg, session_key=None):
        await self.initialize()
        key = session_key or msg.session_key
        if msg.require_existing_session and not self.sessions._path_for_key(key).exists() and not self.sessions.get_cached(key):
            return None
        ctx = TurnContext(msg, key, f"{key}:{time.time_ns()}",
                          TurnKind.SYSTEM if msg.sender_id == "system" or msg.metadata.get("internal_continuation") else TurnKind.USER,
                          pending_queue=self._pending_queues.get(key))
        await self._run_turn_stage(ctx, "restore", self._restore_turn)
        scope = self.scope_resolver.for_message(msg, ctx.require_session().metadata)
        ctx.context = self.context if scope.project_path == self.workspace else ContextBuilder(scope.project_path)
        scope_token = bind_workspace_scope(scope)
        file_token = bind_file_states(self.file_states.for_session(key))
        rc = RequestContext(channel=msg.channel, chat_id=msg.chat_id, session_key=key,
                            runtime=ctx.runtime, turn_id=ctx.turn_id, workspace=scope.project_path,
                            original_user_text=msg.content, metadata=dict(msg.metadata), sender_id=msg.sender_id)
        try:
            with request_context(rc):
                await self._run_turn_stage(ctx, "compact", self._compact_session)
                if await self._run_turn_stage(ctx, "command", self._dispatch_command):
                    return ctx.outbound
                await self._run_turn_stage(ctx, "build", self._build_turn)
                await self._run_turn_stage(ctx, "run", self._run_turn)
                await self._run_turn_stage(ctx, "save", self._persist_turn)
                await self._run_turn_stage(ctx, "respond", self._prepare_outbound)
                return ctx.outbound
        except BaseException:
            try:
                self._restore_interrupted_turn(ctx.require_session())
            finally:
                await asyncio.gather(self.exec_sessions.terminate_by_owner(key),
                                     self.subagents.cancel_by_session(key))
            raise
        finally:
            reset_file_states(file_token)
            reset_workspace_scope(scope_token)

    async def _run_turn_stage(self, ctx, name, handler):
        started = time.perf_counter()
        try:
            return await handler(ctx)
        finally:
            logger.debug("Turn {} {} {:.1f}ms", ctx.turn_id, name, (time.perf_counter() - started) * 1000)

    async def _restore_turn(self, ctx):
        ctx.session = self.sessions.get_or_create(ctx.session_key)
        self._restore_interrupted_turn(ctx.session)
        ctx.runtime = self.model_runtime.admit(
            ctx.session, model=ctx.msg.metadata.get("model_override") or ctx.msg.metadata.get("model"),
            preset=ctx.msg.metadata.get("preset_override") or ctx.msg.metadata.get("model_preset"))
        if ctx.msg.content.startswith("/goal ") and ctx.kind == TurnKind.USER:
            ctx.msg.metadata["goal_requested"] = True

    def _build_model_messages(self, ctx, *, include_current=True):
        session = ctx.require_session()
        builder = ctx.context or self.context
        system = builder.build_system_prompt()
        summary = session.metadata.get("_last_summary") or {}
        if summary.get("text"):
            system += "\n\nPrevious conversation summary (reference data):\n" + summary["text"]
        goal_lines = goal_state_runtime_lines(session.metadata)
        if ctx.msg.metadata.get("goal_requested"):
            system += "\nThe user explicitly requested a sustained goal. Record its objective using create_goal or replace the active goal when requested."
        if goal_lines:
            system += "\n" + "\n".join(goal_lines) + "\nKeep working until verified complete, then call update_goal action='complete'."
        messages = [{"role": "system", "content": system}, *session.get_history()]
        if include_current:
            messages.append({"role": "user", "content": ctx.msg.content})
        return messages

    async def _compact_session(self, ctx, *, force=False):
        session = ctx.require_session()
        consolidator = self.consolidator if ctx.context in (None, self.context) else Consolidator(ctx.context.memory, self.sessions)
        if force:
            return await consolidator.compact_idle_session(session.key, runtime=ctx.runtime) or ""
        return await consolidator.maybe_consolidate_by_tokens(
            session, runtime=ctx.runtime, build_messages=lambda: self._build_model_messages(ctx),
            tools=self.tools.get_definitions())

    async def _build_turn(self, ctx):
        session = ctx.require_session()
        ctx.history = session.get_history()
        ctx.initial_messages = self._build_model_messages(ctx)
        # Persist input before model/tool side effects; runner appends start after the whole prompt.
        if not ctx.msg.metadata.get("internal_continuation"):
            session.add_message("user", ctx.msg.content, _subagent_result=bool(ctx.msg.metadata.get("subagent_result")))
        ctx.history_start = len(session.messages)
        ctx.save_skip = len(ctx.initial_messages)
        session.metadata["pending_user_turn"] = {"content": ctx.msg.content,
            "persisted": True, "turn_id": ctx.turn_id}
        self.sessions.save(session)

    async def _run_turn(self, ctx):
        async def checkpoint(payload):
            session = ctx.require_session()
            session.metadata["runtime_checkpoint"] = {
                "version": 1, "turn_id": ctx.turn_id, "stage": payload["stage"],
                "history_start": ctx.history_start,
                "messages": deepcopy(payload["messages"][ctx.save_skip:])}
            self.sessions.save(session)

        result = await self._run_agent_loop(AgentRunSpec(
            initial_messages=ctx.initial_messages, runtime=ctx.runtime, tools=self.tools,
            max_tool_iterations=self.max_tool_iterations, max_tool_result_chars=self.max_tool_result_chars,
            concurrent_tools=self.concurrent_tools, workspace=(ctx.context or self.context).workspace,
            session_key=ctx.session_key, on_content_delta=self._stream_callback(ctx.msg) if self.stream else None,
            injection_callback=lambda limit: self._drain_pending_messages(ctx.session_key, limit),
            checkpoint_callback=checkpoint,
            goal_continue_message=lambda: _goal_continuation_prompt(ctx.require_session().metadata)
            if sustained_goal_active(ctx.require_session().metadata) else None))
        ctx.final_content, ctx.all_messages, ctx.stop_reason = result.final_content, result.messages, result.stop_reason

    async def _run_agent_loop(self, spec: AgentRunSpec):
        """Keep the same orchestration-to-runner seam as nanobot."""
        return await self.runner.run(spec)

    async def _persist_turn(self, ctx):
        session = ctx.require_session()
        previous_messages, previous_metadata = list(session.messages), deepcopy(session.metadata)
        try:
            session.messages.extend(deepcopy(ctx.all_messages[ctx.save_skip:]))
            session.metadata.pop("pending_user_turn", None)
            session.metadata.pop("runtime_checkpoint", None)
            ctx.continuation = continuation_message(ctx, max_rounds=self.max_continuation_turns)
            self.sessions.save(session)
        except BaseException:
            session.messages[:] = previous_messages
            session.metadata.clear()
            session.metadata.update(previous_metadata)
            raise
        if ctx.continuation:
            ctx.pending_queue.put_nowait(ctx.continuation)

    async def _prepare_outbound(self, ctx):
        if ctx.continuation:
            return
        content = ctx.final_content or ("Iteration budget reached." if ctx.stop_reason == "max_tool_iterations"
                                        else "Agent did not generate a final response.")
        ctx.outbound = self._assemble_outbound(ctx.msg, content, ctx.stop_reason)

    def _assemble_outbound(self, msg, content, stop_reason):
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                               content=content, metadata={"stop_reason": stop_reason})

    def _restore_interrupted_turn(self, session):
        checkpoint = session.metadata.get("runtime_checkpoint")
        pending = session.metadata.get("pending_user_turn")
        if not checkpoint and not pending:
            return
        previous_messages, previous_metadata = list(session.messages), deepcopy(session.metadata)
        try:
            if isinstance(pending, dict) and not pending.get("persisted") and pending.get("content"):
                session.add_message("user", pending["content"])
            if isinstance(checkpoint, dict):
                if checkpoint.get("version") != 1 or checkpoint.get("stage") not in {
                    "awaiting_tools", "tools_completed", "final_response"
                }:
                    raise ValueError("Unsupported runtime checkpoint")
                recorded = normalize_messages(checkpoint.get("messages", []))
                # Only recorded effects are restored. Never rerun uncertain side effects automatically.
                recorded = ContextGovernor().backfill_missing_tool_results(recorded)
                session.messages.extend(recorded)
            elif isinstance(pending, dict):
                session.add_message("assistant", "Error: previous run was interrupted before completion.")
            session.metadata.pop("pending_user_turn", None)
            session.metadata.pop("runtime_checkpoint", None)
            self.sessions.save(session)
        except BaseException:
            session.messages[:] = previous_messages
            session.metadata.clear()
            session.metadata.update(previous_metadata)
            raise

    def _stream_callback(self, msg):
        async def publish(delta):
            if delta:
                await self.bus.publish_outbound(OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                    content=delta, metadata={"event": "stream_delta"}))
        return publish

    async def _drain_pending_messages(self, key, limit=3):
        queue = self._pending_queues.get(key)
        if queue is None or key in self._deferred_messages:
            return []
        messages = []
        while len(messages) < limit:
            try:
                msg = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if msg.content.lstrip().startswith("/") or any(k in msg.metadata for k in
                ("model", "model_override", "model_preset", "preset_override", "internal_continuation")):
                self._deferred_messages[key] = msg
                break
            messages.append({"role": "user", "content": msg.content})
        return messages

    async def cancel_session(self, key):
        owner = self._active_tasks.get(key)
        if owner and owner is not asyncio.current_task():
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
        await self.exec_sessions.terminate_by_owner(key)
        await self.subagents.cancel_by_session(key)

    def stop(self):
        self._running = False
        for task in list(self._dispatch_tasks):
            task.cancel()

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        self.stop()
        tasks = [task for task in self._dispatch_tasks if task is not asyncio.current_task()]
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.subagents.aclose()
        await self.exec_sessions.close_all()
        await self.mcp.aclose()
        await self.model_runtime.aclose()


    async def _dispatch_command(
        self,
        ctx: TurnContext,
    ) -> bool:
        """Handle migrated slash commands before building a model request."""

        if ctx.kind == TurnKind.SYSTEM:
            return False
        raw = ctx.msg.content.strip()
        if raw.startswith('/model '):
            preset = raw.split(maxsplit=1)[1]
            self.model_runtime.admit(ctx.session, preset=preset)
            ctx.require_session().metadata['model_preset'] = preset
            ctx.outbound = self._command_response(ctx, f'Model preset set to {preset}')
            return True
        if raw == '/reload':
            self.model_runtime.invalidate()
            ctx.outbound = self._command_response(ctx, 'Runtime config invalidated; next turn refreshes it.')
            return True
        if raw == '/goal':
            ctx.outbound = self._command_response(ctx, str(ctx.require_session().metadata.get('goal_state') or 'No goal.'))
            return True

        if raw == "/clear":

            ctx.require_session().clear()
            self.sessions.save(ctx.require_session())

            ctx.outbound = OutboundMessage(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                content="Session 已清空。",
            )

            return True

        if raw in {"/help", "/?"}:
            ctx.outbound = self._command_response(
                ctx,
                "\n".join(
                    [
                        "Commands:",
                        "/help - show commands",
                        "/clear - clear current session",
                        "/history [n] - show recent session messages",
                        "/sessions - list persisted sessions",
                        "/compact - force context compaction",
                        "/memory - show long-term memory",
                        "/dream - summarize this session into long-term memory",
                        "/tools - list registered tools",
                        "/config - show runtime config",
                        "/model <preset> - select this session's model preset",
                        "/reload - invalidate model config; refresh at next admission",
                        "/goal <objective> - explicitly request a sustained goal",
                        "/stop - cancel the session and its active resources",
                    ]
                ),
            )
            return True

        if raw.startswith("/history"):
            parts = raw.split()
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            messages = ctx.require_session().messages[-limit:]
            lines = [
                f"{idx + 1}. {m.get('role')}: {str(m.get('content'))[:200]}"
                for idx, m in enumerate(messages)
            ]
            ctx.outbound = self._command_response(ctx, "\n".join(lines) or "No history.")
            return True

        if raw == "/sessions":
            rows = self.sessions.list_sessions()
            lines = [
                f"- {item['key']} ({item['message_count']} messages) {item.get('preview', '')}"
                for item in rows[:20]
            ]
            ctx.outbound = self._command_response(ctx, "\n".join(lines) or "No sessions.")
            return True

        if raw == "/compact":
            summary = await self._compact_session(ctx, force=True)
            content = (
                "Session compacted. Archived conversation history for Dream."
                if summary
                else "Nothing to compact: this session has no non-command conversation history yet."
            )
            ctx.outbound = self._command_response(ctx, content)
            return True

        if raw == "/memory":
            ctx.outbound = self._command_response(
                ctx,
                self.context.memory.read_memory() or "No long-term memory yet.",
            )
            return True

        if raw == "/dream":
            summary = await self._run_dream(runtime=ctx.runtime)
            ctx.outbound = self._command_response(
                ctx,
                summary or (
                    "Dream has no new archived history to process. "
                    "Run /compact or wait for auto-compact first."
                ),
            )
            return True

        if raw == "/tools":
            ctx.outbound = self._command_response(ctx, "\n".join(self.tools.tool_names))
            return True

        if raw == "/config":
            ctx.outbound = self._command_response(
                ctx,
                "\n".join(
                    [
                        f"workspace={self.workspace}",
                        f"model={ctx.runtime.model}",
                        f"context_window_tokens={ctx.runtime.context_window_tokens}",
                        f"max_tool_iterations={self.max_tool_iterations}",
                        f"max_context_tokens={self.max_context_tokens}",
                        f"compact_target_tokens={self.compact_target_tokens}",
                        f"concurrent_tools={self.concurrent_tools}",
                        f"stream={self.stream}",
                    ]
                ),
            )
            return True

        return False

    async def _run_dream(self, *, runtime=None) -> str:
        started_at = time.monotonic()
        built = self.context.memory.build_dream_prompt()
        if built is None:
            return ""

        prompt, last_cursor = built
        dream_result = await self.runner.run(
            AgentRunSpec(
                initial_messages=[{"role": "user", "content": prompt}],
                runtime=runtime or self.runtime,
                tools=self.context.memory.build_dream_tools(),
                max_tool_iterations=self.max_tool_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                concurrent_tools=False,
            )
        )
        elapsed = time.monotonic() - started_at
        completed = self.context.memory.dream_run_completed(
            dream_result,
            had_tool_errors=dream_result.had_tool_errors,
        )
        if completed:
            self.context.memory.set_last_dream_cursor(last_cursor)
            self.context.memory.compact_history()
            return f"Dream completed in {elapsed:.1f}s."
        return (
            f"Dream did not complete after {elapsed:.1f}s; "
            "memory cursor was not advanced."
        )

    def _command_response(self, ctx: TurnContext, content: str) -> OutboundMessage:
        ctx.require_session().add_message("user", ctx.msg.content, _command=True)
        ctx.require_session().add_message("assistant", content, _command=True)
        self.sessions.save(ctx.require_session())
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=content,
            metadata={"stop_reason": "command"},
        )
