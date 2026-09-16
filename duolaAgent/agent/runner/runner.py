"""Shared bounded runner; provider wire formats are outside this module."""
from __future__ import annotations

import asyncio
from copy import deepcopy

from duolaAgent.agent.context_governance import ContextGovernanceConfig, ContextGovernor
from duolaAgent.agent.tools.registry import is_tool_error_result
from duolaAgent.utils.helpers import normalize_messages

from .specs import AgentRunResult, AgentRunSpec

_MAX_INJECTIONS_PER_TURN = 3


class AgentRunner:
    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        messages = normalize_messages(spec.initial_messages)
        tools_used, compacted = [], set()
        had_errors = False
        injected = 0
        final_content = None
        governor = ContextGovernor()
        config = ContextGovernanceConfig(
            provider=spec.runtime.provider, model=spec.runtime.model, tools=spec.tools,
            workspace=spec.workspace, session_key=spec.session_key,
            max_tool_result_chars=spec.max_tool_result_chars,
            context_window_tokens=spec.runtime.context_window_tokens,
            max_tokens=spec.runtime.generation.max_tokens, inflight_start_index=len(messages))
        checkpoint_lock = asyncio.Lock()

        async def checkpoint(stage):
            if spec.checkpoint_callback:
                await spec.checkpoint_callback({"stage": stage, "messages": deepcopy(messages)})

        async def drain():
            nonlocal injected
            if not spec.injection_callback or injected >= _MAX_INJECTIONS_PER_TURN:
                return False
            # Callback consumes no more than this allowance; unconsumed items stay queued.
            additions = await spec.injection_callback(_MAX_INJECTIONS_PER_TURN - injected)
            if not additions:
                return False
            messages.extend(normalize_messages(additions))
            injected += len(additions)
            await checkpoint("tools_completed")
            return True

        for _ in range(max(1, spec.max_tool_iterations)):
            model_messages = governor.prepare_for_model(config, messages, compacted)
            response = await self._request_model(spec, model_messages)
            if response.finish_reason == "error":
                final_content = response.content or "Error calling LLM."
                messages.append({"role": "assistant", "content": final_content})
                await checkpoint("final_response")
                return AgentRunResult(final_content, messages, "error", tools_used,
                                      bool(injected), had_errors)
            # Degenerate provider calls cannot be executed or replayed.
            calls = [c for c in response.tool_calls if c.has_valid_name and c.id]
            if calls:
                messages.append({"role": "assistant", "content": response.content or "",
                                 "tool_calls": [c.to_openai_tool_call() for c in calls]})
                await checkpoint("awaiting_tools")

                async def execute(call):
                    nonlocal had_errors
                    result = await spec.tools.execute(call.name, call.arguments)
                    error = is_tool_error_result(result)
                    content = governor.normalize_tool_result(config, call.id, call.name, result)
                    # Serial checkpoint writes persist each completed tool, including partial batches.
                    async with checkpoint_lock:
                        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                         "content": str(content), "is_error": error})
                        tools_used.append(call.name)
                        had_errors = had_errors or error
                        await checkpoint("awaiting_tools")

                for batch in self._tool_batches(spec, calls):
                    tasks = [asyncio.create_task(execute(call)) for call in batch]
                    try:
                        await asyncio.gather(*tasks)
                    except BaseException:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        raise
                await checkpoint("tools_completed")
                await drain()
                continue

            final_content = response.content
            if final_content:
                messages.append({"role": "assistant", "content": final_content})
            await checkpoint("final_response")
            if await drain():
                continue
            continuation = spec.goal_continue_message() if spec.goal_continue_message else None
            if continuation:
                messages.append({"role": "user", "content": continuation, "_internal": True})
                await checkpoint("tools_completed")
                continue
            reason = "completed" if response.finish_reason in {None, "", "stop", "end_turn"} else response.finish_reason
            return AgentRunResult(final_content, messages, reason, tools_used, bool(injected), had_errors)

        await checkpoint("tools_completed")
        return AgentRunResult(final_content, messages, "max_tool_iterations", tools_used,
                              bool(injected), had_errors)

    @staticmethod
    def _tool_batches(spec, calls):
        batch = []
        for call in calls:
            tool = spec.tools.get(call.name)
            safe = spec.concurrent_tools and tool is not None and tool.concurrency_safe
            if safe:
                batch.append(call)
            else:
                if batch:
                    yield batch
                    batch = []
                yield [call]
        if batch:
            yield batch

    async def _request_model(self, spec, messages):
        kwargs = dict(messages=messages, model=spec.runtime.model,
                      max_tokens=spec.runtime.generation.max_tokens,
                      temperature=spec.runtime.generation.temperature)
        if len(spec.tools):
            kwargs["tools"] = spec.tools.get_definitions()
        if spec.on_content_delta:
            return await spec.runtime.provider.chat_stream_with_retry(
                **kwargs, on_content_delta=spec.on_content_delta)
        return await spec.runtime.provider.chat_with_retry(**kwargs)
