from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from duolaAgent.agent.tools.registry import ToolRegistry
from duolaAgent.llm.llm_runtime import LLMRuntime


@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict[str, Any]]
    runtime: LLMRuntime
    tools: ToolRegistry
    max_tool_iterations: int = 3
    max_tool_result_chars: int = 20000
    concurrent_tools: bool = False
    on_content_delta: Callable[[str], Awaitable[None]] | None = None
    injection_callback: Callable[..., Awaitable[list[dict[str, Any]]]] | None = None
    checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    goal_continue_message: Callable[[], str | None] | None = None
    workspace: Path | None = None
    session_key: str | None = None


@dataclass(slots=True)
class AgentRunResult:
    final_content: str | None
    messages: list[dict[str, Any]]
    stop_reason: str = "completed"
    tools_used: list[str] = field(default_factory=list)
    had_injections: bool = False
    had_tool_errors: bool = False
