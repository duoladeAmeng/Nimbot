from dataclasses import dataclass
from typing import Any
from duolaAgent.llm.llm_runtime import LLMRunTime

@dataclass(slots=True)
class AgentRunSpec:
    initial_messages: list[dict[str,Any]]
    runtime: LLMRunTime


@dataclass(slots=True)
class AgentRunResult:
    final_content: str|None


