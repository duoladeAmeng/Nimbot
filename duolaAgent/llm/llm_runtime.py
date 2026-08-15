
from dataclasses import dataclass

from duolaAgent.llm.providers.base import LLMProvider


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.7
    max_tokens: int = 4096

@dataclass(slots=True,frozen=True)
class LLMRunTime:
    privider:LLMProvider
    model:str
    generation:GenerationSettings
    context_window_tokens:int