"""Immutable per-turn configuration; provider client identity may be shared."""
from dataclasses import dataclass, replace

from duolaAgent.llm.providers.base import GenerationSettings, LLMProvider


@dataclass(slots=True, frozen=True)
class LLMRuntime:
    provider: LLMProvider
    model: str
    generation: GenerationSettings = GenerationSettings()
    context_window_tokens: int = 12000

    def with_overrides(self, **kwargs):
        return replace(self, **kwargs)


LLMRunTime = LLMRuntime  # Compatibility with the initial migration.
