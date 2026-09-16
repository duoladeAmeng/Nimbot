"""Cross-provider fallback, primary circuit breaker and stream timeout recovery."""
from __future__ import annotations

import time
from dataclasses import dataclass

from duolaAgent.llm.providers.base import FALLBACK_ERRORS, GenerationSettings, LLMProvider


@dataclass(frozen=True, slots=True)
class ProviderCandidate:
    provider: LLMProvider
    model: str
    generation: GenerationSettings
    context_window_tokens: int


class FallbackProvider(LLMProvider):
    def __init__(self, primary, fallbacks, *, failure_threshold=3, cooldown_s=60):
        super().__init__()
        self.primary = primary
        self.fallbacks = tuple(fallbacks)
        self.context_output_reservation = max(c.generation.max_tokens for c in (primary, *self.fallbacks))
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._primary_failures = 0
        self._primary_open_until = 0.0
        self._half_open_inflight = False

    async def chat(self, **kwargs):
        return await self._call(False, **kwargs)

    async def chat_with_retry(self, **kwargs):
        return await self._call(False, **kwargs)

    async def chat_stream(self, **kwargs):
        return await self._call(True, **kwargs)

    async def chat_stream_with_retry(self, **kwargs):
        return await self._call(True, **kwargs)

    async def _call(self, streaming, **kwargs):
        open_circuit = bool(self._primary_open_until)
        probe = open_circuit and time.monotonic() >= self._primary_open_until
        use_primary = not open_circuit or (probe and not self._half_open_inflight)
        if probe and use_primary:
            self._half_open_inflight = True
        candidates = ([self.primary] if use_primary else []) + list(self.fallbacks)
        prefix = ""
        response = None
        try:
            for candidate in candidates:
                request = dict(kwargs)
                request["model"] = kwargs.get("model", candidate.model) if candidate is self.primary else candidate.model
                if candidate is not self.primary:
                    request.update(max_tokens=candidate.generation.max_tokens,
                                   temperature=candidate.generation.temperature)
                if prefix:
                    request["messages"] = list(kwargs["messages"]) + [
                        {"role": "assistant", "content": prefix},
                        {"role": "user", "content": "The response was interrupted. Continue exactly where it stopped without repeating the visible text."}]
                method = candidate.provider.chat_stream_with_retry if streaming else candidate.provider.chat_with_retry
                response = await method(**request)
                if candidate is self.primary:
                    if response.finish_reason != "error":
                        self._primary_failures, self._primary_open_until = 0, 0.0
                    elif response.error_type in FALLBACK_ERRORS:
                        self._primary_failures += 1
                        if self._primary_failures >= self.failure_threshold:
                            self._primary_open_until = time.monotonic() + self.cooldown_s
                if response.finish_reason != "error":
                    if prefix:
                        response.content = prefix + (response.content or "")
                    return response
                if response.error_type not in FALLBACK_ERRORS:
                    return response
                if streaming and response.streamed_content:
                    if response.error_type != "timeout":
                        return response
                    prefix += response.streamed_content
            if response is None:
                return await self.primary.provider.chat_with_retry(**kwargs)
            return response
        finally:
            if probe and use_primary:
                self._half_open_inflight = False

    async def aclose(self):
        seen = set()
        for candidate in (self.primary, *self.fallbacks):
            if id(candidate.provider) not in seen:
                seen.add(id(candidate.provider))
                await candidate.provider.aclose()
