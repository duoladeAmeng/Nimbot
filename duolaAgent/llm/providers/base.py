"""Provider contract, classified failures and bounded retries."""
from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: Any

    @property
    def has_valid_name(self):
        return isinstance(self.name, str) and bool(self.name.strip())

    def to_openai_tool_call(self):
        return {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": self.arguments if isinstance(self.arguments, str)
            else json.dumps(self.arguments, ensure_ascii=False)}}


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    error_type: str | None = None
    streamed_content: str = ""

    @property
    def has_tool_calls(self):
        return bool(self.tool_calls)


LLMReponse = LLMResponse
RECOVERABLE_ERRORS = frozenset({"timeout", "connection", "rate_limit", "server_error"})
FALLBACK_ERRORS = RECOVERABLE_ERRORS | {"authentication", "quota"}


def classify_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", getattr(exc, "code", None))
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    text = str(exc).lower()
    if response is not None:
        try:
            text += str(response.text).lower()
        except Exception:
            pass
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timeout"
    if any(word in text for word in ("context_length", "context window", "too many tokens")):
        return "context_overflow"
    if any(word in text for word in ("insufficient_quota", "quota exceeded", "arrearage")):
        return "quota"
    if status in (401, 403):
        return "authentication"
    if status == 429:
        return "rate_limit"
    if isinstance(status, int) and status >= 500:
        return "server_error"
    if status in (400, 404, 422):
        return "invalid_request"
    if isinstance(exc, (ConnectionError, OSError)) or any(
        s in type(exc).__name__.lower() for s in ("connection", "connecterror")
    ):
        return "connection"
    return "unknown"


class LLMProvider(ABC):
    def __init__(self, api_key=None, api_base=None):
        self.api_key, self.api_base = api_key, api_base
        self.timeout_s = float(os.environ.get("DUOLA_LLM_TIMEOUT_S", "60"))
        self.retry_attempts = max(1, int(os.environ.get("DUOLA_RETRY_ATTEMPTS", "2")))

    @abstractmethod
    async def chat(self, messages, model, max_tokens=4096, temperature=0.7, tools=None):
        ...

    async def chat_stream(self, *, on_content_delta=None, **kwargs):
        response = await self.chat(**kwargs)
        if on_content_delta and response.content and response.finish_reason != "error":
            await on_content_delta(response.content)
        return response

    async def chat_with_retry(self, **kwargs):
        for attempt in range(self.retry_attempts):
            try:
                response = await asyncio.wait_for(self.chat(**kwargs), timeout=self.timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = LLMResponse(f"Error calling LLM: {type(exc).__name__}: {exc}",
                                       finish_reason="error", error_type=classify_error(exc))
            if response.finish_reason != "error" or response.error_type not in RECOVERABLE_ERRORS:
                return response
            if attempt + 1 < self.retry_attempts:
                await asyncio.sleep(min(2 ** attempt, 4))
        return response

    async def chat_stream_with_retry(self, *, on_content_delta=None, **kwargs):
        segments = []
        prefix = ""
        request = dict(kwargs)

        async def emit(delta):
            segments.append(delta)
            if on_content_delta:
                await on_content_delta(delta)

        for attempt in range(self.retry_attempts):
            try:
                response = await asyncio.wait_for(
                    self.chat_stream(**request, on_content_delta=emit), timeout=self.timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = LLMResponse(f"Error calling LLM: {type(exc).__name__}: {exc}",
                                       finish_reason="error", error_type=classify_error(exc))
            response.streamed_content = "".join(segments)
            if response.finish_reason != "error":
                if prefix:
                    response.content = prefix + (response.content or "")
                return response
            if response.error_type not in RECOVERABLE_ERRORS:
                return response
            if segments:
                if response.error_type != "timeout":
                    return response
                prefix = "".join(segments)
                request["messages"] = list(kwargs["messages"]) + [
                    {"role": "assistant", "content": prefix},
                    {"role": "user", "content": "The stream timed out. Continue from the visible text without repeating it."}]
            if attempt + 1 < self.retry_attempts:
                await asyncio.sleep(min(2 ** attempt, 4))
        return response

    async def aclose(self):
        pass
