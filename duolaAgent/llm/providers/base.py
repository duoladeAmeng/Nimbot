from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class LLMReponse:
    content: str


class LLMProvider(ABC):
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    async def chat(
            self,
            messages: list[dict[str, Any]],
            model: str,
            max_tokens: int = 4096,
            temperature: float = 0.7,
    ) -> LLMReponse:
        pass

    async def chat_stream(
            self,
            messages: list[dict[str, Any]],
            model: str,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            on_content_delta: Callable[[str], Awaitable[None]] | None = None
    )->LLMReponse:
        response= await self.chat(
            messages=messages,
            model= model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

    async def chat_with_retry(
            self,
            messages: list[dict[str, Any]],
            model: str,
            max_tokens: int = 4096,
            temperature: float = 0.7,
    ) -> LLMReponse:
        pass
