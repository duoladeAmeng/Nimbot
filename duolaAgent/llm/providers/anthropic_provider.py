import asyncio
from typing import Any, Callable, Awaitable

from duolaAgent.llm.providers.base import LLMProvider, LLMReponse


class AnthropicProvider(LLMProvider):
    def __init__(
            self,
            api_key: str | None = None,
            api_base: str | None = None,
            default_model: str = "claude-sonnet-4-6",
    ):
        super().__init__(api_key,api_base)

        from anthropic import AsyncAnthropic
        client_kw:dict[str, Any] = {}
        if api_key:
            client_kw["api_key"] = api_key
        if api_base:
            client_kw["base_url"] = api_base

        self._client = AsyncAnthropic(**client_kw)

    async def chat(self,
                   messages: list[dict[str, Any]],
                   model: str,
                   max_tokens: int = 4096,
                   temperature: float = 0.7) -> LLMReponse:

        try:
            response= await self._client.messages.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return self._parse_reponse(response)
        except Exception as e:
            raise e



    async def chat_stream(
            self,
            messages: list[dict[str, Any]],
            model: str,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            on_content_delta: Callable[[str], Awaitable[None]] | None = None
    )->LLMReponse:
        idle_timeout_s: int = 60
        try:
            async with self._client.messages.stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            ) as stream:
                if on_content_delta:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                stream.__anext__(),
                                timeout=idle_timeout_s,
                            )
                        except StopAsyncIteration:
                            break
                        if (
                            chunk.type == "content_block_delta"
                            and getattr(chunk.delta, "type", None) == "text_delta"
                        ):
                            text = getattr(chunk.delta, "text", None) or ""
                            if text and on_content_delta:
                                await on_content_delta(text)
                response= await asyncio.wait_for(
                    stream.get_final_message(),
                    timeout=idle_timeout_s,
                )
                return self._parse_reponse(response)
        except Exception as e:
            raise e



    @staticmethod
    def _parse_reponse(response: Any) -> LLMReponse:
        content_parts: list[str] = []

        for block in response.content:
            if block.type=="text":
                content_parts.append(block.text)

        return LLMReponse(
            content="".join(content_parts) or None,
        )