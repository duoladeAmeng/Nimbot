import asyncio
from typing import Any, Awaitable, Callable

from duolaAgent.llm.providers.base import LLMProvider, LLMReponse, ToolCallRequest
from duolaAgent.llm.providers.messages import to_anthropic_messages


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
            # Anthropic-compatible providers do not always mirror Anthropic's
            # x-api-key header. MiMo's endpoint uses Bearer auth, while the
            # official Anthropic endpoint uses api_key.
            if api_base and "xiaomimimo.com" in api_base.lower():
                client_kw["auth_token"] = api_key
            else:
                client_kw["api_key"] = api_key
        if api_base:
            client_kw["base_url"] = api_base
        client_kw["timeout"] = self.timeout_s
        client_kw["max_retries"] = 0

        self._client = AsyncAnthropic(**client_kw)

    async def chat(self,
                   messages: list[dict[str, Any]],
                   model: str,
                   max_tokens: int = 4096,
                   temperature: float = 0.7,
                   tools: list[dict[str, Any]] | None = None) -> LLMReponse:

        request_messages, system_prompt = self._split_system_messages(messages)
        kwargs: dict[str, Any] = {
            "messages": request_messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        response= await self._client.messages.create(**kwargs)
        return self._parse_reponse(response)



    async def chat_stream(
            self,
            messages: list[dict[str, Any]],
            model: str,
            max_tokens: int = 4096,
            temperature: float = 0.7,
            tools: list[dict[str, Any]] | None = None,
            on_content_delta: Callable[[str], Awaitable[None]] | None = None
    )->LLMReponse:
        idle_timeout_s: int = 60
        request_messages, system_prompt = self._split_system_messages(messages)
        kwargs: dict[str, Any] = {
            "messages": request_messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        async with self._client.messages.stream(
            **kwargs,
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



    @staticmethod
    def _split_system_messages(
            messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        system_parts: list[str] = []
        request_messages: list[dict[str, Any]] = []

        for message in to_anthropic_messages(messages):
            if message.get("role") != "system":
                request_messages.append(message)
                continue

            content = message.get("content")
            if isinstance(content, str):
                if content.strip():
                    system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            system_parts.append(text)
                    elif block is not None:
                        system_parts.append(str(block))
            elif content is not None:
                system_parts.append(str(content))

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return request_messages, system_prompt


    @staticmethod
    def _format_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []

        for tool in tools:
            function = tool.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                description = function.get("description", "")
                parameters = function.get("parameters") or {
                    "type": "object",
                    "properties": {},
                }
            else:
                name = tool.get("name")
                description = tool.get("description", "")
                parameters = tool.get("input_schema") or tool.get("parameters") or {
                    "type": "object",
                    "properties": {},
                }

            if not isinstance(name, str) or not name:
                continue

            formatted.append(
                {
                    "name": name,
                    "description": str(description or ""),
                    "input_schema": parameters,
                }
            )

        return formatted


    @staticmethod
    def _parse_reponse(response: Any) -> LLMReponse:
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMReponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage={"input_tokens": getattr(response.usage, "input_tokens", 0),
                   "output_tokens": getattr(response.usage, "output_tokens", 0)}
            if getattr(response, "usage", None) else {},
        )

    async def aclose(self):
        await self._client.close()
