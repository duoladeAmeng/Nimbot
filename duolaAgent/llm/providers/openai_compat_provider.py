"""Cancellable HTTP Chat Completions provider with SSE tool-call assembly."""
from __future__ import annotations

import json

import httpx

from duolaAgent.llm.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from duolaAgent.utils.helpers import normalize_messages


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key=None, api_base=None, *, api_key_header="Authorization", client=None):
        super().__init__(api_key, api_base)
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key_header = api_key_header
        headers = {}
        if api_key:
            headers[api_key_header] = f"Bearer {api_key}" if api_key_header.lower() == "authorization" else api_key
        self._client = client or httpx.AsyncClient(headers=headers, timeout=self.timeout_s)

    @staticmethod
    def _format_messages(messages):
        return [{k: v for k, v in m.items() if k in
                 {"role", "content", "tool_calls", "tool_call_id", "name"}}
                for m in normalize_messages(messages)]

    async def chat(self, messages, model, max_tokens=4096, temperature=0.7, tools=None):
        payload = dict(messages=self._format_messages(messages), model=model,
                       max_tokens=max_tokens, temperature=temperature)
        if tools:
            payload["tools"] = tools
        response = await self._client.post(f"{self.api_base}/chat/completions", json=payload)
        response.raise_for_status()
        return self._parse_response(response.json())

    async def chat_stream(self, messages, model, max_tokens=4096, temperature=0.7,
                          tools=None, on_content_delta=None):
        payload = dict(messages=self._format_messages(messages), model=model, stream=True,
                       max_tokens=max_tokens, temperature=temperature)
        if tools:
            payload["tools"] = tools
        parts, calls, usage, finish = [], {}, {}, None
        async with self._client.stream("POST", f"{self.api_base}/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                data = json.loads(raw)
                if data.get("error"):
                    raise ValueError(str(data["error"]))
                usage = data.get("usage") or usage
                # Some OpenAI-compatible APIs emit usage/heartbeat chunks with
                # explicit JSON nulls instead of omitting these optional fields.
                for choice in data.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    if choice.get("index", 0) != 0:
                        continue
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        continue
                    if delta.get("content"):
                        parts.append(delta["content"])
                        if on_content_delta:
                            await on_content_delta(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        if not isinstance(tc, dict):
                            continue
                        saved = calls.setdefault(tc["index"], {"id": "", "type": "function",
                                                              "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            saved["id"] = tc["id"]
                        for key in ("name", "arguments"):
                            saved["function"][key] += tc.get("function", {}).get(key) or ""
        if finish is None:
            raise TimeoutError("Stream ended before a finish marker")
        return self._parse_response({"choices": [{"message": {
            "content": "".join(parts) or None, "tool_calls": [calls[k] for k in sorted(calls)]},
            "finish_reason": finish}], "usage": usage})

    @staticmethod
    def _parse_response(data):
        choices = data.get("choices") or []
        if (not choices or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("message"), dict)):
            raise ValueError("Chat Completions response missing message")
        message = choices[0]["message"]
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass  # Registry returns a schema error to the model.
            calls.append(ToolCallRequest(raw.get("id") or "", function.get("name"), arguments))
        return LLMResponse(message.get("content"), calls, choices[0].get("finish_reason") or "stop",
                           usage=data.get("usage") or {})

    async def aclose(self):
        await self._client.aclose()
