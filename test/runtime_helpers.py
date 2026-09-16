import asyncio
from copy import deepcopy

from duolaAgent.llm.providers.base import LLMProvider, LLMResponse
from duolaAgent.agent.tools.base import Tool
from duolaAgent.bus.message import InboundMessage


class ScriptedProvider(LLMProvider):
    def __init__(self, responses=None, handler=None):
        super().__init__()
        self.responses = list(responses or [])
        self.handler = handler
        self.calls = []
        self.retry_attempts = 1

    async def chat(self, **kwargs):
        self.calls.append(deepcopy({k: v for k, v in kwargs.items() if k != "on_content_delta"}))
        if self.handler:
            return await self.handler(kwargs)
        if not self.responses:
            return LLMResponse("done")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FunctionTool(Tool):
    def __init__(self, name, fn, *, safe=False, schema=None):
        self._name, self.fn, self._safe = name, fn, safe
        self._schema = schema or {"type": "object", "properties": {}, "additionalProperties": False}

    @property
    def name(self):
        return self._name

    description = "Test capability"

    @property
    def parameters(self):
        return self._schema

    @property
    def concurrency_safe(self):
        return self._safe

    async def execute(self, **kwargs):
        return await self.fn(**kwargs)


def inbound(content="hello", key="one", **metadata):
    return InboundMessage("cli", "user", key, content, metadata=metadata)


async def wait_until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)
