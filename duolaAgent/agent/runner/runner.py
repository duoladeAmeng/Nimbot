import asyncio
from typing import Any
from duolaAgent.llm.providers.base import LLMReponse
from test.testanthropicapi import messages
from .specs import AgentRunSpec, AgentRunResult


class AgentRunner:

    async def run(self, spec:AgentRunSpec) -> AgentRunResult:
        messages=list(spec.initial_messages)
        try:
            result= await self.run_core(spec,messages)
        except asyncio.CancelledError as exc:
            raise
        return result


    async def run_core(
            self,
            spec:AgentRunSpec,
            messages:list[dict[str, Any]],
            )->AgentRunResult:
        response= await self._request_model(spec,messages)
        return AgentRunResult(
            final_content=response.content
        )

    # 构造请求参数
    def _build_request_kwargs(
            self,
            spec:AgentRunSpec,
            messages:list[dict[str, Any]],
    ):
        kwargs={
            "messages": messages,
            "model":spec.runtime.model
        }
        generation=spec.runtime.generation
        kwargs["temperature"]=generation.temperature
        kwargs["max_tokens"]=generation.max_tokens
        return kwargs

    async def _request_model(
            self,
            spec:AgentRunSpec,
            messages:list[dict[str, Any]],
    ):
        # 构造参数
        kwargs=self._build_request_kwargs(spec,messages)
        # 创建协程对象
        coro=spec.runtime.privider.chat(
           **kwargs
        )
        try:
            response= await coro
        except asyncio.TimeoutError:
            return LLMReponse(
                content=f"Error calling LLM: timed out",
            )


