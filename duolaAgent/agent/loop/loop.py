from duolaAgent.agent.runner.runner import AgentRunner
from duolaAgent.agent.runner.specs import AgentRunSpec
from duolaAgent.llm.providers.base import LLMProvider
from duolaAgent.llm.llm_runtime import LLMRunTime

class AgentLoop:
    def __init__(
            self,
            provider: LLMProvider,
            model: str
    ):
        self.runner = AgentRunner()

    async def _run_agent_loop(
            self,
            initial_messages: list[dict],
            runtime: LLMRunTime,
    ):
        result= await self.runner.run(
            AgentRunSpec(
                initial_messages=initial_messages,
                runtime=runtime
            )
        )
        return result.final_content
