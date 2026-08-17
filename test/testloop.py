import asyncio
import os

from dotenv import load_dotenv

from duolaAgent.agent.loop import AgentLoop
from duolaAgent.bus.message import InboundMessage
from duolaAgent.bus.queue import MessageBus
from duolaAgent.llm.providers.anthropic_provider import AnthropicProvider


load_dotenv(override=True)

base_url = os.environ.get("ANTHROPIC_BASE_URL")
api_key = os.environ.get("ANTHROPIC_API_KEY")
model_id = os.environ.get("ANTHROPIC_MODEL_ID")


bus = MessageBus()

provider = AnthropicProvider(
    api_key=api_key,
    api_base=base_url,
)

loop = AgentLoop(
    bus=bus,
    provider=provider,
    model=model_id,
)


async def main():

    loop_task = asyncio.create_task(
        loop.run()
    )

    try:

        while True:

            text = await asyncio.to_thread(
                input,
                "You: "
            )

            command = text.strip()

            if command.lower() in {
                "exit",
                "quit",
            }:
                break

            if not command:
                continue

            await bus.publish_inbound(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="test",
                    content=text,
                )
            )

            msg = await bus.consume_outbound()

            print(
                f"Agent: {msg.content}"
            )

    finally:

        loop.stop()
        loop_task.cancel()

        await asyncio.gather(
            loop_task,
            return_exceptions=True,
        )

if __name__ == "__main__":
    asyncio.run(main())


