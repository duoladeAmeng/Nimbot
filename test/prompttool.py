import asyncio
from prompt_toolkit import PromptSession


async def main():
    session = PromptSession()

    while True:
        text = await session.prompt_async("You: ")

        print("收到:", text)


asyncio.run(main())