import asyncio

from duolaAgent.llm.providers.anthropic_provider import  AnthropicProvider
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
base_url = os.environ.get("ANTHROPIC_BASE_URL")
api_key = os.environ.get("ANTHROPIC_API_KEY")  # 建议将 API Key 存放在环境变量中
model_id=os.environ.get("ANTHROPIC_MODEL_ID")


async def on_delta(text: str) -> None:
    print(text, end="", flush=True)


async def testAnthropicProvider(messages):
    provider = AnthropicProvider(api_key=api_key, api_base=base_url)
    re = await provider.chat_stream(messages=messages, model=model_id, on_content_delta=on_delta)
    print()  # 流结束后换行
    print(re)


messages=[
    {"role":"user","content":"我是张三，一名Java程序员，请夸奖我"}
]

asyncio.run(testAnthropicProvider(messages=messages))