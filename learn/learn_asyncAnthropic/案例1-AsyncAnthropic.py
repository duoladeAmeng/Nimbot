import asyncio
import os
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import time

load_dotenv(override=True)

# 1. 初始化异步客户端
async_client = AsyncAnthropic(
    base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

model_id=os.environ.get("ANTHROPIC_MODEL_ID")


async def get_async_response(prompt_text):
    print(f"开始异步请求: {prompt_text}")
    # 2. 发起请求（注意前方的 await，它会将控制权交还给事件循环，不去阻塞其他任务）
    response = await async_client.messages.create(
        model=model_id,
        max_tokens=100,
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        system="只输出纯文本的答案"
    )
    print(f"异步请求完成: {prompt_text}")
    print(response.content[0].text)


async def main():
    start_time = time.time()

    # 3. 使用 asyncio.gather 同时并发这三个请求
    # 它们会同时向 Anthropic 服务器发起请求，无需排队
    tasks = [
        get_async_response("讲个一句话的冷笑话"),
        get_async_response("写一句鼓励人的短句"),
        get_async_response("描述一下夏天的微风")
    ]

    # 等待所有并发任务完成
    results = await asyncio.gather(*tasks)

    print(f"\n全部完成，总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    # 运行异步主程序
    asyncio.run(main())