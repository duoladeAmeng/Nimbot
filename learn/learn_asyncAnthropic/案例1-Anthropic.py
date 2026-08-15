import os
from anthropic import Anthropic
from dotenv import load_dotenv

import time

load_dotenv(override=True)


# 1. 初始化同步客户端
client = Anthropic(
    base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    api_key=os.environ.get("ANTHROPIC_API_KEY")  # 建议将 API Key 存放在环境变量中
)
model_id=os.environ.get("ANTHROPIC_MODEL_ID")

def get_sync_response(prompt_text):
    print(f"开始同步请求: {prompt_text}")
    # 2. 发起请求（程序会在这里卡住，直到拿到结果）
    response = client.messages.create(
        model=model_id,
        max_tokens=100,
        messages=[
            {"role": "user", "content": prompt_text}
        ],
        system="只输出纯文本的答案"
    )
    print(f"同步请求完成: {prompt_text}")
    print(response.content[0].text)

if __name__ == "__main__":
    start_time = time.time()

    # 因为是同步的，这三个请求会挨个排队执行
    ans1 = get_sync_response("讲个一句话的冷笑话")
    ans2 = get_sync_response("写一句鼓励人的短句")
    ans3 = get_sync_response("描述一下夏天的微风")

    print(f"\n全部完成，总耗时: {time.time() - start_time:.2f} 秒")