import asyncio

from duolaAgent.bus.message import InboundMessage, OutboundMessage

# 全局异步队列 (Message Bus)
inbound_queue: asyncio.Queue[InboundMessage] = asyncio.Queue()
outbound_queue: asyncio.Queue[OutboundMessage] = asyncio.Queue()

async def wechat_listener():
    """模拟微信网关：接收用户消息并放入 InboundQueue"""
    print("[微信渠道] 启动监听...")
    await asyncio.sleep(1)  # 模拟网络延迟

    # 构造并发送第一条消息
    msg1 = InboundMessage(
        channel="wechat",
        sender_id="user_001",
        chat_id="group_123",
        content="帮我写一段Python代码实现冒泡排序",
        metadata={"msg_id": "wx_msg_998", "is_vip": True}
    )
    print(f"[微信渠道] 收到用户消息: '{msg1.content}' -> 推入 InboundQueue")
    await inbound_queue.put(msg1)


async def wechat_sender():
    """模拟微信发送端：从 OutboundQueue 取出消息并调用微信API发送"""
    print("[微信发送端] 启动，等待处理回复...")
    while True:
        out_msg = await outbound_queue.get()

        # 仅处理本渠道的消息 (路由)
        if out_msg.channel == "wechat":
            print(f"[微信发送端] (向 {out_msg.chat_id} 发送文本) {out_msg.content}")
            if out_msg.reply_to:
                print(f"            * 回复了原始消息 ID: {out_msg.reply_to}")

        outbound_queue.task_done()


# ================= 核心层 (Agent Core) =================

async def agent_loop():
    """模拟 Agent 核心引擎：处理 InboundQueue，输出到 OutboundQueue"""
    print("[Agent核心] 引擎启动，等待事件...")
    while True:
        in_msg = await inbound_queue.get()
        print(f"[Agent核心] 从队列获取任务，SessionKey: {in_msg.session_key}")

        #  模拟大模型推理耗时 (I/O 阻塞被 asyncio.sleep 模拟，不卡死主线程)
        print(f"[Agent核心] (处理 Session {in_msg.session_key}) 正在调用 LLM...")
        await asyncio.sleep(2)

        #  构造最终回复
        reply_content = f"你好，{in_msg.sender_id}！这是你要的冒泡排序代码：\n```python\n# code here\n```"

        # 这里提取 metadata 里的 msg_id 用于精确回复
        original_msg_id = in_msg.metadata.get("msg_id")

        final_msg = OutboundMessage(
            channel=in_msg.channel,
            chat_id=in_msg.chat_id,
            content=reply_content,
            reply_to=original_msg_id
        )
        await outbound_queue.put(final_msg)

        print(f"[Agent核心] 任务处理完成，结果已推入 OutboundQueue")
        inbound_queue.task_done()


# ================= 启动应用 =================

async def main():
    # 并发运行三个主要组件，它们通过 Queue 彻底解耦
    tasks = [
        asyncio.create_task(wechat_listener()),
        asyncio.create_task(wechat_sender()),
        asyncio.create_task(agent_loop())
    ]

    # 运行5秒后退出（仅用于演示）
    await asyncio.sleep(5)
    print("演示结束。")


if __name__ == "__main__":
    asyncio.run(main())