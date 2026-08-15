# agent命令对应的逻辑

import asyncio

from duolaAgent.cli import terminal

async def run_interactive():
    terminal.init_prompt_session()
    terminal.print_welcome(
        bot_name="Duola",
        model="deepseek-chat",
    )
    try:
        while True:
            user_input = await terminal.read_input()

            command = user_input.strip()

            if not command:
                continue

            if terminal.is_exit_command(command):
                terminal.print_goodbye()
                break

            # 这里以后换成 AgentLoop
            response = f"你刚才说的是：{user_input}"

            terminal.print_agent_response(
                response,
                bot_name="Duola",
            )

    except (KeyboardInterrupt, EOFError):
        terminal.print_goodbye()
def agent():
    asyncio.run(run_interactive())