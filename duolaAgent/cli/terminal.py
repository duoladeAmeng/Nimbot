from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

_prompt_session: PromptSession[str] | None = None


def init_prompt_session(history_file: str = ".duola_history") -> None:
    """初始化终端输入会话。"""
    global _prompt_session

    _prompt_session = PromptSession(
        history=FileHistory(history_file),
    )


async def read_input(prompt: str = "You: ") -> str:
    global _prompt_session

    if _prompt_session is None:
        init_prompt_session()

    assert _prompt_session is not None
    return await _prompt_session.prompt_async(prompt)


console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}


def is_exit_command(command: str) -> bool:
    """判断是否退出。"""
    return command.strip().lower() in EXIT_COMMANDS


def print_welcome(
    bot_name: str = "Duola",
    model: str = "deepseek-chat",
) -> None:
    """打印启动提示。"""
    console.print(
        f"🐻 [bold]{bot_name}[/bold] Interactive mode "
        f"[bold blue]({model})[/bold blue] "
        "— type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit\n"
    )


def print_agent_response(
    content: str,
    bot_name: str = "Duola",
    render_markdown: bool = True,
) -> None:
    """打印 Agent 回复。"""
    console.print()
    console.print(f"[cyan]🐻 {bot_name}[/cyan]")

    if render_markdown:
        console.print(Markdown(content))
    else:
        console.print(content)

    console.print()


def print_agent_header(bot_name: str = "Duola") -> None:
    console.print()
    console.print(f"[cyan]🐻 {bot_name}[/cyan]")


def print_stream_delta(content: str) -> None:
    console.print(content, end="")


def print_error(content: str) -> None:
    console.print(f"[red]{content}[/red]")


def print_goodbye() -> None:
    """打印退出提示。"""
    console.print("\nGoodbye!")
