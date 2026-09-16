import asyncio

from duolaAgent.agent.loop import AgentLoop
from duolaAgent.bus.message import InboundMessage
from duolaAgent.bus.queue import MessageBus
from duolaAgent.cli import terminal
from duolaAgent.config.loader import load_runtime_config
from duolaAgent.config.schema import Config


def _load_config(
    *,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    workspace: str | None = None,
) -> Config:
    return load_runtime_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        workspace=workspace,
    )


def _build_loop(config: Config, *, config_loader=None) -> tuple[AgentLoop, MessageBus]:
    from duolaAgent.agent.model_runtime import ModelRuntimeResolver
    from duolaAgent.security.network import configure_ssrf_whitelist
    bus = MessageBus()
    defaults = config.agents.defaults
    resolver = ModelRuntimeResolver(config=config, config_loader=config_loader or load_runtime_config)
    runtime = resolver.admit()
    configure_ssrf_whitelist(config.tools.ssrf_whitelist)
    loop = AgentLoop(
        bus=bus, provider=runtime.provider, model=runtime.model,
        runtime_resolver=resolver, workspace=config.workspace_path,
        max_tool_iterations=defaults.max_tool_iterations,
        max_context_tokens=defaults.max_context_tokens,
        compact_target_tokens=defaults.compact_target_tokens,
        max_tool_result_chars=defaults.max_tool_result_chars,
        restrict_to_workspace=defaults.restrict_to_workspace,
        concurrent_tools=defaults.concurrent_tools, stream=defaults.stream,
        max_concurrent_sessions=defaults.max_concurrent_sessions,
        max_pending_messages=defaults.max_pending_messages,
        max_subagents=defaults.max_subagents,
        max_continuation_turns=defaults.max_continuation_turns,
        plugins=config.tools.plugins, mcp_configs=config.tools.mcp_servers, tool_config=config.tools,
    )
    return loop, bus


def _normalize_mimo_base_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    lowered = base.lower()
    if lowered.endswith("/anthropic"):
        return base[: -len("/anthropic")] + "/v1"
    if lowered.endswith("/v1"):
        return base
    return base + "/v1"



async def run_interactive(
    *,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    workspace: str | None = None,
) -> None:
    config = _load_config(
        model=model,
        api_key=api_key,
        api_base=api_base,
        workspace=workspace,
    )
    loop, bus = _build_loop(config, config_loader=lambda: _load_config(
        model=model, api_key=api_key, api_base=api_base, workspace=workspace))

    terminal.init_prompt_session()
    terminal.print_welcome(
        bot_name="Duola",
        model=config.agents.defaults.model or "",
    )

    loop_task = asyncio.create_task(loop.run())

    try:
        while True:
            user_input = await terminal.read_input()

            command = user_input.strip()

            if not command:
                continue

            if terminal.is_exit_command(command):
                terminal.print_goodbye()
                break

            await bus.publish_inbound(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="terminal",
                    content=user_input,
                )
            )

            streamed = False
            while True:
                response = await bus.consume_outbound()
                if response.metadata.get("event") == "stream_delta":
                    if not streamed:
                        terminal.print_agent_header("Duola")
                        streamed = True
                    terminal.print_stream_delta(response.content)
                    continue

                if streamed:
                    terminal.console.print("\n")
                else:
                    terminal.print_agent_response(
                        response.content,
                        bot_name="Duola",
                    )
                break

    except (KeyboardInterrupt, EOFError):
        terminal.print_goodbye()
    finally:
        loop.stop()
        loop_task.cancel()
        await asyncio.gather(
            loop_task,
            return_exceptions=True,
        )


def agent(
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    workspace: str | None = None,
) -> None:
    try:
        asyncio.run(
            run_interactive(
                model=model,
                api_key=api_key,
                api_base=api_base,
                workspace=workspace,
            )
        )
    except ValueError as exc:
        terminal.print_error(str(exc))
