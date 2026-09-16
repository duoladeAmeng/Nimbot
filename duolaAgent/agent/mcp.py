from __future__ import annotations

import asyncio
import hashlib
import re
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from duolaAgent.agent.tools.base import Tool, ToolResult
from duolaAgent.agent.tools.registry import ToolRegistry

_SANITIZE_RE = re.compile(r"_+")
_MAX_TOOL_NAME_LENGTH = 64
_HASH_LENGTH = 8


class MCPConnection(Protocol):
    async def aclose(self) -> None:
        ...


@dataclass(slots=True)
class MCPServerConfig:
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    type: str | None = None
    enabled_tools: list[str] = field(default_factory=lambda: ["*"])
    tool_timeout: int = 30


class _OwnedMCPConnection:
    """Enter and exit SDK AnyIO cancel scopes in the same owner task."""
    def __init__(self, name, config, registry):
        self._stop = asyncio.Event()
        self._ready = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._serve(name, config, registry))

    async def _serve(self, name, config, registry):
        try:
            async with AsyncExitStack() as stack:
                session = await _open_mcp_session(config, stack)
                await _register_mcp_capabilities(name, config, session, registry)
                self._ready.set_result(None)
                await self._stop.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                from loguru import logger
                logger.warning("MCP connection {} closed with {}", name, type(exc).__name__)
        finally:
            _unregister_server_tools(registry, name)

    async def ready(self, timeout):
        await asyncio.wait_for(asyncio.shield(self._ready), timeout)

    async def aclose(self):
        self._stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), 5)
        except TimeoutError:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._ready.done() and not self._ready.cancelled():
            self._ready.exception()  # Retrieve initialization errors on timed-out admission.


def _sanitize_name(name: str) -> str:
    return _SANITIZE_RE.sub("_", re.sub(r"[^a-zA-Z0-9_-]", "_", name))


def _limit_tool_name(name: str, max_length: int = _MAX_TOOL_NAME_LENGTH) -> str:
    if len(name) <= max_length:
        return name

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    prefix_length = max_length - _HASH_LENGTH - 1
    return f"{name[:prefix_length]}_{digest}"


def _sanitize_mcp_tool_name(name: str) -> str:
    return _limit_tool_name(_sanitize_name(name))


def _normalize_schema_for_openai(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {},
        }

    normalized = dict(schema)
    normalized.setdefault("type", "object")
    if normalized.get("type") == "object":
        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])
    return normalized


def _text_from_content_block(block: Any) -> str:
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text

    if isinstance(block, Mapping):
        raw_text = block.get("text")
        if isinstance(raw_text, str):
            return raw_text

    return str(block)


def _get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class _MCPWrapperBase(Tool):
    _session: Any
    _server_name: str
    _name: str

    def _set_mcp_connection(self, session: Any, server_name: str) -> None:
        self._session = session
        self._server_name = server_name


class MCPToolWrapper(_MCPWrapperBase):
    def __init__(
        self,
        session: Any,
        server_name: str,
        tool_def: Any,
        tool_timeout: int = 30,
    ) -> None:
        self._set_mcp_connection(session, server_name)
        self._original_name = str(_get_attr_or_key(tool_def, "name", ""))
        self._name = _sanitize_mcp_tool_name(f"mcp_{server_name}_{self._original_name}")
        self._description = str(
            _get_attr_or_key(tool_def, "description", None) or self._original_name
        )
        raw_schema = _get_attr_or_key(tool_def, "inputSchema", None)
        if raw_schema is None:
            raw_schema = _get_attr_or_key(tool_def, "input_schema", None)
        self._parameters = _normalize_schema_for_openai(raw_schema)
        self._tool_timeout = tool_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult.error(f"(MCP tool call timed out after {self._tool_timeout}s)")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult.error(f"(MCP tool call failed: {type(exc).__name__}: {exc})")

        rendered = self._render_call_result(_get_attr_or_key(result, "content", []))
        if _get_attr_or_key(result, "isError", False):
            return ToolResult.error(rendered)
        return rendered

    def _render_call_result(self, content: Any) -> str:
        if content is None:
            return "(no output)"
        if isinstance(content, str):
            return content or "(no output)"
        if not isinstance(content, list):
            return _text_from_content_block(content)

        parts = [_text_from_content_block(block) for block in content]
        return "\n".join(part for part in parts if part) or "(no output)"


class MCPResourceWrapper(_MCPWrapperBase):
    def __init__(
        self,
        session: Any,
        server_name: str,
        resource_def: Any,
        resource_timeout: int = 30,
    ) -> None:
        self._set_mcp_connection(session, server_name)
        self._uri = _get_attr_or_key(resource_def, "uri", "")
        resource_name = str(_get_attr_or_key(resource_def, "name", self._uri))
        self._name = _sanitize_mcp_tool_name(f"mcp_{server_name}_resource_{resource_name}")
        description = _get_attr_or_key(resource_def, "description", None) or resource_name
        self._description = f"[MCP Resource] {description}\nURI: {self._uri}"
        self._parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self._resource_timeout = resource_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        try:
            result = await asyncio.wait_for(
                self._session.read_resource(self._uri),
                timeout=self._resource_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult.error(
                f"(MCP resource read timed out after {self._resource_timeout}s)"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult.error(f"(MCP resource read failed: {type(exc).__name__}: {exc})")

        contents = _get_attr_or_key(result, "contents", [])
        if not isinstance(contents, list):
            contents = [contents]
        parts = [_text_from_content_block(block) for block in contents]
        return "\n".join(part for part in parts if part) or "(no output)"


class MCPPromptWrapper(_MCPWrapperBase):
    def __init__(
        self,
        session: Any,
        server_name: str,
        prompt_def: Any,
        prompt_timeout: int = 30,
    ) -> None:
        self._set_mcp_connection(session, server_name)
        self._prompt_name = str(_get_attr_or_key(prompt_def, "name", ""))
        self._name = _sanitize_mcp_tool_name(f"mcp_{server_name}_prompt_{self._prompt_name}")
        description = _get_attr_or_key(prompt_def, "description", None) or self._prompt_name
        self._description = (
            f"[MCP Prompt] {description}\n"
            "Returns a filled prompt template that can be used as a workflow guide."
        )
        self._parameters = self._build_parameters(_get_attr_or_key(prompt_def, "arguments", []))
        self._prompt_timeout = prompt_timeout

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        try:
            result = await asyncio.wait_for(
                self._session.get_prompt(self._prompt_name, arguments=kwargs),
                timeout=self._prompt_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult.error(f"(MCP prompt call timed out after {self._prompt_timeout}s)")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult.error(f"(MCP prompt call failed: {type(exc).__name__}: {exc})")

        messages = _get_attr_or_key(result, "messages", [])
        if not isinstance(messages, list):
            messages = [messages]

        parts: list[str] = []
        for message in messages:
            content = _get_attr_or_key(message, "content", message)
            if isinstance(content, list):
                parts.extend(_text_from_content_block(block) for block in content)
            else:
                parts.append(_text_from_content_block(content))
        return "\n".join(part for part in parts if part) or "(no output)"

    @staticmethod
    def _build_parameters(arguments: Any) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for argument in arguments or []:
            name = _get_attr_or_key(argument, "name", None)
            if not isinstance(name, str) or not name:
                continue
            properties[name] = {
                "type": "string",
                "description": str(_get_attr_or_key(argument, "description", "") or ""),
            }
            if bool(_get_attr_or_key(argument, "required", False)):
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


def _coerce_server_config(config: MCPServerConfig | Mapping[str, Any]) -> MCPServerConfig:
    if isinstance(config, MCPServerConfig):
        return config

    if "enabled_tools" in config:
        enabled_tools = list(config.get("enabled_tools") or [])
    elif "enabledTools" in config:
        enabled_tools = list(config.get("enabledTools") or [])
    else:
        enabled_tools = ["*"]

    return MCPServerConfig(
        command=config.get("command"),
        args=list(config.get("args") or []),
        env=dict(config.get("env") or {}),
        cwd=config.get("cwd"),
        url=config.get("url"),
        headers=dict(config.get("headers") or {}),
        type=config.get("type"),
        enabled_tools=enabled_tools,
        tool_timeout=int(config.get("tool_timeout") or config.get("toolTimeout") or 30),
    )


async def connect_mcp_servers(
    mcp_servers: Mapping[str, MCPServerConfig | Mapping[str, Any]],
    registry: ToolRegistry,
) -> dict[str, MCPConnection]:
    connections: dict[str, MCPConnection] = {}

    for name, raw_config in mcp_servers.items():
        config = _coerce_server_config(raw_config)
        connection = _OwnedMCPConnection(name, config, registry)
        try:
            await connection.ready(config.tool_timeout)
        except asyncio.CancelledError:
            await connection.aclose()
            for opened in connections.values():
                await opened.aclose()
            raise
        except Exception as exc:
            await connection.aclose()
            from loguru import logger
            logger.warning("MCP {} unavailable: {}", name, type(exc).__name__)
            continue
        connections[name] = connection

    return connections


def _safe_mcp_http_client(headers=None, timeout=None, auth=None):
    # MCP 2 uses httpx2, while Anthropic and Web use httpx. Validate the SDK's
    # every outgoing request, including SSE endpoint announcements and redirects.
    import httpx2

    from duolaAgent.security.network import (
        PinnedDNSAsyncTransport,
        pin_resolved_url_dns,
        resolve_url_target,
    )

    class Transport(httpx2.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx2.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            url = str(request.url)
            ok, error, ips = resolve_url_target(url)
            if not ok:
                raise httpx2.RequestError(error, request=request)
            async with PinnedDNSAsyncTransport._resolver_lock:
                with pin_resolved_url_dns(url, ips):
                    return await self.inner.handle_async_request(request)

        async def aclose(self):
            await self.inner.aclose()

    return httpx2.AsyncClient(headers=headers, timeout=timeout or 30, auth=auth,
                             transport=Transport(), trust_env=False, follow_redirects=True)


async def _open_mcp_session(config: MCPServerConfig, stack: AsyncExitStack) -> Any:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise RuntimeError("The 'mcp' package is required to connect MCP servers.") from exc

    transport_type = config.type
    if config.url:
        from duolaAgent.security.network import validate_url_target
        ok, error = validate_url_target(config.url)
        if not ok:
            raise ValueError(error)
    if not transport_type:
        if config.command:
            transport_type = "stdio"
        elif config.url:
            transport_type = "sse" if config.url.rstrip("/").endswith("/sse") else "streamableHttp"

    if transport_type == "stdio":
        if not config.command:
            raise ValueError("stdio MCP server requires command")
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env or None,
            cwd=config.cwd,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
    elif transport_type == "sse":
        if not config.url:
            raise ValueError("sse MCP server requires url")
        read, write = await stack.enter_async_context(
            sse_client(config.url, headers=config.headers or None,
                       httpx_client_factory=_safe_mcp_http_client)
        )
    elif transport_type == "streamableHttp":
        if not config.url:
            raise ValueError("streamableHttp MCP server requires url")
        http_client = await stack.enter_async_context(_safe_mcp_http_client(headers=config.headers))
        read, write, _ = await stack.enter_async_context(
            streamable_http_client(config.url, http_client=http_client))
    else:
        raise ValueError(f"unknown MCP transport type: {transport_type}")

    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def _register_mcp_capabilities(
    server_name: str,
    config: MCPServerConfig,
    session: Any,
    registry: ToolRegistry,
) -> int:
    registered = 0
    enabled_tools = set(config.enabled_tools)
    allow_all_tools = "*" in enabled_tools

    tools_result = await session.list_tools()
    for tool_def in getattr(tools_result, "tools", []):
        original_name = str(_get_attr_or_key(tool_def, "name", ""))
        wrapped_name = _sanitize_mcp_tool_name(f"mcp_{server_name}_{original_name}")
        if (
            not allow_all_tools
            and original_name not in enabled_tools
            and wrapped_name not in enabled_tools
        ):
            continue
        registry.register(MCPToolWrapper(session, server_name, tool_def, config.tool_timeout))
        registered += 1

    if not allow_all_tools:
        return registered

    with suppress(Exception):
        resources_result = await session.list_resources()
        for resource_def in getattr(resources_result, "resources", []):
            registry.register(
                MCPResourceWrapper(session, server_name, resource_def, config.tool_timeout)
            )
            registered += 1

    with suppress(Exception):
        prompts_result = await session.list_prompts()
        for prompt_def in getattr(prompts_result, "prompts", []):
            registry.register(
                MCPPromptWrapper(session, server_name, prompt_def, config.tool_timeout)
            )
            registered += 1

    return registered


def _tool_prefix(server_name: str) -> str:
    return _sanitize_name(f"mcp_{server_name}_")


def _tool_belongs_to_server(tool: Tool | None, tool_name: str, server_name: str) -> bool:
    if isinstance(tool, _MCPWrapperBase):
        return getattr(tool, "_server_name", None) == server_name
    return tool_name.startswith(_tool_prefix(server_name))


def _unregister_server_tools(registry: ToolRegistry, server_name: str) -> int:
    removed = 0
    for tool_name in list(registry.tool_names):
        tool = registry.get(tool_name)
        if _tool_belongs_to_server(tool, tool_name, server_name):
            registry.unregister(tool_name)
            removed += 1
    return removed


class MCPProvider:
    def __init__(
        self,
        servers: Mapping[str, MCPServerConfig | Mapping[str, Any]],
        registry: ToolRegistry,
    ) -> None:
        self._servers = {
            name: _coerce_server_config(config)
            for name, config in servers.items()
        }
        self._registry = registry
        self._connections: dict[str, MCPConnection] = {}
        self._runtime_statuses: dict[str, str] = {}

    @property
    def configured_server_names(self) -> set[str]:
        return set(self._servers)

    @property
    def connected_server_names(self) -> set[str]:
        return set(self._connections)

    def runtime_status(self) -> dict[str, str]:
        return dict(self._runtime_statuses)

    async def connect(self) -> None:
        missing = {
            name: config
            for name, config in self._servers.items()
            if name not in self._connections
        }
        for name in missing:
            self._runtime_statuses[name] = "connecting"

        connected = await connect_mcp_servers(missing, self._registry)
        self._connections.update(connected)

        for name in missing:
            self._runtime_statuses[name] = "connected" if name in connected else "failed"

    async def reload(
        self,
        servers: Mapping[str, MCPServerConfig | Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        await self.aclose()
        if servers is not None:
            self._servers = {
                name: _coerce_server_config(config)
                for name, config in servers.items()
            }
        await self.connect()
        return {
            "ok": all(status == "connected" for status in self._runtime_statuses.values()),
            "configured": sorted(self._servers),
            "connected": sorted(self._connections),
            "failed": sorted(
                name for name, status in self._runtime_statuses.items() if status == "failed"
            ),
            "requires_restart": False,
        }

    async def aclose(self) -> None:
        for server_name in list(self._servers):
            _unregister_server_tools(self._registry, server_name)

        connections = list(self._connections.values())
        self._connections.clear()
        self._runtime_statuses.clear()

        for connection in connections:
            await connection.aclose()
