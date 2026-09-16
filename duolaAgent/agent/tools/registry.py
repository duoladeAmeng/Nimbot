from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duolaAgent.agent.tools.base import Tool, ToolResult
from duolaAgent.agent.tools.tool_impl.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from duolaAgent.agent.tools.tool_impl.shell import ExecTool


def is_tool_error_result(result: Any) -> bool:
    return isinstance(result, ToolResult) and result.is_error


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_definitions(self) -> list[dict[str, Any]]:
        if self._cached_definitions is None:
            self._cached_definitions = sorted(
                (tool.to_schema() for tool in self._tools.values()),
                key=self._schema_name,
            )
        return self._cached_definitions

    async def execute(self, name: str, params: Any) -> Any:
        tool, params, error = self.prepare_call(name, params)
        if error is not None:
            return error

        try:
            assert tool is not None
            return await tool.execute(**params)
        except Exception as exc:
            return ToolResult.error(f"Error executing {name}: {exc}")

    def prepare_call(
        self,
        name: str,
        params: Any,
    ) -> tuple[Tool | None, dict[str, Any], ToolResult | None]:
        tool = self._tools.get(name)
        if tool is None:
            suggestion = self._suggest_name(name)
            hint = f" Did you mean '{suggestion}'?" if suggestion else ""
            available = ", ".join(self.tool_names)
            return None, {}, ToolResult.error(
                f"Error: Tool '{name}' not found.{hint} Available: {available}"
            )

        params = self._coerce_params(params)
        if not isinstance(params, dict):
            return tool, {}, ToolResult.error(
                f"Error: Tool '{name}' parameters must be a JSON object."
            )

        try:
            params = tool.cast_params(params)
            errors = tool.validate_params(params)
        except (ValueError, TypeError) as exc:
            errors = [str(exc)]
        if errors:
            return tool, params, ToolResult.error(
                f"Error: Invalid parameters for {name}: " + "; ".join(errors)
            )

        return tool, params, None

    def _validate_required(
        self,
        tool: Tool,
        params: dict[str, Any],
    ) -> ToolResult | None:
        required = tool.parameters.get("required", [])
        if not isinstance(required, list):
            return None

        missing = [
            field
            for field in required
            if field not in params or params[field] is None
        ]
        if missing:
            return ToolResult.error(
                f"Error: Missing required parameters for tool '{tool.name}': "
                + ", ".join(str(field) for field in missing)
            )

        return None

    @staticmethod
    def _coerce_params(params: Any) -> Any:
        if params is None:
            return {}
        if not isinstance(params, str):
            return params

        stripped = params.strip()
        if not stripped:
            return {}
        if not stripped.startswith(("{", "[")):
            return params

        try:
            return json.loads(stripped)
        except Exception:
            return params

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        function = schema.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    @staticmethod
    def _lookup_key(name: str) -> str:
        return "".join(ch.lower() for ch in str(name or "") if ch.isalnum())

    def _suggest_name(self, name: str) -> str | None:
        key = self._lookup_key(name)
        matches = [
            registered
            for registered in self._tools
            if self._lookup_key(registered) == key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def create_default_tool_registry(
    workspace: str | Path | None = None,
    *,
    restrict_to_workspace: bool = False,
    exec_session_manager=None,
    plugins: bool = True,
    plugin_scope: str = "core",
    tool_config=None,
) -> ToolRegistry:
    from duolaAgent.agent.tools.exec_session import (
        ExecSessionManager,
        ListExecSessionsTool,
        WriteStdinTool,
    )
    from duolaAgent.agent.tools.loader import load_tool_plugins
    from duolaAgent.agent.tools.tool_impl.filesystem import ApplyPatchTool, EditFileTool
    from duolaAgent.agent.tools.web import WebFetchTool
    manager = exec_session_manager or ExecSessionManager()
    registry = ToolRegistry()
    registry.register(ReadFileTool(workspace, restrict_to_workspace=restrict_to_workspace))
    registry.register(WriteFileTool(workspace, restrict_to_workspace=restrict_to_workspace))
    registry.register(ListDirTool(workspace, restrict_to_workspace=restrict_to_workspace))
    registry.register(EditFileTool(workspace, restrict_to_workspace=restrict_to_workspace))
    registry.register(ApplyPatchTool(workspace, restrict_to_workspace=restrict_to_workspace))
    from duolaAgent.agent.tools.context import ToolContext
    from duolaAgent.config.schema import ToolsConfig
    config = tool_config.model_copy(deep=True) if tool_config else ToolsConfig()
    config.restrict_to_workspace = restrict_to_workspace
    if config.exec.enable:
        registry.register(ExecTool.create(ToolContext(config=config, workspace=str(workspace or "."),
                                                      exec_session_manager=manager)))
        registry.register(WriteStdinTool(manager=manager))
        registry.register(ListExecSessionsTool(manager=manager))
    registry.register(WebFetchTool())
    if plugins:
        load_tool_plugins(registry, workspace=Path(workspace or "."), config=config, scope=plugin_scope)
    return registry
