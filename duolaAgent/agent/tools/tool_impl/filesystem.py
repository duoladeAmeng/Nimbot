from __future__ import annotations

from pathlib import Path
from typing import Any

from duolaAgent.agent.tools.base import Tool, ToolResult
from duolaAgent.agent.tools.file_state import check_read, record_read, record_write
from duolaAgent.agent.tools.schema import (
    boolean_schema,
    integer_schema,
    string_schema,
    tool_parameters_schema,
)
from duolaAgent.security.workspace_access import current_tool_workspace
from duolaAgent.security.workspace_policy import resolve_allowed_path


class _FsTool(Tool):
    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        restrict_to_workspace: bool = False,
        allowed_dir: str | Path | None = None,
        extra_write_allowed_files: list[str | Path] | None = None,
        require_read: bool = False,
    ) -> None:
        self.workspace = Path(workspace or ".").resolve()
        self.require_read = require_read
        self.restrict_to_workspace = restrict_to_workspace
        self.allowed_dir = Path(allowed_dir).resolve() if allowed_dir is not None else None
        self.extra_write_allowed_files = {
            Path(path).absolute()
            for path in (extra_write_allowed_files or [])
        }

    def _resolve(self, path: str) -> Path:
        policy = current_tool_workspace(self.workspace, restrict_to_workspace=self.restrict_to_workspace)
        return resolve_allowed_path(path, workspace=policy.project_path, allowed_root=policy.allowed_root)

    def _is_inside_workspace(self, path: Path) -> bool:
        return path == self.workspace or self.workspace in path.parents

    def _check_write_allowed(self, path: Path) -> None:
        if self.allowed_dir is None:
            return
        if path == self.allowed_dir or self.allowed_dir in path.parents:
            return
        if path in self.extra_write_allowed_files:
            return
        raise PermissionError(f"Path is outside allowed directory: {path}")


class ReadFileTool(_FsTool):
    _DEFAULT_LIMIT = 2000
    _MAX_CHARS = 128_000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a text file. Text output format: LINE_NUM|CONTENT. "
            "Use offset and limit for large files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return tool_parameters_schema(
            properties={
                "path": string_schema("The file path to read"),
                "offset": integer_schema(
                    "Line number to start reading from (1-indexed, default 1)",
                    minimum=1,
                ),
                "limit": integer_schema(
                    "Maximum number of lines to read (default 2000)",
                    minimum=1,
                ),
                "force": boolean_schema(
                    "Compatibility flag; currently ignored.",
                    default=False,
                ),
            },
            required=["path"],
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        path: str | None = None,
        offset: int = 1,
        limit: int | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> str:
        try:
            if not path:
                return ToolResult.error("Error reading file: Unknown path")

            fp = self._resolve(path)
            if not fp.exists():
                return ToolResult.error(f"Error: File not found: {path}")
            if not fp.is_file():
                return ToolResult.error(f"Error: Not a file: {path}")

            start = max(offset, 1)
            cap = limit or self._DEFAULT_LIMIT
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[start - 1 : start - 1 + cap]
            record_read(fp, start, cap)

            output_lines: list[str] = []
            total_chars = 0
            for line_no, line in enumerate(selected, start=start):
                rendered = f"{line_no}|{line}"
                total_chars += len(rendered) + 1
                if total_chars > self._MAX_CHARS:
                    output_lines.append("(truncated)")
                    break
                output_lines.append(rendered)

            if not output_lines:
                return ""

            if start - 1 + cap < len(lines):
                output_lines.append(
                    f"(truncated, showing lines {start}-{start + len(selected) - 1} of {len(lines)})"
                )

            return "\n".join(output_lines)
        except PermissionError as exc:
            return ToolResult.error(f"Error: {exc}")
        except Exception as exc:
            return ToolResult.error(f"Error reading file: {exc}")


class WriteFileTool(_FsTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Create a new file or replace an entire file with the provided "
            "content. Creates parent directories as needed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return tool_parameters_schema(
            properties={
                "path": string_schema("The file path to write to"),
                "content": string_schema("The content to write"),
            },
            required=["path", "content"],
        )

    async def execute(
        self,
        path: str | None = None,
        content: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if not path:
                return ToolResult.error("Error writing file: Unknown path")
            if content is None:
                return ToolResult.error("Error writing file: Unknown content")

            fp = self._resolve(path)
            self._check_write_allowed(fp)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            record_write(fp)
            return f"Successfully wrote {len(content)} characters to {fp}"
        except PermissionError as exc:
            return ToolResult.error(f"Error: {exc}")
        except Exception as exc:
            return ToolResult.error(f"Error writing file: {exc}")


class EditFileTool(_FsTool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Replace one exact text span in a UTF-8 text file."

    @property
    def parameters(self) -> dict[str, Any]:
        return tool_parameters_schema(
            properties={
                "path": string_schema("The file path to edit"),
                "old_text": string_schema("Exact text to replace"),
                "new_text": string_schema("Replacement text"),
            },
            required=["path", "old_text", "new_text"],
        )

    async def execute(
        self,
        path: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if not path:
                return ToolResult.error("Error editing file: Unknown path")
            if old_text is None:
                return ToolResult.error("Error editing file: Unknown old_text")
            if new_text is None:
                return ToolResult.error("Error editing file: Unknown new_text")

            fp = self._resolve(path)
            self._check_write_allowed(fp)
            if not fp.exists():
                return ToolResult.error(f"Error: File not found: {path}")
            read_error = check_read(fp)
            if self.require_read and read_error:
                return ToolResult.error(read_error)
            content = fp.read_text(encoding="utf-8", errors="replace")
            if old_text not in content:
                return ToolResult.error("Error editing file: old_text not found")
            fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
            record_write(fp)
            return f"Successfully edited {fp}"
        except PermissionError as exc:
            return ToolResult.error(f"Error: {exc}")
        except Exception as exc:
            return ToolResult.error(f"Error editing file: {exc}")


class ApplyPatchTool(_FsTool):
    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return (
            "Apply structured text edits. Each edit has path, action "
            "(add, replace, delete), and text fields."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return tool_parameters_schema(
            properties={
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": string_schema("The file path to edit"),
                            "action": string_schema("add, replace, or delete"),
                            "old_text": string_schema("Exact text to replace or delete"),
                            "new_text": string_schema("Text to add or insert"),
                        },
                        "required": ["path", "action"],
                    },
                }
            },
            required=["edits"],
        )

    async def execute(self, edits: list[dict[str, Any]] | None = None, **kwargs: Any) -> str:
        try:
            if not isinstance(edits, list) or not edits:
                return ToolResult.error("Error applying patch: edits must be a non-empty list")

            changed: list[Path] = []
            for edit in edits:
                if not isinstance(edit, dict):
                    return ToolResult.error("Error applying patch: edit must be an object")
                path = edit.get("path")
                action = str(edit.get("action") or "").lower()
                if not isinstance(path, str) or not path:
                    return ToolResult.error("Error applying patch: edit path is required")
                fp = self._resolve(path)
                self._check_write_allowed(fp)
                old_text = edit.get("old_text")
                new_text = edit.get("new_text")

                if action == "add":
                    if new_text is None:
                        return ToolResult.error("Error applying patch: add requires new_text")
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    if fp.exists():
                        return ToolResult.error(f"Error applying patch: file already exists: {path}")
                    fp.write_text(str(new_text), encoding="utf-8")
                elif action == "replace":
                    if old_text is None or new_text is None:
                        return ToolResult.error("Error applying patch: replace requires old_text and new_text")
                    if not fp.exists():
                        return ToolResult.error(f"Error applying patch: file not found: {path}")
                    content = fp.read_text(encoding="utf-8", errors="replace")
                    if str(old_text) not in content:
                        return ToolResult.error(f"Error applying patch: old_text not found in {path}")
                    fp.write_text(content.replace(str(old_text), str(new_text), 1), encoding="utf-8")
                elif action == "delete":
                    if old_text is None:
                        return ToolResult.error("Error applying patch: delete requires old_text")
                    if not fp.exists():
                        return ToolResult.error(f"Error applying patch: file not found: {path}")
                    content = fp.read_text(encoding="utf-8", errors="replace")
                    if str(old_text) not in content:
                        return ToolResult.error(f"Error applying patch: old_text not found in {path}")
                    fp.write_text(content.replace(str(old_text), "", 1), encoding="utf-8")
                else:
                    return ToolResult.error(f"Error applying patch: unknown action {action!r}")
                changed.append(fp)

            return "Patch applied: " + ", ".join(str(path) for path in changed)
        except PermissionError as exc:
            return ToolResult.error(f"Error: {exc}")
        except Exception as exc:
            return ToolResult.error(f"Error applying patch: {exc}")


class ListDirTool(_FsTool):
    _DEFAULT_MAX = 200
    _IGNORE_DIRS = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "venv",
    }

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. Set recursive=true to explore "
            "nested structure. Common noise directories are auto-ignored."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return tool_parameters_schema(
            properties={
                "path": string_schema("The directory path to list"),
                "recursive": boolean_schema(
                    "Recursively list all files (default false)",
                    default=False,
                ),
                "max_entries": integer_schema(
                    "Maximum entries to return (default 200)",
                    minimum=1,
                ),
            },
            required=["path"],
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        path: str | None = None,
        recursive: bool = False,
        max_entries: int | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            if path is None:
                return ToolResult.error("Error listing directory: Unknown path")

            dp = self._resolve(path)
            if not dp.exists():
                return ToolResult.error(f"Error: Directory not found: {path}")
            if not dp.is_dir():
                return ToolResult.error(f"Error: Not a directory: {path}")

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            if recursive:
                iterator = sorted(dp.rglob("*"))
                for item in iterator:
                    if any(part in self._IGNORE_DIRS for part in item.parts):
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue
                    total += 1
                    if len(items) < cap:
                        suffix = "/" if item.is_dir() else ""
                        items.append(f"{item.name}{suffix}")

            if not items and total == 0:
                return f"Directory {path} is empty"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as exc:
            return ToolResult.error(f"Error: {exc}")
        except Exception as exc:
            return ToolResult.error(f"Error listing directory: {exc}")
