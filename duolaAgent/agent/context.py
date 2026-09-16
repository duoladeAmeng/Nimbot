"""Assemble identity, workspace guidance, memory, skills and conversation context."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from duolaAgent.agent.memory import MemoryStore
from duolaAgent.agent.skills import SkillsLoader
from duolaAgent.utils.prompt_templates import render_template


class ContextBuilder:
    """Build system prompt + replay history + current user message."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md"]

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self) -> str:
        """Build the stable system prompt for each model call."""

        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        skills = self.skills.build_skills_section()
        if skills:
            parts.append(f"# Active Skills\n\n{skills}")

        parts.append(render_template("agent/tool_contract.md"))

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""

        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}"

        return render_template(
            "agent/identity.md",
            workspace_path=workspace_path,
            runtime=f"{runtime}, Python {platform.python_version()}",
            platform_policy=render_template("agent/platform_policy.md", system=system),
        )

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        *,
        current_role: str = "user",
    ) -> list[dict[str, Any]]:
        """Build the complete message list for a single LLM request."""

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(),
            },
        ]
        messages.extend(history)
        current = self.build_current_message(
            current_message,
            current_role=current_role,
        )

        # nanobot merges adjacent same-role messages before sending them to the
        # provider. The mini version keeps the same idea so replay history stays
        # legal for chat APIs without pulling in the full context-governance layer.
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = self._merge_message_content(
                last.get("content"),
                current.get("content"),
            )
            messages[-1] = last
            return messages

        messages.append(current)
        return messages

    @staticmethod
    def build_current_message(
        current_message: str,
        *,
        current_role: str = "user",
    ) -> dict[str, Any]:
        return {
            "role": current_role,
            "content": current_message,
        }

    def _load_bootstrap_files(self) -> str:
        parts: list[str] = []
        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / filename
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## {filename}\n\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            if not left:
                return right
            if not right:
                return left
            return f"{left}\n\n{right}"

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    item if isinstance(item, dict) else {"type": "text", "text": str(item)}
                    for item in value
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Cheap token estimate for context budgeting.

    nanobot uses richer tokenizer-aware helpers. The mini version deliberately
    keeps a transparent estimate: roughly four characters per token, plus role
    overhead.
    """

    text = f"{message.get('role', '')}\n{_content_text(message.get('content'))}"
    if message.get("tool_calls"):
        text += f"\n{message.get('tool_calls')}"
    return max(1, len(text) // 4 + 4)


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("name")
                if value is not None:
                    parts.append(str(value))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
