"""Shared token estimates, legal replay boundaries and tool-result offload."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def truncate_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max(0, max_chars - 15)] + "\n[truncated]"


def estimate_message_tokens(message: dict[str, Any]) -> int:
    # Explicit fallback estimate, not an exact tokenizer count. Include tool schemas/arguments.
    return max(1, (len(json.dumps(message, ensure_ascii=False)) + 3) // 4) + 4


def estimate_prompt_tokens_chain(provider, model, messages, tools=None) -> tuple[int, str]:
    counter = getattr(provider, "count_tokens", None)
    if callable(counter):
        try:
            count = counter(messages=messages, tools=tools, model=model)
            if isinstance(count, int) and count > 0:
                return count, "provider"
        except Exception:
            pass
    return sum(estimate_message_tokens(m) for m in messages) + (
        estimate_message_tokens({"tools": tools}) if tools else 0
    ), "character-estimate"


def find_legal_message_start(messages: list[dict[str, Any]]) -> int:
    for i, message in enumerate(messages):
        if message.get("role") != "tool":
            return i
    return len(messages)


def is_user_turn(message: dict[str, Any]) -> bool:
    if message.get("role") != "user" or message.get("_command"):
        return False
    content = message.get("content")
    return not (isinstance(content, list) and content and all(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    ))


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert legacy Anthropic history to canonical messages on a deep copy."""
    result = []
    for message in deepcopy(messages):
        content = message.get("content")
        if not isinstance(content, list):
            result.append(message)
            continue
        regular, calls, outputs = [], [], []
        for block in content:
            if not isinstance(block, dict):
                regular.append(block)
            elif block.get("type") == "tool_use":
                calls.append({"id": block.get("id"), "type": "function", "function": {
                    "name": block.get("name"), "arguments": json.dumps(block.get("input", {}))}})
            elif block.get("type") == "tool_result":
                outputs.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                                "content": block.get("content", ""),
                                "is_error": bool(block.get("is_error"))})
            else:
                regular.append(block)
        if regular or calls or not outputs:
            message["content"] = regular or ""
            if calls:
                message["tool_calls"] = calls
            result.append(message)
        result.extend(outputs)
    return result


def maybe_persist_tool_result(workspace, session_key, tool_call_id, result, *, max_chars):
    if not isinstance(result, str) or len(result) <= max_chars or workspace is None:
        return result
    session = hashlib.sha256((session_key or "default").encode()).hexdigest()[:16]
    digest = hashlib.sha256((tool_call_id + result).encode()).hexdigest()[:24]
    path = Path(workspace) / ".duola" / "tool-results" / session / f"{digest}.txt"
    from duolaAgent.security.workspace_policy import require_path_within
    require_path_within(path, workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result, encoding="utf-8")
    return f"Tool output saved to {path}. Use read_file with a narrow range.\n" + result[:1000]
