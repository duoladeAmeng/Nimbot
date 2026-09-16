"""Canonical transcript to Anthropic wire messages."""
import json

from duolaAgent.utils.helpers import normalize_messages


def to_anthropic_messages(messages):
    result = []
    for message in normalize_messages(messages):
        role = message["role"]
        if role == "system":
            result.append(message)
            continue
        content = message.get("content")
        blocks = list(content) if isinstance(content, list) else (
            [{"type": "text", "text": str(content)}] if content else [])
        if role == "tool":
            role = "user"
            blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                       "content": content or "", "is_error": bool(message.get("is_error"))}]
        for call in message.get("tool_calls") or []:
            args = call["function"].get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_invalid_arguments": args}
            blocks.append({"type": "tool_use", "id": call["id"],
                           "name": call["function"]["name"], "input": args})
        if not blocks:
            continue
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({"role": role, "content": blocks})
    return result
