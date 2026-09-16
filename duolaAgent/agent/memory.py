from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import weakref
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from duolaAgent.agent.tools.registry import ToolRegistry
from duolaAgent.agent.tools.tool_impl.filesystem import (
    ApplyPatchTool,
    EditFileTool,
    ReadFileTool,
    WriteFileTool,
)
from duolaAgent.llm.llm_runtime import LLMRunTime
from duolaAgent.session.manager import Session, SessionManager
from duolaAgent.utils.prompt_templates import render_template

_HISTORY_ENTRY_HARD_CAP = 64_000
_RAW_ARCHIVE_MAX_CHARS = 32_000
_ARCHIVE_SUMMARY_MAX_CHARS = 8_000
_DREAM_FILE_EMBED_CAP = 8000
_WORKSPACE_PROMPT_MAX_CHARS = 32_000


def _consolidator_prompt() -> str:
    return render_template("agent/consolidator_archive.md").rstrip()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 17)].rstrip() + "\n... (truncated)"


def _strip_think(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think\b.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<channel\|>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


class DreamRunProgress:
    """Track tool failures that make a Dream run unsafe to advance."""

    def __init__(self) -> None:
        self.had_tool_errors = False

    async def __call__(
        self,
        *_args: Any,
        tool_events: list[dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> None:
        if any(isinstance(event, dict) and event.get("phase") == "error" for event in tool_events or ()):
            self.had_tool_errors = True


class MemoryStore:
    """Nanobot-compatible file store for durable memory and Dream input."""

    _DEFAULT_MAX_HISTORY = 1000
    _DREAM_CONTENT_PATHS = (
        ".duola/memory/SOUL.md",
        ".duola/memory/USER.md",
        ".duola/memory/MEMORY.md",
    )
    _INTERNAL_HISTORY_SESSION_PREFIXES = ("cron:", "dream:")
    _INTERNAL_HISTORY_SESSION_KEYS = {"heartbeat"}

    def __init__(self, workspace: str | Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = Path(workspace).expanduser().resolve()
        self.max_history_entries = max_history_entries
        self.state_dir = self.workspace / ".duola"
        self.memory_dir = self.state_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = self.memory_dir / "SOUL.md"
        self.user_file = self.memory_dir / "USER.md"
        self.cursor_file = self.memory_dir / ".cursor"
        self.dream_cursor_file = self.memory_dir / ".dream_cursor"
        self.legacy_root_memory_dir = self.workspace / "memory"
        self._append_lock = threading.Lock()
        self._maybe_migrate_root_memory_dir()
        self._maybe_migrate_legacy_history()

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    def append_history(
        self,
        entry: str,
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> int:
        limit = max_chars if max_chars is not None else _HISTORY_ENTRY_HARD_CAP
        raw = entry.rstrip()
        if len(raw) > limit:
            raw = _truncate_text(raw, limit)
        content = _strip_think(raw)
        with self._append_lock:
            cursor = self._next_cursor()
            record: dict[str, Any] = {
                "cursor": cursor,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "content": content,
            }
            if session_key:
                record["session_key"] = session_key
            with self.history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.cursor_file.write_text(str(cursor), encoding="utf-8")
        return cursor

    def archive_summary(
        self,
        messages: list[dict[str, Any]],
        *,
        session_key: str,
        reason: str,
    ) -> str:
        summary = self.summarize_messages(messages)
        if not summary:
            return ""
        self.append_history(
            f"[{reason}] {summary}",
            session_key=session_key,
            max_chars=8000,
        )
        return summary

    def raw_archive(
        self,
        messages: list[dict[str, Any]],
        *,
        max_chars: int | None = None,
        session_key: str | None = None,
    ) -> None:
        limit = max_chars if max_chars is not None else _RAW_ARCHIVE_MAX_CHARS
        formatted = _truncate_text(self._format_messages(messages), limit)
        self.append_history(
            f"[RAW] {len(messages)} messages\n{formatted}",
            session_key=session_key,
        )

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        return [entry for entry, cursor in self._iter_valid_entries() if cursor > since_cursor]

    @classmethod
    def _is_internal_history_session(cls, session_key: str | None) -> bool:
        if not session_key:
            return False
        return (
            session_key in cls._INTERNAL_HISTORY_SESSION_KEYS
            or session_key.startswith(cls._INTERNAL_HISTORY_SESSION_PREFIXES)
        )

    def read_recent_history_for_prompt(
        self,
        since_cursor: int,
        *,
        session_key: str | None,
        unified_session: bool = False,
    ) -> list[dict[str, Any]]:
        entries = self.read_unprocessed_history(since_cursor)
        if session_key is None:
            return entries
        if not unified_session:
            return [entry for entry in entries if entry.get("session_key") == session_key]
        return [
            entry
            for entry in entries
            if (entry_session := entry.get("session_key")) == session_key
            or not self._is_internal_history_session(entry_session)
        ]

    def compact_history(self) -> None:
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        last_dream_cursor = self.get_last_dream_cursor()
        first_unprocessed = next(
            (
                index
                for index, entry in enumerate(entries)
                if (cursor := self._valid_cursor(entry.get("cursor"))) is not None
                and cursor > last_dream_cursor
            ),
            len(entries),
        )
        keep_from = min(len(entries) - self.max_history_entries, first_unprocessed)
        self._write_entries(entries[keep_from:])

    def get_last_dream_cursor(self) -> int:
        with suppress(ValueError, OSError):
            return int(self.dream_cursor_file.read_text(encoding="utf-8").strip())
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        self.dream_cursor_file.write_text(str(max(0, cursor)), encoding="utf-8")

    def get_latest_cursor(self) -> int:
        return max(self._next_cursor() - 1, 0)

    @property
    def dream_prompt_file(self) -> Path:
        return self.workspace / "prompts" / "dream.md"

    def has_dream_prompt_override(self) -> bool:
        try:
            return bool(self.dream_prompt_file.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            return False

    @staticmethod
    def default_dream_prompt() -> str:
        return render_template("agent/dream.md").rstrip()

    def _dream_template(self) -> str:
        try:
            text = self.dream_prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self.default_dream_prompt()
        if not text.strip():
            return self.default_dream_prompt()
        return _truncate_text(text, _WORKSPACE_PROMPT_MAX_CHARS)

    def build_dream_prompt(self, *, max_entries: int = 20) -> tuple[str, int] | None:
        last_cursor = self.get_last_dream_cursor()
        entries = self.read_unprocessed_history(last_cursor)
        if not entries:
            return None
        batch = entries[:max_entries]
        history_text = "\n".join(
            f"[{entry['timestamp']}] {_truncate_text(entry['content'], 1000)}"
            for entry in batch
        )
        prompt = (
            f"{self._dream_template().strip()}\n\n"
            f"{self._render_current_memory_files()}\n\n"
            f"## Conversation History\n{history_text}"
        )
        return prompt, int(batch[-1]["cursor"])

    def _render_current_memory_files(self) -> str:
        blocks: list[str] = []
        for label, path in (
            (".duola/memory/SOUL.md", self.soul_file),
            (".duola/memory/USER.md", self.user_file),
            (".duola/memory/MEMORY.md", self.memory_file),
        ):
            with suppress(OSError):
                content = path.read_text(encoding="utf-8") if path.exists() else ""
                if len(content) > _DREAM_FILE_EMBED_CAP:
                    content = _truncate_text(content, _DREAM_FILE_EMBED_CAP)
                blocks.append(f"### {label}\n{content}" if content.strip() else f"### {label}\n(empty)")
                continue
            blocks.append(f"### {label}\n(empty)")
        return "## Current Memory Files\n" + "\n\n".join(blocks)

    def dream_content_diff(self) -> str:
        return ""

    def build_dream_tools(self) -> ToolRegistry:
        tools = ToolRegistry()
        skills_dir = self.workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        editable_files = [self.memory_file, self.soul_file, self.user_file]
        tools.register(ReadFileTool(self.workspace, restrict_to_workspace=True))
        tools.register(EditFileTool(self.workspace, allowed_dir=skills_dir, extra_write_allowed_files=editable_files))
        tools.register(ApplyPatchTool(self.workspace, allowed_dir=skills_dir, extra_write_allowed_files=editable_files))
        tools.register(WriteFileTool(self.workspace, allowed_dir=skills_dir, extra_write_allowed_files=editable_files))
        return tools

    @staticmethod
    def dream_run_completed(resp: object | None, *, had_tool_errors: bool = False) -> bool:
        if had_tool_errors or resp is None:
            return False
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason is None:
            metadata = getattr(resp, "metadata", None)
            if isinstance(metadata, dict):
                stop_reason = metadata.get("_stop_reason") or metadata.get("stop_reason")
        return stop_reason == "completed"

    @staticmethod
    def dream_session_key() -> str:
        return f"dream:{datetime.now():%Y%m%d-%H%M%S}"

    @staticmethod
    def build_dream_commit_message(prefix: str, diff_body: str) -> str:
        diff_body = (diff_body or "").strip()
        return f"{prefix}\n\n{diff_body}" if diff_body else prefix

    @staticmethod
    def prune_dream_sessions(_sessions_dir: Path, *, keep: int = 10) -> None:
        return None

    @staticmethod
    def summarize_messages(messages: list[dict[str, Any]], *, limit: int = 1200) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "unknown")
            if role not in {"user", "assistant"} or message.get("_command"):
                continue
            text = MemoryStore._message_text(message)
            if text:
                parts.append(f"{role}: {text}")
        summary = " | ".join(parts)
        return _truncate_text(summary, limit) if len(summary) > limit else summary

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in {"tool_use", "tool_result"}:
                        continue
                    value = block.get("text") or block.get("content")
                    if isinstance(value, str):
                        texts.append(value)
                elif isinstance(block, str):
                    texts.append(block)
            return " ".join(texts).strip()
        return str(content).strip()

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role") or "unknown")
            text = MemoryStore._message_text(message)
            if not text:
                continue
            timestamp = str(message.get("timestamp") or "?")[:16]
            tools = message.get("tools_used")
            suffix = f" [tools: {', '.join(tools)}]" if isinstance(tools, list) and tools else ""
            lines.append(f"[{timestamp}] {role.upper()}{suffix}: {text}")
        return "\n".join(lines)

    def _next_cursor(self) -> int:
        cursor_counter = self._read_cursor_counter()
        last = self._read_last_entry() or {}
        last_cursor = self._valid_cursor(last.get("cursor"))
        if cursor_counter is not None:
            if last_cursor is not None:
                return max(cursor_counter, last_cursor) + 1
            max_history_cursor = max((cursor for _entry, cursor in self._iter_valid_entries()), default=0)
            return max(cursor_counter, max_history_cursor) + 1
        if last_cursor is not None:
            return last_cursor + 1
        return max((cursor for _entry, cursor in self._iter_valid_entries()), default=0) + 1

    def _read_cursor_counter(self) -> int | None:
        with suppress(ValueError, OSError):
            cursor = int(self.cursor_file.read_text(encoding="utf-8").strip())
            if cursor >= 0:
                return cursor
        return None

    def _read_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with suppress(FileNotFoundError):
            for line in self.history_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                with suppress(json.JSONDecodeError):
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        entries.append(parsed)
        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        try:
            with self.history_file.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                if size == 0:
                    return None
                handle.seek(max(0, size - 4096))
                data = handle.read().decode("utf-8")
            lines = [line for line in data.splitlines() if line.strip()]
            if not lines:
                return None
            parsed = json.loads(lines[-1])
            return parsed if isinstance(parsed, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _iter_valid_entries(self) -> Iterator[tuple[dict[str, Any], int]]:
        for entry in self._read_entries():
            cursor = self._valid_cursor(entry.get("cursor"))
            if cursor is None or not self._valid_history_payload(entry):
                continue
            yield entry, cursor

    @staticmethod
    def _valid_history_payload(entry: dict[str, Any]) -> bool:
        if not isinstance(entry.get("timestamp"), str):
            return False
        if not isinstance(entry.get("content"), str):
            return False
        session_key = entry.get("session_key")
        return session_key is None or isinstance(session_key, str)

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        tmp = self.history_file.with_suffix(self.history_file.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.history_file)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _valid_cursor(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    def _maybe_migrate_legacy_history(self) -> None:
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return
        text = self.read_file(self.legacy_history_file).strip()
        if not text:
            return
        entries = [
            {
                "cursor": 1,
                "timestamp": datetime.fromtimestamp(self.legacy_history_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "content": text,
            }
        ]
        self._write_entries(entries)
        self.cursor_file.write_text("1", encoding="utf-8")
        self.dream_cursor_file.write_text("1", encoding="utf-8")
        self.legacy_history_file.replace(self.memory_dir / "HISTORY.md.bak")

    def _maybe_migrate_root_memory_dir(self) -> None:
        """Best-effort migration from the earlier root memory/ layout."""

        if not self.legacy_root_memory_dir.exists():
            return
        mapping = {
            self.legacy_root_memory_dir / "MEMORY.md": self.memory_file,
            self.legacy_root_memory_dir / "history.jsonl": self.history_file,
            self.legacy_root_memory_dir / ".cursor": self.cursor_file,
            self.legacy_root_memory_dir / ".dream_cursor": self.dream_cursor_file,
            self.legacy_root_memory_dir / "SOUL.md": self.soul_file,
            self.legacy_root_memory_dir / "USER.md": self.user_file,
            self.legacy_root_memory_dir / "HISTORY.md": self.legacy_history_file,
        }
        for source, target in mapping.items():
            if not source.exists() or target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with suppress(OSError):
                target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


class Consolidator:
    """Summarize compacted session messages into .duola/memory/history.jsonl."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    def __init__(
        self,
        store: MemoryStore,
        sessions: SessionManager,
        consolidation_ratio: float = 0.5,
    ) -> None:
        self.store = store
        self.sessions = sessions
        self.consolidation_ratio = consolidation_ratio
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def archive(
        self,
        messages: list[dict[str, Any]],
        *,
        runtime: LLMRunTime,
        session_key: str | None = None,
        summary_messages: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Summarize messages and append the summary to history.jsonl.

        On provider failure, this matches nanobot's degraded path: raw archive
        the original messages so Dream still has an audit breadcrumb to process.
        """

        if not messages:
            return None

        messages_to_summarize = self._public_history_messages(
            summary_messages if summary_messages is not None else messages
        )
        formatted = _truncate_text(
            MemoryStore._format_messages(messages_to_summarize),
            _RAW_ARCHIVE_MAX_CHARS,
        )
        try:
            response = await runtime.provider.chat_with_retry(
                model=runtime.model,
                messages=[
                    {"role": "system", "content": _consolidator_prompt()},
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                temperature=runtime.generation.temperature,
                max_tokens=runtime.generation.max_tokens,
            )
        except Exception:
            self.store.raw_archive(messages, session_key=session_key)
            return None

        if response.finish_reason == "error":
            self.store.raw_archive(messages, session_key=session_key)
            return None

        summary = response.content or "[no summary]"
        self.store.append_history(
            summary,
            max_chars=_ARCHIVE_SUMMARY_MAX_CHARS,
            session_key=session_key,
        )
        return summary

    async def compact_idle_session(
        self,
        session_key: str,
        *,
        runtime: LLMRunTime,
    ) -> str | None:
        """Archive all unconsolidated messages in a session."""

        lock = self.get_lock(session_key)
        async with lock:
            session = self.sessions.get_or_create(session_key)
            archive_start = min(max(session.last_consolidated, 0), len(session.messages))
            messages_to_archive = list(session.messages[archive_start:])
            if not self._public_history_messages(messages_to_archive):
                return ""

            archive_end = archive_start + len(messages_to_archive)
            summary = await self.archive(
                messages_to_archive,
                runtime=runtime,
                session_key=session_key,
            )
            if summary and summary != "(nothing)":
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": session.updated_at.isoformat(),
                }

            session.last_consolidated = archive_end
            session.provider_state = None
            self.sessions.save(session)
            return summary if summary is not None else "[RAW archive fallback]"

    async def archive_until_boundary(
        self,
        session: Session,
        end_idx: int,
        *,
        runtime: LLMRunTime,
    ) -> str | None:
        """Archive messages from last_consolidated up to end_idx."""

        lock = self.get_lock(session.key)
        async with lock:
            fresh = self.sessions.get_or_create(session.key)
            start = min(max(fresh.last_consolidated, 0), len(fresh.messages))
            end = min(max(end_idx, start), len(fresh.messages))
            if end < len(fresh.messages):
                from duolaAgent.utils.helpers import is_user_turn
                end = max((i for i in range(start, end + 1)
                           if is_user_turn(fresh.messages[i])), default=start)
            chunk = list(fresh.messages[start:end])
            if not self._public_history_messages(chunk):
                return ""

            summary = await self.archive(
                chunk,
                runtime=runtime,
                session_key=fresh.key,
            )
            if summary and summary != "(nothing)":
                fresh.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": fresh.updated_at.isoformat(),
                }
            fresh.last_consolidated = end
            fresh.provider_state = None
            self.sessions.save(fresh)
            session.last_consolidated = fresh.last_consolidated
            session.metadata = dict(fresh.metadata)
            return summary if summary is not None else "[RAW archive fallback]"

    @staticmethod
    def _public_history_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            message
            for message in messages
            if not message.get("_command")
            and message.get("role") in {"user", "assistant"}
            and MemoryStore._message_text(message)
        ]

    def pick_consolidation_boundary(self, session: Session) -> int:
        from duolaAgent.utils.helpers import estimate_message_tokens, is_user_turn
        start = session.last_consolidated
        total = sum(estimate_message_tokens(m) for m in session.messages[start:])
        target = total * self.consolidation_ratio
        used = 0
        for i in range(start, len(session.messages)):
            if i > start and is_user_turn(session.messages[i]) and used >= target:
                return i
            used += estimate_message_tokens(session.messages[i])
        return start  # Never split the only remaining user turn.

    async def maybe_consolidate_by_tokens(self, session, *, runtime, build_messages, tools=None):
        from duolaAgent.utils.helpers import estimate_prompt_tokens_chain
        reservation = max(runtime.generation.max_tokens,
                          getattr(runtime.provider, "context_output_reservation", 0))
        budget = runtime.context_window_tokens - reservation - 1024
        if budget <= 0:
            return ""
        summary = ""
        for _ in range(self._MAX_CONSOLIDATION_ROUNDS):
            estimate, _source = estimate_prompt_tokens_chain(
                runtime.provider, runtime.model, build_messages(), tools)
            if estimate <= budget:
                break
            end = self.pick_consolidation_boundary(session)
            if end <= session.last_consolidated:
                break
            summary = await self.archive_until_boundary(session, end, runtime=runtime) or ""
        return summary
