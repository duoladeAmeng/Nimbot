from __future__ import annotations

import base64
import json
import os
import weakref
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from duolaAgent.utils.helpers import is_user_turn, normalize_messages

FILE_MAX_MESSAGES = 2000
SESSION_CACHE_MAX_SIZE = 128
MIN_COMPACTED_REPLAY_MESSAGES = 6


@dataclass
class Session:
    """A conversation session."""

    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0
    provider_state: Any | None = field(default=None, repr=False)

    def add_message(self, role: str, content: Any, **kwargs: Any) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = FILE_MAX_MESSAGES) -> list[dict[str, Any]]:
        """Return recent replayable messages for the next LLM call."""

        replay_start = min(max(self.last_consolidated, 0), len(self.messages))
        if replay_start:
            replay_start = min(
                replay_start,
                max(0, len(self.messages) - MIN_COMPACTED_REPLAY_MESSAGES),
            )
        # A raw replay suffix includes complete user turns, never half a tool exchange.
        starts = [i for i, message in enumerate(self.messages) if is_user_turn(message)]
        preceding = [i for i in starts if i <= replay_start]
        if preceding:
            replay_start = preceding[-1]
        replayable = self.messages[replay_start:]
        raw_messages = replayable
        if max_messages > 0 and len(replayable) > max_messages:
            desired = len(replayable) - max_messages
            start = next((i for i in range(desired, len(replayable))
                          if is_user_turn(replayable[i])), 0)
            raw_messages = replayable[start:]
        history: list[dict[str, Any]] = []
        for message in raw_messages:
            if message.get("_command"):
                continue
            entry = {
                key: deepcopy(value)
                for key, value in message.items()
                if key in {"role", "content", "tool_calls", "tool_call_id", "name", "is_error"}
            }
            if entry.get("role") == "assistant" and not entry.get("content"):
                if not entry.get("tool_calls"):
                    continue
            history.append(entry)
        return self._legal_history(normalize_messages(history))

    def clear(self) -> None:
        self.messages = []
        self.last_consolidated = 0
        self.provider_state = None
        self.metadata.clear()
        self.updated_at = datetime.now()

    @staticmethod
    def _legal_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Avoid replaying a history suffix that starts with orphan tool results."""

        start = 0
        while start < len(messages) and messages[start].get("role") == "tool":
            start += 1
        return messages[start:]


class SessionManager:
    """Manage session identity, caching, and JSONL persistence."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        self.workspace = Path(workspace or ".").expanduser().resolve()
        self.sessions_dir = self.workspace / ".duola" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, Session] = OrderedDict()
        self._max_cached_sessions = SESSION_CACHE_MAX_SIZE
        self._live: weakref.WeakValueDictionary[str, Session] = weakref.WeakValueDictionary()

    def get_or_create(self, key: str) -> Session:
        session = self._live.get(key) or self._cache.get(key)
        if session is None:
            session = self._load(key) or Session(key=key)
        self._remember(session)
        return session

    def get_cached(self, key: str) -> Session | None:
        session = self._cache.get(key)
        if session is not None:
            self._remember(session)
        return session

    def save(self, session: Session) -> None:
        session.updated_at = datetime.now()
        self._write(session)
        self._remember(session)

    def delete_session(self, key: str) -> bool:
        self._cache.pop(key, None)
        self._live.pop(key, None)
        path = self._path_for_key(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for session in self._cache.values():
            items.append(self._session_info(session))
            seen.add(session.key)
        for path in self.sessions_dir.glob("*.jsonl"):
            key = self._decode_key(path.stem)
            if key is None or key in seen:
                continue
            loaded = self._load(key)
            if loaded is not None:
                items.append(self._session_info(loaded))
        return sorted(items, key=lambda item: item["updated_at"], reverse=True)

    def _load(self, key: str) -> Session | None:
        path = self._path_for_key(key)
        if not path.is_file():
            return None

        metadata: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        created_at: datetime | None = None
        updated_at: datetime | None = None
        last_consolidated = 0

        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    continue
                if record.get("_type") == "metadata":
                    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                    created_at = self._parse_dt(record.get("created_at"))
                    updated_at = self._parse_dt(record.get("updated_at"))
                    raw_lc = record.get("last_consolidated")
                    last_consolidated = raw_lc if isinstance(raw_lc, int) else 0
                else:
                    messages.append(record)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load session {key}: persisted history is unreadable") from exc

        return Session(
            key=key,
            messages=messages,
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=metadata,
            last_consolidated=min(max(last_consolidated, 0), len(messages)),
        )

    def _write(self, session: Session) -> None:
        path = self._path_for_key(session.key)
        tmp = path.with_suffix(".jsonl.tmp")
        records = [
            {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated,
            },
            *session.messages,
        ]
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)

    def _remember(self, session: Session) -> None:
        self._live[session.key] = session
        self._cache[session.key] = session
        self._cache.move_to_end(session.key)
        while len(self._cache) > self._max_cached_sessions:
            self._cache.popitem(last=False)

    def _path_for_key(self, key: str) -> Path:
        encoded = base64.urlsafe_b64encode(key.encode("utf-8")).decode("ascii").rstrip("=")
        return self.sessions_dir / f"{encoded}.jsonl"

    @staticmethod
    def _decode_key(stem: str) -> str | None:
        try:
            padding = "=" * (-len(stem) % 4)
            return base64.urlsafe_b64decode((stem + padding).encode("ascii")).decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _parse_dt(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _session_info(session: Session) -> dict[str, Any]:
        preview = ""
        for message in reversed(session.messages):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                preview = content.strip().replace("\n", " ")[:120]
                break
        return {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "preview": preview,
        }
