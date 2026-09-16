from __future__ import annotations

import os
from pathlib import Path


def get_media_dir() -> Path:
    path = Path(os.environ.get("DUOLA_WORKSPACE", ".")).expanduser().resolve() / ".duola" / "media"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_workspace_path(workspace: str | Path | None = None) -> Path:
    """Resolve and ensure the active duolaAgent workspace path."""

    path = Path(workspace).expanduser() if workspace else Path(".")
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def is_default_workspace(workspace: str | Path | None) -> bool:
    current = Path(workspace).expanduser() if workspace is not None else Path(".")
    return current.resolve() == Path(".").resolve()
