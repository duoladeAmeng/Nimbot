"""Load and render agent system prompt templates (Jinja2) under duolaAgent/prompts/.

Agent prompts live in ``prompts/agent/`` (pass names like
``agent/identity.md``). This mirrors nanobot's template helper while keeping the
mini project layout intentionally small.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def _environment() -> Environment:
    # Plain-text prompts: do not HTML-escape variable values.
    return Environment(
        loader=FileSystemLoader(str(_PROMPTS_ROOT)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, *, strip: bool = False, **kwargs: Any) -> str:
    """Render ``name`` under ``prompts/``.

    Use ``strip=True`` for single-line strings when trailing newlines are not
    desired.
    """

    text = _environment().get_template(name).render(**kwargs)
    return text.rstrip() if strip else text

