from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from duolaAgent.config.paths import get_workspace_path
from duolaAgent.config.schema import Config

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_config(config_path: str | Path | None = None) -> Config:
    """Load duolaAgent config in the same role as nanobot.config.loader."""

    load_dotenv(override=False)
    path = _config_path(config_path)
    if path is None or not path.is_file():
        return Config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid config at {path}: {exc}") from exc


def load_runtime_config(
    *,
    config_path: str | Path | None = None,
    workspace: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> Config:
    """Load config, resolve env refs, and apply CLI/env overrides."""

    config = resolve_config_env_vars(load_config(config_path))
    defaults = config.agents.defaults

    defaults.workspace = str(
        get_workspace_path(
            workspace
            or os.environ.get("DUOLA_WORKSPACE")
            or defaults.workspace
        )
    )

    override_model = (
        model
        or os.environ.get("DUOLA_MODEL")
        or os.environ.get("ANTHROPIC_MODEL_ID")
    )
    defaults.model = override_model or defaults.model
    if not defaults.model and not defaults.model_preset:
        raise ValueError(
            "Missing model. Set DUOLA_MODEL/ANTHROPIC_MODEL_ID, pass --model, "
            "or add agents.defaults.model to duola.json."
        )

    env_provider = os.environ.get("DUOLA_PROVIDER")
    if env_provider:
        defaults.provider = env_provider

    selected = config.model_presets.get(defaults.model_preset) if defaults.model_preset else None
    if defaults.model_preset and selected is None:
        raise ValueError(f"Unknown model preset: {defaults.model_preset}")
    if selected is not None:
        if override_model:
            selected.model = override_model
        if env_provider:
            selected.provider = env_provider
    provider = provider_config(config, selected.provider if selected else defaults.provider)
    provider.api_key = (
        api_key
        or os.environ.get("DUOLA_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or provider.api_key
    )
    provider.api_base = (
        api_base
        or os.environ.get("DUOLA_API_BASE")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or provider.api_base
    )

    defaults.max_tool_iterations = _int_env("DUOLA_MAX_TOOL_ITERATIONS", defaults.max_tool_iterations)
    defaults.max_context_tokens = _int_env("DUOLA_MAX_CONTEXT_TOKENS", defaults.max_context_tokens)
    defaults.compact_target_tokens = _int_env("DUOLA_COMPACT_TARGET_TOKENS", defaults.compact_target_tokens)
    defaults.max_tool_result_chars = _int_env("DUOLA_MAX_TOOL_RESULT_CHARS", defaults.max_tool_result_chars)
    defaults.restrict_to_workspace = _bool_env("DUOLA_RESTRICT_TO_WORKSPACE", defaults.restrict_to_workspace)
    defaults.concurrent_tools = _bool_env("DUOLA_CONCURRENT_TOOLS", defaults.concurrent_tools)
    defaults.stream = _bool_env("DUOLA_STREAM", defaults.stream)
    defaults.retry_attempts = _int_env("DUOLA_RETRY_ATTEMPTS", defaults.retry_attempts)

    return Config.model_validate(config.model_dump())


def resolve_config_env_vars(config: Config) -> Config:
    data = _resolve_env_value(config.model_dump(mode="json"))
    return Config.model_validate(data)


def provider_config(config: Config, provider_name: str):
    normalized = provider_name.replace("-", "_")
    providers = config.providers
    current = getattr(providers, normalized, None)
    if current is not None:
        return current
    if "mimo" in normalized or "openai" in normalized:
        return providers.openai_compatible
    raise ValueError(f"Unknown provider: {provider_name}")


def _config_path(config_path: str | Path | None) -> Path | None:
    raw = config_path or os.environ.get("DUOLA_CONFIG") or "duola.json"
    path = Path(raw).expanduser()
    return path


def _resolve_env_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_value(item) for item in value]
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda match: os.environ.get(match.group(1), ""), value)
    return value


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default
