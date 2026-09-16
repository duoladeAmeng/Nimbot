"""Build immutable provider snapshots from validated configuration."""
from dataclasses import dataclass

from duolaAgent.llm.llm_runtime import LLMRuntime
from duolaAgent.llm.providers.base import GenerationSettings
from duolaAgent.llm.providers.fallback_provider import FallbackProvider, ProviderCandidate


@dataclass(frozen=True, slots=True)
class ProviderSnapshot:
    primary: ProviderCandidate
    fallbacks: tuple[ProviderCandidate, ...]
    runtime: LLMRuntime


def create_provider(config, provider_name):
    from duolaAgent.config.loader import provider_config
    from duolaAgent.llm.providers.anthropic_provider import AnthropicProvider
    from duolaAgent.llm.providers.openai_compat_provider import OpenAICompatibleProvider
    settings = provider_config(config, provider_name)
    name = provider_name.replace("-", "_")
    if settings.api_base and "xiaomimimo.com" in settings.api_base.lower():
        base = settings.api_base.rstrip("/")
        if base.endswith("/anthropic"):
            base = base[:-len("/anthropic")]
        if not base.endswith("/v1"):
            base += "/v1"
        provider = OpenAICompatibleProvider(settings.api_key, base, api_key_header="api-key")
    elif name == "anthropic":
        provider = AnthropicProvider(settings.api_key, settings.api_base)
    elif name in {"openai", "openai_compatible", "deepseek", "mimo"}:
        base = settings.api_base or ("https://api.deepseek.com/v1" if name == "deepseek" else None)
        provider = OpenAICompatibleProvider(settings.api_key, base)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
    provider.retry_attempts = config.agents.defaults.retry_attempts
    return provider


def build_provider_snapshot(config, *, preset=None, model=None):
    defaults = config.agents.defaults
    selected = preset or defaults.model_preset
    if selected and selected not in config.model_presets:
        raise ValueError(f"Unknown model preset: {selected}")
    primary_settings = config.model_presets[selected] if selected else defaults
    for fallback in primary_settings.fallbacks:
        if fallback not in config.model_presets:
            raise ValueError(f"Unknown fallback preset: {fallback}")
    def candidate(settings, override=None):
        chosen = override or settings.model
        if not chosen:
            raise ValueError("A model must be configured")
        return ProviderCandidate(create_provider(config, settings.provider), chosen,
                                 GenerationSettings(settings.temperature, settings.max_tokens),
                                 settings.context_window_tokens or defaults.max_context_tokens)
    primary = candidate(primary_settings, model)
    names = primary_settings.fallbacks
    fallbacks = tuple(candidate(config.model_presets[name]) for name in names)
    provider = FallbackProvider(primary, fallbacks) if fallbacks else primary.provider
    budget = min(c.context_window_tokens for c in (primary, *fallbacks))
    # Validate every configured output reservation against the common window.
    if budget <= max(c.generation.max_tokens for c in (primary, *fallbacks)) + 1024:
        raise ValueError("Context window must exceed max output tokens plus 1024 safety tokens")
    runtime = LLMRuntime(provider, primary.model, primary.generation, budget)
    return ProviderSnapshot(primary, fallbacks, runtime)
