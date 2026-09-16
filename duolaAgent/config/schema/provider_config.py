from pydantic import Field

from duolaAgent.config.schema.config_base import Base


class ProviderConfig(Base):
    api_key: str | None = Field(default=None, repr=False)
    api_base: str | None = None


class ProvidersConfig(Base):
    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai_compatible: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
