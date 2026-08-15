from dataclasses import Field

from pydantic_settings import BaseSettings

from duolaAgent.config.schema.provider_config import ProvidersConfig


class Config(BaseSettings):
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
