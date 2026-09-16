from pathlib import Path

from pydantic import Field

from duolaAgent.config.schema.config_base import Base
from duolaAgent.config.schema.provider_config import ProvidersConfig
from duolaAgent.config.schema.tool_config import ExecToolConfig


class ModelPresetConfig(Base):
    provider: str = "anthropic"
    model: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    context_window_tokens: int | None = Field(default=None, gt=0)
    fallbacks: list[str] = Field(default_factory=list)


class AgentDefaultsConfig(ModelPresetConfig):
    provider: str = "anthropic"
    model: str | None = None
    workspace: str | None = "."
    max_tool_iterations: int = Field(default=3, gt=0)
    max_context_tokens: int = Field(default=12000, gt=1024)
    compact_target_tokens: int = Field(default=8000, gt=0)
    max_tool_result_chars: int = Field(default=20000, ge=1000)
    restrict_to_workspace: bool = True
    concurrent_tools: bool = False
    stream: bool = True
    retry_attempts: int = Field(default=2, gt=0)
    model_preset: str | None = None
    max_concurrent_sessions: int = Field(default=4, gt=0)
    max_pending_messages: int = Field(default=64, gt=0)
    max_subagents: int = Field(default=4, gt=0)
    max_continuation_turns: int = Field(default=12, ge=0)


class ToolsConfig(Base):
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = True
    webui_allow_local_service_access: bool = False
    mcp_servers: dict[str, dict] = Field(default_factory=dict)
    plugins: bool = True
    ssrf_whitelist: list[str] = Field(default_factory=list)


class AgentsConfig(Base):
    defaults: AgentDefaultsConfig = Field(default_factory=AgentDefaultsConfig)


class Config(Base):
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    model_presets: dict[str, ModelPresetConfig] = Field(default_factory=dict)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        return Path(self.agents.defaults.workspace or ".").expanduser().resolve()
