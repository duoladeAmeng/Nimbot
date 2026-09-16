"""Python entry-point tools; each registry constructs its own tool instances."""
from importlib.metadata import entry_points

from loguru import logger

from duolaAgent.agent.tools.base import Tool
from duolaAgent.agent.tools.context import ToolContext


def load_tool_plugins(registry, *, workspace, config=None, scope="core"):
    # Match nanobot's entry-point group, plus the migrated package's group.
    for group in ("nanobot.tools", "duolaAgent.tools"):
        for entry in entry_points(group=group):
            try:
                cls = entry.load()
                if not isinstance(cls, type) or not issubclass(cls, Tool):
                    raise TypeError("Tool entry point must export a Tool subclass")
                if scope not in cls._scopes:
                    continue
                ctx = ToolContext(config=config, workspace=str(workspace))
                if cls.enabled(ctx):
                    tool = cls.create(ctx)
                    if registry.has(tool.name):
                        raise ValueError(f"Duplicate tool name: {tool.name}")
                    registry.register(tool)
            except Exception:
                logger.exception("Could not load tool plugin {}", entry.name)
