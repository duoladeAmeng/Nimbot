"""Nanobot-style import path; retain the migration's existing agent.mcp API."""
from duolaAgent.agent.mcp import (
    MCPConnection,
    MCPPromptWrapper,
    MCPProvider,
    MCPResourceWrapper,
    MCPServerConfig,
    MCPToolWrapper,
    connect_mcp_servers,
)

__all__ = ["MCPConnection", "MCPPromptWrapper", "MCPProvider", "MCPResourceWrapper",
           "MCPServerConfig", "MCPToolWrapper", "connect_mcp_servers"]
