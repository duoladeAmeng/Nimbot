"""HTTP fetch with DNS validation on every request, including redirects."""
import httpx

from duolaAgent.agent.tools.base import Tool, tool_parameters
from duolaAgent.security.network import PinnedDNSAsyncTransport


@tool_parameters({"type": "object", "properties": {
    "url": {"type": "string"}, "max_chars": {"type": "integer", "minimum": 1, "maximum": 200000}},
    "required": ["url"], "additionalProperties": False})
class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a public HTTP/HTTPS page. Private network targets are blocked."
    read_only = True

    async def execute(self, url, max_chars=20000):
        async with httpx.AsyncClient(transport=PinnedDNSAsyncTransport(), trust_env=False,
                                     follow_redirects=True, max_redirects=5, timeout=30) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                chunks, size = [], 0
                async for chunk in response.aiter_text():
                    chunks.append(chunk[:max_chars - size])
                    size += len(chunks[-1])
                    if size >= max_chars:
                        break
                return "".join(chunks)
