"""Subagent tool uses the immutable runtime and scope of the current turn."""
from duolaAgent.agent.tools.base import Tool, tool_parameters
from duolaAgent.agent.tools.context import current_request_context
from duolaAgent.security.workspace_access import current_workspace_scope


@tool_parameters({"type": "object", "properties": {
    "task": {"type": "string", "minLength": 1}, "label": {"type": "string"},
    "wait": {"type": "boolean"}, "temperature": {"type": "number", "minimum": 0, "maximum": 2}},
    "required": ["task"], "additionalProperties": False})
class SpawnTool(Tool):
    name = "spawn"
    description = "Run an isolated subagent; wait=true returns its answer inline, otherwise its result returns to this session."

    def __init__(self, manager):
        self._manager = manager

    async def execute(self, task, label=None, wait=False, temperature=None):
        ctx = current_request_context()
        if not ctx or not ctx.runtime or not ctx.session_key:
            return self.error("Spawn requires an admitted session runtime")
        method = self._manager.run_inline if wait else self._manager.spawn
        return await method(task=task, runtime=ctx.runtime, label=label,
                            session_key=ctx.session_key, origin_channel=ctx.channel,
                            origin_chat_id=ctx.chat_id, workspace_scope=current_workspace_scope(),
                            temperature=temperature)


@tool_parameters({"type": "object", "properties": {"task_id": {"type": "string"}},
                  "required": ["task_id"], "additionalProperties": False})
class SubagentStatusTool(Tool):
    name = "subagent_status"
    description = "Read status and result of a child task belonging to this session."
    read_only = True

    def __init__(self, manager):
        self._manager = manager

    async def execute(self, task_id):
        import json
        ctx = current_request_context()
        status = self._manager.get_status(task_id)
        if not status or not ctx or status["parent_session_key"] != ctx.session_key:
            return self.error("Subagent not found in this session")
        return json.dumps(status, ensure_ascii=False)
