"""Durable explicit goals; ordinary chat does not implicitly enable continuation."""
import json
from copy import deepcopy
from datetime import datetime

from duolaAgent.agent.tools.base import Tool, tool_parameters
from duolaAgent.agent.tools.context import current_request_context
from duolaAgent.session.goal_state import (
    GOAL_STATE_KEY,
    goal_state_raw,
    parse_goal_state,
    sustained_goal_active,
)


class _GoalTool(Tool):
    def __init__(self, sessions):
        self._sessions = sessions

    def _session(self):
        ctx = current_request_context()
        if not ctx or not ctx.session_key:
            raise ValueError("Goal tools require an active session")
        return self._sessions.get_or_create(ctx.session_key)

    def _mutation_allowed(self):
        ctx = current_request_context()
        return bool(ctx and ctx.metadata.get("goal_requested"))

    def _save_goal_state(self, session, goal, *, reset_continuation=False):
        previous = deepcopy(session.metadata)
        session.metadata[GOAL_STATE_KEY] = goal
        session.metadata.pop("thread_goal", None)
        if reset_continuation:
            session.metadata.pop("goal_continuation_rounds", None)
        try:
            self._sessions.save(session)
        except BaseException:
            session.metadata.clear()
            session.metadata.update(previous)
            raise


@tool_parameters({"type": "object", "properties": {
    "objective": {"type": "string", "minLength": 1, "maxLength": 4000},
    "ui_summary": {"type": ["string", "null"], "maxLength": 120}},
    "required": ["objective"], "additionalProperties": False})
class CreateGoalTool(_GoalTool):
    name = "create_goal"
    description = "Record the explicitly requested /goal objective, with verifiable completion criteria."

    async def execute(self, objective, ui_summary=None):
        session = self._session()
        if not self._mutation_allowed():
            return self.error("Goal creation requires an explicit /goal request.")
        if sustained_goal_active(session.metadata):
            return self.error("A goal is active. Replace it only when explicitly requested.")
        if not objective.strip():
            return self.error("Objective must not be blank.")
        self._save_goal_state(session, {"status": "active", "objective": objective.strip(),
                              "ui_summary": ui_summary or "",
                              "started_at": datetime.now().isoformat()}, reset_continuation=True)
        return "Goal recorded. Continue working; call update_goal action='complete' after verification."


@tool_parameters({"type": "object", "properties": {
    "action": {"type": "string", "enum": ["complete", "cancel", "block", "replace"]},
    "recap": {"type": ["string", "null"], "maxLength": 8000},
    "objective": {"type": ["string", "null"], "maxLength": 4000},
    "ui_summary": {"type": ["string", "null"], "maxLength": 120}},
    "required": ["action"], "additionalProperties": False})
class UpdateGoalTool(_GoalTool):
    name = "update_goal"
    description = "Complete only after verified success, block when unable to progress, cancel on user request, or explicitly replace the goal."

    async def execute(self, action, recap=None, objective=None, ui_summary=None):
        session = self._session()
        prior = parse_goal_state(goal_state_raw(session.metadata))
        if not prior or prior.get("status") != "active":
            return self.error("No active goal.")
        if action == "replace":
            if not self._mutation_allowed() or not (objective or "").strip():
                return self.error("Replacement requires an explicit /goal request and objective.")
            goal = {"status": "active", "objective": objective.strip(), "ui_summary": ui_summary or "",
                    "started_at": datetime.now().isoformat(), "previous_objective": prior["objective"]}
        else:
            goal = {**prior, "status": {"complete": "completed", "cancel": "cancelled", "block": "blocked"}[action],
                    "recap": recap or "", "ended_at": datetime.now().isoformat()}
        self._save_goal_state(session, goal, reset_continuation=True)
        return json.dumps(goal, ensure_ascii=False)


@tool_parameters({"type": "object", "properties": {}, "additionalProperties": False})
class GetGoalTool(_GoalTool):
    name = "get_goal"
    description = "Read the current persisted sustained goal."
    read_only = True

    async def execute(self):
        return json.dumps(goal_state_raw(self._session().metadata), ensure_ascii=False)
