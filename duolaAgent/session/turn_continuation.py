"""Bounded internal continuation at a persisted runner-slice boundary."""
from dataclasses import replace

from duolaAgent.session.goal_state import goal_state_runtime_lines, sustained_goal_active

INTERNAL_CONTINUATION_META = "internal_continuation"
_GOAL_CONTINUATION_ROUNDS_KEY = "goal_continuation_rounds"


def _goal_continuation_prompt(metadata):
    return ("Continue the active goal from the saved context. Use tools as needed; "
            "call update_goal action='complete' only after verified completion.\n" +
            "\n".join(goal_state_runtime_lines(metadata)))


def continuation_message(ctx, *, max_rounds=12):
    if ctx.pending_queue is None or ctx.stop_reason not in {"max_iterations", "max_tool_iterations"}:
        return None
    metadata = ctx.require_session().metadata
    if not sustained_goal_active(metadata):
        return None
    rounds = int(metadata.get(_GOAL_CONTINUATION_ROUNDS_KEY, 0))
    if rounds >= max_rounds:
        return None
    if ctx.pending_queue.full():
        return None
    metadata[_GOAL_CONTINUATION_ROUNDS_KEY] = rounds + 1
    # Overrides remain part of this visible run; goal-creation permission does not.
    inbound_metadata = {k: v for k, v in ctx.msg.metadata.items()
                        if k in {"model", "model_override", "model_preset", "preset_override"}}
    inbound_metadata[INTERNAL_CONTINUATION_META] = True
    return replace(ctx.msg, sender_id="goal_continuation", content=_goal_continuation_prompt(metadata),
                   media=[], metadata=inbound_metadata, session_key_override=ctx.session_key)
