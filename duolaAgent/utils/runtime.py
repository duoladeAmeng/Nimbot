"""Runner output normalization shared with context governance."""
def ensure_nonempty_tool_result(tool_name, result):
    return result if result is not None and str(result).strip() else f"[{tool_name} completed with no output]"
