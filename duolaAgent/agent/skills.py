from __future__ import annotations

from pathlib import Path


class SkillsLoader:
    """Load project skills from SKILL.md files.

    This preserves the nanobot idea that capabilities can be documented outside
    Python code and injected into the prompt. duolaAgent loads a compact summary
    instead of implementing the full marketplace/plugin system.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()

    def build_skills_section(self, *, max_chars: int = 12000) -> str:
        skill_files = self._skill_files()
        if not skill_files:
            return ""

        parts: list[str] = []
        used = 0
        for path in skill_files:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
            rel = path.relative_to(self.workspace)
            rendered = f"## {rel}\n\n{content}"
            used += len(rendered)
            if used > max_chars:
                parts.append("(skills truncated)")
                break
            parts.append(rendered)
        return "\n\n".join(parts)

    def _skill_files(self) -> list[Path]:
        roots = [
            self.workspace / "skills",
            self.workspace / ".duola" / "skills",
        ]
        files: list[Path] = []
        for root in roots:
            if root.is_dir():
                files.extend(sorted(root.rglob("SKILL.md")))
        return files
