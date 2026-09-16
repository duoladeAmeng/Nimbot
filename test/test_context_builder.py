import tempfile
import unittest
from pathlib import Path

from duolaAgent.agent.context import ContextBuilder
from duolaAgent.utils.prompt_templates import render_template


class ContextBuilderTest(unittest.TestCase):
    def test_system_prompt_is_rendered_from_markdown_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            builder = ContextBuilder(workspace)

            prompt = builder.build_system_prompt()

            self.assertIn("You are Duola", prompt)
            self.assertIn(f"Workspace: {workspace.resolve()}", prompt)
            self.assertIn("Runtime:", prompt)
            self.assertIn("Tool contract:", prompt)

    def test_render_template_supports_jinja_conditionals(self) -> None:
        rendered = render_template("agent/platform_policy.md", system="Windows")

        self.assertIn("Platform Policy (Windows)", rendered)
        self.assertNotIn("Platform Policy (POSIX)", rendered)


if __name__ == "__main__":
    unittest.main()
