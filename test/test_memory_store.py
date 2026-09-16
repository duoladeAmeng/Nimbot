import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from duolaAgent.agent.memory import MemoryStore


class MemoryStoreTest(unittest.TestCase):
    def test_memory_files_live_under_duola_state_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))

            self.assertEqual(store.memory_dir, Path(tmp).resolve() / ".duola" / "memory")
            self.assertEqual(store.memory_file, Path(tmp).resolve() / ".duola" / "memory" / "MEMORY.md")
            self.assertEqual(store.history_file, Path(tmp).resolve() / ".duola" / "memory" / "history.jsonl")

    def test_migrates_earlier_root_memory_layout_without_deleting_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "memory"
            old.mkdir()
            (old / "MEMORY.md").write_text("old memory", encoding="utf-8")
            (old / "history.jsonl").write_text(
                '{"cursor": 1, "timestamp": "2026-01-01 00:00", "content": "old event"}\n',
                encoding="utf-8",
            )

            store = MemoryStore(root)

            self.assertEqual(store.read_memory(), "old memory")
            self.assertTrue(store.history_file.exists())
            self.assertTrue((old / "MEMORY.md").exists())

    def test_archive_summary_writes_jsonl_and_filters_tool_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))

            summary = store.archive_summary(
                [
                    {"role": "user", "content": "你好"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "我来看看。"},
                            {"type": "tool_use", "name": "list_dir"},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": "raw file listing should not be memorized",
                            }
                        ],
                    },
                    {"role": "assistant", "content": "完成了。"},
                ],
                session_key="cli:test",
                reason="dream",
            )

            self.assertIn("user: 你好", summary)
            self.assertIn("assistant: 我来看看。", summary)
            self.assertIn("assistant: 完成了。", summary)
            self.assertNotIn("tool_use", summary)
            self.assertNotIn("tool_result", summary)
            self.assertNotIn("raw file listing", summary)
            self.assertFalse(store.memory_file.exists())

            lines = store.history_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["cursor"], 1)
            self.assertEqual(record["session_key"], "cli:test")
            self.assertIn("[dream]", record["content"])
            self.assertIn(summary, record["content"])

    def test_build_dream_prompt_embeds_current_files_and_does_not_advance_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.write_soul("# Soul\n- Precise")
            store.write_user("# User\n- Chinese replies")
            store.write_memory("# Memory\n- Project active")
            cursor = store.append_history("[durable] remember this", session_key="cli:test")

            built = store.build_dream_prompt()

            self.assertIsNotNone(built)
            assert built is not None
            prompt, last_cursor = built
            self.assertEqual(last_cursor, cursor)
            self.assertEqual(store.get_last_dream_cursor(), 0)
            self.assertIn("## Current Memory Files", prompt)
            self.assertIn("### .duola/memory/SOUL.md", prompt)
            self.assertIn("Precise", prompt)
            self.assertIn("### .duola/memory/USER.md", prompt)
            self.assertIn("Chinese replies", prompt)
            self.assertIn("### .duola/memory/MEMORY.md", prompt)
            self.assertIn("Project active", prompt)
            self.assertIn("## Conversation History", prompt)
            self.assertIn("remember this", prompt)

            store.set_last_dream_cursor(last_cursor)
            self.assertIsNone(store.build_dream_prompt())

    def test_default_dream_prompt_is_loaded_from_markdown_template(self) -> None:
        prompt = MemoryStore.default_dream_prompt()

        self.assertIn("You are a memory consolidation engine", prompt)
        self.assertIn(".duola/memory/MEMORY.md", prompt)

    def test_consolidator_prompt_is_loaded_from_markdown_template(self) -> None:
        from duolaAgent.agent.memory import _consolidator_prompt

        prompt = _consolidator_prompt()

        self.assertIn("Extract key facts from this conversation", prompt)
        self.assertIn("[durable]", prompt)

    def test_prompt_history_filters_session_like_nanobot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.append_history("legacy")
            store.append_history("current", session_key="cli:a")
            store.append_history("other", session_key="cli:b")
            store.append_history("internal", session_key="dream:run")

            self.assertEqual(
                [entry["content"] for entry in store.read_recent_history_for_prompt(0, session_key="cli:a")],
                ["current"],
            )
            self.assertEqual(
                [
                    entry["content"]
                    for entry in store.read_recent_history_for_prompt(
                        0,
                        session_key="cli:a",
                        unified_session=True,
                    )
                ],
                ["legacy", "current", "other"],
            )

    def test_dream_tools_can_edit_allowed_files_and_reject_internal_memory_files(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                store = MemoryStore(Path(tmp))
                store.write_memory("# Memory\n- Project X")
                store.write_soul("# Soul\n- Helpful")
                tools = store.build_dream_tools()

                self.assertEqual(
                    set(tools.tool_names),
                    {"apply_patch", "edit_file", "read_file", "write_file"},
                )

                memory_result = await tools.execute(
                    "apply_patch",
                    {
                        "edits": [
                            {
                                "path": ".duola/memory/MEMORY.md",
                                "action": "replace",
                                "old_text": "Project X",
                                "new_text": "Project Y",
                            }
                        ]
                    },
                )
                soul_result = await tools.execute(
                    "edit_file",
                    {"path": ".duola/memory/SOUL.md", "old_text": "Helpful", "new_text": "Precise"},
                )
                skill_result = await tools.execute(
                    "write_file",
                    {
                        "path": "skills/demo/SKILL.md",
                        "content": "---\nname: demo\n---\n",
                    },
                )
                blocked_result = await tools.execute(
                    "write_file",
                    {"path": ".duola/memory/history.jsonl", "content": "bad"},
                )
                cursor_result = await tools.execute(
                    "write_file",
                    {"path": ".duola/memory/.dream_cursor", "content": "99"},
                )

                self.assertIn("Patch applied", str(memory_result))
                self.assertIn("Successfully edited", str(soul_result))
                self.assertIn("Successfully wrote", str(skill_result))
                self.assertIn("outside allowed directory", str(blocked_result))
                self.assertIn("outside allowed directory", str(cursor_result))
                self.assertIn("Project Y", store.read_memory())
                self.assertIn("Precise", store.read_soul())
                self.assertTrue((store.workspace / "skills" / "demo" / "SKILL.md").exists())
                self.assertFalse(store.history_file.exists())
                self.assertFalse(store.dream_cursor_file.exists())

        asyncio.run(run_case())


if __name__ == "__main__":
    unittest.main()
