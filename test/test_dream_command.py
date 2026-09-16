import tempfile
import unittest
from pathlib import Path
from typing import Any

from duolaAgent.agent.loop import AgentLoop
from duolaAgent.bus.message import InboundMessage
from duolaAgent.bus.queue import MessageBus
from duolaAgent.llm.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _DreamProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key=None, api_base=None)
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if tools is None:
            return LLMResponse(
                content="- [durable] 项目必须和 nanobot 核心能力一致。",
                finish_reason="stop",
            )
        if tools is not None:
            if any(message.get("role") == "tool" or "tool_result" in str(message.get("content"))
                   for message in messages):
                return LLMResponse(content="Memory updated.", finish_reason="stop")
            assert "## Conversation History" in messages[0]["content"]
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call-1",
                        name="write_file",
                        arguments={
                            "path": ".duola/memory/MEMORY.md",
                            "content": "# Memory\n\n- Migrated Dream wrote durable memory.\n",
                        },
                    )
                ],
            )
        return LLMResponse(content="Memory updated.", finish_reason="stop")


class _FailingConsolidatorProvider(LLMProvider):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="provider failed", finish_reason="error")


class DreamCommandTest(unittest.IsolatedAsyncioTestCase):
    async def test_compact_reports_noop_when_only_commands_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(
                MessageBus(),
                _DreamProvider(),
                "test-model",
                workspace=Path(tmp),
                stream=False,
            )

            outbound = await loop._process_message(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="test",
                    content="/compact",
                )
            )

            self.assertIsNotNone(outbound)
            assert outbound is not None
            self.assertIn("Nothing to compact", outbound.content)
            self.assertFalse(loop.context.memory.history_file.exists())

    async def test_compact_archives_regular_history_for_dream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(
                MessageBus(),
                _DreamProvider(),
                "test-model",
                workspace=Path(tmp),
                stream=False,
            )
            session = loop.sessions.get_or_create("cli:test")
            session.add_message("user", "记住项目必须和 nanobot 核心能力一致。")
            session.add_message("assistant", "收到。")
            loop.sessions.save(session)

            outbound = await loop._process_message(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="test",
                    content="/compact",
                )
            )

            self.assertIsNotNone(outbound)
            assert outbound is not None
            self.assertIn("Session compacted", outbound.content)
            self.assertEqual(loop.sessions.get_or_create("cli:test").last_consolidated, 2)
            built = loop.context.memory.build_dream_prompt()
            self.assertIsNotNone(built)
            assert built is not None
            prompt, _cursor = built
            self.assertIn("nanobot 核心能力一致", prompt)

    async def test_compact_raw_archive_fallback_still_advances_last_consolidated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = AgentLoop(
                MessageBus(),
                _FailingConsolidatorProvider(None, None),
                "test-model",
                workspace=Path(tmp),
                stream=False,
            )
            session = loop.sessions.get_or_create("cli:test")
            session.add_message("user", "需要 raw fallback 也推进归档游标。")
            session.add_message("assistant", "收到。")
            loop.sessions.save(session)

            outbound = await loop._process_message(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="test",
                    content="/compact",
                )
            )

            self.assertIsNotNone(outbound)
            assert outbound is not None
            self.assertIn("Session compacted", outbound.content)
            reloaded = loop.sessions.get_or_create("cli:test")
            self.assertEqual(reloaded.last_consolidated, 2)
            history_text = loop.context.memory.history_file.read_text(encoding="utf-8")
            self.assertIn("[RAW]", history_text)
            self.assertIn("raw fallback", history_text)

    async def test_dream_runs_agent_tools_and_advances_cursor_on_clean_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            provider = _DreamProvider()
            loop = AgentLoop(
                MessageBus(),
                provider,
                "test-model",
                workspace=workspace,
                stream=False,
            )
            cursor = loop.context.memory.append_history(
                "[durable] Project uses nanobot-style Dream.",
                session_key="cli:test",
            )

            outbound = await loop._process_message(
                InboundMessage(
                    channel="cli",
                    sender_id="user",
                    chat_id="test",
                    content="/dream",
                )
            )

            self.assertIsNotNone(outbound)
            assert outbound is not None
            self.assertIn("Dream completed", outbound.content)
            self.assertEqual(loop.context.memory.get_last_dream_cursor(), cursor)
            self.assertIn(
                "Migrated Dream wrote durable memory",
                loop.context.memory.read_memory(),
            )
            self.assertEqual(provider.calls, 2)


if __name__ == "__main__":
    unittest.main()
