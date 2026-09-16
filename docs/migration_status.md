# duolaAgent Migration Status

本次五组核心 Runtime 的实现映射、验证记录和明确边界见 [core_runtime_migration.md](core_runtime_migration.md)。以下保留原迁移记录中的配置与 Memory/Dream 使用说明。

duolaAgent is a nanobot migration. Core agent behavior should stay capability
compatible with nanobot; simplification is limited to compatibility layers,
product surfaces, and integrations that are intentionally out of scope for the
current workspace.

## Core Call Chain

```text
InboundMessage
  -> AgentLoop._dispatch
  -> AgentLoop._process_message
  -> restore / compact / command / build / run / save / respond
  -> AgentRunner.run
  -> LLMProvider.chat_with_retry or chat_stream_with_retry
  -> ToolRegistry.execute
  -> SessionManager.save
  -> OutboundMessage
```

## Implemented Capabilities

- Persistent sessions: JSONL files are stored in `.duola/sessions` under the configured workspace.
- Token-aware compaction: `AgentLoop._compact_session` estimates prompt size, asks `Consolidator` to archive older messages, and replays only the unconsolidated tail plus a small continuity suffix.
- Consolidator: compacted session chunks are summarized by the configured LLM into `.duola/memory/history.jsonl`; provider failures fall back to `[RAW]` archive entries and still advance `last_consolidated` to avoid duplicate archive spam.
- Archived memory history: Consolidator outputs are stored in `.duola/memory/history.jsonl`.
- Long-term memory: `.duola/memory/MEMORY.md` is injected into future system prompts.
- Dream command: `/dream` reads unprocessed `.duola/memory/history.jsonl` entries after `.duola/memory/.dream_cursor`, runs a Dream agent with restricted file-editing tools, and advances the cursor only after a clean completed run.
- Skills loading: `skills/**/SKILL.md` and `.duola/skills/**/SKILL.md` are loaded into the system prompt.
- Command handling: `/help`, `/clear`, `/history`, `/sessions`, `/compact`, `/memory`, `/dream`, `/tools`, `/config`.
- Streaming output: providers can stream deltas through `on_content_delta`; the CLI consumes stream delta messages before the final response.
- Provider retry and timeout: model calls have timeout protection and lightweight retry for transient errors.
- Checkpoint/crash recovery: persist awaiting_tools / tools_completed / final_response, restore recorded assistant/tool messages, and backfill missing interrupted results.
- Session lock and pending queue: one active turn per session; messages arriving mid-turn are queued and can be injected into the runner.
- Concurrent tool execution: enabled by `concurrent_tools`, partitioned into batches by each tool's `concurrency_safe` property.
- Runtime snapshots and fallback: immutable per-turn model/generation/window selection, presets/overrides, admission refresh, classified retries, primary circuit breaker and stream timeout recovery.
- Subagents and goals: isolated child runners; durable explicit goals with bounded continuation slices through the pending queue.
- Workspace policy: file tools and shell working directories can be restricted to the configured workspace.
- Configurable workspace: via CLI `--workspace`, `.env`, or `duola.json`.

## Configuration

Configuration follows the nanobot-style split:

- `duolaAgent.config.schema` defines the typed `Config`.
- `duolaAgent.config.loader` loads `duola.json`, resolves `${ENV}` references, and applies CLI/env overrides.
- `duolaAgent.config.paths` resolves the active workspace.

Configuration priority:

1. Explicit CLI arguments
2. Environment variables / `.env`
3. `duola.json`
4. Built-in defaults

Supported environment variables:

```text
DUOLA_MODEL
DUOLA_API_KEY
DUOLA_API_BASE
DUOLA_PROVIDER
DUOLA_WORKSPACE
DUOLA_MAX_TOOL_ITERATIONS
DUOLA_MAX_CONTEXT_TOKENS
DUOLA_COMPACT_TARGET_TOKENS
DUOLA_MAX_TOOL_RESULT_CHARS
DUOLA_RESTRICT_TO_WORKSPACE
DUOLA_CONCURRENT_TOOLS
DUOLA_STREAM
DUOLA_RETRY_ATTEMPTS
DUOLA_LLM_TIMEOUT_S
```

Legacy Anthropic variable names are still accepted:

```text
ANTHROPIC_MODEL_ID
ANTHROPIC_API_KEY
ANTHROPIC_BASE_URL
```

Example `duola.json`:

```json
{
  "agents": {
    "defaults": {
      "provider": "openai-compatible",
      "model": "mimo-v2.5",
      "workspace": "E:/CodeDir/duolaAgent",
      "max_tool_iterations": 3,
      "max_context_tokens": 12000,
      "compact_target_tokens": 8000,
      "restrict_to_workspace": true,
      "concurrent_tools": false,
      "stream": true,
      "retry_attempts": 2
    }
  },
  "providers": {
    "openai_compatible": {
      "api_key": "${DUOLA_API_KEY}",
      "api_base": "https://api.xiaomimimo.com/anthropic"
    }
  }
}
```

MiMo URLs containing `xiaomimimo.com` are automatically routed through the
OpenAI-compatible provider and normalized to `/v1/chat/completions`.

## Parity Boundaries

- Session persistence, memory files, Dream cursor handling, restricted Dream
  tools, context compaction, streaming, retries, checkpoint recovery, session
  locks, pending injection, configurable workspace, and configurable runtime
  settings are implemented as core migration features, not optional mini
  substitutes.
- Remaining gaps should be treated as migration backlog, not intentional
  feature reductions. Current known gaps include git-backed Dream audit/restore
  commands, nanobot's full command router surface, tokenizer-specific context
  accounting and provider-state sidecars. An optional bwrap backend is available;
  the default application guards do not provide OS-level sandbox isolation.
- Compatibility/product simplifications may remove WebUI-specific metadata,
  marketplace plumbing, and unused channel adapters, but must not change the
  core agent capability contract.

The goal is functional parity for the migrated core: context construction,
session/history, LLM calls, tool calling, tool results, compaction, memory,
configuration, safety boundaries, and recovery.
