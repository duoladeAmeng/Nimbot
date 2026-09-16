# 核心 Runtime 迁移验收

本次范围是简历中的五组核心能力。对照本地 nanobot 源码（核对时 HEAD：edaef4e4f50bb1a00a9c76eac351e11e4c99ad4c），保留 duolaAgent 已有 CLI、配置、Context、Memory/Dream 和兼容入口，把运行时能力接入实际执行路径。

## 源码映射

| 能力 | nanobot | duolaAgent |
| --- | --- | --- |
| 七阶段、锁、队列、取消 | agent/loop.py | agent/loop.py |
| 工具循环与 checkpoint | agent/runner.py | agent/runner/runner.py、specs.py |
| 会话恢复与持久化 | session/manager.py | session/manager.py |
| immutable runtime | utils/llm_runtime.py | llm/llm_runtime.py；utils/llm_runtime.py 兼容入口 |
| snapshot、fallback | providers/factory.py、fallback_provider.py、base.py | llm/providers/ 下同名模块 |
| admission 与配置失效 | agent/model_runtime.py | agent/model_runtime.py |
| model-facing copy 治理 | agent/context_governance.py | agent/context_governance.py |
| token 估算及 offload | utils/helpers.py 的相关函数 | utils/helpers.py |
| 归档及 replay suffix | agent/memory.py、session/manager.py | 同名模块 |
| schema、参数转换、错误 | agent/tools/base.py、schema.py、registry.py | 同名模块 |
| Python entry-point plugin | agent/tools/loader.py | agent/tools/loader.py |
| shell / exec session | agent/tools/shell.py、exec_session.py、_windows_job.py、sandbox.py | 同名模块 |
| MCP | agent/tools/mcp.py | agent/mcp.py；tools/mcp.py 兼容入口 |
| 文件工具 | agent/tools/filesystem.py | agent/tools/tool_impl/filesystem.py；tools/filesystem.py 兼容入口 |
| Workspace / SSRF | security/workspace_access.py、workspace_policy.py、network.py | 同名模块 |
| 子 Agent | agent/subagent.py、tools/spawn.py、tools/context.py、tools/file_state.py | 同名模块 |
| 持续目标 | agent/tools/long_task.py、session/goal_state.py、turn_continuation.py | 同名模块 |

表中模块路径均相对于各自的 Python package 根目录。

## 1. Agent Turn Runtime

实际执行链：

```text
MessageBus -> AgentLoop.run -> _dispatch
  -> per-session lock -> concurrency gate -> _process_message
  -> restore / compact / command / build / run / save / respond
  -> _run_agent_loop -> AgentRunner.run -> Provider / ToolRegistry
```

- 同会话只有一个 worker；跨会话由 Semaphore 限制运行数。
- 后续消息进入有界 pending queue，每个 runner slice 最多注入 3 条；剩余消息保留排队。命令和模型 override 消息留到独立 turn。
- BUILD 先保存用户输入。checkpoint 保存 runner 新增消息，不重复保存整个历史前缀。
- 工具执行前保存 awaiting_tools；每个工具完成后再次保存，保留并行批次的部分结果；批次结束保存 tools_completed；最终文本生成后保存 final_response。
- 取消和下一轮 RESTORE 恢复已记录的 assistant/tool 消息，补齐没有结果的工具调用，成功保存后清除 checkpoint。
- /stop 取消会话 worker、所属 exec sessions 和子 Agent；aclose 关闭整个 runtime 的资源。

恢复的是已记录的上下文，不是进程内存或外部副作用事务。不自动重跑无法确认是否完成的工具，不承诺工具 exactly-once。

验证：test/test_runtime_pipeline.py。

## 2. LLM Runtime Snapshot / Fallback

- GenerationSettings、LLMRuntime、ProviderCandidate、ProviderSnapshot 均使用 frozen dataclass。不可变的是本轮选择及配置值；客户端仍可复用。
- 支持 defaults、命名 preset、session metadata 的 model_preset，以及消息 metadata 的 model_override / preset_override。
- ModelRuntimeResolver.invalidate 使下一次 admission 刷新，已开始的 turn 继续使用旧快照。旧客户端保留到 runtime 关闭。
- 快照取 primary 和直接配置的 fallbacks 的最小 context window，并为整条 fallback 链保留最大输出预留量。不递归展开 fallback preset 自己的 fallbacks。
- 错误分类区分 timeout、connection、rate_limit、server_error、authentication、quota、context_overflow、invalid_request；retry 与 fallback 的适用范围不同。
- primary 连续 3 次可 fallback 错误触发熔断，默认冷却 60 秒；半开阶段只允许一个探测请求。
- 已流出文本的非超时错误不重复请求；超时可在有限 retry / fallback 预算内携带已输出片段请求续写。模型续写不等价于网络字节流断点续传。
- OpenAI-compatible 使用可取消 HTTPX 请求与 SSE 工具参数组装。Anthropic 仅在 Provider 边缘转换 canonical tool_calls/tool 消息。

验证：test/test_provider_runtime.py。未调用付费 LLM API；协议与 SSE 使用模拟 HTTP transport。

## 3. Context Governance / Memory

- 在 model-facing 副本上修复缺失工具名、孤儿/重复结果、缺失结果，不原地改写持久 transcript。
- 旧版 Anthropic 历史读取后在副本上转换；新的持久记录使用 canonical assistant tool_calls / role=tool。
- 超大输出可写入 .duola/tool-results/<session-hash>/*.txt，返回路径和预览。保留 read_file 的 offload 豁免，避免 offload/read 循环。
- 先压缩当前 slice 中可压缩的工具结果，再裁剪历史；保留 user tail，并修复工具协议边界。
- COMPACT 估算完整 prompt，包括 system、历史、当前消息和工具 schemas；沿 user-turn 边界选择旧消息归档。
- 摘要失败则写有长度限制的 RAW 文本归档；归档完成后推进 last_consolidated。完整原始 session transcript 仍保留。
- replay 保留未归档尾部和近期 raw suffix：以最近 6 条为候选，向 user-turn 边界对齐。因此可能多于 6 条。
- 内置 Provider 当前使用字符估算，并提供同步 count_tokens 扩展入口；不宣称精确 tokenizer。固定 system/当前 user turn 本身过大时，不能保证仅靠历史裁剪装入窗口。

验证：test/test_context_governance.py，以及原有 Memory/Dream 测试。

## 4. Tool Execution Plane

- Registry 做 JSON object 解析、schema-driven 转换，以及基础类型、required、enum、范围、数组、对象、additionalProperties 校验；不是完整 JSON Schema Draft 校验器。
- 参数与执行错误通过 ToolResult.error 归一化，取消向上传递。
- concurrency_safe 工具组成并行批次，不安全工具形成顺序屏障。
- Python entry point 支持 nanobot.tools 和 duolaAgent.tools。每个 registry 创建独立插件实例；子 Agent 按 subagent scope 选择插件。
- shell 保留原项目进程树实现：Windows Job Object，Unix 进程组。exec session 支持 stdin、poll、终止、owner 校验、输出缓冲限制；增加未再次 poll 的 session deadline watcher。
- MCP 支持 stdio、SSE、streamable HTTP、动态工具/资源/提示注册和 reload；SDK AnyIO context 由独立 owner task 成对进入/退出。
- Web/MCP HTTP 的每次请求均校验目标，覆盖重定向和 SSE endpoint；DNS pinning；默认阻止私网、loopback、metadata 地址，可显式配置 CIDR whitelist。
- Workspace Scope 通过 ContextVar 绑定，文件路径与 shell 工作目录按 scope 校验。

默认 shell 防护是应用层策略，不等价于系统沙箱。Linux 可选 bwrap 已接入，但未在本次 Windows 环境运行。远端 MCP 服务内部行为也不由本地文件策略替代约束。

验证：test/test_tool_plane.py、test/test_mcp_subagent.py，包含真实 shell、Windows 后代进程取消和本地 MCP stdio 服务。

## 5. Subagent / Sustained Goal

- 子 Agent 构造独立 Registry、FileStates、RequestContext、exec manager、MCP 连接；绑定父 turn 的 Workspace Scope 和 immutable runtime。
- 支持 background 与 spawn(wait=true) inline；有并发上限、状态查询、父 session 关联、取消与结果回注。
- background 结果通过 bus/pending queue 返回父会话；inline 返回 spawn 工具结果。
- 子 Agent 运行状态为进程内状态，不是跨重启子进程恢复。
- /goal <objective> 明确启用目标创建权限。普通聊天不能通过 create_goal 自动启用持续执行。
- goal state 持久化至 session metadata；create/get/update 工具支持 complete/cancel/block/replace，保存失败回滚内存 metadata。
- goal 仍 active、slice 达到 iteration limit、有 pending queue 且接续次数未耗尽时，SAVE 先持久化会话与次数，再排入内部 continuation。
- 默认最多 12 次内部接续；耗尽后保留 active 状态并返回控制权，不伪造完成。直接调用内部 _process_message 没有 pending queue 时不自动接续。
- 内部队列不是 durable scheduler；重启后由下一条入站消息从已保存上下文继续，不承诺进程退出后自行唤醒。

验证：test/test_runtime_pipeline.py、test/test_mcp_subagent.py。

## 配置与使用

范例见 [core_runtime.example.json](core_runtime.example.json)，其中模型名与密钥变量是占位配置。

```powershell
uv sync --dev
uv run pytest -q
uv run ruff check duolaAgent
uv run duolaAgent agent --help
```

- /model <preset>：持久修改本 session preset。
- /reload：使后续 admission 重读模型配置；不等价于所有工具热更新。MCP 配置用 MCPProvider.reload，其他工具配置重建 runtime。
- /goal <objective>：明确请求持续目标。
- /stop：调度入口取消会话与关联资源。

消息级 override：

```python
await bus.publish_inbound(InboundMessage(
    channel="cli", sender_id="user", chat_id="demo", content="继续",
    metadata={"preset_override": "backup", "model_override": "provider-specific-model"},
))
```

交互 CLI 仍逐次输入和消费响应；mid-turn follow-up 的并发入口是 MessageBus。SDK/渠道发送同 session 的第二条消息即可触发队列注入。

## 验证与兼容

- 本轮验证：56 项 pytest 通过；ruff check duolaAgent 通过。
- 原有 16 项配置、Context、Memory/Dream 行为保留；Dream fake provider 增加 canonical tool 消息识别。
- 保留 LLMRunTime、LLMReponse 旧拼写 alias；runner 包导出 AgentRunner/AgentRunSpec。
- 保留旧 tool_impl/filesystem、agent/mcp 等入口，并提供 nanobot 风格兼容导出。
- 未迁移完整 WebUI/渠道/Provider-state sidecar/git-backed Dream；它们不属于本次五项验收。
- 原项目移植代码的 MIT 许可见仓库 LICENSE.nanobot。
