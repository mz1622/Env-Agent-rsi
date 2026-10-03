# Agent Runtime

这是 provider 与环境之间的低层循环：`AgentRunner` 把动态工具契约按 Qwen3 Hermes 模板
写入 system，`ModelClient.generate(messages, ())` 返回从 `<tool_call>` 解析的
`ModelOutput`，环境 observation 再以 user-role 的 `<tool_response>` 回到多轮上下文。

`info`、真实环境状态和官方 evaluator 数据不会进入 Agent 上下文。上层 `agent_system`
负责从 JSON 加载角色、system prompt、skills 与 provider；AppWorld 的训练执行则由 Agent0
接管，并复用同一个协议 renderer/parser。
