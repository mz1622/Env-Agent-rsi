# Agent Runtime

这是 provider 与环境之间的低层循环：`ModelClient.generate(messages, tools)` 返回一个规范化 `ModelOutput`，`AgentRunner` 执行工具并把 observation 作为 tool message 放回多轮上下文。

`info`、真实环境状态和官方 evaluator 数据不会进入 Agent 上下文。上层 `agent_system` 负责从 JSON 加载角色、system prompt、skills 与 provider；AppWorld 的训练 rollout 则由 Agent0 接管。
