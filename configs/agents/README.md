# Agent Configurations

Target 与 Diagnostic 使用同一个 JSON schema：

- `role` 决定角色约束；
- `system_prompt` 与 `skills` 是相对当前配置文件解析的 JSON 路径；
- `max_steps` 和 `max_context_messages` 控制 episode 与消息历史；
- `provider.type` 选择 `api` 或 `adk`，`model` 与 `args` 原样交给对应 provider；
- API key 只写环境变量名；ADK 配置另外使用 `executor: "module:function"`。

代码也可以通过 `provider_overrides` 覆盖 model、base URL 或 args，同一个 Agent 类无需针对不同服务修改。
