# Agent System

当前环境进化侧有两个配置驱动角色：

- `DiagnosticAgent` 从失败 episode 中选一个最高优先级、可证伪的根因；
- `EnvironmentModificationAgent` 把该根因翻译为一个通过 catalog 与 `MutationSurface` 校验的
  `MutationSpec`。

两者不能改 task、官方 evaluator 或 Target 权重。system prompt、skills 和 provider 参数
都从 JSON 加载。`TargetAgent` 通用类仍保留给轻量审计，但 AppWorld 训练基线已经使用
Agent0 的 manager 和 Qwen3-4B，不再维护专用 Target 配置。

Target 上下文统一使用 Qwen3 Hermes 协议：工具 schema 位于 system 的 `<tools>` 中，模型
动作位于 assistant 的 `<tool_call>` 中，环境结果作为 user-role 的 `<tool_response>` 返回。
仓库不再提供 Target 长期 memory 模块，避免把 Agent 经验变化混入环境变化实验。
