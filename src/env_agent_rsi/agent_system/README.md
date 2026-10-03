# Agent System

当前环境进化侧有两个配置驱动角色：

- `DiagnosticAgent` 从失败 episode 中选一个最高优先级、可证伪的根因；
- `EnvironmentModificationAgent` 把该根因翻译为一个通过 catalog 与 `MutationSurface` 校验的
  `MutationSpec`。

两者不能改 task、官方 evaluator 或 Target 权重。system prompt、skills、只读 memory 和
provider 参数都从 JSON 加载。`TargetAgent` 通用类仍保留给轻量审计，但 AppWorld 训练基线
已经使用 Agent0 的 rollout manager 和 Qwen3-4B，不再维护专用 Target 配置。
