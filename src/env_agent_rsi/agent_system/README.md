# Agent System

该目录实现两个职责严格分开的 Agent：

- `TargetAgent` 只执行 benchmark 任务，是环境变化希望帮助的对象。
- `DiagnosticAgent` 只分析 episode 轨迹和 verifier 结果，输出失败签名与候选变化，不拥有环境写权限，也不能改判成功。

两者都由同一个 `AgentConfig` 装配。`system_prompt` 和 `skills` 是 JSON 文件路径；模型差异放在 `provider.type/model/args`。`ConversationContext` 固定保留系统信息和原始任务，并以 assistant tool call + tool result 为完整轮次维护历史，支持真正的多轮工具任务。

环境、Agent 和 provider 三者没有相互硬编码：Target 只依赖 `ActionableEnv`，Diagnostic 只依赖 `EpisodeResult`，两者都只依赖 `ModelClient`。
