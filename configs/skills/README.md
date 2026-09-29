# Skills

Skill 是可复用的任务策略，不是 Python 代码。每个 JSON 文件包含 `name`、`description` 和有序 `instructions`；`ConversationContext` 会把配置选择的 skills 渲染到 system message 中。

Skill 不自动获得额外工具，也不能访问 verifier 或 hidden state。环境实验应记录加载了哪些 skill，防止把 Agent 侧改进误记为环境侧改进。
