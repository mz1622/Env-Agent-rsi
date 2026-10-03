# AppWorld official adapter

本目录不保存 AppWorld 任务、数据库、参考解或 evaluator。`AppWorldProcessBackend` 为每个
trajectory 启动一个 Python 3.12 worker，并用 JSON Lines 交换标准动作和 observation。

Agent 可见工具是 `get_api_docs`、`execute_python`、`finish`。官方 evaluator 和真实数据库
状态不会进入 Agent 上下文。`save_state/load_state` 只恢复官方数据库，不恢复 Python shell
变量或模型上下文，因此标记为 `official_database_only`。

首版 `MutationSurface` 开放 Setup、Contract、Action (`f_A`) 与 Budget。`f_T/f_O` 需要原生
事务边界和具体响应字段映射，尚未开放。Agent0 接入通过同一个 backend 完成，见
`src/env_agent_rsi/integrations/agent0/README.md`。

