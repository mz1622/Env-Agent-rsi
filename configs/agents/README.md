# 环境侧 Agent 配置

`diagnostic_agent.json` 与 `environment_modifier_agent.json` 分别输出结构化失败签名和一个
allowlist mutation；两者不执行 AppWorld 任务。Target 的训练路径由
`configs/agent0/appworld_minimal.json` 中的本地 `Qwen/Qwen3-4B-Base` 和 Agent0 负责。
