# 环境侧 Agent 配置

`target_qwen3_4b.json` 固定本地 Qwen3 4B Target，并通过 Ollama 使用 Agent0/Hermes
文本工具协议。`diagnostic_agent.json` 与 `environment_modifier_agent.json` 分别通过
DeepSeek API 输出结构化失败签名和一个 allowlist mutation；两者不执行 AppWorld
任务。训练路径仍由 `configs/agent0/appworld_minimal.json` 与 Agent0 负责。
