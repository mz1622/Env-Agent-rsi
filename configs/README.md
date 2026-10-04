# 配置目录

当前配置只指向 AppWorld：

- `agent0/`：Qwen3-4B、Agent0 ADPO、parquet 与工具服务参数；
- `agents/`：本地 Qwen Target 与 DeepSeek Diagnostic/Modifier 的模型配置；
- `benchmarks/`：AppWorld 官方 revision、数据版本和本地布局；
- `environment_changes/`：AppWorld 当前支持的环境变化组合；
- `prompts/`、`skills/`：提示与工具使用策略。

任务正文、数据库、ground truth、生成后的 parquet 与密钥不进入配置目录。
