# Agent0 配置

`appworld_minimal.json` 只保存 Env-Agent-RSI 相对 Agent0 官方 Qwen3-4B ADPO 示例的差异：
AppWorld parquet 路径、工具服务地址、trajectory 长度和单机训练规模。PPO/ADPO、veRL、
vLLM、FSDP 与 checkpoint 配置仍由 `third_party/Agent0` 的上游代码解释。

启动器固定启用 MTRL user role，使 AppWorld 返回按 Qwen3 原生
`<tool_response>` 进入下一轮；动作边界仍使用 `</tool_call>`。

这里的 4 train + 1 validation 默认切分只用于确认接入可以运行，不是论文最终评测。
