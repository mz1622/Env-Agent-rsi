# 测试边界

本目录只测试 Env-Agent-RSI 自身：AppWorld adapter、Agent0 薄桥、签名奖励、环境变化和
诊断/修改模块。`pyproject.toml` 将 pytest 收集范围限制在这里；Agent0 上游的 GPU/Ray/vLLM
测试由其官方环境单独运行。

