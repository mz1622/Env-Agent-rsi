# AppWorld 接入说明

AppWorld 运行在独立 Python 3.12 worker 中，主包用 JSON Lines 请求 `reset`、`step`、
`observe`、`evaluate`、`save_state` 与 `load_state`。Agent 可见工具只有
`get_api_docs`、`execute_python`、`finish`。

Agent0 通过原生 `BaseTool`/`AsyncToolServer` 调用同一个 backend。一个 trajectory 对应一个
worker，finish 清理会关闭 worker。正式奖励只来自官方 evaluator 的签名结果，不把模型
自报成功当作分数。运行顺序见 `src/env_agent_rsi/integrations/agent0/README.md`。

