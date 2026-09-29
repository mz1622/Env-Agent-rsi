# Model Providers

- `api.py`：通过标准 HTTP 调用 OpenAI-compatible chat completions API。
- `adk.py`：加载配置中的 `module:function` executor，把 Google ADK、OpenAI Agents SDK 或内部 ADK 的运行方式包到统一接口。
- `factory.py`：唯一的 provider 分派位置。Agent 本身只传 `ProviderSettings`，不出现 SDK 条件分支。

API key 只通过配置指定的环境变量名读取，不写入 JSON。ADK 是可选接入：未安装任何 ADK 时，本地环境、replay 和测试仍可运行。
