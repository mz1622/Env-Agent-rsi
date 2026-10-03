# Model Providers

- `api.py`：通过标准 HTTP 调用 OpenAI-compatible chat completions API。
- `adk.py`：加载配置中的 `module:function` executor，把 Google ADK、OpenAI Agents SDK 或内部 ADK 的运行方式包到统一接口。
- `factory.py`：唯一的 provider 分派位置。Agent 本身只传 `ProviderSettings`，不出现 SDK 条件分支。

API key 优先通过配置指定的本地文件读取，也可回退到环境变量；密钥内容不写入 JSON、日志或运行记录。当前三个 LLM 角色统一读取仓库根目录的 `api.txt`。ADK 是可选接入：未安装任何 ADK 时，本地环境、replay 和测试仍可运行。
