# Benchmark Adapter

`BenchmarkAdapter` 是外部数据集与项目内部协议的隔离层。每个 benchmark 只需实现 `BenchmarkBackend`：提供任务、原生 reset/step、只读 observe、独立 evaluate、快照和允许变化的 `MutationSurface`。

Adapter 负责把工具转换成统一的 OpenAI function schema、在进入后端前校验 action，并把各种原生返回值封装成 `EnvResponse`。后端的 hidden state 与评分器不会进入 Target Agent 上下文。

接入新 benchmark 时建议为它建立独立子目录，并在其中说明：数据来源与许可证、下载方式、原生 action 到 tool 的映射、状态快照方法、verifier 信任边界，以及六类环境变化中哪些被允许。
