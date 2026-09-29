# Python Package Map

`env_agent_rsi` 是项目的可安装 Python 包。公共链路由 `core` 协议开始，经 `environments`/`benchmarks`、`transforms`、`harness` 进入 `agent_runtime` 与 `agent_system`；`evolution` 保存诊断、变化和环境 DAG。

每个子目录都有自己的 README 和中文文件头设计说明。顶层 `__init__.py` 只导出常用稳定接口；新增具体实现应进入对应子模块，并通过协议或注册表接入。
