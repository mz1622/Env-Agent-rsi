# Python 包结构

```text
env_agent_rsi/
├── benchmarks/appworld/     官方 AppWorld 的隔离进程 backend
├── integrations/agent0/     Agent0 数据、工具、奖励、训练启动薄层
├── agent_runtime/           Agent0/Qwen3 工具协议与通用模型边界
├── agent_system/            Target、Diagnostic 与 Modifier 配置层
├── core/                    ActionableEnv、descriptor、response
├── transforms/              六类环境变化机制
├── harness/                 规则分层、装配、checkpoint
├── evolution/               diagnosis、mutation、surface、DAG
└── orchestration/           paired rollout、轨迹与隔离
```

依赖从通用协议指向具体适配；`core` 不引用 benchmark。仓库不再含第二套数据集环境、
verifier 注册表或 hand-written scenario。
