# Core Protocols

核心层定义所有模块共享但不包含业务实现的边界：`Action`、`EnvResponse`、`EnvDescriptor`、`EvaluationResult`、`ActionableEnv`、工具 schema/校验、组件注册表和只读 verifier 协议。

该层不允许导入具体环境、Agent 或变化规则；它是 benchmark adapter、Harness 和 runner 能够独立替换的基础。
