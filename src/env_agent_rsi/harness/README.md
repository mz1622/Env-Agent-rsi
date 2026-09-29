# Rule Harness

Harness 把基础环境、verifier 与六类变化装配成一个 `ActionableEnv`。固定顺序是 Setup → Contract → schema validation → Action → Budget pre-check → base transition → Transition → Observation → Budget accounting。

`factory.py` 使用显式注册表从 JSON 构建组件；`wrapper.py` 执行规则、维护动态 contract version 和完整快照；`rules.py` 只保留旧导入路径兼容。
