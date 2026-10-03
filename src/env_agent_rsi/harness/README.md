# Rule Harness

Harness 把唯一的 `appworld_process` 环境与环境变化规则装配成 `ActionableEnv`。固定顺序是
Setup → Contract → schema validation → Action → Budget pre-check → base transition →
Transition → Observation → Budget accounting。

- `factory.py`：只注册 AppWorld 和六类通用规则；
- `layers.py`：稳定的组件类型、轴、执行槽和独立状态；
- `wrapper.py`：执行 pipeline 并维护动态 contract version；
- `checkpoint.py`：分别保存 base state 与 layer state。

成功判定由 AppWorld 官方 evaluator 完成，不再有按数据集注册的 verifier。
