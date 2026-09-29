# Environment transforms

本模块提供项目允许搜索和组合的六类环境变化。它们只作用于 `ActionableEnv` 边界，不能修改任务 verifier。

| 类型 | 文件 | 执行位置 | 用途 |
| --- | --- | --- | --- |
| Setup | `setup.py` | `reset` 之后、Agent 开始之前 | 重放合法动作前缀，构造可达初始状态 |
| Contract | `contract.py` | descriptor 和执行前 | 同源修改工具 schema 与显式约束 |
| `f_A` | `action.py` | 基础 action 之前 | 隐藏 guard、阻止或改写 action |
| `f_T` | `transition.py` | 基础状态转移之后 | 表达提交后 timeout 等转移语义 |
| `f_O` | `observation.py` | 返回 Agent 之前 | 改写可见 observation，不改变真实状态 |
| Budget | `budget.py` | action 前后 | 限制总步骤、写操作或其他资源 |

规则都必须实现 `reset/save_state/load_state`。有状态计数必须进入快照，确保环境 DAG 中的节点可以确定性回放。
