# Environment transforms

本模块提供六个可执行机制，但它们不是六个同层级的 EnvHarness 组件。它们只作用于 `ActionableEnv` 边界，不能修改任务 verifier。

| 组件 | 主要轴/执行槽 | 文件 | 用途 |
| --- | --- | --- | --- |
| Stage | Setup | `setup.py` | 重放合法动作前缀，构造可达初始状态 |
| Contract | `f_A` / visible contract | `contract.py` | 同源修改工具 schema 与显式约束；可要求参数或追加不含答案的工具使用引导 |
| Contract | `f_A` / action | `action.py` | 隐藏 guard、阻止或改写 action |
| Contract | `f_T` / transition | `transition.py` | 表达提交后 timeout 等转移语义 |
| Contract | `f_O` / observation | `observation.py`、`stale_field.py` | 改写可见 observation，不改变真实状态；后者支持从通用快照点路径取旧字段 |
| Extension | Budget | `budget.py` | 限制总步骤、写操作或其他资源 |

Chain 已进入分类，但 MVP 尚不允许 Modification Agent 自动生成。规则都必须实现 `reset/save_state/load_state`。有状态计数必须进入独立层快照，确保环境 DAG 中的节点可以确定性回放。

`add_tool_guidance` 只改变一个现有工具的 Agent 可见 description，不改变参数 schema、
工具执行、任务正文或 verifier。它适合“Agent 已看到工具，但没有完成 API 发现顺序”的
失败；它不是 skill 注入，也不能携带具体任务答案。
