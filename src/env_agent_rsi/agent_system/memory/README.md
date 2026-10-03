# Agent Memory

该模块是 Target Agent 的只读长期记忆边界。Memory 不属于环境，也不进入 verifier；它保存在模型上下文之外，每个 episode 开始时根据 `task_id`、任务指令和初始观察检索，返回 JSON 记录后作为低权限任务数据注入 user message。

当前实现：

- `protocol.py`：`MemoryQuery`、`MemoryRetriever` 和空实现；
- `json_store.py`：读取版本化 JSON，支持 `task_ids` 作用域、`keywords` 和 `priority` 的透明确定性排序；
- `factory.py`：从 Agent 配置选择 `null` 或 `json` 实现；
- 不提供 writer，不允许一次 rollout 自动修改长期记忆。

记录格式：

```json
{
  "id": "container-prerequisite-v1",
  "kind": "semantic",
  "content": {
    "lesson": "Before taking an object from a closed container, open it."
  },
  "task_ids": [],
  "keywords": ["container", "closed"],
  "priority": 0
}
```

`task_ids` 为空表示全局候选；非空时只对指定任务可见。当前默认文件没有记录，因此接入接口不会改变已有 Target 行为。后续 Agent RSI 的记忆提炼、验证、写入、回滚和训练/测试冻结策略应作为独立模块实现。
