# AppWorld × Agent0 架构

Target Agent 的能力更新交给 Agent0 的 Qwen3-4B + ADPO。Env-Agent-RSI 不重写训练器，只让
同一 Target 在不同 AppWorld 环境版本上产生轨迹，并让 Diagnostic/Modifier 逐次提出一个
环境变化。

## 一次闭环

1. AppWorld task 转成 Agent0 已有 parquet schema。
2. Agent0 rollout 生成 `<tool_call>`，原生异步工具服务器调用 `AppWorldAgent0Tool`。
3. action 进入隔离的 `AppWorldProcessBackend`，官方 evaluator 产生签名 reward。
4. 失败轨迹交给 Diagnostic，形成一个有证据的 `FailureSignature`。
5. Modifier 只提出当前最高优先级、且在 `MutationSurface` allowlist 内的一次变化。
6. paired rollout 用相同 task/seed 比较 parent 与 child，结果和完整 spec 写入持久化
   `EnvironmentBucket`。
7. bucket 保留所有 child，但只有严格优于 incumbent 的节点成为新 best；下一轮
   `BestFirstEnvironmentSearch` 从该 best 继续扩展。

```text
core <- benchmarks/appworld <- integrations/agent0 -> Agent0 submodule
  ^              ^
  |              |
transforms <- harness <- evolution + orchestration <- diagnostic/modifier
```

```text
EnvironmentBucket
├── bucket.json       # DAG、评测聚合、best 指针与 best 历史
└── specs/
    └── <sha256>.json # 内容寻址的完整环境配置

best(parent) --MutationSpec--> candidate(child) --paired eval--> 保留或移动 best
```

bucket 的 scope 必须固定评测语义。同一任务可使用一棵 task-level 树；跨任务搜索则应
保存环境模板，并用同一任务集合、seed 集与 evaluator 聚合每个节点，不能直接比较来自
不同任务的单次分数。

AppWorld evaluator 是唯一成功判据。Diagnostic/Modifier 不得改 task、ground truth、评分器或
Target 权重。通用协议有六类变化，但 AppWorld MVP 先开放 Setup/Contract/Action/Budget；
`f_T/f_O` 等 worker 具备可靠事务事件和 observation 映射后再开放。
