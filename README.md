# Env-Agent-rsi

本仓库先研究环境侧的 evolve：冻结 Agent，把变化限制在环境的初始状态、动作/观察契约、转移故障、资源预算与任务组合上。

## Current research deliverables

- `research/env_evolution_report.tex`：基于 EnvHarness 的环境演化路线、相关工作、搜索策略与最小落地方案。
- `research/minimal_env_plan.md`：可直接进入实现的里程碑、接口、首批环境变体与验收条件。
- `outputs/env-evolution-research/benchmark_landscape.xlsx`：Agent harness、Agent RSI、通用 Agent 与环境演化工作的 benchmark 对照、例子、优先级、评分和来源。

## Starting decision

第一版使用本地、确定性、状态化 API 微环境，任务是“在提交前/提交后故障下恰好写入一次”。先实现 EnvHarness 风格的 Setup/Stage 与 Rules/Contract，验证环境生成、回放、独立 verifier 和 lineage DAG；Link/Chain 与 Agent 自我改写延后。

## Minimal runnable environment

现在仓库内已经有一个可运行的环境侧 MVP。任务是把 `target-item` 恰好写入一次，然后结束 episode。

- 基础环境：`append_item`、`list_items`、`get_item`、`finish` 四个工具。
- `f_T`：`post_commit_timeout` 在写入已经提交后，把成功响应替换为超时。
- `f_O`：`stale_read_after_write` 让写入后的前若干次列表查询看到旧状态。
- verifier：直接读取真实状态，检查目标值恰好出现一次，并且初始数据没有被修改。
- snapshot：同时保存基础环境和规则内部状态，恢复后故障时序保持一致。

### Run

无需第三方运行时依赖：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

安全策略使用稳定的幂等键，并在不确定写入后确认真实结果：

```bash
env-agent-rsi-demo --agent oracle
```

对照策略收到超时后直接重复写入，因此会产生两个目标对象并被 verifier 判定失败：

```bash
env-agent-rsi-demo --agent naive
```

默认故障组合定义在 `configs/micro_api/postcommit_stale.json`。通过 JSON 可以调整超时触发次数和陈旧读取次数，不需要修改 Agent。

### Interaction order

每一步严格按以下顺序运行：

```text
Action
  -> base environment transition
  -> f_T transition rules
  -> f_O observation rules
  -> agent-visible observation
```

其中 `EnvResponse.observation` 提供给 Agent，`EnvResponse.info` 只供 harness 记录故障触发与调试信息；最终 verifier 始终读取未经 `f_O` 修改的真实状态。
