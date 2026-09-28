# 01 — Exactly-once write

## 任务是什么

Agent 必须把目标值写入状态化 API，最终恰好存在一次，并且不修改已有数据。动作空间是 `list_items`、`get_item`、`append_item` 和 `finish`。目标 verifier 始终读取真实状态，检查目标数量为一、初始数据不变且 episode 已结束。

该场景研究的核心不是数据库语法，而是一个经典分布式系统问题：写入可能已经提交，但客户端在收到确认前超时。Agent 必须使用幂等键、读后确认和有限重试，不能把 timeout 直接解释成失败。

## 来源是什么

- Stripe 的官方 API 文档说明所有 POST 请求可以携带 idempotency key，从而在连接错误后安全重试，避免重复创建对象：<https://docs.stripe.com/api/idempotent_requests>
- AWS Builders' Library 讨论了请求超时、迟到响应和带副作用重试，并用 client request ID 提供 at-most-once 语义：<https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/>
- 本仓库把这些生产系统语义缩小成可确定性重放的最小环境。

## 哪些工作用了这个问题

- Stripe 和 AWS 的生产 API 把幂等标识作为安全重试协议的一部分。
- ToolSandbox 研究带有隐式状态依赖的 stateful tool use，而不是只评价无状态函数调用：<https://arxiv.org/abs/2408.04682>
- EnvHarness 的 Contract/Rules 思路允许在不改原 verifier 的前提下改变动作、转移和观察协议：<https://arxiv.org/abs/2608.19880>

## 哪些 benchmark 与它有关

- **AppWorld**：多应用 API、真实数据库状态和 collateral-damage 检查。<https://arxiv.org/abs/2407.18901>
- **τ-bench / τ³-bench**：订单与数据库状态修改，最终按数据库目标状态评分。<https://arxiv.org/abs/2406.12045>
- **ToolSandbox**：stateful、interactive tool-use 任务和中间状态依赖。<https://arxiv.org/abs/2408.04682>

## 环境课程

1. 正常写入、最新读取、宽松预算。
2. `f_A` 要求显式 idempotency key，并返回可操作的结构化错误。
3. `f_T` 在提交后返回带 operation ID 的 unknown 状态。
4. 去掉 operation ID，但保持读取最新。
5. `f_O` 返回带 stale 标记的旧数据。
6. 去掉 stale 标记，恢复目标环境中的 post-commit timeout 与陈旧读取。

每一层成功后撤掉一项辅助并重新测试父节点；解决父节点后，其更容易的后代不再运行。

## 当前实现

- 可执行任务配置：[`task.json`](task.json)
- 基础环境：`src/env_agent_rsi/micro_api/item_env.py`
- verifier：`src/env_agent_rsi/verifiers/exactly_once.py`
- `f_A`：`src/env_agent_rsi/transforms/action.py`
- `f_T`：`src/env_agent_rsi/transforms/transition.py`
- `f_O`：`src/env_agent_rsi/transforms/observation.py`
- 配置：`configs/micro_api/`

```bash
env-agent-rsi-task scenarios/01_exactly_once_write/task.json --agent oracle
```
