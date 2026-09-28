# 02 — Order lifecycle

## 任务是什么

Agent 根据订单状态、商品、付款方式和用户约束，执行取消、修改、退货或换货。代表任务是：“如果订单尚未发货，取消空气净化器并退款到礼品卡；若礼品卡退款不可行，则不要修改订单。”正确答案可能是写操作，也可能是保持数据库不变。

动作覆盖身份确认、订单搜索、精确读取、policy 检查、不可逆修改、退款确认和结束。Verifier 比较最终订单/退款表，并检查其他商品、订单和付款方式没有 collateral change。

## 来源是什么

- τ-bench 把真实业务 policy、LM 模拟用户、工具 API 和最终数据库状态结合起来：<https://arxiv.org/abs/2406.12045>
- τ-bench 的零售 policy 明确规定 pending/delivered 状态、取消理由、退款方式和某些修改工具只能调用一次：<https://github.com/sierra-research/tau-bench/blob/main/tau_bench/envs/retail/wiki.md>
- AppWorld 提供本地可控应用、API、数据库和状态单元测试：<https://arxiv.org/abs/2407.18901>

本目录的可执行任务具体改编自 τ-bench `tasks.py` 的 `tasks[66]`：Aarav Lee 希望把 luggage set 换成 coat，如果不可行则取消订单。这里保留身份验证、读取订单、禁止不兼容替换、取消、原路退款和 collateral-damage 检查，并替换了原 benchmark 的合成 ID。来源定位写在 `task.json`，因此它不是对 τ-bench 原环境的逐字复制。

## 哪些工作用了这个问题

- τ-bench 使用 retail 和 airline 场景研究 Agent、用户、policy 与工具之间的交互。
- τ²/τ³-bench 延伸了多方控制、修订任务和更多业务域；新实验应固定具体版本，避免旧任务与新版 policy 混用。
- EnvHarness 把 Web/API 环境包装成可改变 Setup 与 Contract 的训练环境。

## 哪些 benchmark 与它有关

- **τ-bench / τ³-bench**：最直接的订单与航空服务 benchmark。
- **AppWorld**：Amazon 等模拟应用、多 API 长程任务和状态 verifier。
- **WebArena Shopping / Shopping Admin**：浏览器界面的搜索、购物和后台管理任务。<https://arxiv.org/abs/2307.13854>

## 环境课程

1. 已认证、单个订单、直接给订单 ID、操作正常返回。
2. Agent 自己搜索订单，但环境提供明确 policy 错误。
3. 增加相似订单和商品，仍保持读取一致。
4. 取消已提交但返回 operation ID。
5. 去掉 operation ID；订单状态立即可查询。
6. 订单和退款分别延迟可见，并逐步撤掉 stale 标记。
7. 恢复目标预算与完整 policy。

课程只改变信息和交互辅助，不改变最终订单目标与 collateral-damage 检查。

## 可执行实现

- 任务配置：[`task.json`](task.json)
- 环境：`src/env_agent_rsi/environments/order_lifecycle.py`
- verifier：`src/env_agent_rsi/verifiers/order_goal.py`
- Oracle：`src/env_agent_rsi/scenario_agents.py::order_oracle`

```bash
env-agent-rsi-task scenarios/02_order_lifecycle/task.json --agent oracle
```
