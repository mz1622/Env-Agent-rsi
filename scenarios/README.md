# Scenario catalog

本目录定义 Env RSI 的首批五个任务族。每个目录都把研究说明与运行实现分开：README 负责问题、来源和实验设计，`scenario.json` 提供机器可读元数据；具体环境、verifier 和变换组件放在 `src/env_agent_rsi/`，不会复制到场景目录中。

| 顺序 | 场景 | 当前状态 | 首要能力 |
|---|---|---|---|
| 01 | [Exactly-once write](01_exactly_once_write/README.md) | 已有最小实现 | 不确定提交、幂等、确认后重试 |
| 02 | [Order lifecycle](02_order_lifecycle/README.md) | 设计完成 | policy、条件写入、退款与 collateral damage |
| 03 | [Issue workflow](03_issue_workflow/README.md) | 设计完成 | 多步骤状态流转与部分完成 |
| 04 | [Calendar and email](04_calendar_email/README.md) | 设计完成 | 跨应用依赖、时间约束、重复通知 |
| 05 | [Code repair](05_code_repair/README.md) | 设计完成 | 搜索、编辑、测试、提交 |

推荐实现顺序与编号一致。一个场景只有在具备确定性 reset、独立 verifier、Oracle 成功轨迹、至少一条从辅助环境回到目标环境的课程，以及无状态泄漏测试后，才能标记为 `implemented`。

