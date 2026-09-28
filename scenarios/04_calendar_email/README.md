# 04 — Calendar and email

## 任务是什么

Agent 找到参会者、比较日历和时区、选择符合约束的空闲时间、创建一次会议，然后只发送一次通知邮件。代表任务是：“找到 Alice 和 Bob 下周共同空闲的 30 分钟，避开 Alice 上午十点前的时间，创建会议并通知两人。”

Verifier 检查唯一事件、参会者、时间、时区、会议时长和唯一邮件；错误收件人、时间冲突、重复事件或重复邮件均为 collateral damage。

## 来源是什么

- WorkBench 是包含五个数据库、26 个工具和 690 个 workplace 任务的 sandbox，明确覆盖发送邮件和安排会议：<https://openreview.net/forum?id=4HNAwZFDcH>
- ToolSandbox 提供联系人、消息、提醒、时间等带状态依赖的交互式工具任务：<https://arxiv.org/abs/2408.04682>
- AppWorld 提供 Gmail、Todoist、Phone 等本地应用和状态 evaluator，可用于相邻的跨应用任务：<https://github.com/StonyBrookNLP/appworld>

本目录的具体样本改编自 WorkBench `multi_domain_tasks_and_outcomes.csv` 的零基数据行 151：Leila 有 overdue tasks 时，在第二天最早空闲时间创建 30 分钟会议，并发送指定主题和正文的邮件。最小环境把条件预先设为 true，但保留原任务的联系人、事件标题、收件人、邮件内容、时长和 `13:00` 最早空闲结果。

## 哪些工作用了这个问题

- WorkBench 直接以 workplace email 与 meeting scheduling 评价规划、工具选择和多动作执行。
- ToolSandbox 研究联系人、消息、时间和系统状态之间的隐式依赖。
- OSWorld 使用真实桌面应用、截图和文件状态执行办公任务：<https://arxiv.org/abs/2404.07972>
- AppWorld 研究跨多个日常应用的长程 API 调用。

## 哪些 benchmark 与它有关

- **WorkBench**：最直接的邮件与会议 sandbox。
- **ToolSandbox**：轻量、状态化的联系人/消息/时间工具环境。
- **AppWorld**：Gmail 等 API 应用和跨应用状态验证。
- **OSWorld**：真实邮件、日历或办公 GUI 的后期迁移目标。

## 环境课程

1. 直接提供联系人 ID、统一时区和唯一空闲时间。
2. Agent 自己查联系人与空闲时间，但环境标记冲突。
3. 增加时区和多个可行时间，保留结构化约束提示。
4. 创建事件后返回 operation ID，邮件保持正常。
5. 去掉 operation ID；日历读取短暂陈旧但有版本号。
6. 邮件发送也可能返回不确定结果，Agent 必须避免重复通知。
7. 撤掉辅助字段，恢复跨应用目标和严格副作用预算。

这是首批场景中最典型的 Chain：邮件步骤依赖创建事件的真实结果，因此应在前三个单应用场景稳定后再实现。

## 可执行实现

- 任务配置：[`task.json`](task.json)
- 环境：`src/env_agent_rsi/environments/calendar_email.py`
- verifier：`src/env_agent_rsi/verifiers/calendar_email_goal.py`
- Oracle：`src/env_agent_rsi/scenario_agents.py::calendar_email_oracle`

```bash
env-agent-rsi-task scenarios/04_calendar_email/task.json --agent oracle
```
