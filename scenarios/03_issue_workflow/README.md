# 03 — Multi-step issue workflow

## 任务是什么

Agent 找到指定工单或代码 issue，确认身份和当前状态，设置负责人，添加诊断评论，再把状态推进到目标阶段。例如：“找到支付超时问题，分配给 Bob，添加复现信息，并标记为 In Progress。”

Verifier 独立检查 issue ID、assignee、评论内容与状态，同时确认没有修改同名 issue。这个任务天然包含多个有依赖关系的写操作，适合研究部分完成、补偿和 Chain。

## 来源是什么

- WebArena 提供可复现的 GitLab 网站以及 issue、merge request 和 repository 任务：<https://arxiv.org/abs/2307.13854>
- WebArena-Verified 对任务与 evaluator 做人工审计，并提供确定性离线评分工具：<https://github.com/ServiceNow/webarena-verified>
- WorkArena 使用 ServiceNow 的企业工作流任务评价知识工作 Agent：<https://www.servicenow.com/research/publication/alexandre-drouin-work-icml2024.html>

本目录以 WebArena-Verified 数据中的 `task_id=446` 为种子：定位 a11yproject 中关于 404 错误的 issue 并分配给 Roshanjossey。为了得到一个能研究 Chain 和部分完成的最小环境，本任务明确增加“添加诊断评论”和“推进到 In Progress”两个写步骤；这是标注过的组合改编，不宣称是 benchmark 原题。

## 哪些工作用了这个问题

- WebArena 的 GitLab 子集包含打开 issue、处理 merge request、创建 repository 等协作开发任务。
- WorkArena/WorkArena++ 使用企业表单、问题单、服务目录和组合工作流。
- EnvHarness 在 WebArena 上应用统一的环境包装与技能诱导流程：<https://github.com/google-research/envharness>
- WonderBread 使用 WebArena GitLab/CRM 等工作流研究过程级任务与 demonstration：<https://papers.neurips.cc/paper_files/paper/2024/file/d1fa821312040303b089ae529dbf81a6-Paper-Datasets_and_Benchmarks_Track.pdf>

## 哪些 benchmark 与它有关

- **WebArena GitLab**：浏览器级 issue 和 merge request 操作。
- **WebArena-Verified**：更可靠的 task/evaluator 版本。
- **WorkArena / WorkArena++**：ServiceNow 企业工单和组合任务。
- **AppWorld Todoist**：API 级任务、列表和状态更新，可作为更轻的 Bridge。

## 环境课程

1. 直接给 issue/user ID，并逐步确认每个写操作。
2. Agent 自己搜索唯一 issue 与用户。
3. 增加相似标题、多个用户和分页。
4. `assign_issue` 提交后返回不确定结果，但读取立即可见。
5. 评论或状态更新延迟可见，并提供版本信息。
6. 撤掉版本提示、收紧写入预算，恢复完整多步骤目标。

如果 Agent 学会了可靠的“读—写—确认”模式，父节点成功后，分别训练 assign/comment/status 的更容易子节点应全部剪枝。

## 可执行实现

- 任务配置：[`task.json`](task.json)
- 环境：`src/env_agent_rsi/environments/issue_workflow.py`
- verifier：`src/env_agent_rsi/verifiers/issue_goal.py`
- Oracle：`src/env_agent_rsi/scenario_agents.py::issue_oracle`

```bash
env-agent-rsi-task scenarios/03_issue_workflow/task.json --agent oracle
```
