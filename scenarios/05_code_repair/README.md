# 05 — Code repair

## 任务是什么

Agent 根据 issue 描述在真实或缩小的代码仓库中定位问题、修改文件、运行测试并提交 patch。Verifier 执行 FAIL_TO_PASS 与 PASS_TO_PASS 测试，并可附加修改范围、安全性和禁止网络访问等约束。

动作空间覆盖 `list_files`、`search_code`、`read_file`、`edit_file`、`run_tests` 和 `submit`。环境辅助只能改善可获得的诊断信息或缩小无关搜索空间，不能提供 gold patch 或直接泄露修改行。

## 来源是什么

- SWE-bench 从真实 GitHub issue 和对应修复 PR 构造 repository-level 问题，以测试验证生成 patch：<https://arxiv.org/abs/2310.06770>
- SWE-bench Verified 通过人工核验提高问题与测试的一致性：<https://openai.com/index/introducing-swe-bench-verified/>
- LiveCodeBench 提供持续更新、控制污染的代码能力评价，可作为函数级和新鲜任务补充：<https://arxiv.org/abs/2403.07974>

本目录的可执行 mini-repo 是 SWE-bench Lite 实例 `astropy__astropy-14365` 的语义缩小版；上游 issue 是 Astropy #14365：QDP 命令本应大小写不敏感，但读取器只接受大写。这里重新实现了一个只使用 Python 标准库的两文件仓库，保留 lower-case FAIL_TO_PASS 与 upper-case PASS_TO_PASS 行为，但不复制 Astropy 仓库，也不向 Agent 提供 gold patch。

## 哪些工作用了这个问题

- Darwin Gödel Machine 在 SWE-bench 和 Polyglot 上评价自修改 Coding Agent：<https://arxiv.org/abs/2505.22954>
- SICA 在 SWE-bench Verified、LiveCodeBench 和合成 Agent benchmark 上让 Coding Agent 修改自己：<https://arxiv.org/abs/2504.15228>
- EnvHarness 把 SWE-bench Verified 接入统一环境接口，并保留测试 verifier：<https://github.com/google-research/envharness>
- SWE-agent、OpenHands 等 Coding Agent 广泛使用 SWE-bench 系列作为评价环境。

## 哪些 benchmark 与它有关

- **SWE-bench / SWE-bench Verified**：真实仓库、issue 和单元测试。
- **SWE-bench Live / SWE-rebench**：更新、更抗污染的 issue 修复任务。
- **LiveCodeBench**：较轻量的新鲜代码问题。
- **Polyglot**：多语言 Coding Agent 能力。
- **Terminal-Bench**：更广的终端工作流，可作为后期迁移目标。

## 环境课程

1. 给出相关文件、完整 stack trace 和单个失败测试。
2. 去掉相关文件路径，Agent 使用搜索定位。
3. 只给失败测试名，Agent 自己复现和读取日志。
4. 只给 issue 描述，但保留完整确定性测试输出。
5. 逐步恢复日志截断、命令预算和测试成本。
6. 恢复原始仓库快照和 benchmark harness。

课程不得把 gold diff、PR 评论中的答案或隐藏测试内容放进 observation。代码场景运行慢且容易出现依赖与 flaky-test 噪声，因此排在状态化 API 场景之后。

## 可执行实现

- 任务配置：[`task.json`](task.json)
- mini-repo 与测试：`src/env_agent_rsi/code_tasks/qdp_case.py`
- 环境：`src/env_agent_rsi/environments/code_repair.py`
- verifier：`src/env_agent_rsi/verifiers/code_repair_goal.py`
- Oracle：`src/env_agent_rsi/scenario_agents.py::code_repair_oracle`

测试代码会在一次性临时目录的独立 Python 进程中执行；这是研究用隔离，不是面向不可信代码的安全沙箱。

```bash
env-agent-rsi-task scenarios/05_code_repair/task.json --agent oracle
```
