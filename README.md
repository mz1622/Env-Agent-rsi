# Env-Agent-rsi

本仓库先研究环境侧的 evolve：冻结 Agent，把变化限制在环境的初始状态、动作/观察契约、转移故障、资源预算与任务组合上。

## 端到端链路

```text
task.json
   │ environment / task / verifier / rules
   ▼
Factory + Registries
   │ 构建 BaseEnv、Verifier、Setup/Contract/f_A/f_T/f_O/Budget
   ▼
RuleHarness.reset()
   │ reset 基础环境并重放 Setup actions
   ▼
RuleHarness.describe()
   │ EnvDescriptor(task, tools, contract_version)
   ▼
TargetAgent + AgentRunner
   │ messages 与 tools 分开传给 ModelClient
   ▼
ModelOutput(tool_name, arguments)
   ▼
每轮：Contract → schema → f_A → Budget → Base → f_T → f_O → Budget accounting
   ▼
Agent-visible observation
   │ 只有 observation 进入模型历史；info 留给诊断
   ▼
独立 Verifier 读取真实状态；DiagnosticAgent 只分析失败轨迹
```

环境通过 `EnvDescriptor` 告诉 Agent 当前任务、允许的工具和 JSON 参数约束。Agent 不导入具体环境类，也不读取 handler；切换环境时，`AgentRunner` 在 reset 后重新调用 `env.describe()` 并把当前 `tools` 绑定给模型。若 Contract 在 episode 中变化，`contract_version` 会变化，下一轮模型调用自动加载新工具集。

工具契约与执行约束使用同一来源：`ContractRule` 同时改变 schema 和执行前校验；仅希望在调用后暴露约束时，使用隐藏的 `ActionRule`。这样可以分别实验“Agent 预先看见帮助”和“Agent 从错误中学习”。

## 代码结构总览

下面列出仓库内全部 Python 文件及其职责。依赖方向是 `core → environment/transform/verifier/benchmark → harness/evolution → agent_runtime/agent_system → CLI`；`core` 不导入任何具体场景，verifier 不读取 Agent observation。

```text
src/env_agent_rsi/
├── __init__.py                         顶层稳定 API
├── agents.py                          exactly-once naive/oracle 基线
├── demo.py                            旧版微环境演示命令
├── run_task.py                        oracle/manual/replay 统一 CLI
├── scenario_agents.py                 四个业务场景的确定性 Oracle
│
├── agent_runtime/
│   ├── __init__.py                    Agent runtime 公共导出
│   ├── model.py                       ModelClient、ModelOutput、脚本/函数 adapter
│   └── runner.py                      descriptor 加载、模型循环、消息与轨迹管理
│
├── agent_system/
│   ├── __init__.py                    配置与上下文公共导出
│   ├── config.py                      Agent/Provider JSON 配置与参数覆盖
│   ├── prompts.py                     system prompt 与 skill JSON 加载
│   ├── context.py                     多轮 tool-call/result 上下文与裁剪
│   ├── agent.py                       两类 Agent 的统一资源/provider 装配
│   ├── target.py                      执行环境任务的 Target Agent
│   ├── diagnostic.py                  只读失败归因 Diagnostic Agent
│   └── providers/
│       ├── __init__.py                provider 公共导出
│       ├── api.py                     OpenAI-compatible HTTP 调用
│       ├── adk.py                     module:function ADK executor 桥接
│       └── factory.py                 根据统一 args 选择 provider
│
├── core/
│   ├── __init__.py                    核心类型公共导出
│   ├── protocol.py                    Action、EnvResponse、EnvDescriptor、ActionableEnv
│   ├── tooling.py                     tool schema、contract hash、action 参数校验
│   ├── registry.py                    可插拔组件注册表
│   └── verifier.py                    只读 StateVerifier 协议
│
├── environments/
│   ├── __init__.py                    业务环境公共导出
│   ├── base.py                        状态、descriptor、路由、审计、结束与快照骨架
│   ├── order_lifecycle.py             订单认证、查询、修改、取消和退款
│   ├── issue_workflow.py              issue 搜索、分配、评论和状态推进
│   ├── calendar_email.py              联系人、忙闲时间、会议和邮件
│   └── code_repair.py                 mini repository 搜索、编辑、测试和提交
│
├── micro_api/
│   ├── __init__.py                    微环境公共导出
│   └── item_env.py                    exactly-once 状态写入基础环境
│
├── code_tasks/
│   ├── __init__.py                    代码任务资源公共导出
│   └── qdp_case.py                    QDP mini-repo 与独立测试进程
│
├── transforms/
│   ├── __init__.py                    所有规则协议和实现的公共导出
│   ├── protocols.py                   六类环境变化协议
│   ├── setup.py                       reset 后初始状态动作重放
│   ├── contract.py                    Agent 可见 schema 与同步执行约束
│   ├── action.py                      隐藏的执行前 f_A guard
│   ├── transition.py                  提交后 timeout 等 f_T 规则
│   ├── observation.py                 stale read 等 f_O 规则
│   └── budget.py                      step/write 资源限制与计数
│
├── harness/
│   ├── __init__.py                    Harness 公共入口
│   ├── factory.py                     配置解析、注册表与组件装配
│   ├── wrapper.py                     规则顺序、动态契约、快照和状态隔离
│   └── rules.py                       旧规则导入路径兼容层
│
├── benchmarks/
│   ├── __init__.py                    benchmark adapter 公共导出
│   └── adapter.py                     外部后端到 ActionableEnv 的统一桥
│
├── evolution/
│   ├── __init__.py                    环境演化数据结构公共导出
│   ├── surface.py                     adapter 声明的可变表面
│   ├── mutation.py                    六类 MutationSpec 与稳定摘要
│   ├── failure.py                     确定性失败签名与诊断回退
│   └── lineage.py                     可去重、多父节点环境 DAG
│
└── verifiers/
    ├── __init__.py                    verifier 公共导出
    ├── exactly_once.py                恰好一次写入目标
    ├── order_goal.py                  订单取消、退款及旁路保护
    ├── issue_goal.py                  issue 多步骤最终状态
    ├── calendar_email_goal.py         唯一会议与唯一邮件
    └── code_repair_goal.py            独立重跑测试与修改范围检查

tests/
├── test_agent_runtime.py              descriptor/contract/model runner/隐私边界
├── test_agent_system.py               JSON 配置、多轮上下文与两类 Agent
├── test_architecture.py               注册表、规则和场景清单
├── test_evolution.py                  六类变化、DAG 与 benchmark adapter
├── test_minimal_env.py                timeout/stale/idempotency/snapshot
└── test_scenarios.py                  五个场景的端到端行为与 Oracle
```

每个 Python 文件开头都有中文模块 docstring，说明该文件在架构中的设计职责。新增文件也必须遵循这一规则；测试会扫描并验证这一约束。

## Agent、环境和工具如何解耦

`ActionableEnv` 是 benchmark adapter 的统一边界：

```python
descriptor = env.describe()       # task + 当前允许的 tools
reset = env.reset(seed=0)         # 确定性初始状态
response = env.step(action)       # 标准 Action → 标准 EnvResponse
result = env.evaluate()           # 独立读取真实状态
snapshot = env.save_state()       # 环境和规则可回放
```

不同环境可以有完全不同的工具名和参数；只统一 action/response envelope，不强迫订单、浏览器和代码环境共享业务 action。接真实 benchmark 时实现 `BenchmarkBackend`，再由 `BenchmarkAdapter` 把标准 `Action` 转为原生 API/浏览器/终端调用，并明确公布 `MutationSurface`。

模型 provider 也通过 `ModelClient` 解耦：

```python
def generate(messages, tools) -> ModelOutput:
    ...
```

OpenAI-compatible API、ADK、本地模型或 replay 只需把返回值转换为 `ModelOutput`。API 与 ADK 分别位于独立文件，并由 provider factory 根据同一个配置入口选择。`AgentRunner` 永远只把 `EnvResponse.observation` 送入 Target；`response.info`、`get_env_state()`、verifier 和 hidden tests 不进入其上下文。

## Agent JSON 配置

`configs/agents/` 提供 Target 与 Diagnostic 示例，`configs/prompts/` 和 `configs/skills/` 保存可版本化资源。Agent 类没有硬编码 system prompt 或 skill；路径相对 Agent 配置文件解析。provider 的 `type`、`model`、`base_url`、`executor` 与任意 `args` 使用同一结构，运行时只在 provider factory 分派。

Target 维护完整的多轮工具上下文；Diagnostic 不执行工具，只读取 episode 轨迹和 verifier 结果，返回 failure signature 与候选 `MutationSpec`。环境搜索仍需通过 adapter 的 `MutationSurface` 和 Harness 注册表才能执行候选变化。

## Current research deliverables

- `research/env_evolution_report.tex`：基于 EnvHarness 的环境演化路线、相关工作、搜索策略与最小落地方案。
- `research/env_evolution_method.tex`：完整定义失败条件驱动、保持 verifier 不变的最小辅助搜索，以及环境 DAG、Learner 和迁移评估方法。
- `research/minimal_env_plan.md`：可直接进入实现的里程碑、接口、首批环境变体与验收条件。
- `outputs/env-evolution-research/benchmark_landscape.xlsx`：Agent harness、Agent RSI、通用 Agent 与环境演化工作的 benchmark 对照、例子、优先级、评分和来源。
- `benchmark.md`：从上述对照中筛选公开且适合 Env–Agent co-evolve 的 benchmark，给出下载入口、bridge 适配方式和分阶段接入顺序。

## Scenario catalog

首批实验按独立任务族组织在 [`scenarios/`](scenarios/README.md)：

1. exactly-once 状态写入；
2. 订单取消与修改；
3. 多步骤工单流转；
4. 日历预约与邮件通知；
5. repository-level 代码修复。

每个目录都说明任务、来源、使用过相关问题的工作、相关 benchmark、动作空间、verifier 和从辅助环境回到目标环境的课程；五个场景现在均包含可执行 `task.json`、本地状态环境、独立 verifier 和通过性 Oracle。

代码分层与扩展规则见 [`docs/architecture.md`](docs/architecture.md)。核心原则是环境、verifier、六类环境变化、Target/Diagnostic、provider 和 curriculum 相互解耦，并通过显式注册表组合。每个源码子模块都有自己的 README，解释边界和扩展方式。

## Starting decision

第一版使用本地、确定性、状态化 API 微环境，任务是“在提交前/提交后故障下恰好写入一次”。先实现 EnvHarness 风格的 Setup/Stage 与 Rules/Contract，验证环境生成、回放、独立 verifier 和 lineage DAG；Link/Chain 与 Agent 自我改写延后。

## Minimal runnable environment

现在仓库内已经有一个可运行的环境侧 MVP。任务是把 `target-item` 恰好写入一次，然后结束 episode。

- 基础环境：`append_item`、`list_items`、`get_item`、`finish` 四个工具。
- `f_T`：`post_commit_timeout` 在写入已经提交后，把成功响应替换为超时。
- `f_O`：`stale_read_after_write` 让写入后的前若干次列表查询看到旧状态。
- `Contract`：`require_argument` 同时把幂等键标为 required，并在真实写入前阻止缺失参数的 action。
- verifier：直接读取真实状态，检查目标值恰好出现一次，并且初始数据没有被修改。
- snapshot：同时保存基础环境和规则内部状态，恢复后故障时序保持一致。
- `Setup`：用合法动作重放构造 episode 初态，且不消耗正式交互预算。
- `Budget`：统一限制总步骤和写工具次数，计数进入快照。

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

运行其余完整场景：

```bash
env-agent-rsi-task scenarios/02_order_lifecycle/task.json --agent oracle
env-agent-rsi-task scenarios/03_issue_workflow/task.json --agent oracle
env-agent-rsi-task scenarios/04_calendar_email/task.json --agent oracle
env-agent-rsi-task scenarios/05_code_repair/task.json --agent oracle
```

把 `--agent oracle` 改为 `--agent manual` 后，命令会输出任务与工具 schema，并通过 JSON Lines 接收外部 Agent 的 tool call。环境只给 Agent 返回 observation；verifier 始终读取未经 `f_O` 变换的真实状态。

用可复放的假模型走完整 `ModelClient → AgentRunner → Env` 链路：

```bash
env-agent-rsi-task \
  scenarios/01_exactly_once_write/task.json \
  --agent replay \
  --replay examples/replays/exactly_once.json
```

用 JSON 配置运行真实 Target Agent，并可在失败后调用 Diagnostic Agent：

```bash
env-agent-rsi-task \
  scenarios/01_exactly_once_write/task.json \
  --agent target \
  --agent-config configs/agents/target_agent.json \
  --diagnostic-config configs/agents/diagnostic_agent.json
```

`OPENAI_API_KEY` 由 API provider 在运行时读取；JSON 中不保存密钥。若使用 ADK，把 provider type 设为 `adk` 并配置 `executor: "your_module:run_agent"`。

代码中接入任意模型 SDK：

```python
from env_agent_rsi import AgentRunner, ModelOutput, build_environment, load_spec
from env_agent_rsi.agent_runtime import CallableModelClient

def provider_adapter(messages, tools):
    native = provider.generate(messages=messages, tools=tools)
    call = native.tool_call
    return ModelOutput(call.name, call.arguments, call_id=call.id)

env = build_environment(load_spec("scenarios/02_order_lifecycle/task.json"))
result = AgentRunner(env, CallableModelClient(provider_adapter)).run(seed=0)
```

`baseline.json` 是无辅助规则的基线，`assistive_idempotency.json` 提供写入前安全护栏，`postcommit_stale.json` 是当前目标故障组合。通过 JSON 可以组合规则，不需要修改 Agent 或中央工厂。

### Interaction order

每一步严格按以下顺序运行：

```text
reset
  -> Setup rules（Agent 开始前）
Action
  -> Contract rules（Agent 可见约束）
  -> current EnvDescriptor schema validation
  -> f_A action rules
  -> Budget pre-check
  -> base environment transition
  -> f_T transition rules
  -> f_O observation rules
  -> Budget accounting
  -> agent-visible observation
```

其中 `EnvResponse.observation` 提供给 Agent，`EnvResponse.info` 只供 harness 记录故障触发与调试信息；最终 verifier 始终读取未经 `f_O` 修改的真实状态。
