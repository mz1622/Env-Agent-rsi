# Env–Agent Co-evolve benchmark 接入清单

更新日期：2026-09-28

本文从 [`benchmark_landscape.xlsx`](outputs/env-evolution-research/benchmark_landscape.xlsx) 中筛选公开提供任务数据、环境代码或可执行 evaluator，并且适合测试本项目环境侧演化的 benchmark。优先级表示接入顺序，不表示 benchmark 的学术质量。

## 1. 筛选标准

只有满足下列大部分条件的 benchmark 才进入主清单：

1. **任务可取得**：任务定义、数据、初始状态或生成器可以公开下载。需要免费申请或接受使用条款的数据会标记为“公开申请”。
2. **环境可执行**：Agent 的 action 会进入真实或模拟环境，而不是只对静态题目生成一个答案。
3. **可重置或可重建**：同一个 task/seed 可以恢复到相同初始状态。
4. **有可信 verifier**：优先使用数据库终态、测试、目标谓词或程序化 evaluator。LLM-as-a-judge 不能作为首批主指标。
5. **适合包装而不是篡改**：可以在环境边界实现 `Setup / Contract / f_A / f_T / f_O / Budget`，同时保持原始任务语义和 verifier 不变。
6. **能够区分环境改进与 Agent 改进**：同一个冻结 Agent 可以在原环境、辅助环境和撤去辅助后的目标环境上重复运行。

不把“有公开代码”等同于“适合接入”。AgentBoard 这类评测层、LiveCodeBench 这类近似无状态题集，以及 POET/Procgen 这类控制领域环境生成器，都有研究价值，但不应占用第一阶段的 benchmark bridge 工作量。

## 2. 推荐顺序

| 优先级 | Benchmark | 数据状态 | 原生 verifier | 适合改变的层 | 建议用途 |
| --- | --- | --- | --- | --- | --- |
| P0-smoke | EnvHarness toy24 | 公开代码内置 | 确定性目标谓词 | Setup、Contract、`f_A/f_T/f_O` | 验证我们的 `ActionableEnv` 兼容性，不作为主实验结论 |
| P0 | AppWorld | 公开下载，部分内容以 bundle 发布 | 数据库状态单测，检查 collateral damage | Setup、Contract、`f_A/f_T/f_O`、Budget | 第一外部主 benchmark；多应用、强状态、强 verifier |
| P0 | τ³-bench | 公开代码与任务 | action/数据库状态和任务 criteria | Contract、`f_A/f_T/f_O`、Budget | 订单、客服政策、重复调用和提交后不确定性 |
| P1 | ALFWorld | 公开下载 | 环境目标谓词 | Setup、Contract、`f_A/f_T/f_O` | 低成本验证跨域 Stage/Rule 泛化 |
| P1 | WebShop | 公开下载 | 商品属性匹配和成功奖励 | Setup、Contract、`f_A/f_T/f_O`、Budget | 稳定的模拟网页环境，适合导航与观察变化 |
| P1 | SWE-bench Verified | Hugging Face 公开 | FAIL_TO_PASS / PASS_TO_PASS 测试 | Setup、Contract、`f_A/f_T/f_O`、Budget | 代码环境和终端工具边界；运行成本较高 |
| P1 | WebArena-Verified | 公开仓库和 Hugging Face | 确定性 response/network-trace evaluator | Setup、Contract、`f_A/f_T/f_O`、Budget | 长程 Web 环境；优先于原始 WebArena |
| P1 | SpreadsheetBench 2 | Hugging Face 公开 | 工作簿结果 evaluator；可视化类含 VLM evaluator | Setup、Contract、`f_A/f_T/f_O`、Budget | 文件副作用、重算、输出保存；首版排除可视化类 |
| P2 | WorkArena / WorkArena++ | 公开申请，可自动批准或等待审批 | 程序化任务 evaluator | Setup、Contract、`f_A/f_T/f_O`、Budget | 企业流程和组合任务；ServiceNow 运行时较重 |
| P2 | AndroidWorld | 公开代码和动态任务生成器 | setup/success/teardown 程序 | Setup、Contract、`f_A/f_T/f_O`、Budget | 参数化移动端任务；适合测试环境泛化 |
| P2 | Terminal-Bench 2.0 | Harbor 公开下载 | 容器内测试脚本 | Contract、`f_A/f_T/f_O`、Budget | 异构终端任务；容器构建和快照成本高 |
| P2 | OSWorld 2.1 | 公开申请，任务和资产分版本发布 | 任务级执行式 evaluator | Setup、Contract、`f_A/f_T/f_O`、Budget | 真实桌面长程任务；最后接入 |
| 对照 | OfficeQA | 公开申请，含题目与语料 | 官方 reward 脚本 | Contract、`f_O`、Budget | 观察/检索契约对照，不代表完整状态化环境 |

## 3. 与本仓库的统一适配协议

每个 benchmark 只实现一个 bridge，不修改 `AgentRunner`、搜索器或 verifier 逻辑。适配器对齐现有 `ActionableEnv`：

```text
公开 task record / native environment
                 │
                 ▼
BenchmarkAdapter.describe()
  EnvDescriptor(task_id, instruction, tools, contract_version)
                 │
                 ▼
AgentRunner → Action(tool, arguments)
                 │
Contract → schema validation → f_A
                 │
                 ▼
BenchmarkAdapter.step() → native action
                 │
native transition → f_T → f_O
                 │
                 ▼
EnvResponse(observation, terminated, truncated, info)
                 │
                 ▼
BenchmarkAdapter.evaluate() → original evaluator/verifier
```

### 3.1 Bridge 必须完成的映射

| 本仓库接口 | benchmark adapter 的职责 |
| --- | --- |
| `describe()` | 从任务记录生成 instruction；把原生 action space 转成 JSON tool schemas；不得暴露答案、隐藏测试、真实数据库或 evaluator 细节 |
| `reset(seed, options)` | 选择固定 task ID 和 seed；创建或恢复原生初始状态；返回只对 Agent 可见的初始 observation |
| `step(Action)` | 把统一 tool call 转成原生 API、文本动作、浏览器动作、shell 命令或 GUI action；原生结果规范化为 `EnvResponse` |
| `observe()` | 返回当前 Agent 可见状态，不能绕过 `f_O` 读取真实状态 |
| `evaluate()` | 委托 benchmark 原始 evaluator；不得由 mutation 修改成功条件 |
| `get_env_state()` | 只供 verifier、诊断和状态哈希使用，不进入模型上下文 |
| `save_state()/load_state()` | 优先保存原生 checkpoint；否则保存 task ID、seed、动作日志和可确定性重建所需版本信息 |

### 3.2 统一数据目录建议

公开数据不要提交进 Git。adapter 配置只记录来源和版本：

```text
external_data/                         # 加入 .gitignore
  appworld/
  tau3/
  alfworld/
  swebench/
  webarena_verified/

src/env_agent_rsi/benchmarks/
  <benchmark>/
    adapter.py                         # ActionableEnv 实现
    loader.py                          # task/dataset 加载与版本检查
    action_codec.py                    # Action 与原生 action 的双向转换
    state_codec.py                     # snapshot/rebuild 逻辑

configs/benchmarks/<benchmark>/
  baseline.json
  smoke.json
  mutations/
```

如果后续新增 Python 文件，文件头部继续使用中文模块 docstring，说明该文件在 bridge 中的职责。

### 3.3 任何 benchmark 都不能改变的内容

- task 的语义目标；
- 原始 verifier/evaluator 的代码和哈希；
- hidden tests、golden answer 或特权数据库状态对 Agent 的可见性；
- 同一对比实验中的模型、system prompt 和 skills；
- benchmark 数据版本和容器/镜像版本。

允许变化的只是环境边界：初始状态重放、工具 schema、action guard、转移故障、观察格式、预算和可验证的环境组合。

## 4. P0：立即接入

### 4.1 EnvHarness toy24：协议 smoke test

**来源与下载**

- 代码与说明：[google-research/envharness](https://github.com/google-research/envharness)
- toy24 运行说明：[experiments/toy24/README.md](https://github.com/google-research/envharness/blob/main/experiments/toy24/README.md)

```bash
git clone https://github.com/google-research/envharness.git
cd envharness
pip install -e .
python scripts/check_env.py toy24
```

**为什么接入**

toy24 是纯内存环境，没有 Docker、浏览器或外部数据依赖。它适合验证：

- `describe/reset/step/evaluate/save_state/load_state` 的语义是否与 EnvHarness 一致；
- wrapper 顺序是否确实是 `Contract → f_A → base → f_T → f_O`；
- snapshot 恢复后 mutation 的内部计数器是否一致；
- 原生目标谓词是否保持不变。

**适配方式**

- `describe()` 暴露一个 `apply_operation(expression)` 或原生动作工具。
- `reset()` 绑定固定数字集合和 task ID。
- `step()` 将 JSON action 编码为 toy24 原生 action。
- `evaluate()` 直接调用原生 24 点目标判断。
- 先运行官方 `tests/test_actionable_env_toy24.py`，再让同一组测试跑我们的 adapter。

它是桥接测试，不是主实验，因为它不覆盖持久化副作用、订单政策或真实工具失败。

### 4.2 AppWorld：第一外部主 benchmark

**来源与下载**

- 代码、数据安装和 evaluator：[StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld)
- 论文：[AppWorld](https://arxiv.org/abs/2407.18901)

```bash
git lfs install
git clone https://github.com/StonyBrookNLP/appworld.git
cd appworld
pip install -e .
git lfs pull
appworld install --repo
appworld download data
```

AppWorld 的 app 实现、任务和数据部分以 bundle 发布。可以本地解包使用，但不要把解包后的 benchmark 内容重新提交到本仓库。首轮只使用 `train` 和 `dev`，把 `test_normal`、`test_challenge` 留作最终评估。

**适配方式**

- 首版采用 function-calling 路线，不开放任意 Python code execution，降低 action codec 和沙箱变量。
- `describe()` 把当前任务 instruction、可用 app APIs 和参数 schema 转成 `EnvDescriptor`。
- `reset()` 用固定 `task_id` 创建 `AppWorld`，让 AppWorld 自己恢复任务数据库和时间。
- `step()` 将 tool call 映射到 app API；API 返回值进入 `observation`，调用日志和故障标签进入 `info`。
- `evaluate()` 调用 AppWorld 的数据库状态单测，同时保留 collateral-damage 检查。
- snapshot 优先保存 AppWorld 任务数据库/输出状态；若原生 checkpoint 不完整，则保存 `task_id + dataset version + action replay` 并做状态哈希校验。

**首批环境 mutation**

1. Contract：必填稳定资源 ID、显式确认破坏性操作、限制跨 app 工具集合。
2. `f_A`：阻止缺少幂等键或确认 token 的写操作。
3. `f_T`：API 已提交但返回 timeout；跨 app 第二步失败；部分事务成功。
4. `f_O`：一次 stale read、分页截断、错误消息保留稳定 error code 但隐藏实现细节。
5. Budget：限制总 API 次数、写操作次数或跨 app 切换次数。

**首轮子集**

从 `train/dev` 中选 20–50 个具备写副作用、存在读后写或跨应用依赖的任务。排除只需一次只读查询的任务。每个 task 必须先通过原环境 Oracle、adapter Oracle 和 snapshot round-trip 三道门禁。

### 4.3 τ³-bench：订单与政策约束

**来源与下载**

- 当前代码和任务：[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- 原始 τ-bench 论文：[τ-bench](https://arxiv.org/abs/2406.12045)

```bash
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
uv sync
```

仓库名称仍是 `tau2-bench`，但其当前发布已经是 τ³-bench。旧的 [`sierra-research/tau-bench`](https://github.com/sierra-research/tau-bench) 明确标记为过时，不应用于新实验。实验必须固定 release/tag；若使用 v1.0.1 及之后的任务修复版，结果不能与更早版本直接比较。

**适配方式**

- 第一阶段使用 `base` split 和文本 half-duplex 模式，不接 voice。
- 从 `mock` 域做 smoke，再接 `retail`；`airline`、`telecom` 和 knowledge 域后移。
- `describe()` 暴露用户请求、政策摘要和 domain tools，但不暴露 expected actions。
- `reset()` 加载固定 domain/task/seed 和数据库状态。
- `step()` 同时支持 Agent tool call 与对话消息。为了降低 user simulator 方差，主实验使用固定 simulator model、temperature 和 seed，并保存完整 user trajectory；另加 scripted/replay user 对照。
- `evaluate()` 委托原生 task criteria 和 reward 计算。
- snapshot 保存数据库、对话状态、user simulator 状态和 mutation counter。

**首批环境 mutation**

- Contract：把订单 ID、确认标志和政策前置条件显式放入 schema。
- `f_A`：在不满足政策或缺少确认时阻止破坏性 action。
- `f_T`：取消/退款已经提交后 timeout，或写成功但后续通知失败。
- `f_O`：订单读取短暂陈旧、工具错误被截断、分页隐藏目标订单。
- Budget：一次破坏性写、有限澄清轮次、有限数据库读取。

这里最适合复用仓库现有的订单生命周期场景：先让相同 mutation 在本地 `scenarios/02_order_lifecycle` 通过，再迁移到 τ³-bench retail adapter。

## 5. P1：运行隔离稳定后接入

### 5.1 ALFWorld

**下载**：[alfworld/alfworld](https://github.com/alfworld/alfworld)

```bash
git clone https://github.com/alfworld/alfworld.git
cd alfworld
pip install -e '.[full]'
export ALFWORLD_DATA=/absolute/path/to/alfworld-data
python scripts/alfworld-download
```

**适配**：将文本动作包装成单一 `act(command)` 工具；`reset()` 固定 game/task seed；`evaluate()` 保留原生目标谓词。Setup 可以在 reset 后重放一段合法动作，把 Agent 放到较接近目标的状态。Contract 和 `f_A` 可以限制或规范动作格式，`f_O` 可以控制可见描述，`f_T` 可以注入已执行但响应丢失等界面故障。

**首轮范围**：只跑 TextWorld 模式，暂不接 THOR 视觉环境。它主要验证环境层的跨域性，不应成为订单/API 副作用结论的主要证据。

### 5.2 WebShop

**下载**：[princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop)

```bash
git clone https://github.com/princeton-nlp/WebShop.git webshop
cd webshop
./setup.sh -d small
```

`-d small` 下载 1,000 个商品用于适配和 smoke；正式实验再使用 `-d all`。仓库提供 1.18M 商品和 12,087 条众包 instruction。

**适配**：将原生 action 拆成 `search(query)`、`click(target)`、`back()`、`buy(options)` 等 schema，或先用单一 `browser_act(command)` 保持原生语义。`reset()` 选择固定 instruction；`evaluate()` 保留商品属性匹配奖励。Setup 用动作重放到中间页面，`f_O` 用于属性隐藏、分页和 DOM/文本视图变化，`f_T` 用于点击/购买后的响应丢失，Budget 用于页面步数和购买次数。

### 5.3 SWE-bench Verified

**来源与下载**

- 代码和 evaluation harness：[SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench)
- 数据：[SWE-bench/SWE-bench_Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)

```python
from datasets import load_dataset

tasks = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
```

**适配方式**

- `reset()` 按 `instance_id` 从 `base_commit` 重建容器和工作树。
- `describe()` 提供 issue 描述和 `shell/read_file/write_file/run_tests/finish` 等工具 schema。
- `step()` 将工具调用映射到隔离容器，stdout/stderr 成为 observation。
- `evaluate()` 只调用官方 FAIL_TO_PASS / PASS_TO_PASS harness，hidden tests 永不进入 Agent 上下文。
- snapshot 不强求保存整个容器内存；保存固定镜像 digest、task record、git diff、未跟踪文件清单、动作日志和规则状态，并验证重建后的工作树哈希。

**适合的 mutation**：shell contract、允许命令列表、命令 timeout、命令执行后输出丢失、stdout 截断、测试预算、文件编辑范围。不能修改测试、gold patch、issue 文本的任务语义或依赖版本。

**首轮范围**：先选 10–20 个 Python、单仓库、构建时间短的 Verified task；每个候选 mutation 最多跑 3–5 个 task 做内环，完整子集只用于已通过门禁的节点。

### 5.4 WebArena-Verified

**来源与下载**

- 代码、数据和离线 evaluator：[ServiceNow/webarena-verified](https://github.com/ServiceNow/webarena-verified)
- Hugging Face 数据：[AmineHA/WebArena-Verified](https://huggingface.co/datasets/AmineHA/WebArena-Verified)
- 统一浏览器接口：[ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym)

```bash
pip install webarena-verified browsergym-webarena-verified
webarena-verified dataset-get --output webarena-verified.json
webarena-verified subset-export --name webarena-verified-hard --output webarena-verified-hard.json
```

WebArena-Verified 提供 812 个审计后的任务和 258 个 hard 子集，并用 agent response 和网络 trace 做确定性离线评分。它比原始 WebArena 更适合作为本项目的默认 Web 评测版本。

**适配方式**

- 直接在 BrowserGym 上实现通用 `BrowserGymAdapter`，后续 WorkArena 复用。
- `describe()` 暴露 task goal 与浏览器 action schema。
- `reset()` 传入固定 task ID，并调用站点 full reset 或从固定镜像重建。
- `step()` 将 click/type/scroll/navigation action 交给 BrowserGym；保存 HAR/network trace。
- `evaluate()` 委托 WebArena-Verified 离线 evaluator。
- `f_O` 可过滤 accessibility tree、截图或可见 DOM；`f_T` 可注入请求失败和提交后响应丢失；Contract/`f_A` 可改变 action grammar 与危险操作确认。

首版只选择无需外部账号、能够完全自托管和确定性 reset 的任务。

### 5.5 SpreadsheetBench 2

**来源与下载**

- 代码和 evaluator：[RUCKBReasoning/SpreadsheetBench-2](https://github.com/RUCKBReasoning/SpreadsheetBench-2)
- 数据：[KAKA22/SpreadsheetBench-v2](https://huggingface.co/datasets/KAKA22/SpreadsheetBench-v2)

```bash
git clone https://github.com/RUCKBReasoning/SpreadsheetBench-2.git
huggingface-cli download KAKA22/SpreadsheetBench-v2 \
  --repo-type dataset --local-dir data/spreadsheetbench-v2
```

**适配方式**

- 每个 episode 使用输入工作簿的 copy-on-write 副本，`reset()` 删除输出并重新复制原文件。
- `describe()` 提供任务 instruction 和工作簿工具，例如 `list_sheets/read_range/write_range/set_formula/recalculate/save_output`。
- `step()` 调用隔离的 spreadsheet backend；文件变化摘要进入 `info`，只把工具返回和可见单元格交给 Agent。
- `evaluate()` 调用官方结果 evaluator。首版只用 `Debugging`、`Financial_Model`、`Template`，因为 `Visualization` 当前依赖 VLM checklist 和 Windows Excel/WPS COM，难以作为确定性内环 verifier。
- `save_state()` 保存输入文件哈希、当前工作副本、已保存输出、backend 版本和 mutation state。

**适合的 mutation**：公式/值工具契约、只读与可写范围、重算时机、保存失败、写成功但响应 timeout、读取缓存陈旧、行列截断和工具预算。不能改变 golden workbook 或 evaluator。

## 6. P2：重型环境

### 6.1 WorkArena / WorkArena++

**来源与下载**

- 代码：[ServiceNow/WorkArena](https://github.com/ServiceNow/WorkArena)
- BrowserGym：[ServiceNow/BrowserGym](https://github.com/ServiceNow/BrowserGym)
- 实例数据（公开申请）：[ServiceNow/WorkArena-Instances](https://huggingface.co/datasets/ServiceNow/WorkArena-Instances)

```bash
pip install browsergym-workarena
playwright install chromium
```

当前版本需要取得 ServiceNow instance 数据或配置开发实例。先使用 L1 原子任务验证 `BrowserGymAdapter`，再进入 WorkArena++ 组合任务。复用 WebArena 的 action codec、observation codec 和 trace schema，但由 WorkArena 原生 evaluator 判分，两个 benchmark 的 verifier 不能混用。

适合研究企业表单、知识库、订单和审批中的 Contract、确认、状态恢复和长程预算；不适合在 MVP 阶段承担高频候选搜索。

### 6.2 AndroidWorld

**下载**：[google-research/android_world](https://github.com/google-research/android_world)

```bash
git clone https://github.com/google-research/android_world.git
cd android_world
pip install -e .
```

AndroidWorld 提供动态参数化任务、setup/success/teardown 和可重复 emulator 环境。adapter 将 touch、type、back、home 等 action 编码成 JSON tools；`reset()` 调用任务 setup 并记录参数 seed；`evaluate()` 调用原生 success；teardown 保证跨 episode 隔离。

初始只选 Contacts、Clock、Simple Calendar 等不依赖外部账号的任务。Apple Silicon 上的嵌套 emulator 会显著变慢，运行节点应优先使用原生 Android emulator 或固定 Linux worker。

### 6.3 Terminal-Bench 2.0

**来源与下载**

- 官方执行框架：[Harbor](https://github.com/harbor-framework/harbor)
- Terminal-Bench 2 数据入口：[Harbor Hub](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2/latest)
- 数据仓库：[laude-institute/terminal-bench-datasets](https://github.com/laude-institute/terminal-bench-datasets)

```bash
uv tool install harbor
harbor run -d terminal-bench/terminal-bench-2 -a oracle
```

Harbor 会下载任务、构建隔离环境并运行测试。adapter 最好位于 Harbor agent 边界，而不是复制任务：`describe()` 提供 instruction 和 `terminal.exec`；`step()` 执行命令并返回 stdout/stderr；`evaluate()` 只读取 Harbor verifier 产物。

适合 Contract、命令 allowlist、timeout、命令已执行但输出丢失、stdout 截断、磁盘/时间预算。由于任务高度异构，首轮只选 5–10 个 verifier 快、无网络和无 GPU 的任务，不对整个数据集自动生成同一种 mutation。

### 6.4 OSWorld 2.1

**来源与下载**

- 当前代码和版本说明：[xlang-ai/OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2)
- 任务（公开申请）：[xlangai/osworld_v2_tasks](https://huggingface.co/datasets/xlangai/osworld_v2_tasks)
- 完整资产（公开申请）：[xlangai/osworld_v2_assets_gated](https://huggingface.co/datasets/xlangai/osworld_v2_assets_gated)

```bash
git clone --branch osworld-v2.1 https://github.com/xlang-ai/OSWorld-V2.git
cd OSWorld-V2
uv sync --frozen
uv run scripts/tools/download_osworld_v2_tasks.py --benchmark-release osworld-v2.1
uv run scripts/tools/download_osworld_v2_assets.py \
  --benchmark-release osworld-v2.1 \
  --target-dir cache/osworld_v2_assets_v2.1
```

必须同时固定代码、task、asset、mocked website 和 VM/provider image 的同一 release；不能把 `main`、`latest` 或不同版本的组件混在一次实验里。

adapter 将 screenshot/accessibility observation 和 click/type/hotkey 等 action 转成统一协议；`reset()` 恢复固定 VM snapshot 并执行 task setup；`evaluate()` 调用原生任务 evaluator。初期只选择离线、无 OAuth、无真实账号的任务。OSWorld 用于最后验证环境演化是否能跨 GUI 和多应用保持效果，不进入早期搜索内环。

## 7. 观察契约对照：OfficeQA

**来源与下载**

- 代码和 reward：[databricks/officeqa](https://github.com/databricks/officeqa)
- Pro / Full 数据（公开申请）：[databricks/officeqa](https://huggingface.co/datasets/databricks/officeqa)
- Pro V2 数据（公开申请）：[databricks/officeqa-pro-v2](https://huggingface.co/datasets/databricks/officeqa-pro-v2)

```python
from datasets import load_dataset

questions = load_dataset(
    "databricks/officeqa",
    data_files="officeqa_pro.csv",
    split="train",
)
```

OfficeQA 本身是静态问答，不具备 AppWorld 那样的可写世界状态。要用于本项目，需要显式构造一个 retrieval environment：

- `search(query, filters)`；
- `open_document(document_id)`；
- `read_page(document_id, page)`；
- `extract_table(document_id, page, region)`；
- `submit_answer(answer, citations)`。

原始问题、语料和 reward 脚本保持不变，只改变检索/观察 contract、上下文截断和工具预算。因此 OfficeQA 只能回答“环境接口是否帮助检索和观察”，不能单独证明我们的状态转移、恢复或副作用处理有效。

## 8. 暂不接入的公开 benchmark

| Benchmark | 暂缓原因 | 后续用途 |
| --- | --- | --- |
| AgentBench | 八类子环境接口、reset 和 verifier 不统一 | 外部 benchmark bridge 稳定后的泛化评测 |
| AgentBoard | 主要是评测与 progress-rate 分析层，不是环境底座 | 借鉴中间进度指标 |
| 原始 WebArena | 部分 evaluator 含模糊匹配或 LLM 判断，站点部署重 | 默认改用 WebArena-Verified；必要时再做结果对照 |
| SkillsBench | 主要测 skill 带来的增益，环境不是主变量 | 用于评估从成功轨迹归纳出的 skill，而不是生成环境 |
| LiveCodeBench | 近似无状态代码提交，action space 和环境状态都很弱 | 只做污染控制或代码能力回归 |
| Procgen / PAIRED / ACCEL / POET | 适合验证课程和环境搜索算法，但离工具 Agent 任务语义较远 | 借鉴 regret、局部编辑、archive 和 minimal criterion；不作为外部 Agent benchmark bridge |
| Factorio / Pokémon | 长程但 verifier、重置或搜索成本不适合首版 | 研究持续学习和不重置 episode |
| OMNI-EPIC | 开放式生成环境和 reward，verifier 漂移风险高 | 远期研究，不进入可信 benchmark 主线 |

## 9. 每个 bridge 的验收门禁

一个 benchmark 只有同时通过以下门禁，才允许进入 co-evolve 搜索：

1. 固定版本、task ID 和 seed 连续重置，初始状态哈希一致。
2. 原环境 Oracle 通过；通过 adapter 的 Oracle 也通过。
3. baseline Agent 在原生 runner 与 adapter 无 mutation 模式下结果等价，或差异有明确解释。
4. `save_state → load_state` 后 observation、可见工具、真实状态和 evaluator 结果一致。不能原生快照时，必须证明动作重放可以确定性重建。
5. verifier/evaluator 文件哈希在所有 candidate 中保持不变。
6. mutation 触发时产生命名 `fault_event`，未触发路径不改变原生行为。
7. Agent 只看到 `observation`，不能看到 `info`、hidden tests、gold answer、真实数据库或 expected actions。
8. mutation 后 Oracle 仍可完成任务；无法通过的 candidate 标为 invalid，而不是“更难”。
9. 每个 episode 使用独立工作副本、数据库、容器或 VM，测试结束后不存在跨 episode 状态泄漏。
10. 记录 benchmark commit/tag、dataset revision、镜像 digest、task ID、seed、Agent 配置、mutation spec、verifier hash 和完整 trajectory。

## 10. 推荐的实际实施顺序

### M0：adapter conformance

- 接 EnvHarness toy24。
- 将官方 contract test 和本仓库 `ActionableEnv` test 合并成 benchmark adapter conformance suite。
- 只验证协议、snapshot、状态隔离和 verifier 不变量。

### M1：状态化 API

- 接 AppWorld 的 20–50 个 `train/dev` 写任务。
- 接 τ³-bench `mock` 和 `retail` 小子集。
- 把本仓库 exactly-once、订单、工单、日历邮件场景中的 mutation 迁移到两个外部环境。
- 主要指标：成功率、重复写率、collateral damage、确认读取率、故障触发覆盖和重置方差。

### M2：跨接口泛化

- ALFWorld TextWorld 和 WebShop small。
- 检查从 API 环境学到的恢复策略是否能迁移到文本动作和模拟网页。
- 加随机 observation/action perturbation 作为相同预算的控制组。

### M3：文件和长程任务

- SWE-bench Verified 小子集。
- SpreadsheetBench 2 的三个确定性类别。
- WebArena-Verified 自托管子集。
- 把候选搜索与昂贵完整评估分开：小子集负责筛选，固定 holdout 只评估已入 archive 的节点。

### M4：重型系统

- 复用 BrowserGym bridge 接 WorkArena。
- 接 AndroidWorld、Terminal-Bench 2.0 和 OSWorld 2.1。
- 这些环境只用于验证已经稳定的 mutation family，不在其中从零进行开放式搜索。

## 11. 首个外部实验建议

第一项正式实验采用 AppWorld，第二项采用 τ³-bench retail：

1. 冻结同一个 Agent、模型、prompt、tool-call codec 和预算。
2. 每个 benchmark 选 20 个可写、可确定性验证的任务。
3. 比较原环境、随机 mutation、诊断驱动 mutation、撤去辅助后的目标环境四组。
4. 每个 mutation 先跑 Oracle gate，再跑冻结 Agent。
5. 当辅助节点成功后，从父节点重新测试；父节点已经成功时，不继续优化更容易的子节点。
6. 报告 task success、违规写入、重复副作用、collateral damage、步骤数、故障覆盖、snapshot 成功率和 evaluator 方差。

这样，benchmark 不是用来不断制造更难的题，而是提供有可信目标和真实失败模式的环境。环境搜索只添加当前 Agent 完成原任务所需的最小帮助，并最终验证帮助能否被撤去。
