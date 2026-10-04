# Env-Agent-RSI

这个仓库现在只保留一个可运行 benchmark：**AppWorld**。Target Agent 的训练与多轮
rollout 复用 [Agent0](https://github.com/aiming-lab/Agent0)；本项目负责 AppWorld 环境、
诊断、环境修改、候选环境评估与版本 DAG。其他 toy、τ³、手写订单/代码任务及其 adapter
已经移除，避免同时维护多套不一致接口。

## 当前完成的最小链路

```text
AppWorld task ──> Agent0 parquet ──> Qwen/Qwen3-4B-Base + Agent0 ADPO
                                             │
                                  Agent0 AsyncToolServer
                                             │
                                  AppWorldAgent0Tool
                                             │
                                  AppWorldProcessBackend
                                             │
                                    官方 evaluator 奖励
```

- `third_party/Agent0`：固定 commit 的官方 submodule；训练器不复制、不 fork。
- `src/env_agent_rsi/benchmarks/appworld`：隔离 AppWorld Python 3.12 worker。
- `src/env_agent_rsi/integrations/agent0`：数据、工具、奖励、训练参数四个薄适配点。
- `src/env_agent_rsi/agent_system`：Diagnostic 与 Environment Modifier。
- `src/env_agent_rsi/transforms`：Setup、Contract、Action、Transition、Observation、Budget。
- `src/env_agent_rsi/evolution`：失败签名、mutation allowlist、配置物化和环境 DAG。
- `src/env_agent_rsi/orchestration`：同 seed 配对评估、轨迹保存和隔离运行。
- `experiments/appworld_env_evolution`：本地 Qwen Target → DeepSeek Diagnose →
  DeepSeek Modify → 相同 Target 重跑的三任务真实闭环。

AppWorld 第一版 mutation surface 只开放已能可靠验证的 Setup、Contract、Action、Budget；
Transition 与 Observation 实现仍保留为通用模块，但不会伪装成 AppWorld 已支持能力。
Contract 目前还支持 `add_tool_guidance`：只向既有工具说明追加不含答案的最小流程提示，
用于验证信息发现类失败能否由环境契约辅助，而不改 Target 权重、任务或 verifier。

## 安装与运行

```bash
python scripts/setup_appworld.py
PYTHONPATH=src python experiments/appworld_train_adapter/select_tasks.py

git submodule update --init --recursive third_party/Agent0
python -m pip install -e '.[data]'
env-agent-rsi-agent0-data
env-agent-rsi-agent0-server
env-agent-rsi-agent0-train          # 先打印可审计命令
env-agent-rsi-agent0-train --execute

# 单步环境进化配对实验（需本地 Ollama qwen3:4b-direct 与 api.txt）
PYTHONPATH=src ../.venv/bin/python \
  experiments/appworld_env_evolution/run_three_tasks.py
```

AppWorld 版本固定在 `configs/benchmarks/appworld_train.json`。默认模型和训练参数位于
`configs/agent0/appworld_minimal.json`。4-train/1-validation 只是接入 smoke split，不是论文
最终训练/测试划分。

## 奖励边界

AppWorld 官方 evaluator 在隔离 worker 中运行。环境使用每条 parquet 记录的随机 key 为
evaluator payload 生成 HMAC；reward 函数仅接受签名正确、task id 一致的结果。模型自行
输出“成功”或伪造 XML 不能得分。

## 测试

```bash
PYTHONPATH=src python -m pytest -q
```

本仓库只收集 `tests/`；Agent0 submodule 自带的 Ray/vLLM/GPU 测试应在官方训练环境中单独
运行。详细说明见 `src/env_agent_rsi/integrations/agent0/README.md`、
`src/env_agent_rsi/benchmarks/appworld/README.md` 和 `docs/architecture.md`。
