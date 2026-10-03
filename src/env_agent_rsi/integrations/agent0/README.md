# Agent0 × AppWorld 最小接入

## 复用边界

Agent0 固定为 Git submodule `third_party/Agent0`。本目录不复制其 curriculum、ADPO、veRL、
rollout manager、异步工具服务器或 checkpoint 代码，只补 Agent0 原本不知道的 AppWorld
边界：

1. `dataset.py` 把官方 AppWorld instruction 变成 `VerlToolRLHFDataset` 已有字段；
2. `tool.py` 继承 Agent0 `BaseTool`，把三种 action 转发给现有 `AppWorldProcessBackend`；
3. `tool_server.py` 在运行时把该类注册到 Agent0 `AsyncToolServer`；
4. `reward.py` 将官方 evaluator 结果变成 Agent0 可加载的规则奖励；
5. `training.py` 只向上游 `verl_tool.trainer.main_ppo` 传 Hydra overrides。

上游自带的 `mcp_interface` 目前没有实现 `conduct_action`，所以不能直接承担 AppWorld
执行；这里选择官方 `BaseTool` 扩展点，而不是另写 rollout 或训练框架。

## 最小运行顺序

```bash
git submodule update --init --recursive third_party/Agent0
python -m pip install -e '.[data]'
env-agent-rsi-agent0-data
env-agent-rsi-agent0-server
env-agent-rsi-agent0-train          # 先只打印可审计命令
env-agent-rsi-agent0-train --execute
```

工具服务和训练命令应在按 Agent0 README 创建、已安装 veRL/vLLM/FastAPI 的训练环境里运行。
本仓库的 AppWorld worker 仍使用独立 `.venv-appworld`，不会把 AppWorld 依赖灌进训练环境。

## 奖励可信边界

AppWorld 官方 evaluator 在 worker 内执行。工具返回的 evaluator payload 使用 parquet
样本中的逐任务随机 key 做 HMAC；reward 函数同时验证签名和 task id。因此模型仅输出
`success=true` 或伪造 XML 标记都得不到奖励。parquet 与 key 位于被 Git 忽略的
`artifacts/appworld_train/agent0/`，不会提交任务正文或 protected ground truth。
