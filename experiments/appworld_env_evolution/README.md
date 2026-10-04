# AppWorld 单步环境进化实验

这个实验固定 Target Agent，按同一流程执行三个 AppWorld Train 任务：

1. 本地 Qwen3 4B 在原始环境运行一次，并由官方 evaluator 评分；
2. DeepSeek Diagnostic Agent 读取失败轨迹并给出唯一最高优先级根因；
3. DeepSeek Modifier 把诊断转换成一个通过 mutation surface 白名单校验的环境变化；
4. 在相同任务、seed、Target 配置上重新运行，并比较改动前后的官方得分。

默认任务是先前已经稳定失败的 `e85d92a_1`、`e85d92a_2`、`e85d92a_3`。
Target 使用 `configs/agents/target_qwen3_4b.json`，环境侧两个 Agent 使用各自的
DeepSeek API 配置。完整轨迹只写入 Git 忽略的 `artifacts/appworld_train/`，避免提交
AppWorld 受保护任务内容；摘要不保存任务文本、答案或工具返回正文。

```bash
PYTHONPATH=src ../.venv/bin/python \
  experiments/appworld_env_evolution/run_three_tasks.py
```

如果诊断判断失败不可由当前环境白名单修复，实验会明确记录 `modifier_skipped`，不会
伪造改动或绕过 schema 校验。

## 1024 与 8192 生成上限对照

以下是同一本地 Qwen3 4B 权重、seed 0 和三个任务的一次本机对照；表格只记录官方
evaluator 聚合结果，不包含任务答案或工具返回正文。

| 任务 | 1024 baseline | 8192 baseline | 8192 candidate | 8192 有效步骤（base → candidate） |
| --- | --- | --- | --- | --- |
| `e85d92a_1` | 1/2 | **2/2** | baseline 成功，不修改 | 6 → — |
| `e85d92a_2` | 1/2 | 1/2 | 1/2 | 5 → 12 |
| `e85d92a_3` | 1/2 | 1/2 | 1/2 | 12 → 5 |

8192 让第一个任务从截断失败变为成功，也让另外两个任务形成更完整的轨迹，但没有让
它们通过。三任务 8192 实验耗时约 72.5 分钟，1024 实验约 5.8 分钟，在本机上约慢
12.5 倍。因此 8192 适合复核“无动作/输出截断”失败，不宜在没有 episode 总 token
预算的情况下直接作为大规模 rollout 的无条件默认值。
