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
