# AppWorld Train 准备实验

- `select_tasks.py` 从官方 Train split 按 difficulty、应用/API 数和解法长度排序，默认保留
  五个不同 generator 的任务；输出只含 task id 与聚合复杂度。
- `smoke.py` 检查官方任务、数据库与 evaluator 能在隔离 Python 3.12 环境中加载。

```bash
python scripts/setup_appworld.py
PYTHONPATH=src python experiments/appworld_train_adapter/select_tasks.py
PYTHONPATH=src python experiments/appworld_train_adapter/smoke.py
env-agent-rsi-agent0-data
```

选择结果与 Agent0 parquet 位于被 Git 忽略的 `artifacts/appworld_train/`。

