# Benchmark 适配层

`adapter.py` 定义 `BenchmarkBackend` 最小协议并规范化为 `ActionableEnv`；`appworld/` 是当前
唯一实现。AppWorld 原生任务、工具、checkpoint 和 evaluator 仍由官方包拥有。
