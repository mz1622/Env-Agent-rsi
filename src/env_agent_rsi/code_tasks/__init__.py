"""代码修复任务资源的公共导出。

这里集中暴露小型仓库快照与确定性测试函数，环境和 verifier 复用同一测试语义，
但 verifier 会独立重跑而不信任 Agent 可见的测试结果。
"""

from .qdp_case import INITIAL_FILES, run_qdp_tests

__all__ = ["INITIAL_FILES", "run_qdp_tests"]
