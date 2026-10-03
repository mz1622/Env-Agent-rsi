# Orchestration

这个模块负责候选环境的运行基础设施，不负责让 LLM 决定候选是否接受。

- `rollouts.py`：让 baseline 和 candidate 使用完全相同的 seed 集执行 K 次。
- `storage.py`：把包含消息、动作、观察和 verifier 结果的完整 episode 追加到 JSONL。
- `isolation.py`：在一次性子进程中运行候选；崩溃和超时不会终止主进化流程。

推荐链路是：先编译/物化声明式候选，再经子进程执行，随后把成对结果交给独立的固定
acceptance gate。子进程隔离只处理故障传播，不提供操作系统级文件、网络或秘密权限沙箱。
