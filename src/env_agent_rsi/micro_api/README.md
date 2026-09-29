# Minimal Stateful API

`ItemEnv` 是最小可运行环境：Agent 需要把目标值恰好写入一次。四个工具覆盖分页读、按 ID 读、带可选幂等键的写入和终止。

该环境刻意保持小而确定，适合验证提交后超时、陈旧观察、幂等重试、Setup、Budget、快照和独立 verifier，再把相同协议迁移到真实 benchmark。
