# Minimal API Configurations

- `baseline.json`：无辅助规则的目标环境。
- `postcommit_stale.json`：写入已提交但响应超时，并让后续列表读取短暂陈旧。
- `assistive_idempotency.json`：在工具契约中公开幂等键要求，同时在执行前阻止不安全写入。

三份配置保持同一任务和 verifier，用来比较环境变化是否真正提高固定 Target Agent 的成功率。
