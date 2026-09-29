# Configuration Catalog

配置按职责分开：`agents/` 选择角色与 provider，`prompts/` 保存 system prompt，`skills/` 保存可复用策略，`environment_changes/` 展示六阶段变化，`micro_api/` 保存最小环境基线与故障组合。

所有配置均为 JSON；路径类字段相对其所属 Agent 配置解析。密钥、token 和 verifier 私有状态不得写入该目录。
