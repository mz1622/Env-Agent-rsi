# 实验目录

当前包含 `appworld_train_adapter/`（固定任务选择与官方 worker 验证）和
`appworld_env_evolution/`（Target → Diagnose → Modify → Target 的三任务配对实验）。
Agent0 数据和训练入口位于可安装包中，避免在实验目录复制 rollout/training 实现。
