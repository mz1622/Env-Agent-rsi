# Memory Configurations

这里保存 Agent 侧、版本化、可检索的 JSON 长期记忆。当前 `empty_target_memory.json` 只固定未来接口，不包含任务经验；Agent0 最小基线暂不把 memory 注入训练轨迹。

Memory 与 Skill 的区别：Skill 是经过验证并作为可信系统策略加载的程序性规则；Memory 是按任务检索的历史信息，以 user-level 数据注入，不能覆盖 System Prompt。环境进化实验期间应冻结该文件，避免把环境改善与 Agent 记忆更新混为一谈。
