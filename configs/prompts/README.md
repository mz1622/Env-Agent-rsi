# System Prompts

这里保存 Agent 的系统提示词，每个文件都是带 `schema_version`、`name` 和 `content` 的 JSON 对象。运行时只从配置指定的文件加载，不在 Agent Python 类中维护第二份隐藏提示词。

Target prompt 由 Agent0 数据桥读取，并与 AppWorld skill、动态 Tool Register 组合；Diagnostic prompt 强制只返回一个最高优先级根因；Modifier prompt 强制只返回一个白名单内的可执行环境变化。改变 prompt 时应记录版本，以便实验可复现。
