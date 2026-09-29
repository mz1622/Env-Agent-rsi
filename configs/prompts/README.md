# System Prompts

这里保存 Agent 的系统提示词，每个文件都是带 `schema_version`、`name` 和 `content` 的 JSON 对象。运行时只从配置指定的文件加载，不在 Agent Python 类中维护第二份隐藏提示词。

Target prompt 约束任务执行和工具使用；Diagnostic prompt 约束只读归因及结构化 JSON 输出。改变 prompt 时应保留旧文件或记录版本，以便实验可复现。
