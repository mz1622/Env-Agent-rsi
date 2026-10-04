# AppWorld 环境变化配置

`appworld_supported.json` 只声明 AppWorld 当前真实支持的 Setup、Contract、Action (`f_A`)
和 Budget。通用代码仍定义 Transition (`f_T`) 与 Observation (`f_O`)，但 AppWorld 的第一版
`MutationSurface` 会拒绝它们，直到 worker 暴露可验证的事务边界和字段级 observation 映射。

AppWorld 的 Contract 白名单包括：

- `require_argument`：让现有参数在可见 schema 与运行校验中同时成为必填；
- `add_tool_guidance`：向一个现有工具说明追加最小操作顺序提示，不改变工具语义且不得
  包含答案。
