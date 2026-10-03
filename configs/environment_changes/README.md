# AppWorld 环境变化配置

`appworld_supported.json` 只声明 AppWorld 当前真实支持的 Setup、Contract、Action (`f_A`)
和 Budget。通用代码仍定义 Transition (`f_T`) 与 Observation (`f_O`)，但 AppWorld 的第一版
`MutationSurface` 会拒绝它们，直到 worker 暴露可验证的事务边界和字段级 observation 映射。
