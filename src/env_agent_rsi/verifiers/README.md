# Verifiers

Verifier 只读取环境真实快照来判断任务成功，不读取 Agent 看见的 observation，也不被 Diagnostic Agent 修改。每个内置任务族有独立实现，检查目标状态和不允许的旁路修改。

环境演化必须保持 verifier 代码或哈希不变，否则得到的是换题，而不是帮助同一个 Target Agent 完成原任务。
