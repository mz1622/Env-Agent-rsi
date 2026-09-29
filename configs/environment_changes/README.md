# Six Environment Changes

`all_six.json` 展示一个环境节点如何同时声明六类变化：

1. `setup`：reset 后、Agent 开始前重放动作来塑造初始状态；
2. `contract`：改变 Agent 可见工具 schema，并同步做执行前校验；
3. `action`：在基础环境执行前进行隐藏 guard 或 action rewrite；
4. `transition`：在真实状态转移后改变故障/提交语义；
5. `observation`：只改变 Agent 看到的结果，不改变 verifier 读取的真实状态；
6. `budget`：限制 step、write 等资源并报告消耗。

Factory 按固定顺序装配这六组规则。搜索器生成的 `MutationSpec` 只能选择其中一个阶段和一个已注册实现，不能直接注入 Python 源码。
