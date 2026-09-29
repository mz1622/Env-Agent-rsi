# Environment Evolution

这个模块把“搜索环境”限制为可记录、可复放的结构化变化，而不是让模型直接改环境源码。

- `surface.py`：由 adapter 公布可变工具、观察通道、预算维度和支持的变化阶段。
- `mutation.py`：一次变化的 JSON 结构，阶段固定为 Setup、Contract、Action、Transition、Observation、Budget 六类。
- `failure.py`：把 episode 轨迹与独立 verifier 结果压缩为失败签名；无模型时也有确定性回退。
- `lineage.py`：保存环境版本、变化边和评估元数据。内容相同的环境合并成一个节点，因此它是 DAG 而不是必须重复展开的树。

诊断 Agent 只提出 `candidate_changes`；真正可执行的变化必须通过 `MutationSurface` 校验，并由 Harness 注册表物化。这样 verifier、Agent 参数和 benchmark 原始目标不会被环境搜索偷偷改写。
