# Environment Evolution

环境搜索被限制为可记录、可回放的结构化变化：

- `surface.py`：AppWorld adapter 公布真实可变边界；
- `mutation.py`：一次只记录一个 add/replace/remove；
- `catalog.py`：可执行实现与参数 schema 白名单；
- `failure.py`：唯一最高优先级诊断；
- `materializer.py`：保留 task 与官方 evaluator，生成子环境配置；
- `lineage.py`：保存环境版本 DAG，并合并内容相同的节点。
- `bucket.py`：原子持久化完整 spec、DAG、分批评测、聚合分数与当前 best 指针；
- `search.py`：只从当前 best 物化下一条 mutation edge，评测后再决定是否移动 best。

通用分类包含 Setup、Contract/`f_A`、Transition/`f_T`、Observation/`f_O`、Budget。Modifier
只能从当前 AppWorld `MutationSurface` 支持的实现中选一个；候选必须经过同 task/seed 的
paired rollout 与回归守卫，不能只生成表面合法 JSON。

## Environment bucket 与 best-first 搜索

一个 bucket 对应一个固定 search scope，例如同一 AppWorld task，或同一任务集合与完全
一致的评测协议。若对任务集合搜索，node 应保存不含具体 task id 的环境模板，评测批次
则聚合该集合上的 paired rollouts；不能把不同任务各自的完整 spec 混入同一棵树比较。

节点选择使用稳定字典序：`success_rate` → 官方 `verifier_score` → 成功节点的较少步骤。
未评测 child 不会成为 best；同分时保留 incumbent，避免 best 指针抖动。较差、崩溃或
无收益节点继续留在 archive 中，使后续搜索知道哪些 mutation 路径已经尝试过。
