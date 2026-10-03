# Environment Evolution

环境搜索被限制为可记录、可回放的结构化变化：

- `surface.py`：AppWorld adapter 公布真实可变边界；
- `mutation.py`：一次只记录一个 add/replace/remove；
- `catalog.py`：可执行实现与参数 schema 白名单；
- `failure.py`：唯一最高优先级诊断；
- `materializer.py`：保留 task 与官方 evaluator，生成子环境配置；
- `lineage.py`：保存环境版本 DAG，并合并内容相同的节点。

通用分类包含 Setup、Contract/`f_A`、Transition/`f_T`、Observation/`f_O`、Budget。Modifier
只能从当前 AppWorld `MutationSurface` 支持的实现中选一个；候选必须经过同 task/seed 的
paired rollout 与回归守卫，不能只生成表面合法 JSON。
