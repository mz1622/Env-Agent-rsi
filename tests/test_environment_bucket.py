"""持久化环境 bucket、best 选择与 best-first expansion 测试。

这些测试保证差候选仍留在树上、best 不回退、优候选能成为下一轮父节点，并且进程
重启后完整 spec、DAG、评测和 best 指针可以恢复。
"""

from __future__ import annotations

from env_agent_rsi.evolution import (
    BestFirstEnvironmentSearch,
    EnvironmentBucket,
    EvaluationBatch,
    MutationSpec,
    MutationSurface,
    ToolSemantics,
    default_mutation_catalog,
)


def _root_spec() -> dict:
    return {
        "schema_version": 2,
        "environment": {"type": "appworld_process"},
        "task": {"id": "bucket-task"},
        "components": [],
    }


def _surface() -> MutationSurface:
    return MutationSurface(
        tools=(ToolSemantics("get_api_docs", "read"),),
        supported_phases=("contract",),
        supported_components=("contract",),
        supported_contract_axes=("f_A",),
        supported_implementations={"contract": ("add_tool_guidance",)},
        budget_dimensions=(),
    )


def _guidance(text: str) -> MutationSpec:
    return MutationSpec(
        "contract",
        "add_tool_guidance",
        {"tool": "get_api_docs", "guidance": text},
    )


def test_bucket_persists_tree_and_keeps_worse_candidate(tmp_path) -> None:
    bucket = EnvironmentBucket(tmp_path / "bucket", scope="appworld-task")
    search = BestFirstEnvironmentSearch(
        bucket,
        surface=_surface(),
        catalog=default_mutation_catalog(),
    )
    root = search.initialize(
        _root_spec(),
        EvaluationBatch(0.0, 0.5, 2, mean_steps=3.0, seeds=(1, 2)),
    )
    worse = search.expand_best(_guidance("Inspect an exact API contract."))
    search.record_result(
        worse,
        EvaluationBatch(0.0, 0.25, 2, mean_steps=6.0, seeds=(1, 2)),
    )

    assert bucket.best_node_id == root.node_id
    assert worse.child_node_id in bucket.dag.nodes
    assert len(bucket.dag.edges) == 1

    restored = EnvironmentBucket(tmp_path / "bucket", scope="appworld-task")
    assert restored.best_node_id == root.node_id
    assert restored.get_spec(worse.child_node_id) == worse.environment_spec
    assert len(restored.evaluations[worse.child_node_id]) == 1


def test_better_child_becomes_parent_of_next_expansion(tmp_path) -> None:
    bucket = EnvironmentBucket(tmp_path / "bucket")
    search = BestFirstEnvironmentSearch(
        bucket,
        surface=_surface(),
        catalog=default_mutation_catalog(),
    )
    root = search.initialize(
        _root_spec(),
        EvaluationBatch(0.0, 0.5, 1, mean_steps=4.0, seeds=(0,)),
    )
    better = search.expand_best(_guidance("Resolve the entity identifier first."))
    search.record_result(
        better,
        EvaluationBatch(1.0, 1.0, 1, mean_steps=7.0, seeds=(0,)),
    )
    assert bucket.best_node_id == better.child_node_id

    next_expansion = search.expand_best(
        _guidance("Print the evidence before finishing.")
    )
    assert next_expansion.parent_node_id == better.child_node_id
    assert next_expansion.parent_node_id != root.node_id
    assert bucket.best_node_id == better.child_node_id
    assert bucket.aggregate(better.child_node_id) == {
        "success_rate": 1.0,
        "verifier_score": 1.0,
        "mean_steps": 7.0,
        "rollout_count": 1,
        "evaluation_batches": 1,
    }


def test_evaluation_batch_extracts_official_partial_score() -> None:
    batch = EvaluationBatch.from_episodes(
        [
            {
                "evaluation": {
                    "success": False,
                    "metrics": {"passed": 1, "total": 2},
                },
                "steps": 5,
            },
            {
                "evaluation": {
                    "success": True,
                    "metrics": {"passed": 2, "total": 2},
                },
                "steps": 7,
            },
        ],
        seeds=(10, 11),
        source="paired-rollout",
    )
    assert batch.success_rate == 0.5
    assert batch.verifier_score == 0.75
    assert batch.mean_steps == 6.0
    assert batch.rollout_count == 2
