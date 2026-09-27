from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from env_agent_rsi.harness.rules import PostCommitTimeoutRule, StaleReadAfterWriteRule
from env_agent_rsi.harness.wrapper import RuleHarness
from env_agent_rsi.micro_api.item_env import ItemEnv


def load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict):
        raise ValueError("environment spec must be a JSON object")
    return spec


def build_environment(spec: Mapping[str, Any]) -> RuleHarness:
    task = dict(spec.get("task", {}))
    base = ItemEnv(
        target_value=str(task.get("target_value", "target-item")),
        page_size=int(task.get("page_size", 2)),
    )
    rules = dict(spec.get("rules", {}))
    transition_rules = []
    for rule_spec in rules.get("transition", []):
        rule_type = rule_spec.get("type")
        if rule_type == "post_commit_timeout":
            transition_rules.append(
                PostCommitTimeoutRule(
                    tool=str(rule_spec.get("tool", "append_item")),
                    trigger_on_nth=int(rule_spec.get("trigger_on_nth", 1)),
                )
            )
        else:
            raise ValueError(f"unknown transition rule: {rule_type!r}")

    observation_rules = []
    for rule_spec in rules.get("observation", []):
        rule_type = rule_spec.get("type")
        if rule_type == "stale_read_after_write":
            observation_rules.append(
                StaleReadAfterWriteRule(
                    stale_reads=int(rule_spec.get("stale_reads", 1))
                )
            )
        else:
            raise ValueError(f"unknown observation rule: {rule_type!r}")

    return RuleHarness(base, transition_rules, observation_rules)

