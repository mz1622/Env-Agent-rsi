from __future__ import annotations

import json
import unittest
from pathlib import Path

from env_agent_rsi.agents import run_naive_agent, run_oracle_agent
from env_agent_rsi.core.protocol import Action
from env_agent_rsi.harness.factory import build_environment, load_spec


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/micro_api/baseline.json"
FAULTED = ROOT / "configs/micro_api/postcommit_stale.json"


class MinimalEnvironmentTests(unittest.TestCase):
    def test_same_seed_reset_is_deterministic(self) -> None:
        env = build_environment(load_spec(BASELINE))
        env.reset(seed=17)
        first = json.dumps(env.save_state(), sort_keys=True)
        env.reset(seed=17)
        second = json.dumps(env.save_state(), sort_keys=True)
        self.assertEqual(first, second)

    def test_post_commit_timeout_preserves_real_write(self) -> None:
        env = build_environment(load_spec(FAULTED))
        env.reset(seed=0)
        response = env.step(Action("append_item", {"value": "target-item"}))
        self.assertFalse(response.observation["ok"])
        self.assertEqual(response.observation["error"]["code"], "TIMEOUT")
        self.assertEqual(
            sum(
                item["value"] == "target-item" for item in env.get_env_state()["items"]
            ),
            1,
        )

    def test_first_read_after_write_is_stale(self) -> None:
        env = build_environment(load_spec(FAULTED))
        env.reset(seed=0)
        env.step(Action("append_item", {"value": "target-item"}))
        stale = env.step(Action("list_items", {"limit": 100}))
        fresh = env.step(Action("list_items", {"limit": 100}))
        self.assertFalse(
            any(x["value"] == "target-item" for x in stale.observation["items"])
        )
        self.assertTrue(
            any(x["value"] == "target-item" for x in fresh.observation["items"])
        )

    def test_observe_also_uses_observation_rules(self) -> None:
        env = build_environment(load_spec(FAULTED))
        env.reset(seed=0)
        env.step(Action("append_item", {"value": "target-item"}))
        stale = env.observe()
        fresh = env.observe()
        self.assertFalse(any(x["value"] == "target-item" for x in stale["items"]))
        self.assertTrue(any(x["value"] == "target-item" for x in fresh["items"]))

    def test_naive_retry_creates_duplicate_and_fails(self) -> None:
        env = build_environment(load_spec(FAULTED))
        env.reset(seed=0)
        result, _ = run_naive_agent(env, "target-item")
        self.assertFalse(result.success)
        self.assertEqual(result.metrics["target_count"], 2)

    def test_oracle_handles_timeout_and_stale_read(self) -> None:
        env = build_environment(load_spec(FAULTED))
        env.reset(seed=0)
        result, _ = run_oracle_agent(env, "target-item")
        self.assertTrue(result.success)
        self.assertEqual(result.metrics["target_count"], 1)

    def test_snapshot_restores_rule_state(self) -> None:
        spec = load_spec(FAULTED)
        env = build_environment(spec)
        env.reset(seed=5)
        env.step(Action("append_item", {"value": "target-item"}))
        snapshot = env.save_state()

        restored = build_environment(spec)
        restored.load_state(snapshot)
        first_read = restored.step(Action("list_items", {"limit": 100}))
        self.assertFalse(
            any(x["value"] == "target-item" for x in first_read.observation["items"])
        )
        second_read = restored.step(Action("list_items", {"limit": 100}))
        self.assertTrue(
            any(x["value"] == "target-item" for x in second_read.observation["items"])
        )


if __name__ == "__main__":
    unittest.main()
