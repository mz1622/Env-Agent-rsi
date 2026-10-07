# Copyright 2026 The EnvHarness Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for the local-Qwen / DeepSeek split model profile."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from envharness.infra.model import api_key_for, client_kwargs
from scripts.run_harness import _apply_model_overrides, _client_from_block


ROOT = Path(__file__).resolve().parents[1]
POLICY_MODEL = "ollama/qwen3:4b-direct"
AGENT_MODEL = "deepseek/deepseek-flash"


def test_deepseek_key_file_and_provider_defaults(tmp_path, monkeypatch) -> None:
    key_file = tmp_path / "deepseek.key"
    key_file.write_text("test-secret-value\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEYS", raising=False)
    monkeypatch.setenv("EH_DEEPSEEK_API_KEY_FILE", str(key_file))

    assert api_key_for(AGENT_MODEL) == "test-secret-value"
    resolved = client_kwargs(AGENT_MODEL, apply_global_override=False)
    assert resolved["model"] == AGENT_MODEL
    assert resolved["api_base"] == "https://api.deepseek.com"
    assert resolved["api_key"] == "test-secret-value"
    assert resolved["thinking"] == {"type": "disabled"}


def test_local_qwen_provider_defaults(monkeypatch) -> None:
    monkeypatch.delenv("EH_MODEL", raising=False)
    resolved = client_kwargs(POLICY_MODEL)
    assert resolved == {
        "model": POLICY_MODEL,
        "drop_params": True,
        "api_base": "http://127.0.0.1:11434",
    }


def test_role_overrides_take_precedence_over_shared_model() -> None:
    cfg = {
        "policy": {
            "client_factory": "old.Policy",
            "client_kwargs": {"model": "old-policy", "temperature": 0},
        },
        "agent": {
            "client_factory": "old.Agent",
            "client_kwargs": {"model": "old-agent"},
        },
    }
    _apply_model_overrides(
        cfg,
        model="openai/shared",
        policy_model=POLICY_MODEL,
        agent_model=AGENT_MODEL,
    )
    assert cfg["policy"]["model"] == POLICY_MODEL
    assert cfg["agent"]["model"] == AGENT_MODEL
    assert "client_factory" not in cfg["policy"]
    assert "client_factory" not in cfg["agent"]
    assert "model" not in cfg["policy"]["client_kwargs"]
    assert "model" not in cfg["agent"]["client_kwargs"]


def test_explicit_cli_choice_can_ignore_legacy_global_override(monkeypatch) -> None:
    monkeypatch.setenv("EH_MODEL", "openai/should-not-win")
    cfg = {
        "policy": {"model": POLICY_MODEL},
        "agent": {"model": AGENT_MODEL},
    }
    _apply_model_overrides(
        cfg,
        model="openai/shared",
        policy_model=POLICY_MODEL,
        agent_model=AGENT_MODEL,
    )

    _, policy_kwargs = _client_from_block(
        cfg["policy"], apply_global_override=False
    )
    _, agent_kwargs = _client_from_block(
        cfg["agent"], apply_global_override=False
    )
    assert policy_kwargs["model"] == POLICY_MODEL
    assert agent_kwargs["model"] == AGENT_MODEL


def test_every_corpus_config_uses_split_roles() -> None:
    paths = sorted((ROOT / "experiments").glob("*/corpus*.yaml"))
    paths.extend(sorted((ROOT / "experiments/toy24").glob("*.yaml")))
    assert paths

    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        policy = cfg.get("policy") or {}
        policy_model = policy.get("model") or (policy.get("client_kwargs") or {}).get("model")
        assert policy_model == POLICY_MODEL, path

        agent = cfg.get("agent") or cfg.get("mutator") or {}
        if str(agent.get("type", "llm")).lower() == "noop":
            continue
        agent_model = agent.get("model") or (agent.get("client_kwargs") or {}).get("model")
        assert agent_model == AGENT_MODEL, path


def test_reasoning_bank_eval_configs_use_local_qwen() -> None:
    paths = sorted((ROOT / "experiments").glob("*/reasoning_bank_eval*.yaml"))
    assert paths
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (cfg.get("model") or {}).get("name") == POLICY_MODEL, path


def test_credentials_are_not_present_in_experiment_configs() -> None:
    for path in (ROOT / "experiments").glob("**/*.yaml"):
        payload = json.dumps(
            yaml.safe_load(path.read_text(encoding="utf-8")), sort_keys=True
        ).lower()
        assert "api_key" not in payload, path
        assert "test-secret-value" not in payload, path
