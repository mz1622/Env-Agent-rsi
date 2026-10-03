"""AppWorld 环境 Harness 的公共入口。

向外提供配置加载、环境与六类规则注册、装配和 checkpoint；任务评分由 AppWorld
backend 的官方 evaluator 负责，不再维护一套按数据集注册的 verifier。
"""

from .factory import (
    available_components,
    build_environment,
    load_spec,
    register_action_rule,
    register_budget_rule,
    register_contract_rule,
    register_environment,
    register_observation_rule,
    register_setup_rule,
    register_transition_rule,
)
from .wrapper import RuleHarness
from .layers import HarnessLayer, RuleLayer
from .checkpoint import (
    EnvironmentCheckpoint,
    build_stack,
    dump_stack,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "RuleHarness",
    "HarnessLayer",
    "RuleLayer",
    "EnvironmentCheckpoint",
    "build_stack",
    "dump_stack",
    "load_checkpoint",
    "save_checkpoint",
    "available_components",
    "build_environment",
    "load_spec",
    "register_action_rule",
    "register_budget_rule",
    "register_contract_rule",
    "register_environment",
    "register_observation_rule",
    "register_setup_rule",
    "register_transition_rule",
]
