"""环境 Harness 的公共入口。

向外提供配置加载、组件注册、环境装配和 RuleHarness；具体场景代码只需实现协议并
注册，不应修改 Agent runner。
"""

from .factory import (
    available_components,
    build_environment,
    load_spec,
    register_action_rule,
    register_contract_rule,
    register_environment,
    register_observation_rule,
    register_transition_rule,
    register_verifier,
)
from .wrapper import RuleHarness

__all__ = [
    "RuleHarness",
    "available_components",
    "build_environment",
    "load_spec",
    "register_action_rule",
    "register_contract_rule",
    "register_environment",
    "register_observation_rule",
    "register_transition_rule",
    "register_verifier",
]
