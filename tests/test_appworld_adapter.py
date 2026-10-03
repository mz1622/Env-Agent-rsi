"""验证 AppWorld adapter 的公开契约不依赖 AppWorld 包即可导入。"""

from __future__ import annotations

from env_agent_rsi.benchmarks.appworld.backend import APPWORLD_TOOLS


def test_appworld_tools_are_small_dynamic_gateway() -> None:
    names = {tool["function"]["name"] for tool in APPWORLD_TOOLS}
    assert names == {"get_api_docs", "execute_python", "finish"}
    execute = next(
        tool for tool in APPWORLD_TOOLS if tool["function"]["name"] == "execute_python"
    )
    assert execute["function"]["parameters"]["required"] == ["code"]
