"""启动复用 Agent0 AsyncToolServer 的 AppWorld 工具服务。

Agent0 的 get_tool_cls 假定扩展位于其源码包内；为保持 submodule 原样，本启动器只在进程
内注册本仓库的 AppWorldAgent0Tool，并把解析函数定向到该类。
"""

from __future__ import annotations

import argparse
from typing import Any

from env_agent_rsi.integrations.agent0.tool import AppWorldAgent0Tool
from env_agent_rsi.integrations.agent0.upstream import validate_agent0_checkout


def _load_server() -> tuple[Any, Any]:
    validate_agent0_checkout()
    from verl_tool.servers.tools import base
    from verl_tool.servers import serve

    if "appworld" not in base.ALL_TOOLS:
        base.ALL_TOOLS.append("appworld")
    original = serve.get_tool_cls

    def resolve(tool_type: str) -> Any:
        if tool_type == "appworld":
            return AppWorldAgent0Tool
        return original(tool_type)

    serve.get_tool_cls = resolve
    return serve.AsyncToolServer, serve.ServerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent0 AppWorld tool server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--workers-per-tool", type=int, default=4)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args()
    server_type, config_type = _load_server()
    config = config_type(
        host=args.host,
        port=args.port,
        workers_per_tool=args.workers_per_tool,
        max_concurrent_requests=args.max_concurrent_requests,
        request_timeout=args.request_timeout,
        enable_hashing=False,
    )
    server_type(tool_types=("appworld",), config=config).start()


if __name__ == "__main__":
    main()

