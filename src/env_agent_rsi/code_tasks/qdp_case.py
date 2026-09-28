"""SWE-bench QDP 大小写问题的最小可执行快照。

模块保存初始源码并在一次性目录的独立进程中运行 FAIL_TO_PASS/PASS_TO_PASS 测试，
用于快速研究代码环境演化；该隔离保证复现性，但不构成恶意代码安全沙箱。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

from env_agent_rsi.core.protocol import JsonObject


INITIAL_FILES = {
    "qdp_parser.py": '''"""Tiny QDP line classifier adapted from Astropy's QDP reader."""

import re

_COMMAND_RE = re.compile(r"^(READ|SKIP)(?:\\s|$)")


def line_type(line: str) -> str:
    stripped = line.strip()
    if not stripped or stripped.startswith("!"):
        return "comment"
    if _COMMAND_RE.match(stripped):
        return "command"
    tokens = stripped.split()
    if tokens and all(_is_number_or_no(token) for token in tokens):
        return "data"
    raise ValueError(f"Unrecognized QDP line: {line}")


def _is_number_or_no(token: str) -> bool:
    if token == "NO":
        return True
    try:
        float(token)
    except ValueError:
        return False
    return True
''',
    "README.md": """# qdp-mini

`qdp_parser.line_type` classifies command, data, and comment lines in a QDP file.
QDP commands are case-insensitive, but the current implementation rejects lower-case
commands and lower-case `no` values. Fix the implementation without changing tests.
""",
}


_TEST_RUNNER = r"""
import importlib.util
import json
from pathlib import Path

results = []

try:
    spec = importlib.util.spec_from_file_location(
        "qdp_parser", Path(__file__).with_name("qdp_parser.py")
    )
    qdp_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qdp_parser)
except Exception as exc:
    print(json.dumps({"import_error": f"{type(exc).__name__}: {exc}", "tests": []}))
    raise SystemExit(0)


def check(name, fn):
    try:
        fn()
    except Exception as exc:
        results.append({"name": name, "passed": False, "message": f"{type(exc).__name__}: {exc}"})
    else:
        results.append({"name": name, "passed": True, "message": "passed"})


def pass_uppercase():
    assert qdp_parser.line_type("READ SERR 1 2") == "command"
    assert qdp_parser.line_type("NO 1.0") == "data"
    assert qdp_parser.line_type("! comment") == "comment"


def fail_lowercase_command():
    assert qdp_parser.line_type("read serr 1 2") == "command"


def fail_lowercase_missing_value():
    assert qdp_parser.line_type("no 1.0") == "data"


check("pass_to_pass::uppercase_and_comment", pass_uppercase)
check("fail_to_pass::lowercase_command", fail_lowercase_command)
check("fail_to_pass::lowercase_missing_value", fail_lowercase_missing_value)
print(json.dumps({"tests": results}))
"""


def run_qdp_tests(files: Mapping[str, str], timeout_seconds: float = 3.0) -> JsonObject:
    """Run real Python tests in a disposable process.

    This is intentionally a lightweight research harness, not a security sandbox.
    Only run it with agents and code that are trusted to execute on the host.
    """

    source = files.get("qdp_parser.py")
    if not isinstance(source, str):
        return {
            "passed": False,
            "tests": [],
            "error": "qdp_parser.py is missing",
        }
    with tempfile.TemporaryDirectory(prefix="env-agent-rsi-code-") as directory:
        root = Path(directory)
        (root / "qdp_parser.py").write_text(source, encoding="utf-8")
        runner = root / "run_tests.py"
        runner.write_text(_TEST_RUNNER, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runner)],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "tests": [], "error": "test process timed out"}
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {
            "passed": False,
            "tests": [],
            "error": completed.stderr.strip() or "test process produced no result",
        }
    tests = result.get("tests", [])
    result["passed"] = bool(tests) and all(test.get("passed") for test in tests)
    result["return_code"] = completed.returncode
    return result
