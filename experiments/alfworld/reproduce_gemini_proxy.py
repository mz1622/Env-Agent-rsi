"""Run the resumable ALFWorld EnvHarness-skills condition from this checkout.

    python3 experiments/alfworld/reproduce_gemini_proxy.py check
    python3 experiments/alfworld/reproduce_gemini_proxy.py run
    python3 experiments/alfworld/reproduce_gemini_proxy.py status

Repeat run with the same ROOT_RUN to resume. --through corpus|skills|bank
stops after that stage. A separate native Gemini embedding service takes
EH_GEMINI_EMBED_API_BASE (including the version) and EH_GEMINI_EMBED_API_KEY.
For a locally served OpenAI-compatible embedding model, set EH_EMBED_MODEL
to openai/<served-name> and EH_LOCAL_EMBED_API_BASE to its /v1 URL.
"""
import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
CHAT = "gemini/gemini-3.1-flash-lite"
EMBED = "gemini/gemini-embedding-001"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def check_tool():
    from envharness.agents.harness_agent import PROPOSE_TOOL
    from envharness.infra.llm import Message
    from envharness.infra.model import build_client

    name = PROPOSE_TOOL["function"]["name"]
    response = build_client(CHAT, max_retries=0).chat(
        messages=[Message(role="user", content=(
            "Connectivity check. Call propose_candidate with a short rationale, "
            "empty rules_code and no in_env_actions."
        ))],
        tools=[PROPOSE_TOOL],
        tool_choice={"type": "function", "function": {"name": name}},
        temperature=0.7, max_tokens=256, timeout=30,
    )
    if not any(call.name == name and isinstance(call.arguments, dict)
               for call in response.tool_calls):
        raise ValueError("Missing named tool call")
    print(f"Chat and tool call OK ({CHAT}, {name}).")


def embedding_profile():
    model = os.getenv("EH_EMBED_MODEL") or EMBED
    if model.startswith("openai/"):
        base = os.getenv("EH_LOCAL_EMBED_API_BASE", "").rstrip("/")
        if not base:
            raise ValueError("Set EH_LOCAL_EMBED_API_BASE to the local embedding server's /v1 URL.")
        # Never inherit a cloud credential when selecting a local service.
        os.environ.update(OPENAI_API_BASE=base, OPENAI_BASE_URL=base,
                          OPENAI_API_KEY=os.getenv("EH_LOCAL_EMBED_API_KEY") or "local")
    elif model.startswith("gemini/"):
        base = (os.getenv("EH_GEMINI_EMBED_API_BASE") or os.getenv("GEMINI_API_BASE") or GEMINI_BASE).rstrip("/")
    else:
        raise ValueError("Expected gemini/<model> or openai/<local-embedding-model>.")
    url = urlsplit(base)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Use an API base URL without embedded credentials or query parameters.")
    return dict(model=model, base=base, revision=os.getenv("EH_EMBED_REVISION") or "")


def check_embedding(profile):
    import litellm
    from envharness.infra.model import embedding_kwargs

    response = litellm.embedding(
        input=["connectivity check", "second input"],
        **embedding_kwargs(model=profile["model"]), timeout=20, num_retries=0,
    )
    dimensions = [len(item["embedding"]) for item in response.data]
    if len(dimensions) != 2 or dimensions[0] <= 0 or dimensions[0] != dimensions[1]:
        raise ValueError(f"Unexpected embedding dimensions {dimensions}")
    if profile["model"] == EMBED and dimensions[0] != 3072:
        raise ValueError(f"Expected gemini-embedding-001 at native width, got {dimensions}")
    print(f"Batch embedding OK ({profile['model']}, 2 x {dimensions[0]}).")
    return dimensions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "run", "status"], nargs="?", default="check")
    parser.add_argument("--run-dir", type=Path, default=os.getenv("ROOT_RUN") or REPO / "runs/alfworld_gemini_ours")
    parser.add_argument("--through", choices=["corpus", "skills", "bank", "eval"],
                        default="bank" if os.getenv("SKIP_EVAL") else "eval")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    sys.path.insert(0, str(REPO))
    from experiments.alfworld import ours_resume
    if args.mode == "status":
        ours_resume.status(root)
        return 0
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        parser.error("Set GEMINI_API_KEY or GOOGLE_API_KEY for the chat endpoint.")
    os.environ.update(
        MODEL=CHAT, EH_MODEL=CHAT, EH_EMBED_MODEL=os.getenv("EH_EMBED_MODEL") or EMBED,
        PY=sys.executable, PYTHONPATH=str(REPO), LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    os.environ.setdefault("GEMINI_API_BASE", GEMINI_BASE)
    os.environ.setdefault("ALFWORLD_DATA", str(Path.home() / ".cache/alfworld"))
    os.chdir(REPO)
    profile = embedding_profile() if args.mode == "check" or args.through in ("bank", "eval") else None
    if args.mode == "check":
        ok = True
        for label, check in [("chat/tool", check_tool), ("embedding", lambda: check_embedding(profile))]:
            try:
                check()
            except Exception as exc:
                ok = False
                print(f"{label} failed ({getattr(exc, 'status_code', None) or type(exc).__name__}).",
                      file=sys.stderr)
        return 0 if ok else 1
    ours_resume.run(root, REPO, args.through, profile, lambda: check_embedding(profile))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        message = str(exc)
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "EH_GEMINI_EMBED_API_KEY", "EH_LOCAL_EMBED_API_KEY", "OPENAI_API_KEY"):
            value = os.getenv(name)
            if value and value != "local":
                message = message.replace(value, "[redacted]")
        sys.exit(f"{type(exc).__name__}: {message[:700]}")
