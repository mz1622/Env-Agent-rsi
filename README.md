# *EnvHarness*: Awakening Static Worlds for Agent Learning

Check out our [paper](https://arxiv.org/abs/2608.19880) and [webpage](https://envharness.com/) for more details.

## 🔥 Updates

<!-- FILL: one dated bullet per release / acceptance / follow-up, newest first. -->
- [2026-08-21] We released our [paper](https://arxiv.org/abs/2608.19880) and [website](https://envharness.com/).

## 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview

As LLMs become autonomous agents, they learn less from curated text and more from
interactive environments. But those environments are expensive to build and, once
built, stay **static** — they behave identically no matter which agent interacts
with them or how much it has improved, so they can neither target a particular
agent's weaknesses nor keep teaching once its tasks are solved. **EnvHarness**
applies the *agent harness* idea to the other side of the interaction: just as an
agent harness makes a frozen LLM capable through plug-in components (skills,
memory, tools) without changing its weights, EnvHarness wraps a **frozen
environment** with its own plug-in components to make it dynamically controllable
— without touching the environment's internal code.

The environment search uses two change protocols. **Stage** is implemented by
`Setup`: it replays normal tool actions after reset to reshape the initial
state. **Contract** is implemented by `Rules`: it changes which actions are
allowed, what transitions return, and what the agent observes through the A/T/O
hooks. Each Candidate returns four independent slots—Stage, f_A, f_T, and
f_O—and may leave any inapplicable slot empty. The goal predicate remains
untouched, preserving the benchmark's trusted verifier.

An LLM **designer agent** drives a diagnostic loop: it reads the
agent's trajectories to diagnose a specific weakness, writes components that
reshape the environment to target it, tests the policy in the new environment,
and revises until the environment can actually *teach* what the agent lacks. The
signal is targeted (written against diagnosed flaws) and lasting (the loop repeats
as the agent improves, co-evolving the two).

Across ALFWorld, WebArena, SWE-bench Verified, OfficeQA, and SpreadsheetBench,
skills learned in EnvHarness environments beat both the no-skill baseline and
skills learned in the original environments — more effective (up to +9 points on
held-out tasks) and more efficient (~9.8% fewer interaction steps). The same
dynamic environments also produce stronger policies under reinforcement learning,
and repeating the designer loop compounds the gains round after round.

### Key Features
* **Frozen environments, no internal edits:** the benchmark's task set, its
  dynamics and its grading stay exactly as published. Only the layer the agent
  acts *through* changes.
* **Code as the Envharness:** the designer emits real Python — a `_Rules(Rules)`
  subclass — not a selection from a fixed menu. It is compiled and executed in
  an isolated subprocess, so a bad mutation becomes a recorded trace instead of
  a dead run.
* **Four independent change slots:** every proposal evaluates Stage plus
  Contract f_A/f_T/f_O independently and fills only applicable slots.
* **External-search boundary:** EnvRigger emits up to four independent changes
  before external selection. MCTS may select a current or historical proposal
  together with its source environment; EnvHarness then applies and validates
  it as a new node. See
  [`envharness/orchestration/TREE_SEARCH.md`](envharness/orchestration/TREE_SEARCH.md).
* **Benchmark-agnostic:** adding an environment means implementing one
  interface; the designer, the components, the loop and the evaluation stages
  need no changes.

---

## ⚡️ Quickstart Guide

### 0. LLM Configuration

This fork uses two deliberately separate model roles by default:

- **Policy / Target Agent:** official Hugging Face
  `Qwen/Qwen3-4B-Instruct-2507`, loaded directly by Python/Transformers from
  `models/Qwen3-4B-Instruct-2507`. No Ollama or local HTTP server is used.
- **HarnessAgent / Mutator (Env Rigger):** DeepSeek API
  `deepseek/deepseek-flash` at `https://api.deepseek.com`.

Install the local-model dependencies and download the weights once:

```bash
python -m pip install -e '.[local-qwen]'
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir models/Qwen3-4B-Instruct-2507
```

Set `EH_LOCAL_MODEL_PATH` only when the weights live somewhere else. The
default `models/` directory is gitignored. The direct client caps each response
at 8192 new tokens; generation stops earlier when the model emits EOS.

DeepSeek credentials are resolved in this order: `DEEPSEEK_API_KEY`,
`DEEPSEEK_API_KEYS`, then the file named by `EH_DEEPSEEK_API_KEY_FILE`. With no
file override, the ignored repository-root `api.txt` is used. Never commit that
file.

```bash
printf '%s\n' 'your-deepseek-api-key' > api.txt
python scripts/check_env.py toy24
```

Every corpus YAML in `experiments/` keeps the two roles explicit:

```yaml
policy:
  model: local/Qwen3-4B-Instruct-2507
agent:  # some legacy configs call this block `mutator`
  type: llm
  model: deepseek/deepseek-flash
```

Override roles independently when needed:

```bash
python scripts/run_harness.py --config experiments/toy24/mutated_smoke.yaml \
  --policy-model local/Qwen3-4B-Instruct-2507 \
  --agent-model deepseek/deepseek-flash
```

The legacy `--model` flag and `EH_MODEL` variable are still supported and
intentionally override **both** roles. Do not use them for the default split
setup. Full reproduction drivers expose `POLICY_MODEL`, `AGENT_MODEL`, and
`INDUCTION_MODEL` separately.

The direct local Qwen backend keeps one cached model per Python process and
serializes generation on that model. Corpus configs therefore use the
in-process runner: K environments may exist concurrently, but they share one
copy of the 4B weights instead of loading one copy per subprocess. Remote
HarnessAgent concurrency still follows the provider's rate limit; a pool large
enough to saturate its tokens-per-minute tier turns into 429s that truncate
episodes mid-task.

#### Embeddings

Skill retrieval needs an embedding model too. OpenAI, Vertex and Gemini have
paired defaults. DeepSeek chat and the local Transformers Policy do not imply a
compatible embedding space, so set `EH_EMBED_MODEL` explicitly before building
or querying reasoning banks.

| `MODEL` provider | embedding model | dim |
| --- | --- | --- |
| `openai/...` | `openai/text-embedding-3-small` | 1536 |
| `vertex_ai/...` | `vertex_ai/text-embedding-004` (same ADC) | 768 |
| `gemini/...` | `gemini/gemini-embedding-001` | 3072 |
| `deepseek/...`, `local/...` or `ollama/...` | set `EH_EMBED_MODEL` explicitly | provider-dependent |

`EH_EMBED_MODEL` overrides the pairing for both bank building and retrieval,
and takes any embedding model litellm supports (with that provider's own
credentials set):

```bash
EH_EMBED_MODEL=openai/text-embedding-3-small \
MODEL=vertex_ai/claude-sonnet-4-6 python experiments/alfworld/reproduce.py
```

That is also why the override exists: it holds one embedding space fixed while
the policy provider changes. A bank stores its vectors, and `Bank.retrieve`
rejects a query vector of a different width, so changing the embedding model —
including by changing provider — means rebuilding the bank.

### 1. Run a benchmark

Each benchmark has its own environment and its own one-command driver.
**Open the README in the experiment folder you want to run** — it carries the
environment setup, the run commands and the knobs for that benchmark:

- [`experiments/toy24`](experiments/toy24/README.md)
- [`experiments/alfworld`](experiments/alfworld/README.md)
- [`experiments/swebench`](experiments/swebench/README.md)
- [`experiments/webarena`](experiments/webarena/README.md)
- [`experiments/officeqa`](experiments/officeqa/README.md)
- [`experiments/spreadsheetbench`](experiments/spreadsheetbench/README.md)

Every folder follows the same shape:

```bash
python scripts/check_env.py <benchmark>           # preflight
bash experiments/<benchmark>/reproduce_smoke.sh   # the same stages, fewer tasks
python experiments/<benchmark>/reproduce.py       # the full protocol
```

The experiments above distill/evaluate **skills**. For **RL training** — a policy
trained with GRPO directly inside EnvHarness environments (via verl-agent) — see
[`rl/`](rl/README.md).



## 📊 Results

Skills induced from EnvHarness-adapted environments transfer back to the
**untouched** benchmark and beat both controls — no skills at all, and skills
induced from the original environments. All numbers are the mean over three
independent runs, with standard deviations as subscripts. A dash marks a
baseline that is benchmark-specific and cannot be applied to the other domain;
EnvHarness covers every benchmark through the same interface.

### ALFWorld and WebArena

| Skill Source | ALFWorld In-Dist | ALFWorld OOD | ALFWorld Avg. | Reddit | Shopping | Shop Admin | GitLab | WebArena Avg. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| No Skills | 62.6<sub>1.7</sub> | 60.7<sub>5.2</sub> | 61.7<sub>3.4</sub> | 39.6<sub>2.3</sub> | 35.2<sub>3.3</sub> | 44.1<sub>2.3</sub> | 35.8<sub>8.4</sub> | 38.7<sub>2.3</sub> |
| Original Envs | 63.3<sub>2.8</sub> | 61.4<sub>4.3</sub> | 62.4<sub>3.4</sub> | 38.7<sub>9.7</sub> | 35.2<sub>1.3</sub> | 44.6<sub>3.0</sub> | 35.4<sub>4.0</sub> | 38.5<sub>3.1</sub> |
| GenEnv | 63.3<sub>1.2</sub> | 61.9<sub>2.7</sub> | 62.6<sub>1.9</sub> | — | — | — | — | — |
| VeriEnv | — | — | — | 39.6<sub>4.2</sub> | 30.2<sub>0.0</sub> | 49.7<sub>2.4</sub> | **38.9**<sub>5.6</sub> | 39.6<sub>1.4</sub> |
| **EnvHarness Envs** | **66.2**<sub>0.3</sub> | **70.4**<sub>2.3</sub> | **68.3**<sub>1.3</sub> | **40.6**<sub>4.7</sub> | **37.4**<sub>0.3</sub> | **50.8**<sub>1.5</sub> | 37.7<sub>3.1</sub> | **41.6**<sub>1.8</sub> |
| *Δ (EnvHarness − Original)* | *+2.9* | *+9.0* | *+5.9* | *+1.9* | *+2.2* | *+6.2* | *+2.3* | *+3.1* |

### SWE-bench Verified, OfficeQA and SpreadsheetBench

| Skill Source | SWE-verified SR ↑ | SWE-verified AS ↓ | OfficeQA EM ↑ | OfficeQA F1 ↑ | SpreadsheetBench Pass@1 ↑ | SpreadsheetBench Mean Score ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| No Skills | 47.67<sub>0.93</sub> | 53.58<sub>2.93</sub> | 54.23<sub>2.84</sub> | 55.77<sub>2.98</sub> | 46.44<sub>0.15</sub> | 61.32<sub>0.37</sub> |
| Original Envs | 49.88<sub>2.59</sub> | 55.01<sub>1.69</sub> | 54.40<sub>1.84</sub> | 55.77<sub>1.59</sub> | 45.88<sub>1.19</sub> | 61.47<sub>0.59</sub> |
| SWE-smith | 50.12<sub>1.74</sub> | 54.72<sub>2.03</sub> | — | — | — | — |
| **EnvHarness Envs** | **52.58**<sub>2.72</sub> | **49.61**<sub>2.49</sub> | **56.20**<sub>2.34</sub> | **57.73**<sub>2.29</sub> | **49.15**<sub>0.36</sub> | **62.48**<sub>0.27</sub> |
| *Δ (EnvHarness − Original)* | *+2.70* | *−5.40* | *+1.80* | *+1.97* | *+3.27* | *+1.01* |

SR = success rate, AS = agent steps (lower is better), EM = exact match.
The three skill sources correspond to the conditions each `reproduce.py` prints:
`nobank` (No Skills), `orig` (Original Envs) and `ours` (EnvHarness Envs).

**Models.** Every number in both tables was produced with **Gemini**. This fork
defaults to local Qwen for Policy rollouts and DeepSeek for environment
rigging, so exact reproduction of the upstream tables requires intentionally
overriding both roles back to Gemini:

```bash
python scripts/run_harness.py --config experiments/swebench/corpus.yaml \
  --model gemini/gemini-3.5-flash
```

For skill-bank stages, set `EH_EMBED_MODEL` to the embedding space used to build
the bank. Absolute numbers move with the model; what the tables compare is skill
sources at a fixed model.

## 🧱 Adding a New Benchmark

A benchmark joins EnvHarness by implementing **one interface**, `ActionableEnv`
(`reset / step / observe / evaluate / get_env_state / save_state / from_state`).
Everything downstream — the Environment Designer, the three components, the loop,
the evaluation — is benchmark-agnostic and needs no changes.

### 1. Implement the Bridge

The Bridge is the **only** layer that may know about a docker container, a
browser session, or a simulator. Subclass `ActionableEnv`, register a stable
tag, and implement the seven required methods.

`tool_registry` declares the action space. Its schemas are what the Policy is
given to act with, and what the Environment Designer is shown so a Rule's
action hook can match on `action.name`. Each entry is a `Tool` whose schema is
introspected from its `invoke` signature — see any
`envharness/bridges/*/tools.py` for the shape.

```python
# envharness/bridges/mybench/bridge.py
from envharness.core.actionable_env import ActionableEnv
from envharness.core.registry import register_env
from envharness.core.types import (
    Action, EnvResetResponse, EnvResponse, EvaluationResult, Observation,
)
from .tools import Search      # declares name="search"

@register_env("mybench")                  # tag written into save files
class MyBenchEnv(ActionableEnv):
    tool_registry = [Search]              # -> tool_schemas() for the Policy

    def reset(self, seed=None, options=None) -> EnvResetResponse:
        # The orchestrator passes the per-task identifier as `seed`; it indexes
        # into the benchmark's task library, it is not a randomness source.
        ...
        return EnvResetResponse(observation=self.observe(), info={})

    def step(self, action: Action) -> EnvResponse:
        # Dispatch on action.name -- the same names the Tools declare.
        if action.name == "search":
            result = self._search(**action.kwargs)
        else:
            result = {"error": f"unknown action {action.name!r}"}
        return EnvResponse(observation=self.observe(), reward=0.0,
                           terminated=self._done, truncated=False,
                           info={"result": result})   # by convention: info["result"]

    def observe(self) -> Observation: ...
    def evaluate(self) -> EvaluationResult: ...
    def get_env_state(self): ...          # data only -- NO runtime handles
    def save_state(self) -> dict: ...
    @classmethod
    def from_state(cls, state: dict): ...
```

Two contracts matter:

* **`get_env_state()` must carry data, never handles.** It is what the
  designer's generated hooks receive, and it crosses a subprocess boundary. A
  docker client or a browser page in there breaks both.
* **`save_state` / `from_state` are yours to define.** In-memory benchmarks can
  snapshot everything; for a container or a browser, store
  `{"reset_seed": ..., "reset_options": {...}}` and let `from_state` re-run
  `reset` — valid at episode boundaries, which is where checkpoints are taken.
  Override `close()` if there is external state to release; subprocess death
  does not free it for you.

### 2. Describe the state to the Environment Designer

`env_state_schema()` is injected verbatim into the designer's prompt. It is the
*only* thing telling it which fields its generated hooks may read, so be
explicit:

```python
    @classmethod
    def env_state_schema(cls) -> str:
        return ("MyBenchState = {\n"
                "  query: str,          # the task's question\n"
                "  hits: list[str],     # results of the last search\n"
                "  submitted: bool,\n"
                "}")
```

Optional hooks, all with safe defaults: `list_tasks()` (enables agent-driven
task selection), `notify_replay_complete()` (rewind per-episode counters after
a `Setup` replay), `default_reset_args()` / `reset_after_load()` (checkpoint
loading), `step_reward()` (dense per-step signal; non-fatal).

### 3. Point a corpus config at it

Corpus generation needs no new code — `scripts/run_harness.py` is
bridge-agnostic. Copy the closest existing `corpus.yaml` and change the import
path:

```yaml
env:
  import_path: envharness.bridges.mybench.bridge:MyBenchEnv
  reset_options: { ... }                 # forwarded to your reset()
policy:
  client_factory: envharness.infra.llm:LiteLLMClient
  client_kwargs: { model: openai/gpt-4.1-mini }
  action_format: function_calling        # or think_action for single-tool text
objective:
  type: difficulty_zone
  target_band: [0.4, 0.6]
```

Verify it boots before spending a run:

```bash
python -c "from envharness.bridges.mybench.bridge import MyBenchEnv; MyBenchEnv(); print('OK')"
python scripts/run_harness.py --config experiments/mybench/corpus.yaml --n-tasks 1
```

### 4. Reuse the downstream stages

Skill induction and evaluation are shared. `scripts/induce_pair.py` works
unchanged on per-task rollouts; write a benchmark-local `induce.py` only when the
induction prompt needs domain phrasing. For the evaluation, copy the driver
closest to your grading style — they differ only in how they load tasks and score
them. Then chain the stages in a `reproduce.py` mirroring an existing one.

### 5. Add a preflight target and a test

Add a `check_<bench>()` to `scripts/check_env.py` and register it in `CHECKS`,
so a missing dependency or dataset fails loudly before a run rather than
silently grading everything as failure. Then copy
`tests/test_actionable_env_toy24.py` — it walks the whole `ActionableEnv`
contract, including a `save_state` / `from_state` round-trip, and needs no GPU,
docker, or API key:

```bash
pytest
```

## Supported environment changes

- **Stage (`Setup`)**: `stage.in_env_actions` is replayed after reset and before
  the Policy sees its first observation; `stage.rationale` explains that change.
- **Contract (`Rules`)**: `contract.f_A`, `contract.f_T`, and `contract.f_O` are
  independent `{code, rationale}` objects defining `filter_action`,
  `modify_transition`, and `filter_observation`.

A Candidate may populate zero to four slots. The loader combines non-empty
Contract `code` values into native `rules_code`; every populated slot has its
own rationale, while an unused slot leaves both its payload and rationale empty.

## 🙏 Acknowledgements

We adopt the memory design of [**ReasoningBank**](https://arxiv.org/abs/2509.25140) in our agent implementation, and we are grateful for their work.

## 💬 Citation

If our work is useful for you, please consider citing our paper:

```
@article{huang2026envharness,
      title={EnvHarness: Awakening Static Worlds for Agent Learning}, 
      author={Chengsong Huang and Zifeng Wang and Rujun Han and Jun Yan and Yanfei Chen and Zoey CuiZhu and Ke Jiang and Peng Xia and Han Yu and Yufan Zhuang and Yifei Ming and Jiaqi Pan and Bhavana Dalvi Mishra and Jiaxin Huang and Burak Gokturk and Tomas Pfister and Chen-Yu Lee},
      year={2026},
      eprint={2608.19880},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.19880}, 
}
```

This is not an officially supported Google product. This project is not eligible for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
