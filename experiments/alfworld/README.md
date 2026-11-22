# EnvHarness on ALFWorld

## Resumable Gemini skills experiment

This branch packages the EnvHarness `ours` condition used on Qiyuan. It collects
100 training tasks over four shards, induces skills from the resulting trajectories
and builds an embedding index. Each validation episode retrieves five skills by
MMR. Evaluation covers 140 seen and 134 unseen games at each of three starting
seeds (0, 1000 and 2000).

From the repository root on macOS or Linux, use Python 3.12 and the text-only
ALFWorld backend:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r experiments/alfworld/requirements-gemini.txt -e .
```

Set `ALFWORLD_DATA` to an existing ALFWorld data directory or use
`.venv/bin/alfworld-download` to populate the default `~/.cache/alfworld`.
Supply `GEMINI_API_KEY` in the process environment. Set `GEMINI_API_BASE` to
the Gemini-compatible base URL for a proxy-issued key. The default is Google's
Gemini API. The policy and skill induction use `gemini-3.1-flash-lite`.
Both skill indexing and task retrieval use local Qwen3-Embedding-0.6B
(FP16, 1024 dimensions). The policy receives the retrieved skill text.

Start Ollama on this machine (`ollama serve` if it is not already running),
then pull the model and start the experiment in another shell.

```bash
ollama pull qwen3-embedding:0.6b-fp16
.venv/bin/python experiments/alfworld/reproduce_gemini_proxy.py check
.venv/bin/python experiments/alfworld/reproduce_gemini_proxy.py run
.venv/bin/python experiments/alfworld/reproduce_gemini_proxy.py status
```

The default output is `runs/alfworld_gemini_ours/`. Repeat the same run command
to resume. The default Ollama endpoint is `http://127.0.0.1:11434/v1`.
The index records the installed model digest. Set `EH_EMBED_REVISION` to an
expected digest to verify it before resuming. The FP16 manifest used for this
run was `sha256:67a7592a8852b7b36f6328cdec193cfb7639769f5edf344ba709b088b2a51042`.
For Gemini embeddings set
`EH_EMBED_MODEL=gemini/gemini-embedding-001`. For a separate Gemini embedding
service also set `EH_GEMINI_EMBED_API_BASE` and `EH_GEMINI_EMBED_API_KEY`. A local
OpenAI-compatible embedding service uses `EH_EMBED_MODEL=openai/<served-name>`
and `EH_LOCAL_EMBED_API_BASE=http://127.0.0.1:<port>/v1`. To collect trajectories
and induce skills before an embedding service is available, use
`run --through skills`, then resume with `run`.

## Environment

ALFWorld pins to Python 3.12 (3.13 hits a latent TextWorld bug).

```bash
conda create -n eh-alfworld python=3.12 && conda activate eh-alfworld
pip install -e .                       # pydantic, pyyaml, litellm

# ALFWorld runtime. On a clean 3.12 env the plain install fails: the
# transitive dependency `visdom` imports pkg_resources at build time, which
# setuptools >= 81 no longer ships. Constrain the build env:
printf 'setuptools<81\n' > /tmp/eh_constraints.txt
PIP_CONSTRAINT=/tmp/eh_constraints.txt pip install "alfworld[full]" textworld==1.7.0

# Game data (~2.3 GB). Honours $ALFWORLD_DATA, else ~/.cache/alfworld.
export ALFWORLD_DATA=~/eh_alfworld_data
alfworld-download

# Credentials for whatever MODEL names (default: openai/gpt-4.1-mini)
export OPENAI_API_KEY='...'             # GPT
# export GEMINI_API_KEYS=k1,k2,...      # Gemini: a pool spreads its per-key quota
# Claude on Vertex needs ADC + GOOGLE_CLOUD_PROJECT, no key

python scripts/check_env.py alfworld    # preflight (python, imports, data)
```

Check the bridge loads:

```bash
python -c "from envharness.bridges.alfworld import AlfworldEnv; AlfworldEnv(); print('OK')"
```

## Run

```bash
# Reduced task count; the values it sets are at the top of the script.
bash experiments/alfworld/reproduce_smoke.sh

# Full run: corpus -> induce -> subset -> eval. Every shard and the eval
# run with the interpreter you launch this with.
python experiments/alfworld/reproduce.py
```

Shards round-robin whatever key pool the provider has; more Gemini keys
spread that per-key quota, and one key (or none, on Vertex) simply shares.

Output lands under `runs/alfworld_headline_<timestamp>/`, with the eval
summary at `$ROOT_RUN/eval/summary.json`:

```json
{
  "rows": [
    {"condition": "<nobank|orig|ours>", "split": "eval_in_distribution",
     "n_won": 0, "n": 140, "sr": 0.0, "start_seed": 0, "round": 1},
    {"condition": "<nobank|orig|ours>", "split": "eval_out_of_distribution",
     "n_won": 0, "n": 134, "sr": 0.0, "start_seed": 0, "round": 1}
  ]
}
```

### Knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `openai/gpt-4.1-mini` | model for **all** stages — corpus policy, harness agent, induction, eval. Any provider (`openai/…`, `vertex_ai/claude-…`, `gemini/…`) |
| `<PROVIDER>_API_KEY(S)` | per `MODEL` | `$OPENAI_API_KEY`, `$GEMINI_API_KEYS`, or ADC on Vertex. Shards round-robin the pool; key 0 induces, the last one evals |
| `PY` | the launching interpreter | interpreter for every stage |
| `N_TASKS_TOTAL` | `100` | corpus size |
| `N_SHARDS` | `4` | parallel corpus shards |
| `TASK_BASE_OFFSET` | `0` | first task_id |
| `EVAL_START_SEEDS` | `0,1000,2000` | one eval round per start seed |
| `EVAL_CONCURRENCY` | `24` | inner parallelism of `reasoning_bank_eval.py` |
| `EVAL_N_INDIST` / `EVAL_N_OOD` | all (140 / 134) | cap eval tasks per split |
| `ROOT_RUN` | `runs/alfworld_headline_<ts>` | output dir |
| `SKIP_CORPUS=1` | — | reuse `$ROOT_RUN/corpus/traces.jsonl` |
| `SKIP_INDUCE=1` | — | reuse `$ROOT_RUN/banks/*_full.jsonl` |
| `SKIP_SUBSET=1` | — | reuse `$ROOT_RUN/banks/*_subset*.jsonl` |
| `SKIP_EVAL=1` | — | stop after Stage 3 |

## Files

| File | Purpose |
|---|---|
| `reproduce.py` | one-command entry point (chains stages 1-4) |
| `reproduce_smoke.sh` | the same four stages at a reduced task count |
| `corpus.yaml` | Stage 1 Orchestrator + Harness Agent config (per-shard) |
| `reasoning_bank_eval.py` | Stage 4 eval driver (`AlfworldEnv` + Bank + LiteLLM) |
| `reasoning_bank_eval.yaml` | Stage 4 eval config (model, retrieval top-K, splits) |
| `../../scripts/run_harness.py` | bridge-agnostic Stage 1 worker (one shard) |
| `../../scripts/induce_pair.py` | Stage 2 induction |
| `../../scripts/subset.py` | Stage 3 1-per-task subset |
| `../../envharness/bridges/alfworld/` | the ALFWorld Bridge (`AlfworldEnv`) |
| `../../envharness/prompts/alfworld_skill_prompt.py` | skill prompt + retrieval helpers used by `reasoning_bank_eval.py` |
