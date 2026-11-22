"""Task-level checkpoints for the ALFWorld EnvHarness-skills condition."""
from __future__ import annotations

import concurrent.futures as futures
import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import sys
import tempfile
from functools import partial
from importlib.metadata import version
from itertools import chain, islice
from multiprocessing.connection import wait as wait_processes
from pathlib import Path

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, parts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as f:
        temp = Path(f.name)
        try:
            f.writelines(parts)
            f.flush()
            os.fsync(f.fileno())
            f.close()
            os.replace(temp, path)
            fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temp.unlink(missing_ok=True)


def put(path, value):
    atomic_write(path, [json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"])


def manifest(path, settings):
    if path.exists():
        previous = read(path)
        if previous != settings:
            changed = sorted(k for k in previous.keys() | settings.keys()
                             if previous.get(k) != settings.get(k))
            raise ValueError(f"Run settings changed ({', '.join(changed)}). "
                             f"Use the original settings or a new ROOT_RUN. See {path}")
    else:
        put(path, settings)


@contextlib.contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another run is active in {root}. Use status.") from None
        # Closing the descriptor releases the lock. Forked evaluation workers
        # retain it if the coordinator dies while they finish their current job.
        yield


def run_bounded(pool, worker, jobs, limit, stop=None):
    """At most limit in flight. Workers commit their own completed units."""
    jobs = iter(jobs)
    live = {pool.submit(worker, job) for job in islice(jobs, limit)}
    try:
        while live:
            done, live = futures.wait(live, return_when=futures.FIRST_COMPLETED)
            for job in done:
                job.result()
            if stop is None or not stop.is_set():
                live.update(pool.submit(worker, item) for item in islice(jobs, len(done)))
    finally:
        for job in live:
            job.cancel()


class StopFlag:
    def __init__(self, flag, parent):
        self.flag = flag
        self.parent = parent

    def is_set(self):
        return bool(self.flag.value) or os.getppid() != self.parent

    def set(self):
        self.flag.value = 1


def fork_jobs(worker, jobs, limit):
    """Qiyuan/Linux. A child holds the lock through its current unit, then
    exits on parent loss. No persistent executor waits for a dead parent."""
    jobs = list(jobs)
    if not jobs:
        return
    ctx = multiprocessing.get_context("fork")
    cursor = ctx.RawValue("i", 0)
    cursor_lock = ctx.Lock()
    # Every writer sets the same one-way byte. A hard-killed worker cannot
    # strand the stop signal behind a process-held lock.
    stop = StopFlag(ctx.RawValue("B", 0), os.getpid())

    def work():
        try:
            while not stop.is_set():
                if not cursor_lock.acquire(timeout=0.25):
                    continue  # recheck the stop flag if the lock holder died
                try:
                    index = cursor.value
                    if index >= len(jobs):
                        return
                    cursor.value += 1
                finally:
                    cursor_lock.release()
                worker(jobs[index], stop)
        except Exception as exc:
            stop.set()
            message = str(exc)
            for name in ("GEMINI_API_KEY", "EH_GEMINI_EMBED_API_KEY", "EH_LOCAL_EMBED_API_KEY", "OPENAI_API_KEY"):
                value = os.getenv(name)
                if value and value != "local":
                    message = message.replace(value, "[redacted]")
            print(f"{type(exc).__name__}: {message[:700]}", file=sys.stderr, flush=True)
            raise SystemExit(1) from None
        except BaseException:
            stop.set()
            raise

    processes = []
    try:
        for _ in range(min(limit, len(jobs))):
            p = ctx.Process(target=work)
            p.start()
            processes.append(p)
        remaining = {p.sentinel: p for p in processes}
        while remaining:
            for handle in wait_processes(remaining):
                child = remaining.pop(handle)
                child.join()
                if child.exitcode:
                    raise RuntimeError(f"Worker exited with code {child.exitcode}; unfinished units remain pending.")
    finally:
        stop.set()
        for p in processes:
            p.join()
            p.close()


def partition(count, offset, shards):
    if min(count, shards) < 1:
        raise ValueError("N_TASKS_TOTAL and N_SHARDS must be positive")
    q, r = divmod(count, shards)
    groups = []
    for i in range(min(shards, count)):
        size = q + int(i < r)
        groups.append(range(offset, offset + size))
        offset += size
    return groups


class ShardHistory(list):
    """The pinned orchestrator reads a snapshot before it adds baselines."""
    add = list.append
    all = list.copy


class TaskLog:
    def __init__(self, inner, stop):
        self.inner = inner
        self.log_dir = inner.log_dir
        self.stop = stop

    def event(self, kind, **fields):
        self.inner.event(kind, **fields)
        error = fields.get("error") or ""
        # A generated rule can legitimately fail. Keep those traces for the
        # designer. Provider and worker failures must leave the task pending.
        invalid_rollout = (error.startswith(("policy.act raised:", "subprocess "))
                           or (fields.get("candidate_id") == "baseline" and bool(error)))
        if kind == "task_aborted" or (kind == "rollout_done" and invalid_rollout):
            self.stop.set()
            raise RuntimeError(f"Unfinished training task ({kind}): {error[:240]}")
        if self.stop.is_set():
            raise InterruptedError("Corpus collection stopped")


def keep_baseline_batch(orch, directory):
    """Reuse all K original rollouts if an interruption occurred during mutation."""
    from envharness.core.types import Trace

    collect_original = orch._rollout_baseline_k

    def collect_or_load(task_idx, task_id):
        path = directory / f"{task_idx}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                traces = [Trace.model_validate_json(line) for line in f]
        else:
            traces = collect_original(task_idx, task_id)
        if ([t.rollout_idx for t in traces] != list(range(orch.config.k_per_candidate))
                or any(t.error or t.rollout_seed != task_id or t.candidate_id != "baseline"
                       for t in traces)):
            raise ValueError(f"Incomplete baseline batch for task {task_id}")
        if not path.is_file():
            atomic_write(path, (t.model_dump_json() + "\n" for t in traces))
        return traces

    orch._rollout_baseline_k = collect_or_load


def collect_shard(source, cfg, corpus_dir, indexed_group, stop):
    from envharness.core.types import Trace
    from scripts.run_harness import build_from_config

    shard, seeds = indexed_group
    directory = corpus_dir / f"shard{shard}"
    paths = [directory / "tasks" / f"{s}.jsonl" for s in seeds]
    try:
        first = next((i for i, path in enumerate(paths) if not path.is_file()), len(paths))
        if any(p.exists() for p in paths[first:]):
            raise ValueError(f"Missing earlier task in serial shard {shard}: {paths[first]}")
        if first == len(paths):
            return
        name = os.path.relpath(directory, source / "runs")
        orch = build_from_config(cfg, name, overrides={
            "n_tasks": len(seeds), "n_iterations": len(seeds), "task_id_base_offset": seeds.start,
        })
        history = ShardHistory()
        orch.trace_store = history
        orch.log = TaskLog(orch.log, stop)
        baselines = directory / "pending_baselines"
        keep_baseline_batch(orch, baselines)
        orch._preflight_check()
        for index, path in enumerate(paths):
            if stop.is_set():
                return
            if index < first:
                with path.open(encoding="utf-8") as f:
                    history.extend(Trace.model_validate_json(line) for line in f)
                continue
            before = len(history)
            # _iterate appends the K baseline traces and returns all candidate
            # traces. Keep both, in the same order as Orchestrator.run().
            history.extend(orch._iterate(index))
            traces = history[before:]
            expected = orch.config.k_per_candidate
            baseline = [t.rollout_idx for t in traces if t.kind == "baseline"]
            task_seed = orch.config.base_seed + index * orch.config.task_id_stride + seeds.start
            if sorted(baseline) != list(range(expected)) or any(t.rollout_seed != task_seed for t in traces):
                raise ValueError(f"Incomplete or inconsistent rollouts for task {seeds[index]}")
            atomic_write(path, (t.model_dump_json() + "\n" for t in traces))
            (baselines / f"{index}.jsonl").unlink(missing_ok=True)
    except BaseException:
        stop.set()
        raise


def corpus_files(corpus_dir, groups):
    return [corpus_dir / f"shard{i}" / "tasks" / f"{seed}.jsonl"
            for i, seeds in enumerate(groups) for seed in seeds]


def collect(source, cfg, corpus_dir, groups):
    fork_jobs(partial(collect_shard, source, cfg, corpus_dir), enumerate(groups), len(groups))
    paths = corpus_files(corpus_dir, groups)
    combined = corpus_dir / "traces.jsonl"
    if not combined.is_file():
        atomic_write(combined, (p.read_text(encoding="utf-8") for p in paths))
    return paths


def induce_task(skills_dir, chat, trace_path):
    from scripts.induce_pair import _items_for_task, _pick_pair

    task = trace_path.stem
    output = skills_dir / f"{task}.json"
    with trace_path.open(encoding="utf-8") as f:
        traces = [json.loads(line) for line in f]
    accepted = [t for t in traces if t["kind"] == "accepted"]
    selected = accepted or [t for t in traces if t["kind"] == "baseline"]
    good, bad = _pick_pair(selected)
    items = _items_for_task(str(selected[0]["rollout_seed"]), "ours", good, bad, chat,
                            raise_on_exhaustion=True)
    put(output, items)  # [] is a completed model response, never a transport error.


def make_skills(paths, skills_dir, chat, stop=None):
    missing = [p for p in paths if not (skills_dir / (p.stem + ".json")).is_file()]
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        run_bounded(pool, partial(induce_task, skills_dir, chat), missing, 8, stop)
    if stop is not None and stop.is_set():
        raise InterruptedError("Skill induction stopped")
    return list(chain.from_iterable(read(skills_dir / (p.stem + ".json")) for p in paths))


def valid_vectors(vectors, count, dimension):
    return len(vectors) == count and all(len(v) == dimension and
                                         all(math.isfinite(x) for x in v) for v in vectors)


def make_bank(root, items, profile, dimension, stop=None):
    from envharness.reasoning_bank.embed import EMBED_BATCH, embed_texts

    if not items:
        raise ValueError("The full ours bank has no skills")
    settings = {"embedding": profile, "dimension": dimension, "skills_sha256": digest(items)}
    directory = root / "banks" / digest(settings)[:20]
    manifest(directory / "settings.json", settings)
    bank_path = directory / "ours_full.jsonl"
    if not bank_path.exists():
        def rows():
            for start in range(0, len(items), EMBED_BATCH):
                batch = items[start:start + EMBED_BATCH]
                cache = directory / "chunks" / f"{start}.json"
                if cache.is_file():
                    vectors = read(cache)
                else:
                    if stop is not None and stop.is_set():
                        raise InterruptedError("Embedding stopped")
                    texts = [f"## {it['title']}\n{it['description']}\n{it['content']}" for it in batch]
                    vectors = embed_texts(texts, model=profile["model"])
                    if not valid_vectors(vectors, len(batch), dimension):
                        raise ValueError("Invalid embedding response; batch left pending")
                    put(cache, vectors)
                if not valid_vectors(vectors, len(batch), dimension):
                    raise ValueError(f"Invalid cached vectors in {cache}")
                for item, vector in zip(batch, vectors):
                    yield json.dumps(dict(item, embedding=vector), ensure_ascii=False) + "\n"
        atomic_write(bank_path, rows())
    return bank_path


def evaluate_seed(cfg, bank, split, folder, seed, stop):
    from experiments.alfworld.reasoning_bank_eval import _worker
    from envharness.infra.model import key_pool

    rec = _worker(seed, cfg=cfg, bank=bank, split=split,
                  gemini_api_keys=key_pool(cfg["model"]["name"]))
    if rec.get("error"):
        raise RuntimeError(f"Unfinished eval episode {split}/{seed}: {rec['error'][:240]}")
    if rec["seed"] != seed or rec["split"] != split or type(rec["success"]) is not bool:
        raise ValueError(f"Invalid episode record for {split}/{seed}")
    put(folder / f"{seed}.json", rec)


def saved_episode(folder, split, seed):
    path = folder / f"{seed}.json"
    try:
        record = read(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if (isinstance(record, dict) and record.get("seed") == seed and record.get("split") == split
            and type(record.get("success")) is bool and record.get("error") == ""):
        return record
    return None


def evaluate(root, cfg, bank_path, starts, splits, workers):
    from envharness.reasoning_bank import Bank

    from experiments.alfworld import reasoning_bank_eval
    settings = {"condition": "ours", "config": cfg, "bank_profile": bank_path.parent.name,
                "evaluator_sha256": file_hash(Path(reasoning_bank_eval.__file__)),
                "bank_sha256": file_hash(bank_path), "start_seeds": starts, "splits": splits}
    directory = root / "eval" / digest(settings)[:20]
    manifest(directory / "settings.json", settings)
    bank = Bank.load(bank_path)
    rows = []
    for idx, start in enumerate(starts, 1):
        round_dir = directory / f"round{idx}"
        for split, size in splits.items():
            folder = round_dir / "episodes" / split
            seeds = range(start, start + size)
            missing = (s for s in seeds if saved_episode(folder, split, s) is None)
            fork_jobs(partial(evaluate_seed, cfg, bank, split, folder), missing, workers)
            records = [read(folder / f"{s}.json") for s in seeds]
            wins = sum(rec["success"] for rec in records)
            atomic_write(round_dir / f"ours_{split}.jsonl", (json.dumps(r) + "\n" for r in records))
            rows.append(dict(condition="ours", split=split, round=idx, start_seed=start,
                             n_won=wins, n=size, sr=wins / max(1, size)))
    put(directory / "summary.json", dict(rows=rows, start_seeds=starts, complete=True))
    print(f"Evaluation complete: {directory / 'summary.json'}", flush=True)
    return directory


def source_signature(source):
    # Fingerprint the code actually used to create the task histories and
    # skills. Database files, logs, credentials and Python caches are excluded.
    files = sorted((source / "envharness").rglob("*.py"))
    files += [source / "scripts/run_harness.py", source / "scripts/induce_pair.py"]
    return digest({str(p.relative_to(source)): file_hash(p) for p in files})


def run(root, source, through, profile, embedding_check):
    root = root.resolve()
    cfg = (source / os.getenv("CORPUS_YAML", "experiments/alfworld/corpus.yaml")).resolve()
    eval_path = (source / os.getenv("EVAL_YAML", "experiments/alfworld/reasoning_bank_eval.yaml")).resolve()
    count = int(os.getenv("N_TASKS_TOTAL") or 100)
    offset = int(os.getenv("TASK_BASE_OFFSET") or 0)
    shards = int(os.getenv("N_SHARDS") or 4)
    groups = partition(count, offset, shards)
    chat = os.environ["EH_MODEL"]
    settings = dict(format=1, condition="ours", revision="fab7d57441f06b75c73a900e04561d4d7600f361",
                    source_sha256=source_signature(source), corpus_sha256=file_hash(cfg),
                    driver_sha256=file_hash(Path(__file__)),
                    tasks=count, offset=offset, shards=shards, model=chat,
                    chat_base=os.environ["GEMINI_API_BASE"], data=str(Path(os.environ["ALFWORLD_DATA"]).resolve()),
                    packages={p: version(p) for p in ("litellm", "alfworld", "textworld")},
                    python=list(sys.version_info[:3]))
    custom = os.environ.get("ALFWORLD_CONFIG")
    default_config = source / "envharness/third_party/alfworld/base_config.yaml"
    settings["alfworld_config_sha256"] = file_hash(Path(custom) if custom else default_config)
    with run_lock(root):
        marker = root / "protocol.json"
        if not marker.exists() and any(p.name != ".run.lock" and not p.name.startswith(".pending-")
                                       for p in root.iterdir()):
            raise ValueError("Existing directory has no ours checkpoint manifest; use a new ROOT_RUN.")
        manifest(marker, settings)
        dimension = None
        if through in ("bank", "eval"):
            verified = root / "preflight" / (digest(profile) + ".json")
            if not verified.is_file():
                # Keep live HTTP clients out of the parent that forks workers.
                def probe(_, stop):
                    put(verified, {"dimension": embedding_check()})
                fork_jobs(probe, [None], 1)
            dimension = read(verified)["dimension"]
        print(f"Run directory: {root}", flush=True)
        paths = collect(source, cfg, root / "corpus", groups)
        if through == "corpus":
            return
        def prepare_bank(_, stop):
            items = make_skills(paths, root / "skills/ours", chat, stop)
            if through != "skills":
                path = make_bank(root, items, profile, dimension, stop)
                put(root / "banks" / "current.json", str(path.relative_to(root)))

        fork_jobs(prepare_bank, [None], 1)
        if through == "skills":
            return
        bank = root / read(root / "banks" / "current.json")
        print(f"Full ours bank: {bank}", flush=True)
        if through == "bank":
            return
        eval_cfg = yaml.safe_load(eval_path.read_text())
        eval_cfg["conditions"] = {"ours": eval_cfg["conditions"]["ours"]}
        eval_cfg["model"]["name"] = chat
        starts = [int(x) for x in (os.getenv("EVAL_START_SEEDS") or "0,1000,2000").split(",")]
        splits = dict(eval_cfg["eval"]["splits"])
        for split, name in [("eval_in_distribution", "EVAL_N_INDIST"), ("eval_out_of_distribution", "EVAL_N_OOD")]:
            if os.getenv(name):
                splits[split] = int(os.environ[name])
        workers = int(os.getenv("EVAL_CONCURRENCY") or 24)
        if workers < 1 or any(n < 0 for n in splits.values()):
            raise ValueError("Invalid eval concurrency or split size")
        return evaluate(root, eval_cfg, bank, starts, splits, workers)


def status(root):
    protocol = root / "protocol.json"
    if not protocol.exists():
        print(f"No checkpointed ours run at {root}")
        return
    with (root / ".run.lock").open() as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            active = True
    n = read(protocol)["tasks"]
    print(f"{root} ({'active' if active else 'idle'})")
    print(f"Corpus tasks: {sum(1 for _ in root.glob('corpus/shard*/tasks/*.jsonl'))}/{n}")
    print(f"Skill tasks: {sum(1 for _ in root.glob('skills/ours/*.json'))}/{n}")
    for path in sorted(root.glob("banks/*/ours_full.jsonl")):
        print(f"Full bank: {path}")
    for meta in sorted(root.glob("eval/*/settings.json")):
        cfg = read(meta)
        total = len(cfg["start_seeds"]) * sum(cfg["splits"].values())
        saved = sum(1 for _ in meta.parent.glob("round*/episodes/*/*.json"))
        print(f"Evaluation {meta.parent.name}: {saved}/{total} episodes")
