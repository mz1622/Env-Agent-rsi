"""Offline regression tests for the Qiyuan single-condition driver (Linux)."""
import contextlib
import io
import json
import multiprocessing
import os
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from experiments.alfworld import ours_resume as resume
from experiments.alfworld import reproduce_gemini_proxy as proxy


SOURCE = Path(__file__).resolve().parents[1]
PROFILE = {"model": "openai/offline", "base": "http://127.0.0.1:9/v1", "revision": "v1"}
SPLIT = "eval_in_distribution"
OOD = "eval_out_of_distribution"


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def append(path, data):
    with path.open("a") as out:
        out.write(json.dumps(data) + "\n")


def vectors(texts, **kwargs):
    return [[float(len(t)), 0.5, 1.0] for t in texts]


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.base = Path(self.tmp)
        self.root = self.base / "run"
        self.enterContext(contextlib.chdir(SOURCE))
        self.enterContext(patch("socket.socket.connect", side_effect=AssertionError("unexpected network")))
        self.enterContext(patch.dict(os.environ, dict(
            GEMINI_API_KEY="offline-dummy", GEMINI_API_BASE="https://example.invalid/v1beta",
            EH_MODEL=proxy.CHAT, MODEL=proxy.CHAT, EH_EMBED_MODEL=PROFILE["model"],
            ALFWORLD_DATA=self.tmp, LITELLM_LOCAL_MODEL_COST_MAP="True",
            N_SHARDS="1", N_TASKS_TOTAL="3", TASK_BASE_OFFSET="0",
            EVAL_N_INDIST="3", EVAL_N_OOD="1", EVAL_START_SEEDS="0,100", EVAL_CONCURRENCY="1",
        )))

    def test_atomicity_and_lock(self):
        target = self.root / "saved.json"
        resume.put(target, ["old"])

        def truncated():
            yield "["
            raise InterruptedError()

        with self.assertRaises(InterruptedError):
            resume.atomic_write(target, truncated())
        self.assertEqual(resume.read(target), ["old"])
        with resume.run_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "Another run"):
                with resume.run_lock(self.root):
                    self.fail("duplicate writer")
        with resume.run_lock(self.root):
            pass
        with contextlib.redirect_stdout(io.StringIO()):
            resume.status(self.base / "absent")
        self.assertFalse((self.base / "absent").exists())

    def test_loss_of_owner(self):
        def work(i, stop):
            (self.base / "started").touch()
            deadline = time.monotonic() + 6
            while not (self.base / "finish").exists() and time.monotonic() < deadline:
                time.sleep(.02)
            resume.put(self.root / str(i), i)

        def owner():
            with resume.run_lock(self.root):
                resume.fork_jobs(work, [1, 2], 1)

        proc = multiprocessing.get_context("fork").Process(target=owner)
        proc.start()
        try:
            for _ in range(200):
                if (self.base / "started").exists():
                    break
                time.sleep(.015)
            self.assertTrue((self.base / "started").exists())
            proc.kill()
            proc.join(3)
            self.assertIsNotNone(proc.exitcode)
            with self.assertRaisesRegex(RuntimeError, "Another run"):
                with resume.run_lock(self.root):
                    pass
        finally:
            (self.base / "finish").touch()
            if proc.is_alive():
                proc.kill()
            proc.join(3)
        for _ in range(200):
            try:
                with resume.run_lock(self.root):
                    break
            except RuntimeError:
                time.sleep(.015)
        with resume.run_lock(self.root):
            self.assertEqual(resume.read(self.root / "1"), 1)
            self.assertFalse((self.root / "2").exists())

    def test_embedding_batch_resume(self):
        calls = []
        items = [dict(title=str(i), content="c", description="d", source={}) for i in range(13)]

        def interrupted(texts, **kwargs):
            calls.append(len(texts))
            if calls == [10, 3]:
                raise TimeoutError("interrupted")
            return vectors(texts)

        with patch("envharness.reasoning_bank.embed.embed_texts", interrupted):
            with self.assertRaises(TimeoutError):
                resume.make_bank(self.root, items, PROFILE, 3)
            self.assertFalse(list(self.root.rglob("ours_full.jsonl")))
            bank = resume.make_bank(self.root, items, PROFILE, 3)
            self.assertEqual(resume.make_bank(self.root, items, PROFILE, 3), bank)
        self.assertEqual(calls, [10, 3, 3])
        self.assertEqual([r["title"] for r in lines(bank)], [str(i) for i in range(13)])

    def test_hard_worker_exit_releases_lock(self):
        def crash(i, stop):
            if i == 1:
                os._exit(37)
            resume.put(self.root / str(i), i)

        with self.assertRaisesRegex(RuntimeError, "37"), resume.run_lock(self.root):
            resume.fork_jobs(crash, [0, 1, 2], 1)
        with resume.run_lock(self.root):
            self.assertEqual(resume.read(self.root / "0"), 0)
            self.assertFalse((self.root / "1").exists() or (self.root / "2").exists())

    def test_transport_error_and_empty_induction(self):
        from envharness.reasoning_bank import induce
        calls = [(induce.induce_memory_items, ("task", "trace", True)),
                 (induce.induce_paired_diff, ("task", "failed", "won"))]
        for method, args in calls:
            with self.subTest(method=method.__name__):
                with patch("litellm.completion", side_effect=RuntimeError("temporary offline error")):
                    self.assertEqual(method(*args, retries=1), [])
                    with self.assertRaises(RuntimeError):
                        method(*args, retries=1, raise_on_exhaustion=True)
                response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="NOTHING"))])
                with patch("litellm.completion", return_value=response):
                    self.assertEqual(method(*args, retries=1, raise_on_exhaustion=True), [])

    def test_failed_embedding_preflight_leaves_rollouts_unstarted(self):
        def unavailable():
            raise RuntimeError("embedding unavailable")

        with self.assertRaisesRegex(RuntimeError, "unfinished units"):
            resume.run(self.root, SOURCE, "eval", PROFILE, unavailable)
        self.assertFalse((self.root / "corpus").exists())
        self.assertFalse(list(self.root.glob("preflight/*.json")))
        legacy = self.base / "legacy"
        legacy.mkdir()
        (legacy / "traces.jsonl").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "no ours checkpoint"):
            resume.run(legacy, SOURCE, "corpus", None, None)
        self.assertEqual((legacy / "traces.jsonl").read_text(), "preserve")

    def test_evaluation_preserves_original_prompt_retrieval_and_score(self):
        from envharness.core.types import Observation
        from envharness.reasoning_bank import Bank
        from experiments.alfworld import reasoning_bank_eval as upstream
        requests = self.base / "requests.jsonl"
        obs = Observation(text="You are in a room.\nYour task is to: find a lamp.",
                          data={"admissible_commands": ["look"]})

        class Env:
            def reset(self, *, seed, options):
                return SimpleNamespace(observation=obs)

            def step(self, action):
                return SimpleNamespace(observation=obs, reward=1., info={"success": True},
                                       terminated=True, truncated=False)

            def close(self):
                pass

        def chat(**kwargs):
            append(requests, [kwargs["model"], kwargs["messages"], kwargs["temperature"], kwargs["max_tokens"]])
            msg = SimpleNamespace(content="<action>look</action>")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.enterContext(patch.object(upstream, "AlfworldEnv", Env))
        self.enterContext(patch.object(upstream, "completion_with_retry", chat))
        self.enterContext(patch("envharness.reasoning_bank.bank.embed_texts", vectors))
        self.enterContext(patch("envharness.reasoning_bank.embed.embed_texts", vectors))
        cfg = yaml.safe_load((SOURCE / "experiments/alfworld/reasoning_bank_eval.yaml").read_text())
        cfg["model"]["name"] = proxy.CHAT
        items = [dict(title=f"skill-{i}", description="d", content="c", source={}) for i in range(6)]
        bank = resume.make_bank(self.root, items, PROFILE, 3)
        expected = upstream._worker(0, cfg=cfg, bank=Bank.load(bank), split=SPLIT,
                                    gemini_api_keys=["offline-dummy"])
        result = resume.evaluate(self.root, cfg, bank, [0], {SPLIT: 1}, 1)
        actual = resume.read(result / "round1/episodes" / SPLIT / "0.json")
        self.assertTrue(actual["success"])
        self.assertEqual(len(actual["retrieved_titles"]), 5)
        self.assertEqual({k:v for k,v in actual.items() if k != "duration_ms"},
                         {k:v for k,v in expected.items() if k != "duration_ms"})
        prompts = lines(requests)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0], prompts[1])

    def test_local_embedding_route(self):
        import httpx
        seen = []

        def send(client, request, **kwargs):
            seen.append((str(request.url), request.headers["authorization"]))
            return httpx.Response(200, request=request, json=dict(object="list", model="offline",
                data=[dict(index=i, object="embedding", embedding=[1., 2.]) for i in range(2)],
                usage=dict(prompt_tokens=2, total_tokens=2)))

        with patch.dict(os.environ, {"EH_LOCAL_EMBED_API_BASE": PROFILE["base"],
                                     "OPENAI_API_KEY": "unrelated-cloud-key",
                                     "EH_LOCAL_EMBED_API_KEY": ""}), patch.object(httpx.Client, "send", send):
            settings = proxy.embedding_profile()
            self.assertEqual(proxy.check_embedding(settings), 2)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "offline-dummy")
        self.assertEqual(seen, [(PROFILE["base"] + "/embeddings", "Bearer local")])

    def test_resume_matches_upstream_and_uninterrupted(self):
        from envharness.core.types import Action, Candidate, DecideResult, Decision, Observation, Step, Trace
        from scripts import run_harness
        original = run_harness.build_from_config
        cfg = yaml.safe_load((SOURCE / "experiments/alfworld/corpus.yaml").read_text())
        cfg["orchestrator"]["rollout_concurrency"] = 1
        cfg_path = self.base / "corpus.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        os.environ["CORPUS_YAML"] = str(cfg_path)
        journal = self.base / "calls.jsonl"
        failure = self.base / "break"

        def failing(stage):
            return failure.exists() and failure.read_text() == stage

        class Runner:
            def __init__(self):
                self.count = Counter()

            def run(self, spec):
                seed = spec.env.reset_seed
                stage = "mut" if spec.candidate.rules_code else "base"
                counter_key = seed, stage
                k = self.count[counter_key]
                self.count[counter_key] += 1
                key = f"{seed}-{stage}-{k}"
                append(journal, ["roll", key])
                command = Action(name="do", kwargs={"text": "look"})
                step = Step(raw_action=command, filtered_action=command,
                            filtered_observation=Observation(text=f"Task: {seed}-{stage}"))
                return Trace(episode_id=key, iteration_id=spec.iteration_id, task_id=spec.task_id,
                             rollout_seed=seed, candidate=spec.candidate, steps=[step], duration_steps=1,
                             success=k % 2 == 0,
                             error="policy.act raised: quota" if seed == 1 and stage == "mut" and failing("corpus") else None)

        class Designer:
            def propose(self, ctx):
                append(journal, ["history", ctx.task_id,
                                  [[t.episode_id, t.candidate_id] for t in ctx.history_traces]])
                return Candidate(rules_code="class _Rules: pass" if ctx.task_id % 2 else "")

            def decide(self, candidate, traces, ctx):
                return DecideResult(decision=Decision.ACCEPT)

        def build(*args, **kwargs):
            orch = original(*args, **kwargs)
            self.assertEqual(orch.policy_spec.client_kwargs["inner_kwargs"]["model"], proxy.CHAT)
            orch.runner, orch.harness_agent = Runner(), Designer()
            return orch

        def induction(task_query, **kwargs):
            append(journal, ["induce", task_query])
            if task_query == "1-mut" and failing("skills"):
                raise RuntimeError("quota")
            if task_query == "2-base":
                return []
            return [dict(title=task_query, description="d", content="c")]

        def episode(seed, *, cfg, bank, split, gemini_api_keys):
            self.assertEqual(set(cfg["conditions"]), {"ours"})
            self.assertEqual(cfg["model"], dict(name=proxy.CHAT, temperature=.4, max_response_tokens=2048))
            self.assertEqual([it.title for it in bank.items], ["0-base", "1-mut"])
            append(journal, ["eval", split, seed])
            return dict(seed=seed, split=split, success=seed % 2 == 0,
                        error="quota" if split == SPLIT and seed == 2 and failing("eval") else "")

        def probe():
            append(journal, ["probe"])
            return 3

        self.enterContext(patch.object(run_harness, "build_from_config", build))
        self.enterContext(patch("scripts.induce_pair.induce_paired_diff", induction))
        self.enterContext(patch("envharness.reasoning_bank.embed.embed_texts", vectors))
        self.enterContext(patch("experiments.alfworld.reasoning_bank_eval._worker", episode))

        oracle_path = self.base / "oracle"
        oracle = build(cfg_path, os.path.relpath(oracle_path, SOURCE / "runs"),
                       overrides=dict(n_tasks=3, n_iterations=3, task_id_base_offset=0))
        oracle.run()
        def semantic(rows):
            return [{k: v for k, v in t.items() if k not in ("iteration_id", "candidate_id")}
                    for t in rows]
        oracle_traces = semantic(lines(oracle_path / "traces.jsonl"))
        reference = self.base / "complete"
        ref_eval = resume.run(reference, SOURCE, "eval", PROFILE, probe)
        journal.unlink()

        failure.write_text("corpus")
        with self.assertRaises(RuntimeError):
            resume.run(self.root, SOURCE, "skills", None, None)
        self.assertEqual([p.stem for p in self.root.glob("corpus/shard0/tasks/*.jsonl")], ["0"])
        self.assertEqual(len(lines(self.root / "corpus/shard0/pending_baselines/1.jsonl")), 5)
        self.assertNotIn(["probe"], lines(journal))
        failure.write_text("skills")
        with self.assertRaises(RuntimeError):
            resume.run(self.root, SOURCE, "skills", None, None)
        self.assertEqual(sorted(p.stem for p in self.root.glob("skills/ours/*.json")), ["0", "2"])
        self.assertEqual(resume.read(self.root / "skills/ours/2.json"), [])
        snapshots = [r for r in lines(journal) if r[:2] == ["history", 2]]
        stored = [t for seed in (0, 1) for t in lines(self.root / f"corpus/shard0/tasks/{seed}.jsonl")]
        self.assertEqual(snapshots[0][2], [[t["episode_id"], t["candidate_id"]] for t in stored])
        failure.write_text("eval")
        with self.assertRaises(RuntimeError):
            resume.run(self.root, SOURCE, "eval", PROFILE, probe)
        folder = next(self.root.glob("eval/*/round1/episodes/" + SPLIT))
        self.assertEqual(sorted(p.stem for p in folder.glob("*.json")), ["0", "1"])
        self.assertFalse(resume.read(folder / "1.json")["success"])
        self.assertFalse(list(self.root.glob("eval/*/summary.json")))
        failure.unlink()
        evaluation = resume.run(self.root, SOURCE, "eval", PROFILE, probe)
        self.assertEqual(semantic(lines(self.root / "corpus/traces.jsonl")), oracle_traces)
        self.assertEqual(resume.read(evaluation / "summary.json"), resume.read(ref_eval / "summary.json"))
        for output in ref_eval.rglob("ours_*.jsonl"):
            self.assertEqual(lines(output), lines(evaluation / output.relative_to(ref_eval)))
        bank = self.root / resume.read(self.root / "banks/current.json")
        self.assertEqual(lines(bank), lines(reference / resume.read(reference / "banks/current.json")))
        calls = lines(journal)
        self.assertEqual(calls.count(["eval", SPLIT, 1]), 1)
        self.assertEqual(calls.count(["eval", SPLIT, 2]), 2)
        self.assertEqual(calls.count(["roll", "0-base-0"]), 1)
        self.assertEqual(calls.count(["roll", "1-base-0"]), 1)
        self.assertEqual(calls.count(["roll", "1-mut-0"]), 2)
        self.assertEqual(calls.count(["induce", "0-base"]), 1)
        self.assertEqual(calls.count(["induce", "2-base"]), 1)
        self.assertEqual(calls.count(["probe"]), 1)
        resume.run(self.root, SOURCE, "eval", PROFILE, probe)
        self.assertEqual(lines(journal), calls)
        resume.run(self.root, SOURCE, "bank", dict(PROFILE, revision="v2"), probe)
        self.assertEqual(lines(journal), calls + [["probe"]])
        self.assertEqual(len(list(self.root.glob("banks/*/ours_full.jsonl"))), 2)
        self.assertFalse(list(self.root.rglob("orig_*.jsonl")) or list(self.root.rglob("nobank_*.jsonl")))
        with patch.dict(os.environ, {"N_SHARDS": "2"}), self.assertRaisesRegex(ValueError, "shards"):
            resume.run(self.root, SOURCE, "skills", None, None)
        self.assertEqual([list(g) for g in resume.partition(5, 10, 2)], [[10, 11, 12], [13, 14]])
        from scripts.induce_pair import _build_bank
        grouped = {}
        for seed in (0, 1, 2):
            traces = [t for t in oracle_traces if t["rollout_seed"] == seed]
            grouped[str(seed)] = [t for t in traces if t["kind"] == "accepted"] or traces
        official_bank = self.base / "official.jsonl"
        with patch("scripts.induce_pair.embed_texts", vectors):
            _build_bank(condition="ours", traces_by_task=grouped, llm_model=proxy.CHAT,
                        concurrency=1, embed_model=PROFILE["model"], out_path=official_bank)
        self.assertEqual(lines(bank), lines(official_bank))
