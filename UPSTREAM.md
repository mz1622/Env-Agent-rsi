# Upstream provenance

This repository is based directly on
[google-research/envharness](https://github.com/google-research/envharness).

- Imported branch: `main`
- Imported commit: `fab7d57441f06b75c73a900e04561d4d7600f361`
- Imported on: 2026-10-06
- Upstream remote: `https://github.com/google-research/envharness.git`

The local changes intentionally keep the EnvHarness architecture and add a
split-model runtime profile:

- Policy/Target: local Ollama `qwen3:4b-direct`
- HarnessAgent/Mutator: DeepSeek API `deepseek-flash`
- Role-specific CLI overrides for policy and environment-rigging models
- DeepSeek credentials from an environment variable or ignored local key file
- Environment search exposes four independent Candidate slots: Stage (`Setup`)
  plus Contract f_A/f_T/f_O (`Rules` A/T/O), each with its own rationale

The original Apache-2.0 license and upstream copyright notices are retained.
