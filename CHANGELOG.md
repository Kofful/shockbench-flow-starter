# Changelog

## 0.1.1 (2026-09-30)

- shockbench-flow 0.1.1: a public scoring API
  (`shockbench_flow_agent.EpisodeSet`: `build`, `score`, `compare`),
  plain-language reports (the score, its interval, costs in USD), and a loader
  that keeps each submission's modules to itself.
- The kit: your work lives in `agents/` and `my/`, which updates never touch
  (README, "Who owns what"); `make update` pulls a new wheel and resolves a
  `uv.lock` conflict; `sbf compare` (a paired interval); `sbf check` runs the
  agent in a process holding only the scoring image's packages and fails an
  import the server lacks in any module; `sbf runs`; every tool defaults to Tiny
  and `--quick` means the same everywhere; example 08 is `sbf pack`, and example
  02 is now `02_play_agents.py`.

## 0.1.0 (2026-09-30)

- shockbench-flow 0.1.0: the first release of the starter kit (environments
  `ShockBench/Tiny-v0`, `Small-v0` and `Full-v0`, the agent kit, the local
  evaluation), vendored as `vendor/shockbench_flow-0.1.0-py3-none-any.whl`.
