# Changelog

## Unreleased

A simpler kit, on shockbench-flow 0.1.2 from PyPI.

- shockbench-flow comes from PyPI, `>= 0.1.2`, its exact version locked in
  `uv.lock`: `vendor/` and its wheel are gone, and
  `uv sync --upgrade-package shockbench-flow` takes a new release. 0.1.2 scores exactly
  as 0.1.1. The kit now calls its APIs instead of its own copies:
  `play_isolated` (`sbf check`), `build_image` and `play_container`
  (`sbf check --docker`, so `docker/policy/` is gone: the image's files ship in
  the package), `EpisodeSet.build(quick=True)`, report names, `agent.py` paths,
  `LIMITS`, `agent_config_from_reset`, `play_episode`,
  `markdown(generated_by=...)`; and the benchmark is quiet without the kit's
  logging toggles.

- Your fork is yours to change: `my/`, the "who owns what" rules, `make update`
  and `sbf version` are gone. Pull new releases with `git pull upstream main`
  (README, "Updating").
- The shipped agents are ordinary folders of `agents/` (`template`, `random`,
  `heuristic`); start yours with `cp -r agents/template agents/mine`.
- The examples moved to `examples/`. Each is one self-contained file whose
  options are `--flags`, as for `sbf` (Fire): Hydra, `configs/` and the run
  folders' `meta.json` are gone. The PPO training and the policy search live in
  their examples.
- The heuristic agent reads its numbers from a `params.json` beside it, which
  the policy search writes.
- `sbf upload` takes an agent's name or folder and packs it. It reads
  `CODABENCH_COMPETITION` and `CODABENCH_TOKEN` only, and refuses an `agent.py`
  whose imports the server lacks. `CODABENCH_URL` and the username and
  password variables are gone: `sbf token` asks for them once and saves the
  token in `.env`.
- The README no longer lists reference scores.
- The README starts with numbered setup steps: install uv, `uv sync`, activate
  `.venv`, then run, score and submit.
- `sbf runs`, `sbf fields` (now `scripts/fields_docs.py`) and `sbf version` are
  gone; so are the logging and display helpers, pre-commit, and the
  Hydra, OmegaConf, pandas, pydantic, rich, rootutils and joblib dependencies.

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
