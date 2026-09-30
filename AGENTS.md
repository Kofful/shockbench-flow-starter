# AGENTS.md

What a coding agent (or you) needs to work in this repository: the task, who
owns which files, the layout, the commands, the rules that decide a score, and
how to evaluate reliably.

## The task

ShockBench-Flow is a Gymnasium control task. Every week of an episode an agent
decides how much of each good to send along each route of a supply network.
Disruptions (a strait closes, a route is sanctioned, a tariff jumps, a factory
goes down) are drawn before the episode starts and nothing the agent does
changes them; the agent sees the network as it is this week, its stock and
shipments, a demand forecast, and noisy early warnings and announcements. An
episode costs money (USD); lower is better.

The score compares an agent's cost with two references on the same scenarios:
**0** is the naive rule (keep shipping the normal plan), **1** is the
clairvoyant plan (it knew every disruption in advance); below 0 is worse than
naive. The package and the board call it RSS.

- Networks: `tiny` (practice, 26 weeks, every tool's default), `small` (the
  public board's, 52 weeks), `full` (the private board's, 104 weeks). Shapes
  differ: read them from `config["spaces"]`, never hard-code Tiny's.
- A submission is a zip with `agent.py` at its root, defining
  `class Agent: __init__(self, config)` (once per episode) and
  `act(self, observation) -> {"flows": ..., ...}` (once per week).
- The details (every field, the RL wrappers, the scoring formula, the noise):
  [docs/GUIDE.md](docs/GUIDE.md) and [docs/fields/](docs/fields/).

## Who owns what

Edit only the participant's side. The organisers ship fixes into their side, and
`make update` merges them; an edit there conflicts at the next update.

| Participant's (edit freely)                                                                            | Organisers' (do not edit)                                                                             |
| ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `agents/<name>/`: one folder per submission, `agent.py` and the files it loads (weights are committed) | `vendor/` (the benchmark's wheel), `src/sbf_starter/`, `src/helper/`, `src/hydra_plugins/`            |
| `my/scripts/`, `my/configs/`, `my/tests/`, any module under `my/` (importable as `my`)                 | `scripts/python/0N_*.py`, `scripts/bash/`, `configs/` (main, hydra, task, the examples' configs)      |
| `[dependency-groups] mine` in `pyproject.toml` (`uv add --group mine <package>`); `outputs/`, `.env`   | `tests/`, `docs/`, `README.md`, `AGENTS.md`, `Makefile`, `CHANGELOG.md`, the rest of `pyproject.toml` |

To change an example, copy it to `my/scripts/`; to change an example's settings,
write a config in `my/configs/` that starts with
`defaults: [<the example's config>, _self_]` and run the example with
`--config-name=<yours>`.

## Layout

```
agents/                 # yours: one submission folder per agent (make new-agent NAME=x)
my/                     # yours: the package `my`
  scripts/              #   your Hydra entry points (the new-script skill writes them here)
  configs/              #   their configs, on every entry point's Hydra search path
  tests/                #   collected by `uv run pytest`
src/
  sbf_starter/          # the kit: agent names (agents/), the `sbf` CLI, scoring, the isolated check,
                        #   search and PPO pieces, run folders; shipped agents in agents/{template,random,heuristic}
  hydra_plugins/        # puts configs/ and my/configs/ on Hydra's search path
  helper/               # logging (Loguru) and rich display helpers
scripts/python/         # the examples 01_quickstart.py ... 07_dashboard.py (Hydra)
scripts/bash/update.sh  # make update
configs/                # main.yaml, hydra/, task/ (tiny, small, full), one config per example
tests/                  # the kit's tests
docs/                   # GUIDE.md, fields/ (every observation and action field), img/
docker/policy/          # the scoring container's Dockerfile and pinned packages
vendor/                 # the shockbench-flow wheel
outputs/                # run folders outputs/<script>/<run_name>/ and packed zips (gitignored)
```

## Commands

Always run Python through `uv run` (the locked environment). `AGENT` is a name
(a folder of `agents/`, else `template`, `random`, `heuristic`) or a path.

| Task                             | Command                                                                  |
| -------------------------------- | ------------------------------------------------------------------------ |
| Install (the rl extra too)       | `make install` (`make install_rl`)                                       |
| Update to the organisers' latest | `make update` (git pull, relock, resync keeping the rl extra, version)   |
| A new agent                      | `make new-agent NAME=mine` (copies the template to `agents/mine/`)       |
| Score locally                    | `uv run sbf evaluate mine` (`--task=small`, `--quick`, `--episodes=...`) |
| Compare two agents               | `uv run sbf compare mine template` (a paired interval)                   |
| Check as the server does         | `uv run sbf check mine --task=small` (`--docker`: the real container)    |
| Pack a zip                       | `uv run sbf pack mine` (to `outputs/mine.zip`)                           |
| Upload (only when asked)         | `uv run sbf upload outputs/mine.zip --dry_run`, then without it          |
| Your submissions                 | `uv run sbf status`                                                      |
| Past runs                        | `uv run sbf runs`                                                        |
| Run an example                   | `uv run python scripts/python/0N_name.py task=small key=value`           |
| Add a dependency of yours        | `uv add --group mine <package>`                                          |
| Tests (the kit's and `my/tests`) | `uv run pytest -n 3`                                                     |
| Lint / format                    | `make lint`                                                              |
| Every CLI command                | `uv run sbf --help`, `uv run sbf <command> --help`                       |

## Rules that decide a score

The full list is [docs/GUIDE.md, "Rules"](docs/GUIDE.md#rules). The ones code
must respect:

- **Imports**: only Python 3.13's standard library, numpy, SciPy and PyTorch
  (CPU) exist on the server, in `agent.py` and in every module it imports.
  Anything else in this environment (gymnasium, Stable-Baselines3, pandas,
  `sbf_starter`, `my`) is for training only. `sbf check` fails a violation.
- **CPU per week**: 2 s on the public board (Small), 4 s on the private board
  (Full). `Agent(config)` counts toward week 1; load weights at module level. A
  week over budget, crashed or malformed is played by the naive rule.
- **Seeding**: seed every random generator from `config["policy_seed"]`.
- **Files**: load them relative to `Path(__file__).parent`; at most 500 MB
  unpacked and 1,000 files; `print` goes to stderr.
- **Never upload unless the participant asks**: each upload spends one of the
  day's 3 submissions. Use `--dry_run` to test. Never print or commit `.env`.

## Evaluating reliably

- A score on 20 episodes is noisy (one standard error 0.17 to 0.20). To decide
  whether a change helped, run `uv run sbf compare new old`: a paired interval
  that holds 0 means the episodes cannot tell them apart.
- Tune on a root of your own (`--entropy=12345 --episodes=64`, or
  `EpisodeSet.build(task, 64, entropy=12345)` in Python) and keep the dev
  episodes (root 0) for confirmation; a search that sees only the dev episodes
  fits them.
- `--quick` (the examples' `quick=true`) is for smoke tests: a rough naive rule,
  no harm levels, not the board's numbers.
- `sbf check` times `act` on this machine and `--cpu_budget` meters it here; the
  server meters its own (`sbf check --docker` is the closest local copy).

## Working in this repository

- New experiments go in `my/scripts/<name>.py` with `my/configs/<name>.yaml`
  (the `new-script` skill in `.claude/skills/` scaffolds both, and a test in
  `my/tests/`). They compose `main` and the `task` group like the examples, and
  write their run folder `outputs/<name>/<run_name>/` through
  `sbf_starter.runs.start(cfg)`.
- Hydra for entry points (`@hydra.main`, called under
  `if __name__ == "__main__":`), Fire for the `sbf` CLI; no argparse or click.
- Log through `from helper.logging import logger`; draw randomness from local
  seeded generators (`np.random.default_rng(seed)`).
- Verify before reporting done: `uv run sbf check <agent> --task=small` for an
  agent, a run with small settings for a script, `uv run pytest -n 3` for code.
  If a check cannot run (no Docker, no network), say so.
