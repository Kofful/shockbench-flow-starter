# Recurrent residual PPO agent

This folder contains both halves of the local-only workflow:

- `agent.py` is the submission runtime. It imports only NumPy, Torch and the
  standard library, loads `policy.pt` once, maintains previous-week features
  and recurrent state, and returns deterministic flows.
- `train.py` is the training program. It may use the repository's RL-only
  packages; it is never imported by `agent.py`.
- `policy.pt` is the installed TorchScript actor. The committed initial file is
  a zero-residual network, so it starts from the same tuned controller as
  `agents/rl` instead of random actions. A serious training run should replace
  it using `--install`.

## What was changed relative to `examples/05_train_ppo.py`

The actor uses shared per-route encoders and pooled network context, so one
architecture supports Tiny, Small and Full shapes. Its compact features retain
stock, inbound pipeline, WIP, queued cargo, capacity, freight/tariff changes,
transit time, prohibitions, chokepoint state, warnings, pending sanctions,
closure timing, last execution, demand pressure and shortages. Every decision
contains the current values, previous values and their difference. A GRU also
retains information from earlier weeks.

PPO predicts bounded residuals around a strong nominal controller. Known
illegal, closed and too-late routes are projected out. Training uses `gamma=1`
by default because the benchmark scores undiscounted episode cost. The trainer
also provides:

- parallel scenario environments and CUDA-batched policy updates;
- deterministic non-dev training roots refreshed during long runs;
- a distinct fixed validation root with exact RSS model selection;
- root 0 only as an optional final report, never as a selection signal;
- GAE, PPO/value clipping, entropy regularisation, gradient clipping, target-KL
  stopping and learning-rate annealing;
- periodic best/latest exports, atomic resumable checkpoints, early stopping,
  full settings, and `history.csv` with policy/value losses and RSS.

The deployment actor deliberately leaves chokepoint release overrides at their
simulator default. The default release is a safe deterministic controller;
learning mixed continuous/categorical override actions should only be added
after a paired held-out comparison demonstrates an improvement.

## Commands

Install the training dependencies once:

```bash
uv sync --extra rl
```

Run a cheap wiring smoke test:

```bash
uv run python agents/ppo/train.py --task=tiny --updates=1 --n_envs=1 \
  --train_scenarios=2 --validation_episodes=1 --quick_validation
```

Train the public-board network and install only the best held-out checkpoint:

```bash
uv run python agents/ppo/train.py --task=small --updates=300 --n_envs=8 \
  --train_scenarios=4096 --validation_episodes=100 --device=auto --install
```

Resume to a new total update count; the original training settings are restored
from `checkpoint.pt`:

```bash
uv run python agents/ppo/train.py --resume=outputs/ppo/<run> --updates=500 --install
```

For final confirmation only, add `--evaluate_dev`. Do not repeatedly tune from
that result. Exact validation can be expensive because references include the
clairvoyant solution; `--quick_validation` is only for smoke tests.

The run folder contains `best/` and `latest/`, each a clean submission folder,
plus `checkpoint.pt`, `history.csv`, `settings.json` and `summary.json`.

Verify the installed agent without uploading anything:

```bash
uv run sbf check ppo --task=small
```

Because `train.py` is intentionally kept beside the agent, `sbf check ppo` may
list its training-only imports as nonfatal warnings. The generated `best/`
folder contains only `agent.py` and `policy.pt` and has no such warnings.

For a robust final choice, run several seeds into separate run folders and use
the same held-out validation root for all of them. Select by held-out RSS, then
inspect root 0 only once after the choice is frozen.
