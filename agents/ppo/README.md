# Recurrent PPO with dispatch and queue control

`agent.py` imports only the standard library, NumPy and Torch, loads
`policy.pt` once, and uses current/previous weekly features plus GRU memory.
Evaluation performs neural inference and action masking. `train.py` runs
PPO; `pretrain.py` optionally initializes it from a planner that sees only
public observations. Both training programs stay local.

0.8–0.9 RSS is an experimental target, not a guaranteed result. RSS measures
savings relative to naive and clairvoyant references, not demand served.

The installed Small checkpoint completed 100 PPO updates (41,600 weekly
transitions, 800 training episodes). Its best exact 100-scenario validation
RSS was **0.56061**, at update 40. It did not meet the target. Both local teacher
pretraining experiments decreased imitation loss but hurt evaluation RSS,
so neither was installed. See `agents/rl/LOG.md` for measured outcomes.

Independent confirmation on 100 fresh Small scenarios scored **0.56579**
versus **0.56274** for the previous PPO: gain 0.00306, with a 90% paired
interval of [0.00234, 0.00379]. Both had zero fallback weeks. This is a small
measured improvement, not a demonstrated 0.8–0.9 agent.

## Fixes to the original PPO implementation

- Full-range additive actions: `clip(baseline + tanh(latent), 0, 1)` retains
  the baseline at zero but allows zero/full routes to change substantially.
- Contiguous sequence minibatches, GRU backpropagation across weeks, memory
  burn-in, and resets at episode boundaries.
- Destination stock/inbound/WIP, connected sink forecasts/backlogs/shortages,
  connected grid shedding, factory capacity/power, and source availability.
  Reachability includes production transformations. Inputs retain current
  values, previous values, and differences.
- Learned override quantities and categorical default/override/hold decisions.
  Queue features include observed shipments arriving this week.
- Explicit dispatch and override masks. `action_mask` is an observation field,
  not a return field. Mask confidence respects blackout flags. Closures and
  nominal deadlines are preferences/features rather than hard exclusions.
- Independent critic encoder, Huber value loss, optional value clipping,
  `gamma=1`, longer GAE traces, PPO/KL clipping, and baseline regularization.
- Potential shaping preserves undiscounted episode cost. Shortage and shed
  penalties are not artificially increased.
- Optional counterfactual rewards subtract a fixed controller's cost on the
  identical scenario. This reduces exogenous reward variation without changing
  the cost-minimizing policy; reference rollouts never enter observations.
- CUDA policy batches when available; simulator workers remain CPU processes.
- Nonzero training roots, independent exact validation, checkpoint-zero
  validation, and atomic best/latest exports. Any validation fallback fails.
- `history.csv` records losses, shortage/shed USD, RSS by harm level, and
  fallback counts. Summaries explicitly report whether the target was reached.

The initial queue-mode head chooses default release. The full-range action
change requires retraining; old optimizer checkpoints cannot resume unchanged.

## Training

Install the existing training extra if needed: `uv sync --extra rl`.

Optional public-observation planner warm-start:

```bash
uv run python agents/ppo/pretrain.py --task=small --episodes=128 --epochs=80 \
  --validation_episodes=100 --device=auto --out=outputs/ppo/teacher
```

The teacher dataset is written to `demonstrations.npz` under the run folder.
Reuse it with `--dataset=<path>` and the same task/training entropy. Selection
uses validation RSS, never teacher training loss alone.

If imitation improves loss but hurts RSS, collect labels at learner states:
pass `--behavior_policy=<candidate/policy.pt> --teacher_probability=0.5`
and `--aggregate_dataset=<previous/demonstrations.npz>` with a fresh seed.
These mixed trajectories reduce the mismatch between teacher and learner
inventory states. The installed initializer remains a selection safeguard.

Continue with PPO:

```bash
uv run python agents/ppo/train.py --task=small --updates=1000 --n_envs=8 \
  --train_scenarios=256 --validation_episodes=100 --device=auto \
  --initial_policy=outputs/ppo/teacher/best/policy.pt --patience=0 --install
```

Without a warm-start, omit `--initial_policy`. `--install` replaces local
weights with the best validation checkpoint, including checkpoint zero.
The local imitation experiments did not improve RSS, so the teacher warm-start
remains experimental; direct PPO is the primary workflow. New runs use
`--counterfactual_reward=True`; compatible resumes retain their original
reward definition and do not silently reinterpret a saved cost critic.

Early stopping is disabled by default. When enabled, `--patience` counts
validation checks after `--min_updates` (default 200). The default target is
`--target_rss=0.85`; three consecutive checks must meet it after the minimum
budget before target stopping. Use an untouched root for final confirmation.

Resume a compatible run, explicitly overriding patience if desired:

```bash
uv run python agents/ppo/train.py --resume=outputs/ppo/<run> \
  --updates=1500 --patience=0 --install
```

Resume restores optimizer/RNG state and restarts simulator episodes; it is not
an exact continuation of partially played episodes. Old optimizer schemas are
rejected. `--migrate_only` saves a previous-weights backup under `outputs/ppo/`
and creates a compatible initializer; start a new run afterward.

The default scenario pool is 256, refreshed every 100 updates and populated
once before workers start. `--scenario_cache` can reuse an existing cache;
reference caches follow `SBF_CACHE_DIR`.

Tiny wiring smoke (quick scores are not leaderboard evidence):

```bash
uv run python agents/ppo/train.py --task=tiny --updates=2 --n_envs=1 \
  --train_scenarios=2 --validation_episodes=1 --quick_validation
```

Verify CPU deployment without uploading:

```bash
uv run sbf check ppo --task=small
uv run sbf check ppo --task=full
```

The agent folder also contains training scripts, so package checks may warn
about their training-only imports. `agent.py` does not import them. Generated
`best/` folders contain only runtime files. Root 0 is optional final reporting
via `--evaluate_dev`, excluded from checkpoint selection.
