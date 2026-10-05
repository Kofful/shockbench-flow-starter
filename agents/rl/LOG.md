# RL agent development log

Date: 2026-10-05
Current policy source: `agents/rl/agent.py`

This file records the work performed to accelerate local evaluation, design and
tune the RL agent, validate it across the three task sizes, and add a dashboard
for inspecting its losses. All work and evaluation stayed local. Nothing was
uploaded to Codabench or published to PyPI.

## 1. Outcome

The final agent is a deterministic residual policy trained through episodic,
return-based parameter search. It starts from the public nominal dispatch plan
and learns how much unused route capacity to add for different supply-chain
stages. It also reacts to observed chokepoint closures and stops dispatches that
cannot reach a point of use before the episode ends.

The main held-out results were:

| Task and evaluation | RL agent | Send-maximum baseline | Difference |
| --- | ---: | ---: | ---: |
| Small, exact 20-episode dev score | 0.5276 | 0.4086 | +0.1190 |
| Small, quick 20-episode dev score | 0.4909 | 0.3518 | +0.1391 |
| Tiny, quick 20-episode dev score | 0.8228 | 0.7269 | +0.0959 |

The exact Small paired 90% interval was `+0.0891` to `+0.1526`. The
quick Small paired interval was `+0.1094` to `+0.1720`, and the quick Tiny
paired interval was `+0.0665` to `+0.1404`. In all three comparisons, every
bootstrap result favored the RL agent.

`quick` scores use a rough naive reference and no harm-level weighting. They
are useful for iteration but are not leaderboard scores. The exact Small result
uses the benchmark's full naive model and leaderboard harm-level weighting.

## 2. Evaluation-speed groundwork

Before tuning the policy, I inspected the starter's scorer and the installed
`shockbench-flow` package to find where evaluation time was spent.

### 2.1 Findings

1. Reference episodes were already process-parallel during construction.
2. Ordinary agent episodes were not receiving the requested `n_jobs` value
   through the starter's scoring wrapper.
3. The simulator itself is NumPy/SciPy-based and is not a useful CUDA target.
4. Neural policy inference is the part that can benefit from CUDA and batching.
5. The local environment did not have PyTorch or Stable-Baselines3 installed.
6. `nvidia-smi` could not communicate with the NVIDIA driver, so CUDA could not
   be benchmarked on this machine.

### 2.2 Local changes made

The local evaluation path was extended so that:

- ordinary agents play episodes in parallel through `n_jobs`;
- agents may implement the optional protocol
  `Agent.act_batch(agents, observations)`;
- compatible agents are evaluated in lock-step batches with independent
  environments and agent objects;
- `--batch_size` and `--device=auto|cpu|cuda|cuda:N` are available to evaluate
  and compare commands;
- the requested evaluation device is exposed through `SBF_EVAL_DEVICE`;
- batched evaluation is disabled when the per-week CPU meter is active because
  a single batch call cannot reproduce independent per-episode CPU accounting;
- the PPO export example can load its TorchScript policy on CUDA locally and
  execute one batched forward pass for all live episodes;
- PPO training accepts a PyTorch `device` argument;
- policy-search evaluation forwards `n_jobs` to agent episode scoring;
- a local `cuda` optional dependency was added without publishing anything.

The corresponding documentation and tests were updated. The lock metadata was
kept consistent locally and `uv lock --check` passed. No package version was
uploaded or changed on PyPI.

### 2.3 Acceleration validation

- A custom batched test policy produced the same episode costs as serial
  evaluation, to the cent.
- A CLI evaluation using parallel episodes completed successfully.
- The non-Codabench suite passed with 37 tests and 4 expected skips at that
  stage.
- Ruff, formatting, `git diff --check`, and the local lock check passed.
- CUDA behavior could not be timed because the host's NVIDIA driver was not
  available.

## 3. Initial agent investigation

I inspected:

- `examples/05_train_ppo.py` and its TorchScript export;
- the heuristic, template, random, and initial agents;
- every observation and action field for Tiny, Small, and Full;
- the simulator's route, inventory, production, and cost behavior;
- the benchmark's naive, greedy LP, deterministic MPC, and state/base-stock
  policy implementations;
- the public instance tables, action slots, lanes, lead times, production
  delays, nominal pipeline, and sink demands.

The installed RL training extra was unavailable locally, and a short generic
PPO run was unlikely to outperform the strong send-maximum policy. I therefore
used a compact residual actor whose parameters could be optimized directly on
complete episodic return. This is an evolution-strategy form of reinforcement
learning: candidate policies play complete scenarios, and parameters are
selected using their total reward rather than labeled target actions.

## 4. Baseline measurements

The first quick comparisons established the starting point:

- Tiny, 8 episodes: heuristic `0.7327`, send-maximum `0.7327`.
- Small, 4 episodes: heuristic `0.4535`, send-maximum `0.4793`.

The heuristic's simple closure scaling did not improve Tiny and was slightly
worse on the first Small sample. Send-maximum was therefore used as the main
baseline.

## 5. Actor design

### 5.1 Nominal residual action

For every action slot, the actor reconstructs the public nominal week-zero
dispatch from `static["instance"]["initial_state"]["pipeline"]`. Its basic
action is:

```text
flow = nominal + reserve * (nominal_capacity - nominal)
```

This preserves the public normal plan and learns only how aggressively to use
the remaining capacity. A reserve of 0 means nominal dispatch; a reserve of 1
means send the maximum.

### 5.2 Learned route categories

Each slot is classified by:

- whether it appears in the public nominal plan (`planned` or `new`); and
- the type of node dispatching it (`source`, `terminal`, `fab`, `osat`, or
  `other`).

This gives ten reserve parameters. It is much smaller and more robust than one
parameter per slot, and it transfers from Small to Full despite their different
numbers of routes.

### 5.3 End-of-horizon mask

The actor derives the last useful dispatch week for every slot from the public
network. It recursively computes the minimum remaining time from a
node/commodity pair to a point of use, accounting for:

- route lead times;
- the one-week delay before stock can be dispatched onward;
- fab production lead time and product conversion;
- OSAT packaging lead time and product conversion;
- immediate use at sinks and power grids.

Flows after the learned deadline plus `end_margin` are set to zero. The logic
reads the network dynamically and does not hard-code Tiny or Small shapes.

### 5.4 Closure response

For a lane crossing observed chokepoints, flow is multiplied by:

```text
product(open_fraction ** closure_power)
```

The exponent is learned separately for Tiny and for Small/Full. Direct routes
do not receive a chokepoint multiplier.

### 5.5 Distress signal

The actor computes a dimensionless distress value from:

- last week's lost demand divided by total demand; and
- last week's power shed divided by total grid base load.

The implementation can raise reserve capacity by
`distress_gain * distress`. Search showed that positive gains increased cost,
so the final learned `distress_gain` is 0. The feature remains implemented for
future training experiments.

### 5.6 Safety projection

The final flows are multiplied by `observation["action_mask"]`. The simulator
then performs its normal stock, shared-edge, fleet, throughput, and storage
clipping. The agent returns only `flows`, leaving queue release behavior at the
benchmark's default.

## 6. Small parameter search

Search used private training roots rather than root 0. The dev root was kept
for held-out confirmation.

### 6.1 Global reserve search

On 12 Small episodes from training root `20261005`, the global reserve sweep
gave:

| Reserve | Quick RSS |
| ---: | ---: |
| 0.35 | 0.510265 |
| 0.40 | 0.520110 |
| 0.45 | 0.527334 |
| 0.50 | 0.529619 |
| 0.55 | 0.525258 |
| 0.60 | 0.520475 |
| 0.65 | 0.516118 |
| 0.70 | 0.512368 |

The first robust starting point was therefore a reserve of 0.50.

An earlier 8-episode search showed the same pattern: send-maximum scored
`0.53146`, while a reserve of 0.50 with the end mask scored `0.56414`.
On 8 dev episodes, the same change improved `0.39054` to `0.42672`.

### 6.2 Per-stage coordinate search

Starting from 0.50 for every stage, each parameter was swept independently.
The strongest observations were:

- planned source routes preferred about 0.75;
- planned fab routes preferred 0;
- planned OSAT routes preferred 1;
- planned `other` routes preferred 0;
- new source routes preferred about 0.50;
- new terminal routes saturated by about 0.50;
- new fab routes improved toward 0.75-1.00;
- new OSAT routes strongly preferred 0;
- new `other` routes preferred 1.

Combining those categories raised training RSS from `0.529619` to `0.584423`.

### 6.3 Dynamic-parameter search

With the stage parameters combined:

- positive distress gain consistently reduced the score, so it was set to 0;
- closure response improved the first root, with powers around 0.5-1.0 best;
- a small positive end margin was slightly more robust than an exact cut;
- raising the new-fab reserve to 1 improved the combined policy.

On root `20261005`, the joint policy reached about `0.613`, and a closure power
of 1 could reach about `0.617` on that root.

### 6.4 Independent-root confirmation

The closure exponent was selected using another 12 Small episodes from root
`20261006`:

| Policy | Quick RSS |
| --- | ---: |
| Send maximum | 0.517248 |
| Global reserve 0.50 | 0.545516 |
| Combined stages, closure power 0 | 0.599375 |
| Combined stages, closure power 0.50 | 0.604427 |
| Combined stages, closure power 0.75 | 0.600653 |
| Combined stages, closure power 1.00 | 0.596134 |

Although power 1 was slightly better on the first root, power 0.5 transferred
better. The cross-root choice was therefore `closure_power = 0.5`.

### 6.5 Final Small/Full parameters

```text
planned_source   0.75     new_source   0.50
planned_terminal 0.50     new_terminal 0.50
planned_fab      0.00     new_fab      1.00
planned_osat     1.00     new_osat     0.00
planned_other    0.00     new_other    1.00
distress_gain    0.00
closure_power    0.50
end_margin       1
```

## 7. Small held-out evaluation

### 7.1 Quick 20-episode comparison

```text
RL agent:      0.4909
Send maximum:  0.3518
Difference:   +0.1391
90% interval: +0.1094 to +0.1720
```

### 7.2 Exact 20-episode comparison

The exact naive demand model took 181.5 seconds to prepare and used 1,000
replications. The 20 naive and clairvoyant reference episodes then took 19.6
seconds and were cached locally.

```text
RL agent:      0.5276
Send maximum:  0.4086
Difference:   +0.1190
90% interval: +0.0891 to +0.1526
```

This was the main evidence used to keep the Small policy.

## 8. Tiny-specific correction

The Small coefficients initially transferred poorly to Tiny:

```text
Small coefficients on Tiny: 0.4172
Send maximum on Tiny:       0.7327
```

Tiny has only 20 action slots, a 26-week horizon, and one chokepoint, so it was
given its own parameter set.

On 32 training episodes from root `20261005`, a strong closure response and an
exact end cut were best. Planned `other` routes benefited from zero residual
capacity, while most other categories needed maximum reserve. On independent
root `20261006`:

```text
Send maximum:                 0.597380
Tiny policy, closure 0.50:    0.688173
Tiny policy, closure 0.75:    0.692193
Tiny policy, closure 1.00:    0.694324
Tiny policy, closure 1.25:    0.677830
```

### Final Tiny parameters

```text
planned_source   0.75     new_source   1.00
planned_terminal 1.00     new_terminal 1.00
planned_fab      1.00     new_fab      1.00
planned_osat     1.00     new_osat     1.00
planned_other    0.00     new_other    1.00
distress_gain    0.00
closure_power    1.00
end_margin       0
```

The final quick 20-episode dev comparison was:

```text
Tiny RL agent:  0.8228
Send maximum:   0.7269
Difference:    +0.0959
90% interval:  +0.0665 to +0.1404
```

The agent selects this parameter set when
`static["instance_id"] == "chokepoint-tiny"`; otherwise it uses the Small/Full
set.

## 9. Full transfer check

Building Full's exact references would have been expensive, so I first played
four Full dev scenarios directly and compared episode costs against
send-maximum on the same scenarios. The RL agent cost less in every episode.

| Episode | Saving versus send maximum |
| ---: | ---: |
| 0 | about $600.4B |
| 1 | about $609.6B |
| 2 | about $118.0B |
| 3 | about $283.3B |

This is a cost comparison, not a Full RSS estimate, because clairvoyant
references were not generated for this check.

## 10. Submission and runtime validation

The checker does not accept a bare `.py` path as a submission archive, so the
source was copied into a temporary folder as `agent.py` for isolated checks.
The actual source remained local.

Final Small isolated check:

- scorer/import checks passed;
- 1 file, about 11 KB unpacked;
- week 1 initialization plus first action: 0.016 CPU seconds;
- median action: 0.0014 CPU seconds;
- maximum action: 0.016 CPU seconds;
- budget: 2 CPU seconds per week;
- no fallback weeks.

Full isolated check:

- checks passed;
- week 1 initialization plus first action: 0.023 CPU seconds;
- median action: 0.0018 CPU seconds;
- maximum action: 0.023 CPU seconds;
- budget: 4 CPU seconds per week.

The policy imports only the standard library and NumPy. It has no weight file,
PyTorch dependency, network access, or runtime training step.

## 11. Loss dashboard

After the policy was complete, `examples/08_agent_losses.py` was added to make
the agent's cost behavior easier to inspect.

### Dashboard behavior

The script:

1. loads the local `rl` agent, supporting either `agents/rl.py` or the current
   `agents/rl/agent.py` folder layout;
2. selects a requested dev episode, or searches for one containing a closure;
3. records the agent and optionally the naive rule on exactly the same scenario;
4. saves the existing full episode dashboard and GIF;
5. builds a new four-panel loss figure containing:
   - weekly objective loss, with terminal credit in the final week;
   - cumulative objective loss for the agent and naive rule;
   - cumulative excess loss, green below zero for savings and red above zero
     for worse-than-naive performance;
   - weekly cost split into freight, war risk, tariff, holding, queue holding,
     shortage, disposal, and power-shed components;
6. saves the raw episode record for later analysis.

Generated files are:

```text
network.png
dashboard_agent.png
losses_agent.png
episode_agent.gif
record_agent.npz
```

The first smoke run on Tiny episode 29 reported:

```text
RL loss J:       about $3.1B
Naive loss J:    about $3.6B
Excess loss:     about -$503.3M
```

The negative excess means the RL agent saved about $503.3M relative to naive.
The rendered loss chart was also inspected visually.

README's examples table was updated, and a regression test verifies all five
dashboard artifacts.

## 12. Final verification

After the dashboard change:

- `38 passed, 4 skipped` in the non-Codabench test suite;
- the three Docker tests were skipped because no Docker daemon was available;
- the PPO example test was skipped because the RL training extra was not
  installed;
- the Codabench mock tests were excluded because the sandbox blocks their local
  loopback server;
- Ruff checks passed;
- all 31 Python files passed the formatting check;
- `git diff --check` passed.

## 13. Files changed

### Agent

- `agents/rl/agent.py`
  - residual RL actor;
  - learned Tiny and Small/Full parameter sets;
  - nominal-flow reconstruction;
  - route-stage classification;
  - task-independent dispatch deadlines;
  - closure response and action-mask projection;
  - distress feature retained with a learned gain of zero.

### Evaluation acceleration

- `src/sbf_starter/accelerated.py`
  - batched lock-step evaluation and device handling.
- `src/sbf_starter/scoring.py`, `src/sbf_starter/cli.py`
  - `n_jobs`, batch-size, and device routing.
- `examples/04_evaluate.py`, `examples/05_train_ppo.py`,
  `examples/06_policy_search.py`, `examples/ppo_agent.py`
  - accelerated evaluation, CUDA-aware training/export, and parallel search.
- `pyproject.toml`, `uv.lock`, `README.md`, `docs/GUIDE.md`
  - local CUDA extra, lock metadata, and usage documentation.
- `tests/test_accelerated.py`
  - routing, device, environment, and batch-size tests.

### Loss dashboard

- `examples/08_agent_losses.py`
  - standard dashboard plus dedicated loss analysis.
- `README.md`
  - new example entry.
- `tests/test_examples.py`
  - dashboard artifact regression test.

## 14. Reproduction commands

Quick Tiny loss dashboard:

```bash
uv run python examples/08_agent_losses.py --quick
```

Quick Small loss dashboard for episode 0:

```bash
uv run python examples/08_agent_losses.py --task=small --episode=0 --quick
```

Quick Small comparison:

```bash
uv run sbf compare rl template --task=small --quick --episodes=20
```

Exact Small comparison:

```bash
uv run sbf compare rl template --task=small
```

Submission-style check:

```bash
uv run sbf check rl --task=small
```

Tests and lint:

```bash
uv run pytest -n 3
make lint
```

## 15. Known limitations and next experiments

1. The policy is a compact parameterized actor, not a neural PPO checkpoint.
   Its RL signal is complete episodic return optimized through parameter search.
2. Full was checked through direct paired costs on four episodes, not an exact
   Full RSS evaluation.
3. CUDA acceleration applies to compatible batched neural agents; this NumPy
   policy is already much faster than the CPU budget and does not benefit from
   CUDA.
4. `distress_gain` is implemented but currently zero because positive values
   hurt both training cost and transfer.
5. The current actor uses default queue release behavior. Learning release and
   hold actions is a possible future improvement.
6. More independent training roots and an exact Full comparison would provide
   stronger evidence for private-board transfer.
