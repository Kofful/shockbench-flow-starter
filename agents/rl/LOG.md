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

## 16. Recurrent PPO agent and separate trainer

I added `agents/ppo/` as a second, neural RL implementation. It contains
`agent.py`, `train.py`, `policy.pt`, and a focused `README.md`. Training and
evaluation are separated: `train.py` owns Gymnasium, Stable-Baselines3's vector
environment helpers, optimization, scoring, checkpointing, and export;
`agent.py` imports only the standard library, NumPy, and Torch. At module load,
the evaluator loads the TorchScript checkpoint once. During an episode it only
extracts features, performs deterministic inference, updates recurrent state,
and returns flows.

The original PPO example discarded the padded dynamic state and used a flat,
memoryless, task-specific MLP. The new feature builder instead creates one
fixed-width record per route. It aggregates stock, inbound pipeline, WIP,
queued cargo, current capacities, costs, tariffs, transit times, prohibitions,
chokepoint state, early warnings, pending prohibitions, announced reopening,
last requested and executed flow, forecast demand, shortage, shed power, and
observation confidence. The input contains the current record, previous week's
record, and their difference. A GRU adds longer episode memory. The encoder is
shared across routes and pools route embeddings for network-wide context, so
the same architecture accepts Tiny, Small, and Full shapes.

The actor is residual rather than absolute. The zero-residual checkpoint
reproduces the tuned `agents/rl` controller. PPO applies bounded corrections
in logit space, while deterministic projection removes known illegal, closed,
zero-capacity, and too-late dispatches. This makes exploration start from a
working controller rather than destructive random flows. The committed
`policy.pt` is this safe initializer; a long training run replaces it only
when `--install` is explicitly passed.

The trainer implements finite-horizon PPO with `gamma=1` by default, GAE,
policy and value clipping, entropy regularization, gradient clipping,
target-KL stopping, learning-rate annealing, parallel simulator processes, and
CUDA-batched neural updates when CUDA is available. It rotates deterministic
nonzero training roots during long runs. Model selection uses exact RSS on a
fixed, distinct, nonzero validation root. Root 0 is excluded from training and
selection and is evaluated only when the user explicitly passes
`--evaluate_dev`. Every run records settings and optimization losses, exports
`latest/` and `best/` clean agent folders, writes an atomic resumable
`checkpoint.pt`, and supports early stopping. Resume restores the original run
settings and accepts a new total update count.

Tiny exposes queue lots as padded records, while Small and Full expose a dense
lot-key-by-week matrix. The first Small server-style check revealed that schema
difference. I added both aggregation paths and repeated the check. The final
Small check completed all 52 weeks with no fallback, a median action time of
about 0.0069 seconds, and a maximum of about 0.036 seconds against the 2-second
budget. The Full check completed all 104 weeks with no fallback, a median of
about 0.0242 seconds, and a maximum of about 0.073 seconds against the 4-second
budget.

I exercised three training paths directly without pytest or the `tests/`
folder: a four-week optimizer/export smoke, a complete 26-week Tiny rollout,
and a two-update scenario-refresh run followed by resume to update three. The
complete rollout recorded an episode return, PPO losses, exact quick held-out
RSS, a best agent, a checkpoint, history, and summary. A paired four-scenario
Small quick comparison showed the initialized PPO and `agents/rl` at the same
displayed RSS (`0.5375`, difference `-0.0000`), confirming the intended safe
initialization. Quick scores are only wiring checks, not leaderboard evidence.

Static checks passed for both new Python files. `sbf check ppo` reports
nonfatal warnings for training-only imports because the requested trainer is
kept inside the agent folder; generated `best/` folders contain only
`agent.py` and `policy.pt`. No pytest command or file under `tests/` was run.
Nothing was uploaded to Codabench or published to PyPI.

## 17. PPO action, memory, distress, and queue-control revision (2026-10-06)

### Diagnosis and preserved baseline

I read the actual Small run under `outputs/ppo/2026-10-05_17-46-51/`.
It requested 300 updates but stopped at update 40 because the best checkpoint
was update 10 and patience was 30 updates. It played 320 complete episodes,
16,640 weekly transitions. RSS moved from 0.5581887 at update 1 to 0.5583070
at update 10, then fell to 0.5555537. SHA-256 checks confirmed that the installed
weights were the best checkpoint. The training device had resolved to CPU.

The logit residual was a stronger restriction than intended: its correction
was limited to +/-2. A baseline-zero route could reach only 0.0007384 of capacity;
a baseline-full route could fall only to 0.9992616. In one Small observation,
38 of 78 active routes had less than one percentage point of possible movement.
I preserved the old agent and weights at `outputs/ppo/pre_v2/`, and migration
also keeps a previous-weights backup under `outputs/ppo/migration_backup/`.

### Action and observation changes

I replaced the logit window with `clip(base + tanh(latent), 0, 1)` in both
training and deployment. Zero means the exact baseline; large positive/negative
latents can now move zero/full routes across the capacity range. A direct
NumPy/Torch parity check confirmed endpoint movement and a masked route stayed
zero. Closure and nominal deadline restrictions became baseline preferences
and observed features rather than removing legal actions from exploration.
Known prohibitions and zero first-edge capacity remain projected out.

The new inputs add route/commodity/destination identity, destination stock,
inbound shipments and WIP, forecasts/backlog/shortage at reachable sinks,
shedding at reachable grids, source availability, factory capacity/power,
storage pressure, and clipping losses. Reachability follows both transport
and production transformations. Separate shortage and shed signals replace
reliance on a single global distress multiplier. `distress_gain` belongs to
the older compact RL controller; PPO learns its response through these inputs.
Mask confidence now reads `action_mask.observed` rather than testing whether
the mask's numeric value is nonnegative. Deployment explicitly applies both
dispatch and override masks. Masks are inputs, not action return fields.

### Recurrent optimization and stopping

The trainer now uses contiguous sequence minibatches, recent-memory burn-in,
backpropagation across weeks, and hidden-state resets at episode boundaries.
A three-week gradient diagnostic confirmed nonzero gradient from the final
decision into the first week's input. The critic has an independent encoder;
Huber loss and disabled-by-default value clipping reduce sensitivity to large
episode costs. Gamma remains 1 and GAE lambda becomes 0.99. Optional potential
shaping preserves the undiscounted cost objective up to the fixed initial-state
potential; no arbitrary shortage/shed penalty multipliers were introduced.

Patience is disabled by default, counts validation checks when enabled, and
cannot stop before the minimum update budget. Explicit patience overrides work
on resume: a Tiny resume check successfully changed it to 7. Checkpoint zero
is validated before optimization and retained if trained checkpoints are worse.
The target defaults to 0.85 RSS, requires repeated successful checks, and is
reported as met/unmet rather than asserted. Resume rejects incompatible old
optimizer schemas, restores compatible optimizer/RNG state, and restarts
simulator episodes. Completed episode counters persist in new checkpoints.

Training history includes shortage/shed USD, RSS by harm stratum, fallback
counts, anchor loss, and stale validation-check counts. Old CSV headers are
migrated when needed. Validation fallback is an error. Scenario pools default
to 256, refresh every 100 updates, and populate once in the parent before
workers start, avoiding duplicate generation and many unused cached scenarios.

### Queue control and optional teacher initialization

A paired single-scenario planner diagnostic cost $3.170970 trillion when only
its dispatch flows were played, versus $2.681561 trillion with its queue
overrides. This motivated adding shared neural features for override routes,
learned release quantities, and categorical default/override/hold decisions.
The evaluator returns all three valid action fields and includes observed
shipments arriving this week when determining release availability. PPO uses
joint Gaussian-latent and categorical likelihoods; override latent likelihoods
are included only when override mode is selected. The initializer chooses
default release. These changes are the v3 feature/action/checkpoint schema.

I added `agents/ppo/pretrain.py`, which collects demonstrations from `mpc_det`
on non-dev roots using only public standard observations. No omega or future
disruption payload is passed to the teacher. Solver dependencies stay in
training; deployment remains NumPy/Torch inference. It saves datasets, fits
full episode sequences, and selects by exact held-out RSS. Dataset reuse and
aggregation are supported. A first 32-scenario/40-epoch experiment reduced
imitation loss from 9.3927 to 0.4861 but hurt RSS, so its initializer (0.5583411)
was retained. These failed candidates were not installed.

The planner itself scored 0.6980455 on a 20-scenario held-out diagnostic
(RSS by harm level: 0.6856961, 0.6730946, 0.8136731, 0.6160403). This is a
diagnostic, not independent final evidence or a promised 0.8-0.9 result.
To address imitation distribution shift, I added collection on mixed
learner/teacher trajectories with teacher labels at the learner's states,
dataset aggregation, and a quantity-space imitation loss instead of giving
extreme inverse-tanh targets disproportionate weight. A second experiment
uses 16 mixed trajectories plus the original 32 demonstrations. A separate
100-update Small PPO experiment starts from the retained initializer.

### Verification and environment limitations

Tiny full-horizon optimizer/export and resume smoke runs completed without
pytest. Small v3 deployment completed all weeks with no fallback, maximum CPU
time about 0.050 seconds; Full completed with no fallback, maximum about
0.114 seconds. Both are below their 2/4-second limits. Ruff passed. Training
scripts in the agent folder produce nonfatal training-import warnings; clean
export folders contain only runtime files.

`nvidia-smi` could not communicate with the NVIDIA driver; Torch also selected
CPU. CUDA code remains available, but no local CUDA speedup was demonstrated.
The sandbox blocked the multiprocessing forkserver socket for the eight-worker
PPO experiment. I requested and received execution approval, then relaunched
that local training outside the sandbox. References were copied to a writable
temporary cache and existing scenarios reused without modifying the original
cache. An earlier v2 experiment was interrupted before optimization to add
the missing queue-control action space; no trained checkpoint was lost.

No pytest command, script under `tests/`, Codabench mock, upload, or PyPI
publication was performed. Follow-up measured results are recorded below.

### Follow-up training and implementation outcomes

The mixed-trajectory imitation experiment completed all 30 epochs. Training
loss fell from 7.4839 to approximately 0.3110, but exact validation RSS was
worse than the initial 0.5583411. Selection kept epoch zero and this candidate
was not installed. Lower imitation loss was not evidence of a better control
policy in either experiment. The optional teacher script remains experimental,
not a claimed route to the requested score.

The main Small PPO run completed all 100 requested updates rather than stopping
at 40: 41,600 weekly transitions and 800 complete training episodes using eight
CPU simulator workers and a 256-scenario training pool. It used training root
1503053992 and exact 100-episode validation root 20261006, with no quick scoring
or dev-root selection. Best validation RSS progressed from 0.5583411 at update
zero to 0.5588967 at update 1, 0.5602212 at update 20, and 0.5606087 at update
40. Later checkpoints did not improve the best. The trainer installed update
40; matching SHA-256 hashes confirmed that `agents/ppo/policy.pt` is the exported
best checkpoint. `outputs/ppo/v3_small/summary.json` explicitly records
`target_met: false`. The 0.8-0.9 target has **not** been achieved.

I changed validation routing to use batched inference when CUDA is available
or evaluation uses one worker, and CPU process-parallel evaluation otherwise.
This keeps the CUDA batch path while avoiding serial CPU feature extraction
when parallel workers are requested. A Tiny batched-validation smoke completed.
Training imports now set a feature-only flag before importing the deployment
module. This avoids loading a checkpoint or initializing CUDA in simulator
workers that need only feature construction; normal evaluation still loads
weights once at module scope. A Tiny teacher smoke verified this import path.

I added an optional, default-on counterfactual reward for **new** runs: subtract
the weekly reward of a fixed nominal/reserve/closure-aware controller played
on the identical scenario. The resulting episode objective is the reference
cost minus policy cost, preserving the cost-minimizing optimum while removing
an action-independent source of variation. The reference uses default queue
release, and its trajectory never enters the learner's observations. The
reference is cached per scenario inside each simulator worker.

Two direct Tiny diagnostics checked the reward accounting: following that
same controller gave exactly zero relative episode reward; a zero-dispatch
policy gave -15.98322314607343, matching the scaled cost difference to floating
point precision. A two-update Small smoke and resume to update three completed
with the new reward path and persisted episode counters. These are wiring and
objective-accounting checks, not evidence of an RSS improvement. The 100-update
run had already started with the original dense cost rewards; compatible
resumes retain their saved reward definition. A new relative-reward run resets
the cost critic while retaining the actor, rather than silently reusing a
critic trained on a different target.

Finally, minibatch permutation uses a local seeded NumPy generator and saves
its state in new checkpoints. The patience baseline is tracked separately
from the absolute best checkpoint, so cumulative small improvements can reset
patience once they exceed `min_delta`. These changes received optimizer/resume
smoke coverage; older compatible checkpoints retain legacy RNG support.

The installed trained weights passed fresh isolated CPU checks: all 52 Small
weeks, median act 0.0214 s and maximum 0.054 s versus the 2 s limit; all 104 Full
weeks, median 0.0663 s and maximum 0.132 s versus the 4 s limit. Neither used
fallback. These are local runtime checks, not claims of Full score quality.
The expected training-only-import warnings remain nonfatal; clean exported
agent folders contain only the permitted runtime module and weights. No upload
was performed despite the check command printing an upload suggestion.

For independent confirmation I froze the installed checkpoint and started a
paired comparison against the preserved old PPO on 100 fresh scenarios,
root 20261008. This root was not used for checkpoint selection. The old agent
alone scored 0.5627375 there (90% interval 0.5332728-0.5919509, zero fallback).
The paired comparison result is recorded below when complete.

### Independent confirmation and final handoff

The exact paired 100-scenario confirmation completed successfully. The new
PPO scored 0.5657928153 and the preserved old PPO scored 0.5627375484. Their
difference was +0.0030552669, with a paired 90% resampling interval
[0.0023415494, 0.0037934002]. This interval excludes zero: the measured gain is
small but detectable on these scenarios. The new agent's absolute 90% RSS
interval was [0.5362950015, 0.5951166821]. Mean cost decreased from approximately
$2.767628 trillion to $2.764215 trillion, a saving of about $3.413403 billion
per episode. Both agents had zero fallback weeks. Results are saved in
`outputs/ppo/confirmation_comparison.json`.

The commands used were the direct training and CLI workflows, never pytest:

```bash
uv run python agents/ppo/train.py --task=small --updates=100 --n_envs=8 \
  --train_scenarios=256 --entropy=1503053992 --refresh_every=0 \
  --scenario_cache=/home/kofful/.cache/shockbench-flow \
  --validation_episodes=100 --validation_jobs=4 --eval_every=20 \
  --out=outputs/ppo/v3_small --install
uv run sbf compare ppo outputs/ppo/pre_v2 --task=small --episodes=100 \
  --entropy=20261008 --device=cpu --batch_size=1 --n_jobs=4 \
  --out=outputs/ppo/confirmation_comparison.json
uv run sbf check ppo --task=small
uv run sbf check ppo --task=full
```

Actual invocations used `SBF_CACHE_DIR=/tmp/sbf-ppo-cache`, a writable uv cache
and `--no-sync` to avoid changing the locked dependencies. The training command
above records the experiment's arguments; the current trainer additionally
defaults to relative rewards, so reproducing this original reward run requires
`--counterfactual_reward=False`. I did not start a longer experiment after the
participant chose “Finish this run and report the measured results.”

Final Ruff lint/format checks and `git diff --check` passed. The unrelated
`dotvenv.zip` was left untouched. All modifications are local, the installed
weights are the selected trained checkpoint, and no tests under `tests/`,
pytest, Codabench mock/upload, or PyPI publication/update was performed. The
reported outcome is a working revised PPO with a small confirmed RSS gain;
the requested 0.8-0.9 performance remains unmet.

## 18. Clairvoyant comparison dashboards and decision diagnostics (2026-10-06)

### Request and investigation

The participant requested the same episode dashboards for the clairvoyant
reference, comparisons against their agent and naive, and more detail about
decisions that caused extra costs. I inspected examples 07/08, the installed
dashboard recorder, the Gym/core environment snapshot APIs, the reference LP,
its replay verifier and weekly cost formation. No pytest or files in `tests/`
were run. No Codabench mock, upload, PyPI publication, dependency upgrade or
training run was performed. Existing PPO weights and training changes were
left intact during this dashboard task; `dotvenv.zip` was untouched.

The important finding is that clairvoyant is a full-horizon optimization LP,
not a submission Agent whose actions can reproduce the reference cost. The LP
can choose production, energy allocation, service and other auxiliaries in
addition to dispatches. Ordinary agents get only shipping and queue controls,
with production/allocation performed by simulator rules. Merely feeding the
LP's dispatch vector to the simulator would produce a different trajectory and
must not be labeled the scored clairvoyant plan.

### Implementation and accounting

Added `examples/09_compare_plans.py`, following the existing Fire keyword-flag
pattern. It supports any agent name/folder/agent.py, all task sizes, a scenario
root/index, information regime, reset/policy seeds, exact or quick naive,
audit-week selection, candidate-route count, optional GIFs and output path.
Outputs normally live under `outputs/09_compare_plans/<date_time>/`.

Added training/local-analysis-only `src/sbf_starter/diagnostics.py`. The tool:

1. Resolves one explicit scenario and uses its identical omega for the agent,
   prediction-free naive, and optimized reference. It records the scenario hash
   and seeds so the comparison can be reproduced.
2. Solves the original reference LP with the benchmark's oracle solver and
   refuses to label a nonoptimal result clairvoyant. `--quick` changes only the
   naive quantile replication count (2 instead of 1,000), never the LP solve.
3. Records complete agent/naive simulator trajectories with the existing episode
   recorder. The official LP replay checks must find each trajectory feasible
   and its simulator/LP accounting equal. Invalid/fallback actions are rejected
   before causal diagnostics rather than silently attributed to the agent.
4. Projects the actual optimized stock, dispatch, backlog, service, shedding,
   weekly component costs and terminal credit into the existing dashboard
   format. The record is explicitly labeled `clairvoyant_lp_projection`, not
   an agent replay, and its action arrays are explicitly non-executable.
   Exogenous map/warning signals use the same information view as the agent.
   LP lot identities, pipeline and WIP are unavailable and masked rather than
   copying the agent's unrelated state. Optimized queue totals appear in the
   full quantity export, not in the plain-stock heatmap.
5. Writes three standard dashboards and records, `comparison.png`,
   `cost_attribution.csv`, `route_decisions.csv`, `quantity_comparison.csv`,
   `lp_vectors.npz`, `summary.json`, `decision_audit.json`, and an offline,
   self-contained `index.html`. `--animation` adds three GIFs.
6. Decomposes all eight weekly cost components into actual edge/node LP terms
   for agent, naive and clairvoyant, preserving the official lane aggregation.
   Each component must reconcile to its recorded weekly cost. Terminal credit
   is reported separately. The route table includes zero/blocked routes,
   requested/executed/clipped quantities, optimized quantities, source stock,
   destinations, transit times, observed prohibitions/open fractions and the
   realized (hindsight) route state. Observed unknowns remain unknown.

The offline HTML includes side-by-side dashboards, cumulative/weekly costs,
component comparisons, a week/entity search, sortable tables, readable action
interventions and every raw-data download. It needs neither a web server nor
an external JavaScript service. Large tables render the first 600 filtered
matches for responsiveness; CSVs contain all rows. Embedded JSON escapes `<`
and text is inserted with `textContent` rather than treating agent labels as
HTML. Headless Chrome successfully rendered the Small dashboard; its initial
sandbox socket restriction required approved execution outside the sandbox.
The layout was inspected in a browser screenshot and the comparison PNG.

### Same-state causal interventions

Weekly cost gaps and different optimal flows are descriptive, not proof that
the current week's shipping decision was wrong. Inventories already differ,
optimal plans may not be unique, and shipping affects later weeks. To measure
a tested alternative, the audit rebuilds the original agent's complete prefix
and recurrent memory, snapshots its same pre-action simulator state, changes
only one week's action, then lets a copied original agent react through the
remaining episode. No future scenario information is passed to the actor.

Alternatives include naive's whole action recomputed at the agent's state,
individual naive quantities, a 25% smaller request, a request increased by
25% of nominal capacity, and default/held queue releases. Naive keeps its
initial reset-time plan; it is not copied from naive's separate trajectory.
Candidate routes exclude publicly known empty sources and observed masked
routes; unknown supply is not treated as zero. Default week selection ranks
net weekly hindsight cost gaps, and explicit `--audit_start` supports earlier
decisions. Selection is diagnostic hindsight and can miss causes outside the
selected weeks; it is not a deployable policy or an exhaustive action search.

For every audited week, an unchanged-action control must reproduce the final
cost in exact integer cents. All prefix actions are also checked against the
record. Failed deterministic replay/deepcopy is reported as an audit error,
not used to claim savings. A safe audit failure preserves accounting outputs
and reports failure rather than presenting an empty audit as successful.

Each result contains net episode savings, immediate savings, the first week
with a cost effect, a per-week savings trace, all cost-component savings,
terminal-credit change, exact old/new requests and executions, and changed
queue modes/overrides. Component savings plus credit must reconcile to total
savings within rounding tolerance. Different interventions' savings are not
additive. The report explicitly distinguishes requests with unchanged current
execution from physical dispatch changes: a changed request can alter later
PPO behavior through observation feedback without delivering extra cargo now.

### Corrections found during smoke runs

The first Tiny smoke exposed that internal stock slots include chokepoints,
but the dashboard's `stock.qty` contains only plain stocks. I corrected all
stock indexing through the flat layout's `(node, commodity)` positions and
kept optimized queues in the quantity export. A subsequent smoke showed that
step info does not repeat reset's `static`; the naive policy is now reset
once at the original initial state before replaying the prefix. I also used
the actual schema's edge `tau` (not `tau0`) and the scalar
`action_mask.observed[0]` (not a per-route observed array). Intervention
branches use the public core snapshot/restore and action/observation conversion
APIs, without changing the Gym adapter's private week cursor.

### Measured Small results and useful decision example

The exact-naive comparison on Small, dev episode 0, root 0, policy seed 0:

- PPO J: $3,539,512,276,730.44.
- Naive J: $3,935,505,276,240.43 (1,000 quantile replications).
- Clairvoyant LP J: $3,152,354,959,380.31; optimal status 0, highs-ipm.
- Single-scenario gap recovery: 0.5056411151. This is diagnostic only, not
  the leaderboard's harm-stratified/pooled RSS or a multi-scenario estimate.
- Extra PPO shortage cost against clairvoyant: $337,632,507,026.35.
- Extra PPO shedding cost against clairvoyant: $38,271,248,498.04.
  Other components and the terminal-credit difference complete the net gap.

The first three-high-gap-week quick audit (43, 44, 52) completed in about 14
seconds after model construction. Each no-change control saved exactly $0.
It found a week-52 whole-naive-action alternative saving $1,780,386.88; these
late-week results alone did not explain the much larger total opportunity gap.

I therefore also audited earlier weeks 17-22 against exact naive, initially
with four candidate routes: 90 total rollouts including controls, about 74
seconds after model construction. This exposed several request-only feedback
effects. I refined candidate selection to skip known empty sources and added
an explicit feedback-only explanation to the report.

The final earlier-week run, weeks 17-19 with three candidate routes, completed
36 paired intervention rollouts in about 35 seconds after model construction.
All three controls reproduced the original cost exactly. Its largest tested
improvement was week 17, slot 83, `sea.ct.osat_my.chk_taiwan`, mature chips
(`chip_mat`, lane 17): replacing PPO's request 36,256.139 with naive's
same-state request 175,554.429 increased actual execution from 977.965 to
1,621.102. Net episode savings were $6,409,361.58. Immediate saving was $0;
the first cost effect was week 18, with the large shortage benefit in week 19.
Shortage savings were $6,601,717.24, offset by additional transport/tariff and
other costs and a $3,686.66 reduction in terminal credit. This is a real
executed-dispatch change, not a request-only feedback effect. Other tested
week-19 and week-18 naive quantity substitutions saved about $5.64M and
$4.45M individually; these savings cannot be added or assumed to generalize.

Final report: `outputs/09_compare_plans/ppo_small_episode0_early/index.html`.
The larger earlier exploratory audit is preserved in
`outputs/09_compare_plans/ppo_small_episode0_exact/`; the initial quick view is
in `outputs/09_compare_plans/ppo_small_episode0/`. No PPO weights were changed
and no claimed achievement of the earlier 0.8-0.9 target follows from this
single-scenario diagnostic.

### Compatibility and verification

Added opt-in `--clairvoyant` / `--audit_weeks` to example 08, delegating to
example 09; its original loss-dashboard path remains unchanged. Updated the
README with usage, exports, replay semantics and limitations.

Executed local script checks, not pytest:

```bash
uv run --no-sync python examples/09_compare_plans.py --agent=template \
  --task=tiny --episode=0 --quick --audit_weeks=1 --route_candidates=1 \
  --out=outputs/09_compare_plans/tiny_smoke
uv run --no-sync python examples/09_compare_plans.py --agent=ppo \
  --task=small --episode=0 --audit_start=17 --audit_weeks=3 \
  --route_candidates=3 --out=outputs/09_compare_plans/ppo_small_episode0_early
uv run --no-sync python examples/09_compare_plans.py --agent=ppo \
  --task=full --entropy=20261009 --episode=0 --quick --audit_weeks=0 \
  --out=outputs/09_compare_plans/full_smoke
uv run --no-sync python examples/09_compare_plans.py --agent=random \
  --task=tiny --episode=29 --policy_seed=7 --quick --audit_start=5 \
  --audit_weeks=1 --route_candidates=1 --animation \
  --out=outputs/09_compare_plans/random_animation_smoke
uv run --no-sync python examples/08_agent_losses.py --agent=template \
  --task=tiny --episode=0 --quick --clairvoyant --audit_weeks=0 --n_jobs=1 \
  --out=outputs/09_compare_plans/wrapper_smoke
uv run --no-sync python examples/08_agent_losses.py --agent=template \
  --task=tiny --episode=0 --nonaive --n_jobs=1 \
  --out=outputs/09_compare_plans/legacy_losses_smoke
```

Actual commands used writable caches: `SBF_CACHE_DIR=/tmp/sbf-ppo-cache`,
`UV_CACHE_DIR=/tmp/sbf-uv-cache`, `MPLCONFIGDIR=/tmp/sbf-matplotlib`. The seeded
random-agent smoke passed its replay control and wrote all three GIFs. The
Full smoke passed reference replay and component reconciliation on its
104-week/large-network dimensions; its costs were PPO $5.923T, quick naive
$6.504T and optimal clairvoyant $4.743T (about 17.5 seconds for the LP solve).
These are smoke/one-scenario results, not Full leaderboard performance claims.

Additional direct checks verified all three matched scenario hashes, exact
cent net totals, every component attribution, zero unchanged-control savings,
masked unavailable LP fields, embedded HTML data and every report artifact
link. Ruff lint/format and `git diff --check` passed on the new/modified
dashboard code. No files in `tests/` were run, and all changes remain local.

## 19. History-aware heuristic and complete Small field coverage

Date: 2026-10-06. Request: update `agents/heuristic` to consider available
actions, forecasts, current state and previous weeks using
`docs/fields/small.md`, minimizing losses. This is a separate deterministic
controller change; neither the RL/PPO weights nor their training code changed.

### 19.1 Inspect and preserve the baseline

Read the complete Small field reference, the guide's interface/rules, the
existing heuristic, public instance tables, scoring/check code and installed
simulation/production/clip/queue/cost implementations. These were read-only
inspections; the benchmark package was not edited. The old heuristic used only
maximum capacities, masks and observed closures, ignoring forecasts and most
state. Preserved it in `outputs/heuristic_update/before/agent.py` for paired
comparison before changing the local source. The unrelated `dotvenv.zip` was
left untouched.

### 19.2 Replace the simple rule with a rolling planner

Implemented a sparse SciPy LP inside `agents/heuristic/agent.py`, importing
only standard library, NumPy and SciPy. Each week plans up to 24 weeks ahead
and executes only the first week's flow and queue decisions. All dispatch and
override slots enter the optimization, not just nominal-plan routes. It returns
all three action arrays; continuous quantities are optimized jointly, not
exhaustively enumerated. Shapes are read from `config['spaces']['action']` and
checked against public slot/release tables.

The model balances stocks, backlog, scheduled pipeline arrivals, existing WIP,
source replenishment, wafer-to-raw-chip production, packaging and grid fuel.
Ordinary dispatch is bounded by previous end-of-week stock; it cannot consume
today's later arrivals or source lift. Chokepoint arrivals are available before
tanker release. Shared edge, source stock, estimated downstream pool throughput
and extra-transit fleet limits couple routes. Costs include all eight published
components, with shortage/shed penalties taken from public economic tables.

The real simulator, not the submission, controls production and generation.
The model's production/energy variables are planning approximations, not
additional agent actions. New sea-lane trips are aggregated, existing container
queue releases are estimated, and future graph conditions persist unless an
announced reopening/prohibition supplies another date. This does not reproduce
the exact clairvoyant oracle or establish global optimality.

At the true episode end use public stock salvage. At earlier window boundaries
use bounded downstream continuation values and safety buffers, preventing
myopic draining of the supply chain. Default solve wall limit is 0.65 seconds.
An unsuccessful solve plays an internal nominal/reserve controller, records
`solver_failures` and exposes status/horizon/size in `last_plan`. This is
separate from the scorer's naive fallback.

### 19.3 Preserve and use history; respect hidden values

Store independent copies of every weekly observation/mask and returned action
in episode-local `history`/`actions`. Use all observed past forecast residuals
for shrunk demand-bias calibration, cumulative requested/executed quantities
for a weak reliability tie-break and throughput buffers, demand/served/lost
totals for sink reserves, all prior shed measurements for grid reserves, and
all eight accumulated cost components for bounded distress/waste adjustments.
The latest observed graph/stock/backlog survives a blackout, initialized from
public nominal values and reset prohibitions. Preserve previously announced
pending dates through missing coverage; an observed lifting supersedes them.

Every visible demand forecast column participates. Beyond forecast coverage
use public seasonal means with historical calibration. Region, dyad and
chokepoint warnings add soft route risk, not false hard sanctions. Live message
identifiers/masks select threads; target/region/commodity map affected routes,
channel/kind sets confidence, and announcement/effective weeks set age/urgency.
Withdrawals do not predict future events. Current cargo lists supersede older
ones: arrived pipeline entries must not be counted again merely because history
retains them. Prior raw observations are retained for inspection, not blindly
summed as fresh stock. `agents/heuristic/README.md` contains a field-by-field
coverage table and the precise approximation/interface limitations.

### 19.4 Development checks and corrections

Initial smoke checks exposed source stocks without holding/salvage fields and
release pairs without a declared commodity stock. Added safe public defaults
instead of assuming every raw node has every stock key. Read supply rates from
`stock[commodity]['supply_rate']`. Tiny has unpowered fabs (`grid=None`);
only powered fabs contribute grid draw. Both Tiny's padded lots and
Small/Full's dense cohort matrix are supported.

Added shared downstream edge/pool and fleet-divergence constraints after the
first version, plus dated incoming-cargo checks and FIFO cohort congestion.
This refinement improved the independent eight-scenario quick comparison
against the first LP version by 0.062718 (90% paired interval
0.030954–0.093060, rounded). These are development diagnostics, not exact
leaderboard results.

The complete Tiny diagnostics caught masked zero overrides: the flat converter
emits EVERY slot in a pair when mode 1 is used, including zero quantities.
Masked zeros are still logged as invalid. Corrected the agent to use custom
release only when all pair slots are visibly valid, otherwise default release
for a desired positive release or hold for a desired zero release. It cannot
encode arbitrary partial overrides in this flat action interface. No benchmark
converter was patched to conceal the limitation.

Added `examples/10_heuristic_diagnostics.py`, a self-contained Fire example
defaulting to Tiny. It records full NPZ episodes and JSON with actual costs,
planner statuses, internal fallbacks, CPU timing and history length. It checks
action space/dtypes, finite nonnegative values, known masks, observation
immutability and simulator validity. Its initial wrapper-constructor and
trajectory-property mistakes were corrected during smoke runs. A separate
masked-value check initially used an integer too large for int8; corrected the
check to use dtype-safe sentinels, not by changing agent semantics.

Updated the main README and the old `03_heuristic_agent.py` description: the
heuristic no longer acts identically to send-maximum in non-closure episodes.
The only final constructor addition after scoring was public shape validation;
it does not change decisions for valid Tiny/Small/Full configs.

### 19.5 Measured performance

All comparisons use the same scenarios for the new and preserved old agent,
standard regime, `n_jobs=2` and `cpu_budget=True`. Independent development
roots were 20261010/20261011. The final independent root was 20261012; no policy
tuning followed its results. Dev root 0 was used for final confirmation.

| Comparison | Updated | Old heuristic | Paired gain | 90% paired interval |
| --- | ---: | ---: | ---: | --- |
| Initial LP, 8 independent Small, quick, root 20261010 | 0.496864 | 0.342094 | +0.154770 | +0.0774 to +0.2394 |
| Intermediate, 20 independent Small, exact, root 20261011 | 0.5474 | 0.4156 | +0.1318 | +0.0839 to +0.1820 |
| Final, 64 independent Small, exact, root 20261012 | **0.615812** | **0.413256** | **+0.202556** | **+0.172224 to +0.236780** |
| Final, 20-episode Small dev, exact, root 0 | **0.644382** | **0.400373** | **+0.244009** | **+0.180722 to +0.312204** |

The final 64-scenario comparison reports mean cost $2,681,860,697,844.65 versus
$2,884,445,447,248.57, a **7.02% reduction**. Both final comparisons report
zero scorer fallback weeks. Quick numbers use rough naive references; the two
final rows use exact 1,000-replication naive references and exact oracles.
RSS is the scorer's reported score, not percentage demand served. These
measurements do not establish a 0.8–0.9 score or the minimum attainable loss.

Saved comparisons under `outputs/heuristic_update/` as
`initial_comparison.json`, `confirmation.json`,
`refinement_comparison.json`, `final_independent.json`, `final_dev.json`.

### 19.6 Submission and diagnostic validation

Isolated `sbf check heuristic` passed on Small and Full with allowed imports,
valid submissions and no over-budget weeks. Small week 1/max was 0.208 CPU
seconds, median 0.1208; Full week 1 was 0.503, median 0.3991 and max 0.563
seconds, against budgets 2 and 4 seconds. A final Small check repeated after
the shape-validation guard passed with week 1/max 0.245 seconds and median
0.1032 seconds. These are this machine's CPU measurements, not a
claim about another server or a Docker run.

Complete diagnostic episodes passed on Tiny, Small and Full, with no internal
solver fallbacks or invalid actions in the successful records. Standard Small
root 20261012 episodes 0/1 cost $2,450,978,795,244.99 /
$1,634,109,911,955.87; the same root's prediction-free episode 0 cost
$2,456,197,624,141.74 (a compatibility check, not an information-value study).
Tiny episode 0 cost $3,221,714,754.81. Full episode 0 cost
$2,152,939,164,958.07, with max weekly CPU 0.527 seconds. Task sizes/scenarios
differ, so these costs are not comparable across tasks. Diagnostic JSON/NPZ
files are in `tiny_diagnostics`, `small_diagnostics`, `full_diagnostics` and
`prediction_free_diagnostics` under the run directory.

Separate fresh-agent checks poisoned every hidden/padded value and verified
identical actions to tolerance: 30,836 entries on Tiny, 41,341 on Small,
59,707 on Full. They also verified separate episode histories and that changing
the caller's stock array cannot mutate stored history. These were direct
in-process checks, not pytest or execution of anything under `tests/`.

Generated the existing three-plan dashboard for the new heuristic on dev
Small episode 0, using exact references and no intervention audits:
`outputs/heuristic_update/dev_episode0_dashboard/index.html`.
Costs: heuristic **$3,439,687,116,611.93**, naive
**$3,935,505,276,240.43**, clairvoyant **$3,152,354,959,380.31**.
The dashboard's replay, component reconciliation and invalid-action checks
passed. This one-case comparison is not the multi-episode RSS above.

Reproduce with:

```bash
uv run --no-sync sbf compare heuristic outputs/heuristic_update/before \
  --task=small --episodes=64 --entropy=20261012 --n_jobs=2 --cpu_budget \
  --out=outputs/heuristic_update/final_independent.json
uv run --no-sync sbf compare heuristic outputs/heuristic_update/before \
  --task=small --episodes=dev --n_jobs=2 --cpu_budget \
  --out=outputs/heuristic_update/final_dev.json
uv run --no-sync sbf check heuristic --task=small
uv run --no-sync sbf check heuristic --task=full
uv run --no-sync python examples/10_heuristic_diagnostics.py --task=small --episodes=2
uv run --no-sync python examples/09_compare_plans.py --agent=heuristic \
  --task=small --episode=0 --audit_weeks=0
```

Commands used writable caches `UV_CACHE_DIR=/tmp/sbf-uv-cache` and
`SBF_CACHE_DIR=/tmp/sbf-ppo-cache`; dashboard rendering also used
`MPLCONFIGDIR=/tmp/sbf-matplotlib`. Lint, formatting and `git diff --check`
were run on changed source. No pytest, `tests/` scripts, Codabench mocks,
uploads, dependency upgrades, PyPI publication or `.venv` edits occurred.

## 20. Correct zero-transit fuel timing at the episode end (2026-10-10)

### 20.1 Request, diagnosis and baseline

The participant asked to remove avoidable end-of-episode excess from the
heuristic, following the final-week spike in the Full comparison dashboard.
The matched baseline is Full, standard regime, root 0, episode 1, policy seed
0, in `outputs/09_compare_plans/2026-10-09_22-15-33/`. Its scenario hash is
`b0518c7bd5e03119b31e114244b2add6adb7d6df21b6dbac783d1e9e636619fb`.
Saved the pre-change agent from Git HEAD to
`outputs/heuristic_end_fix/before/agent.py` using apply_patch, so independent
paired comparisons use the immediate previous policy, not the much older
baseline from section 19. The worktree was clean before this change.

Read the route LP, inventory/dispatch constraints, energy allocation and
installed simulator ordering. The heuristic's `_route` returned
`max(1, travel)` even for public zero-transit terminal-to-grid fuel edges.
The simulator dispatches first, then adds arrivals, then burns fuel for
generation: these transfers really can supply the current week. The false
delay made the LP undervalue final-week generation and plan terminal/grid
replenishment incorrectly before the end. All terminal fuel stocks were
already empty at the beginning of the baseline final week, so fixing only
the final action would be too late.

The earlier diagnosis also identified real US/India generation shocks during
the final week. A positive final-week cost remains even in the clairvoyant
solution; no artificial zero-loss terminal reward or scenario-specific action
was added. Scenario truth was used only for offline diagnosis, never as an
agent input.

### 20.2 Implementation

Changed `agents/heuristic/agent.py` to return the accumulated nonnegative
transit time, preserving zero rather than clamping it to one. Waiting for an
announced reopening still contributes to travel; positive-transit routes and
blocked-route handling are unchanged. This correction applies throughout
every episode, so the rolling planner can prepare fuel stocks before the final
week instead of adding a last-week-only patch.

The existing LP now places zero-transit cargo into the current-week inventory
balance before production. Its separate dispatch-stock constraint still
prevents ordinary same-week arrivals, supply lifts and new zero-transit
arrivals from being redispatched at an intermediate node. Gas rationing
continues to use previous-week grid stock, as the simulator does, rather than
allowing fresh arrivals to bypass the threshold. Horizon shrinking, stock
salvage, masks and release legality were not changed.

Updated `agents/heuristic/README.md` to explain the corrected ordering and
these constraints. No dependency, installed benchmark, training or weights
were changed. The fix needs only the usual public observations and metadata;
no new configuration parameters are required.

### 20.3 Focused checks and measured dashboard outcome

Ran direct in-process regression checks, not pytest or files under `tests/`.
A synthetic final-week case based on the recorded Full observation supplies
10,000 units of crude at the US terminal and none at its grid. To isolate
current-week generation from terminal-credit/holding degeneracy, the final
check sets synthetic salvage, holding and disposal to zero, removes storage
caps, and sets positive freight of $1,000 per unit on the zero-transit edge.
The old model then requests zero, whereas the corrected model requests
637.952846 units with zero solver fallbacks. Old/new modeled transit is
one/zero weeks. These artificial economics are only a mechanics regression,
not evaluation settings or tuning data.

Two initial variants incorrectly assumed that the old model would always
request zero even with free transport or salvage: it requested 10,000 units
in those variants. Corrected the *diagnostic assumption*, not the policy;
late cargo can have terminal-credit value or be an LP tie even when the old
model cannot use it for generation.

A second regression starts the terminal empty with 10,000 units scheduled to
arrive there this week. The corrected model requests zero terminal-to-grid
redispatch, and has no fallback. Also checked that all feasible strictly
positive-transit paths retain positive lead time.

Re-ran the complete updated heuristic from week 1 on the exact dashboard
scenario, not just its last 12 weeks. Generated the three-plan dashboard,
NPZ records and cost/quantity/route CSVs at
`outputs/heuristic_end_fix/full_episode1_dashboard/`. Scenario hashes match;
the dashboard's simulator validity and LP accounting reconciliation passed.
Quick affects only the rough naive reference; the oracle is solved exactly.

| Metric, same Full scenario | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Final-week net cost | $166.700851B | $131.508355B | $35.192496B (21.11%) |
| Final-week power shed | $148.195157B | $114.321511B | $33.873646B |
| Final-week shortage | $19.687031B | $18.368633B | $1.318398B |
| Total episode net cost | $7,620.960758B | $7,576.677914B | $44.282844B |

The exact clairvoyant final-week net cost is $117.046184B; the update closes
about 70.9% of the original final-week net-cost gap on this scenario. A
$14.462171B final-week gap remains. Imperfect production/queue/future-shock
modeling can still cause excess; this change does not establish that all
avoidable losses have disappeared or that Full RSS is 0.8–0.9. The separate
earlier last-12-week-only counterfactual is not the full-episode result above.

The detailed CSV shows that final-week shed now matches the oracle (up to
rounding) on Taiwan, Korea, China, Japan, India and Southeast Asia. Residual
excess shed is concentrated in the US ($7.538989B) and EU ($1.282129B).
Those gaps remain follow-up model-fidelity work, not a plotting correction or
evidence that the remaining cost is entirely unavoidable.

### 20.4 Submission checks and reproducible commands

Isolated submission checks passed on Small (max/week 1 0.238 CPU seconds,
median 0.1392) and Full (week 1 0.670, median 0.5478, max 0.738), against
2/4-second limits. These are local measurements, not Docker/server timings.
Ruff lint and format checks passed for the changed agent source.

A complete independent Tiny episode (root 20261013, episode 0) passed the
diagnostics script's shape/dtype, finite/nonnegative action, mask, observation
immutability, simulator validity and 26-week history checks. Cost was
$522,003,670.54, internal solver fallbacks zero, maximum weekly CPU 0.121
seconds. These costs cannot be compared with a different network/scenario.
The record and summary are in `outputs/heuristic_end_fix/tiny_diagnostics/`.

Commands use `UV_CACHE_DIR=/tmp/sbf-uv-cache` and
`SBF_CACHE_DIR=/tmp/sbf-ppo-cache`; rendering also uses
`MPLCONFIGDIR=/tmp/sbf-matplotlib`:

```bash
uv run --no-sync sbf check heuristic --task=small
uv run --no-sync sbf check heuristic --task=full
uv run --no-sync python examples/10_heuristic_diagnostics.py --task=tiny \
  --episodes=1 --entropy=20261013 --out=outputs/heuristic_end_fix/tiny_diagnostics
uv run --no-sync python examples/09_compare_plans.py --agent=heuristic \
  --task=full --entropy=0 --episode=1 --quick --audit_weeks=0 \
  --out=outputs/heuristic_end_fix/full_episode1_dashboard
uv run --no-sync sbf compare heuristic outputs/heuristic_end_fix/before \
  --task=full --episodes=8 --entropy=20261013 --quick --n_jobs=2 --cpu_budget \
  --out=outputs/heuristic_end_fix/full_independent_quick.json
uv run --no-sync sbf compare heuristic outputs/heuristic_end_fix/before \
  --task=small --episodes=16 --entropy=20261013 --n_jobs=2 --cpu_budget \
  --out=outputs/heuristic_end_fix/small_independent.json
uv run --no-sync ruff check agents/heuristic/agent.py
uv run --no-sync ruff format --check agents/heuristic/agent.py
git diff --check
```

Independent comparisons use a new root, 20261013, and identical scenarios for
both versions. Full uses quick reference quantiles and is explicitly not a
board-quality score; Small uses exact references. No pytest, `tests/` scripts,
Codabench mocks/uploads, PyPI writes or `.venv` modifications were performed.

### 20.5 Independent paired results

The eight-scenario Full quick comparison completed with updated score
0.677274684 versus 0.637656999 for the preserved immediate baseline. The
paired gain is 0.039617685, with a 90% interval [0.027265519, 0.050927170].
Mean episode cost fell from $6,253,020,166,997.27 to $6,125,129,995,689.77
(2.0453%). Cost decreased on all eight paired scenarios. Both versions had
zero scorer fallback weeks and zero over-budget CPU weeks. This is stronger
evidence than the selected dashboard case, but eight episodes with rough
naive quantiles and no harm strata are still not a Full leaderboard result.
Saved the full comparison, per-scenario costs and settings in
`outputs/heuristic_end_fix/full_independent_quick.json`.

The exact Small run needed fresh reference-cache preparation (its 1,000-draw
naive quantiles took 343.5 seconds before harm cut points and episode
references). Offered to hand off the already-completed checks/Full results or
wait for Small; the participant explicitly chose to wait. No further policy
changes were made while those paired evaluations were running.

Inspected the installed reference-generation path read-only to explain the
delay: exact evaluation computes 2,000 public harm-threshold draws before
the scenario references. A separate one-draw timing probe took 1.534 wall
seconds. Its first invocation used the wrong `shockbench_flow.tasks` import;
corrected that diagnostic to `shockbench_flow.hosting.tasks`. This probe does
not change the policy or replace any scored reference.

The exact 16-scenario Small comparison completed after that preparation.
Updated score is 0.692273445 versus 0.653835761 for the immediate baseline;
paired gain 0.038437684, 90% interval [0.031444032, 0.046626510]. Mean
episode cost fell from $2,433,996,171,224.08 to $2,386,331,777,818.50
(1.9583%). Cost decreased on all 16 paired scenarios, with savings from
$3.112185B to $78.599165B. Both versions had zero scorer fallback weeks and
zero over-budget CPU weeks. The run uses standard exact references and harm
thresholds (`quick=False`), but the reported score is overall-episode RSS
(`pooled=False`), not a full harm-stratified leaderboard estimate. No tuning
followed either independent comparison. Saved the per-episode comparison in
`outputs/heuristic_end_fix/small_independent.json`.

All final dashboard, compatibility diagnostic, submission-check and paired
comparison runs finished successfully. Final lint/format and `git diff --check` passed; only
the heuristic source, its README and this log have tracked changes. No claim
is made that the remaining US/EU end-week excess has been eliminated.

## 21. Full service-loss refinement (2026-10-10)

### 21.1 Request, baseline and diagnosis

The participant requested reducing shed and shortages toward the clairvoyant
reference, with the strongest attainable improvement on Full. Preserved the
exact current heuristic (including section 20's zero-transit correction) in
`outputs/heuristic_service_fix/before/agent.py`. The earlier source, README
and log changes were left intact. No dependency/package changes, PyPI writes,
uploads, Codabench mocks, pytest or scripts under `tests/` are permitted/used.

Inspected the standalone planner, Full field documentation, public static
network, installed simulator's dispatch/production/energy rules, public
forecast implementation and the existing three-plan Full episode-1 records.
These are read-only diagnostics; hidden episode marks and oracle actions are
not inputs to the agent. In the existing example, shortage cost is $1,822.65B
against $1,235.30B for the oracle, whereas shed is $5,715.10B against
$5,675.76B. Major shortage gaps are China mature chips, SEA leading chips,
US leading chips and other mature-chip sinks. Several energy-limited fabs
have large wafer stocks but little actual output. Simply sending more wafer
cannot fix a grid with no industrial headroom.

Identified model risks: reserves are predominantly rewarded at the artificial
end of the planning window, the LP chooses fuel segments independently even
though the simulator burns available segments pro rata, industrial production
can displace base load in the LP even at `base_first` grids, and observed
generation impairments are persisted over the entire future horizon. Public
forecasts have roughly 9–11% one-week relative error in this example, making
zero-inventory just-in-time service risky where spare capacity exists.

Two initial record-inspection commands guessed incorrect `FlatLayout` import
locations; corrected to `shockbench_flow.information.flat`. A read of a guessed
disruption scenario path failed; used the actual `policies/scenarios.py`.
These diagnostic mistakes did not change any policy or evaluation data.

### 21.2 Controlled candidate experiments

Added configurable candidate mechanisms: soft weekly fuel/terminal/service
inventory targets, proportional fuel-use planning, a base-first industrial
headroom constraint and future generation mean reversion. Experiments are on
an independent tuning root, 20261016, not dev root 0. Candidate mechanisms
start disabled until their actual full-episode costs are measured; infeasible
reserve targets use penalized slack, not hard constraints or oracle input.

Created a local experiment harness and parameter files under
`outputs/heuristic_service_fix/`. It plays complete standard-regime episodes,
collects actual eight-component USD costs, maximum weekly CPU, internal solver
fallbacks and invalid actions. It does not calculate RSS or substitute its own
scoring references. Its first parallel invocation failed because spawned
workers had not imported the environment registration module. Corrected by
importing it inside the worker; reran the experiment. No tests were involved.

### 21.3 Selection results and rejected alternatives

The first two tuning episodes favored weekly reserves; a rigid equality tying
all fuel burn to fixed shares was disastrous when one fuel was unavailable:
mean cost $13.257T against the baseline's $5.955T. Removed that mechanism.
Also tried a softer load-factor/segment-deficit model, first-week common fab
production ratios and automatic OSAT starts, base-first industrial headroom,
four-week generation recovery, normal-approximation shortage bands, larger
fuel targets and a 32-week horizon. The experimental source snapshots and
parameter/result JSON files remain in `outputs/heuristic_service_fix/v2/`,
`v3/`, `variants_1.json` through `variants_6.json`, and `sweep_*.json`.

Expanded three finalists and the immediate baseline to tuning episodes 0–3
of root 20261016. Selection uses actual total episode cost, not just shed:

| Candidate | Mean total USD | Mean shed USD | Mean shortage USD | Internal planner fallbacks |
| --- | ---: | ---: | ---: | ---: |
| Immediate baseline | $5,114.377B | $2,652.724B | $2,414.850B | 0 |
| Smaller reserves + recovery | $4,943.390B | $2,531.570B | $2,364.521B | 0 |
| Two-week fuel/terminal, 0.6-week service | $4,923.613B | $2,493.464B | $2,382.366B | 0 |
| Same reserves, 32-week horizon | $4,924.046B | $2,493.205B | $2,384.577B | 1 |

Retained the simplest lowest-cost finalist: a 24-week horizon with
`reserve_gain=0.1`, `fuel_reserve=2`, `terminal_reserve=2` and
`service_reserve=0.6`. Across these four tuning scenarios, mean total cost
decreased 3.7300% ($190.764B), shed 6.0036% and shortage 1.3452%.
All four paired costs decreased. The differences between neighboring
candidates are small, so this is a selected candidate, not a claim of a
globally optimal parameter set. Three-/four-week fuel buffers reduced shed
more but worsened shortages, and had higher total mean cost on the initial
two episodes. The other experimental mechanisms were removed from the
submission rather than being enabled on the basis of unproven benefits.

### 21.4 Final implementation and mechanical checks

Added optional weekly reserve-shortfall variables to the LP inventory rows.
Their penalty is 10% of the existing downstream marginal value, making
pre-positioning beneficial without forcing infeasible inventory or overriding
higher-valued immediate service. Each target is capped by public storage.
Grid fuel targets include two weeks of nominal segment burn plus the existing
gas rationing threshold; terminals cover two weeks of unique downstream-grid
fuel consumption. Sinks cover 0.6 weeks of the history-calibrated forecast.
Terminal targets taper to remaining weeks. Final-week grid/terminal targets
are zero; final-week sink protection remains because realized demand is noisy.
This preserves the section-20 zero-transit correction, actual dispatch stock
availability, masks, fleet/route constraints, previous-stock gas rationing
and true terminal credit. Fab/OSAT input stocks are not blindly inflated.

Cached terminal fuel rates at construction and deduplicated source/destination
pairs so alternative routes do not multiply consumption. The local
`verify_reserves.py` checks Tiny/Small/Full storage bounds, service targets,
final-week taper, zero-transit semantics and rate calculation, including a
synthetic duplicated route in a private config copy. No synthetic config is
used in any score. All checks passed.

A solver-options diagnostic showed that requesting HiGHS `threads=1` after
its SciPy scheduler was initialized with automatic threads returns status 4;
the same option works when first to initialize it. Because evaluation may
already have initialized that scheduler through reference/nominal solves,
did not change the submission's solver threading. Some initial parallel
experimental runs had an internal timeout/fallback (recorded in their JSON),
but the selected finalist's four tuning episodes did not. Ruff formatting
initially reported the new source needed formatting; formatted it and lint
passed. No claim is made that every rejected diagnostic run succeeded.

Isolated scorer-style checks passed: Full week 1 0.567 CPU seconds, median
0.4576, maximum 0.577 (budget 4 seconds); Small week 1/maximum 0.233,
median 0.1357 (budget 2 seconds). These are local, not Docker/server timings.
The check tool's generic suggestion to upload was not acted on.

A complete Small diagnostics episode on fresh root 20261018 passed action
shape/dtype/finiteness/sign, mask, observation immutability, simulator-validity
and full-history assertions. Cost $3,000,307,804,151.24, maximum weekly CPU
0.269 seconds, internal fallbacks zero; this standalone cost is not RSS.
Records are in `outputs/heuristic_service_fix/small_diagnostics/`.

### 21.5 Same-scenario three-plan dashboard

Replayed the final submission from week 1 on the previous Full screenshot's
root-0 episode 1, with exactly the same scenario hash
`b0518c7bd5e03119b31e114244b2add6adb7d6df21b6dbac783d1e9e636619fb`.
The oracle solve remained exact (19.79 seconds); naive quantiles were the
same rough two-draw setting as the prior dashboard, not a board reference.

| Metric | Immediate baseline | Updated | Exact oracle |
| --- | ---: | ---: | ---: |
| Episode total | $7,576.678B | $7,433.421B | $6,948.111B |
| Episode shed | $5,715.102B | $5,588.897B | $5,675.757B |
| Episode shortage | $1,822.653B | $1,804.219B | $1,235.298B |
| Final-week net | $131.508B | $122.542B | $117.046B |
| Final-week shed | $114.322B | $105.500B | $105.500B |

Episode savings are $143.256897B (1.8908%). Final shed now agrees with the
oracle within about $1,307 on a $105.5B total, including elimination of the
previously residual US/EU gap. The tiny remaining difference is at China
(0.000317 generation units); the other seven grids match exactly.
Episode shed is actually lower than the oracle: the oracle can trade extra
base-load shed for more industrial output. Lower shed alone is not the
scoring target, and the remaining $568.921B shortage excess is substantial.
This change does not eliminate unobserved-shock losses, real capacity
shortfalls or the simulator/oracle production-control difference.
The refreshed HTML/PNG/NPZ/CSV dashboard is
`outputs/heuristic_service_fix/full_episode1_dashboard/index.html`.

### 21.6 Independent holdout and reproducible commands

Froze the selected source before looking at holdout results. Prepared 16
new Full scenarios on root 20261017 using the package's explicit
`EpisodeSet.build(..., fq_replications=1000, cut_draws=0)` option. This keeps
standard exact-cost naive/oracle references but omits expensive 2,000-draw
harm-stratum thresholds. The package consequently labels the score
non-board/quick; it must not be described as a harm-pooled leaderboard score
or as a two-replication naive approximation. No reference/scoring formula
was modified. The completed paired results are recorded below.

Reproducible commands (use `UV_CACHE_DIR=/tmp/sbf-uv-cache`,
`SBF_CACHE_DIR=/tmp/sbf-ppo-cache`, and for rendering
`MPLCONFIGDIR=/tmp/sbf-matplotlib`):

```bash
# Initial ablations: substitute N=1,2,3,4,5 for each saved parameter file.
uv run --no-sync python outputs/heuristic_service_fix/sweep.py \
  --variants=outputs/heuristic_service_fix/variants_1.json \
  --out=outputs/heuristic_service_fix/sweep_1.json --jobs=3
# Finalists on additional tuning episodes, never the holdout root.
uv run --no-sync python outputs/heuristic_service_fix/sweep.py \
  --variants=outputs/heuristic_service_fix/variants_6.json \
  --out=outputs/heuristic_service_fix/sweep_6.json --start=2 --episodes=2 --jobs=3
uv run --no-sync python outputs/heuristic_service_fix/verify_reserves.py
uv run --no-sync sbf check heuristic --task=full
uv run --no-sync sbf check heuristic --task=small
uv run --no-sync python examples/10_heuristic_diagnostics.py \
  --task=small --entropy=20261018 --episodes=1 \
  --out=outputs/heuristic_service_fix/small_diagnostics
uv run --no-sync python examples/09_compare_plans.py --agent=heuristic \
  --task=full --episode=1 --entropy=0 --quick --audit_weeks=0 \
  --out=outputs/heuristic_service_fix/full_episode1_dashboard
uv run --no-sync python outputs/heuristic_service_fix/compare_exact_references.py \
  --task=full --episodes=16 --entropy=20261017 --n_jobs=2 \
  --out=outputs/heuristic_service_fix/full_holdout.json
uv run --no-sync ruff check agents/heuristic/agent.py
uv run --no-sync ruff format --check agents/heuristic/agent.py
git diff --check
```

Reference preparation was run separately before that paired comparison to
avoid regenerating the same in-progress cache. The four tuning episodes use
fixed local seed 0; the actual scorer-style folder comparison uses the
submission-hash-salted seeds and CPU meter. The heuristic's decisions are
deterministic; its seeded local RNG is not used to draw actions.

### 21.7 Completed independent Full results and limitations

The 16-scenario holdout completed successfully:

| Metric | Updated | Immediate baseline |
| --- | ---: | ---: |
| Overall-episode RSS | 0.669635960 | 0.634008048 |
| Mean episode cost | $6,496.056B | $6,611.365B |
| Scorer fallback weeks | 0 | 0 |
| Over-budget CPU weeks | 0 | 0 |
| Invalid entries | 0 | 0 |

Paired RSS improvement is **0.035627913**, 90% interval
**[0.018913698, 0.053206132]**. All 2,000 resampled episode sets favored the
updated version; this is a bootstrap result, not a guarantee or a Bayesian
probability. Mean episode savings are $115.308838B (1.7441%). Eleven of sixteen
individual costs improved; five worsened. The worst regression was $70.518573B
(episode 14); the largest saving was $359.228525B (episode 11). Those five
regressions demonstrate the real reserve-versus-service tradeoff, not a claim
that more shipments help every scenario. No parameter adjustment followed
these holdout results.

Mean reference costs were $8,663.316B naive and $5,426.841B clairvoyant. The
remaining updated cost gap is $1,069.215B per episode. RSS 0.8–0.9 is **not**
achieved on this independent Full set. Further improvements need better
automatic-production/energy prediction and shortage-aware allocation, rather
than blindly increasing fuel/material buffers again.

Results, all scenario costs, hashes, CPU/fallback/invalid/error counts and the
exact-reference settings are saved in
`outputs/heuristic_service_fix/full_holdout.json`. Its `quick=True` and
`pooled=False` flags are the package's non-board classification for
`cut_draws=0`. The generic CLI warning calls such a result a rough-naive quick
score, but in this explicit configuration the saved `fq_replications=1000`
shows that naive costs use the standard 1,000 replications; only harm-stratum
thresholds/pooling were omitted. The source/scoring implementation was not
patched to change this classification.

Final Ruff lint/format and `git diff --check` passed. Only the heuristic
source, its README and this log have tracked modifications (including the
preserved previous section-20 changes). Experiment snapshots and reports are
local gitignored artifacts. A documentation patch initially had a missing
diff-line prefix and was rejected atomically; corrected and reapplied it.
All final verification/evaluation processes finished. No pytest, `tests/`
execution, PyPI/dependency modifications, `.venv` edits or uploads were made.
