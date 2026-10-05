# ShockBench-Flow participant guide

What you need beyond the [README](../README.md): the interface in detail, the RL
wrappers, the rules, how the score is computed, how noisy a local score is, and
ideas for approaches. The last section tells the supply-chain story behind the
network; it is optional.

Contents: [the task](#the-task) · [the interface](#the-interface) ·
[Small and Full](#small-and-full) · [common mistakes](#common-mistakes) ·
[wrappers for RL](#wrappers-for-rl) · [rules](#rules) · [scoring](#scoring) ·
[local evaluation and its noise](#local-evaluation-and-its-noise) ·
[submitting](#submitting) · [approaches](#approaches) ·
[background (optional)](#background-optional)

## The task

An episode is T weeks (26 on Tiny, 52 on Small, 104 on Full). Each week your
agent chooses how much of each commodity to send along each route (an _action
slot_: an edge, or the first edge of a sea lane through one or more straits),
and, for tanker cargo waiting at a strait, whether it leaves by the default
rule, by your own quantities, or waits. The environment then clips your orders
to what is in stock and what the routes can carry this week, moves the goods,
runs the factories and power grids, serves demand and charges the week's cost.

The cost J of an episode is the sum over weeks of freight, war-risk surcharges,
tariffs, holding (higher for cargo queued at a strait), a penalty for every unit
of demand not served, disposal and power shed at the grids, minus the value of
what is left at the end. Lower is better. The disruptions of an episode
(closures, sanctions, tariffs, conflicts, factory outages) are drawn before it
starts from a public generator; nothing your agent does changes them.

**What your agent sees: the `standard` regime**, the one the leaderboards use.
Besides the network as it is this week, your own state and the demand forecast,
it gives three early signals of disruptions that have not acted yet:

- `warning.score`: an early-warning score per region, pair of rival regions and
  strait, with a one-week lag;
- `messages.*`: announcement threads (tariff proposals and final notices,
  sanction threats, military threats); some are false alarms that never take
  effect;
- `pending_prohibitions.*`: announced sanctions not yet in force, with the week
  each takes effect.

The naive rule that anchors the score sees none of these and ignores
disruptions, so using them well is where an agent can gain.

## The interface

```python
class Agent:
    def __init__(self, config=None):  # once per episode
        ...

    def act(self, observation):       # once per week
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
```

- `config` is a dict: `static` (the network's tables: `nodes`, `edges`, `lanes`,
  `commodities`, `action_slots`, `override_slots`, `sinks`, and the whole public
  instance under `static["instance"]`), `regime`, `T`, `policy_seed` (seed your
  random generators with it), `layout` (what each position of a dense
  observation block stands for), `release_modes` and `spaces` (every array's
  shape and dtype). Nothing hidden is in it.
- `observation` is a dict of numpy arrays with fixed shapes, keyed by strings
  (`observation["stock.qty"]`). Every field `x` comes with `x.observed`, 1 where
  the value is shown this week and 0 where it is hidden or padding. Lists of
  varying length (shipments in transit, messages) are padded to a fixed size.
- The action is a dict: `flows` (one quantity per action slot, 0 or more),
  `override_qty` (one per override slot) and `release_mode` (per strait and
  tanker commodity: 0 the default release, 1 your `override_qty`, 2 hold). The
  last two may be left out.
- `action_mask` is 1 on every slot that may carry goods this week (no sanction
  on its route). It does not show closures or capacity: read `graph_now.open`
  (how open each strait is, 1 to 0) and `graph_now.u` (each edge's capacity this
  week).
- The nominal weekly flows (the normal plan) are the week-0 shipments of
  `static["instance"]["initial_state"]["pipeline"]`.

Every field, with its shape, dtype, index set and meaning, is in
[fields/tiny.md](fields/tiny.md), [fields/small.md](fields/small.md) and
[fields/full.md](fields/full.md), generated from the installed package by
`uv run python scripts/fields_docs.py`. The fields that matter most at first:

| key                                           | what it is                                                                                                  |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `week`                                        | the week to decide, 1 to T                                                                                  |
| `stock.qty`                                   | stock on hand per (node, commodity), rows `layout["stock_slots"]`                                           |
| `backlog.qty`                                 | unserved demand carried at each market                                                                      |
| `graph_now.u`, `graph_now.c`, `graph_now.tau` | each edge's capacity, freight cost and lead time this week                                                  |
| `graph_now.open`                              | each strait's open fraction, rows `layout["chokepoints"]`                                                   |
| `graph_now.prohibited`, `graph_now.tariff`    | sanctions and tariffs per (edge, commodity)                                                                 |
| `demand_forecast.qty`                         | the demand forecast for the next 8 weeks                                                                    |
| `last_week.cost_components`                   | last week's cost by component (freight, war risk, tariff, holding, queue holding, shortage, disposal, shed) |
| `warning.score`                               | the early-warning scores (`standard` only)                                                                  |
| `action_mask`, `override_mask`                | the slots you may use this week                                                                             |

Under gymnasium your agent needs the `config` the server builds; `agent_config`
makes it from the reset:

```python
import gymnasium as gym
import shockbench_flow_gym
from shockbench_flow_agent import agent_config

env = gym.make("ShockBench/Small-v0")
obs, info = env.reset(options={"episode": 0})
agent = Agent(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
```

`gym.make` options: `regime` (`"standard"`, the scored one; `"prediction_free"`
hides the three signals), `dense_reward` (a shaped reward whose sum is still
minus the cost, up to a constant), `render_mode="rgb_array"`, `entropy` (the
scenarios' root: 0, the default, is the public dev root; any other integer below
2\*\*128 is a training root of your own) and `gamma` (the disruption intensity:
0.62, the default and the scored one, 0.79, 0.95, 0.97). The gymnasium
environment plays no fallback: an exception in `act` stops your script.

## Small and Full

Small (the public board's) and Full (the private board's) have the same keys as
Tiny except two blocks, stored compactly:

- `pipeline.*` is grouped: one entry per (edge, commodity, lane, arrival week),
  its `qty` the total of the shipments.
- The cargo queued at the straits is one dense array, `queue_lots.qty`, of shape
  (number of lot keys, T). Row `i` is `config["layout"]["lot_keys"][i]`, a
  (strait node, commodity, lane, next edge) tuple; column `w - 1` holds the
  quantity that reached the strait in week `w` and still waits.
- Tiny's per-lot lists (`queue_lots.lot_id`, `.chokepoint`, `.k`, ...) do not
  exist there: an agent that reads them raises `KeyError` every week. Test
  `"lot_keys" in config["layout"]` to tell the layouts apart.

Read every shape from `config["spaces"]`, never from Tiny's tables.

## Common mistakes

- **Indexing the observation by position.** `observation[2]` raises `KeyError`:
  it is a dict keyed by strings.
- **`__init__` without `config`.** The server calls `Agent(config)`;
  `def __init__(self)` raises, and naive plays the whole episode.
- **Returning the wrong thing.** `act` returns a dict with at least `flows`, a
  float array with one entry per action slot (20, 108 or 395); `release_mode` is
  an array, not a scalar.
- **Flows on sanctioned routes.** Entries on a prohibited slot, and negative or
  non-finite quantities, are dropped (the rest of the action stands). Multiply
  `flows` by `observation["action_mask"]`.
- **Trusting `action_mask` for closures.** What you send into a closed strait
  waits in its queue: check `graph_now.open`.
- **Imports the server lacks.** Only the standard library, numpy, scipy and
  torch exist there. `sbf check` fails an agent that imports anything else.
- **Files next to `agent.py`.** Load them relative to it:
  `Path(__file__).parent / "weights.npz"`.

## Wrappers for RL

`shockbench_flow_gym.wrappers`:

| wrapper                  | what it does                                                                                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CapacityFractionAction` | the action becomes a `Box` [0, 1] per slot: flows = fraction x the slot's nominal capacity x `action_mask`; all ones is "send the maximum"                      |
| `ScaleReward`            | divides the reward by naive's mean weekly cost on the dev split (`wrappers.REWARD_SCALE_USD`), so an average naive week is about -1                             |
| `ScenarioPool`           | draws N scenarios of your own root once, caches them on disk and serves them on reset (`options={"pool_index": i}` picks one); `draw_scenarios` fills the cache |
| `EpisodeRecorder`        | saves each finished episode to one `.npz` (read back by `load_record`, drawn by the dashboard)                                                                  |
| `SlimInfo`               | keeps only a few `info` keys, so vectorised training over subprocesses stays fast                                                                               |

They compose with gymnasium's own wrappers; the action and recorder wrappers go
inside any observation wrapper:

```python
import gymnasium as gym
from gymnasium.wrappers import FilterObservation, FlattenObservation, RescaleAction

import shockbench_flow_gym
from shockbench_flow_gym.wrappers import CapacityFractionAction, ScaleReward, ScenarioPool

PADDED = ("pipeline.", "queue_lots.", "wip.", "messages.", "pending_prohibitions.", "closure_end.")

env = gym.make("ShockBench/Small-v0")
env = ScenarioPool(env, n_scenarios=64, entropy=20261002)  # your own training root, drawn once and cached
env = ScaleReward(CapacityFractionAction(env))              # actions in [0, 1], reward in naive's weeks
env = FlattenObservation(FilterObservation(env, [k for k in env.observation_space.spaces if not k.startswith(PADDED)]))
env = RescaleAction(env, -1.0, 1.0)                         # what Stable-Baselines3's PPO expects
```

Most of the observation's size is the padded lists: dropping them (as above)
leaves 850 numbers on Tiny, 5,684 on Small and 17,803 on Full (masks included).
[05_train_ppo.py](../examples/05_train_ppo.py) trains PPO on this stack
and exports the policy as TorchScript (`torch.jit.load` in `agent.py`), since
the server has torch but not Stable-Baselines3.

## Rules

The competition's rules, as they bear on your code. Everything else in this
repository links here.

- **The boards.** The public board scores the submission your team keeps there
  on 200 private episodes of Small, with **2 s of CPU per week**. The private
  board re-scores that same submission automatically on 400 private episodes of
  Full, with **4 s of CPU per week**: you do not submit to it. Each team keeps
  one submission on the public board and may swap it for another at any time.
  The dates are on the competition page.
- **Submissions**: 3 per day (UTC) per team. Participation needs the organisers'
  approval.
- **Imports**: the server runs Python 3.13 with its standard library, numpy
  2.4.5, SciPy 1.18.1 and PyTorch 2.14.0 (CPU build, one thread), and nothing
  else, in `agent.py` or in any module it imports. No network, no GPU. Train
  with anything (external data and pretrained models are allowed), then ship
  weights and plain numpy or torch code. `sbf check` fails an agent that imports
  anything else.
- **Time**: the CPU budget above is metered from your container, every process
  and thread, and a week also has 10 s of wall clock. The container has 60 s to
  start (the interpreter and the import of `agent.py`, not metered), so load
  weights at module level: `Agent(config)` counts toward week 1. An episode
  stops 180 s (Small) or 480 s (Full) after its container starts.
- **The container**: one per episode, one CPU, 4 GB of memory, at most 128
  processes and threads, a read-only file system except a 256 MB `/tmp`, an
  unprivileged user. `print` goes to stderr; stdout belongs to the scorer, and
  no output of scored runs is shown.
- **Seeding**: one `Agent` per episode, nothing carries over between episodes.
  Seed every random generator from `config["policy_seed"]`; it is salted by the
  zip's SHA-256, so any byte change of the zip changes it (`sbf pack` writes
  repeatable bytes).
- **Fallbacks**: a week over the CPU budget or the wall clock, an exception or a
  malformed action is played by the naive rule from your own observation. It
  counts in your cost and in the board's Fallbacks column; it does not
  invalidate the submission.
- **Size**: at most 500 MB unpacked and 1,000 files; no symbolic links; no file
  named `predictions.json` at the root. The kit also refuses a zip over 100 MiB
  (a limit that may change).
- **Development**: language models and coding assistants are allowed while you
  develop; there is no limit on the compute you use off the platform. Never
  upload on someone's behalf without being asked: each upload spends one of the
  day's submissions.
- **Prizes**: $600, $400 and $250 (the competition page says how they are
  awarded). Winners release their code.

Local runs are a guide to the server's meter, not its count: `sbf check` times
your `act` in an isolated process on this machine, `sbf evaluate --cpu_budget`
hands the weeks over budget to the naive rule as the server does, and
`sbf check --docker` plays in a local copy of the scoring container with the
server's CPU meter. The scoring host is a Linux x86_64 server; on an Apple
silicon Mac the container runs under x86_64 emulation, so keep a margin.

## Scoring

For each episode n the scorer knows three costs: yours J, the naive rule's
J_naive and the clairvoyant plan's J_clairvoyant (a linear program that knows
the whole scenario in advance). Your saving is g_n = J_naive − J and the
attainable saving is D_n = J_naive − J_clairvoyant. The score, the Resilience
Skill Score (RSS):

- within a harm level s, the sum of the savings over the sum of the attainable
  savings of its episodes, Σ g_n / Σ D_n;
- overall, Σ_s p_s ḡ_s / Σ_s p_s D̄_s, with the mean savings ḡ_s and D̄_s of each
  harm level and the weights p = (0.50, 0.30, 0.15, 0.05).

The naive rule scores exactly 0 and the clairvoyant plan exactly 1; below 0 is
not clipped. The four harm levels sort scenarios by how much damage their
disruptions would do to the normal plan: level 1 holds the calmest half of the
generator's scenarios, level 4 the 5 % most harmful. The episode sets hold
equally many episodes per level and the weights restore the generator's mix. The
naive rule sees no warnings, whatever regime you play. An episode whose
clairvoyant plan is not solved exactly is left out, and the report says so.

## Local evaluation and its noise

`uv run sbf evaluate <agent>` (or `examples/04_evaluate.py`, the same function) scores
your agent on the public dev episodes of Tiny by default, with the scorer's own
computation: `--task=small` for the public board's network, `--task=full` for
the private board's. The dev split is 20 episodes, 5 per harm level. It builds
shockbench-flow's `EpisodeSet`, which computes the naive rule's and the clairvoyant
plan's costs of the episodes once and caches them on disk
(`~/.cache/shockbench-flow` or `SBF_CACHE_DIR`); afterwards a score costs one
run of your agent per episode. In Python:

```python
from shockbench_flow_agent import EpisodeSet

episodes = EpisodeSet.build("small", "dev", n_jobs=-1)  # references cached; all CPU cores
print(episodes.score("agents/mine", n_jobs=-1))          # parallel agent episodes
print(episodes.compare("agents/mine", "agents/other", n_jobs=-1))
```

The starter's `sbf evaluate` and `sbf compare` also parallelize the agent episodes
with `--n_jobs` (the package API above defaults to one worker). An agent can opt
into lock-step policy batching by defining the static method
`Agent.act_batch(agents, observations) -> actions`; then `--batch_size=32`
advances that many independent episodes per inference call. The exported PPO
agent implements it and reads `--device=auto|cpu|cuda|cuda:N`. Install a
CUDA-capable PyTorch without the training stack with `uv sync --extra cuda`.
CUDA accelerates policy inference only—the simulator and the SciPy/HiGHS
clairvoyant references remain CPU work. Runs with `--cpu_budget` use the normal
per-episode evaluator because a batched call cannot reproduce the server's
per-week CPU meter. CUDA and batched matrix kernels can round differently from
the board's CPU inference; use `--device=cpu --batch_size=1` (and `sbf check`)
when checking exact submission behavior.

- **The first run on a network is slow**: it computes the naive rule's demand
  model and the harm levels' cut points, and the clairvoyant plan of every
  episode. On Tiny it takes about a minute on a laptop. On the organisers' runs
  with 8 workers the naive rule's model alone took about 2.5 minutes on Small
  and on Full, and a clairvoyant plan takes seconds per episode on Small and
  more than a minute per episode on Full. Small is the practical local target.
- **`--quick`** (a rough naive rule, no harm levels, the first 4 episodes) runs
  in seconds; its numbers are not the board's. The examples' `--quick` means
  the same.
- **The dev split is small.** One standard error of a score over the choice of
  20 episodes is about 0.17 to 0.20. `evaluate` prints a 90 % interval; to tell
  whether a change helped, use
  **`sbf compare new old`**: both agents play the same episodes and the interval
  of the difference is paired, so the noise they share cancels. When it holds 0,
  the episodes cannot tell them apart.
- **Tune on your own root.**
  `uv run sbf evaluate mine --entropy=12345 --episodes=64` scores 64 episodes of
  a root of your own (their references cached too), and
  [06_policy_search.py](../examples/06_policy_search.py) trains on one.
  Keep the dev episodes for confirmation.
- **Same machine, same numbers.** Costs are reproducible to the cent on one
  machine type, not across CPU types, so your local score can differ from the
  server's in the last decimals.

**Reading the report.** The score on its 0 to 1 scale and its 90 % interval,
whether the harm levels are weighted as on the board (with `--quick` all
episodes count alike), the score per harm level, the mean cost per episode in
USD of your agent, the naive rule and the clairvoyant plan, and the weeks the
naive rule played for your agent (with the first error, when there was one).

## Submitting

1. `uv run sbf check <agent>` (a name, a folder or a zip; `--task=small` for the
   public board's budget): the server's validator (the same code the scorer
   runs, which never imports your agent), every file's imports against the
   scoring image, mistakes it accepts but that would hand weeks to the naive
   rule, and a timed run in a process that holds only the submission and the
   scoring image's packages, so a package of this repository cannot hide a
   missing import. Add `--docker` to run in a local copy of the scoring
   container (needs Docker; the first build downloads the pinned torch and SciPy
   wheels).
2. `uv run sbf upload <agent>` packs the agent into `outputs/<name>.zip`, runs
   the server's static checks and submits it. It reads two variables from the
   environment or from a `.env` file (copy `.env.example`; `.gitignore` lists
   `.env`):
   - `CODABENCH_COMPETITION`: the competition's URL. The tool talks to that
     URL's host only.
   - `CODABENCH_TOKEN`: your Codabench API token, never printed. Codabench's
     pages do not show it: `uv run sbf token` asks once for your username (or
     email) and password (not echoed, not stored) and saves the token in
     `.env`. An account created with "Sign in with GitHub" needs a password
     first (Codabench's password reset).

   `--dry_run` checks the zip, your token, your registration, the phase and your
   remaining daily submissions, and uploads nothing. `--wait` follows the run
   and prints the score. `--phase` names a phase (default: the one open now).
   It stops, without retrying, on anything Codabench refuses (registration
   pending, phase closed, daily limit); a Failed run does not count against
   Codabench's limits.
3. `uv run sbf status` lists your submissions and scores.

The upload uses Codabench's own web API with your account, in the order the web
page uses it, one request at a time; it is not a documented interface and may
change with Codabench. The web page always works: `uv run sbf pack <agent>`
writes the zip (`outputs/<name>.zip`, with `agent.py` at its root) to upload
there.

## Approaches

- **Heuristics and control.** Start from send-the-maximum and add rules that
  read the observation: strait closures (`graph_now.open`), sanctions
  (`action_mask`, `pending_prohibitions.*`), warnings, stock against demand.
  Measure each rule with `sbf compare`: a plausible rule can score below naive.
- **Planning (MPC).** The package's `mpc_det` baseline solves the clairvoyant
  plan's linear program over the next weeks on a forecast in which observed
  disruptions persist. SciPy's `linprog` (HiGHS) is on the server; mind the 2 s
  and 4 s CPU budgets.
- **Reinforcement learning.**
  [05_train_ppo.py](../examples/05_train_ppo.py): the wrappers above, PPO,
  and an export the server can run. Train on your own root (`ScenarioPool`),
  watch the CPU cost of your network per week.
- **Evolutionary and program search (AlphaEvolve style).**
  [06_policy_search.py](../examples/06_policy_search.py): candidates are
  submission folders (here the heuristic agent with its numbers in a
  `params.json`), the fitness is the score on cached references of your own
  root under the CPU budget, and the proposer is a mutation you can replace with
  a model that writes `agent.py`. Fitness on a few episodes is noisy: the example keeps
  its best only if it beats the starting point on the held-out dev split
  (`EpisodeSet.compare`, a paired interval).

## Background (optional)

The networks are stylised models of two real supply chains: energy (LNG, crude
oil, gas, nuclear fuel) from producing regions to power grids, and
semiconductors (wafers and neon, fabs, packaging plants) to markets in the US,
Europe, China and Japan. They are coupled: the grids power the fabs, so a fuel
shortage becomes a chip shortage weeks later. Several sea routes pass through
straits (Hormuz, Malacca, Taiwan, Suez, Panama, the Cape, the Turkish straits on
Small and Full), where a closure queues cargo and air or longer routes take over
at a higher cost.

The disruption generator draws closures of the straits, sanctions and export
controls on routes, tariff changes, conflicts between regions and outages of
factories, with rates and durations calibrated on historical records; each event
may be preceded by warnings and announcements, some of which are false alarms.
Tiny keeps the same mechanics on 12 nodes: LNG from three sources to two grids,
partly through one strait; wafers to three fabs; chips to one packaging plant
and one market, through the strait or by air around it.
