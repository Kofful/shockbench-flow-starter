# History-aware heuristic

`agent.py` is a deterministic, CPU-only submission. No training, package imports,
weights, hidden scenarios or clairvoyant results are needed during evaluation.
Optional numbers are loaded once from `params.json` beside the agent.

Every week it solves a sparse 24-week inventory/routing LP and executes only
the first week's transport and legal tanker-release decisions. The horizon
shrinks at the episode end. It compares all dispatch and override slots jointly,
including unused nominal routes, air alternatives, longer sea routes and holds.
Actions are continuous quantities: this is joint optimization, not an exhaustive
enumeration of every possible action vector.

The objective includes freight, tariffs, war risk, holding, queue holding,
shortage, disposal and power shed. Shared source stocks, edge capacities,
estimated downstream throughput and extra-transit fleet capacity constrain the
plan. Stock dispatched today cannot use supplies or ordinary arrivals that only
become available later today. Pipeline arrivals and WIP maturity are scheduled
by their observed weeks. Fab, packaging and fuel constraints link the stages.

## Field coverage

The reference is [docs/fields/small.md](../../docs/fields/small.md). Shapes and
indices come from public layout/static tables, not Small-specific constants.
All observations and returned actions, including masks, are copied into
episode-local `history` and `actions`. Earlier state is retained for inspection;
current own-state lists supersede older cargo lists, which must not be counted
again after they have arrived. Persistent graph beliefs, forecast-error
calibration, execution/service totals, loss totals and shed statistics carry
information from previous weeks into current decisions.

| Fields | Decision use |
| --- | --- |
| `week` | Absolute announcements/arrivals; rolling horizon and terminal credit. |
| `stock.qty`, `backlog.qty` | Dispatch availability, inventory balances and backlog carried to future weeks. Latest observed stock is retained if hidden; nominal stock initializes the belief. |
| `pipeline.edge/k/lane/qty/arrival_week` | Scheduled incoming inventory; tanker arrivals can join this week's chokepoint release, ordinary arrivals cannot be dispatched until next week. |
| `queue_lots.qty` and `layout.lot_keys` | Tanker queue content available to override/hold; container cohort order and pool congestion estimate automatic release timing. Tiny's per-lot format is handled separately. |
| `wip.node/k/qty/out_week` | Already-started production enters the correct output stock at maturity; never counted as immediately dispatchable. |
| `graph_now.u/c/tau` | Current capacity, freight and transit; last-observed beliefs survive blackout. |
| `graph_now.prohibited/tariff` | Legal routes and commodity-valued tariff costs. Reset prohibitions initialize missing graph entries. |
| `graph_now.open/kappa.tb/kappa.ct/war_risk` | Route closure response, shared pool throughput, queue holding and transit risk costs. |
| `graph_now.supply.avail` | Source/material replenishment, after current dispatch. |
| `graph_now.fab.R/alpha_bar/cap_eff` | Wafer starts, effective production and power draw constraints. |
| `graph_now.grid.G_bar/y_bar` | Available generation, base-load requirements, fuel stock and rationing constraints. |
| `graph_now.osat.R/thr_eff` | Packaging throughput and output delay. |
| `slot_mask`, `action_mask`, `action_mask.observed` | Current dispatch legality, distinct from closure/capacity. Scalar visibility flags are handled correctly. |
| `last_week.clip.requested/executed` | Episode-wide execution totals; weak route reliability tie-break and actual throughput contribution to continuation buffers. Stock clipping is not misinterpreted as a hard sanction. |
| `last_week.cost_components` | Cumulative distress versus transport/storage/disposal costs adjusts bounded safety buffers. All eight components contribute. |
| `last_week.sinks.demand/served/lost` | Forecast residual calibration and accumulated unmet service increase sink reserves. |
| `last_week.shed.qty` | All observed past grid-shed quantities contribute to fuel reserves. |
| `demand_forecast.qty` | Every visible horizon column, with shrunk historical forecast bias; public seasonal mean beyond coverage. |
| `warning.score`, `layout.warning_units`, `dyads` | Region, chokepoint and dyad route risk. Warnings affect soft costs, not hard legal masks. |
| `messages.msg_id/channel/kind/region/target_kind/target/k/announced_week/stated_effective_week` | Identify live threads and their targets/commodity; channel/kind confidence, age and effective date shape soft risk. Withdrawals do not predict new disruptions. |
| `pending_prohibitions.edge/k/effective_week` | Block cargo that would reach a route edge after its announced prohibition; remember previously announced dates through a blackout, superseded by observed lifting. |
| `closure_end.chokepoint/end_week` | Wait for announced reopening and restore nominal pool availability afterward. Unknown reopening is not invented. |
| `override_mask`, `override_mask.observed` | Custom tanker release only when the entire pair is visibly legal; otherwise automatic release or hold. |
| Every `<field>.observed` | Padded/hidden values are not facts. Latest observed graph/state entries or nominal priors fill supported beliefs; padded cargo/message rows are never used. |
| `flows/override_qty/release_mode` | All three action arrays are returned with their public dimensions and dtypes. |

Important flat-interface detail: `release_mode=1` sends **all** override slots
of its pair, including zero quantities. A masked zero is still logged as invalid.
The agent therefore uses the default release when only a legal subset exists
and the plan wants a positive release; it holds when the plan wants zero.
It cannot express an arbitrary legal subset through this flat API without
invalid entries. This constraint is specific to the public action interface.

## Limits and fallback

The LP is a forecast model, not an exact simulator or proof of minimum cost.
It aggregates new lane trips, estimates container queue delays, persists current
conditions and approximates future generation/production choices. The real
simulator chooses fab starts, energy allocation, pro-rata packaging and
container releases automatically; the agent cannot command those LP variables.
Pipeline already in transit can face unannounced disruptions. Replanning and
reserves reduce, but do not eliminate, this mismatch.

At the true episode end it uses public stock salvage. Earlier window ends use
bounded downstream continuation values, preventing artificial end-of-window
draining. Forecasts, warnings and announcements remain uncertain; no guaranteed
RSS or globally optimal loss is claimed.

An unsuccessful/timed-out LP uses a local nominal/reserve controller and
increments `solver_failures`. This internal fallback differs from the scorer's
naive fallback. `last_plan` exposes status, horizon and variable count. Solves
have a 0.65-second wall-clock limit, but import/build/CPU accounting still needs
the isolated server-style check.

## Run

```bash
uv run --no-sync sbf check heuristic --task=small
uv run --no-sync sbf evaluate heuristic --task=small --n_jobs=2 --cpu_budget
uv run --no-sync python examples/10_heuristic_diagnostics.py --task=small --episodes=4
uv run --no-sync python examples/09_compare_plans.py --agent=heuristic --task=small --episode=0 --quick
```

The diagnostics script writes complete EpisodeRecorder NPZ files and JSON
containing actual costs, weekly CPU times, planner statuses and internal
fallback counts. It verifies dimensions, finite nonnegative actions, known
masks, observation immutability and complete history. It does not calculate RSS;
use paired `sbf compare` with exact references for performance decisions.

Development outcomes and final measured comparisons are recorded in
[agents/rl/LOG.md](../rl/LOG.md), section 19. All changes are local; no pytest,
`tests/` scripts, Codabench calls or PyPI modifications were used.
