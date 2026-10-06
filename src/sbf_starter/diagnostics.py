"""Local hindsight dashboards and paired, single-decision intervention diagnostics.

Never imported by submission agents. The oracle record is an LP projection, NOT
a simulator replay: the reference can control auxiliaries an Agent cannot.
"""

from __future__ import annotations

import copy
import csv
import html
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from shockbench_flow.dynamics.env import Env
from shockbench_flow.dynamics.state import COST_COMPONENTS, cents
from shockbench_flow.information.flat import FlatLayout, flat_from_action, observation_arrays
from shockbench_flow.oracle.lp import lp_costs
from shockbench_flow.oracle.replay import replay, trajectory_vector
from shockbench_flow_agent.convert import action_to_wire, agent_config
from shockbench_flow_gym.env import ShockBenchFlowEnv
from shockbench_flow_gym.wrappers import EpisodeRecorder, load_record


def money(value, _position=None):
    for magnitude, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= magnitude:
            return f"${value / magnitude:.2f}{suffix}"
    return f"${value:,.2f}"


def _flat(layout, action):
    flow, override, modes = flat_from_action(layout, action)
    return {"flows": flow, "override_qty": override, "release_mode": modes}


def record_run(env, agent_class=None, *, policy=None, seed=0, episode=0, label="agent"):
    """Record either submission actions or a benchmark wire policy, retaining ground truth."""
    with tempfile.TemporaryDirectory(prefix="sbf-diagnostic-") as tmp:
        recorder = EpisodeRecorder(env, tmp, compress=False)
        obs, info = recorder.reset(seed=seed)
        layout = env.layout
        if policy is not None:
            policy.reset(info["static"], info["obs"], info["policy_seed"])
        else:
            actor = agent_class(agent_config(info["static"], info["policy_seed"], layout, obs))
        done = False
        while not done:
            action = _flat(layout, policy.act(info["obs"])) if policy is not None else actor.act(obs)
            obs, _, terminated, truncated, info = recorder.step(action)
            done = terminated or truncated
        record = load_record(recorder.paths[-1])
    record["meta"].update(agent=label, episode=episode, omega_hash=env.core.trajectory.omega_hash)
    return record, env.core.trajectory


def oracle_record(template, inst, model, oracle, marks):
    """Project actual optimized stocks/dispatches/costs into the standard plot format.

    Exogenous signals are the template's same information view. Unsupported own
    state (lot identities, pipeline and WIP) is explicitly masked, never copied
    from the template. Full LP variables are exported separately for inspection.
    """
    rec = copy.deepcopy(template)
    rec.pop("naive", None)
    T, nc = model.T, model.meta["nc"]
    point = oracle.x.reshape(T, nc)
    weekly, salvage = lp_costs(model, oracle.x)
    rec["costs"] = np.array([[getattr(w, c) for c in COST_COMPONENTS] for w in weekly])
    rewards = np.array([-cents(w.total()) for w in weekly], dtype=np.int64)
    rewards[-1] += cents(salvage)
    rec["reward_cents"], rec["reward"] = rewards, rewards.astype(float) / 100
    if -sum(map(int, rewards)) != oracle.J_cents:
        raise ValueError("oracle weekly cost/terminal credit failed to reconcile")
    obs = rec["obs"]
    unsupported = []
    for key in obs:
        if key.startswith(("pipeline.", "queue_lots.", "wip.", "last_week.")):
            obs[key][...] = 0
            if key.endswith(".observed"):
                unsupported.append(key.removesuffix(".observed"))
    rec["action"] = {k: np.zeros_like(v) for k, v in rec["action"].items()}
    # Week 1 stock/backlog are the simulator's identical initial state. Rows
    # 2..T+1 are LP end-of-week stocks/backlogs, not zero-action stocks.
    obs["stock.qty"][1:] = 0
    obs["stock.qty.observed"][...] = 1
    obs["backlog.qty"][1:] = 0
    obs["backlog.qty.observed"][...] = 1
    layout = FlatLayout.from_static(rec["static"])
    stock_positions = {pair: i for i, pair in enumerate(layout.stock_slots)}
    for field in ("clip.requested", "clip.executed", "sinks.served", "sinks.lost", "shed.qty"):
        obs[f"last_week.{field}.observed"][1:] = 1
    slots = {slot: i for i, slot in enumerate(inst.action_slots)}
    for j, column in enumerate(model.columns):
        tag, *key = column
        values = point[:, j]
        if tag == "I":
            stock_slot = inst.stock_slots[key[0]]
            obs["stock.qty"][1:, stock_positions[(stock_slot.node, stock_slot.k)]] = values
        elif tag == "B":
            obs["backlog.qty"][1:, key[0]] = values
        elif tag == "x" and tuple(key) in slots:
            s = slots[tuple(key)]
            rec["action"]["flows"][:, s] = values
            for field in ("requested", "executed"):
                obs[f"last_week.clip.{field}"][1:, s] = values
                obs[f"last_week.clip.{field}.observed"][1:, s] = 1
        elif tag in ("D", "U"):
            field = "served" if tag == "D" else "lost"
            obs[f"last_week.sinks.{field}"][1:, key[0]] = values
            obs[f"last_week.sinks.{field}.observed"][1:, key[0]] = 1
        elif tag == "ysh":
            obs["last_week.shed.qty"][1:, key[0]] = values
            obs["last_week.shed.qty.observed"][1:, key[0]] = 1
    obs["last_week.sinks.demand"][1:] = marks.demand
    obs["last_week.sinks.demand.observed"][1:] = 1
    obs["last_week.cost_components"][1:] = rec["costs"]
    obs["last_week.cost_components.observed"][1:] = 1
    represented = {
        "last_week.clip.requested",
        "last_week.clip.executed",
        "last_week.sinks.served",
        "last_week.sinks.lost",
        "last_week.sinks.demand",
        "last_week.shed.qty",
        "last_week.cost_components",
    }
    rec["meta"].update(
        agent="clairvoyant LP (not an agent replay)",
        regime="hindsight_lp_projection",
        record_kind="clairvoyant_lp_projection",
        J_cents=int(oracle.J_cents),
        salvage_cents=cents(salvage),
        oracle_status=int(oracle.status),
        oracle_solver=oracle.solver,
        unavailable_fields=[k for k in unsupported if k not in represented],
        projection_note="Optimized LP quantities; action arrays are NOT executable agent actions. "
        "Map/warnings show the same exogenous information view as the agent. Queue lot identities, "
        "pipeline and WIP are unavailable; optimized queue totals are in quantity_comparison.csv.",
    )
    return rec


def _entity(inst, column):
    tag, *key = column

    def node(n):
        return inst.nodes[n].id

    def commodity(k):
        return inst.commodities[k].id

    if tag == "x":
        e, k, lane = key
        return f"{inst.edges[e].id} / {commodity(k)} / lane {lane}"
    if tag in ("I", "O", "lift"):
        s = inst.stock_slots[key[0]]
        return f"{node(s.node)} / {commodity(s.k)}"
    if tag == "Q":
        return f"{node(key[0])} / {commodity(key[1])} / lane {key[2]}"
    if tag in ("D", "U", "B"):
        d = inst.demands[key[0]]
        return f"{node(d.node)} / {commodity(d.k)}"
    if tag in ("p", "E"):
        return node(inst.fabs[key[0]])
    if tag == "xi":
        return f"{node(inst.osats[key[0]])} / {commodity(key[1])}"
    if tag in ("y", "ysh", "G"):
        return node(inst.grids[key[0]]) + (f" / {commodity(key[1])}" if tag == "G" and key[1] is not None else "")
    return str(column)


def cost_attribution(inst, model, vectors, records):
    """Reconcile each weekly objective component to its exact edge/node LP terms."""
    nc, T = model.meta["nc"], model.T
    packed = {name: z.reshape(T, nc).copy() for name, z in vectors.items()}
    # Match the official formation of x_ek and queue totals across lanes.
    for _, members in model.meta["lane_groups"]:
        cols = [j for _, j in members]
        for z in packed.values():
            summed = np.array([math.fsum(row) for row in z[:, cols]])
            z[:, cols] = 0
            z[:, cols[0]] = summed
    rows = []
    for component, coefficients in model.meta["cost_parts"].items():
        coef = coefficients.reshape(T, nc)
        terms = {name: coef * z for name, z in packed.items()}
        for name, values in terms.items():
            expected = records[name]["costs"][:, COST_COMPONENTS.index(component)]
            if not np.allclose(values.sum(axis=1), expected, rtol=1e-9, atol=0.02):
                raise ValueError(f"{name} {component}: entity cost attribution does not reconcile")
        for t, j in np.argwhere(np.any(np.stack([v != 0 for v in terms.values()]), axis=0)):
            row = {"week": int(t + 1), "component": component, "entity": _entity(inst, model.columns[j])}
            row.update({f"{name}_usd": float(v[t, j]) for name, v in terms.items()})
            row["excess_vs_clairvoyant_usd"] = row["agent_usd"] - row["clairvoyant_usd"]
            row["excess_vs_naive_usd"] = row["agent_usd"] - row["naive_usd"]
            rows.append(row)
    return sorted(rows, key=lambda r: r["excess_vs_clairvoyant_usd"], reverse=True)


def quantity_comparison(inst, model, vectors):
    rows = []
    points = {name: z.reshape(model.T, model.meta["nc"]) for name, z in vectors.items()}
    tags = {
        "x": "edge flow",
        "p": "fab starts",
        "E": "fab energy",
        "xi": "packaging",
        "G": "generation",
        "y": "served grid load",
        "ysh": "shed load",
        "D": "served demand",
        "U": "lost demand",
        "B": "backlog",
        "O": "disposal",
        "Q": "queue",
        "I": "stock",
        "lift": "supply lift",
    }
    for j, column in enumerate(model.columns):
        if column[0] not in tags:
            continue
        for t in range(model.T):
            values = {name: float(z[t, j]) for name, z in points.items()}
            if not any(values.values()):
                continue
            rows.append({"week": t + 1, "kind": tags[column[0]], "entity": _entity(inst, column), **values})
    return rows


def route_comparison(inst, record, oracle, marks):
    rows = []
    layout = FlatLayout.from_static(record["static"])
    stock_positions = {pair: i for i, pair in enumerate(layout.stock_slots)}
    for t in range(inst.T):
        obs = {k: v[t] for k, v in record["obs"].items()}
        for s, (e, k, lane) in enumerate(inst.action_slots):
            req = float(record["action"]["flows"][t, s])
            executed = float(record["obs"]["last_week.clip.executed"][t + 1, s])
            ideal = float(oracle["action"]["flows"][t, s])
            sl = stock_positions[(inst.edges[e].tail, k)]
            path = [e] if lane is None else list(inst.lanes[lane].edges)
            chokepoints = sorted(
                {
                    inst.chokepoint_ordinal[inst.edges[edge].head]
                    for edge in path
                    if inst.edges[edge].head in inst.chokepoint_ordinal
                }
            )
            seen_opens = [float(obs["graph_now.open"][c]) for c in chokepoints if obs["graph_now.open.observed"][c]]
            visible_bans = [
                bool(obs["graph_now.prohibited"][edge, k])
                for edge in path
                if obs["graph_now.prohibited.observed"][edge, k]
            ]
            state = (
                "prohibited" if marks.prohibited[t, e, k] else "zero capacity" if marks.u[t, e] == 0 else "available"
            )
            rows.append(
                {
                    "week": t + 1,
                    "slot": s,
                    "entity": _entity(inst, ("x", e, k, lane)),
                    "destination": inst.nodes[inst.edges[path[-1]].head].id,
                    "requested": req,
                    "executed": executed,
                    "clipped": max(0.0, req - executed),
                    "clairvoyant_flow": ideal,
                    "flow_gap": executed - ideal,
                    "source_stock_before": float(obs["stock.qty"][sl]),
                    "nominal_transit_weeks": sum(inst.edges[edge].tau for edge in path),
                    "observed_route_min_open": min(seen_opens) if seen_opens else (1.0 if not chokepoints else None),
                    "observed_route_prohibition": any(visible_bans) if len(visible_bans) == len(path) else None,
                    "realized_route_min_open": min(float(marks.o[t, c]) for c in chokepoints) if chokepoints else 1.0,
                    "realized_edge_state": state,
                    "observed_capacity": float(obs["graph_now.u"][e]) if obs["graph_now.u.observed"][e] else None,
                    "observed_prohibition": bool(obs["graph_now.prohibited"][e, k])
                    if obs["graph_now.prohibited.observed"][e, k]
                    else None,
                    "realized_unit_transport_usd": float(
                        marks.c[t, e] + marks.c_wr[t, e, k] + marks.tariff[t, e, k] * inst.commodities[k].v
                    ),
                }
            )
    return rows


def _prediction_free(obs):
    return {**obs, **dict.fromkeys(("demand_forecast", "warning", "messages", "pending_prohibitions", "closure_end"))}


def audit_decisions(inst, omega, agent_class, record, baseline_policy, *, regime, seed, weeks, route_candidates=2):
    """Full-horizon paired interventions: same past/scenario, one changed action, adaptive future agent.

    Rank candidates using the current observation and naive action only. Selecting
    weeks by realized excess cost is diagnostic hindsight, not deployable policy.
    Rebuild recurrent memory by calling the agent on every unchanged prefix week.
    A no-change replay must match cents/actions before claiming causal savings.
    """
    results = []
    policy_seed = int(record["meta"]["policy_seed"])
    for week in weeks:
        print(f"auditing week {week}: one-week alternatives, then the agent resumes", flush=True)
        env = ShockBenchFlowEnv(inst, omega, regime, policy_seed=policy_seed)
        try:
            obs, info = env.reset(seed=seed)
            actor = agent_class(agent_config(info["static"], policy_seed, env.layout, obs))
            baseline_policy.reset(info["static"], _prediction_free(info["obs"]), policy_seed)
            for t in range(week):
                predicted = actor.act(copy.deepcopy(obs))
                original = {k: v[t].copy() for k, v in record["action"].items()}
                for k, values in original.items():
                    if not np.allclose(predicted.get(k, np.zeros_like(values)), values, rtol=1e-7, atol=1e-7):
                        raise ValueError("agent replay is not deterministic; intervention attribution is unsafe")
                if t + 1 < week:
                    obs, _, _, _, info = env.step(original)
            snap = env.core.snapshot()
            raw = _prediction_free(info["obs"])
            naive = _flat(env.layout, baseline_policy.act(raw))
            candidates = [("unchanged replay (control)", original, None), ("naive whole action", naive, None)]
            capacities = np.array([inst.edges[e].u0 for e, _, _ in inst.action_slots])
            gap = np.abs(naive["flows"] - original["flows"]) / np.maximum(capacities, 1)
            active = np.flatnonzero((naive["flows"] > 0) | (original["flows"] > 0))
            stock_index = {pair: i for i, pair in enumerate(env.layout.stock_slots)}
            supply_index = {pair: i for i, pair in enumerate(env.layout.supply_slots)}

            def can_dispatch(s):
                edge, commodity, _ = inst.action_slots[s]
                pair = (inst.edges[edge].tail, commodity)
                if obs["action_mask.observed"][0] and not obs["action_mask"][s]:
                    return False
                available = float(obs["stock.qty"][stock_index[pair]])
                sup = supply_index.get(pair)
                if sup is not None:
                    if not obs["graph_now.supply.avail.observed"][sup]:
                        return True  # unknown supply is not a known-zero source
                    available += float(obs["graph_now.supply.avail"][sup])
                return available > 1e-9

            active = [s for s in active if can_dispatch(s)]
            chosen = sorted(active, key=lambda s: gap[s], reverse=True)[:route_candidates]
            for s in chosen:
                for title, qty in (
                    ("naive quantity", naive["flows"][s]),
                    ("reduce request 25%", original["flows"][s] * 0.75),
                    ("increase request 25% capacity", original["flows"][s] + capacities[s] * 0.25),
                ):
                    if np.isclose(qty, original["flows"][s]):
                        continue
                    alt = {k: v.copy() for k, v in original.items()}
                    alt["flows"][s] = qty
                    candidates.append((title, alt, int(s)))
            # Queue control is a separate intervention, not hidden inside flow differences.
            for mode, title in ((0, "default all queue releases"), (2, "hold all queues for this week")):
                if len(original["release_mode"]) and np.any(original["release_mode"] != mode):
                    alt = {k: v.copy() for k, v in original.items()}
                    alt["release_mode"][...] = mode
                    alt["override_qty"][...] = 0
                    candidates.append((title, alt, None))
            for title, alternative, slot in candidates:
                branch = Env()
                restored, _ = branch.restore(snap)
                continuation = copy.deepcopy(actor)
                next_raw, _, terminated, truncated, _ = branch.step(
                    action_to_wire(env.layout, restored["week"], alternative)
                )
                next_obs = observation_arrays(env.layout, next_raw)
                intervention_rec = branch.trajectory.records[-1]
                while not (terminated or truncated):
                    next_raw, _, terminated, truncated, _ = branch.step(
                        action_to_wire(env.layout, next_raw["week"], continuation.act(next_obs))
                    )
                    next_obs = observation_arrays(env.layout, next_raw)
                traj = branch.trajectory
                if any(r.invalid for r in traj.records):
                    raise ValueError(f"invalid action in intervention {title}: cannot attribute cleanly")
                saving = (int(record["meta"]["J_cents"]) - traj.J_cents) / 100
                if title.startswith("unchanged") and traj.J_cents != int(record["meta"]["J_cents"]):
                    raise ValueError("unchanged full-horizon replay failed; no causal claims can be made")
                changes = []
                for s in np.flatnonzero(alternative["flows"] != original["flows"]):
                    changes.append(
                        {
                            "slot": int(s),
                            "entity": _entity(inst, ("x", *inst.action_slots[s])),
                            "old_request": float(original["flows"][s]),
                            "new_request": float(alternative["flows"][s]),
                            "old_executed": float(record["obs"]["last_week.clip.executed"][week, s]),
                            "new_executed": float(intervention_rec.executed.get(int(s), 0)),
                        }
                    )
                executed = np.array([intervention_rec.executed.get(s, 0.0) for s in range(env.layout.n_slots)])
                same_execution = np.allclose(
                    executed, record["obs"]["last_week.clip.executed"][week], rtol=1e-9, atol=1e-7
                )
                same_queue_controls = np.array_equal(alternative["release_mode"], original["release_mode"]) and (
                    np.array_equal(alternative["override_qty"], original["override_qty"])
                )
                before = record["costs"][week - 1 :].sum(axis=0)
                after = np.array(
                    [[getattr(r.costs, c) for c in COST_COMPONENTS] for r in traj.records[week - 1 :]]
                ).sum(axis=0)
                components = {c: float(before[j] - after[j]) for j, c in enumerate(COST_COMPONENTS)}
                credit_change = (traj.salvage_cents - int(record["meta"]["salvage_cents"])) / 100
                if not math.isclose(
                    math.fsum(components.values()) + credit_change, saving, rel_tol=1e-9, abs_tol=inst.T * 0.02
                ):
                    raise ValueError("intervention savings/components do not reconcile")
                delay = (
                    -record["reward_cents"][week - 1 :].astype(float)
                    + np.array(traj.rewards_cents()[week - 1 :], dtype=float)
                ) / 100
                changed_weeks = np.flatnonzero(np.abs(delay) > 0.02) + week
                results.append(
                    {
                        "week": int(week),
                        "alternative": title,
                        "slot": slot,
                        "entity": None if slot is None else _entity(inst, ("x", *inst.action_slots[slot])),
                        "saving_usd": saving,
                        "alternative_J_usd": traj.J_cents / 100,
                        "immediate_saving_usd": float(delay[0]),
                        "first_cost_effect_week": int(changed_weeks[0]) if len(changed_weeks) else None,
                        "component_savings_usd": components,
                        "terminal_credit_change_usd": credit_change,
                        "weekly_savings_usd": {str(t + week): float(v) for t, v in enumerate(delay)},
                        "changed_flows": changes,
                        "current_dispatch_execution_changed": not bool(same_execution),
                        "feedback_only_request_change": bool(changes and same_execution and same_queue_controls),
                        "changed_override_slots": [
                            int(i) for i in np.flatnonzero(alternative["override_qty"] != original["override_qty"])
                        ],
                        "changed_release_pairs": [
                            int(i) for i in np.flatnonzero(alternative["release_mode"] != original["release_mode"])
                        ],
                        "old_release_modes": original["release_mode"].tolist(),
                        "new_release_modes": alternative["release_mode"].tolist(),
                    }
                )
        finally:
            env.close()
    return sorted(results, key=lambda r: r["saving_usd"], reverse=True)


def verify_trajectories(model, trajectories):
    for label, traj in trajectories.items():
        result = replay(model, traj)
        if not result.feasible or not result.cost_equal:
            raise ValueError(f"{label} trajectory is inconsistent with the reference LP: {result}")
    return {label: trajectory_vector(model, traj) for label, traj in trajectories.items()}


def comparison_plot(records, path):
    fig = Figure(figsize=(15, 10), facecolor="#fcfcfb")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    palette = {"agent": "#2a78d6", "naive": "#eb6834", "clairvoyant": "#1baf7a"}
    T = len(records["agent"]["reward_cents"])
    weeks = np.arange(1, T + 1)
    for name, rec in records.items():
        weekly = -rec["reward_cents"].astype(float) / 100
        axes[0, 0].plot(weeks, weekly, label=name, color=palette[name])
        axes[0, 1].plot(
            weeks, np.cumsum(weekly), label=f"{name} J {money(rec['meta']['J_cents'] / 100)}", color=palette[name]
        )
        for c, style in (("shed", "-"), ("shortage", "--")):
            axes[1, 0].plot(
                weeks, rec["costs"][:, COST_COMPONENTS.index(c)], style, color=palette[name], label=f"{name} {c}"
            )
    names = [*COST_COMPONENTS, "terminal credit"]
    for i, (name, rec) in enumerate(records.items()):
        values = [*rec["costs"].sum(axis=0), -rec["meta"]["salvage_cents"] / 100]
        axes[1, 1].bar(np.arange(len(names)) + (i - 1) * 0.25, values, width=0.25, color=palette[name], label=name)
    axes[1, 1].set_xticks(range(len(names)), names, rotation=35, ha="right")
    for ax, title in zip(
        axes.flat,
        (
            "Weekly net cost (final week includes credit)",
            "Cumulative net cost",
            "Shed and shortage: costs, not learning losses",
            "Episode cost components",
        ),
        strict=True,
    ):
        ax.set_title(title, loc="left")
        ax.yaxis.set_major_formatter(FuncFormatter(money))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Same scenario, three plans · Clairvoyant = hindsight LP, not a replayable agent", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["week"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(out, summary, costs, routes, quantities, interventions):
    """An offline HTML dashboard: no server, third-party JavaScript or network."""
    payload = json.dumps(
        {
            "summary": summary,
            "costs": costs,
            "routes": routes,
            "quantities": quantities,
            "interventions": interventions,
        },
        allow_nan=False,
    )
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    title = html.escape(f"{summary['agent']} · {summary['task']} · episode {summary['episode']}")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TITLE</title>
<style>body{font:15px system-ui;background:#fcfcfb;color:#222;margin:24px auto;max-width:1500px;padding:0 20px}
h1{font-size:24px}h2{font-size:20px;margin-top:32px}.note{background:#fff2d6;padding:16px;border-radius:6px}
.cards,.gallery{display:flex;gap:16px;flex-wrap:wrap}.card{padding:16px;border:1px solid #ddd;border-radius:6px}
.gallery a{flex:1;min-width:300px}.gallery img{width:100%}img.chart{width:100%}input,select{padding:8px;margin:8px}
.scroll{max-height:540px;overflow:auto;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:12px}
th{position:sticky;top:0;background:#e8eef5;cursor:pointer}
td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}
tr:nth-child(even){background:#f5f5f5}details{margin:12px 0}pre{white-space:pre-wrap;font-size:12px}
.good{color:#087341}.bad{color:#b32929}a{color:#246cc0}</style>
<h1>TITLE</h1><p id="identity"></p><div id="cards" class="cards"></div>
<p class="note">Clairvoyant is the benchmark's full-horizon LP, not an agent replay. It knows future disruptions
and can choose production/energy allocations beyond the agent's shipping controls. Its stock and cost panels
come directly from the optimum; maps/warnings use the same signal view as your agent. Missing LP lot identities,
pipeline and WIP are masked. A different flow is not automatically a mistake: inventories differ and optimal plans
may not be unique. This is one scenario, not a leaderboard RSS estimate. Quick changes only the naive baseline.</p>
<img class="chart" src="comparison.png" alt="Three-plan cost comparison">
<h2>Same episode dashboards</h2><div class="gallery">
<a href="dashboard_agent.png">Your agent<img src="dashboard_agent.png" alt="Agent dashboard"></a>
<a href="dashboard_naive.png">Naive<img src="dashboard_naive.png" alt="Naive dashboard"></a>
<a href="dashboard_clairvoyant.png">Clairvoyant LP
<img src="dashboard_clairvoyant.png" alt="Clairvoyant dashboard"></a></div>
<h2>Tested decision alternatives</h2><p>Each experiment changes only one week's action from the same agent state
and scenario. The original agent then resumes and reacts to the changed state through the end of the episode.
Positive saving means this tested alternative helped; negative means it hurt. No-change controls verify replay.
Savings from different interventions are not additive. Alternatives tested are not an exhaustive optimum.
Week selection uses hindsight costs and can miss earlier causes. Expand an experiment for delayed shed/shortage,
terminal credit, and exact changed requests. Naive quantities here are recomputed at your agent's state,
not copied from naive's separate trajectory.</p><div id="audits"></div>
<h2>Weekly detail explorer</h2><label>Week <select id="week"><option value="">All weeks</option></select></label>
<label>Find entity/component <input id="find" placeholder="src_ru_gas, shed, fab..." type="search"></label>
<p>Click a table heading to sort. Amounts are USD. Quantity units depend on commodity (energy GWh, goods units);
they must not be added across commodities. Realized route state is hindsight; observed fields show what was visible.</p>
<h3>Exact costs by node/edge and component</h3><p>These accounting differences reconcile to weekly costs.
They locate the losses but do not, by themselves, identify the causal earlier action. Terminal credit is separate.</p>
<div id="costs" class="scroll"></div><h3>Requested vs executed vs optimized dispatch</h3>
<p>Clipping means a request was not fully executed, not an extra monetary penalty. Capacity, stock, shared fleet
or legal constraints may cause it. A prohibition on the entry edge is distinct from a closure later on its route.</p>
<div id="routes" class="scroll"></div><h3>Flows, production, energy, stock, backlog and disposal quantities</h3>
<p>Production and energy allocations are outcomes of simulator rules for the agent/naive, but optimized LP
variables for clairvoyant. Stock and queues are end-of-week values.</p><div id="quantities" class="scroll"></div>
<h2>Raw data</h2><p><a href="summary.json">Summary</a> · <a href="cost_attribution.csv">Cost attribution CSV</a> ·
<a href="route_decisions.csv">Routes CSV</a> · <a href="quantity_comparison.csv">Quantities CSV</a> ·
<a href="decision_audit.json">Interventions JSON</a> · <a href="lp_vectors.npz">Full LP vectors</a> ·
<a href="record_agent.npz">Agent record</a> · <a href="record_naive.npz">Naive record</a> ·
<a href="record_clairvoyant.npz">Projected LP record</a></p>
<script type="application/json" id="data">PAYLOAD</script><script>
const d=JSON.parse(document.getElementById('data').textContent),s=d.summary;
const usd=x=>new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(x);
document.getElementById('identity').textContent=
`Root ${s.entropy}; policy seed ${s.policy_seed}; scenario ${s.omega_hash}; quick naive: ${s.quick}`;
for(const [name,cost] of Object.entries(s.cost_usd)){let e=document.createElement('div');e.className='card';
e.textContent=`${name}: ${usd(cost)}`;document.getElementById('cards').append(e);}
let e=document.createElement('div');e.className='card';
const recovery=s.gap_recovery===null?'undefined':(s.gap_recovery*100).toFixed(2)+'%';
e.textContent=`Single-scenario gap recovery: ${recovery} (not board RSS)`;
document.getElementById('cards').append(e);
for(const a of d.interventions){let detail=document.createElement('details'),head=document.createElement('summary');
head.textContent=`Week ${a.week}: ${a.alternative}${a.entity?' · '+a.entity:''} · saving ${usd(a.saving_usd)}`;
head.className=a.saving_usd>0?'good':a.saving_usd<0?'bad':'';detail.append(head);
let explanation=document.createElement('p');explanation.textContent=
`Immediate saving ${usd(a.immediate_saving_usd)}; final saving ${usd(a.saving_usd)}; `+
`first cost effect ${a.first_cost_effect_week===null?'none':'week '+a.first_cost_effect_week}. `+
`Terminal credit change ${usd(a.terminal_credit_change_usd)}.`;detail.append(explanation);
if(a.feedback_only_request_change){let note=document.createElement('p');note.className='note';note.textContent=
'The request changed, but current executed dispatches and queue controls did not. Any later savings came through '+
'feedback and the agent reacting differently later, not extra cargo delivered in this week.';detail.append(note);}
let breakdown=document.createElement('p');breakdown.textContent='Savings by component: '+
Object.entries(a.component_savings_usd).map(([k,v])=>`${k}: ${usd(v)}`).join(' · ');detail.append(breakdown);
let changes=document.createElement('ul');for(const c of a.changed_flows){let li=document.createElement('li');
li.textContent=`Slot ${c.slot}, ${c.entity}: request ${c.old_request.toFixed(3)} → ${c.new_request.toFixed(3)}; `+
`executed ${c.old_executed.toFixed(3)} → ${c.new_executed.toFixed(3)}`;changes.append(li);}detail.append(changes);
let raw=document.createElement('details'),label=document.createElement('summary'),pre=document.createElement('pre');
label.textContent='Full experiment data, including per-week effects and queue modes';raw.append(label);
pre.textContent=JSON.stringify(a,null,2);raw.append(pre);detail.append(raw);
document.getElementById('audits').append(detail);}
if(!d.interventions.length)document.getElementById('audits').textContent=
s.audit_error?`Audit failed: ${s.audit_error}`:'No interventions requested (--audit_weeks=0).';
for(let i=1;i<=s.weeks;i++){let o=document.createElement('option');o.value=i;o.textContent=i;
document.getElementById('week').append(o);}
let sorting={};function render(key){const host=document.getElementById(key);host.replaceChildren();
const selected=document.getElementById('week').value;
const find=document.getElementById('find').value.toLowerCase();
let rows=d[key].filter(r=>(!selected||r.week===Number(selected))&&JSON.stringify(r).toLowerCase().includes(find));
const cols=Object.keys(d[key][0]||{});
if(sorting[key]){const [col,dir]=sorting[key];rows.sort((a,b)=>dir*
(typeof a[col]==='number'&&typeof b[col]==='number'?a[col]-b[col]:String(a[col]).localeCompare(String(b[col]))));}
let table=document.createElement('table'),head=document.createElement('tr');
for(const col of cols){let th=document.createElement('th');th.textContent=col;
th.onclick=()=>{sorting[key]=[col,sorting[key]?.[0]===col?-sorting[key][1]:-1];render(key)};
head.append(th);}table.append(head);
for(const row of rows.slice(0,600)){let tr=document.createElement('tr');
for(const col of cols){let td=document.createElement('td'),v=row[col];
td.textContent=v===null?'unobserved / none':typeof v==='number'?
v.toLocaleString('en-US',{maximumFractionDigits:3}):String(v);tr.append(td);}
table.append(tr);}host.append(table);
let p=document.createElement('p');p.textContent=
`${rows.length} matches; showing first ${Math.min(rows.length,600)}. CSV exports include every row.`;
host.append(p);}
function all(){['costs','routes','quantities'].forEach(render)}
document.getElementById('week').onchange=all;document.getElementById('find').oninput=all;all();
</script></html>"""
    (out / "index.html").write_text(page.replace("TITLE", title).replace("PAYLOAD", payload))
