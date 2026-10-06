"""Offline dashboards and decision diagnostics for agent, naive and clairvoyant.

    uv run python examples/09_compare_plans.py --agent=ppo --task=small --episode=0 --quick
    uv run python examples/09_compare_plans.py --agent=rl --episode=29 --audit_weeks=0
    uv run python examples/09_compare_plans.py --agent=ppo --task=small --entropy=12345 --episode=0

Open the printed index.html locally. No server, upload or external service is
needed. Quick uses a rough naive reference; the clairvoyant LP is always solved
exactly. Single-scenario gaps are diagnostics, not the leaderboard's pooled RSS.
"""

import json
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
from shockbench_flow.evaluation.results import oracle_optimal
from shockbench_flow.marks import compute_marks
from shockbench_flow.oracle.lp import build_lp, solve_oracle
from shockbench_flow.policies.naive_fq import anchor_policy, fallback_spec
from shockbench_flow_agent import NAIVE_REPLICATIONS, QUICK
from shockbench_flow_gym import dashboard
from shockbench_flow_gym.env import ShockBenchFlowEnv

from sbf_starter import env_id
from sbf_starter.agents import load
from sbf_starter.diagnostics import (
    audit_decisions,
    comparison_plot,
    cost_attribution,
    money,
    oracle_record,
    quantity_comparison,
    record_run,
    route_comparison,
    verify_trajectories,
    write_csv,
    write_report,
)
from sbf_starter.play import episodes_with_closure


def main(
    agent: str = "ppo",
    task: str = "tiny",
    episode: int | None = None,
    entropy: int = 0,
    search: int = 60,
    min_open: float = 0.5,
    quick: bool = False,
    n_jobs: int = 1,
    regime: str = "standard",
    seed: int = 0,
    policy_seed: int = 0,
    audit_weeks: int = 3,
    audit_start: int | None = None,
    route_candidates: int = 2,
    animation: bool = False,
    out: str | None = None,
) -> None:
    """Compare three plans on exactly one scenario and audit selected decisions.

    Args:
        agent: agent name, submission folder or agent.py (default ppo).
        task: tiny, small or full (all dimensions come from the instance).
        episode: scenario index; omitted searches for a chokepoint closure.
        entropy: scenario root; zero is dev, use your own root for tuning.
        search: episode search limit when episode is omitted.
        min_open: closure threshold for scenario selection.
        quick: rough naive quantiles only; oracle always exact, not quick RSS.
        n_jobs: workers for initial naive quantile calculation.
        regime: agent information regime; naive is always prediction-free.
        seed: Gym reset seed; does not change the explicit scenario index.
        policy_seed: local agent seed, shared by all compared simulator runs.
        audit_weeks: number of weeks to audit; zero skips interventions.
        audit_start: first audit week (1-based); omitted ranks net cost gaps.
        route_candidates: routes per audit week, each tested at naive/-25%/+25% capacity.
        animation: also write three week-by-week GIFs (slower).
        out: output folder (default outputs/09_compare_plans/<date_time>).

    """
    if audit_weeks < 0 or route_candidates < 0:
        raise ValueError("audit_weeks and route_candidates must be nonnegative")
    out = Path(out or f"outputs/09_compare_plans/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    agent_class = load(agent)
    registered = gym.make(env_id(task), regime=regime, entropy=entropy, policy_seed=policy_seed)
    try:
        if episode is None:
            (episode,) = episodes_with_closure(registered, 1, search=search, min_open=min_open, seed=seed)
        if isinstance(episode, bool) or int(episode) != episode or episode < 0:
            raise ValueError("episode must be a nonnegative integer")
        episode = int(episode)
        inst = registered.unwrapped.instance
        omega = registered.unwrapped.omega_source(episode)
    finally:
        registered.close()
    if audit_start is not None and not 1 <= audit_start <= inst.T:
        raise ValueError(f"audit_start must be in 1..{inst.T}")

    print(f"{task}, episode {episode}, root {entropy}: solving exact clairvoyant LP", flush=True)
    marks = compute_marks(inst, omega)
    model = build_lp(inst, marks)
    started = time.perf_counter()
    oracle = solve_oracle(model)
    if not oracle_optimal(oracle):
        raise RuntimeError(f"clairvoyant LP not optimal (status {oracle.status}): {oracle.message}")
    print(f"clairvoyant cost {money(oracle.J_cents / 100)}; solve {oracle.seconds:.2f}s", flush=True)

    replications = QUICK["fq_replications"] if quick else NAIVE_REPLICATIONS
    params = dashboard.scenario_params(inst, omega)
    print(f"loading/computing naive quantiles ({replications} replications)", flush=True)
    fq_quantiles(inst, params, replications, n_jobs=n_jobs, cache_dir=default_cache_dir())
    fallback = fallback_spec(inst, params, replications, n_jobs)
    records, trajectories = {}, {}
    for name in ("agent", "naive"):
        env = ShockBenchFlowEnv(
            inst, omega, regime if name == "agent" else "prediction_free", policy_seed=policy_seed, fallback=fallback
        )
        try:
            records[name], trajectories[name] = record_run(
                env,
                agent_class,
                policy=anchor_policy(inst, params, replications, n_jobs) if name == "naive" else None,
                seed=seed,
                episode=episode,
                label=agent if name == "agent" else "naive",
            )
        finally:
            env.close()
        print(f"{name}: {money(records[name]['meta']['J_cents'] / 100)}", flush=True)
    if any(r.invalid for traj in trajectories.values() for r in traj.records):
        raise ValueError("recorded invalid/fallback actions; resolve them before causal diagnosis")
    vectors = verify_trajectories(model, trajectories) | {"clairvoyant": oracle.x}
    records["clairvoyant"] = oracle_record(records["agent"], inst, model, oracle, marks)
    for name, rec in records.items():
        rec["meta"].update(entropy=entropy, quick_naive=bool(quick))
        if name != "naive":
            rec["naive"] = {key: records["naive"][key] for key in ("reward_cents", "costs")}
            rec["meta"]["naive_J_cents"] = records["naive"]["meta"]["J_cents"]
        dashboard.save_record(rec, out / f"record_{name}.npz")
        dashboard.episode_dashboard(rec, out / f"dashboard_{name}.png")
        if animation:
            dashboard.episode_animation(rec, out / f"episode_{name}.gif")
    dashboard.plot_network(task).savefig(out / "network.png", dpi=100)
    comparison_plot(records, out / "comparison.png")
    np.savez_compressed(
        out / "lp_vectors.npz",
        **vectors,
        columns_json=np.array(json.dumps(model.columns)),
        omega_hash=np.array(omega.hash),
    )

    costs = cost_attribution(inst, model, vectors, records)
    routes = route_comparison(inst, records["agent"], records["clairvoyant"], marks)
    quantities = quantity_comparison(inst, model, vectors)
    write_csv(out / "cost_attribution.csv", costs)
    write_csv(out / "route_decisions.csv", routes)
    write_csv(out / "quantity_comparison.csv", quantities)
    weekly_gap = (
        -records["agent"]["reward_cents"].astype(float) + records["clairvoyant"]["reward_cents"].astype(float)
    ) / 100
    if audit_start is None:
        weeks = sorted((np.argsort(-weekly_gap, kind="stable")[:audit_weeks] + 1).tolist())
    else:
        weeks = list(range(audit_start, min(inst.T + 1, audit_start + audit_weeks)))
    # All accounting outputs survive if a user agent cannot be safely replayed.
    summary = {
        "agent": agent,
        "task": task,
        "episode": episode,
        "entropy": entropy,
        "weeks": inst.T,
        "regime": regime,
        "seed": seed,
        "policy_seed": policy_seed,
        "omega_hash": omega.hash,
        "quick": bool(quick),
        "naive_replications": replications,
        "cost_usd": {name: rec["meta"]["J_cents"] / 100 for name, rec in records.items()},
        "cost_components_usd": {
            name: dict(zip(rec["meta"]["cost_components"], rec["costs"].sum(axis=0).tolist(), strict=True))
            for name, rec in records.items()
        },
        "terminal_credit_usd": {name: rec["meta"]["salvage_cents"] / 100 for name, rec in records.items()},
        "oracle_status": oracle.status,
        "oracle_solver": oracle.solver,
        "oracle_seconds": oracle.seconds,
        "audit_weeks": weeks,
        "agent_salvage_usd": records["agent"]["meta"]["salvage_cents"] / 100,
        "gap_recovery": None,
        "audit_error": None,
        "limitations": [
            "single-scenario diagnostic, not pooled board RSS",
            "LP auxiliaries are not all available as agent controls",
            "week cost differences and flow differences are not causal blame",
            "interventions are hindsight-selected, limited, and not additive",
        ],
    }
    headroom = summary["cost_usd"]["naive"] - summary["cost_usd"]["clairvoyant"]
    if headroom > 0:
        summary["gap_recovery"] = (summary["cost_usd"]["naive"] - summary["cost_usd"]["agent"]) / headroom
    interventions = []
    try:
        interventions = audit_decisions(
            inst,
            omega,
            agent_class,
            records["agent"],
            anchor_policy(inst, params, replications, n_jobs),
            regime=regime,
            seed=seed,
            weeks=weeks,
            route_candidates=route_candidates,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        summary["audit_error"] = f"{type(exc).__name__}: {exc}"
        print(f"Intervention audit unavailable: {summary['audit_error']}", flush=True)
    summary["elapsed_seconds"] = time.perf_counter() - started
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "decision_audit.json").write_text(json.dumps(interventions, indent=2, allow_nan=False) + "\n")
    write_report(out, summary, costs, routes, quantities, interventions)
    print(f"Open {out.resolve() / 'index.html'}", flush=True)
    for item in [r for r in interventions if r["saving_usd"] > 0.02][:3]:
        print(f"Tested improvement: week {item['week']}, {item['alternative']}, saving {money(item['saving_usd'])}")
    if summary["audit_error"]:
        raise RuntimeError(summary["audit_error"])


if __name__ == "__main__":
    fire.Fire(main)
