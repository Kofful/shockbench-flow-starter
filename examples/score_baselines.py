"""Score the benchmark package's own baselines (mpc_det, mpc_scen, greedy_lp, ...) on `sbf evaluate`'s episodes.

    uv run python examples/score_baselines.py                                     # mpc_det on Tiny's dev episodes
    uv run python examples/score_baselines.py --policies=mpc_det,mpc_scen,greedy_lp --task=small
    uv run python examples/score_baselines.py --task=small --entropy=9001 --episodes=64 --cpu_budget --against=rl

The baselines live in ``shockbench_flow.policies`` and speak the package's own protocol, not ``Agent``: they are played
here in this process with ``rollout``, as the scorer plays a submission, on the scenarios, policy seeds and cached
reference costs of ``EpisodeSet``, so their scores sit on the same scale as `sbf evaluate`'s. They import highspy, which
the server lacks: this measures how far a planner can go, it does not make a submission. CPU seconds per week are
measured with ``time.process_time`` like ``sbf evaluate --cpu_budget``; with ``--cpu_budget`` a week over the task's
budget is played by the naive rule, as on the server.

Names: zero, naive, nd, sz_state_base_stock, human_ref, greedy_lp, mpc_det, mpc_scen, hindsight_consensus (slow).

Diagnostic variants of mpc_det (not agents: two of them read the scenario, which no agent can):

- ``mpc_det@<H>``: mpc_det with another horizon: H a number of weeks (``mpc_det@30``) or a label of the package's
  sweep, L, L+4, L+8 or max(26,2L) (L = 24 weeks on small and full);
- ``mpc_perfect``: mpc_det planning on the TRUE future of the next H = L weeks instead of "what is broken stays broken";
- ``mpc_perfect_T``: the same with the window to the end of the episode;
- ``mpc_perfect_duration``: mpc_det that knows only when a strait closed now reopens (its true state until then, open
  after), nothing of closures to come: the ceiling of a classifier of closure durations.

mpc_det against mpc_perfect is the value of knowing the future (what forecasting can win at most); mpc_perfect against
mpc_perfect_T is the cost of the horizon; mpc_perfect_T against 1 is mostly out of any policy's reach: the clairvoyant
plan is an LP that replaces some of the simulator's fixed rules (the default release at straits, the grids' energy
priority, the allocation of short supply) by free choices (``oracle/lp.py``), a lower bound on any trajectory's cost.

    uv run python examples/score_baselines.py --policies=mpc_det,mpc_perfect,mpc_perfect_T --task=small
"""

import json
import re
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed

from sbf_starter import scoring
from sbf_starter.agents import resolve


def _names(policies) -> list[str]:
    """The comma-separated names, a comma inside parentheses kept (``mpc_det@max(26,2L)``)."""
    text = policies if isinstance(policies, str) else ",".join(str(p) for p in policies)
    return [p.strip() for p in re.split(r",(?![^(]*\))", text) if p.strip()]


class _Metered:
    """The baseline with its CPU per week measured; a week over ``budget_s`` returns no action (naive plays it)."""

    def __init__(self, policy, budget_s: float | None) -> None:
        self.policy, self.budget_s, self.carry = policy, budget_s, 0.0
        self.cpu, self.over = [], 0

    def __getattr__(self, attr):
        return getattr(self.policy, attr)

    def reset(self, static: dict, obs: dict, policy_seed: int) -> None:
        start = time.process_time()
        self.policy.reset(static, obs, policy_seed)
        self.carry = time.process_time() - start  # counts toward week 1, as Agent(config) does

    def act(self, obs: dict):
        start = time.process_time()
        action = self.policy.act(obs)
        used = self.carry + time.process_time() - start
        self.cpu.append(used)
        self.carry = 0.0
        if self.budget_s is not None and used > self.budget_s:
            self.over += 1
            return None
        return action


def _make(name: str, context, marks):
    """Baseline ``name`` of the registry, or a diagnostic variant of mpc_det (module docstring)."""
    import dataclasses

    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet, MpcDetParams
    from shockbench_flow.policies.registry import make_policy

    if name.startswith("mpc_det@"):
        H = name.removeprefix("mpc_det@")
        if not H.isdigit():
            return MpcDet(MpcDetParams(H=H), context)

        class FixedHorizon(MpcDet):
            """mpc_det with a window of H weeks (the package's own variants take only its sweep's labels)."""

            def _horizon(self, inst, plan) -> int:
                return int(H)

        policy = FixedHorizon(context=context)
        policy.name = name
        return policy
    if name == "mpc_perfect_duration":

        class PerfectDuration(MpcDet):
            """mpc_det's persistence forecast, but a strait closed now follows the truth until it reopens, then open."""

            def _window_arrays(self, inst, obs, H_t):
                arrays = {k: np.array(v) for k, v in super()._window_arrays(inst, obs, H_t).items()}
                t = int(obs["week"])
                kappa0 = L.ObservedGraph.nominal(inst).values["kappa"]
                for c in np.flatnonzero(self._memory.values["open"] < 1.0):
                    truth = np.asarray(marks.o)[t - 1 : t - 1 + H_t, c]
                    r = next((k for k, x in enumerate(truth) if x >= 0.999), H_t)  # the window week it reopens
                    for f, nominal in (
                        ("o", 1.0),
                        ("o_now", 1.0),
                        ("kappa", kappa0[c]),
                        ("kappa_now", kappa0[c]),
                        ("wr_class", 0),
                    ):
                        arrays[f][:r, c] = np.asarray(getattr(marks, f))[t - 1 : t - 1 + r, c]
                        arrays[f][r:, c] = nominal
                arrays["h_queue"], arrays["c_wr"] = L.window_queue_and_transit(inst, arrays["wr_class"])
                return arrays

        policy = PerfectDuration(context=context)
        policy.name = name
        return policy
    if name not in ("mpc_perfect", "mpc_perfect_T"):
        return make_policy(name, context)

    class PerfectForecast(MpcDet):
        """mpc_det whose window holds the scenario's true marks of weeks t .. t + H - 1 (and its future fab hits)."""

        def _horizon(self, inst, plan) -> int:
            return inst.T if name == "mpc_perfect_T" else super()._horizon(inst, plan)

        def _window_arrays(self, inst, obs, H_t):
            t = int(obs["week"])
            return {f: np.asarray(getattr(marks, f))[t - 1 : t - 1 + H_t] for f in L.WINDOW_FIELDS}

        def act(self, obs: dict) -> dict:
            inst, t = self._inst, int(obs["week"])
            self._memory.update(inst, obs)
            H_t = L.window_length(self._H, t, inst.T)
            hits = tuple(  # hits of the window's weeks, renumbered t -> 1 (earlier ones are in the observed WIP)
                dataclasses.replace(h, onset=h.onset - (t - 1)) for h in marks.fab_hits if t <= h.onset_week < t + H_t
            )
            arrays = self._window_arrays(inst, obs, H_t)
            model = L.rolled_lp(inst, obs, arrays, H_t, fab_hits=hits, planning_rules=self.params.planning_rules)
            res = self._session.solve(L.to_highs_lp(model), L.WindowShape.of(model))
            if res.ok:
                return L.week1_action(inst, model, res.x, obs, L.prohibited_now(self._memory, t))
            return self._fallback.act(obs)

    policy = PerfectForecast(context=context)
    policy.name = name
    return policy


def _play(name: str, spec: tuple, n: int, budget_s: float | None) -> dict:
    """One episode of baseline ``name``, as ``EpisodeSet`` plays an agent's (``shockbench_flow_agent.scoring``)."""
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.evaluation.cache import fq_quantiles
    from shockbench_flow.hosting.tasks import get_task, task_generator
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    task, entropy, regime, fq_replications, cache = spec
    inst, omega, marks, fallback = _world(task, entropy, n, fq_replications, cache)
    _inst, params = task_generator(task)
    context = PolicyContext(
        fq_quantile=fq_quantiles(inst, params, fq_replications, cache_dir=cache).quantiles,
        generator=GeneratorRef(task, get_task(task).gamma),
        fq_replications=fq_replications,
    )
    policy = _Metered(_make(name, context, marks), budget_s)
    start = time.perf_counter()
    traj = rollout(inst, policy, omega, regime, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    fell = sum(took_fallback(r) for r in traj.records)
    return {
        "episode": n,
        "J_policy_cents": traj.J_cents,
        "fallback_weeks": fell,
        "cpu_weeks": policy.over,
        "invalid_entries": sum(len(r.invalid) for r in traj.records if not took_fallback(r)),
        "first_error": None,
        "weeks": inst.T,
        "seconds": round(time.perf_counter() - start, 3),
        "omega_hash": omega.hash,
        "cpu_median_s": float(np.median(policy.cpu)),
        "cpu_max_s": float(np.max(policy.cpu)),
    }


def main(
    policies: str | list[str] = "mpc_det",
    task: str = "tiny",
    episodes: str | int | list[int] = "dev",
    entropy: int = 0,
    quick: bool = False,
    cpu_budget: bool = False,
    against: str | None = None,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """Score each baseline of ``policies`` (comma-separated), and optionally compare each with agent ``against``.

    Args:
        policies: baseline names, comma-separated (see the module docstring).
        task: tiny, small or full.
        episodes: dev (20 episodes), a count k (episodes 0..k-1) or a list.
        entropy: 0 for the public dev episodes; any other integer for scenarios of your own.
        quick: seconds, not the leaderboard's numbers.
        cpu_budget: a week over the task's CPU budget is played by the naive rule, as on the server.
        against: an agent (a name of agents/, or a folder) to compare every baseline with, paired.
        n_jobs: parallel episodes (-1: all cores).
        out: also write the scores as JSON there.
    """
    from shockbench_flow_agent.scoring import LEVEL, N_BOOT, Comparison, _boot_stats, _interval

    es = scoring.episode_set(task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
    budget = es._budget(cpu_budget)
    ref = es.score(str(resolve(against)), name=against, cpu_budget=cpu_budget, n_jobs=n_jobs) if against else None
    summary, first = [], None
    for name in _names(policies):
        print(f"\n=== {name} on {task}, {len(es.episodes)} episodes ...", flush=True)
        start = time.perf_counter()
        rows = Parallel(n_jobs=n_jobs)(delayed(_play)(name, es._spec, n, budget) for n in es.episodes)
        for r, row in zip(es.references, rows):
            if r["omega_hash"] != row["omega_hash"]:
                raise RuntimeError(f"episode {r['episode']}: the cached reference was computed on another scenario")
        J = [r["J_policy_cents"] for r in rows]
        (boot,) = _boot_stats(es.references, [J], es.rss(J)["pooled"], N_BOOT, 0)
        score = es._score(name, rows, budget, LEVEL, boot)
        cpu_med = float(np.median([r["cpu_median_s"] for r in rows]))
        cpu_max = float(np.max([r["cpu_max_s"] for r in rows]))
        print(score)
        print(f"CPU per week: median {cpu_med:.3f} s, max {cpu_max:.3f} s (budget {es._budget(True)} s on {task})")
        print(f"played in {time.perf_counter() - start:.0f} s")
        entry = {
            "policy": name,
            "task": task,
            "entropy": entropy,
            "episodes": len(es.episodes),
            "quick": quick,
            "cpu_budget": cpu_budget,
            "rss": score.rss,
            "interval": score.interval,
            "rss_by_stratum": score.rss_by_stratum,
            "fallback_weeks": score.fallback_weeks,
            "cpu_median_s": cpu_med,
            "cpu_max_s": cpu_max,
            "cost_usd": score.cost_usd,
        }

        def paired(other):
            pooled = score.pooled and other.pooled
            ba, bb = _boot_stats(es.references, [score.J_cents, other.J_cents], pooled, N_BOOT, 0)
            d = ba - bb
            ok = d[np.isfinite(d)]
            diff = None if score.rss is None or other.rss is None else score.rss - other.rss
            share = float((ok > 0).mean()) if ok.size else None
            return Comparison(score, other, diff, _interval(d, LEVEL), LEVEL, share)

        if ref is not None:
            cmp = paired(ref)
            print(f"\npaired with {against}:\n{cmp}")
            entry |= {"against": against, "against_rss": ref.rss, "diff": cmp.diff, "diff_interval": cmp.interval}
        if first is None:
            first = score
        else:
            cmp = paired(first)
            print(f"\npaired with {first.agent}:\n{cmp}")
            entry |= {"vs_first": cmp.diff, "vs_first_interval": cmp.interval}
        summary.append(entry)

    print("\nsummary (score: 0 = naive rule, 1 = clairvoyant plan)")
    if ref is not None:
        print(f"  {against:22s} {ref.rss:.4f}  (the agent compared with)")
    for e in summary:
        lo, hi = e["interval"] or (float("nan"), float("nan"))
        extra = f", vs {against} {e['diff']:+.4f}" if ref is not None else ""
        if "vs_first" in e:
            a, b = e["vs_first_interval"] or (float("nan"), float("nan"))
            extra += f", vs {summary[0]['policy']} {e['vs_first']:+.4f} [{a:+.4f}, {b:+.4f}]"
        print(
            f"  {e['policy']:22s} {e['rss']:.4f}  [{lo:.4f}, {hi:.4f}]  CPU/week max {e['cpu_max_s']:.2f} s, "
            f"naive played {e['fallback_weeks']} weeks{extra}"
        )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=2))
        print(f"written {out}")


if __name__ == "__main__":
    fire.Fire(main)
