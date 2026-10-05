"""Local scores on the wheel's ``EpisodeSet``, whose reference costs are computed once and cached on disk.

``quick`` gives a rough score in seconds (a rough naive rule and no harm levels; ``"dev"`` becomes 4 episodes): not the
leaderboard's numbers, and the report says so.
"""

from __future__ import annotations

from pathlib import Path

from sbf_starter import DEFAULT_TASK, check_task


SCALE = "score (0 = naive rule, 1 = clairvoyant plan)"


def episode_set(
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    *,
    quick: bool = False,
    entropy: int = 0,
    n_jobs: int = -1,
    verbose: bool = True,
):
    from shockbench_flow_agent import EpisodeSet

    check_task(task)
    if isinstance(episodes, tuple):
        episodes = list(episodes)
    return EpisodeSet.build(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=verbose, quick=quick)


def _scorable(agent):
    """A name becomes its folder. An agent.py plays in this process (a debugger works); a folder or zip is validated
    and seeded by its zip's SHA-256, as on the server.
    """
    from sbf_starter.agents import resolve

    return str(resolve(agent)) if isinstance(agent, (str, Path)) else agent


def _name(agent) -> str | None:
    return str(agent) if isinstance(agent, (str, Path)) else None


def evaluate(
    agent,
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    *,
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool | float = False,
    n_jobs: int = -1,
    batch_size: int = 32,
    device: str = "auto",
    verbose: bool = True,
):
    """One agent's ``Score``; process-parallel by default, with optional batched CUDA inference."""
    from sbf_starter.accelerated import check_batch_size, check_device, evaluation_device, score_one

    check_batch_size(batch_size)
    check_device(device)
    with evaluation_device(device):
        # Enter before reference generation so any reusable joblib workers inherit the requested policy device too.
        es = episode_set(task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs, verbose=verbose)
        return score_one(
            es,
            _scorable(agent),
            name=_name(agent),
            cpu_budget=cpu_budget,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )


def compare(
    a,
    b,
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    *,
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool | float = False,
    n_jobs: int = -1,
    batch_size: int = 32,
    device: str = "auto",
    verbose: bool = True,
):
    """A's score minus b's on the same episodes, with a paired interval (a ``Comparison``)."""
    from sbf_starter.accelerated import check_batch_size, check_device, comparison, evaluation_device

    check_batch_size(batch_size)
    check_device(device)
    with evaluation_device(device):
        es = episode_set(task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs, verbose=verbose)
        names = (_name(a), _name(b)) if _name(a) and _name(b) else None
        return comparison(
            es,
            _scorable(a),
            _scorable(b),
            names=names or (None, None),
            cpu_budget=cpu_budget,
            n_jobs=n_jobs,
            batch_size=batch_size,
        )


def as_dict(result) -> dict:
    """A ``Score`` or ``Comparison`` as JSON values."""
    from shockbench_flow_agent import Comparison

    if isinstance(result, Comparison):
        return {
            "a": as_dict(result.a),
            "b": as_dict(result.b),
            "difference": result.diff,
            "interval": result.interval,
            "p_a_better": result.p_a_better,
        }
    keys = ("agent", "task", "regime", "rss", "pooled", "rss_all", "interval", "episodes", "fallback_weeks", "quick")
    out = {k: getattr(result, k) for k in keys}
    out |= {"cost_usd": result.cost_usd, "naive_cost_usd": result.naive_cost_usd}
    out |= {"clairvoyant_cost_usd": result.clairvoyant_cost_usd, "cpu_weeks": result.cpu_weeks}
    out["per_episode"] = [
        {k: r.get(k) for k in ("episode", "stratum", "J_policy_cents", "J_naive_cents", "J_clairvoyant_cents")}
        for r in result.rows
    ]
    return out
