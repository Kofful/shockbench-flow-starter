"""Local scores for ``sbf evaluate``, ``sbf compare`` and the examples: one function each, one set of defaults.

A score puts an agent's cost between two references on the same scenarios: 0 is the naive rule (keep shipping the
normal plan, ignore disruptions), 1 is the clairvoyant plan (a plan that knew every disruption in advance), below 0 is
worse than naive. The references do not depend on the agent, so the wheel's ``EpisodeSet`` computes them once and
caches them on disk (``SBF_CACHE_DIR``, else ``~/.cache/shockbench-flow``); a score then costs one run of the agent per
episode. The numbers are the leaderboard's computation on public episodes.

- ``episodes="dev"`` is the local dev split, 20 public episodes, 5 per harm level (the leaderboard's episodes are
  private; these follow the same rule); a count k is episodes 0..k-1, and a list names them.
- ``quick=True`` means the same everywhere (``sbf evaluate --quick``, ``quick=true`` in an example's config): a rough
  naive rule (``QUICK``, 2 replications of its demand model instead of 1,000) and no harm levels, so ``"dev"``
  becomes the first ``QUICK_EPISODES`` episodes; seconds instead of minutes, not the leaderboard's numbers.
- ``entropy`` other than 0 draws the episodes from a root of your own: tune there, confirm on the dev split.
- ``cpu_budget=True`` hands a week over the task's CPU budget to the naive rule, as the server does (measured in this
  process, so it is a guide: ``sbf check --docker`` meters as the server does).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from sbf_starter import DEFAULT_TASK, check_task


QUICK = {"fq_replications": 2, "cut_draws": 0}  # the rough naive rule, no harm levels
QUICK_EPISODES = 4  # what "dev" means under quick (the dev split needs the harm levels)
QUICK_NOTE = "quick: a rough naive rule and no harm levels, so these are not the leaderboard's numbers"
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
    """The wheel's ``EpisodeSet`` of these episodes, its references from the cache or computed once."""
    from loguru import logger
    from shockbench_flow_agent import EpisodeSet

    check_task(task)
    if quick and episodes == "dev":
        episodes = QUICK_EPISODES
    if isinstance(episodes, tuple):
        episodes = list(episodes)
    options = QUICK if quick else {}
    logger.disable("shockbench_flow")  # the engine's own cache lines; ``verbose`` keeps the plain progress lines
    try:
        return EpisodeSet.build(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=verbose, **options)
    finally:
        logger.enable("shockbench_flow")


def scorable(agent):
    """What ``EpisodeSet.score`` takes: an agent's name or folder or zip as its path, an ``agent.py`` file as its class.

    A class is played in this process, so a debugger stops in it; a folder or zip is checked by the scorer's validator
    first and its random seed is salted by the zip's SHA-256, as on the server.
    """
    from sbf_starter.agents import load, resolve

    if not isinstance(agent, (str, Path)):
        return agent
    path = resolve(agent)
    return load(path) if path.suffix == ".py" else str(path)


def _label(agent) -> str:
    """An agent as the reports name it: as given (a name or a path), or a class's name."""
    return str(agent) if isinstance(agent, (str, Path)) else getattr(agent, "__qualname__", type(agent).__qualname__)


def evaluate(
    agent,
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    *,
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool | float = False,
    n_jobs: int = -1,
    verbose: bool = True,
):
    """One agent's ``Score`` (``str()`` is the report: the score, its 90 % interval, costs in USD, naive's weeks)."""
    es = episode_set(task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs, verbose=verbose)
    return replace(es.score(scorable(agent), cpu_budget=cpu_budget), agent=_label(agent))


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
    verbose: bool = True,
):
    """Two agents on the same episodes (``Comparison``): a's score minus b's, with a paired bootstrap interval."""
    es = episode_set(task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs, verbose=verbose)
    c = es.compare(scorable(a), scorable(b), cpu_budget=cpu_budget)
    return replace(c, a=replace(c.a, agent=_label(a)), b=replace(c.b, agent=_label(b)))


def as_dict(result) -> dict:
    """A ``Score`` or ``Comparison`` as plain JSON values (for ``--out`` and a run's ``meta.json``)."""
    from shockbench_flow_agent import Comparison

    if isinstance(result, Comparison):
        return {
            "a": as_dict(result.a),
            "b": as_dict(result.b),
            "difference": result.diff,
            "interval": result.interval,
            "p_a_better": result.p_a_better,
        }
    keys = ("agent", "task", "regime", "rss", "pooled", "rss_all", "interval", "episodes", "fallback_weeks")
    out = {k: getattr(result, k) for k in keys}
    out |= {"cost_usd": result.cost_usd, "naive_cost_usd": result.naive_cost_usd}
    out |= {"clairvoyant_cost_usd": result.oracle_cost_usd, "cpu_weeks": result.cpu_weeks}
    out["per_episode"] = [
        {k: r.get(k) for k in ("episode", "stratum", "J_policy_cents", "J_naive_cents", "J_oracle_cents")}
        for r in result.rows
    ]
    return out
