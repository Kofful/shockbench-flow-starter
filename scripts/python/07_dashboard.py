"""Dashboards of one dev episode: a random agent and a send-the-maximum agent on the same scenario, with naive.

    uv run python scripts/python/07_dashboard.py                             # Tiny: a dev episode with a closure
    uv run python scripts/python/07_dashboard.py episode=29 quick=true       # a named episode, a rough naive rule
    uv run python scripts/python/07_dashboard.py task=small episode=0        # the public board's network

It finds the first dev episode in which a strait is fully open in week 1 and later falls below ``min_open`` open (a
disruption the dashboards show), plays both agents on it (``reset(options={"episode": n})``) with the naive rule, the
score's zero, on the same scenario, and writes to the run folder: the network map ``network.png``,
``dashboard_random.png``, ``dashboard_max.png``, ``episode_max.gif`` (one frame per week) and both episode records
(``record_*.npz``, which ``dashboard.load_record`` and ``wrappers.load_record`` read back).

The naive rule needs its demand model, computed once and kept in the local evaluation's cache (``~/.cache/
shockbench-flow`` or ``SBF_CACHE_DIR``; ``sbf evaluate`` fills the same entry): README, "Examples", says how long the
first run takes. ``naive=false`` leaves it out; ``quick=true`` uses a rough one (not the board's). The dashboard was
designed on Tiny; it runs on Small and Full too, where the network map is dense.
"""

import time

import gymnasium as gym
import hydra
import loguru
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* ids
from omegaconf import DictConfig
from shockbench_flow.policies.naive_fq import REPLICATIONS  # the naive rule's demand model, as the board computes it
from shockbench_flow_gym import dashboard

from helper.logging import logger
from sbf_starter import runs, scoring
from sbf_starter.agents import load
from sbf_starter.play import episodes_with_closure


@hydra.main(version_base=None, config_path="../../configs", config_name="07_dashboard")
def main(cfg: DictConfig) -> None:
    """Write the map, both dashboards, the GIF and the records (module docstring); ``episode`` skips the search."""
    out = runs.start(cfg)
    env = gym.make(cfg.task.env_id, regime=cfg.regime)
    if cfg.episode is None:
        (n,) = episodes_with_closure(env, 1, search=cfg.search, min_open=cfg.min_open, seed=cfg.seed)
        logger.info("dev episode {}: a strait closes below {:.0%} open", n, cfg.min_open)
    else:
        n = int(cfg.episode)
        logger.info("dev episode {}", n)
    replications = None if not cfg.naive else scoring.QUICK["fq_replications"] if cfg.quick else REPLICATIONS
    dashboard.plot_network(cfg.task.name).savefig(out / "network.png", dpi=100)
    written = ["network.png"]
    for name, agent in (("random", load("random")), ("max", load("template"))):
        start = time.perf_counter()
        loguru.logger.disable("shockbench_flow")  # the engine's cache lines
        try:
            rec = dashboard.record_episode(
                env, agent, seed=cfg.seed, options={"episode": n}, naive_replications=replications, n_jobs=cfg.n_jobs
            )
        finally:
            loguru.logger.enable("shockbench_flow")
        meta = rec["meta"]  # costs in integer cents: total cost less the terminal credit (lower is better)
        naive = f", the naive rule's ${meta['naive_J_cents'] / 100:,.0f}" if "naive" in rec else ""
        logger.info("{}: cost ${:,.0f}{} ({:.1f} s)", name, meta["J_cents"] / 100, naive, time.perf_counter() - start)
        dashboard.save_record(rec, out / f"record_{name}.npz")
        dashboard.episode_dashboard(rec, out / f"dashboard_{name}.png")
        written += [f"record_{name}.npz", f"dashboard_{name}.png"]
    dashboard.episode_animation(rec, out / "episode_max.gif")
    logger.info("written in {}: {}", runs.shown(out), ", ".join([*written, "episode_max.gif"]))
    runs.finish(out, episode=n)


if __name__ == "__main__":
    main()
