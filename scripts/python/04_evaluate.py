"""Score an agent locally, as ``sbf evaluate`` does (the same function, the same defaults), or compare two.

The score puts your cost between two references on the same scenarios: 0 is the naive rule (keep shipping the normal
plan, ignore disruptions), 1 is the clairvoyant plan (a plan that knew every disruption in advance), below 0 is worse
than naive. ``sbf_starter.scoring.evaluate`` builds the wheel's ``EpisodeSet`` (the references cached on disk after the
first run) and scores the agent on it; ``against`` compares two agents on the same episodes with a paired interval,
which is how to tell whether a change helped.

    uv run python scripts/python/04_evaluate.py                                    # template, Tiny's dev episodes
    uv run python scripts/python/04_evaluate.py agent=heuristic against=template   # a paired comparison
    uv run python scripts/python/04_evaluate.py agent=path/to/agent.py             # in this process (a debugger works)
    uv run python scripts/python/04_evaluate.py task=small quick=true              # seconds, not the board's numbers

The first run on a network computes the naive rule's and the clairvoyant plan's costs of the episodes and caches them
(``~/.cache/shockbench-flow`` or ``SBF_CACHE_DIR``); later runs play only your agent.
"""

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from helper.logging import logger
from sbf_starter import runs, scoring


@hydra.main(version_base=None, config_path="../../configs", config_name="04_evaluate")
def main(cfg: DictConfig) -> None:
    """Score ``cfg.agent`` (or compare it with ``cfg.against``) and log the report; the result goes to the run."""
    out = runs.start(cfg)
    episodes = OmegaConf.to_container(cfg.episodes) if OmegaConf.is_list(cfg.episodes) else cfg.episodes
    options = {"quick": cfg.quick, "entropy": cfg.entropy, "cpu_budget": cfg.cpu_budget, "n_jobs": cfg.n_jobs}
    if cfg.against is None:
        result = scoring.evaluate(cfg.agent, cfg.task.name, episodes, **options)
        summary = {"score": result.rss}
    else:
        result = scoring.compare(cfg.agent, cfg.against, cfg.task.name, episodes, **options)
        summary = {"score": result.a.rss, "against": result.b.rss, "difference": result.diff}
    logger.info("\n{}", result)
    if cfg.quick:
        logger.info(scoring.QUICK_NOTE)
    (out / "result.json").write_text(json.dumps(scoring.as_dict(result), indent=1) + "\n")
    runs.finish(out, **summary)


if __name__ == "__main__":
    main()
