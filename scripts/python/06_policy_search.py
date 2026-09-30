"""Policy search: evolve a small parametric policy against a fast fitness function (numpy only, no model calls).

The loop is the skeleton of an evolutionary program search (AlphaEvolve-style); its pieces are in
``sbf_starter.search``:

1. a **candidate** is the source code of an ``agent.py`` (here a template with numbers filled in, ``search.render``);
2. **fitness** scores the candidate on training episodes of your own root, with the CPU budget of the task
   (``cpu_budget=True``: a week over it is played by the naive rule, as on the server), on the wheel's
   ``EpisodeSet`` (the references cached; one run of the candidate per episode, milliseconds on Tiny);
3. a **proposer** makes new candidates from the best ones (here: Gaussian mutation of the numbers, ``search.mutate``);
4. the best candidate is compared with the starting point on held-out episodes (the public dev split) with a paired
   interval, and written as a submission folder only if it beats the starting point there.

The policy: every action slot ships ``fraction[s]`` of its capacity (1 = send the maximum), and a route through a
strait is scaled by the strait's observed open fraction raised to ``closure_power`` (0: ignore closures). The search
starts from send-the-maximum.

Where an LLM proposer plugs in: replace ``mutate`` (numbers in a fixed template) with a function that takes the best
candidates' source and scores and returns new source code (a language model editing ``act``, say). Everything else
stays: a candidate is any ``agent.py`` source, so rewritten logic is scored the same way. Keep the held-out check: a
search that sees only 20 dev episodes will fit them, so train on your own root (``entropy``) and use the dev episodes
to confirm. This example calls no model and needs no API key.

    uv run python scripts/python/06_policy_search.py                  # Tiny
    uv run python scripts/python/06_policy_search.py task=small generations=10 population=12

The first run on a network computes the naive rule's and the clairvoyant plan's costs of every training and held-out
episode and caches them (README, "Examples", says how long); later runs play only the candidates.
"""

import math
import tempfile
from dataclasses import replace
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from helper.logging import logger
from sbf_starter import runs, scoring
from sbf_starter.search import mutate, render, slot_count, start_params, write_candidate


@hydra.main(version_base=None, config_path="../../configs", config_name="06_policy_search")
def main(cfg: DictConfig) -> None:
    """Search on your root, compare the best with the start on held-out dev episodes, write it if it is better."""
    out = runs.start(cfg)
    task = cfg.task.name
    if cfg.entropy == 0:
        raise ValueError("train on a root of your own (entropy=...): root 0 holds the dev episodes kept for the check")
    rng = np.random.default_rng(cfg.seed)
    holdout_episodes = OmegaConf.to_container(cfg.holdout) if OmegaConf.is_list(cfg.holdout) else cfg.holdout
    train = scoring.episode_set(task, cfg.train_episodes, quick=cfg.quick, entropy=cfg.entropy, n_jobs=cfg.n_jobs)
    holdout = scoring.episode_set(task, holdout_episodes, quick=cfg.quick, n_jobs=cfg.n_jobs)
    with tempfile.TemporaryDirectory(prefix="sbf-search-") as tmp:
        work = Path(tmp)

        def fitness(params: np.ndarray, name: str) -> float:
            score = train.score(str(write_candidate(render(params), work, name)), cpu_budget=True)
            return -math.inf if score.rss is None else score.rss

        start = start_params(slot_count(task))  # send the maximum, closures ignored
        archive = [(fitness(start, "g0_start"), start)]
        logger.info(
            "{}: {} action slots; training on {} episodes of root {}",
            task,
            len(start) - 1,
            len(train.episodes),
            cfg.entropy,
        )
        logger.info(
            "generation 0: send-the-maximum, training {} {:.4f}",
            scoring.SCALE,
            archive[0][0],
        )
        for g in range(1, cfg.generations + 1):
            parents = [p for _s, p in sorted(archive, key=lambda x: -x[0])[: cfg.elite]]
            children = mutate(parents, rng, cfg.population, sigma=cfg.sigma)  # the proposer: swap in your own here
            scored = [(fitness(c, f"g{g}_{i}"), c) for i, c in enumerate(children)]
            archive += scored
            best_score = max(s for s, _ in archive)
            logger.info(
                "generation {}: best of {} {:.4f}; best so far {:.4f}",
                g,
                cfg.population,
                max(s for s, _ in scored),
                best_score,
            )
        best_score, best = max(archive, key=lambda x: x[0])
        best_dir = write_candidate(render(best), work, "best")
        cmp = holdout.compare(str(best_dir), str(write_candidate(render(start), work, "start")), cpu_budget=True)
        cmp = replace(cmp, a=replace(cmp.a, agent="the best candidate"), b=replace(cmp.b, agent="send-the-maximum"))
        logger.info("held out, on {} dev episodes:\n{}", len(holdout.episodes), cmp)
        beats = cmp.diff is not None and cmp.diff > 0
        if beats:
            (out / "best").mkdir(exist_ok=True)
            (out / "best" / "agent.py").write_text((best_dir / "agent.py").read_text())
            best_out = runs.shown(out / "best")
            logger.info("written {}/agent.py: next, uv run sbf check {} --task={}", best_out, best_out, task)
        else:
            logger.warning(
                "not written: the best candidate does not beat send-the-maximum held out (it fitted its training)"
            )
    if cfg.quick:
        logger.info(scoring.QUICK_NOTE)
    runs.finish(out, train_score=best_score, holdout_score=cmp.a.rss, holdout_start=cmp.b.rss, written=beats)


if __name__ == "__main__":
    main()
