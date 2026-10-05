"""An evolutionary search over the heuristic agent's numbers (AlphaEvolve-style, without a model).

    uv run python examples/06_policy_search.py
    uv run python examples/06_policy_search.py --task=small --generations=10 --population=12

A candidate is the heuristic's agent.py plus a params.json; its fitness is its score on your own root, under the CPU
budget. To plug in an LLM, replace ``mutate`` with a function that writes new agent.py files. The best is kept only if
it beats the start (send the maximum) on the held-out dev episodes. Its numbers are per action slot, so they fit only
the network searched on: use ``--task=small`` for a submission.
"""

import json
import math
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments

from sbf_starter import env_id, scoring
from sbf_starter.agents import resolve


HEURISTIC = resolve("heuristic") / "agent.py"
MAX_CLOSURE_POWER = 3.0  # closure_power lives in [0, 3], the fractions in [0, 1]


def start_params(n_slots: int) -> np.ndarray:
    """Send the maximum as [fraction per slot..., closure_power]."""
    return np.r_[np.ones(n_slots), 0.0]


def write_candidate(params: np.ndarray, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(HEURISTIC, folder / "agent.py")
    numbers = {"fraction": [round(float(x), 4) for x in params[:-1]], "closure_power": round(float(params[-1]), 4)}
    (folder / "params.json").write_text(json.dumps(numbers) + "\n")
    return folder


def mutate(parents: list[np.ndarray], rng: np.random.Generator, n: int, sigma: float = 0.15) -> list[np.ndarray]:
    """Gaussian steps around random parents, scaled and clipped to each parameter's range."""
    out = []
    for _ in range(n):
        p = parents[rng.integers(len(parents))]
        child = p + rng.normal(0.0, sigma, p.shape) * np.r_[np.ones(len(p) - 1), MAX_CLOSURE_POWER]
        child[:-1] = np.clip(child[:-1], 0.0, 1.0)
        child[-1] = np.clip(child[-1], 0.0, MAX_CLOSURE_POWER)
        out.append(child)
    return out


def main(
    task: str = "tiny",
    entropy: int = 20261002,
    train_episodes: int = 16,
    holdout: str | int | list[int] = "dev",
    generations: int = 6,
    population: int = 8,
    elite: int = 3,
    sigma: float = 0.15,
    quick: bool = False,
    n_jobs: int = -1,
    seed: int = 0,
    out: str | None = None,
) -> None:
    """Search, check the best on held-out episodes, and write it to <out>/best if it wins there.

    Args:
        task: tiny, small or full.
        entropy: your training root: any integer but 0 (the dev episodes).
        train_episodes: episodes of that root the fitness plays.
        holdout: held-out dev episodes: dev, a count or a list.
        generations: rounds of the search.
        population: candidates per generation.
        elite: candidates kept as parents.
        sigma: the mutation's scale.
        quick: seconds, not the leaderboard's numbers.
        n_jobs: workers for reference computation and candidate episodes (-1: all cores).
        seed: the search's generator.
        out: the run folder (default: outputs/06_policy_search/<date_time>).

    """
    if entropy == 0:
        raise ValueError("train on a root of your own (--entropy=...): root 0 holds the dev episodes of the check")
    out = Path(out or f"outputs/06_policy_search/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    rng = np.random.default_rng(seed)
    train = scoring.episode_set(task, train_episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
    held_out = scoring.episode_set(task, holdout, quick=quick, n_jobs=n_jobs)
    with tempfile.TemporaryDirectory(prefix="sbf-search-") as tmp:
        work = Path(tmp)

        def fitness(params: np.ndarray, name: str) -> float:
            score = train.score(str(write_candidate(params, work / name)), cpu_budget=True, n_jobs=n_jobs)
            return -math.inf if score.rss is None else score.rss

        start = start_params(gym.make(env_id(task)).action_space["flows"].shape[0])
        archive = [(fitness(start, "g0_start"), start)]
        print(f"{task}: {len(start) - 1} action slots; training on {len(train.episodes)} episodes of root {entropy}")
        print(f"generation 0: send-the-maximum, training {scoring.SCALE} {archive[0][0]:.4f}")
        for g in range(1, generations + 1):
            parents = [p for _s, p in sorted(archive, key=lambda x: -x[0])[:elite]]
            children = mutate(parents, rng, population, sigma=sigma)  # the proposer: swap in your own here
            scored = [(fitness(c, f"g{g}_{i}"), c) for i, c in enumerate(children)]
            archive += scored
            best_score = max(s for s, _ in archive)
            print(f"generation {g}: best of {population} {max(s for s, _ in scored):.4f}; best so far {best_score:.4f}")
        best_score, best = max(archive, key=lambda x: x[0])
        print(f"the best candidate: training {scoring.SCALE} {best_score:.4f}, closure_power {best[-1]:.3f}")
        best_dir = write_candidate(best, work / "best")
        cmp = held_out.compare(
            str(best_dir), str(write_candidate(start, work / "start")), cpu_budget=True, n_jobs=n_jobs
        )
        cmp = replace(cmp, a=replace(cmp.a, agent="the best candidate"), b=replace(cmp.b, agent="send-the-maximum"))
        print(f"held out, on {len(held_out.episodes)} dev episodes:\n{cmp}")
        if cmp.diff is not None and cmp.diff > 0:
            shutil.copytree(best_dir, out / "best", dirs_exist_ok=True)
            print(f"written {out / 'best'}: next, uv run sbf check {out / 'best'} --task={task}")
        else:
            print("not written: the best candidate does not beat send-the-maximum held out (it fitted its training)")


if __name__ == "__main__":
    fire.Fire(main)
