"""Learn agents/mpc_residual's 12 corrections (one per group of routes) on top of the MPC plan by evolution strategy.

    uv run python examples/train_residual.py                                   # small, root 1001, 32 episodes
    uv run python examples/train_residual.py --generations=20 --population=16 --train_episodes=48
    uv run python examples/train_residual.py --start=outputs/train_residual/<run>/best.json   # go on from a run

Each generation samples ``population - 1`` candidates around the current mean theta (plus the mean itself), plays each
on the same ``train_episodes`` episodes of the training root, and moves the mean to the weighted average of the
``elite`` best; when no candidate beats the mean, the mean stays and the spread shrinks faster. The fitness is the
score (RSS) on those episodes, so candidates are compared on equal footing. At the end the best candidate is compared
with theta = 0 (plain MPC) on held-out episodes; only if it wins there is its theta written to
agents/mpc_residual/params.json. A generation's candidates and episodes are played in one pool of workers (the
package's own per-episode player, ``shockbench_flow_agent.scoring._play``, without packing a zip per candidate), and a
theta already scored is not played again. Every generation is logged to outputs/train_residual/<run>/.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _play

from sbf_starter import ROOT, scoring


AGENT = ROOT / "agents" / "mpc_residual"


def _groups() -> tuple[str, ...]:
    import ast

    tree = ast.parse((AGENT / "agent.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "GROUPS")
    return tuple(ast.literal_eval(node.value))


def candidate(theta: np.ndarray, folder: Path) -> Path:
    """A copy of agents/mpc_residual with ``theta`` in its params.json."""
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(AGENT, folder, ignore=shutil.ignore_patterns("params.json", "__pycache__"))
    (folder / "params.json").write_text(json.dumps({"theta": [round(float(x), 6) for x in theta]}))
    return folder


def main(
    task: str = "small",
    entropy: int = 1001,
    train_episodes: int = 32,
    generations: int = 15,
    population: int = 12,
    elite: int = 4,
    sigma: float = 0.1,
    sigma_decay: float = 0.93,
    holdout_entropy: int = 9001,
    holdout_episodes: int = 32,
    start: str | None = None,
    n_jobs: int = -1,
    seed: int = 0,
    write: bool = True,
    out: str | None = None,
) -> None:
    """Search theta, check the best on held-out episodes, and write it to agents/mpc_residual/params.json if it wins.

    Args:
        task: small (the public board's) or full; tiny for a smoke test.
        entropy: the training root (never 0, nor the held-out one).
        train_episodes: episodes of the training root every candidate plays.
        generations: rounds of the search.
        population: candidates per generation, the current mean included.
        elite: the best candidates the mean moves to.
        sigma: the initial spread of the candidates around the mean (theta lives in [-1, 1]).
        sigma_decay: sigma shrinks by this factor every generation.
        holdout_entropy: the root of the final check.
        holdout_episodes: episodes of the final check.
        start: a best.json (or params.json) to start from instead of theta = 0.
        n_jobs: parallel (candidate, episode) games (-1: all cores).
        seed: the search's random generator.
        write: write the winner to agents/mpc_residual/params.json.
        out: the run folder (default: outputs/train_residual/<date_time>).
    """
    if entropy in (0, holdout_entropy):
        raise ValueError("train on a root of your own, neither the dev root 0 nor the held-out root")
    groups = _groups()
    rng = np.random.default_rng(seed)
    run = Path(out or ROOT / "outputs" / "train_residual" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir(parents=True, exist_ok=True)
    mean = np.zeros(len(groups))
    if start:
        mean = np.asarray(json.loads(Path(start).read_text())["theta"], dtype=float)
    train = scoring.episode_set(task, train_episodes, entropy=entropy, n_jobs=n_jobs)
    weights = np.log(elite + 0.5) - np.log(np.arange(1, elite + 1))
    weights /= weights.sum()
    log, best = [], (-np.inf, mean.copy())

    with tempfile.TemporaryDirectory(prefix="sbf-residual-") as tmp:
        work = Path(tmp)

        scored: dict[tuple, float] = {}

        def fitness(thetas: list[np.ndarray], tag: str) -> np.ndarray:
            key = [tuple(np.round(th, 6)) for th in thetas]
            todo = [i for i in range(len(thetas)) if key[i] not in scored]
            folders = {i: str(candidate(thetas[i], work / f"{tag}_{i}")) for i in todo}
            jobs = [(i, j, n) for i in todo for j, n in enumerate(train.episodes)]
            rows = Parallel(n_jobs=n_jobs)(
                delayed(_play)(None, folders[i], NO_ZIP_SHA256, train._spec, [n], None) for i, _j, n in jobs
            )
            J = {i: [0] * len(train.episodes) for i in todo}
            for (i, j, _n), (row,) in zip(jobs, rows):
                if row["omega_hash"] != train.references[j]["omega_hash"]:
                    raise RuntimeError("the cached references were computed on other scenarios")
                J[i][j] = row["J_policy_cents"]
            for i in todo:
                rss = train.rss(J[i])["rss"]
                scored[key[i]] = -np.inf if rss is None else float(rss)
            return np.array([scored[k] for k in key])

        for g in range(generations):
            start_g = time.perf_counter()
            thetas = [mean] + [
                np.clip(mean + sigma * rng.standard_normal(len(groups)), -1, 1) for _ in range(population - 1)
            ]
            scores = fitness(thetas, f"g{g}")
            order = np.argsort(-scores)
            if scores[order[0]] > best[0]:
                best = (float(scores[order[0]]), thetas[order[0]].copy())
            improved = order[0] != 0  # a candidate beat the mean (the fitness has no noise: the same episodes)
            if improved:
                mean = np.clip(sum(w * thetas[i] for w, i in zip(weights, order[:elite])), -1, 1)
            log.append(
                {
                    "generation": g,
                    "sigma": sigma,
                    "mean_score": float(scores[0]),
                    "best_score": float(scores[order[0]]),
                    "best_so_far": best[0],
                    "mean_theta": mean.round(4).tolist(),
                }
            )
            print(
                f"generation {g}: mean's score {scores[0]:.4f}, best {scores[order[0]]:.4f}, "
                f"best so far {best[0]:.4f}, sigma {sigma:.3f} ({time.perf_counter() - start_g:.0f} s)",
                flush=True,
            )
            sigma *= sigma_decay if improved else sigma_decay**4  # none better: search closer to the mean
            (run / "log.json").write_text(json.dumps(log, indent=1))
            (run / "best.json").write_text(json.dumps({"score": best[0], "theta": best[1].round(6).tolist()}, indent=1))

        print(f"\nbest on the training episodes: {best[0]:.4f}")
        for name, th in zip(groups, best[1]):
            print(f"  {name:36s} {th:+.3f}")
        holdout = scoring.episode_set(task, holdout_episodes, entropy=holdout_entropy, n_jobs=n_jobs)
        cmp = holdout.compare(
            str(candidate(best[1], work / "best")),
            str(candidate(np.zeros(len(groups)), work / "zero")),
            names=("mpc_residual (best theta)", "mpc (theta = 0)"),
            n_jobs=n_jobs,
        )
        print(f"\nheld out, {holdout_episodes} episodes of root {holdout_entropy}:\n{cmp}")
        (run / "holdout.txt").write_text(str(cmp))
        if write and cmp.diff is not None and cmp.diff > 0:
            (AGENT / "params.json").write_text(json.dumps({"theta": best[1].round(6).tolist()}, indent=1))
            print(f"written agents/mpc_residual/params.json; next: uv run sbf check mpc_residual --task={task}")
        else:
            print(f"not written (it does not beat plain MPC held out, or --nowrite); the run is in {run}")


if __name__ == "__main__":
    fire.Fire(main)
