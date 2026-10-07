"""Learn agents/mpc_adaptive's weights W (12 groups x 8 features: a correction that depends on the week's state) by
evolution strategy.

    uv run python examples/train_adaptive.py --task=tiny --generations=2 --pairs=2 --batch=4 --pool=8   # a smoke test
    uv run python examples/train_adaptive.py --n_jobs=6                                        # small, root 1002
    uv run python examples/train_adaptive.py --start=outputs/train_adaptive/<run>/mean.json    # go on from a run

OpenAI's evolution strategy: each generation draws ``pairs`` directions eps around the current mean W and plays
W + sigma eps and W - sigma eps (plus the mean itself, and W = 0 for reference) on ``batch`` episodes drawn from a pool
of ``pool`` episodes of the training root, the same episodes for every candidate of the generation. The candidates'
scores (RSS) become centred ranks, and the mean moves along their rank-weighted directions with Adam. A new batch
every generation keeps the weights from fitting a fixed set of episodes. At the end the mean is compared with W = 0
(plain MPC) on held-out episodes; only if it wins there is it written to agents/mpc_adaptive/params.json. Every
generation is logged to outputs/train_adaptive/<run>/ (log.json, mean.json).
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
from sbf_starter.tracking import Tracker


AGENT = ROOT / "agents" / "mpc_adaptive"


def _shape() -> tuple[int, int]:
    import ast

    tree = ast.parse((AGENT / "agent.py").read_text())
    names = {"GROUPS": 0, "FEATURES": 0}
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in names:
            names[n.targets[0].id] = len(ast.literal_eval(n.value))
    return names["GROUPS"], names["FEATURES"]


def candidate(W: np.ndarray, folder: Path) -> Path:
    """A copy of agents/mpc_adaptive with ``W`` in its params.json."""
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(AGENT, folder, ignore=shutil.ignore_patterns("params.json", "__pycache__"))
    (folder / "params.json").write_text(json.dumps({"W": np.round(W, 6).tolist()}))
    return folder


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main(
    task: str = "small",
    entropy: int = 1002,
    pool: int = 256,
    batch: int = 24,
    generations: int = 30,
    pairs: int = 10,
    sigma: float = 0.1,
    lr: float = 0.03,
    decay: float = 0.005,
    holdout_entropy: int = 9001,
    holdout_episodes: int = 64,
    start: str | None = None,
    n_jobs: int = -1,
    seed: int = 0,
    write: bool = True,
    out: str | None = None,
) -> None:
    """Search W, check the mean on held-out episodes, and write it to agents/mpc_adaptive/params.json if it wins.

    Args:
        task: small (the public board's) or full; tiny for a smoke test.
        entropy: the training root (never 0, nor the held-out one; 1001 is train_residual's).
        pool: episodes of the training root the batches are drawn from (their references are computed once).
        batch: episodes every candidate of a generation plays.
        generations: rounds of the search.
        pairs: directions per generation; each is played both ways (2 * pairs candidates, plus the mean and W = 0).
        sigma: the spread of the candidates around the mean, in units of W.
        lr: Adam's step size on W.
        decay: pulls W towards 0 (plain MPC) every step.
        holdout_entropy: the root of the final check.
        holdout_episodes: episodes of the final check.
        start: a mean.json (or params.json) to start from instead of W = 0.
        n_jobs: parallel (candidate, episode) games (-1: all cores).
        seed: the search's random generator.
        write: write the winner to agents/mpc_adaptive/params.json.
        out: the run folder (default: outputs/train_adaptive/<date_time>).
    """
    params = dict(locals())
    if entropy in (0, holdout_entropy):
        raise ValueError("train on a root of your own, neither the dev root 0 nor the held-out root")
    shape = _shape()
    rng = np.random.default_rng(seed)
    run = Path(out or ROOT / "outputs" / "train_adaptive" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir(parents=True, exist_ok=True)
    mean = np.zeros(shape)
    if start:
        mean = np.asarray(json.loads(Path(start).read_text())["W"], dtype=float)
    scoring.episode_set(task, pool, entropy=entropy, n_jobs=n_jobs)  # the pool's references, once (cached on disk)
    m, v = np.zeros(shape), np.zeros(shape)  # Adam's moments
    tracker = Tracker("mpc_adaptive", run.name, params | {"run_folder": run})
    log = []

    with tempfile.TemporaryDirectory(prefix="sbf-adaptive-") as tmp:
        work = Path(tmp)
        played: dict[tuple, int] = {}  # (W's key, episode) -> J in cents

        def scores(Ws: list[np.ndarray], episodes: list[int], es) -> np.ndarray:
            keys = [tuple(np.round(W, 6).ravel()) for W in Ws]
            first = {key: i for i, key in reversed(list(enumerate(keys)))}  # one folder per distinct W
            folders, jobs = {}, []
            for key, i in first.items():
                todo = [n for n in episodes if (key, n) not in played]
                if todo:
                    folders[i] = str(candidate(Ws[i], work / f"c{i}"))
                    jobs += [(key, i, n) for n in todo]
            rows = Parallel(n_jobs=n_jobs)(
                delayed(_play)(None, folders[i], NO_ZIP_SHA256, es._spec, [n], None) for _key, i, n in jobs
            )
            for (key, _i, n), (row,) in zip(jobs, rows):
                if row["omega_hash"] != es.references[episodes.index(n)]["omega_hash"]:
                    raise RuntimeError("the cached references were computed on other scenarios")
                played[(key, n)] = row["J_policy_cents"]
            out = []
            for key in keys:
                rss = es.rss([played[(key, n)] for n in episodes])["rss"]
                out.append(-np.inf if rss is None else float(rss))
            return np.array(out)

        for g in range(generations):
            start_g = time.perf_counter()
            episodes = sorted(rng.choice(pool, size=min(batch, pool), replace=False).tolist())
            es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
            episodes = [int(n) for n in es.episodes]  # the order of es.references
            eps = rng.standard_normal((pairs, *shape))
            Ws = [mean, np.zeros(shape)] + [mean + s * sigma * e for e in eps for s in (1, -1)]
            f = scores(Ws, episodes, es)
            ranks = centred_ranks(f[2:]).reshape(pairs, 2)
            grad = np.einsum("p,pij->ij", ranks[:, 0] - ranks[:, 1], eps) / (2 * pairs * sigma)
            grad -= decay * mean
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            mean = mean + lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            log.append(
                {
                    "generation": g,
                    "episodes": episodes,
                    "mean_score": float(f[0]),
                    "mpc_score": float(f[1]),
                    "mean_minus_mpc": float(f[0] - f[1]),
                    "best_candidate": float(f[2:].max()),
                    "W_norm": float(np.linalg.norm(mean)),
                }
            )
            tracker.log(
                {"mean_score": f[0], "mpc_score": f[1], "mean_minus_mpc": f[0] - f[1], "best_candidate": f[2:].max()},
                step=g,
            )
            last = [r["mean_minus_mpc"] for r in log[-5:]]
            print(
                f"generation {g}: mean's score {f[0]:.4f}, mpc {f[1]:.4f}, mean - mpc {f[0] - f[1]:+.4f} "
                f"(last {len(last)}: {np.mean(last):+.4f}), best candidate {f[2:].max():.4f}, "
                f"|W| {np.linalg.norm(mean):.3f} ({time.perf_counter() - start_g:.0f} s)",
                flush=True,
            )
            (run / "log.json").write_text(json.dumps(log, indent=1))
            (run / "mean.json").write_text(json.dumps({"generation": g, "W": np.round(mean, 6).tolist()}, indent=1))

        holdout = scoring.episode_set(task, holdout_episodes, entropy=holdout_entropy, n_jobs=n_jobs)
        cmp = holdout.compare(
            str(candidate(mean, work / "mean")),
            str(candidate(np.zeros(shape), work / "zero")),
            names=("mpc_adaptive (mean W)", "mpc (W = 0)"),
            n_jobs=n_jobs,
        )
        print(f"\nheld out, {holdout_episodes} episodes of root {holdout_entropy}:\n{cmp}")
        (run / "holdout.txt").write_text(str(cmp))
        tracker.log({f"holdout_{task}_minus_mpc": cmp.diff}, step=generations)
        for name in ("mean.json", "holdout.txt"):
            tracker.artifact(run / name)
        tracker.end()
        if write and cmp.diff is not None and cmp.diff > 0:
            (AGENT / "params.json").write_text(json.dumps({"W": np.round(mean, 6).tolist()}, indent=1))
            print(f"written agents/mpc_adaptive/params.json; next: uv run sbf check mpc_adaptive --task={task}")
        else:
            print(f"not written (it does not beat plain MPC held out, or --nowrite); the run is in {run}")


if __name__ == "__main__":
    fire.Fire(main)
