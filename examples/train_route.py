"""Learn agents/mpc_route's network (a correction for every route) by evolution strategy on small and full together.

    uv run python examples/train_route.py --generations=1 --pairs=1 --batch_small=2 --batch_full=1 --pool_small=4 \
        --pool_full=2 --holdout_small=2 --holdout_full=1 --nowrite                     # a smoke test
    uv run python examples/train_route.py --minutes=270                                 # about 25 generations
    uv run python examples/train_route.py --start=outputs/train_route/<run>/mean.json   # go on from a run

OpenAI's evolution strategy, as examples/train_adaptive.py, on both maps: each generation draws ``pairs`` directions
eps around the current mean parameters and plays mean + sigma eps and mean - sigma eps (plus the mean, and plain MPC
for reference) on ``batch_small`` small and ``batch_full`` full episodes drawn from pools of the training root, the
same episodes for every candidate of the generation. A candidate's fitness is the average of its two scores (RSS);
the fitnesses become centred ranks and the mean moves along their rank-weighted directions with Adam. The number of
routes plays no part (unlike for PPO): only the network's parameters are searched. At the end the mean is compared
with plain MPC on held-out episodes of both maps; only if it wins on both is it written to agents/mpc_route/
params.json. Every generation goes to outputs/train_route/<run>/ (log.json for examples/watch_training.py, mean.json)
and to MLflow.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _play, _world

from sbf_starter import ROOT, scoring
from sbf_starter.tracking import Tracker


AGENT = ROOT / "agents" / "mpc_route"
MAPS = ("small", "full")
MPC_CACHE = ROOT / "outputs" / "train_ppo_residual"  # plain MPC's costs on the training pools, shared with PPO


def n_params(hidden: int) -> int:
    sys.path.insert(0, str(AGENT))
    import agent as mpc_route

    sys.path.remove(str(AGENT))
    return mpc_route.n_params(hidden)


def candidate(params: np.ndarray | None, hidden: int, folder: Path) -> Path:
    """A copy of agents/mpc_route with ``params`` in its params.json (None: without one, plain MPC)."""
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(AGENT, folder, ignore=shutil.ignore_patterns("params.json", "__pycache__"))
    if params is not None:
        (folder / "params.json").write_text(json.dumps({"hidden": hidden, "params": np.round(params, 6).tolist()}))
    return folder


def play(root: str, spec: tuple, n: int) -> dict:
    """One game, the worker's episode memo emptied after it (many episodes over a run would fill the memory)."""
    (row,) = _play(None, root, NO_ZIP_SHA256, spec, [n], None)
    _world.cache_clear()
    return row


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main(
    minutes: float | None = None,
    generations: int = 1000,
    entropy: int = 1003,
    pool_small: int = 256,
    pool_full: int = 32,
    batch_small: int = 16,
    batch_full: int = 4,
    pairs: int = 6,
    hidden: int = 16,
    sigma: float = 0.1,
    lr: float = 0.03,
    decay: float = 0.002,
    holdout_entropy: int = 9001,
    holdout_small: int = 64,
    holdout_full: int = 8,
    start: str | None = None,
    n_jobs: int = 6,
    seed: int = 0,
    write: bool = True,
    out: str | None = None,
) -> None:
    """Search the network, check the mean on held-out small and full episodes, write it if it wins on both.

    Args:
        minutes: stop starting new generations after this long (None: run all ``generations``).
        generations: rounds of the search at most.
        entropy: the training root of both maps (1003 shares its references and MPC's costs with PPO's runs).
        pool_small: small episodes of the training root the batches are drawn from.
        pool_full: full episodes of the training root the batches are drawn from.
        batch_small: small episodes every candidate of a generation plays.
        batch_full: full episodes every candidate of a generation plays (a full game takes about a minute).
        pairs: directions per generation, each played both ways (2 * pairs candidates, plus the mean and MPC).
        hidden: the network's hidden units.
        sigma: the spread of the candidates around the mean, in units of the parameters.
        lr: Adam's step size.
        decay: pulls the parameters towards 0 (plain MPC) every step.
        holdout_entropy: the root of the final check.
        holdout_small: small episodes of the final check.
        holdout_full: full episodes of the final check.
        start: a mean.json (or params.json) to start from instead of 0 (plain MPC).
        n_jobs: parallel games.
        seed: the search's random generator.
        write: write the winner to agents/mpc_route/params.json.
        out: the run folder (default: outputs/train_route/<date_time>).
    """
    params = dict(locals())
    if entropy in (0, holdout_entropy):
        raise ValueError("train on a root of your own, neither the dev root 0 nor the held-out root")
    rng = np.random.default_rng(seed)
    run = Path(out or ROOT / "outputs" / "train_route" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir(parents=True, exist_ok=True)
    size = n_params(hidden)
    mean = np.zeros(size)
    if start:
        doc = json.loads(Path(start).read_text())
        mean, hidden = np.asarray(doc["params"], dtype=float), int(doc["hidden"])
    pools = {"small": pool_small, "full": pool_full}
    batches = {"small": batch_small, "full": batch_full}
    for task in MAPS:  # the pools' references, once (cached on disk)
        scoring.episode_set(task, pools[task], entropy=entropy, n_jobs=n_jobs)
    played: dict[tuple, int] = {}  # (params' key, map, episode) -> J in cents
    zero = tuple(np.zeros(size))
    for task in MAPS:  # plain MPC's costs already known from PPO's runs
        path = MPC_CACHE / f"mpc_{task}_{entropy}.json"
        if path.is_file():
            played |= {(zero, task, int(n)): j for n, j in json.loads(path.read_text()).items()}
    tracker = Tracker("mpc_route", run.name, params | {"run_folder": run, "n_params": size})
    m, v = np.zeros(size), np.zeros(size)  # Adam's moments
    log, began = [], time.monotonic()
    print(f"{size} parameters; the run is in {run}", flush=True)

    with tempfile.TemporaryDirectory(prefix="sbf-route-") as tmp:
        work = Path(tmp)

        def scores(Ps: list[np.ndarray], episodes: dict, sets: dict) -> dict:
            """Each candidate's score per map on this generation's episodes."""
            keys = [tuple(np.round(P, 6)) for P in Ps]
            first = {key: i for i, key in reversed(list(enumerate(keys)))}
            folders, jobs = {}, []
            for key, i in first.items():
                todo = [(t, n) for t in MAPS for n in episodes[t] if (key, t, n) not in played]
                if todo:
                    folders[i] = str(candidate(None if key == zero else Ps[i], hidden, work / f"c{i}"))
                    jobs += [(key, i, t, n) for t, n in todo]
            jobs.sort(key=lambda job: job[2] != "full")  # the long full games first, so the workers end together
            rows = Parallel(n_jobs=n_jobs)(delayed(play)(folders[i], sets[t]._spec, n) for _k, i, t, n in jobs)
            for (key, _i, t, n), row in zip(jobs, rows):
                if row["omega_hash"] != sets[t].references[episodes[t].index(n)]["omega_hash"]:
                    raise RuntimeError("the cached references were computed on other scenarios")
                played[(key, t, n)] = row["J_policy_cents"]
            out = {}
            for t in MAPS:
                rss = [sets[t].rss([played[(key, t, n)] for n in episodes[t]])["rss"] for key in keys]
                out[t] = np.array([-np.inf if r is None else float(r) for r in rss])
            return out

        for g in range(generations):
            if minutes is not None and time.monotonic() - began > 60 * minutes:
                break
            start_g = time.perf_counter()
            episodes, sets = {}, {}
            for t in MAPS:
                drawn = sorted(rng.choice(pools[t], size=min(batches[t], pools[t]), replace=False).tolist())
                sets[t] = scoring.episode_set(t, drawn, entropy=entropy, n_jobs=n_jobs, verbose=False)
                episodes[t] = [int(n) for n in sets[t].episodes]
            eps = rng.standard_normal((pairs, size))
            Ps = [mean, np.zeros(size)] + [mean + s * sigma * e for e in eps for s in (1, -1)]
            f = scores(Ps, episodes, sets)
            fitness = (f["small"] + f["full"]) / 2
            ranks = centred_ranks(fitness[2:]).reshape(pairs, 2)
            grad = (ranks[:, 0] - ranks[:, 1]) @ eps / (2 * pairs * sigma) - decay * mean
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            mean = mean + lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            row = {"minutes": round((time.monotonic() - began) / 60, 1), "generation": g}
            for t in MAPS:
                row[t] = {"episodes": len(episodes[t]), "rss": float(f[t][0]), "minus_mpc": float(f[t][0] - f[t][1])}
            log.append(row)
            tracker.log(
                {f"{t}_{k}": row[t][k] for t in MAPS for k in ("rss", "minus_mpc")}
                | {"best_candidate": float(fitness[2:].max()), "params_norm": float(np.linalg.norm(mean))},
                step=g,
            )
            print(
                f"generation {g} ({row['minutes']:.0f} min): "
                + " | ".join(f"{t}: {f[t][0]:.4f}, mean - mpc {f[t][0] - f[t][1]:+.4f}" for t in MAPS)
                + f" | |params| {np.linalg.norm(mean):.3f} ({time.perf_counter() - start_g:.0f} s)",
                flush=True,
            )
            (run / "log.json").write_text(json.dumps(log, indent=1))
            (run / "mean.json").write_text(
                json.dumps({"generation": g, "hidden": hidden, "params": np.round(mean, 6).tolist()})
            )

        wins = []
        best = str(candidate(mean, hidden, work / "mean"))
        for t, episodes_n in (("small", holdout_small), ("full", holdout_full)):
            if episodes_n <= 0:
                continue
            es = scoring.episode_set(t, episodes_n, entropy=holdout_entropy, n_jobs=n_jobs)
            cmp = es.compare(best, str(ROOT / "agents" / "mpc"), names=("mpc_route", "mpc"), n_jobs=n_jobs)
            print(f"\nheld out, {t}, {episodes_n} episodes of root {holdout_entropy}:\n{cmp}", flush=True)
            (run / f"holdout_{t}.txt").write_text(str(cmp))
            tracker.log({f"holdout_{t}_minus_mpc": cmp.diff}, step=len(log))
            tracker.artifact(run / f"holdout_{t}.txt")
            wins.append(cmp.diff is not None and cmp.diff > 0)
    tracker.artifact(run / "mean.json")
    tracker.end()
    if write and wins and all(wins):
        (AGENT / "params.json").write_text(
            json.dumps({"hidden": hidden, "params": np.round(mean, 6).tolist()}, indent=1)
        )
        print("written agents/mpc_route/params.json; next: uv run sbf check mpc_route --task=small (and --task=full)")
    else:
        print(f"not written (it does not beat plain MPC on every held-out map, or --nowrite); the run is in {run}")


if __name__ == "__main__":
    fire.Fire(main)
