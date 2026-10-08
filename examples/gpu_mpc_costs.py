"""Plain MPC's cost on the training episodes, for the trainers' "minus MPC" lines (sbf_starter.gpu.maps.MPC_CACHE).

    uv run python examples/gpu_mpc_costs.py --task=full --episodes=128 --n_jobs=1

Plays agents/mpc on the episodes of the training root not yet in the cache, one game per worker (a full game takes
about a minute and half a gigabyte), and adds each cost to the cache file after every game, so an interrupted run
loses at most the games under way.
"""

import json

import fire
from joblib import Parallel, delayed
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _play

from sbf_starter import ROOT, scoring
from sbf_starter.gpu.maps import MPC_CACHE


def main(task: str = "full", episodes: int = 128, entropy: int = 1003, n_jobs: int = 1) -> None:
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    path = MPC_CACHE / f"mpc_{task}_{entropy}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.is_file() else {}
    todo = [int(n) for n in es.episodes if int(n) not in done]
    print(f"plain MPC on {len(todo)} {task} episode(s) of root {entropy}, {n_jobs} worker(s) ...", flush=True)
    mpc = str(ROOT / "agents" / "mpc")
    for start in range(0, len(todo), n_jobs):
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_play)(None, mpc, NO_ZIP_SHA256, es._spec, [n], None) for n in todo[start : start + n_jobs]
        )
        for (row,) in rows:
            done[int(row["episode"])] = row["J_policy_cents"]
        path.write_text(json.dumps(done))
        print(f"  {len(done)} of {episodes} known", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
