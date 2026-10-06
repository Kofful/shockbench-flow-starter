"""Record raw training data for forecasting disruptions: what an agent sees each week, and what truly happens.

    uv run python examples/collect_data.py --task=small --entropy=1001 --episodes=2000     # training data
    uv run python examples/collect_data.py --task=small --entropy=1002 --episodes=300      # held out
    uv run python examples/collect_data.py --task=full --entropy=2001 --episodes=300

One compressed file per episode, data/<task>_e<entropy>/ep<n>.npz, and the dataset's tables once in meta.json. The
episodes are played with zero flows: the disruptions, warnings and announcements are drawn before an episode starts and
no action changes them, so every agent sees the same signals. The agent's own state (stock, pipeline, costs) is not
recorded. Episodes already on disk are skipped, so an interrupted run resumes.

Keys of an episode file (T weeks; row t - 1 is week t, the observation the agent decides week t on):

- ``obs/<key>`` (T, ...): the Dict observation's exogenous blocks, ``week``, ``graph_now.*``, ``warning.score``,
  ``demand_forecast.qty``, each with its ``.observed`` mask (docs/fields/<task>.md);
- ``messages/<column>``, ``pending_prohibitions/<column>``, ``closure_end/<column>``: the live entries of those padded
  lists as long tables, one row per (week, entry), with a ``week`` column, each column with its ``.observed`` mask;
- ``truth/<field>`` (T, ...): the scenario's true weekly marks (``shockbench_flow.marks.WeeklyMarks``): ``o`` the
  straits' open fraction, ``u`` the edges' capacities, ``prohibited``, ``tariff``, ``supply``, ``G_bar`` / ``y_bar``
  the grids', ``R`` / ``R_osat`` the fabs' and OSATs' restoration, ``demand``, ``wr_class``, ...; mind the weeks:
  ``truth/<x>_now`` row t - 1 is exactly what the observation of week t shows (the state at the start of week t), and
  ``truth/<x>`` row t - 1 is week t's average, NOT known yet when deciding week t: labels start there;
- ``truth_fab_hits/<column>``: the fab outages (fab ordinal, onset in weeks, severity).

    import numpy as np
    ep = np.load("data/small_e1001/ep00000.npz")
    ep["obs/warning.score"].shape, ep["truth/o"].shape      # (52, 31) (52, 7)
"""

import importlib
import json
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401  registers the ShockBench/* environments
from joblib import Parallel, delayed
from shockbench_flow.policies.lp_common import WINDOW_FIELDS
from shockbench_flow_agent import agent_config

from sbf_starter import ROOT, env_id


DENSE = ("week", "graph_now.", "warning.", "demand_forecast.")  # the exogenous blocks of the observation
LISTS = {"messages": "msg_id", "pending_prohibitions": "edge", "closure_end": "chokepoint"}  # block -> live column


def _small(a: np.ndarray) -> np.ndarray:
    """float32 for floats, int32 for integers, int8 kept: about half the bytes, enough for features and labels."""
    a = np.asarray(a)
    if a.dtype.kind == "f":
        return a.astype(np.float32)
    if a.dtype.kind in "iu" and a.dtype.itemsize > 4:
        return a.astype(np.int32)
    return a


def record(task: str, entropy: int, n: int, path: Path) -> float:
    """Play episode ``n`` of root ``entropy`` with zero flows and write its file; the seconds it took."""
    importlib.import_module("shockbench_flow_gym")  # registers the environments in a joblib worker too
    start = time.perf_counter()
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    marks = env.unwrapped.core._ep.marks  # the scenario's true marks, the oracle's (private: a local tool)
    zero = {"flows": np.zeros(env.action_space["flows"].shape)}
    dense = {k: [] for k in obs if k.startswith(DENSE)}
    lists = {b: {} for b in LISTS}
    for t in range(1, marks.T + 1):
        for k in dense:
            dense[k].append(np.asarray(obs[k]))
        for block, live_col in LISTS.items():
            live = np.flatnonzero(obs[f"{block}.{live_col}.observed"])
            table = lists[block]
            table.setdefault("week", []).append(np.full(live.size, t, dtype=np.int32))
            for key in obs:
                if key.startswith(f"{block}."):
                    table.setdefault(key.removeprefix(f"{block}."), []).append(np.asarray(obs[key])[live])
        obs, _reward, terminated, truncated, info = env.step(zero)
        if terminated or truncated:
            break
    out = {f"obs/{k}": _small(np.stack(v)) for k, v in dense.items()}
    for block, table in lists.items():
        out |= {f"{block}/{col}": _small(np.concatenate(v)) for col, v in table.items()}
    out |= {f"truth/{f}": _small(getattr(marks, f)) for f in WINDOW_FIELDS}
    hits = marks.fab_hits
    out |= {
        "truth_fab_hits/fab": np.array([h.fab for h in hits], dtype=np.int32),
        "truth_fab_hits/onset": np.array([h.onset for h in hits], dtype=np.float32),
        "truth_fab_hits/severity": np.array([h.severity for h in hits], dtype=np.float32),
    }
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)  # whole files only: an interrupted write leaves no half episode behind
    return time.perf_counter() - start


def write_meta(task: str, entropy: int, folder: Path) -> None:
    """The dataset's tables, the same for every episode: what each position of an array stands for."""
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": 0})
    config = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    static = config["static"]
    meta = {
        "task": task,
        "entropy": entropy,
        "T": config["T"],
        "instance_id": static["instance_id"],
        "layout": config["layout"],
        "regions": static["regions"],
        "dyads": static["dyads"],
        "nodes": static["nodes"],
        "commodities": static["commodities"],
        "edges": {k: static["edges"][k] for k in ("id", "tail", "head", "mode", "tau0", "c0", "u0")},
        "lanes": static["lanes"],
        "truth_fields": list(WINDOW_FIELDS),
        "codes": {
            "messages.channel": [
                "tariff_formal",
                "tariff_informal",
                "tariff_final",
                "sanction_legal",
                "ties_threat",
                "mid_threat",
            ],
            "messages.kind": ["proposal", "final_notice", "threat", "publication", "withdrawal"],
            "messages.target_kind": ["chokepoint", "edge", "node", "region"],
            "graph_now.war_risk": ["none", "red_sea", "hormuz_2026"],
        },
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


def main(task: str = "tiny", entropy: int = 1001, episodes: int = 10, first: int = 0, n_jobs: int = -1) -> None:
    """Record episodes ``first`` .. ``first + episodes - 1`` of root ``entropy`` into data/<task>_e<entropy>/.

    Args:
        task: tiny, small or full.
        entropy: the scenarios' root: never 0 (the dev episodes) nor a root you evaluate on (9001 in PLAN.md).
        episodes: how many episodes.
        first: the first episode index (to add more to a dataset later).
        n_jobs: parallel episodes (-1: all cores).
    """
    if entropy == 0:
        raise ValueError("root 0 holds the dev episodes the scores are confirmed on: record a root of your own")
    folder = ROOT / "data" / f"{task}_e{entropy}"
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "meta.json").is_file():
        write_meta(task, entropy, folder)
    todo = [n for n in range(first, first + episodes) if not (folder / f"ep{n:05d}.npz").is_file()]
    print(f"{task}, root {entropy}: {episodes - len(todo)} of {episodes} episodes on disk, recording {len(todo)}")
    start = time.perf_counter()
    seconds = Parallel(n_jobs=n_jobs)(delayed(record)(task, entropy, n, folder / f"ep{n:05d}.npz") for n in todo)
    size = sum(f.stat().st_size for f in folder.glob("ep?????.npz"))
    files = len(list(folder.glob("ep?????.npz")))
    if seconds:
        print(
            f"recorded {len(seconds)} episodes in {time.perf_counter() - start:.0f} s ({np.mean(seconds):.1f} s each)"
        )
    print(f"{folder.relative_to(ROOT)}: {files} episodes, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    fire.Fire(main)
