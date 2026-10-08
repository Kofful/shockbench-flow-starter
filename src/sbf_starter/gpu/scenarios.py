"""Episodes for the tensor simulator, drawn once on the CPU and kept on disk, loaded straight into GPU tensors.

``build(task, entropy, episodes)`` draws each episode's scenario and weekly marks with the package
(``sample_omega``, ``compute_marks``: about 2 s an episode of small) in worker processes and saves the arrays the
simulator and the policy's observation need, one compressed file per episode, under
``~/.cache/shockbench-flow/gpu/<task>/entropy-<root>/``. ``load`` fills a ``Batch`` episode by episode into tensors
allocated once on the device, so a few thousand episodes never pass through the CPU's memory at once (WSL has little).

``build_info`` adds what the observation's information layer shows (``information.observe.wrap``, regime standard):
the demand forecast and the announced prohibitions of every week. They depend on the scenario only, not on what the
agent does, so one pass of the package's environment per episode (any actions) records them; ``load_info`` attaches
them to a ``Batch`` (``forecast`` [B, T, D, H], ``pending`` [B, T, E, K]: the effective week of the announced
prohibition of (edge, commodity), 0 for none).
"""

from pathlib import Path

import numpy as np
import torch

from sbf_starter.gpu.net import Batch


CACHE = Path.home() / ".cache" / "shockbench-flow" / "gpu"
FIELDS = (  # WeeklyMarks fields kept: the simulator's, then the instant values the observation shows
    "u",
    "c",
    "kappa",
    "supply",
    "G_bar",
    "y_bar",
    "R",
    "alpha_bar",
    "demand",
    "prohibited",
    "tariff",
    "c_wr",
    "h_queue",
    "u_now",
    "o_now",
    "supply_now",
)


def folder(task: str, entropy: int) -> Path:
    return CACHE / task / f"entropy-{entropy}"


def _draw(task: str, entropy: int, n: int) -> None:
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import split_label, task_generator
    from shockbench_flow.marks import compute_marks, osat_throughput

    inst, params = task_generator(task)
    m = compute_marks(inst, sample_omega(inst, params, entropy, n, split_label(entropy)))
    scrap = np.ones((inst.T, len(inst.fabs)))
    for hit in m.fab_hits:
        if 1 <= hit.onset_week <= inst.T:
            scrap[hit.onset_week - 1, hit.fab] *= 1.0 - hit.severity
    arrays = {f: np.asarray(getattr(m, f)) for f in FIELDS}
    arrays["osat_thr"] = osat_throughput(inst, m.R_osat)
    arrays["scrap"] = scrap
    path = folder(task, entropy) / f"{n}.npz"
    np.savez_compressed(path.with_suffix(".tmp.npz"), omega_hash=np.array(m.omega_hash), **arrays)
    path.with_suffix(".tmp.npz").rename(path)


def build(task: str, entropy: int, episodes, n_jobs: int = 4) -> list[int]:
    """Draw and save the episodes not cached yet; returns the episode numbers."""
    from joblib import Parallel, delayed

    ns = list(range(episodes)) if isinstance(episodes, int) else [int(n) for n in episodes]
    folder(task, entropy).mkdir(parents=True, exist_ok=True)
    todo = [n for n in ns if not (folder(task, entropy) / f"{n}.npz").is_file()]
    if todo:
        print(f"drawing {len(todo)} {task} episode(s) of root {entropy} ({n_jobs} workers) ...", flush=True)
        Parallel(n_jobs=n_jobs)(delayed(_draw)(task, entropy, n) for n in todo)
    return ns


def load(task: str, entropy: int, episodes, device: str = "cuda") -> Batch:
    """The cached ``episodes`` as one ``Batch`` on ``device`` (each file read and copied in turn)."""
    ns = list(range(episodes)) if isinstance(episodes, int) else [int(n) for n in episodes]
    out: dict[str, torch.Tensor] = {}
    for i, n in enumerate(ns):
        with np.load(folder(task, entropy) / f"{n}.npz") as f:
            for key in (*FIELDS, "osat_thr", "scrap"):
                a = f[key]
                if key not in out:
                    dtype = torch.bool if a.dtype == bool else torch.float64
                    out[key] = torch.empty((len(ns), *a.shape), dtype=dtype, device=device)
                out[key][i] = torch.as_tensor(a, device=device)
    batch = Batch(B=len(ns), **{k: out[k] for k in Batch.__dataclass_fields__ if k != "B"})
    batch.u_now, batch.o_now, batch.supply_now = out["u_now"], out["o_now"], out["supply_now"]
    return batch


def _draw_info(task: str, entropy: int, n: int) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import split_label, task_generator

    from sbf_starter import env_id

    inst, params = task_generator(task)
    env = gym.make(env_id(task), regime="standard")
    env.unwrapped.omega_source = lambda i: sample_omega(inst, params, entropy, i, split_label(entropy))
    obs, _ = env.reset(options={"episode": n})
    zero = {"flows": np.zeros(env.action_space["flows"].shape[0])}
    E, K = len(inst.edges), len(inst.commodities)
    forecast, pending = [], []
    for _t in range(inst.T):
        forecast.append(np.where(obs["demand_forecast.qty.observed"] == 1, obs["demand_forecast.qty"], 0.0))
        week = np.zeros((E, K), dtype=np.int16)
        live = obs["pending_prohibitions.edge.observed"] == 1
        for e, k, w in zip(
            obs["pending_prohibitions.edge"][live],
            obs["pending_prohibitions.k"][live],
            obs["pending_prohibitions.effective_week"][live],
        ):
            week[e, k] = w if week[e, k] == 0 else min(week[e, k], w)
        pending.append(week)
        obs, *_ = env.step(zero)
    path = folder(task, entropy) / f"{n}.info.npz"
    np.savez_compressed(path.with_suffix(".tmp.npz"), forecast=np.array(forecast), pending=np.array(pending))
    path.with_suffix(".tmp.npz").rename(path)


def build_info(task: str, entropy: int, episodes, n_jobs: int = 2) -> list[int]:
    """Record the information layer of the episodes not recorded yet; returns the episode numbers."""
    from joblib import Parallel, delayed

    ns = list(range(episodes)) if isinstance(episodes, int) else [int(n) for n in episodes]
    folder(task, entropy).mkdir(parents=True, exist_ok=True)
    todo = [n for n in ns if not (folder(task, entropy) / f"{n}.info.npz").is_file()]
    if todo:
        print(f"recording the information layer of {len(todo)} {task} episode(s) of root {entropy} ...", flush=True)
        Parallel(n_jobs=n_jobs)(delayed(_draw_info)(task, entropy, n) for n in todo)
    return ns


def load_info(batch: Batch, task: str, entropy: int, episodes, device: str = "cuda") -> Batch:
    """Attach the recorded forecast and announced prohibitions of ``episodes`` to ``batch`` (same order)."""
    ns = list(range(episodes)) if isinstance(episodes, int) else [int(n) for n in episodes]
    for i, n in enumerate(ns):
        with np.load(folder(task, entropy) / f"{n}.info.npz") as f:
            if i == 0:
                batch.forecast = torch.empty((len(ns), *f["forecast"].shape), dtype=torch.float64, device=device)
                batch.pending = torch.empty((len(ns), *f["pending"].shape), dtype=torch.int16, device=device)
            batch.forecast[i] = torch.as_tensor(f["forecast"], device=device)
            batch.pending[i] = torch.as_tensor(f["pending"], device=device)
    return batch
