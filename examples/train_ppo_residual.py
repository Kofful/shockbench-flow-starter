"""Train agents/mpc_ppo's network by PPO (Stable-Baselines3) on small and full together. Needs ``uv sync --extra rl``.

    uv run python examples/train_ppo_residual.py --minutes=10 --n_small=2 --n_full=1 --pool_small=16 --pool_full=4 \
        --holdout_small=8 --holdout_full=2 --nowrite                              # a smoke test
    uv run python examples/train_ppo_residual.py --minutes=480                    # a night: 4 small + 2 full games
    uv run python examples/train_ppo_residual.py --start=outputs/train_ppo_residual/<run>   # go on from a run

Every week the MPC plan is made as in agents/mpc; PPO's policy sees the week's state (agents/mpc_ppo: ``features``)
and picks 12 numbers, theta per group of routes, that correct the plan's flows. The reward of a week is its cost
divided by the episode's (naive - clairvoyant) cost, so an episode's return is its score (RSS) minus a constant: PPO
maximises the score itself. Episodes come from pools of the training root on each map; their references (naive and
clairvoyant costs) and plain MPC's cost are computed once and cached, so the log shows PPO's score minus MPC's on the
same episodes (with the exploration noise of training). Every rollout the model, its observation normaliser and the
exported policy (policy.npz) are saved to outputs/train_ppo_residual/<run>/. At the end the exported policy is
compared with plain MPC on held-out episodes of small and full; only if it wins on both is it written to
agents/mpc_ppo/policy.npz.
"""

import json
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from joblib import Parallel, delayed
from shockbench_flow_agent import agent_config
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _play

from sbf_starter import ROOT, env_id, scoring
from sbf_starter.tracking import Tracker


try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
except ModuleNotFoundError as err:
    raise SystemExit(f"{err.name} is missing: install the rl extra first, uv sync --extra rl") from None


AGENT = ROOT / "agents" / "mpc_ppo"
MPC = ROOT / "agents" / "mpc"
WEIGHTS = "policy.npz"


def pool_table(task: str, entropy: int, size: int, n_jobs: int, cache: Path) -> dict:
    """Per episode of the pool: (naive - clairvoyant) and naive costs in USD, and plain MPC's cost (cached)."""
    es = scoring.episode_set(task, size, entropy=entropy, n_jobs=n_jobs)
    path = cache / f"mpc_{task}_{entropy}.json"
    mpc = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.is_file() else {}
    todo = [int(n) for n in es.episodes if int(n) not in mpc]
    if todo:
        print(f"plain MPC on {len(todo)} {task} episode(s) of root {entropy} (once, cached) ...", flush=True)
        rows = Parallel(n_jobs=n_jobs)(delayed(_play)(None, str(MPC), NO_ZIP_SHA256, es._spec, [n], None) for n in todo)
        for (row,) in rows:
            mpc[int(row["episode"])] = row["J_policy_cents"]
        path.write_text(json.dumps(mpc))
    table = {}
    for n, ref in zip(es.episodes, es.references):
        gap = (ref["J_naive_cents"] - ref["J_oracle_cents"]) / 100
        if ref.get("excluded") is None and gap > 0:
            table[int(n)] = (gap, ref["J_naive_cents"] / 100, mpc[int(n)] / 100)
    return table


class MpcResidualEnv(gym.Env):
    """A week of agents/mpc_ppo: the observation is its ``features``, the action its 12 theta (before THETA_SCALE)."""

    def __init__(self, task: str, entropy: int, table: dict, seed: int) -> None:
        import sys

        import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments in this process
        from shockbench_flow.disruption.sampler import sample_omega
        from shockbench_flow.hosting.tasks import split_label, task_generator

        sys.path.insert(0, str(AGENT))
        import agent as mpc_ppo

        self.mod, self.task, self.table = mpc_ppo, task, table
        self.episodes = sorted(table)
        self.env = gym.make(env_id(task), regime="standard")
        inst, params = task_generator(task)
        self.env.unwrapped.omega_source = lambda n: sample_omega(inst, params, entropy, n, split_label(entropy))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (mpc_ppo.OBS_SIZE,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (len(mpc_ppo.GROUPS),), np.float32)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.n = int(self.rng.choice(self.episodes))
        obs, info = self.env.reset(options={"episode": self.n})
        self.agent = self.mod.Agent(agent_config(info["static"], info["policy_seed"], self.env.unwrapped.layout, obs))
        self.obs, self.cost = obs, 0.0
        return self.agent.features(obs).astype(np.float32), {}

    def step(self, action):
        theta = self.mod.THETA_SCALE * np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = self.env.step(self.agent.week_action(self.obs, theta))
        self.obs, self.cost = obs, self.cost - reward
        gap, naive, mpc = self.table[self.n]
        info = {}
        if terminated or truncated:
            info = {"map": self.task, "rss": (naive - self.cost) / gap, "rss_mpc": (naive - mpc) / gap}
        return self.agent.features(obs).astype(np.float32), float(reward / gap), terminated, truncated, info


def make_env(task: str, entropy: int, table: dict, seed: int):
    return lambda: MpcResidualEnv(task, entropy, table, seed)


def export(model: PPO, venv: VecNormalize, path: Path) -> Path:
    """The policy's deterministic network and the observation normaliser, as numpy arrays (agents/mpc_ppo reads it)."""
    layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    layers.append(model.policy.action_net)
    arrays = {"obs_mean": venv.obs_rms.mean, "obs_var": venv.obs_rms.var, "clip_obs": np.array(venv.clip_obs)}
    for i, layer in enumerate(layers):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    np.savez(path, **arrays)
    return path


class Progress(BaseCallback):
    """Prints the last episodes' score minus MPC's per map after every rollout and saves the run."""

    def __init__(self, run: Path, venv: VecNormalize, minutes: float, tracker: Tracker, window: int = 40) -> None:
        super().__init__()
        self.run, self.venv, self.tracker, self.window = run, venv, tracker, window
        self.deadline = time.monotonic() + 60 * minutes
        self.start = time.monotonic()
        self.recent = {"small": deque(maxlen=window), "full": deque(maxlen=window)}
        self.done = {"small": 0, "full": 0}
        self.log = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "rss" in info:
                self.recent[info["map"]].append((info["rss"], info["rss_mpc"]))
                self.done[info["map"]] += 1
        return time.monotonic() < self.deadline

    def _on_rollout_end(self) -> None:
        row = {"minutes": round((time.monotonic() - self.start) / 60, 1), "steps": int(self.num_timesteps)}
        text = []
        for task, recent in self.recent.items():
            if recent:
                rss, rss_mpc = np.array(recent).T
                row[task] = {"episodes": self.done[task], "rss": rss.mean(), "minus_mpc": (rss - rss_mpc).mean()}
                text.append(
                    f"{task}: {rss.mean():.4f}, ppo - mpc {(rss - rss_mpc).mean():+.4f} ({self.done[task]} ep.)"
                )
        self.log.append(row)
        metrics = {"minutes": row["minutes"], "exploration_std": float(np.exp(self.model.policy.log_std.mean().item()))}
        for task in self.recent:
            if task in row:
                r = row[task]
                metrics |= {f"{task}_score": r["rss"], f"{task}_minus_mpc": r["minus_mpc"]}
                metrics[f"{task}_mpc"] = r["rss"] - r["minus_mpc"]
        self.tracker.log(metrics, step=row["steps"])
        print(f"{row['minutes']:6.1f} min, {row['steps']} weeks | " + " | ".join(text), flush=True)
        self.model.save(self.run / "model.zip")
        self.venv.save(str(self.run / "vecnormalize.pkl"))
        export(self.model, self.venv, self.run / WEIGHTS)
        (self.run / "log.json").write_text(json.dumps(self.log, indent=1))


def candidate(weights: Path | None, folder: Path) -> Path:
    if folder.exists():
        shutil.rmtree(folder)
    shutil.copytree(AGENT, folder, ignore=shutil.ignore_patterns(WEIGHTS, "__pycache__"))
    if weights is not None:
        shutil.copy(weights, folder / WEIGHTS)
    return folder


def main(
    minutes: float = 480,
    n_small: int = 4,
    n_full: int = 2,
    entropy: int = 1003,
    pool_small: int = 256,
    pool_full: int = 32,
    n_steps: int = 104,
    batch_size: int = 104,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    log_std_init: float = -1.6,
    net_arch: tuple[int, ...] = (64, 64),
    holdout_entropy: int = 9001,
    holdout_small: int = 64,
    holdout_full: int = 8,
    n_jobs: int = 4,
    start: str | None = None,
    seed: int = 0,
    write: bool = True,
    out: str | None = None,
) -> None:
    """Train, then check the exported policy against plain MPC on held-out small and full episodes.

    Args:
        minutes: training time (the references and MPC's costs of the pools come first, once).
        n_small: parallel games on small.
        n_full: parallel games on full (about 5 times slower a week: the small games wait for them).
        entropy: the training root of both maps (never 0, nor the held-out one; 1001, 1002 are the ES runs').
        pool_small: small episodes of the training root the games draw from.
        pool_full: full episodes of the training root the games draw from.
        n_steps: weeks per game in a rollout (104: one full episode, two small ones).
        batch_size: PPO's minibatch.
        n_epochs: PPO's passes over a rollout.
        learning_rate: PPO's step size.
        gamma: the discount per week.
        gae_lambda: GAE's lambda.
        log_std_init: the exploration's initial log standard deviation (of the action, before THETA_SCALE).
        net_arch: hidden layers of the policy and of the value network.
        holdout_entropy: the root of the final check.
        holdout_small: small episodes of the final check.
        holdout_full: full episodes of the final check.
        n_jobs: workers of the references, MPC's costs and the final check.
        start: a run folder (model.zip, vecnormalize.pkl) to go on from.
        seed: the seed of PPO and of the episode draws.
        write: write the winner to agents/mpc_ppo/policy.npz.
        out: the run folder (default: outputs/train_ppo_residual/<date_time>).
    """
    params = dict(locals())
    if entropy in (0, holdout_entropy):
        raise ValueError("train on a root of your own, neither the dev root 0 nor the held-out root")
    base = ROOT / "outputs" / "train_ppo_residual"
    run = Path(out or base / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    tables = {
        task: pool_table(task, entropy, size, n_jobs, base)
        for task, size, n in (("small", pool_small, n_small), ("full", pool_full, n_full))
        if n > 0
    }
    thunks = [make_env("small", entropy, tables["small"], seed + i) for i in range(n_small)]
    thunks += [make_env("full", entropy, tables["full"], seed + 100 + i) for i in range(n_full)]
    venv = SubprocVecEnv(thunks)
    if start:
        venv = VecNormalize.load(str(Path(start) / "vecnormalize.pkl"), venv)
        model = PPO.load(Path(start) / "model.zip", env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=gamma)
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            policy_kwargs={"net_arch": list(net_arch), "log_std_init": log_std_init},
            seed=seed,
            device="cpu",
            verbose=0,
        )
    print(f"training {minutes:g} min on {n_small} small + {n_full} full games; the run is in {run}", flush=True)
    tracker = Tracker("mpc_ppo", run.name, params | {"run_folder": run})
    progress = Progress(run, venv, minutes, tracker)
    model.learn(total_timesteps=10**9, callback=progress, reset_num_timesteps=not start)
    progress._on_rollout_end()
    venv.close()

    weights = run / WEIGHTS
    with tempfile.TemporaryDirectory(prefix="sbf-ppo-") as tmp:
        ppo = str(candidate(weights, Path(tmp) / "ppo"))
        wins = []
        for task, episodes in (("small", holdout_small), ("full", holdout_full)):
            if episodes <= 0:
                continue
            es = scoring.episode_set(task, episodes, entropy=holdout_entropy, n_jobs=n_jobs)
            cmp = es.compare(ppo, str(MPC), names=("mpc_ppo", "mpc"), n_jobs=n_jobs)
            print(f"\nheld out, {task}, {episodes} episodes of root {holdout_entropy}:\n{cmp}", flush=True)
            (run / f"holdout_{task}.txt").write_text(str(cmp))
            wins.append(cmp.diff is not None and cmp.diff > 0)
            tracker.log({f"holdout_{task}_minus_mpc": cmp.diff}, step=progress.num_timesteps)
            tracker.artifact(run / f"holdout_{task}.txt")
    tracker.artifact(weights)
    tracker.end()
    if write and wins and all(wins):
        shutil.copy(weights, AGENT / WEIGHTS)
        print(f"written agents/mpc_ppo/{WEIGHTS}; next: uv run sbf check mpc_ppo --task=small (and --task=full)")
    else:
        print(f"not written (it does not beat plain MPC on every held-out map, or --nowrite); the run is in {run}")


if __name__ == "__main__":
    fire.Fire(main)
