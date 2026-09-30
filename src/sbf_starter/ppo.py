"""Train PPO (Stable-Baselines3) on a ShockBench-Flow network and export it as a TorchScript submission.

Needs the rl extra: ``uv sync --extra rl`` (Stable-Baselines3 and torch). ``scripts/python/05_train_ppo.py`` calls
``train(cfg, out)`` with ``configs/05_train_ppo.yaml``, the one place the settings live. The scoring container has
torch (CPU) but not Stable-Baselines3, so the trained policy is exported as one TorchScript file with an ``agent.py``
that loads it.

What it does:

1. Draws ``n_scenarios`` training scenarios from your own root ``entropy`` (``draw_scenarios``, cached under
   ``~/.cache/shockbench-flow`` or ``$SBF_CACHE_DIR``): the public generator, never the dev or the hidden scenarios.
2. Wraps each environment for RL: ``ScenarioPool`` serves those scenarios, ``CapacityFractionAction`` turns the action
   into a fraction in [0, 1] of each route's capacity (masked routes get 0; all ones is "send the maximum"),
   ``ScaleReward`` divides the reward by naive's mean weekly cost, ``FilterObservation`` drops the padded
   variable-length lists (``DROP_PREFIXES``) and ``FlattenObservation`` makes one vector of the rest,
   ``RescaleAction`` maps the action to [-1, 1], ``SlimInfo`` keeps the info small (the subprocesses pickle it);
   ``VecNormalize`` normalises the observation.
3. Trains PPO with an MLP policy (``net_arch`` hidden layers, ``activation`` between them) and saves the model, the
   normaliser's statistics and SB3's ``progress.csv`` in the run folder (the learning curve: ``rollout/ep_rew_mean``
   is minus an episode's cost in units of naive's mean weekly cost, about -26 for an average naive episode on Tiny).
4. Exports the deterministic policy (the mean of the Gaussian, clipped) with the normaliser as one TorchScript module,
   ``policy.pt``: the normaliser, SB3's own float32 layers whatever their sizes and activation, the clip and the map
   to a fraction of capacity, with the observation's field names inside. The ``agent.py`` beside it loads it once at
   module level (the CPU budget does not meter the import) and runs it on CPU torch, which the scoring image holds.
5. Checks the submission: the kit's validator accepts its zip, and the exported agent's flows equal the SB3 policy's
   on every week of ``check_episodes`` episodes, within ``ACTION_ATOL`` of a route's capacity.

Torch 2.14 marks TorchScript deprecated (a FutureWarning, which the agent silences); ``torch.jit.load`` still loads
the module in the scoring image's pinned torch (2.14.0, CPU).
"""

import copy
import csv
import json
import time
import warnings
from collections.abc import Mapping
from pathlib import Path

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* ids
import torch  # the rl extra: a ModuleNotFoundError here means `uv sync --extra rl`
from gymnasium.wrappers import FilterObservation, FlattenObservation, RescaleAction
from shockbench_flow_agent import load_agent_class
from shockbench_flow_agent.submission import agent_warnings, build_submission, check_zip
from shockbench_flow_gym.wrappers import CapacityFractionAction, ScaleReward, ScenarioPool, SlimInfo, draw_scenarios
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from helper.logging import logger
from sbf_starter import env_id
from sbf_starter.play import make_agent


# the padded variable-length lists of the Dict observation, left out of the flat vector (about 62,000 of its 62,800
# numbers on tiny); what is left is about 850 numbers: stock, backlog, the observed graph, masks, forecasts, warnings
DROP_PREFIXES = ("pipeline.", "queue_lots.", "wip.", "messages.", "pending_prohibitions.", "closure_end.")
ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
}
ACTION_ATOL = 1e-4  # the exported agent's flows may differ from the SB3 policy's by this fraction of a capacity
WEIGHTS = "policy.pt"

AGENT = '''"""A PPO policy trained with Stable-Baselines3 (scripts/python/05_train_ppo.py), run as TorchScript."""

import warnings
from pathlib import Path

import numpy as np
import torch


# loaded once, at import: the CPU budget meters Agent(config) and act, not the import of this file
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # torch 2.14 marks TorchScript deprecated; it still loads
    POLICY = torch.jit.load(str(Path(__file__).resolve().parent / "policy.pt"), map_location="cpu")
POLICY.eval()
KEYS = list(POLICY.obs_keys)  # the flat observation: these fields, in this order


class Agent:
    def __init__(self, config):
        u0 = config["static"]["edges"]["u0"]  # each edge's nominal capacity per week
        self.capacity = np.array([u0[e] for e in config["static"]["action_slots"]["edge"]], dtype=float)

    def act(self, observation):
        x = np.concatenate([np.asarray(observation[k], dtype=np.float64).ravel() for k in KEYS])
        with torch.inference_mode():
            fraction = POLICY(torch.from_numpy(x)).numpy()  # in [0, 1]: all ones is "send the maximum"
        return {"flows": fraction * self.capacity * observation["action_mask"]}
'''


def observation_keys(env: gym.Env) -> list[str]:
    """The Dict observation's fields kept in the flat vector (all but ``DROP_PREFIXES``), in its order.

    ``FlattenObservation`` concatenates the fields of ``FilterObservation``'s Dict space in that space's key order
    (gymnasium sorts a Dict space's keys), which the exported agent repeats.
    """
    kept = [k for k in env.observation_space.spaces if not k.startswith(DROP_PREFIXES)]
    return list(FilterObservation(env, kept).observation_space.spaces)


class LastDictObservation(gym.Wrapper):
    """Keeps the last Dict observation in ``last_dict`` (for the check of the exported agent only)."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dict = obs
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self.last_dict = out[0]
        return out


def make_env(task: str, scenarios: list[Path], entropy: int, regime: str, *, check: bool = False):
    """A thunk building one training environment on the pool's scenarios (already in the cache).

    ``check`` builds the variant of ``check_submission``: no ``Monitor``, the last Dict observation kept.
    """

    def thunk() -> gym.Env:
        env = ScenarioPool(gym.make(env_id(task), regime=regime), len(scenarios), entropy)
        if check:
            env = LastDictObservation(env)
        env = ScaleReward(CapacityFractionAction(env))
        env = FlattenObservation(FilterObservation(env, observation_keys(env)))
        n = env.action_space.shape
        env = RescaleAction(env, np.full(n, -1.0, np.float32), np.full(n, 1.0, np.float32))
        return env if check else Monitor(SlimInfo(env))  # SlimInfo: a small info dict through the pipes

    return thunk


class StopAfter(BaseCallback):
    """Stops training after ``minutes`` of wall time (None: never)."""

    def __init__(self, minutes: float | None) -> None:
        super().__init__()
        self.deadline = None if minutes is None else time.monotonic() + 60 * minutes

    def _on_step(self) -> bool:
        return self.deadline is None or time.monotonic() < self.deadline


class ExportedPolicy(torch.nn.Module):
    """The deterministic policy as one module: fraction of capacity in [0, 1] of a flat observation (float64).

    ``VecNormalize``'s normalisation and clip (float64, as SB3's numpy), SB3's own layers (float32, as SB3 runs
    them: any sizes, any activation), the action space's clip and ``RescaleAction``'s map back to [0, 1].
    ``obs_keys`` names the Dict observation's fields the flat vector concatenates, in order.
    """

    obs_keys: list[str]

    def __init__(self, model: PPO, venv: VecNormalize, keys: list[str]) -> None:
        super().__init__()
        policy = model.policy
        if not isinstance(getattr(policy.features_extractor, "flatten", None), torch.nn.Flatten):
            raise ValueError("the export expects the flat observation (SB3's FlattenExtractor)")
        layers = [*policy.mlp_extractor.policy_net, policy.action_net]
        self.net = torch.nn.Sequential(*[copy.deepcopy(m).cpu() for m in layers]).eval()
        space = model.action_space
        self.register_buffer("mean", torch.as_tensor(venv.obs_rms.mean, dtype=torch.float64))
        self.register_buffer("std", torch.sqrt(torch.as_tensor(venv.obs_rms.var, dtype=torch.float64) + venv.epsilon))
        self.register_buffer("low", torch.as_tensor(space.low, dtype=torch.float64))
        self.register_buffer("high", torch.as_tensor(space.high, dtype=torch.float64))
        self.clip_obs = float(venv.clip_obs)
        self.obs_keys = list(keys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.clamp((x - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        a = self.net(z.to(torch.float32)).to(torch.float64)
        a = torch.minimum(torch.maximum(a, self.low), self.high)
        return torch.clamp((a - self.low) / (self.high - self.low), 0.0, 1.0)


def export(model: PPO, venv: VecNormalize, keys: list[str], path: Path) -> Path:
    """The deterministic policy and the normaliser as one TorchScript file (``ExportedPolicy``, ``torch.jit.save``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch 2.14 marks TorchScript deprecated (module docstring)
        torch.jit.save(torch.jit.script(ExportedPolicy(model, venv, keys)), str(path))
    return path


def write_submission(weights: Path, folder: Path) -> Path:
    """``folder`` with ``agent.py`` (``AGENT``) and its weights, zipped by the kit to ``folder.zip``."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agent.py").write_text(AGENT)
    (folder / WEIGHTS).write_bytes(weights.read_bytes())
    return build_submission(folder, folder.with_suffix(".zip"))


def _inner(env: gym.Env, cls: type) -> gym.Env:
    while not isinstance(env, cls):
        env = env.env
    return env


def check_submission(task, model, venv, scenarios, entropy, regime, folder: Path, episodes: int) -> float:
    """The largest difference, as a fraction of a route's capacity, between the exported agent's and SB3's flows.

    Plays ``episodes`` pool scenarios with the SB3 policy through the training wrappers; each week the exported agent
    (``folder/agent.py``, loaded as the scorer loads it) acts on the same Dict observation, which is the one the
    scoring container hands over.

    Raises:
        RuntimeError: if the flat vector is not the concatenation of the kept fields the agent reads.

    """
    agent_class = load_agent_class(folder, "ppo_agent")
    env = make_env(task, scenarios, entropy, regime, check=True)()
    keep, fraction = _inner(env, LastDictObservation), _inner(env, CapacityFractionAction)
    keys = observation_keys(keep)
    worst = 0.0
    for i in range(episodes):
        flat, info = env.reset(seed=i, options={"pool_index": i % len(scenarios)})
        agent = make_agent(env, agent_class, keep.last_dict, info)
        done = False
        while not done:
            obs = keep.last_dict
            if not np.array_equal(flat, np.concatenate([np.asarray(obs[k], np.float64).ravel() for k in keys])):
                raise RuntimeError("the flat observation is not the concatenation of the fields the agent reads")
            action, _ = model.predict(venv.normalize_obs(flat[None]), deterministic=True)
            theirs = fraction.action(env.action(action[0]))["flows"]  # RescaleAction, then capacity times mask
            mine = agent.act(obs)["flows"]
            worst = max(worst, float(np.max(np.abs(mine - theirs) / fraction.capacity)))
            flat, _reward, done, _truncated, _info = env.step(action[0])
    return worst


def learning_curve(progress: Path, points: int = 12) -> list[list[float]]:
    """[timesteps, mean return of the last 100 episodes] from SB3's ``progress.csv``, about ``points`` rows."""
    with progress.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("rollout/ep_rew_mean")]
    keep = sorted({round(i * (len(rows) - 1) / max(1, points - 1)) for i in range(points)}) if rows else []
    return [
        [int(float(rows[i]["time/total_timesteps"])), round(float(rows[i]["rollout/ep_rew_mean"]), 4)] for i in keep
    ]


def train(cfg: Mapping, out: Path) -> dict:
    """Train, export, write and check the submission in the run folder ``out`` (module docstring); returns a summary.

    ``cfg`` holds ``configs/05_train_ppo.yaml``'s keys (``task.name`` the network); nothing here has defaults of its
    own. The summary is also written to ``out/summary.json``.

    Raises:
        ValueError: on an ``activation`` outside ``ACTIVATIONS``.
        RuntimeError: if the exported agent's flows differ from the SB3 policy's by more than ``ACTION_ATOL``.

    """
    task, regime, entropy = cfg["task"]["name"], cfg["regime"], int(cfg["entropy"])
    if cfg["activation"] not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {list(ACTIVATIONS)}, got {cfg['activation']!r}")
    T = gym.make(env_id(task)).unwrapped.instance.T
    n_steps = cfg["n_steps"] or 8 * T
    torch.set_num_threads(int(cfg["torch_threads"]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    n_envs = int(cfg["n_envs"])
    scenarios = draw_scenarios(task, int(cfg["n_scenarios"]), entropy, n_jobs=max(1, n_envs))
    logger.info("{} training scenarios of root {}: {}", cfg["n_scenarios"], entropy, scenarios[0].parent)
    thunks = [make_env(task, scenarios, entropy, regime) for _ in range(n_envs)]
    venv = VecNormalize(
        SubprocVecEnv(thunks) if n_envs > 1 else DummyVecEnv(thunks),
        norm_obs=True,
        norm_reward=False,
        gamma=cfg["gamma"],
    )
    arch = list(cfg["net_arch"])
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=n_steps,
        batch_size=cfg["batch_size"] or n_steps,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        policy_kwargs={"net_arch": {"pi": arch, "vf": arch}, "activation_fn": ACTIVATIONS[cfg["activation"]]},
        seed=int(cfg["seed"]),
    )
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    model.learn(total_timesteps=int(cfg["total_timesteps"]), callback=StopAfter(cfg["max_minutes"]))
    train_seconds = time.monotonic() - start
    venv.close()
    venv.training = False  # the normaliser's statistics are frozen from here on
    model.save(out / f"ppo_{task}.zip")
    venv.save(out / "vecnormalize.pkl")
    keys = observation_keys(gym.make(env_id(task), regime=regime))
    weights = export(model, venv, keys, out / WEIGHTS)
    folder = out / "submission"
    zip_path = write_submission(weights, folder)
    checked = check_zip(zip_path)  # raises SubmissionError on anything the scorer would refuse
    worst = check_submission(task, model, venv, scenarios, entropy, regime, folder, int(cfg["check_episodes"]))
    summary = {
        "timesteps": int(model.num_timesteps),
        "train_seconds": round(train_seconds, 1),
        "observation_size": len(venv.obs_rms.mean),
        "net_arch": arch,
        "activation": cfg["activation"],
        "learning_curve": learning_curve(out / "progress.csv"),
        "model": str(out / f"ppo_{task}.zip"),
        "submission": str(folder),
        "zip": str(zip_path),
        "zip_sha256": checked.sha256,
        "agent_warnings": agent_warnings((folder / "agent.py").read_bytes()),
        "max_flow_difference": worst,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if worst > ACTION_ATOL:
        raise RuntimeError(f"the exported agent's flows differ from the SB3 policy's by {worst:.2e} of a capacity")
    return summary


__all__ = ["AGENT", "ACTIVATIONS", "ExportedPolicy", "export", "train"]
