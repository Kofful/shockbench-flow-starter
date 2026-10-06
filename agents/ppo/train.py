"""Train and export the recurrent residual policy in this folder.

The trainer uses non-development scenario roots, undiscounted finite-horizon
returns, bounded residual actions, recurrent state, CUDA minibatches, exact RSS
validation on a held-out root, best-checkpoint selection, and resumable
checkpoints.  Root 0 is optional final reporting only and never selects a
checkpoint.

Smoke test::

    uv run python agents/ppo/train.py --task=tiny --updates=1 --n_envs=1 \
        --train_scenarios=2 --validation_episodes=1 --quick_validation

Serious Small run::

    uv run python agents/ppo/train.py --task=small --updates=300 --n_envs=8 \
        --train_scenarios=4096 --validation_episodes=100 --install

Resume by passing the prior run folder to ``--resume``.  ``--install`` copies
the best validation checkpoint to ``agents/ppo/policy.pt``; no package is
published and no network service is contacted.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import shutil
import sys
import time
import warnings
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401 - environment registration
import torch
from shockbench_flow_gym.wrappers import CapacityFractionAction, ScaleReward, ScenarioPool, SlimInfo
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sbf_starter import ROOT, env_id, scoring


AGENT_DIR = Path(__file__).resolve().parent
AGENT_FILE = AGENT_DIR / "agent.py"
INSTALLED_POLICY = AGENT_DIR / "policy.pt"
FEATURE_VERSION = 1
FEATURE_DIM = 96
HIDDEN_SIZE = 64
RESIDUAL_SCALE = 2.0
EPS = 1e-4
HISTORY_FIELDS = (
    "update",
    "train_entropy",
    "timesteps",
    "episodes",
    "mean_episode_return",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "learning_rate",
    "validation_rss",
    "validation_cost_usd",
    "best_validation_rss",
    "elapsed_seconds",
)


class RecurrentResidualPolicy(torch.nn.Module):
    """Size-independent shared route encoder, recurrent actor and pooled critic."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        width = 96
        self.feature_version = FEATURE_VERSION
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.residual_scale = RESIDUAL_SCALE
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, width),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
        )
        self.context = torch.nn.Sequential(torch.nn.Linear(3 * width, hidden_size), torch.nn.Tanh())
        self.gru = torch.nn.GRUCell(hidden_size, hidden_size)
        actor_width = hidden_size + 3 * width
        self.actor_body = torch.nn.Sequential(
            torch.nn.Linear(actor_width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width // 2),
            torch.nn.SiLU(),
        )
        self.actor_mean = torch.nn.Linear(width // 2, 1)
        self.actor_log_std = torch.nn.Linear(width // 2, 1)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(2 * width + 2 * hidden_size, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 1),
        )
        # Zero residual means exact baseline inference before any training.
        torch.nn.init.zeros_(self.actor_mean.weight)
        torch.nn.init.zeros_(self.actor_mean.bias)
        torch.nn.init.zeros_(self.actor_log_std.weight)
        torch.nn.init.constant_(self.actor_log_std.bias, -0.7)
        torch.nn.init.zeros_(self.value[-1].weight)
        torch.nn.init.zeros_(self.value[-1].bias)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        local = self.encoder(features)
        pooled_mean = torch.mean(local, dim=1)
        pooled_max = torch.amax(local, dim=1)
        pooled = torch.cat((pooled_mean, pooled_max), dim=-1)
        expanded = pooled.unsqueeze(1).expand(-1, local.shape[1], -1)
        recurrent_input = self.context(torch.cat((local, expanded), dim=-1))
        next_hidden = self.gru(
            recurrent_input.reshape(-1, self.hidden_size), hidden.reshape(-1, self.hidden_size)
        ).reshape_as(hidden)
        actor = self.actor_body(torch.cat((local, expanded, next_hidden), dim=-1))
        mean = self.actor_mean(actor).squeeze(-1)
        log_std = torch.clamp(self.actor_log_std(actor).squeeze(-1), -4.0, 0.5)
        hidden_pool = torch.cat((torch.mean(next_hidden, dim=1), torch.amax(next_hidden, dim=1)), dim=-1)
        value = self.value(torch.cat((pooled, hidden_pool), dim=-1)).squeeze(-1)
        return mean, log_std, value, next_hidden


def _export(model: RecurrentResidualPolicy, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = RecurrentResidualPolicy(model.feature_dim, model.hidden_size)
    cpu.load_state_dict({name: value.detach().cpu() for name, value in model.state_dict().items()})
    cpu.eval()
    with torch.inference_mode(), torch.jit.optimized_execution(True), warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        scripted = torch.jit.script(cpu)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.jit.save(scripted, str(temporary))
        temporary.replace(path)
    return path


def _initialise_policy(path: Path = INSTALLED_POLICY, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    return _export(RecurrentResidualPolicy(), path)


# Let a clean checkout run the trainer to create its first deployable baseline.
# Normal evaluation still requires the committed policy file to exist.
if not INSTALLED_POLICY.is_file():
    _initialise_policy()

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
deployment = importlib.import_module("agent")
FeatureBuilder = deployment.FeatureBuilder


def _make_env(task: str, scenarios: int, entropy: int, regime: str):
    def thunk():
        env = ScenarioPool(gym.make(env_id(task), regime=regime), scenarios, entropy)
        env = ScaleReward(CapacityFractionAction(env))
        return SlimInfo(env)

    return thunk


def _agent_config(task: str, regime: str):
    from shockbench_flow_gym import agent_config_from_reset

    env = gym.make(env_id(task), regime=regime)
    try:
        observation, info = env.reset(seed=1)
        return agent_config_from_reset(env, observation, info)
    finally:
        env.close()


def _encode_batch(builders, observations):
    features = []
    base = []
    active = []
    for i, builder in enumerate(builders):
        observation = {key: value[i] for key, value in observations.items()}
        features.append(builder.encode(observation))
        base.append(builder.last_base)
        active.append(builder.last_active)
    return (
        np.stack(features).astype(np.float32),
        np.stack(base).astype(np.float32),
        np.stack(active).astype(np.float32),
    )


def _fraction_tensor(base: torch.Tensor, active: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(base, EPS, 1.0 - EPS)
    logits = torch.log(clipped) - torch.log1p(-clipped)
    return torch.sigmoid(logits + RESIDUAL_SCALE * torch.tanh(latent)) * active


def _log_prob(distribution, latent: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    # The transformation Jacobian cancels from PPO's old/new likelihood ratio.
    return torch.sum(distribution.log_prob(latent) * active, dim=-1)


def _entropy(distribution, active: torch.Tensor) -> torch.Tensor:
    count = torch.clamp(torch.sum(active, dim=-1), min=1.0)
    return torch.sum(distribution.entropy() * active, dim=-1) / count


def _write_agent(folder: Path, policy: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AGENT_FILE, folder / "agent.py")
    destination = folder / "policy.pt"
    if policy.resolve() != destination.resolve():
        shutil.copy2(policy, destination)
    return folder


def _validation_metrics(episodes, folder: Path, n_jobs: int) -> dict:
    rows = episodes.play(str(folder), cpu_budget=False, n_jobs=n_jobs)
    costs = [int(row["J_policy_cents"]) for row in rows]
    result = episodes.rss(costs)
    return {
        "rss": result["rss"],
        "rss_all": result["rss_all"],
        "cost_usd": float(np.mean(costs) / 100.0),
        "fallback_weeks": int(sum(row["fallback_weeks"] for row in rows)),
    }


def _append_history(path: Path, record: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: record.get(key) for key in HISTORY_FIELDS})


def _save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _save_checkpoint(path, model, optimizer, update, timesteps, best_rss, best_update, settings) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "timesteps": timesteps,
            "best_rss": best_rss,
            "best_update": best_update,
            "settings": settings,
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def _explained_variance(prediction: np.ndarray, target: np.ndarray) -> float:
    variance = float(np.var(target))
    return float("nan") if variance == 0.0 else 1.0 - float(np.var(target - prediction)) / variance


def _training_root(base: int, block: int, forbidden: int) -> int:
    root = int(np.random.SeedSequence([base, block]).generate_state(1, dtype=np.uint32)[0])
    if root == 0 or root == forbidden:
        root = (root + 1) % (2**32)
    return root or 1


def main(
    task: str = "tiny",
    updates: int = 100,
    n_envs: int = 8,
    rollout_steps: int | None = None,
    train_scenarios: int = 512,
    refresh_every: int = 25,
    validation_episodes: int = 100,
    entropy: int = 20261005,
    validation_entropy: int = 20261006,
    regime: str = "standard",
    ppo_epochs: int = 6,
    minibatch_size: int = 64,
    learning_rate: float = 3e-4,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    clip_range: float = 0.15,
    value_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.002,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
    eval_every: int = 5,
    checkpoint_every: int = 5,
    patience: int = 30,
    quick_validation: bool = False,
    validation_jobs: int = -1,
    torch_threads: int = 1,
    device: str = "auto",
    seed: int = 0,
    evaluate_dev: bool = False,
    dev_episodes: int = 20,
    install: bool = False,
    resume: str | None = None,
    out: str | None = None,
    initialize_only: bool = False,
) -> None:
    """Train PPO and save a TorchScript policy consumed directly by ``agent.py``.

    Args:
        task: tiny, small or full.
        updates: PPO rollout/update cycles.
        n_envs: parallel simulator processes.
        rollout_steps: weeks per rollout (default: one task horizon).
        train_scenarios: cached non-dev scenarios sampled throughout training.
        refresh_every: switch to a deterministic fresh training root every N updates; 0 disables.
        validation_episodes: exact held-out scenarios used for model selection.
        entropy: nonzero scenario root used only for training.
        validation_entropy: distinct nonzero root used only for validation.
        regime: information regime; standard is scored.
        ppo_epochs: optimizer passes over each rollout.
        minibatch_size: transitions per CUDA/CPU optimizer minibatch.
        learning_rate: initial Adam learning rate; linearly annealed.
        gamma: reward discount; 1 matches undiscounted benchmark cost.
        gae_lambda: generalized-advantage trace parameter.
        clip_range: PPO likelihood-ratio clip.
        value_clip: clipped value-function update range.
        value_coef: value loss coefficient.
        entropy_coef: exploration entropy coefficient.
        max_grad_norm: gradient clipping norm.
        target_kl: stop an update epoch when approximate KL exceeds this value.
        eval_every: exact validation interval in updates.
        checkpoint_every: resumable checkpoint interval in updates.
        patience: stop after this many updates without validation improvement; 0 disables.
        quick_validation: rough smoke-test references, never leaderboard-quality numbers.
        validation_jobs: process count for exact RSS validation.
        torch_threads: CPU Torch threads; keep one beside environment workers.
        device: auto, cpu, cuda, or cuda:N for PPO optimization.
        seed: NumPy and Torch training seed.
        evaluate_dev: evaluate root 0 once at the end; never used for selection.
        dev_episodes: root-0 episodes in that final report.
        install: atomically replace agents/ppo/policy.pt with the best checkpoint.
        resume: existing run folder or checkpoint.pt.
        out: run folder (default: outputs/ppo/<timestamp>).
        initialize_only: write a zero-residual baseline policy and exit.

    """
    if initialize_only:
        path = _initialise_policy(INSTALLED_POLICY, seed)
        print(f"initialized {path}")
        return
    resume_path = None if resume is None else Path(resume)
    if resume_path is not None and resume_path.is_dir():
        resume_path = resume_path / "checkpoint.pt"
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved = resume_checkpoint["settings"]
        task = saved["task"]
        n_envs = int(saved["n_envs"])
        rollout_steps = int(saved["rollout_steps_resolved"])
        train_scenarios = int(saved["train_scenarios"])
        refresh_every = int(saved.get("refresh_every", 0))
        validation_episodes = int(saved["validation_episodes"])
        entropy = int(saved["entropy"])
        validation_entropy = int(saved["validation_entropy"])
        regime = saved["regime"]
        ppo_epochs = int(saved["ppo_epochs"])
        minibatch_size = int(saved["minibatch_size"])
        learning_rate = float(saved["learning_rate"])
        gamma = float(saved["gamma"])
        gae_lambda = float(saved["gae_lambda"])
        clip_range = float(saved["clip_range"])
        value_clip = float(saved["value_clip"])
        value_coef = float(saved["value_coef"])
        entropy_coef = float(saved["entropy_coef"])
        max_grad_norm = float(saved["max_grad_norm"])
        target_kl = float(saved["target_kl"])
        eval_every = int(saved["eval_every"])
        checkpoint_every = int(saved["checkpoint_every"])
        patience = int(saved["patience"])
        quick_validation = bool(saved["quick_validation"])
        seed = int(saved["seed"])
    if task not in {"tiny", "small", "full"}:
        raise ValueError("task must be tiny, small or full")
    if entropy == 0 or validation_entropy == 0:
        raise ValueError("entropy 0 is reserved for final dev confirmation")
    if entropy == validation_entropy:
        raise ValueError("training and validation entropy roots must be different")
    if updates < 1 or n_envs < 1 or train_scenarios < 1 or validation_episodes < 1:
        raise ValueError("updates, n_envs, train_scenarios and validation_episodes must be positive")
    if refresh_every < 0:
        raise ValueError("refresh_every must be nonnegative")
    if not 0.0 < gamma <= 1.0 or not 0.0 < gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in (0, 1]")

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    chosen_device = torch.device(resolved_device)
    if chosen_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {chosen_device} requested but CUDA is unavailable")
    torch.set_num_threads(torch_threads)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    horizon = int(_agent_config(task, regime)["T"])
    rollout_steps = horizon if rollout_steps is None else int(rollout_steps)
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")

    run = (
        resume_path.parent
        if resume_path is not None
        else Path(out or ROOT / "outputs" / "ppo" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    )
    run.mkdir(parents=True, exist_ok=True)
    settings = {
        key: value
        for key, value in locals().copy().items()
        if key
        not in {
            "chosen_device",
            "resume_path",
            "run",
        }
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    settings["device_resolved"] = str(chosen_device)
    settings["rollout_steps_resolved"] = rollout_steps
    _save_json(run / "settings.json", settings)

    model = RecurrentResidualPolicy().to(chosen_device)
    installed = torch.jit.load(str(INSTALLED_POLICY), map_location="cpu")
    model.load_state_dict(installed.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    start_update = 0
    timesteps = 0
    best_rss = -math.inf
    best_update = 0
    if resume_path is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["update"])
        timesteps = int(checkpoint["timesteps"])
        best_rss = float(checkpoint["best_rss"])
        best_update = int(checkpoint["best_update"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(f"resuming {run} after update {start_update}")

    print(
        f"{task}: {updates} PPO updates, {n_envs} envs x {rollout_steps} weeks; "
        f"training root seed {entropy}, held-out root {validation_entropy}; device {chosen_device}"
    )
    if quick_validation:
        print("quick validation is only a smoke test; its RSS is not leaderboard quality")
    print("dev root 0 is excluded from training and checkpoint selection")

    validation = scoring.episode_set(
        task,
        validation_episodes,
        quick=quick_validation,
        entropy=validation_entropy,
        n_jobs=validation_jobs,
        verbose=True,
    )
    config = _agent_config(task, regime)
    builders = [FeatureBuilder(config) for _ in range(n_envs)]

    def make_vector(root):
        env_fns = [_make_env(task, train_scenarios, root, regime) for _ in range(n_envs)]
        return SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    current_block = start_update // refresh_every if refresh_every else 0
    current_entropy = _training_root(entropy, current_block, validation_entropy) if refresh_every else entropy
    vector = make_vector(current_entropy)
    vector.seed(seed + current_block * n_envs)
    observations = vector.reset()
    current_features, current_base, current_active = _encode_batch(builders, observations)
    hidden = torch.zeros((n_envs, builders[0].n_slots, HIDDEN_SIZE), dtype=torch.float32, device=chosen_device)
    episode_returns = np.zeros(n_envs, dtype=np.float64)
    finished_returns = []
    episodes_finished = 0
    started = time.monotonic()
    history = run / "history.csv"
    latest_policy = run / "latest" / "policy.pt"
    latest_agent = run / "latest"
    best_agent = run / "best"

    try:
        for update_index in range(start_update, updates):
            update = update_index + 1
            desired_block = update_index // refresh_every if refresh_every else 0
            if desired_block != current_block:
                vector.close()
                current_block = desired_block
                current_entropy = _training_root(entropy, current_block, validation_entropy)
                vector = make_vector(current_entropy)
                vector.seed(seed + current_block * n_envs)
                observations = vector.reset()
                for builder in builders:
                    builder.reset()
                current_features, current_base, current_active = _encode_batch(builders, observations)
                hidden.zero_()
                episode_returns.fill(0.0)
                print(f"training scenarios refreshed from root {current_entropy}")
            progress = update_index / max(updates, 1)
            lr_now = learning_rate * (1.0 - progress)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            shape = (rollout_steps, n_envs)
            feature_buffer = np.empty(shape + current_features.shape[1:], dtype=np.float32)
            base_buffer = np.empty(shape + current_base.shape[1:], dtype=np.float32)
            active_buffer = np.empty(shape + current_active.shape[1:], dtype=np.float32)
            latent_buffer = np.empty(shape + current_base.shape[1:], dtype=np.float32)
            hidden_buffer = np.empty(shape + tuple(hidden.shape[1:]), dtype=np.float32)
            log_prob_buffer = np.empty(shape, dtype=np.float32)
            value_buffer = np.empty(shape, dtype=np.float32)
            reward_buffer = np.empty(shape, dtype=np.float32)
            done_buffer = np.empty(shape, dtype=np.float32)

            model.eval()
            for step in range(rollout_steps):
                feature_buffer[step] = current_features
                base_buffer[step] = current_base
                active_buffer[step] = current_active
                hidden_buffer[step] = hidden.detach().cpu().numpy()
                x = torch.as_tensor(current_features, device=chosen_device)
                base_t = torch.as_tensor(current_base, device=chosen_device)
                active_t = torch.as_tensor(current_active, device=chosen_device)
                with torch.no_grad():
                    mean, log_std, value, next_hidden = model(x, hidden)
                    distribution = torch.distributions.Normal(mean, torch.exp(log_std))
                    latent = distribution.sample()
                    fractions = _fraction_tensor(base_t, active_t, latent)
                    log_probability = _log_prob(distribution, latent, active_t)
                next_observations, rewards, dones, _infos = vector.step(fractions.cpu().numpy())
                latent_buffer[step] = latent.cpu().numpy()
                log_prob_buffer[step] = log_probability.cpu().numpy()
                value_buffer[step] = value.cpu().numpy()
                reward_buffer[step] = rewards
                done_buffer[step] = dones.astype(np.float32)
                episode_returns += rewards
                timesteps += n_envs
                for i, done in enumerate(dones):
                    if done:
                        finished_returns.append(float(episode_returns[i]))
                        episode_returns[i] = 0.0
                        episodes_finished += 1
                        builders[i].reset()
                        next_hidden[i].zero_()
                observations = next_observations
                current_features, current_base, current_active = _encode_batch(builders, observations)
                hidden = next_hidden.detach()

            with torch.no_grad():
                _, _, bootstrap, _ = model(torch.as_tensor(current_features, device=chosen_device), hidden)
            advantages = np.zeros_like(reward_buffer)
            last_advantage = np.zeros(n_envs, dtype=np.float32)
            next_value = bootstrap.cpu().numpy()
            for step in range(rollout_steps - 1, -1, -1):
                not_done = 1.0 - done_buffer[step]
                delta = reward_buffer[step] + gamma * next_value * not_done - value_buffer[step]
                last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
                advantages[step] = last_advantage
                next_value = value_buffer[step]
            returns = advantages + value_buffer
            flat_advantage = advantages.reshape(-1)
            flat_advantage = (flat_advantage - flat_advantage.mean()) / (flat_advantage.std() + 1e-8)

            total = rollout_steps * n_envs
            indices = np.arange(total)
            flat_features = feature_buffer.reshape((total,) + feature_buffer.shape[2:])
            flat_base = base_buffer.reshape((total,) + base_buffer.shape[2:])
            flat_active = active_buffer.reshape((total,) + active_buffer.shape[2:])
            flat_latent = latent_buffer.reshape((total,) + latent_buffer.shape[2:])
            flat_hidden = hidden_buffer.reshape((total,) + hidden_buffer.shape[2:])
            flat_old_log_prob = log_prob_buffer.reshape(total)
            flat_old_value = value_buffer.reshape(total)
            flat_return = returns.reshape(total)
            losses = []
            stop_for_kl = False
            model.train()
            for _epoch in range(ppo_epochs):
                np.random.shuffle(indices)
                for first in range(0, total, minibatch_size):
                    batch = indices[first : first + minibatch_size]
                    x = torch.as_tensor(flat_features[batch], device=chosen_device)
                    old_hidden = torch.as_tensor(flat_hidden[batch], device=chosen_device)
                    active_t = torch.as_tensor(flat_active[batch], device=chosen_device)
                    latent_t = torch.as_tensor(flat_latent[batch], device=chosen_device)
                    old_log_prob = torch.as_tensor(flat_old_log_prob[batch], device=chosen_device)
                    old_value = torch.as_tensor(flat_old_value[batch], device=chosen_device)
                    return_t = torch.as_tensor(flat_return[batch], device=chosen_device)
                    advantage_t = torch.as_tensor(flat_advantage[batch], device=chosen_device)
                    mean, log_std, value, _ = model(x, old_hidden)
                    distribution = torch.distributions.Normal(mean, torch.exp(log_std))
                    log_probability = _log_prob(distribution, latent_t, active_t)
                    log_ratio = log_probability - old_log_prob
                    ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                    unclipped = ratio * advantage_t
                    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_t
                    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                    value_delta = torch.clamp(value - old_value, -value_clip, value_clip)
                    value_clipped = old_value + value_delta
                    value_loss = 0.5 * torch.mean(
                        torch.maximum((value - return_t) ** 2, (value_clipped - return_t) ** 2)
                    )
                    entropy_bonus = torch.mean(_entropy(distribution, active_t))
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
                        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
                    losses.append(
                        (
                            float(policy_loss.detach()),
                            float(value_loss.detach()),
                            float(entropy_bonus.detach()),
                            float(approx_kl.detach()),
                            float(clip_fraction.detach()),
                        )
                    )
                    if target_kl > 0.0 and float(approx_kl) > target_kl:
                        stop_for_kl = True
                        break
                if stop_for_kl:
                    break

            metric = np.mean(np.asarray(losses), axis=0)
            validation_metrics = None
            should_validate = update == 1 or update % eval_every == 0 or update == updates
            if should_validate:
                _export(model, latest_policy)
                _write_agent(latest_agent, latest_policy)
                validation_metrics = _validation_metrics(validation, latest_agent, validation_jobs)
                rss = validation_metrics["rss"]
                fitness = -validation_metrics["cost_usd"] if rss is None else float(rss)
                if fitness > best_rss:
                    best_rss = fitness
                    best_update = update
                    _write_agent(best_agent, latest_policy)
                    print(f"update {update}: new best held-out RSS {rss}; cost ${validation_metrics['cost_usd']:,.0f}")

            recent_returns = finished_returns[-max(1, n_envs * 10) :]
            record = {
                "update": update,
                "train_entropy": current_entropy,
                "timesteps": timesteps,
                "episodes": episodes_finished,
                "mean_episode_return": float(np.mean(recent_returns)) if recent_returns else None,
                "policy_loss": float(metric[0]),
                "value_loss": float(metric[1]),
                "entropy": float(metric[2]),
                "approx_kl": float(metric[3]),
                "clip_fraction": float(metric[4]),
                "explained_variance": _explained_variance(value_buffer, returns),
                "learning_rate": lr_now,
                "validation_rss": None if validation_metrics is None else validation_metrics["rss"],
                "validation_cost_usd": None if validation_metrics is None else validation_metrics["cost_usd"],
                "best_validation_rss": best_rss,
                "elapsed_seconds": time.monotonic() - started,
            }
            _append_history(history, record)
            print(
                f"update {update}/{updates} steps={timesteps} return={record['mean_episode_return']} "
                f"pi={record['policy_loss']:.4f} vf={record['value_loss']:.4f} kl={record['approx_kl']:.5f}"
            )
            should_stop = patience > 0 and best_update > 0 and update - best_update >= patience
            if update % checkpoint_every == 0 or update == updates or should_stop:
                _save_checkpoint(
                    run / "checkpoint.pt", model, optimizer, update, timesteps, best_rss, best_update, settings
                )
            if should_stop:
                print(f"early stopping: held-out RSS has not improved for {patience} updates")
                break
    finally:
        vector.close()

    if not (best_agent / "policy.pt").is_file():
        _export(model, latest_policy)
        _write_agent(best_agent, latest_policy)
    summary = {
        "settings": settings,
        "timesteps": timesteps,
        "episodes": episodes_finished,
        "best_update": best_update,
        "best_validation_rss": best_rss,
        "best_agent": str(best_agent),
        "installed": bool(install),
    }
    if evaluate_dev:
        dev = scoring.episode_set(task, dev_episodes, quick=False, entropy=0, n_jobs=validation_jobs, verbose=True)
        dev_score = dev.score(str(best_agent), name="ppo", cpu_budget=False, n_jobs=validation_jobs)
        summary["dev"] = scoring.as_dict(dev_score)
        print("dev was evaluated once after model selection; it did not influence training")
    if install:
        temporary = INSTALLED_POLICY.with_suffix(".pt.tmp")
        shutil.copy2(best_agent / "policy.pt", temporary)
        temporary.replace(INSTALLED_POLICY)
        print(f"installed best update {best_update} in {INSTALLED_POLICY}")
    _save_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"next: uv run sbf check {best_agent} --task={task}")


if __name__ == "__main__":
    fire.Fire(main)
