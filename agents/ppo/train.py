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
import os
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
from shockbench_flow_gym.wrappers import ScaleReward, ScenarioPool, SlimInfo, draw_scenarios
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sbf_starter import ROOT, env_id, scoring


AGENT_DIR = Path(__file__).resolve().parent
AGENT_FILE = AGENT_DIR / "agent.py"
INSTALLED_POLICY = AGENT_DIR / "policy.pt"
FEATURE_VERSION = 3
FEATURE_DIM = 168
HIDDEN_SIZE = 64
RESIDUAL_SCALE = 1.0
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
    "validation_fallback_weeks",
    "validation_rss_all",
    "validation_rss_by_stratum",
    "validation_checks_without_improvement",
    "shortage_usd",
    "shed_usd",
    "anchor_loss",
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
        self.critic_encoder = torch.nn.Sequential(
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
        self.release_head = torch.nn.Linear(width // 2, 3)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(2 * width + 2 * hidden_size, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 1),
        )
        # Zero residual means exact baseline inference before any training.
        torch.nn.init.zeros_(self.actor_mean.weight)
        torch.nn.init.zeros_(self.actor_mean.bias)
        torch.nn.init.zeros_(self.actor_log_std.weight)
        torch.nn.init.constant_(self.actor_log_std.bias, -1.7)
        torch.nn.init.zeros_(self.release_head.weight)
        with torch.no_grad():
            self.release_head.bias.copy_(torch.tensor([3.0, -3.0, -3.0]))
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
        critic_local = self.critic_encoder(features)
        critic_pool = torch.cat((torch.mean(critic_local, dim=1), torch.amax(critic_local, dim=1)), dim=-1)
        value = self.value(torch.cat((critic_pool, hidden_pool.detach()), dim=-1)).squeeze(-1)
        return mean, log_std, value, next_hidden, self.release_head(actor)


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


def _load_weights(model, path: Path, *, reset_actor: bool = False) -> None:
    """Expand v1 feature columns explicitly; never silently resume old optimizer state."""
    installed = torch.jit.load(str(path), map_location="cpu")
    source = installed.state_dict()
    if int(installed.feature_version) == 1:
        expanded = torch.zeros_like(model.encoder[0].weight)
        expanded[:, :12] = source["encoder.0.weight"][:, :12]
        for block in range(3):
            expanded[:, 24 + block * 48 : 24 + block * 48 + 28] = source["encoder.0.weight"][
                :, 12 + block * 28 : 12 + (block + 1) * 28
            ]
        source["encoder.0.weight"] = expanded
        reset_actor = True  # v1 means had different action semantics.
    elif int(installed.feature_version) not in {2, FEATURE_VERSION}:
        raise ValueError("unsupported policy schema")
    for key, value in model.state_dict().items():
        if key not in source:
            source[key] = value
    model.load_state_dict(source)
    if reset_actor:
        torch.nn.init.zeros_(model.actor_mean.weight)
        torch.nn.init.zeros_(model.actor_mean.bias)
        torch.nn.init.zeros_(model.actor_log_std.weight)
        torch.nn.init.constant_(model.actor_log_std.bias, -1.7)


# Let a clean checkout run the trainer to create its first deployable baseline.
# Normal evaluation still requires the committed policy file to exist.
if not INSTALLED_POLICY.is_file():
    _initialise_policy()

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
_previous_feature_flag = os.environ.get("SBF_PPO_FEATURES_ONLY")
os.environ["SBF_PPO_FEATURES_ONLY"] = "1"
try:
    deployment = importlib.import_module("agent")
finally:
    if _previous_feature_flag is None:
        os.environ.pop("SBF_PPO_FEATURES_ONLY", None)
    else:
        os.environ["SBF_PPO_FEATURES_ONLY"] = _previous_feature_flag
FeatureBuilder = deployment.FeatureBuilder


class DispatchAndRelease(gym.ActionWrapper):
    """Fractions plus categorical release modes; matches deployed action conversion."""

    def __init__(self, env):
        super().__init__(env)
        inst, layout = env.unwrapped.instance, env.unwrapped.layout
        self.n_dispatch = layout.n_slots
        self.n_override = layout.n_override_slots
        self.capacity = np.asarray(
            [inst.edges[e].u0 for e, _k, _lane in inst.action_slots]
            + [inst.edges[e].u0 for _cp, _k, e, _lane in inst.override_slots]
        )
        self.n_pairs = len(layout.pairs)
        self.action_space = gym.spaces.Box(0.0, 2.0, (len(self.capacity) + self.n_pairs,), dtype=np.float32)
        self.mask = np.ones_like(self.capacity)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.mask = np.concatenate((obs["action_mask"], obs["override_mask"]))
        return obs, info

    def step(self, action):
        result = self.env.step(self.action(action))
        self.mask = np.concatenate((result[0]["action_mask"], result[0]["override_mask"]))
        return result

    def action(self, action):
        flows = np.clip(action[: len(self.capacity)], 0.0, 1.0) * self.capacity * self.mask
        return {
            "flows": flows[: self.n_dispatch],
            "override_qty": flows[self.n_dispatch :],
            "release_mode": np.rint(action[len(self.capacity) :]).astype(np.int64),
        }


class CounterfactualReward(gym.Wrapper):
    """Subtract a fixed controller's weekly cost on the identical scenario.

    This is a training control variate. Its episode total is J_reference-J,
    so it removes exogenous cost variation without changing the optimal policy.
    Nothing from the reference rollout is added to the agent's observation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reference_cache = {}
        self.reference_rewards = None
        self.reference_week = 0

    def reset(self, **kwargs):
        from shockbench_flow.omega.container import load_omega
        from shockbench_flow_gym import agent_config_from_reset
        from shockbench_flow_gym.env import ShockBenchFlowEnv

        obs, info = self.env.reset(**kwargs)
        index = int(info["pool_index"])
        if index not in self.reference_cache:
            base = self.env.unwrapped
            reference = ShockBenchFlowEnv(
                base.instance,
                omega_source=load_omega(self.env.paths[index]),
                regime=base.regime,
                dense_reward=base.dense_reward,
            )
            rewards = []
            try:
                reference_obs, reference_info = reference.reset(seed=0)
                features = FeatureBuilder(agent_config_from_reset(reference, reference_obs, reference_info))
                done = False
                while not done:
                    week = int(reference_obs["week"].item())
                    flows = features.nominal + features.reserve * (features.capacity - features.nominal)
                    flows = np.where(week <= features.last_useful, flows, 0.0)
                    for slot, chokepoints in enumerate(features.route_chokepoints):
                        for cp in chokepoints:
                            position = features.chokepoint_position[cp]
                            if reference_obs["graph_now.open.observed"][position]:
                                flows[slot] *= max(float(reference_obs["graph_now.open"][position]), 0.0) ** float(
                                    features.parameters["closure_power"]
                                )
                    current_capacity = reference_obs["graph_now.u"][features.slot_edge]
                    observed_capacity = reference_obs["graph_now.u.observed"][features.slot_edge]
                    flows = np.where(observed_capacity & (current_capacity <= EPS), 0.0, flows)
                    reference_obs, reward, terminated, truncated, _info = reference.step(
                        {"flows": flows * reference_obs["action_mask"]}
                    )
                    rewards.append(reward)
                    done = terminated or truncated
            finally:
                reference.close()
            self.reference_cache[index] = np.asarray(rewards, dtype=np.float64)
        self.reference_rewards = self.reference_cache[index]
        self.reference_week = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reference_reward = float(self.reference_rewards[self.reference_week])
        self.reference_week += 1
        return obs, reward - reference_reward, terminated, truncated, info


def _make_env(
    task: str, scenarios: int, entropy: int, regime: str, scenario_cache, dense_reward, counterfactual_reward=False
):
    def thunk():
        env = ScenarioPool(
            gym.make(env_id(task), regime=regime, dense_reward=dense_reward),
            scenarios,
            entropy,
            cache_dir=scenario_cache,
        )
        if counterfactual_reward:
            env = CounterfactualReward(env)
        env = ScaleReward(DispatchAndRelease(env))
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
    releases = []
    for i, builder in enumerate(builders):
        observation = {key: value[i] for key, value in observations.items()}
        features.append(builder.encode(observation))
        base.append(builder.last_base)
        active.append(builder.last_active)
        releases.append(builder.last_release_active.copy())
    return (
        np.stack(features).astype(np.float32),
        np.stack(base).astype(np.float32),
        np.stack(active).astype(np.float32),
        np.stack(releases).astype(np.float32),
    )


def _fraction_tensor(base: torch.Tensor, active: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    return torch.clamp(base + RESIDUAL_SCALE * torch.tanh(latent), 0.0, 1.0) * active


def _log_prob(distribution, latent: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    # PPO optimizes the sampled latent policy. The fixed environment mapping
    # may clip many latents to one action; no inverse/Jacobian is assumed.
    return torch.sum(distribution.log_prob(latent) * active, dim=-1)


def _entropy(distribution, active: torch.Tensor) -> torch.Tensor:
    count = torch.clamp(torch.sum(active, dim=-1), min=1.0)
    return torch.sum(distribution.entropy() * active, dim=-1) / count


def _release_distribution(route_logits, weights):
    return torch.distributions.Categorical(logits=torch.einsum("pr,brc->bpc", weights, route_logits))


def _conditional_active(active, modes, weights, n_dispatch):
    if active.shape[-1] == n_dispatch:
        return active
    override_pair = torch.argmax(weights[:, n_dispatch:], dim=0)
    conditional = (modes[:, override_pair] == 1).float()
    return torch.cat((active[:, :n_dispatch], active[:, n_dispatch:] * conditional), dim=-1)


def _write_agent(folder: Path, policy: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(AGENT_FILE, folder / "agent.py")
    destination = folder / "policy.pt"
    if policy.resolve() != destination.resolve():
        shutil.copy2(policy, destination)
    return folder


def _validation_metrics(episodes, folder: Path, n_jobs: int, batch_size: int = 32) -> dict:
    from sbf_starter.accelerated import play_batched

    requested = os.environ.get("SBF_EVAL_DEVICE", "auto")
    cuda_inference = requested.startswith("cuda") or (requested == "auto" and torch.cuda.is_available())
    use_batch = batch_size > 1 and (cuda_inference or n_jobs == 1)
    rows = play_batched(episodes, str(folder), batch_size=batch_size) if use_batch else None
    if rows is None:
        rows = episodes.play(str(folder), cpu_budget=False, n_jobs=n_jobs)
    costs = [int(row["J_policy_cents"]) for row in rows]
    result = episodes.rss(costs)
    if any(row["fallback_weeks"] for row in rows):
        raise RuntimeError("validation used fallback actions; inspect the agent before selecting weights")
    return {
        "rss": result["rss"],
        "rss_all": result["rss_all"],
        "rss_by_stratum": result["rss_by_stratum"],
        "pooled": result["pooled"],
        "cost_usd": float(np.mean(costs) / 100.0),
        "fallback_weeks": int(sum(row["fallback_weeks"] for row in rows)),
    }


def _append_history(path: Path, record: dict) -> None:
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            previous = list(reader) if reader.fieldnames != list(HISTORY_FIELDS) else None
        if previous is not None:
            temporary = path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerows({key: row.get(key) for key in HISTORY_FIELDS} for row in previous)
            temporary.replace(path)
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


def _save_checkpoint(
    path,
    model,
    optimizer,
    update,
    timesteps,
    best_rss,
    best_update,
    settings,
    stale_checks=0,
    episodes=0,
    reached_checks=0,
    patience_best=None,
    training_rng=None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 3,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "timesteps": timesteps,
            "best_rss": best_rss,
            "best_update": best_update,
            "stale_checks": stale_checks,
            "episodes": episodes,
            "reached_checks": reached_checks,
            "patience_best": best_rss if patience_best is None else patience_best,
            "training_rng": None if training_rng is None else training_rng.bit_generator.state,
            "settings": settings,
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def _unroll(model, features, hidden_states, dones, descriptors, burn_in, device):
    """Recompute recent memory, then backpropagate through contiguous weeks."""
    initial = []
    for env, start, _end in descriptors:
        first = max(0, start - burn_in)
        hidden = torch.as_tensor(hidden_states[first, env : env + 1], device=device)
        with torch.no_grad():
            for step in range(first, start):
                if step > first:
                    hidden = hidden * (1.0 - float(dones[step - 1, env]))
                x = torch.as_tensor(features[step, env : env + 1], device=device)
                _, _, _, hidden, _ = model(x, hidden)
        if start > first:
            hidden = hidden * (1.0 - float(dones[start - 1, env]))
        initial.append(hidden)
    hidden = torch.cat(initial).detach()
    means, stds, values, logits, time_indices, env_indices = [], [], [], [], [], []
    max_length = max(end - start for _env, start, end in descriptors)
    for offset in range(max_length):
        valid = [j for j, (_env, start, end) in enumerate(descriptors) if start + offset < end]
        envs = [descriptors[j][0] for j in valid]
        steps = [descriptors[j][1] + offset for j in valid]
        h = hidden[valid]
        if offset:
            h = h * torch.as_tensor(1.0 - dones[np.asarray(steps) - 1, envs], device=device)[:, None, None]
        x = torch.as_tensor(features[steps, envs], device=device)
        mean, log_std, value, next_hidden, release_logits = model(x, h)
        # Out-of-place assignment preserves autograd across the sequence.
        hidden = hidden.index_copy(0, torch.as_tensor(valid, device=device), next_hidden)
        means.append(mean)
        stds.append(log_std)
        values.append(value)
        logits.append(release_logits)
        time_indices.extend(steps)
        env_indices.extend(envs)
    return (
        torch.cat(means),
        torch.cat(stds),
        torch.cat(values),
        torch.cat(logits),
        np.asarray(time_indices),
        np.asarray(env_indices),
    )


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
    updates: int = 500,
    n_envs: int = 8,
    rollout_steps: int | None = None,
    train_scenarios: int = 256,
    scenario_cache: str | None = None,
    dense_reward: bool = True,
    counterfactual_reward: bool = True,
    refresh_every: int = 100,
    validation_episodes: int = 100,
    entropy: int = 20261005,
    validation_entropy: int = 20261006,
    regime: str = "standard",
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    sequence_length: int = 16,
    burn_in: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 1.0,
    gae_lambda: float = 0.99,
    clip_range: float = 0.15,
    value_clip: float = 0.0,
    value_coef: float = 0.5,
    entropy_coef: float = 0.002,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.03,
    eval_every: int = 5,
    checkpoint_every: int = 5,
    patience: int | None = None,
    min_updates: int = 200,
    min_delta: float = 0.0001,
    target_rss: float = 0.85,
    target_checks: int = 3,
    anchor_coef: float = 0.002,
    quick_validation: bool = False,
    validation_jobs: int = 4,
    torch_threads: int = 1,
    device: str = "auto",
    seed: int = 0,
    evaluate_dev: bool = False,
    dev_episodes: int = 20,
    install: bool = False,
    resume: str | None = None,
    out: str | None = None,
    initialize_only: bool = False,
    migrate_only: bool = False,
    initial_policy: str | None = None,
) -> None:
    """Train PPO and save a TorchScript policy consumed directly by ``agent.py``.

    Args:
        task: tiny, small or full.
        updates: PPO rollout/update cycles.
        n_envs: parallel simulator processes.
        rollout_steps: weeks per rollout (default: one task horizon).
        train_scenarios: cached non-dev scenarios sampled throughout training.
        scenario_cache: optional existing scenario cache; references use SBF_CACHE_DIR.
        dense_reward: stock/pipeline potential shaping with unchanged undiscounted episode cost.
        counterfactual_reward: subtract fixed-controller costs on the same scenario to reduce reward variance.
        refresh_every: switch to a deterministic fresh training root every N updates; 0 disables.
        validation_episodes: exact held-out scenarios used for model selection.
        entropy: nonzero scenario root used only for training.
        validation_entropy: distinct nonzero root used only for validation.
        regime: information regime; standard is scored.
        ppo_epochs: optimizer passes over each rollout.
        minibatch_size: approximate weeks per sequence minibatch.
        sequence_length: contiguous weeks used for recurrent backpropagation.
        burn_in: previous weeks recomputed without gradients before each sequence.
        learning_rate: initial Adam learning rate; linearly annealed.
        gamma: reward discount; 1 matches undiscounted benchmark cost.
        gae_lambda: generalized-advantage trace parameter.
        clip_range: PPO likelihood-ratio clip.
        value_clip: optional value update range; 0 disables reward-scale-dependent clipping.
        value_coef: value loss coefficient.
        entropy_coef: exploration entropy coefficient.
        max_grad_norm: gradient clipping norm.
        target_kl: stop an update epoch when approximate KL exceeds this value.
        eval_every: exact validation interval in updates.
        checkpoint_every: resumable checkpoint interval in updates.
        patience: validation checks without improvement; default 0, overrides saved value on resume.
        min_updates: minimum updates before patience or target stopping.
        min_delta: RSS improvement required to reset patience.
        target_rss: desired held-out RSS, never a promised score.
        target_checks: consecutive checks meeting target before stopping; 0 disables target stopping.
        anchor_coef: initial baseline regularization, annealed to zero.
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
        migrate_only: expand installed v1 features and reset its incompatible actor; saves a backup.
        initial_policy: optional pretrained TorchScript checkpoint; ignored on resume.

    """
    if initialize_only:
        path = _initialise_policy(INSTALLED_POLICY, seed)
        print(f"initialized {path}")
        return
    if migrate_only:
        backup = ROOT / "outputs" / "ppo" / "migration_backup"
        backup.mkdir(parents=True, exist_ok=True)
        backup_policy = backup / "policy.pt"
        if not backup_policy.exists():
            shutil.copy2(INSTALLED_POLICY, backup_policy)
        model = RecurrentResidualPolicy()
        _load_weights(model, INSTALLED_POLICY)
        _export(model, INSTALLED_POLICY)
        print(f"migrated {INSTALLED_POLICY}; previous weights at {backup_policy}")
        return
    resume_path = None if resume is None else Path(resume)
    if resume_path is not None and resume_path.is_dir():
        resume_path = resume_path / "checkpoint.pt"
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("schema_version") != 3:
            raise ValueError(
                "old optimizer checkpoints cannot resume under v3 actions; use --migrate_only, then start a new run"
            )
        saved = resume_checkpoint["settings"]
        task = saved["task"]
        n_envs = int(saved["n_envs"])
        rollout_steps = int(saved["rollout_steps_resolved"])
        train_scenarios = int(saved["train_scenarios"])
        scenario_cache = saved.get("scenario_cache")
        dense_reward = bool(saved.get("dense_reward", True))
        counterfactual_reward = bool(saved.get("counterfactual_reward", False))
        refresh_every = int(saved.get("refresh_every", 0))
        validation_episodes = int(saved["validation_episodes"])
        entropy = int(saved["entropy"])
        validation_entropy = int(saved["validation_entropy"])
        regime = saved["regime"]
        ppo_epochs = int(saved["ppo_epochs"])
        minibatch_size = int(saved["minibatch_size"])
        sequence_length = int(saved["sequence_length"])
        burn_in = int(saved["burn_in"])
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
        if patience is None:
            patience = int(saved["patience"])
        quick_validation = bool(saved["quick_validation"])
        seed = int(saved["seed"])
        min_updates = int(saved["min_updates"])
        min_delta = float(saved["min_delta"])
        target_rss = float(saved["target_rss"])
        target_checks = int(saved["target_checks"])
        anchor_coef = float(saved["anchor_coef"])
    patience = 0 if patience is None else int(patience)
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
    if sequence_length < 2 or burn_in < 0 or minibatch_size < sequence_length:
        raise ValueError("sequence_length >= 2, burn_in >= 0 and minibatch_size >= sequence_length are required")
    if patience < 0 or target_checks < 0 or min_updates < 0 or min_delta < 0:
        raise ValueError("stopping settings must be nonnegative")
    if eval_every < 1 or checkpoint_every < 1 or ppo_epochs < 1:
        raise ValueError("evaluation/checkpoint intervals and ppo_epochs must be positive")
    if regime != "standard":
        raise ValueError("this trainer validates the scored standard information regime; use regime=standard")
    if not 0.0 < gamma <= 1.0 or not 0.0 < gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in (0, 1]")

    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    chosen_device = torch.device(resolved_device)
    if chosen_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {chosen_device} requested but CUDA is unavailable")
    torch.set_num_threads(torch_threads)
    rng = np.random.default_rng(seed)
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
    _load_weights(model, Path(initial_policy) if initial_policy else INSTALLED_POLICY)
    if counterfactual_reward and resume_path is None:
        # A cost critic is not calibrated for relative rewards. Keep the actor,
        # but initialize a fresh critic for the changed value target.
        fresh = RecurrentResidualPolicy()
        model.critic_encoder.load_state_dict(fresh.critic_encoder.state_dict())
        model.value.load_state_dict(fresh.value.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    start_update = 0
    timesteps = 0
    best_rss = -math.inf
    best_update = 0
    stale_checks = 0
    reached_checks = 0
    patience_best = -math.inf
    if resume_path is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["update"])
        timesteps = int(checkpoint["timesteps"])
        best_rss = float(checkpoint["best_rss"])
        best_update = int(checkpoint["best_update"])
        stale_checks = int(checkpoint.get("stale_checks", 0))
        reached_checks = int(checkpoint.get("reached_checks", 0))
        patience_best = float(checkpoint.get("patience_best", best_rss))
        if checkpoint.get("training_rng") is not None:
            rng.bit_generator.state = checkpoint["training_rng"]
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
    builders = [FeatureBuilder(config, queue_control=True) for _ in range(n_envs)]
    release_weights = torch.as_tensor(builders[0].release_weights, device=chosen_device)
    n_dispatch = builders[0].n_dispatch

    def make_vector(root):
        # Populate the pool once in the parent, avoiding worker races and N
        # copies of scenario-generation work on a newly refreshed root.
        probe = gym.make(env_id(task), regime=regime)
        try:
            draw_scenarios(probe.unwrapped.instance, train_scenarios, root, cache_dir=scenario_cache)
        finally:
            probe.close()
        env_fns = [
            _make_env(task, train_scenarios, root, regime, scenario_cache, dense_reward, counterfactual_reward)
            for _ in range(n_envs)
        ]
        return SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

    current_block = start_update // refresh_every if refresh_every else 0
    current_entropy = _training_root(entropy, current_block, validation_entropy) if refresh_every else entropy
    vector = make_vector(current_entropy)
    vector.seed(seed + current_block * n_envs)
    observations = vector.reset()
    current_features, current_base, current_active, current_release_active = _encode_batch(builders, observations)
    hidden = torch.zeros((n_envs, builders[0].n_slots, HIDDEN_SIZE), dtype=torch.float32, device=chosen_device)
    episode_returns = np.zeros(n_envs, dtype=np.float64)
    finished_returns = []
    episodes_finished = 0 if resume_checkpoint is None else int(resume_checkpoint.get("episodes", timesteps // horizon))
    started = time.monotonic()
    history = run / "history.csv"
    latest_policy = run / "latest" / "policy.pt"
    latest_agent = run / "latest"
    best_agent = run / "best"

    # Validate update zero so learning cannot replace a stronger initializer
    # with the first (possibly worse) trained checkpoint.
    if start_update == 0:
        _export(model, latest_policy)
        _write_agent(latest_agent, latest_policy)
        initial_metrics = _validation_metrics(validation, latest_agent, validation_jobs)
        if initial_metrics["rss"] is None:
            raise RuntimeError("held-out RSS is undefined; cannot select a checkpoint")
        best_rss = float(initial_metrics["rss"])
        patience_best = best_rss
        _write_agent(best_agent, latest_policy)
        _save_json(run / "initial_validation.json", initial_metrics)
        print(f"initial held-out RSS {best_rss:.6f}")

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
                current_features, current_base, current_active, current_release_active = _encode_batch(
                    builders, observations
                )
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
            release_buffer = np.empty(shape + current_release_active.shape[1:], dtype=np.int64)
            release_active_buffer = np.empty(shape + current_release_active.shape[1:], dtype=np.float32)
            shortage_usd = shed_usd = 0.0

            model.eval()
            for step in range(rollout_steps):
                feature_buffer[step] = current_features
                base_buffer[step] = current_base
                active_buffer[step] = current_active
                release_active_buffer[step] = current_release_active
                hidden_buffer[step] = hidden.detach().cpu().numpy()
                x = torch.as_tensor(current_features, device=chosen_device)
                base_t = torch.as_tensor(current_base, device=chosen_device)
                active_t = torch.as_tensor(current_active, device=chosen_device)
                with torch.no_grad():
                    mean, log_std, value, next_hidden, release_logits = model(x, hidden)
                    distribution = torch.distributions.Normal(mean, torch.exp(log_std))
                    latent = distribution.sample()
                    fractions = _fraction_tensor(base_t, active_t, latent)
                    release_distribution = _release_distribution(release_logits, release_weights)
                    release_active_t = torch.as_tensor(current_release_active, device=chosen_device)
                    modes = release_distribution.sample() * release_active_t.long()
                    effective_active = _conditional_active(active_t, modes, release_weights, n_dispatch)
                    log_probability = _log_prob(distribution, latent, effective_active) + torch.sum(
                        release_distribution.log_prob(modes) * release_active_t, dim=-1
                    )
                action = torch.cat((fractions, modes.float()), dim=-1)
                next_observations, rewards, dones, _infos = vector.step(action.cpu().numpy())
                release_buffer[step] = modes.cpu().numpy()
                for env_index, done in enumerate(dones):
                    loss_obs = (
                        _infos[env_index].get("terminal_observation", {})
                        if done
                        else {"last_week.cost_components": next_observations["last_week.cost_components"][env_index]}
                    )
                    components = np.asarray(loss_obs.get("last_week.cost_components", np.zeros(8)))
                    shortage_usd += float(components[5])
                    shed_usd += float(components[7])
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
                current_features, current_base, current_active, current_release_active = _encode_batch(
                    builders, observations
                )
                hidden = next_hidden.detach()

            with torch.no_grad():
                _, _, bootstrap, _, _ = model(torch.as_tensor(current_features, device=chosen_device), hidden)
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
            normalized_advantage = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            descriptors = [
                (env, start, min(start + sequence_length, rollout_steps))
                for env in range(n_envs)
                for start in range(0, rollout_steps, sequence_length)
            ]
            sequences_per_batch = max(1, minibatch_size // sequence_length)
            losses = []
            stop_for_kl = False
            model.train()
            for _epoch in range(ppo_epochs):
                order = rng.permutation(len(descriptors))
                for first in range(0, len(order), sequences_per_batch):
                    batch = [descriptors[j] for j in order[first : first + sequences_per_batch]]
                    mean, log_std, value, release_logits, steps, envs = _unroll(
                        model, feature_buffer, hidden_buffer, done_buffer, batch, burn_in, chosen_device
                    )
                    active_t = torch.as_tensor(active_buffer[steps, envs], device=chosen_device)
                    modes = torch.as_tensor(release_buffer[steps, envs], device=chosen_device)
                    release_active_t = torch.as_tensor(release_active_buffer[steps, envs], device=chosen_device)
                    effective_active = _conditional_active(active_t, modes, release_weights, n_dispatch)
                    base_t = torch.as_tensor(base_buffer[steps, envs], device=chosen_device)
                    latent_t = torch.as_tensor(latent_buffer[steps, envs], device=chosen_device)
                    old_log_prob = torch.as_tensor(log_prob_buffer[steps, envs], device=chosen_device)
                    old_value = torch.as_tensor(value_buffer[steps, envs], device=chosen_device)
                    return_t = torch.as_tensor(returns[steps, envs], device=chosen_device)
                    advantage_t = torch.as_tensor(normalized_advantage[steps, envs], device=chosen_device)
                    distribution = torch.distributions.Normal(mean, torch.exp(log_std))
                    release_distribution = _release_distribution(release_logits, release_weights)
                    log_probability = _log_prob(distribution, latent_t, effective_active) + torch.sum(
                        release_distribution.log_prob(modes) * release_active_t, dim=-1
                    )
                    log_ratio = log_probability - old_log_prob
                    ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                    unclipped = ratio * advantage_t
                    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_t
                    policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                    value_error = torch.nn.functional.smooth_l1_loss(value, return_t, reduction="none")
                    if value_clip > 0.0:
                        value_clipped = old_value + torch.clamp(value - old_value, -value_clip, value_clip)
                        value_error = torch.maximum(
                            value_error, torch.nn.functional.smooth_l1_loss(value_clipped, return_t, reduction="none")
                        )
                    value_loss = torch.mean(value_error)
                    release_entropy = torch.sum(
                        release_distribution.entropy() * release_active_t, dim=-1
                    ) / torch.clamp(release_active_t.sum(dim=-1), min=1.0)
                    entropy_bonus = torch.mean(_entropy(distribution, effective_active) + release_entropy)
                    deterministic = _fraction_tensor(base_t, active_t, mean)
                    anchor_loss = torch.sum((deterministic - base_t * active_t) ** 2) / torch.clamp(
                        active_t.sum(), min=1.0
                    )
                    loss = (
                        policy_loss
                        + value_coef * value_loss
                        - entropy_coef * entropy_bonus
                        + anchor_coef * (1.0 - progress) * anchor_loss
                    )
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
                            float(anchor_loss.detach()),
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
                improved_enough = fitness > patience_best + min_delta
                stale_checks = 0 if improved_enough else stale_checks + 1
                if improved_enough:
                    patience_best = fitness
                reached_checks = reached_checks + 1 if fitness >= target_rss else 0
                _save_json(run / "latest_validation.json", validation_metrics)
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
                "validation_fallback_weeks": None
                if validation_metrics is None
                else validation_metrics["fallback_weeks"],
                "validation_rss_all": None if validation_metrics is None else json.dumps(validation_metrics["rss_all"]),
                "validation_rss_by_stratum": None
                if validation_metrics is None
                else json.dumps(validation_metrics["rss_by_stratum"]),
                "validation_checks_without_improvement": stale_checks,
                "shortage_usd": shortage_usd / n_envs,
                "shed_usd": shed_usd / n_envs,
                "anchor_loss": float(metric[5]),
                "best_validation_rss": best_rss,
                "elapsed_seconds": time.monotonic() - started,
            }
            _append_history(history, record)
            print(
                f"update {update}/{updates} steps={timesteps} return={record['mean_episode_return']} "
                f"pi={record['policy_loss']:.4f} vf={record['value_loss']:.4f} kl={record['approx_kl']:.5f}"
            )
            patience_stop = should_validate and update >= min_updates and patience > 0 and stale_checks >= patience
            target_stop = (
                should_validate and update >= min_updates and target_checks > 0 and reached_checks >= target_checks
            )
            should_stop = patience_stop or target_stop
            if update % checkpoint_every == 0 or update == updates or should_stop:
                _save_checkpoint(
                    run / "checkpoint.pt",
                    model,
                    optimizer,
                    update,
                    timesteps,
                    best_rss,
                    best_update,
                    settings,
                    stale_checks,
                    episodes_finished,
                    reached_checks,
                    patience_best,
                    rng,
                )
            if should_stop:
                print(
                    "target reached on consecutive held-out checks"
                    if target_stop
                    else f"early stopping after {stale_checks} validation checks without improvement"
                )
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
        "target_rss": target_rss,
        "target_met": best_rss >= target_rss,
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
