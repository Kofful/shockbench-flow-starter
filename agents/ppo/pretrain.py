"""Warm-start PPO from a planning teacher using only public observations.

The MPC/HiGHS dependencies are training-only. Evaluation uses the exported
neural weights, including learned queue releases, and performs no LP solves.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import torch
import train as training
from shockbench_flow.information.flat import flat_from_action
from shockbench_flow.policies.mpc_det import MpcDet
from shockbench_flow_gym import agent_config_from_reset
from shockbench_flow_gym.wrappers import REWARD_SCALE_USD


def main(
    task: str = "tiny",
    episodes: int = 32,
    epochs: int = 30,
    batch_episodes: int = 4,
    entropy: int = 20261007,
    validation_entropy: int = 20261006,
    validation_episodes: int = 100,
    validation_jobs: int = 1,
    eval_every: int = 5,
    quick_validation: bool = False,
    learning_rate: float = 1e-3,
    device: str = "auto",
    seed: int = 0,
    dataset: str | None = None,
    aggregate_dataset: str | None = None,
    behavior_policy: str | None = None,
    teacher_probability: float = 1.0,
    initial_policy: str | None = None,
    out: str | None = None,
    install: bool = False,
) -> None:
    """Collect non-dev demonstrations, fit full episode sequences, select by RSS."""
    if entropy == 0 or validation_entropy == 0 or entropy == validation_entropy:
        raise ValueError("teacher and validation roots must be distinct and nonzero")
    if min(episodes, epochs, batch_episodes, validation_episodes, eval_every) < 1:
        raise ValueError("episode/epoch/batch/evaluation counts must be positive")
    if not 0.0 <= teacher_probability <= 1.0:
        raise ValueError("teacher_probability must be in [0, 1]")
    if teacher_probability < 1.0 and not behavior_policy:
        raise ValueError("behavior_policy is required for learner-state demonstrations")
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    chosen = torch.device(resolved)
    if chosen.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    run = Path(out or training.ROOT / "outputs" / "ppo_pretrain" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir(parents=True, exist_ok=True)
    settings = {
        "task": task,
        "episodes": episodes,
        "epochs": epochs,
        "entropy": entropy,
        "validation_entropy": validation_entropy,
        "validation_episodes": validation_episodes,
        "seed": seed,
        "device": resolved,
        "teacher": "mpc_det (public standard observations only)",
        "behavior_policy": behavior_policy,
        "teacher_probability": teacher_probability,
    }
    training._save_json(run / "settings.json", settings)
    env = gym.make(training.env_id(task), entropy=entropy, regime="standard")
    observations, info = env.reset(seed=seed)
    config = agent_config_from_reset(env, observations, info)
    builder = training.FeatureBuilder(config, queue_control=True)
    weights = torch.as_tensor(builder.release_weights, device=chosen)
    n_dispatch = builder.n_dispatch
    data = []
    original_policy = training.deployment.POLICY
    if behavior_policy:
        candidate = torch.jit.load(behavior_policy, map_location=training.deployment.DEVICE)
        if int(candidate.feature_version) != training.FEATURE_VERSION:
            raise ValueError("behavior policy schema mismatch")
        candidate.eval()
        training.deployment.POLICY = candidate
    try:
        if dataset:
            with np.load(dataset, allow_pickle=False) as saved:
                if int(saved["feature_version"]) != training.FEATURE_VERSION or str(saved["task"]) != task:
                    raise ValueError("dataset schema/task mismatch")
                if int(saved["entropy"]) != entropy:
                    raise ValueError("dataset root must match --entropy")
                packed = {
                    key: saved[key].copy()
                    for key in ("features", "base", "active", "release_active", "target", "modes", "returns")
                }
        else:
            for episode in range(episodes):
                observations, info = env.reset(seed=int(rng.integers(1, 2**31)))
                builder.reset()
                learner = (
                    training.deployment.Agent(agent_config_from_reset(env, observations, info))
                    if behavior_policy
                    else None
                )
                teacher = MpcDet()
                teacher.reset(info["static"], info["obs"], seed + episode)
                rows = {
                    key: [] for key in ("features", "base", "active", "release_active", "target", "modes", "reward")
                }
                done = False
                cost_cents = 0
                while not done:
                    rows["features"].append(builder.encode(observations))
                    rows["base"].append(builder.last_base.copy())
                    rows["active"].append(builder.last_active.copy())
                    rows["release_active"].append(builder.last_release_active.copy())
                    flows, override, modes = flat_from_action(env.unwrapped.layout, teacher.act(info["obs"]))
                    target = np.concatenate((flows, override)) / builder.last_scale
                    rows["target"].append(np.clip(target, 0.0, 1.0).astype(np.float32))
                    rows["modes"].append(modes.astype(np.int64))
                    teacher_action = {"flows": flows, "override_qty": override, "release_mode": modes}
                    learner_action = learner.act(observations) if learner is not None else teacher_action
                    action = teacher_action if rng.random() < teacher_probability else learner_action
                    observations, reward, terminated, truncated, info = env.step(action)
                    rows["reward"].append(reward / REWARD_SCALE_USD[config["static"]["instance_id"]])
                    cost_cents -= int(info["reward_cents"])
                    done = terminated or truncated
                returns = np.cumsum(np.asarray(rows.pop("reward"), dtype=np.float32)[::-1])[::-1].copy()
                rows["returns"] = returns
                data.append({key: np.asarray(value) for key, value in rows.items()})
                print(f"teacher episode {episode + 1}/{episodes}: cost ${cost_cents / 100:,.0f}", flush=True)
            packed = {key: np.stack([episode[key] for episode in data]) for key in data[0]}
            if aggregate_dataset:
                with np.load(aggregate_dataset, allow_pickle=False) as previous:
                    if (
                        int(previous["feature_version"]) != training.FEATURE_VERSION
                        or str(previous["task"]) != task
                        or int(previous["entropy"]) != entropy
                    ):
                        raise ValueError("aggregate dataset task/schema/root mismatch")
                    packed = {key: np.concatenate((previous[key], value), axis=0) for key, value in packed.items()}
            np.savez_compressed(
                run / "demonstrations.npz",
                **packed,
                task=task,
                entropy=entropy,
                feature_version=training.FEATURE_VERSION,
            )
    finally:
        env.close()
        training.deployment.POLICY = original_policy

    model = training.RecurrentResidualPolicy().to(chosen)
    training._load_weights(model, Path(initial_policy) if initial_policy else training.INSTALLED_POLICY)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation = training.scoring.episode_set(
        task, validation_episodes, entropy=validation_entropy, quick=quick_validation, n_jobs=validation_jobs
    )
    latest = run / "latest"
    best = run / "best"
    policy = latest / "policy.pt"
    training._export(model, policy)
    training._write_agent(latest, policy)
    initial = training._validation_metrics(validation, latest, validation_jobs)
    best_rss = float(initial["rss"])
    best_epoch = 0
    training._write_agent(best, policy)
    if initial_policy:
        baseline = run / "baseline"
        training._write_agent(baseline, training.INSTALLED_POLICY)
        baseline_metrics = training._validation_metrics(validation, baseline, validation_jobs)
        if float(baseline_metrics["rss"]) > best_rss:
            best_rss = float(baseline_metrics["rss"])
            training._write_agent(best, baseline / "policy.pt")
    history = []
    count, horizon = packed["features"].shape[:2]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(count)
        losses = []
        for first in range(0, count, batch_episodes):
            batch = order[first : first + batch_episodes]
            hidden = torch.zeros((len(batch), builder.n_slots, training.HIDDEN_SIZE), device=chosen)
            loss = torch.zeros((), device=chosen)
            for week in range(horizon):
                x = torch.as_tensor(packed["features"][batch, week], device=chosen)
                mean, _log_std, value, hidden, logits = model(x, hidden)
                base = torch.as_tensor(packed["base"][batch, week], device=chosen, dtype=torch.float32)
                target = torch.as_tensor(packed["target"][batch, week], device=chosen, dtype=torch.float32)
                active = torch.as_tensor(packed["active"][batch, week], device=chosen, dtype=torch.float32)
                modes = torch.as_tensor(packed["modes"][batch, week], device=chosen)
                release_active = torch.as_tensor(packed["release_active"][batch, week], device=chosen)
                active = training._conditional_active(active, modes, weights, n_dispatch)
                # Quantity-space loss avoids overweighting endpoint latents.
                # Leave the prediction unclipped here so wrong saturated
                # actions still receive a corrective supervised gradient.
                predicted = base + torch.tanh(mean)
                imitation = torch.sum((predicted - target) ** 2 * active) / torch.clamp(active.sum(), min=1.0)
                distribution = training._release_distribution(logits, weights)
                release_loss = -torch.sum(distribution.log_prob(modes) * release_active) / torch.clamp(
                    release_active.sum(), min=1.0
                )
                returns = torch.as_tensor(packed["returns"][batch, week], device=chosen)
                value_loss = torch.nn.functional.smooth_l1_loss(value, returns)
                loss = loss + (imitation + release_loss + 0.1 * value_loss) / horizon
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            training._export(model, policy)
            training._write_agent(latest, policy)
            metrics = training._validation_metrics(validation, latest, validation_jobs)
            record.update(metrics)
            if float(metrics["rss"]) > best_rss:
                best_rss, best_epoch = float(metrics["rss"]), epoch
                training._write_agent(best, policy)
        history.append(record)
        training._save_json(run / "history.json", history)
        print(f"pretrain epoch {epoch}/{epochs}: loss {record['loss']:.4f}; best RSS {best_rss:.6f}", flush=True)
    if install:
        temporary = training.INSTALLED_POLICY.with_suffix(".pt.tmp")
        import shutil

        shutil.copy2(best / "policy.pt", temporary)
        temporary.replace(training.INSTALLED_POLICY)
    training._save_json(
        run / "summary.json",
        {
            "best_rss": best_rss,
            "best_epoch": best_epoch,
            "initial_rss": initial["rss"],
            "best_agent": str(best),
            "installed": install,
            "quick": quick_validation,
        },
    )
    print(json.dumps({"best_rss": best_rss, "best_agent": str(best)}, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
