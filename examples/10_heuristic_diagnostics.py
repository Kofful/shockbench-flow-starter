"""Inspect the heuristic's complete episodes, planner failures and CPU usage.

    uv run python examples/10_heuristic_diagnostics.py --task=small --episodes=4

No reference shortcuts, training, pytest or external services. This measures
costs and controller health, not RSS; use sbf compare for paired scores.
"""

import json
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from shockbench_flow_agent import agent_config
from shockbench_flow_gym.wrappers import EpisodeRecorder

from sbf_starter import env_id
from sbf_starter.agents import load


def main(
    task: str = "tiny",
    episodes: int = 1,
    entropy: int = 20261012,
    regime: str = "standard",
    policy_seed: int = 0,
    out: str | None = None,
):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    out = Path(out or f"outputs/10_heuristic_diagnostics/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    env = EpisodeRecorder(
        gym.make(env_id(task), entropy=entropy, regime=regime, policy_seed=policy_seed), directory=out
    )
    agent_class = load("heuristic")
    rows = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=0, options={"episode": episode})
            began = time.process_time()
            agent = agent_class(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
            init_cpu = time.process_time() - began
            weeks, total = [], 0
            while True:
                observed = {k: v.copy() for k, v in obs.items()}
                before = time.process_time()
                action = agent.act(obs)
                cpu = time.process_time() - before
                if not weeks:
                    cpu += init_cpu
                if not env.action_space.contains(action):
                    raise AssertionError("action is outside its public shape/dtype/value space")
                if any(not np.array_equal(obs[k], observed[k], equal_nan=True) for k in obs):
                    raise AssertionError("agent mutated its observation")
                if obs["action_mask.observed"].item() and np.any(action["flows"][obs["action_mask"] == 0]):
                    raise AssertionError("agent requested a known prohibited flow")
                if obs["override_mask.observed"].item() and np.any(action["override_qty"][obs["override_mask"] == 0]):
                    raise AssertionError("agent requested a known prohibited override")
                obs, _, terminated, truncated, info = env.step(action)
                total -= int(info["reward_cents"])
                weeks.append(
                    agent.last_plan
                    | {
                        "cpu_s": cpu,
                        "requested_qty": float(action["flows"].sum()),
                        "overridden_pairs": int(np.count_nonzero(action["release_mode"] == 1)),
                        "held_pairs": int(np.count_nonzero(action["release_mode"] == 2)),
                    }
                )
                if terminated or truncated:
                    break
            invalid = [list(r.invalid) for r in env.unwrapped.core.trajectory.records if r.invalid]
            if invalid:
                raise AssertionError(f"invalid simulator actions: {invalid}")
            if len(agent.history) != env.unwrapped.instance.T:
                raise AssertionError("history does not contain every week")
            row = {
                "episode": episode,
                "cost_usd": total / 100,
                "internal_solver_fallbacks": agent.solver_failures,
                "max_week_cpu_s": max(w["cpu_s"] for w in weeks),
                "weeks": weeks,
                "loss_components_usd": dict(
                    zip(agent.layout["cost_components"], agent.loss_totals.tolist(), strict=True)
                ),
                "history_weeks": len(agent.history),
            }
            # The final week's actual cost is not part of a next observation.
            last = env.unwrapped.core.trajectory.records[-1]
            row["loss_components_usd"] = {
                k: row["loss_components_usd"][k] + getattr(last.costs, k) for k in agent.layout["cost_components"]
            }
            rows.append(row)
            print(
                f"{task} episode {episode}: ${total / 100:,.2f}; planner fallbacks {agent.solver_failures}; "
                f"max weekly CPU {row['max_week_cpu_s']:.3f}s",
                flush=True,
            )
    finally:
        env.close()
    summary = {"task": task, "entropy": entropy, "regime": regime, "policy_seed": policy_seed, "episodes": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Written {out / 'summary.json'} and EpisodeRecorder NPZ files", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
