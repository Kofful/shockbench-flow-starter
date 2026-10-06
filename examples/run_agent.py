"""One agent played week by week under gymnasium, printing what it sends and what each week costs.

uv run python examples/run_agent.py
uv run python examples/run_agent.py --agent=heuristic --task=small --episode=3
"""

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401  registers the ShockBench/* environments
from shockbench_flow_agent import agent_config

from sbf_starter import env_id
from sbf_starter.agents import load


def main(agent: str = "template", task: str = "tiny", episode: int = 0) -> None:
    """Play dev episode ``episode`` with ``agents/<agent>/agent.py`` and print every week."""
    env = gym.make(env_id(task))
    obs, info = env.reset(options={"episode": episode})
    config = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    policy = load(agent)(config)  # the class from agent.py, built as the server builds it
    names = config["layout"]["cost_components"]

    total, done = 0.0, False
    while not done:
        week = int(obs["week"].item())
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total += reward
        parts = obs["last_week.cost_components"]
        top = ", ".join(f"{names[i]} {parts[i] / 1e6:,.0f}M" for i in np.argsort(parts)[::-1][:3] if parts[i] > 0)
        print(
            f"week {week:3d}: sent {np.sum(action['flows']):12,.0f} units on {np.count_nonzero(action['flows']):3d} "
            f"routes | cost {-reward / 1e6:10,.1f}M USD | biggest: {top}"
        )
    print(f"{agent} on {task} dev episode {episode}: total cost {-total:,.0f} USD (lower is better)")


if __name__ == "__main__":
    fire.Fire(main)
