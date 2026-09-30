"""Quickstart: one episode with random actions, the standard gymnasium loop.

uv run python scripts/python/01_quickstart.py              # Tiny, the practice network (a second)
uv run python scripts/python/01_quickstart.py task=small   # Small, the public board's network
uv run python scripts/python/01_quickstart.py episode=3    # another dev episode

It plays dev episode ``episode`` (``env.reset(options={"episode": 0})`` by default): the same index replays the same
scenario, and it is the episode of the same index that ``sbf evaluate --episodes=[0]`` scores.
"""

import gymnasium as gym
import hydra
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
from omegaconf import DictConfig

from helper.logging import logger
from sbf_starter import runs


@hydra.main(version_base=None, config_path="../../configs", config_name="01_quickstart")
def main(cfg: DictConfig) -> None:
    """Play one dev episode of ``cfg.task`` with random actions and log its cost."""
    out = runs.start(cfg)
    env = gym.make(cfg.task.env_id)  # regime="standard": what the scored runs see
    obs, info = env.reset(options={"episode": cfg.episode})  # dev episode `episode`: the same index, the same scenario
    env.action_space.seed(cfg.seed)
    logger.info("{}: {} observation arrays; action parts {}", cfg.task.env_id, len(obs), list(env.action_space.spaces))
    logger.info("dev episode {}: env.reset(options={{'episode': {}}}) replays it", info["episode"], info["episode"])
    total, weeks, done = 0.0, 0, False
    while not done:
        action = env.action_space.sample()  # {"flows": ..., "override_qty": ..., "release_mode": ...}
        obs, reward, terminated, truncated, info = env.step(action)  # reward = minus this week's cost in USD
        total, weeks, done = total + reward, weeks + 1, terminated or truncated
    logger.info("{} weeks, total cost {:,.0f} USD with random actions (lower is better)", weeks, -total)
    runs.finish(out, cost_usd=round(-total, 2))


if __name__ == "__main__":
    main()
