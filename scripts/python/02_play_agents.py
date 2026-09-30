"""Two agents in the submission format, played under gymnasium exactly as the scorer plays them.

The scorer builds ``Agent(config)`` once per episode and calls ``act(observation)`` every week. Under gymnasium the same
``config`` comes from the reset's ``info`` (``sbf_starter.play.make_agent``). Both agents are submission folders the
organisers ship (``src/sbf_starter/agents/``):

- ``random``: random flows up to each route's capacity (a lower bar);
- ``template``: every open route ships its capacity (send the maximum), the folder to copy for your own agent.

    uv run python scripts/python/02_play_agents.py              # Tiny (seconds)
    uv run python scripts/python/02_play_agents.py task=small   # Small, the public board's network
"""

import gymnasium as gym
import hydra
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
from omegaconf import DictConfig

from helper.logging import logger
from sbf_starter import runs
from sbf_starter.agents import load
from sbf_starter.play import play_episode


@hydra.main(version_base=None, config_path="../../configs", config_name="02_play_agents")
def main(cfg: DictConfig) -> None:
    """Play both agents on dev episodes 0 .. episodes - 1 and log their costs in USD."""
    out = runs.start(cfg)
    env = gym.make(cfg.task.env_id)
    agents = {name: load(name) for name in ("random", "template")}
    totals = dict.fromkeys(agents, 0.0)
    for n in range(cfg.episodes):
        costs = {name: play_episode(env, cls, n) for name, cls in agents.items()}
        totals = {k: totals[k] + v for k, v in costs.items()}
        logger.info("dev episode {}: {}", n, ", ".join(f"{k} {v:,.0f} USD" for k, v in costs.items()))
    logger.info("costs only (lower is better): `uv run sbf compare template random` puts them on the score's scale")
    runs.finish(out, **{f"{k}_mean_cost_usd": round(v / cfg.episodes, 2) for k, v in totals.items()})


if __name__ == "__main__":
    main()
