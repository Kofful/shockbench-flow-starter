"""A hand-written rule (the ``heuristic`` agent) against send-the-maximum, on dev episodes where a strait closes.

The rule ships everything that can move, less into a strait that is partly closed: a sea lane through a strait ships
its capacity times the strait's observed open fraction. On an episode where no strait closes it plays exactly what
send-the-maximum plays, so this example looks for the first dev episodes in which a strait falls below ``min_open``
open, and shows the difference in USD there. Read its ``agent.py`` (``src/sbf_starter/agents/heuristic/``) for the
details; it is a starting point, not a tuned policy.

    uv run python scripts/python/03_heuristic_agent.py              # Tiny, the first 3 episodes with a closure
    uv run python scripts/python/03_heuristic_agent.py task=small   # Small, the public board's network
    uv run python scripts/python/03_heuristic_agent.py episodes=6   # more of them
"""

import gymnasium as gym
import hydra
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
from omegaconf import DictConfig

from helper.logging import logger
from sbf_starter import runs
from sbf_starter.agents import load
from sbf_starter.play import episodes_with_closure, play_episode


@hydra.main(version_base=None, config_path="../../configs", config_name="03_heuristic_agent")
def main(cfg: DictConfig) -> None:
    """Play the rule and send-the-maximum on the first dev episodes with a closure and log the difference in USD."""
    out = runs.start(cfg)
    env = gym.make(cfg.task.env_id)
    rule, send_max = load("heuristic"), load("template")
    found = episodes_with_closure(env, cfg.episodes, search=cfg.search, min_open=cfg.min_open, seed=cfg.seed)
    logger.info("dev episodes in which a strait falls below {:.0%} open: {}", cfg.min_open, found)
    saved = 0.0
    for n in found:
        mine, base = play_episode(env, rule, n), play_episode(env, send_max, n)
        saved += base - mine
        msg = "dev episode {}: the rule {:,.0f} USD, send-the-maximum {:,.0f} USD: the rule saves {:,.0f} USD ({:+.1%})"
        logger.info(msg, n, mine, base, base - mine, mine / base - 1)
    logger.info(
        "in total the rule saves {:,.0f} USD on these {} episode(s) (negative: it costs more)", saved, len(found)
    )
    logger.info("on the score's scale, over the dev split: uv run sbf compare heuristic template")
    runs.finish(out, episodes=found, saved_usd=round(saved, 2))


if __name__ == "__main__":
    main()
