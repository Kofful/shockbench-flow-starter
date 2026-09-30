"""The heuristic agent against send-the-maximum, on the dev episodes where a strait closes.

    uv run python examples/03_heuristic_agent.py
    uv run python examples/03_heuristic_agent.py --task=small --episodes=6

Where no strait closes the two play the same, so only episodes with a closure are shown.
"""

import fire
import gymnasium as gym
from shockbench_flow_gym import play_episode  # also registers the ShockBench/* environments

from sbf_starter import env_id
from sbf_starter.agents import load
from sbf_starter.play import episodes_with_closure


def main(task: str = "tiny", episodes: int = 3, min_open: float = 0.5, search: int = 60, seed: int = 0) -> None:
    """Print what the rule saves in USD on each episode.

    Args:
        task: tiny, small or full.
        episodes: how many episodes with a closure to play.
        min_open: a strait's observed open fraction must fall below this for an episode to count.
        search: look through dev episodes 0 .. search - 1.
        seed: the reset's seed.

    """
    env = gym.make(env_id(task))
    rule, send_max = load("heuristic"), load("template")
    found = episodes_with_closure(env, episodes, search=search, min_open=min_open, seed=seed)
    print(f"dev episodes in which a strait falls below {min_open:.0%} open: {found}")
    saved = 0.0
    for n in found:
        mine, base = play_episode(env, rule, n), play_episode(env, send_max, n)
        saved += base - mine
        print(
            f"dev episode {n}: the rule {mine:,.0f} USD, send-the-maximum {base:,.0f} USD: "
            f"the rule saves {base - mine:,.0f} USD ({mine / base - 1:+.1%})"
        )
    print(f"in total the rule saves {saved:,.0f} USD on these {len(found)} episode(s) (negative: it costs more)")
    print("on the score's scale, over the dev split: uv run sbf compare heuristic template")


if __name__ == "__main__":
    fire.Fire(main)
