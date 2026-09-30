"""Regenerate docs/fields/ from the installed shockbench-flow, after a new release (and `uv sync`).

uv run python scripts/fields_docs.py
"""

import fire
import gymnasium as gym
from shockbench_flow_agent.spaces import markdown
from shockbench_flow_gym import agent_config_from_reset  # also registers the ShockBench/* environments

from sbf_starter import ROOT, TASKS, env_id


COMMAND = "uv run python scripts/fields_docs.py"


def fields(task: str) -> str:
    env = gym.make(env_id(task))
    obs, info = env.reset(options={"episode": 0})
    config = agent_config_from_reset(env, obs, info)
    return markdown(config, obs, grouped="lot_keys" in config["layout"], generated_by=COMMAND)


def main(tasks: list[str] = tuple(TASKS)) -> None:
    for task in tasks:
        path = ROOT / "docs" / "fields" / f"{task}.md"
        path.write_text(fields(task) + "\n")
        print(f"written {path.relative_to(ROOT)}")


if __name__ == "__main__":
    fire.Fire(main)
