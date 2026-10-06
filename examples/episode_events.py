"""What happened in an episode, week by week: closures, sanctions, capacity cuts, tariffs, outages, announcements.

uv run python examples/episode_events.py --task=small --episode=5
uv run python examples/episode_events.py --task=small --episode=5 --start=35 --end=45
uv run python examples/episode_events.py --task=small --episode=5 --signals=chokepoint,prohibited,message

The events are the changes an agent can observe (``shockbench_flow_gym.timeline``), the same in any agent's episode:
the disruptions are drawn before the episode starts. "During week w" is the week whose cost they hit, the week of the
cost charts (examples/08_agent_losses.py, examples/watch_agent.py); the agent sees them from week w + 1 on. Week 0 is
what is already disrupted when the episode starts.

Signals: chokepoint, prohibited, lifted, capacity, tariff, war_risk, fab, osat, grid, supply, message,
message_update, message_ended, pending, closure_end.
"""

from collections import Counter

import fire
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401  registers the ShockBench/* environments
from shockbench_flow_gym.timeline import episode_events

from sbf_starter import env_id


def _value(x) -> str:
    return "-" if x is None else f"{x:.4g}" if isinstance(x, float) else str(x)


def main(task: str = "tiny", episode: int = 0, start: int = 0, end: int | None = None, signals: str = "") -> None:
    """Print the events of dev episode ``episode`` during weeks ``start`` .. ``end``, optionally some signals only."""
    events = episode_events(gym.make(env_id(task)), options={"episode": episode})
    keep = {s.strip() for s in (signals.split(",") if isinstance(signals, str) else signals) if s.strip()}
    rows = [
        e
        for e in events
        if start <= e["week"] - 1 <= (end if end is not None else 10**9) and (not keep or e["signal"] in keep)
    ]
    print(f"{task} dev episode {episode}: {len(rows)} of {len(events)} events shown")
    print("by signal:", ", ".join(f"{s} {n}" for s, n in Counter(e["signal"] for e in rows).most_common()))
    week = None
    for e in rows:
        if e["week"] - 1 != week:
            week = e["week"] - 1
            print(f"\nduring week {week}:" if week else "\nat the start (week 0):")
        moved = f"  {_value(e['before'])} -> {_value(e['after'])}" if e["before"] is not None else ""
        print(f"  {e['signal']:14s} {e['subject']}{moved}  ({e['note']})")


if __name__ == "__main__":
    fire.Fire(main)
