"""Find dev episodes in which a strait closes, for the examples that show a disruption (03 and 07)."""

from __future__ import annotations


def closes(env, episode: int, min_open: float, seed: int = 0) -> bool:
    """Whether a strait is fully open in week 1 of the episode and later below ``min_open`` open."""
    from shockbench_flow_gym.timeline import episode_events

    events = episode_events(env, seed=seed, options={"episode": episode})
    opens = [e for e in events if e["signal"] == "chokepoint"]
    return bool(opens) and opens[0]["before"] is not None and min(e["after"] for e in opens) < min_open


def episodes_with_closure(env, count: int, *, search: int = 60, min_open: float = 0.5, seed: int = 0) -> list[int]:
    """The first ``count`` dev episodes below ``search`` in which a strait closes; SystemExit when fewer do."""
    found = [n for n in range(search) if closes(env, n, min_open, seed)][:count]
    if len(found) < count:
        raise SystemExit(f"only {len(found)} dev episode(s) below {search} show a strait below {min_open:.0%} open")
    return found
