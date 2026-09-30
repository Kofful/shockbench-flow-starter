"""Play an ``Agent`` class under gymnasium the way the scorer plays it: ``Agent(config)`` once, then ``act`` weekly."""

from __future__ import annotations


def make_agent(env, agent_class, obs, info):
    """``agent_class(config)`` with the ``config`` the scorer would build for this reset (``agent_config``)."""
    from shockbench_flow_agent import agent_config

    return agent_class(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))


def play_episode(env, agent_class, episode: int) -> float:
    """Total cost in USD of ``agent_class`` on dev episode ``episode`` (``reset(options={"episode": episode})``).

    The gymnasium environment plays no fallback: an exception in ``act`` stops here, where the scorer would play the
    naive rule for that week.
    """
    obs, info = env.reset(options={"episode": episode})
    agent = make_agent(env, agent_class, obs, info)
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        total, done = total + reward, terminated or truncated
    return -total


def closes(env, episode: int, min_open: float, seed: int = 0) -> bool:
    """Whether a strait is fully open in week 1 of dev episode ``episode`` and later below ``min_open`` open."""
    from shockbench_flow_gym.timeline import episode_events

    events = episode_events(env, seed=seed, options={"episode": episode})
    opens = [e for e in events if e["signal"] == "chokepoint"]
    return bool(opens) and opens[0]["before"] is not None and min(e["after"] for e in opens) < min_open


def episodes_with_closure(env, count: int, *, search: int = 60, min_open: float = 0.5, seed: int = 0) -> list[int]:
    """The first ``count`` dev episodes below ``search`` in which a strait closes below ``min_open`` open.

    Raises:
        SystemExit: when fewer than ``count`` do.

    """
    found = [n for n in range(search) if closes(env, n, min_open, seed)][:count]
    if len(found) < count:
        raise SystemExit(f"only {len(found)} dev episode(s) below {search} show a strait below {min_open:.0%} open")
    return found
