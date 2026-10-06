"""One agent's episode in a window with a week slider, to step through it at your own pace.

uv run python examples/watch_agent.py
uv run python examples/watch_agent.py --agent=heuristic --episode=3

The episode is played first and every week recorded; then the window opens. Drag the slider, or press the left and
right arrow keys, to move between weeks.

Left: the network after that week (env.render(): red routes closed or banned, amber reduced, the strait coloured by
how open it is). Right, top to bottom: the cost of every week (the chosen one highlighted), the same split by cost
component (the colours of examples/08_agent_losses.py), and that week in numbers with what changed during it
(``shockbench_flow_gym.timeline.observed_events``: closures, sanctions, capacities, tariffs, outages, announcements).
"""

import fire
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import shockbench_flow_gym  # noqa: F401  registers the ShockBench/* environments
from matplotlib.widgets import Slider
from shockbench_flow_agent import agent_config
from shockbench_flow_gym.timeline import observed_events

from sbf_starter import env_id
from sbf_starter.agents import load
from sbf_starter.play import episodes_with_closure


# the cost components' colours of examples/08_agent_losses.py, so the two read alike
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


MAX_EVENTS = 10  # lines of "what changed" in the panel; examples/episode_events.py lists them all


def _value(x) -> str:
    return "-" if x is None else f"{x:.3g}" if isinstance(x, float) else str(x)


def changes(events: list[dict]) -> list[str]:
    """The events of a week as short lines (signal: subject before -> after), at most ``MAX_EVENTS``."""
    lines = []
    for e in events[:MAX_EVENTS]:
        moved = f" {_value(e['before'])} -> {_value(e['after'])}" if e["before"] is not None else ""
        lines.append(f"  {e['signal']}: {e['subject']}{moved}"[:80])
    if len(events) > MAX_EVENTS:
        lines.append(f"  ... and {len(events) - MAX_EVENTS} more (examples/episode_events.py lists them)")
    return lines or ["  nothing"]


def describe(obs: dict, action: dict, config: dict, events: list[dict], week: int) -> str:
    """The week in numbers, from the observation after it, the action that played it and what changed during it."""
    layout, nodes = config["layout"], config["static"]["nodes"]["id"]
    parts = obs["last_week.cost_components"]
    opens = [f"{nodes[c].removeprefix('chk_')} {o:.2f}" for c, o in zip(layout["chokepoints"], obs["graph_now.open"])]
    lines = ["straits, how open (1 open, 0 closed):"]
    lines += ["  " + ", ".join(opens[i : i + 4]) for i in range(0, len(opens), 4)]
    demand, lost = obs["last_week.sinks.demand"].sum(), obs["last_week.sinks.lost"].sum()
    lines += [
        f"customers: wanted {demand:,.0f}, did not get {lost:,.0f}",
        f"power not delivered by the grids: {obs['last_week.shed.qty'].sum():,.1f}",
        f"the agent sent {action['flows'].sum():,.0f} units on {np.count_nonzero(action['flows'])} routes",
        "",
        f"cost of the week: {parts.sum() / 1e6:,.1f}M USD",
    ]
    lines += [f"  {name}: {p / 1e6:,.1f}M" for name, p in zip(layout["cost_components"], parts) if p > 0]
    lines += ["", f"what changed during week {week}:", *changes(events)]
    return "\n".join(lines)


def main(agent: str = "template", task: str = "tiny", episode: int | None = None) -> None:
    """Play dev episode ``episode`` (default: the first where a strait closes) with ``agents/<agent>``; browse it."""
    env = gym.make(env_id(task), render_mode="rgb_array")
    if episode is None:
        (episode,) = episodes_with_closure(env, 1)
    obs, info = env.reset(options={"episode": episode})
    config = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    policy = load(agent)(config)

    frames, texts, weekly, parts, done = [], [], [], [], False
    start = observed_events(config, None, obs)  # what is already disrupted before week 1
    while not done:
        action = policy.act(obs)
        prev = obs
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        week = len(frames) + 1
        events = observed_events(config, prev, obs)  # seen from the next week on: the agent learns of it then
        if week == 1:
            events = [{**e, "signal": f"{e['signal']} (at the start)"} for e in start] + events
        frames.append(env.render())
        texts.append(describe(obs, action, config, events, week))
        weekly.append(-reward / 1e6)
        parts.append(obs["last_week.cost_components"] / 1e6)
    weeks = len(weekly)
    print(f"{agent} on {task} dev episode {episode}: total cost {sum(weekly) * 1e6:,.0f} USD")

    fig = plt.figure(figsize=(15, 10))
    ax_map = fig.add_axes((0.01, 0.1, 0.58, 0.84))
    ax_cost = fig.add_axes((0.65, 0.77, 0.33, 0.18))
    ax_parts = fig.add_axes((0.65, 0.47, 0.33, 0.22))
    ax_text = fig.add_axes((0.61, 0.02, 0.38, 0.39))
    ax_slider = fig.add_axes((0.1, 0.03, 0.8, 0.025))
    ax_map.axis("off")
    ax_text.axis("off")
    image = ax_map.imshow(frames[0])
    bars = ax_cost.bar(range(1, weeks + 1), weekly, color="#bdc3c7")
    ax_cost.set_xlabel("week")
    ax_cost.set_ylabel("million USD")
    ax_cost.set_title("the cost of every week")
    costs = np.array(parts)
    names = config["layout"]["cost_components"]
    active = [j for j in range(costs.shape[1]) if costs[:, j].any()]
    x = np.arange(1, weeks + 1)
    ax_parts.stackplot(
        x,
        *(costs[:, j] for j in active),
        labels=[names[j] for j in active],
        colors=[COLOURS[j % len(COLOURS)] for j in active],
        alpha=0.88,
    )
    ax_parts.set_xlim(1, weeks)
    ax_parts.set_xlabel("week")
    ax_parts.set_ylabel("million USD")
    ax_parts.set_title("weekly loss by cost component")
    ax_parts.legend(frameon=False, ncol=4, fontsize=7, loc="upper left")
    marker = ax_parts.axvline(1, color="#222222", linewidth=1.2)
    text = ax_text.text(0, 1, "", va="top", family="monospace", fontsize=7)
    slider = Slider(ax_slider, "week", 1, weeks, valinit=1, valstep=1)

    def show(week: float) -> None:
        i = int(week) - 1
        image.set_data(frames[i])
        for j, bar in enumerate(bars):
            bar.set_color("#c0392b" if j == i else "#bdc3c7")
        marker.set_xdata([i + 1, i + 1])
        ax_map.set_title(
            f"{agent} on {task}, dev episode {episode}: after week {i + 1} of {weeks}   "
            f"(total so far {sum(weekly[: i + 1]):,.0f}M of {sum(weekly):,.0f}M USD)"
        )
        text.set_text(texts[i])
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        step = {"right": 1, "left": -1}.get(event.key, 0)
        if step:
            slider.set_val(min(max(slider.val + step, 1), weeks))

    slider.on_changed(show)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show(1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
