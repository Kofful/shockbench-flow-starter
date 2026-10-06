"""Dashboard of an agent's losses on one episode, including the naive baseline.

    uv run python examples/08_agent_losses.py --quick
    uv run python examples/08_agent_losses.py --task=small --episode=0 --quick
    uv run python examples/08_agent_losses.py --agent=template --episode=29 --quick
    uv run python examples/08_agent_losses.py --agent=ppo --task=small --episode=0 --clairvoyant

By default this plays the ``rl`` agent (a folder or a standalone ``rl.py``). It
writes the standard episode dashboard and animation, plus
``losses_agent.png``: weekly and cumulative objective loss, excess loss against
naive, and the cost-component breakdown. In ShockBench, loss means episode
cost J (lower is better), not a supervised-learning loss.
With ``--clairvoyant``, example 09 instead writes an offline three-plan report
with the exact reference LP and same-state decision interventions.
"""

import runpy
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from shockbench_flow_agent import NAIVE_REPLICATIONS, QUICK
from shockbench_flow_gym import dashboard  # also registers the ShockBench/* environments

from sbf_starter import ROOT, env_id
from sbf_starter.agents import load
from sbf_starter.play import episodes_with_closure


DEFAULT_AGENT = ROOT / "agents" / "rl.py"
if not DEFAULT_AGENT.is_file():
    DEFAULT_AGENT = ROOT / "agents" / "rl"
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def _money(value, _position=None) -> str:
    """Compact USD tick labels that remain readable for Tiny, Small and Full."""
    value = float(value)
    size = abs(value)
    if size >= 1e12:
        return f"${value / 1e12:.1f}T"
    if size >= 1e9:
        return f"${value / 1e9:.1f}B"
    if size >= 1e6:
        return f"${value / 1e6:.1f}M"
    if size >= 1e3:
        return f"${value / 1e3:.1f}k"
    return f"${value:,.0f}"


def _style(ax, title: str, T: int) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlim(0.5, T + 0.5)
    ax.set_xlabel("week")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(_money))


def loss_dashboard(record: dict, path: str | Path, dpi: int = 110) -> Figure:
    """Write a four-panel loss analysis from ``dashboard.record_episode``."""
    meta = record["meta"]
    T = int(meta["weeks"])
    weeks = np.arange(1, T + 1)
    weekly = -np.asarray(record["reward_cents"], dtype=np.float64) / 100.0
    cumulative = np.cumsum(weekly)
    naive = record.get("naive")
    naive_weekly = None if naive is None else -np.asarray(naive["reward_cents"], dtype=np.float64) / 100.0
    naive_cumulative = None if naive_weekly is None else np.cumsum(naive_weekly)

    fig = Figure(figsize=(15, 10), dpi=dpi, facecolor="#fcfcfb")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.20, left=0.08, right=0.97, top=0.89, bottom=0.08)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(weeks, weekly, color=COLOURS[0], linewidth=1.8, label=str(meta.get("agent", "agent")))
    if naive_weekly is not None:
        ax.plot(weeks, naive_weekly, color=COLOURS[1], linewidth=1.5, alpha=0.85, label="naive")
    ax.axhline(0, color="#898781", linewidth=0.8)
    ax.legend(frameon=False)
    _style(ax, "weekly objective loss (week T includes terminal credit)", T)

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(weeks, cumulative, color=COLOURS[0], linewidth=2.0, label=f"agent J {_money(cumulative[-1])}")
    if naive_cumulative is not None:
        ax.plot(
            weeks,
            naive_cumulative,
            color=COLOURS[1],
            linewidth=1.8,
            label=f"naive J {_money(naive_cumulative[-1])}",
        )
    ax.legend(frameon=False)
    _style(ax, "cumulative objective loss", T)

    ax = fig.add_subplot(grid[1, 0])
    if naive_cumulative is None:
        ax.plot(weeks, cumulative, color=COLOURS[0], linewidth=2.0)
        title = "cumulative loss (run without naive comparison)"
    else:
        excess = cumulative - naive_cumulative
        ax.plot(weeks, excess, color="#52514e", linewidth=1.8)
        ax.fill_between(weeks, 0, excess, where=excess >= 0, color="#e34948", alpha=0.35, label="worse than naive")
        ax.fill_between(weeks, 0, excess, where=excess < 0, color="#1baf7a", alpha=0.35, label="saving vs naive")
        ax.axhline(0, color="#898781", linewidth=0.8)
        ax.legend(frameon=False)
        title = f"cumulative excess loss: final {_money(excess[-1])} (below zero is better)"
    _style(ax, title, T)

    ax = fig.add_subplot(grid[1, 1])
    costs = np.asarray(record["costs"], dtype=np.float64)
    names = list(meta["cost_components"])
    active = [i for i in range(costs.shape[1]) if np.any(costs[:, i])]
    if active:
        ax.stackplot(
            weeks,
            *(costs[:, i] for i in active),
            labels=[names[i] for i in active],
            colors=[COLOURS[i % len(COLOURS)] for i in active],
            alpha=0.88,
        )
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    _style(ax, "weekly loss by cost component", T)

    episode = meta.get("episode")
    episode_text = "" if episode is None else f", episode {episode}"
    fig.suptitle(
        f"{meta.get('agent', 'agent')} loss dashboard · {meta['instance']}{episode_text} · lower is better",
        x=0.08,
        ha="left",
        fontsize=13,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="#fcfcfb")
    return fig


def main(
    agent: str | None = None,
    task: str = "tiny",
    episode: int | None = None,
    search: int = 60,
    min_open: float = 0.5,
    naive: bool = True,
    quick: bool = False,
    n_jobs: int = -1,
    regime: str = "standard",
    seed: int = 0,
    out: str | None = None,
    clairvoyant: bool = False,
    audit_weeks: int = 3,
) -> None:
    """Play an agent and write its standard and loss-focused dashboards.

    Args:
        agent: agent name or path (default: the local rl agent).
        task: tiny, small or full.
        episode: dev episode (default: first with a strait below ``min_open``).
        search: look through dev episodes 0 .. search - 1.
        min_open: open-fraction threshold used when selecting an episode.
        naive: include the naive baseline (``--nonaive`` to skip it).
        quick: use the rough, fast naive model rather than the board's model.
        n_jobs: workers for the naive model's first computation (-1: all cores).
        regime: information regime; standard is scored.
        seed: environment reset seed.
        out: output folder (default: outputs/08_agent_losses/<date_time>).
        clairvoyant: write a three-plan dashboard with decision diagnostics (example 09).
        audit_weeks: weeks to audit in clairvoyant comparison mode (zero skips).

    """
    agent = str(DEFAULT_AGENT if agent is None else agent)
    if clairvoyant:
        if not naive:
            raise ValueError("clairvoyant comparison mode requires the naive baseline")
        compare = runpy.run_path(str(Path(__file__).with_name("09_compare_plans.py")))["main"]
        compare(
            agent=agent,
            task=task,
            episode=episode,
            search=search,
            min_open=min_open,
            quick=quick,
            n_jobs=n_jobs,
            regime=regime,
            seed=seed,
            audit_weeks=audit_weeks,
            out=out,
        )
        return
    agent_class = load(agent)
    label = Path(agent).stem if Path(agent).suffix else Path(agent).name
    out = Path(out or f"outputs/08_agent_losses/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id(task), regime=regime)
    if episode is None:
        (episode,) = episodes_with_closure(env, 1, search=search, min_open=min_open, seed=seed)
        print(f"dev episode {episode}: a strait closes below {min_open:.0%} open")
    else:
        episode = int(episode)
        print(f"dev episode {episode}")

    replications = None if not naive else QUICK["fq_replications"] if quick else NAIVE_REPLICATIONS
    started = time.perf_counter()
    record = dashboard.record_episode(
        env,
        agent_class,
        seed=seed,
        options={"episode": episode},
        naive_replications=replications,
        n_jobs=n_jobs,
    )
    record["meta"]["agent"] = label
    agent_cost = record["meta"]["J_cents"] / 100
    comparison = ""
    if "naive" in record:
        naive_cost = record["meta"]["naive_J_cents"] / 100
        comparison = f", naive {_money(naive_cost)}, excess loss {_money(agent_cost - naive_cost)}"
    print(f"{label}: loss J {_money(agent_cost)}{comparison} ({time.perf_counter() - started:.1f} s)")

    dashboard.plot_network(task).savefig(out / "network.png", dpi=100)
    dashboard.save_record(record, out / "record_agent.npz")
    dashboard.episode_dashboard(record, out / "dashboard_agent.png")
    dashboard.episode_animation(record, out / "episode_agent.gif")
    loss_dashboard(record, out / "losses_agent.png")
    print(f"written in {out}: network.png, dashboard_agent.png, losses_agent.png, episode_agent.gif, record_agent.npz")


if __name__ == "__main__":
    fire.Fire(main)
