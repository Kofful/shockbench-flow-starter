"""Two agents' costs by component on the same episodes: where one gains or loses against the other.

    uv run python examples/compare_costs.py --a=gpu_rl --b=mpc --task=small --episodes=16
    uv run python examples/compare_costs.py --a=gpu_rl --b=mpc --task=full --episodes=4

Plays each agent (a folder of agents/) on episodes of ``entropy`` in the package's environment (regime standard, as
scored), sums every week's eight cost components and the terminal credit, and prints the mean per episode of each,
for both agents and their difference (a - b: negative means a spends less there). The chart goes to
outputs/compare_costs/<a>_vs_<b>_<task>.png: each component's mean for both agents, and the difference.
"""

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow.disruption.sampler import sample_omega
from shockbench_flow.hosting.tasks import split_label, task_generator
from shockbench_flow_agent import agent_config

from sbf_starter import ROOT, env_id
from sbf_starter.agents import load


def costs(agent: str, task: str, episodes: int, entropy: int) -> tuple[list[str], np.ndarray]:
    """[episodes, 9]: the eight components and minus the terminal credit, per episode (USD)."""
    inst, params = task_generator(task)
    cls = load(agent)
    rows = []
    for n in range(episodes):
        env = gym.make(env_id(task), regime="standard")
        env.unwrapped.omega_source = lambda i: sample_omega(inst, params, entropy, i, split_label(entropy))
        obs, info = env.reset(options={"episode": n})
        config = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
        names = list(config["layout"]["cost_components"])
        policy = cls(config)
        parts, total, done = np.zeros(len(names)), 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            parts += obs["last_week.cost_components"]
            total -= reward
            done = terminated or truncated
        rows.append([*parts, total - parts.sum()])  # what is left of J is minus the terminal credit
        print(f"  {agent} episode {n}: J {total / 1e9:,.1f}B", flush=True)
    return [*names, "terminal credit"], np.array(rows)


def main(a: str = "gpu_rl", b: str = "mpc", task: str = "small", episodes: int = 16, entropy: int = 9001) -> None:
    names, ca = costs(a, task, episodes, entropy)
    _, cb = costs(b, task, episodes, entropy)
    ma, mb = ca.mean(0) / 1e9, cb.mean(0) / 1e9
    print(f"\n{task}, {episodes} episodes of root {entropy}, mean per episode in billions of USD:")
    print(f"  {'component':18s} {a:>12s} {b:>12s} {'a - b':>12s}")
    for name, x, y in zip(names, ma, mb):
        print(f"  {name:18s} {x:12,.1f} {y:12,.1f} {x - y:+12,.1f}")
    print(f"  {'total J':18s} {ma.sum():12,.1f} {mb.sum():12,.1f} {ma.sum() - mb.sum():+12,.1f}")
    chart(names, ma, mb, a, b, task, episodes, entropy)


def chart(names, ma, mb, a, b, task, episodes, entropy) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURFACE = "#ffffff", "#c3c2b7", "#33332f", "#1a1a19"
    BLUE, GREY, GREEN, RED = "#3987e5", "#8a8980", "#199e70", "#e66767"
    plt.rcParams.update(
        {
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.labelcolor": INK2,
            "font.size": 9,
        }
    )
    order = np.argsort(-np.maximum(ma, mb))
    labels = [names[i] for i in order]
    y = np.arange(len(order))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    left.barh(y - 0.2, ma[order], height=0.4, color=BLUE, label=a)
    left.barh(y + 0.2, mb[order], height=0.4, color=GREY, label=b)
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlabel("млрд $ на епізод")
    left.set_title(f"Витрати за статтями (J: {a} {ma.sum():,.0f}, {b} {mb.sum():,.0f})", loc="left", color=INK)
    left.legend(frameon=False)
    d = ma[order] - mb[order]
    right.barh(y, d, color=[GREEN if v < 0 else RED for v in d], height=0.6)
    for yi, v in zip(y, d):
        right.text(v, yi, f" {v:+,.1f} ", va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=8)
    right.axvline(0, color=INK2, lw=1)
    right.set_yticks(y, [""] * len(y))
    right.invert_yaxis()
    right.set_xlabel("млрд $ на епізод")
    right.set_title(f"{a} − {b}: зелене — {a} витрачає менше", loc="left", color=INK)
    for ax in (left, right):
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{task}, {episodes} епізодів root {entropy}", x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    out = ROOT / "outputs" / "compare_costs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a}_vs_{b}_{task}.png"
    fig.savefig(path, dpi=120)
    print(f"chart: {path}")


if __name__ == "__main__":
    fire.Fire(main)
