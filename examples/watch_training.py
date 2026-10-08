"""A live chart of training runs: each run's score and plain MPC's on the same episodes, per map, redrawn as it grows.

    uv run python examples/watch_training.py                                   # the newest run of train_ppo_residual
    uv run python examples/watch_training.py --runs=outputs/train_ppo_residual/<run1>,outputs/<trainer>/<run2>
    uv run python examples/watch_training.py --png=outputs/progress.png        # one picture, no window

Any trainer can be shown: its run folder needs a ``log.json``, a list of rows, one per report, each with
``"minutes"`` and, per map it plays, ``{"rss": its score on the last episodes, "minus_mpc": that minus plain MPC's
score on the same episodes}`` (examples/train_ppo_residual.py writes it), and optionally ``"holdout_<map>"`` and
``"holdout_<map>_minus_mpc"`` (examples/gpu_ppo.py: the deterministic policy on held-out episodes, drawn as dots). Top
row: the scores (solid: the run, dashed: MPC on the same episodes); bottom row: the run minus MPC, above 0 when the
run beats MPC.
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt

from sbf_starter import ROOT


MAPS = ("small", "full")
THEMES = {  # series colours in this order, then text, secondary text, grid, background
    "light": (
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
        ("#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"),
    ),
    "dark": (
        ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
        ("#ffffff", "#c3c2b7", "#33332f", "#1a1a19"),
    ),
}
COLORS, (INK, INK2, GRID, SURFACE) = THEMES["dark"]
HELD_OUT = {"light": "#e34948", "dark": "#e66767"}  # red: the held-out evaluation
RED = HELD_OUT["dark"]


def newest(trainer: str = "train_ppo_residual") -> Path:
    runs = sorted(p.parent for p in (ROOT / "outputs" / trainer).glob("*/log.json"))
    if not runs:
        raise SystemExit(f"no run with a log.json in outputs/{trainer}/ yet")
    return runs[-1]


def read(run: Path) -> list[dict]:
    try:
        return json.loads((run / "log.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):  # not written yet, or caught mid-write
        return []


def draw(fig, axes, runs: list[Path], labels: list[str]) -> None:
    for ax in axes.ravel():
        ax.clear()
    for j, task in enumerate(MAPS):
        top, bottom = axes[0, j], axes[1, j]
        latest = []
        for i, (run, label) in enumerate(zip(runs, labels)):
            rows = [r for r in read(run) if task in r]
            if not rows:
                continue
            color = COLORS[i % len(COLORS)]
            t = [r["minutes"] for r in rows]
            rss = [r[task]["rss"] for r in rows]
            diff = [r[task]["minus_mpc"] for r in rows]
            mpc = [a - d for a, d in zip(rss, diff)]
            top.plot(t, rss, color=color, lw=2, label=label)
            top.plot(
                t,
                mpc,
                color=INK2 if len(runs) == 1 else color,
                lw=1.5,
                ls="--",
                alpha=0.8,
                label="MPC на тих самих епізодах" if len(runs) == 1 else f"MPC ({label})",
            )
            bottom.plot(t, diff, color=color, lw=2, label=label)
            held = [r for r in rows if f"holdout_{task}" in r]  # the deterministic policy on held-out episodes
            if held:
                th = [r["minutes"] for r in held]
                red = RED if len(runs) == 1 else color  # several runs: each its own colour
                top.plot(
                    th,
                    [r[f"holdout_{task}"] for r in held],
                    color=red,
                    lw=0,
                    marker="o",
                    ms=6,
                    label="перевірка на 9001 (без шуму)",
                )
                bottom.plot(
                    th, [r[f"holdout_{task}_minus_mpc"] for r in held], color=red, lw=1, marker="o", ms=6, ls=":"
                )
            last = rows[-1][task]
            text = f"{rss[-1]:.3f} проти MPC {mpc[-1]:.3f} ({last.get('episodes', '?')} еп.)"
            if held:
                text += f"; 9001: {held[-1][f'holdout_{task}']:.3f} ({held[-1][f'holdout_{task}_minus_mpc']:+.3f})"
            latest.append(text)
        title = task.capitalize() + (": " + "; ".join(latest) if latest else ": ще немає епізодів")
        top.set_title(title, loc="left", color=INK, fontsize=10)
        top.set_ylabel("score (останні епізоди)")
        bottom.axhline(0, color=INK2, lw=1)
        bottom.set_ylabel("run − MPC")
        bottom.set_xlabel("хвилини тренування")
        for ax in (top, bottom):
            ax.grid(color=GRID, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        if top.lines:
            top.legend(frameon=False, fontsize=8, loc="lower left")
    fig.canvas.draw_idle()


def main(
    runs: str | list | tuple | None = None,
    labels: str | list | tuple | None = None,
    interval: float = 15,
    png: str | None = None,
    theme: str = "dark",
) -> None:
    """Show ``runs`` (run folders, comma-separated; default: the newest train_ppo_residual run), redrawn every
    ``interval`` seconds; with ``png``, write one picture there instead of opening a window. ``theme``: dark or
    light."""
    global COLORS, INK, INK2, GRID, SURFACE, RED
    COLORS, (INK, INK2, GRID, SURFACE) = THEMES[theme]
    RED = HELD_OUT[theme]
    if runs is None:
        folders = [newest()]
    else:
        folders = [Path(r) for r in (runs.split(",") if isinstance(runs, str) else runs)]
    folders = [f if f.is_absolute() else ROOT / f for f in folders]
    if labels is None:
        names = [f"{f.parent.name} {f.name}" for f in folders]
    else:
        names = labels.split(",") if isinstance(labels, str) else list(labels)
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
    fig, axes = plt.subplots(2, len(MAPS), figsize=(13, 7), sharex="col", gridspec_kw={"height_ratios": [3, 2]})
    fig.canvas.manager.set_window_title("Тренування: score проти MPC")
    draw(fig, axes, folders, names)
    fig.tight_layout()
    if png:
        fig.savefig(png, dpi=120)
        print(png)
        return
    timer = fig.canvas.new_timer(interval=int(interval * 1000))
    timer.add_callback(lambda: (draw(fig, axes, folders, names), fig.tight_layout()))
    timer.start()
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
