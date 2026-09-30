"""The checks behind ``sbf check``: every file's imports, then timed episodes in an isolated process.

``play_isolated`` runs the agent with only the submission and the scoring image's packages on its path, so a package
installed here cannot hide a missing import. CPU times are this machine's (``sbf check --docker`` is closer).
"""

from __future__ import annotations

import statistics
from pathlib import Path


def import_warnings(root: Path) -> list[str]:
    """One line per import the scoring image lacks, in every ``.py`` of the folder (read, never imported)."""
    from shockbench_flow_agent.submission import missing_imports

    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    out = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        for name in missing_imports((root / rel).read_bytes(), files):
            out.append(
                f"{rel} imports {name}, which the scoring image does not have (Python's standard library, numpy, "
                "scipy and torch only): if agent.py imports it, naive plays every week there"
            )
    return out


def timed_run(root: Path, task: str, episodes: int = 1) -> list[dict]:
    from shockbench_flow_agent import play_isolated

    return [play_isolated(Path(root).resolve(), task, n) for n in range(episodes)]


def summary(row: dict) -> dict:
    """Week 1 includes ``Agent(config)``."""
    cpu = [c for c in row["cpu_s"] if c is not None]
    return {
        "week1_s": cpu[0] if cpu else 0.0,
        "median_s": statistics.median(cpu) if cpu else 0.0,
        "max_s": max(cpu) if cpu else 0.0,
        "over": list(row["over_budget"]),
    }


def last_error(stderr: str, lines: int = 12) -> str:
    tail = [line for line in stderr.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(tail)
