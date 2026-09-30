"""The checks behind ``sbf check``: every file's imports, then a timed run in a process holding only the server's.

1. ``import_warnings`` reads every ``.py`` of the submission (nothing is imported) and names each import the scoring
   image lacks. A warning is not yet a failure: a module that ``agent.py`` never imports, or an import inside a
   ``try`` that catches ``ImportError``, is harmless.
2. ``timed_run`` plays dev episodes with the agent in a child process started as ``python -I -S -B``, whose path holds
   the submission's folder, the Python standard library and a folder of links to the packages of the scoring image
   (``IMAGE_DISTRIBUTIONS``: numpy, SciPy, torch and what torch needs) and to the agent kit, nothing else. So a
   package of this repository (``helper``, ``sbf_starter``, pandas, ...) cannot hide a missing import or shadow one of
   the submission's own modules (a ``helper.py`` beside ``agent.py`` is the one imported). The child speaks to this
   process over the scorer's own wire (``SubprocessTransport``, 10 s per week, 60 s to start), and times
   ``Agent(config)`` and every ``act`` in CPU seconds. An ``agent.py`` that does not import there is the server's
   fatal case: naive plays every week and the score is 0.

A local run is a guide to the server's CPU meter, not its count: ``sbf check --docker`` meters a copy of the scoring
container. Torch is linked only when it is installed here (``uv sync --extra rl``); without it an agent that imports
torch fails here but not on the server, and the report says so.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import tempfile
from importlib import metadata, util
from pathlib import Path

from sbf_starter import ROOT
from sbf_starter.container import DEADLINE_S, MAX_REPLY_BYTES, STARTUP_S


REQUIREMENTS = ROOT / "docker" / "policy" / "requirements.txt"  # the scoring image's packages, pinned by hash
KIT_PACKAGES = ("shockbench_flow", "shockbench_flow_agent")  # what the scoring image holds of the wheel
# the child: the kit's shim with Agent(config) and each act timed in CPU seconds, one JSON line per week to a file
CHILD = """
import json, sys, time
lib, timing, sub = sys.argv[1:4]
sys.path.insert(0, lib)
import shockbench_flow_agent.shim as shim
reset, act, out = shim.AgentShim.reset, shim.AgentShim.act, open(timing, "w", buffering=1)
def timed_reset(self, *args, **kwargs):
    t = time.process_time()
    try:
        return reset(self, *args, **kwargs)
    finally:
        self._sbf_week1 = time.process_time() - t
def timed_act(self, obs):
    t = time.process_time()
    try:
        return act(self, obs)
    finally:
        used = time.process_time() - t + getattr(self, "_sbf_week1", 0.0)
        self._sbf_week1 = 0.0
        out.write(json.dumps([int(obs["week"]), used]) + "\\n")
shim.AgentShim.reset, shim.AgentShim.act = timed_reset, timed_act
from shockbench_flow_agent.__main__ import main
sys.exit(main([sub]))
"""


def import_warnings(root: Path) -> list[str]:
    """One line per import the scoring image lacks, in every ``.py`` of the submission folder ``root``."""
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


def image_distributions() -> list[str]:
    """The distributions the scoring image installs (``docker/policy/requirements.txt``), as named there."""
    names = []
    for line in REQUIREMENTS.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(==|@)", line)
        if m:
            names.append(m.group(1))
    return names


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def image_lib(dest: Path) -> tuple[Path, list[str]]:
    """``dest`` filled with links to the image's packages and the kit, as installed here; and the ones missing here."""
    wanted = {_norm(d) for d in image_distributions()}
    installed = metadata.packages_distributions()  # top-level module -> the distributions that provide it
    tops = [t for t, dists in installed.items() if any(_norm(d) in wanted for d in dists)]
    found = set()
    dest.mkdir(parents=True, exist_ok=True)
    for top in sorted(set(tops) | set(KIT_PACKAGES)):
        spec = util.find_spec(top)
        if spec is None:
            continue
        if spec.submodule_search_locations:
            src = Path(next(iter(spec.submodule_search_locations)))
        elif spec.origin and spec.origin not in ("built-in", "frozen"):
            src = Path(spec.origin)
        else:
            continue
        (dest / src.name).symlink_to(src)
        libs = src.parent / f"{top}.libs"  # the shared libraries a Linux wheel keeps beside its package
        if libs.is_dir() and not (dest / libs.name).exists():
            (dest / libs.name).symlink_to(libs)
        for d in installed.get(top, []):
            found.add(_norm(d))
    return dest, sorted(wanted - found)


def _episode(root: Path, task: str, n: int, lib: Path, work: Path) -> dict:
    """One dev episode of ``task`` with the agent in the isolated child: timings, fallback weeks, the first error."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.hosting.tasks import scenario, task_generator
    from shockbench_flow.information.runner import SubprocessTransport, play_wire_episode
    from shockbench_flow.information.wire import WireLimits
    from shockbench_flow.marks import compute_marks

    inst, _params = task_generator(task)
    omega = scenario(task, n)
    timing, errlog, home = work / f"timing-{n}.jsonl", work / f"stderr-{n}.txt", work / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "OMP_NUM_THREADS": "1"}
    env |= {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with errlog.open("wb") as err:
        transport = SubprocessTransport(
            [sys.executable, "-I", "-S", "-B", "-c", CHILD, str(lib), str(timing), str(root)],
            deadline_s=DEADLINE_S,
            startup_s=STARTUP_S,
            max_reply_bytes=MAX_REPLY_BYTES,
            env=env,
            stderr=err.fileno(),
        )
        wired = play_wire_episode(
            Env(fallback=None),
            inst,
            transport,
            limits=WireLimits(max_reply_bytes=MAX_REPLY_BYTES, bank_seconds=60.0),
            regime="standard",
            omega=omega,
            policy_seed=0,
            marks=compute_marks(inst, omega),
            policy_name="submission",
        )
    weeks = [json.loads(line) for line in timing.read_text().splitlines()] if timing.exists() else []  # none: no import
    return {
        "episode": n,
        "weeks": inst.T,
        "cpu": [s for _w, s in weeks],
        "substitutions": [(int(w), str(c)) for w, c in wired.substitutions],
        "imported": bool(weeks),
        "stderr": errlog.read_text(errors="replace"),
    }


def timed_run(root: Path, task: str, episodes: int = 1) -> list[dict]:
    """Dev episodes 0..episodes-1 of ``task`` with the submission in ``root`` in the isolated child (module docstring).

    Each row: ``weeks``, ``cpu`` (CPU seconds per week played, week 1 with ``Agent(config)``), ``substitutions``
    ((week, cause) the server would give to naive: ``action`` for an exception or a malformed action, ``timeout``,
    ``killed`` when the process was gone), ``imported`` (whether ``agent.py`` imported), ``stderr`` (the agent's).
    """
    with tempfile.TemporaryDirectory(prefix="sbf-check-run-") as tmp:
        work = Path(tmp)
        lib, _missing = image_lib(work / "lib")
        return [_episode(Path(root).resolve(), task, n, lib, work) for n in range(episodes)]


def missing_here() -> list[str]:
    """The scoring image's packages for agents (numpy, scipy, torch) not installed here (torch without the rl extra)."""
    from shockbench_flow_agent.submission import IMAGE_PACKAGES

    return [name for name in IMAGE_PACKAGES if util.find_spec(name) is None]


def summary(row: dict, budget_s: float) -> dict:
    """Week 1's, the median and the largest CPU seconds of a row, and the weeks over ``budget_s``."""
    cpu = row["cpu"]
    return {
        "week1_s": cpu[0] if cpu else None,
        "median_s": statistics.median(cpu) if cpu else None,
        "max_s": max(cpu) if cpu else None,
        "over": [i + 1 for i, c in enumerate(cpu) if c > budget_s],
    }


def last_error(stderr: str, lines: int = 12) -> str:
    """The end of the agent's stderr (a traceback, when it raised)."""
    tail = [line for line in stderr.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(tail)


__all__ = ["import_warnings", "last_error", "missing_here", "summary", "timed_run"]
