"""The starter kit's command line: score, compare, check, pack, upload and follow a submission.

Run from the repository root: ``uv run sbf <command> --help`` (``make evaluate``, ``make check`` ... run the common
ones). An AGENT is a submission folder (``agent.py`` at its root), a zip, an ``agent.py`` file (``evaluate`` and
``compare``), or a name: a folder of ``agents/`` (yours), else a shipped agent (``template``, ``random``,
``heuristic``). Every command defaults to the Tiny network (``--task=small`` for the public board's).

    evaluate AGENT       the local score (0 = naive rule, 1 = clairvoyant plan) on the dev episodes, with an interval
    compare A B          two agents on the same episodes: the difference and its paired interval
    check AGENT          the scorer's checks, every file's imports, and a timed run in an isolated process
                         (--docker: the real container with the scorer's CPU meter)
    pack FOLDER          the submission zip (repeatable bytes), checked
    upload ZIP           send a zip to the competition on Codabench with your own account (--dry_run to rehearse)
    status [ID]          your submissions on Codabench and their scores (--wait to follow one)
    runs                 the example runs under outputs/ (script, run, task, result)
    fields               every observation and action field of a network (shape, dtype, meaning), as Markdown
    version              the installed shockbench-flow against the wheel this repository vendors

``evaluate`` and ``compare`` share ``--episodes`` (dev, a count, or a list), ``--quick`` (seconds: a rough naive rule
and no harm levels, not the leaderboard's numbers; the examples' ``quick=true`` means the same), ``--entropy`` (a root
of your own) and ``--cpu_budget`` (weeks over the task's CPU budget go to the naive rule, as on the server).

Codabench credentials are read from the environment only (CODABENCH_TOKEN, or CODABENCH_USERNAME and
CODABENCH_PASSWORD; ``main`` also reads a ``.env`` file, see .env.example) and never printed or stored; the competition
is --competition or CODABENCH_COMPETITION (its id or URL). See sbf_starter/codabench.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import fire

from helper.display import DisplayConsole
from sbf_starter import DEFAULT_TASK, check_task, cpu_budget_s
from sbf_starter.agents import resolve


REFUSED, FAILED = 2, 1
OUT_DIR = Path("outputs")  # where ``pack`` writes a zip by default (gitignored)
_console = DisplayConsole()


def _say(*parts) -> None:
    """One line on stdout, as it is: no markup, no highlighting, no wrapping (the lines hold paths and brackets)."""
    _console.print(*parts, markup=False, highlight=False, emoji=False, soft_wrap=True)


def _indent(text: str) -> str:
    return "  " + text.strip().replace("\n", "\n  ")


def _path(path: str) -> Path:
    """A folder or zip as given, or a named agent's folder; a missing path is reported as the scorer would."""
    try:
        return resolve(path)
    except FileNotFoundError:
        return Path(path)


def _as_zip(path: Path, work: Path) -> Path:
    """``path`` itself when it is a zip; a folder zipped into ``work`` as the kit zips it."""
    from shockbench_flow_agent.submission import build_submission

    if path.is_dir():
        return Path(build_submission(path, work / f"{path.resolve().name or 'submission'}.zip"))
    if path.is_file():
        return path
    raise SystemExit(f"{path}: no such folder or zip, and no agent of that name")


def _validate(zip_path: Path):
    """The trusted runner's check of the zip; prints the verdict; returns the Submission or None when refused."""
    from shockbench_flow_agent.submission import REFUSAL_FIXES, SubmissionError, check_zip

    try:
        sub = check_zip(zip_path)
    except SubmissionError as err:
        _say(f"REFUSED: {err}")
        if err.code in REFUSAL_FIXES:
            _say(f"  what to fix: {REFUSAL_FIXES[err.code]}")
        return None
    _say(
        f"OK: the scorer's checks pass: {len(sub.files)} file(s), {sub.zip_bytes:,} bytes zipped, "
        f"{sub.total_bytes:,} unpacked"
    )
    _say(f"  sha256 {sub.sha256} (the submission id; it salts config['policy_seed'])")
    return sub


def _static(root: Path) -> tuple[list[str], list[str]]:
    """(warnings, the fatal ones): agent.py's mistakes, and every file's imports the scoring image lacks."""
    from shockbench_flow_agent.submission import agent_warnings

    from sbf_starter.check import import_warnings

    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    own = [w for w in agent_warnings((root / "agent.py").read_bytes(), files) if not w.startswith("agent.py imports")]
    imports = import_warnings(root)
    return own + imports, [w for w in imports if w.startswith("agent.py imports")]


def check(
    path: str,
    task: str = DEFAULT_TASK,
    episodes: int = 1,
    timing: bool = True,
    docker: bool = False,
    docker_episodes: int = 1,
) -> None:
    """Check a submission as the scorer does, then time it in a process holding only the server's packages.

    Args:
        path: a folder with agent.py at its root, a submission zip, or an agent's name.
        task: the network of the timed run: tiny (the default), small (the public board's) or full (the private's).
        episodes: dev episodes of the timed run.
        timing: play the agent in an isolated process (only the submission and the scoring image's packages on its
            path) and time Agent(config) and each act in CPU seconds (False: the static checks only).
        docker: also build the scoring container locally (Docker needed; the first build downloads about 200 MB) and
            play dev episodes in it with the scorer's CPU meter: the imports, the read-only file system and the
            time rules are the server's.
        docker_episodes: episodes of the container run (on small, or full with --task=full).

    Exit status: 0 when every check passes, 2 when the scorer would refuse the zip, 1 when the agent would not run
    there (an import the scoring image lacks, an exception, a malformed action, a week over the wall clock).

    """
    from shockbench_flow_agent.submission import extract_submission

    from sbf_starter import check as checks

    check_task(task)
    budget = cpu_budget_s(task)
    with tempfile.TemporaryDirectory(prefix="sbf-check-") as tmp:
        work = Path(tmp)
        zip_path = _as_zip(_path(path), work)
        _say(f"1. the scorer's checks of {zip_path.name}, and every file's imports (nothing is imported)")
        if _validate(zip_path) is None:
            sys.exit(REFUSED)
        root = extract_submission(zip_path, work / "sub").root
        warnings, fatal = _static(root)
        for w in warnings:
            _say(f"WARNING: {w}")
        ok = not (fatal and not timing)
        if timing:
            _say(
                f"2. a timed run in an isolated process (only the submission and the scoring image's packages): "
                f"{episodes} dev episode(s) of {task}, CPU seconds per week (budget {budget:g} s)"
            )
            missing = checks.missing_here()
            if missing:
                _say(f"  NOTE: not installed here, so not importable in this run: {missing} (uv sync --extra rl)")
            for r in checks.timed_run(root, task, episodes):
                if not r["imported"]:
                    _say("FAILED: agent.py does not import with the scoring image's packages: on the server naive")
                    _say("  plays every week and the score is 0. The agent's error:")
                    _say(_indent(checks.last_error(r["stderr"])))
                    ok = False
                    break
                s = checks.summary(r, budget)
                over = f", over the budget in weeks {s['over']}" if s["over"] else ""
                _say(
                    f"  episode {r['episode']}: week 1 (Agent(config) + first act) {s['week1_s']:.3f} s, median act "
                    f"{s['median_s']:.4f} s, max {s['max_s']:.3f} s{over}"
                )
                if s["over"]:
                    _say(
                        "  WARNING: over the budget on this machine; the scorer meters its own (a Linux x86_64 server, "
                        "one CPU): check with --docker, and keep a margin"
                    )
                if r["substitutions"]:
                    ok = False
                    subs = r["substitutions"]
                    _say(f"FAILED: {len(subs)} of {r['weeks']} weeks would be played by naive: {subs[:10]}")
                    _say("  (action: an exception or a malformed action; timeout: no reply in 10 s). The error:")
                    _say(_indent(checks.last_error(r["stderr"])))
        if docker and ok:
            from sbf_starter import container

            _say(f"3. the scoring container, {docker_episodes} dev episode(s) of {container.docker_task(task)}")
            rows = container.play(root, container.docker_task(task), docker_episodes, echo=lambda s: _say("  " + s))
            for r in rows:
                subs = r["substitutions"]
                _say(
                    f"  episode {r['episode']} ({r['task']}, {r['weeks']} weeks): ready in {r['ready_s'] or 0:.1f} s, "
                    f"week 1 {r['week1_cpu_s'] or 0:.3f} s CPU, median {r['median_cpu_s'] or 0:.3f} s, max "
                    f"{r['max_cpu_s'] or 0:.3f} s (budget {r['cpu_budget_s']:g} s); {len(subs)} week(s) the scorer "
                    "would give to naive" + (f": {subs[:10]}" if subs else "")
                )
                if r["meter_errors"]:
                    _say(f"  NOTE: the CPU meter could not read the container: {r['meter_errors'][:2]}")
                ok = ok and not subs
        _say("all checks passed" if ok else "a check failed: see above")
        if not ok:
            sys.exit(FAILED)


def pack(folder: str, out: str | None = None, compress: bool = False) -> str:
    """Zip a submission folder (agent.py at its root, and any files it loads) with repeatable bytes, then check it.

    Args:
        folder: the submission folder, or an agent's name (agents/<name>, or a shipped one).
        out: the zip to write (default: outputs/<folder's name>.zip).
        compress: deflate the members (for large weights; the default stores them).

    """
    from shockbench_flow_agent.submission import build_submission

    src = _path(folder)
    if not (src / "agent.py").is_file():
        raise SystemExit(f"{src}/agent.py does not exist: the zip needs agent.py at its root")
    dest = Path(out) if out else OUT_DIR / f"{src.resolve().name}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    build_submission(src, dest, compress=compress)
    _say(f"written {dest}")
    if _validate(dest) is None:
        sys.exit(REFUSED)
    try:
        shown = dest.resolve().relative_to(Path.cwd())
    except ValueError:
        shown = dest
    _say(f"next: uv run sbf check {shown}   then   uv run sbf upload {shown}")
    return str(dest)


def _scored(result, quick: bool, out: str | None) -> None:
    import json

    from sbf_starter.scoring import QUICK_NOTE, as_dict

    _say(str(result))
    if quick:
        _say(QUICK_NOTE)
    if out:
        Path(out).write_text(json.dumps(as_dict(result), indent=1) + "\n")
        _say(f"written {out}")


def evaluate(
    path: str,
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool = False,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """The local score of an agent on the dev episodes (0 = naive rule, 1 = clairvoyant plan), with a 90 % interval.

    Args:
        path: an agent's name, a submission folder or zip (checked by the scorer's validator first), or an agent.py
            file (played in this process, where a debugger works).
        task: tiny (the default), small (the public board's network) or full (the private board's).
        episodes: dev (20 public episodes, 5 per harm level), a count k (episodes 0..k-1) or a list.
        quick: seconds instead of minutes: a rough naive rule, no harm levels, the first 4 episodes for dev; not the
            leaderboard's numbers.
        entropy: 0, the public dev episodes; any other integer, a root of your own (tune there, confirm on dev).
        cpu_budget: hand a week over the task's CPU budget to the naive rule, as the server does (this machine's CPU).
        n_jobs: workers of a first run's reference computation (-1: all cores).
        out: write the result as JSON there.

    The first run on a network computes the naive rule's and the clairvoyant plan's costs and caches them
    (``~/.cache/shockbench-flow`` or ``SBF_CACHE_DIR``): a minute or two on tiny, longer on small and full.

    """
    from sbf_starter.scoring import evaluate as score

    _scored(score(path, task, episodes, quick=quick, entropy=entropy, cpu_budget=cpu_budget, n_jobs=n_jobs), quick, out)


def compare(
    a: str,
    b: str,
    task: str = DEFAULT_TASK,
    episodes: str | int | list[int] = "dev",
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool = False,
    n_jobs: int = -1,
    out: str | None = None,
) -> None:
    """Two agents on the same episodes: A's score minus B's, with a paired 90 % interval (the arguments of evaluate).

    A paired interval resamples both agents on the same episodes, so the noise they share cancels: when it holds 0,
    these episodes cannot tell the two apart.
    """
    from sbf_starter.scoring import compare as cmp

    result = cmp(a, b, task, episodes, quick=quick, entropy=entropy, cpu_budget=cpu_budget, n_jobs=n_jobs)
    _scored(result, quick, out)


def _client(competition: str | int | None = None):
    """The client of the Codabench server that hosts the competition.

    The server is a competition URL's own host, else ``CODABENCH_URL``, else www.codabench.org. Credentials go only to
    that host.
    """
    import urllib.parse

    from sbf_starter import codabench

    value = str(competition or os.environ.get("CODABENCH_COMPETITION") or "").strip()
    parsed = urllib.parse.urlparse(value)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme in ("http", "https") and parsed.netloc else None
    configured = (os.environ.get("CODABENCH_URL") or "").rstrip("/") or None
    if origin and configured and origin != configured:
        raise SystemExit(f"the competition URL is on {origin} but CODABENCH_URL is {configured}: unset one of them")
    creds = codabench.Credentials.from_env()
    return codabench.Client(origin or configured or codabench.DEFAULT_URL, creds)


def _competition(competition: str | int | None):
    from sbf_starter import codabench

    value = competition or os.environ.get("CODABENCH_COMPETITION")
    if not value:
        raise SystemExit("name the competition: --competition=<id or URL>, or CODABENCH_COMPETITION in the environment")
    return codabench.competition_id(value)


def upload(
    zip_path: str,
    competition: str | None = None,
    phase: str | None = None,
    dry_run: bool = False,
    wait: bool = False,
    poll_s: float = 30.0,
) -> None:
    """Upload a submission zip to the competition with your Codabench account (the web page's upload, scripted).

    Args:
        zip_path: the zip (``pack`` writes one; a folder is refused: pack it first).
        competition: the competition's id or URL (default CODABENCH_COMPETITION).
        phase: the phase's name on Codabench (default: the one open now).
        dry_run: check everything (the zip, your login, registration, the phase, your daily slots) and upload nothing.
        wait: follow the submission until it is scored, then print its score.
        poll_s: seconds between two status reads while waiting (at least 10).

    Each upload spends one of the phase's daily submissions (3 per UTC day); nothing is ever retried.

    """
    from sbf_starter import codabench

    path = Path(zip_path)
    if not path.is_file() or path.suffix != ".zip":
        raise SystemExit(f"{path}: not a zip file (pack a folder first: uv run sbf pack <folder>)")
    _say(f"1. the scorer's checks of {path.name}")
    if _validate(path) is None:
        sys.exit(REFUSED)
    pk, secret = _competition(competition)
    client = _client(competition)
    try:
        _say(f"2. Codabench at {client.base_url}, competition {pk}, credentials from {client.credentials.describe()}")
        comp, ph = codabench.preflight(client, pk, secret, phase)
        used = ph.get("used_submissions_per_day")
        limit = ph.get("max_submissions_per_day")
        _say(f"  phase {ph.get('name')!r} (id {ph['id']}) is open; submissions today: {used} of {limit}")
        if dry_run:
            _say("dry run: nothing uploaded")
            return
        _say("3. uploading")
        sub = codabench.upload(client, path, comp, ph)
        _say(f"  submission {sub.get('id')} created, status {sub.get('status')}")
        if wait:
            done = codabench.wait(client, sub["id"], poll_s, echo=lambda s: _say("  " + s))
            _print_submission(done)
        else:
            _say(f"follow it: uv run sbf status {sub.get('id')} --wait")
    except codabench.CodabenchError as err:
        _say(f"ERROR: {err}")
        sys.exit(FAILED)


def _print_submission(sub: dict) -> None:
    from sbf_starter import codabench

    sc = codabench.scores(sub)
    shown = ", ".join(f"{k} {v:.4f}" if "rss" in k else f"{k} {v:g}" for k, v in sc.items()) or "no score"
    when = sub.get("created_when") or sub.get("submitted_at") or ""
    _say(f"  {sub.get('id')}: {sub.get('status')} {when} {sub.get('filename') or ''}  {shown}".rstrip())
    if sub.get("status") == "Failed" and sub.get("status_details"):
        _say(f"    {sub['status_details']}")


def status(
    submission: int | None = None,
    competition: str | None = None,
    phase: str | None = None,
    wait: bool = False,
    poll_s: float = 30.0,
) -> None:
    """One submission's status and score, or (no id) your submissions to the phase.

    Args:
        submission: a submission id (``upload`` prints it).
        competition: the competition's id or URL (default CODABENCH_COMPETITION), for the list.
        phase: the phase's name (default: the one open now), for the list.
        wait: follow the submission until it is Finished, Failed or Cancelled.
        poll_s: seconds between two reads while waiting (at least 10).

    """
    from sbf_starter import codabench

    client = _client(competition)
    try:
        if submission is not None:
            sub = codabench.wait(client, int(submission), poll_s) if wait else client.submission(int(submission))
            _print_submission(sub)
            return
        pk, secret = _competition(competition)
        ph = codabench.pick_phase(client.competition(pk, secret), phase)
        subs = client.submissions(ph["id"])
        _say(f"phase {ph.get('name')!r}: {len(subs)} submission(s)")
        for s in subs:
            _print_submission(s)
    except codabench.CodabenchError as err:
        _say(f"ERROR: {err}")
        sys.exit(FAILED)


def runs(limit: int = 20) -> None:
    """The runs of the examples (and of your scripts) under outputs/, newest first: script, run, task, results."""
    import pandas as pd

    from sbf_starter import runs as run_folders

    rows = run_folders.table()[:limit]
    if not rows:
        _say("no runs under outputs/ yet: every example writes outputs/<script>/<run_name>/")
        return

    def shown(results: dict) -> str:
        return ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in results.items())

    table = pd.DataFrame(
        {
            "script": [r.get("script") for r in rows],
            "run": [r.get("run_name") for r in rows],
            "task": [r.get("task") or "-" for r in rows],
            "wheel": [r.get("shockbench_flow") for r in rows],
            "commit": [(r.get("git_commit") or "-")[:8] + ("+" if r.get("git_dirty") else "") for r in rows],
            "results": [shown(r.get("results") or {}) or "-" for r in rows],
        }
    )
    _console.display_df_as_table(table, max_rows=len(rows), max_col_width=None, title="runs under outputs/")


def fields(task: str = DEFAULT_TASK, out: str | None = None) -> None:
    """Every observation and action field of a network, with its shape, dtype, index set and meaning (Markdown).

    Args:
        task: tiny, small or full.
        out: write the tables to this file instead of printing them.

    """
    import re

    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* ids
    from shockbench_flow_agent.spaces import markdown

    from sbf_starter import env_id
    from sbf_starter.play import make_agent

    env = gym.make(env_id(task))
    obs, info = env.reset(options={"episode": 0})
    config = make_agent(env, lambda c: c, obs, info)
    text = markdown(config, obs, grouped="lot_keys" in config["layout"])
    generator = "by `uv run python scripts/python/build_starter_kit.py`"
    text = text.replace(generator, f"by `uv run sbf fields --task={task}`")
    text = re.sub(r" \(design \u00a7[\d.]+\)|,? design \u00a7[\d.]+", "", text)  # the benchmark's design sections
    if out:
        Path(out).write_text(text + "\n")
        _say(f"written {out}")
    else:
        _say(text)


def version() -> None:
    """The installed shockbench-flow against the vendored wheel and uv.lock; exit 1 when they disagree."""
    import hashlib
    import importlib.metadata
    import re
    import tomllib
    import zipfile

    from sbf_starter import ROOT
    from sbf_starter.container import vendored_wheel

    wheel = vendored_wheel()
    with zipfile.ZipFile(wheel) as zf:
        meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        vendored = re.search(r"^Version: (.+)$", zf.read(meta).decode(), re.MULTILINE).group(1).strip()
    try:
        installed = importlib.metadata.version("shockbench-flow")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    entry = next((p for p in lock.get("package", []) if p.get("name") == "shockbench-flow"), {})
    locked = entry.get("version")
    digest = f"sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}"
    # a wheel of the vendor/ index is locked by its file name (a path source, by its hash)
    names = {Path(str(w.get("path") or w.get("url") or "")).name for w in entry.get("wheels", [])}
    same = wheel.name in names or digest in {w.get("hash") for w in entry.get("wheels", [])}
    _say(f"vendored wheel: {wheel.name} (version {vendored})")
    _say(f"locked in uv.lock: {locked}; installed: {installed}")
    problems = []
    if locked != vendored or not same:
        problems.append("uv.lock does not match the vendored wheel: run `make update` (or `uv lock` then `uv sync`)")
    if installed != vendored:
        problems.append("the installed version is not the vendored wheel's: run `make update` (it keeps the rl extra)")
    for p in problems:
        _say(f"PROBLEM: {p}")
    if problems:
        sys.exit(FAILED)
    _say("ok: up to date with this repository (new wheels are announced in CHANGELOG.md)")


COMMANDS = {
    "evaluate": evaluate,
    "compare": compare,
    "check": check,
    "pack": pack,
    "upload": upload,
    "status": status,
    "runs": runs,
    "fields": fields,
    "version": version,
}


def main() -> None:
    """The ``sbf`` console script: a ``.env`` in the working directory (or above) is read first, then Fire."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=True)
    fire.Fire(COMMANDS, name="sbf", serialize=lambda result: None)  # commands print their own results


if __name__ == "__main__":
    main()
