"""The run folders of the examples: ``outputs/<script>/<run_name>/`` with ``config.yaml`` and ``meta.json``.

Every Hydra entry point (the examples, and yours under ``my/scripts/``) runs in its own folder, named by the config
key ``run_name`` (default: the start time, ``configs/main.yaml``) under ``outputs/<script's name>/``
(``configs/hydra/default.yaml``). ``start`` writes there the resolved config and ``meta.json`` (the wheel's version,
the git commit, the task, the command); ``finish`` adds the run's results to ``meta.json``; ``sbf runs`` lists them.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from sbf_starter import ROOT


OUTPUTS = Path("outputs")  # relative to the working directory, as Hydra's run folder is


def _git(*args: str) -> str | None:
    try:
        res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _shown(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def shown(path: Path) -> str:
    """``path`` relative to the working directory when it lies below it (log lines stay short), else as it is."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def start(cfg) -> Path:
    """The run's folder (Hydra's output directory), with ``config.yaml`` and ``meta.json`` written in it."""
    import shockbench_flow
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    hc = HydraConfig.get()
    out = Path(hc.runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    task = cfg.get("task")
    meta = {
        "script": hc.job.name,
        "run_name": out.name,
        "task": task.get("name") if task is not None else None,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "shockbench_flow": shockbench_flow.__version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "command": " ".join(["python", _shown(Path(sys.argv[0])), *sys.argv[1:]]),
        "results": {},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    from helper.logging import logger

    logger.info("run folder: {}", shown(out))
    return out


def finish(out: Path, **results) -> None:
    """Add ``results`` (plain JSON values: a score, a path) to the run's ``meta.json``."""
    path = Path(out) / "meta.json"
    meta = json.loads(path.read_text())
    meta["results"] |= results
    meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(meta, indent=1) + "\n")


def table(root: Path = OUTPUTS) -> list[dict]:
    """Every run under ``root`` (newest first): its ``meta.json`` with the folder."""
    rows = []
    for path in root.glob("*/*/meta.json"):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rows.append(meta | {"folder": path.parent.as_posix()})
    return sorted(rows, key=lambda r: r.get("started") or "", reverse=True)
