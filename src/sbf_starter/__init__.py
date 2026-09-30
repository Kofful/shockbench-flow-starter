"""The starter kit's importable code: the organisers' agents, the `sbf` tooling, scoring helpers, search and PPO.

- ``sbf_starter.agents``: agent names resolved to folders (yours in ``agents/`` first, then the shipped ``template``,
  ``random`` and ``heuristic``), and loading an ``Agent`` class;
- ``sbf_starter.cli``: the ``sbf`` command line (evaluate, compare, check, pack, upload, status, runs, fields, version);
- ``sbf_starter.scoring``: ``evaluate`` and ``compare`` on cached references (the wheel's ``EpisodeSet``), shared by
  ``sbf`` and the examples, and the one meaning of ``quick``;
- ``sbf_starter.check``: the checks of ``sbf check`` (every file's imports, a timed run in an isolated process);
- ``sbf_starter.codabench``: the Codabench client behind ``upload`` and ``status``;
- ``sbf_starter.container``: a local copy of the scoring container (``sbf check --docker``);
- ``sbf_starter.play``: play an ``Agent`` class under gymnasium as the scorer does;
- ``sbf_starter.runs``: the run folders of the examples (``outputs/<script>/<run>/``) and ``sbf runs``;
- ``sbf_starter.search``: the pieces of the policy search (``scripts/python/06_policy_search.py``);
- ``sbf_starter.ppo``: PPO training and its TorchScript export (needs the ``rl`` extra).

The tasks are the files of ``configs/task/`` (their gymnasium ids) and the CPU budgets are the wheel's
(``shockbench_flow_agent.scoring.CPU_BUDGET_S``); nothing else lists them.
"""

from functools import cache
from pathlib import Path

import rootutils


try:
    from importlib.metadata import version

    __version__ = version("shockbench-flow-starter")
except Exception:
    # Fallback for a checkout that is not installed
    __version__ = "0.0.0.dev0"

ROOT = Path(rootutils.find_root(search_from=__file__, indicator=".project-root"))  # the repository's root
DEFAULT_TASK = "tiny"  # the default network of every example and of `sbf` (configs/main.yaml says the same)


@cache
def tasks() -> dict[str, dict]:
    """Every task by name, in the wheel's order (tiny, small, full): its ``configs/task/<name>.yaml`` as a dict."""
    from omegaconf import OmegaConf
    from shockbench_flow_agent.scoring import CPU_BUDGET_S

    out = {}
    for name in CPU_BUDGET_S:
        path = ROOT / "configs" / "task" / f"{name}.yaml"
        out[name] = OmegaConf.to_container(OmegaConf.load(path)) | {"cpu_budget_s": CPU_BUDGET_S[name]}
    return out


def check_task(task: str) -> str:
    """``task`` when it names a task; else ValueError listing them."""
    if task not in tasks():
        raise ValueError(f"task must be one of {list(tasks())}, got {task!r}")
    return task


def env_id(task: str) -> str:
    """The gymnasium id of a task (``ShockBench/Tiny-v0``, ...)."""
    return tasks()[check_task(task)]["env_id"]


def cpu_budget_s(task: str) -> float:
    """The CPU seconds per week a task's board allows (the wheel's ``CPU_BUDGET_S``)."""
    return tasks()[check_task(task)]["cpu_budget_s"]
