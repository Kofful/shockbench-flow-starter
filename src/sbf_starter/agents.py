"""Agent names: ``mine`` means the folder ``agents/mine/``; an existing path is taken as it is."""

from pathlib import Path

from sbf_starter import ROOT


AGENTS_DIR = ROOT / "agents"


def resolve(agent: str | Path) -> Path:
    path = Path(agent)
    if path.exists():
        return path
    named = AGENTS_DIR / str(agent)
    if (named / "agent.py").is_file():
        return named
    raise FileNotFoundError(f"{agent}: no such folder, zip or file, and no agents/{agent}/agent.py")


def load(agent: str | Path):
    """The ``Agent`` class of a name, a folder or an agent.py, loaded as the scorer loads it."""
    from shockbench_flow_agent import load_agent_class

    path = resolve(agent)
    folder = path.resolve().parent if path.is_file() else path.resolve()
    return load_agent_class(path, f"agent_{folder.name}")
