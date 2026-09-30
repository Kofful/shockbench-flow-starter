"""Agent names, resolved to submission folders: yours in ``agents/`` first, then the three the organisers ship.

A submission folder holds ``agent.py`` at its root, which is what a submission zip contains. Yours live in the
top-level ``agents/`` (``make new-agent NAME=mine`` copies the template there); the organisers' live here:

- ``template``: send the maximum (every open route ships its capacity); start your own agent from it;
- ``random``: random flows up to each route's capacity (a lower bar);
- ``heuristic``: send the maximum, less into a strait that is partly closed (``sbf_starter.search`` renders it).

Every ``agent.py`` uses only numpy (what the scoring image has), so the folder can be checked and packed as it is:
``uv run sbf check template``. A name given to ``sbf`` or to an example (``agent=heuristic``) is looked up in
``agents/``, then here, when it is not a path.
"""

from pathlib import Path

from sbf_starter import ROOT


SHIPPED_DIR = Path(__file__).resolve().parent  # the organisers' agents
YOURS_DIR = ROOT / "agents"  # yours
SHIPPED = ("template", "random", "heuristic")


def resolve(agent: str | Path) -> Path:
    """A path as given when it exists, else ``agents/<agent>``, else the shipped agent of that name.

    Raises:
        FileNotFoundError: when it is none of them.

    """
    path = Path(agent)
    if path.exists():
        return path
    for base in (YOURS_DIR, SHIPPED_DIR):
        named = base / str(agent)
        if (named / "agent.py").is_file():
            return named
    raise FileNotFoundError(f"{agent}: no such folder, zip or file, and no agent of that name in agents/ or shipped")


def names() -> dict[str, Path]:
    """Every agent a name resolves to: yours (``agents/``), then the shipped ones not shadowed by yours."""
    yours = {p.parent.name: p.parent for p in sorted(YOURS_DIR.glob("*/agent.py"))}
    return yours | {n: SHIPPED_DIR / n for n in SHIPPED if n not in yours}


def load(agent: str | Path):
    """The ``Agent`` class of a named agent, a submission folder or an ``agent.py`` file, loaded as the scorer does."""
    from shockbench_flow_agent import load_agent_class

    path = resolve(agent)
    return load_agent_class(path, f"agent_{path.resolve().parent.name if path.is_file() else path.resolve().name}")
