"""Copy the modules of the installed shockbench-flow that `mpc_det` needs into agents/<mpc agent>/sbflow/, server-ready.

uv run python scripts/vendor_mpc.py

The server has numpy, scipy and torch only, so the MPC agents (``AGENTS``: each folder is its own zip) cannot import
shockbench_flow. This copies the import
closure of ``shockbench_flow.policies.mpc_det`` (MIT, its licence copied beside it) as the package ``sbflow`` and
patches the four imports the server lacks:

- ``loguru`` (package __init__, logging only): dropped;
- ``fastjsonschema`` (instance/io.py, the instance file's schema check): a stub that accepts every instance, since the
  agent's instance comes from the scorer's own ``config["static"]``;
- ``joblib`` (parallel.py, parallel loops): serial only;
- ``highspy`` (policies/lp_common.py, mpc_det's solver): unavailable; agents/mpc solves with scipy's ``linprog``.

Rerun it after `uv sync --upgrade-package shockbench-flow`: every patch asserts it still finds its text.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("mpc", "mpc_residual", "mpc_adaptive", "mpc_ppo", "mpc_route")  # every agent folder that imports sbflow gets its own copy
DEST = ROOT / "agents" / AGENTS[0] / "sbflow"
NAME = "sbflow"

PATCHES = {
    "__init__.py": [
        (
            re.compile(r"try:\n    from loguru import logger as _logger\n.*?del _logger\n", re.S),
            "# vendored: the loguru switch-off is dropped (the server has no loguru; nothing here logs)\n",
        ),
    ],
    "instance/io.py": [
        (
            "import fastjsonschema\n",
            "class fastjsonschema:  # vendored stub: no fastjsonschema on the server; the scorer's instance is valid\n"
            "    class JsonSchemaValueException(Exception):\n"
            "        pass\n"
            "\n"
            "    @staticmethod\n"
            "    def compile(*args, **kwargs):\n"
            "        return lambda raw: None\n",
        ),
        (
            "        _schema_validator()(raw)\n",
            "        pass  # vendored: no schema check (its JSON file is not copied; the scorer's instance is valid)\n",
        ),
    ],
    "parallel.py": [
        (
            "    from joblib import Parallel, delayed\n",
            "    return [fn(x) for x in items]  # vendored: serial only (the server has no joblib)\n",
        ),
    ],
    "policies/lp_common.py": [
        (
            "if TYPE_CHECKING:  # annotations only: the solving functions import highspy themselves "
            "(module docstring)\n"
            "    import highspy\n",
            "if TYPE_CHECKING:  # vendored: no highspy on the server; agents/mpc solves with scipy's linprog\n",
        ),
        (
            "    import highspy\n\n    return highspy\n",
            '    raise RuntimeError("vendored copy: no highspy on the server; solve with scipy\'s linprog")\n',
        ),
    ],
}


def closure() -> list[str]:
    """The shockbench_flow modules a fresh interpreter loads to import mpc_det (and the LP solve it calls)."""
    code = (
        "import sys, shockbench_flow.policies.mpc_det, shockbench_flow.oracle.lp\n"
        "names = sorted(m for m in sys.modules if m.partition('.')[0] == 'shockbench_flow')\n"
        "print('\\n'.join(names))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    return out.split()


def main() -> None:
    import shockbench_flow

    src_root = Path(shockbench_flow.__file__).resolve().parent
    if DEST.exists():
        shutil.rmtree(DEST)
    files = []
    for module in closure():
        path = Path(*module.split(".")[1:])
        src = src_root / path / "__init__.py" if (src_root / path).is_dir() else src_root / path.with_suffix(".py")
        files.append(src.relative_to(src_root))
    for rel in sorted(set(files)):
        text = (src_root / rel).read_text(encoding="utf-8")
        text = re.sub(r"\bshockbench_flow\b", NAME, text)
        for old, new in PATCHES.get(rel.as_posix(), []):
            if isinstance(old, re.Pattern):
                text, n = old.subn(new, text)
            else:
                n = text.count(old)
                text = text.replace(old, new)
            if n != 1:
                raise SystemExit(f"{rel}: a patch matched {n} times, not once: the package changed; update PATCHES")
        out = DEST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    for name in PATCHES:
        if not (DEST / name).is_file():
            raise SystemExit(f"{name} is not in the closure any more: update PATCHES")
    dist = next(src_root.parent.glob("shockbench_flow-*.dist-info"))
    licence = next((dist / "licenses").glob("LICENSE*"), None) if (dist / "licenses").is_dir() else None
    if licence is not None:
        shutil.copy(licence, DEST / "LICENSE")
    (DEST / "VENDORED.md").write_text(
        f"Copied from {dist.name.removesuffix('.dist-info')} by scripts/vendor_mpc.py ({len(set(files))} modules, "
        f"renamed shockbench_flow -> {NAME}, patched: {', '.join(PATCHES)}). Do not edit: rerun the script.\n",
        encoding="utf-8",
    )
    for agent in AGENTS[1:]:
        other = ROOT / "agents" / agent / "sbflow"
        if other.parent.is_dir():
            if other.exists():
                shutil.rmtree(other)
            shutil.copytree(DEST, other)
    where = ", ".join(f"agents/{a}/sbflow" for a in AGENTS if (ROOT / "agents" / a).is_dir())
    print(f"copied {len(set(files))} modules of {dist.name.removesuffix('.dist-info')} to {where}")


if __name__ == "__main__":
    main()
