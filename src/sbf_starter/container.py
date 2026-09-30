"""``sbf check --docker``: episodes in a local copy of the scoring container, with the server's CPU meter.

The wheel builds the image from its own files (``build_image``) and plays each episode as the server does
(``play_container``). The image is linux/amd64: on Apple silicon it runs under emulation, slower than the server.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def docker_task(task: str) -> str:
    """Tiny is not hosted, so a check of tiny plays small in the container."""
    return "full" if task == "full" else "small"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
    return r.returncode == 0


def play(submission_dir: Path, task: str = "small", episodes: int = 1, echo=print) -> list[dict]:
    from shockbench_flow_agent import build_image, play_container

    if not docker_available():
        raise RuntimeError("docker is not available: install Docker (or OrbStack, Colima) and start its daemon")
    echo("building the scoring image (the first build downloads torch and SciPy: some hundreds of MB)")
    image = build_image()
    return [play_container(Path(submission_dir).resolve(), image, task, n) for n in range(episodes)]
