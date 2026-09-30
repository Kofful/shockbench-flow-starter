"""Play a submission in a local copy of the scoring container, with the scorer's CPU meter (``sbf check --docker``).

``build_image`` builds ``docker/policy/Dockerfile`` from a context of only that file, its ``requirements.txt``, the
kit's launcher and the kit's packages unpacked from the vendored wheel: the organisers' policy image, but built here.
``play`` then runs dev episodes the way the Codabench ingestion does (``shockbench_flow.hosting.codabench``): one
container per episode with the scorer's flags (no network, read-only file system, one CPU, 4 GB, an unprivileged user),
the wire's 10 s wall clock per week and 60 s start-up, and the week rule of the CPU meter
(``shockbench_flow.hosting.metering.WeekCpu``) read from Docker's own statistics. A week over the task's CPU budget
is reported as the scorer would substitute it (``cpu``).

The CPU seconds measured here are this machine's, not the scoring host's (a Linux x86_64 server). On an ARM machine
(Apple silicon) the image runs under x86_64 emulation, which is slower, so a week near its budget here may be fine
there; a week far over it will not be. The costs are not reported: the episode runs without naive's fallback.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

from sbf_starter import ROOT, cpu_budget_s


DOCKER_DIR = ROOT / "docker" / "policy"
TAG = "sbf-starter-policy:local"
KIT_PACKAGES = ("shockbench_flow", "shockbench_flow_agent")  # what the scoring image holds of the package
LAUNCHER = "shockbench_flow/hosting/launch.py"
DEADLINE_S, STARTUP_S = 10.0, 60.0  # the wall clock per week and to start (docs/GUIDE.md, "Rules")
EPISODE_S = {"small": 180.0, "full": 480.0}  # an episode's container stops this long after it starts
MAX_REPLY_BYTES = 1 << 20  # the local evaluation's reply cap (shockbench_flow_agent.local_eval.SubmissionRunConfig)


def vendored_wheel() -> Path:
    wheels = sorted((ROOT / "vendor").glob("shockbench_flow-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError(f"expected one shockbench-flow wheel in {ROOT / 'vendor'}, found {len(wheels)}")
    return wheels[0]


def _context(wheel: Path) -> bytes:
    """The build context as a tar: Dockerfile, requirements.txt, launch.py and lib/<the kit's packages>."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar, zipfile.ZipFile(wheel) as whl:
        for name in ("Dockerfile", "requirements.txt"):
            tar.add(DOCKER_DIR / name, arcname=name)

        def add(arcname: str, data: bytes) -> None:
            info = tarfile.TarInfo(arcname)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))

        add("launch.py", whl.read(LAUNCHER))
        for member in whl.namelist():
            top = member.split("/", 1)[0]
            if top in KIT_PACKAGES and not member.endswith("/") and "__pycache__" not in member:
                add(f"lib/{member}", whl.read(member))
    return buf.getvalue()


def docker_task(task: str) -> str:
    """The hosted network a check of ``task`` plays in the container: full for full, else small (tiny is not hosted)."""
    return "full" if task == "full" else "small"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
    return r.returncode == 0


def build_image(tag: str = TAG, echo=print) -> str:
    """Build the policy image (module docstring) and return its ID; the first build downloads torch (about 200 MB)."""
    if not docker_available():
        raise RuntimeError("docker is not available: install Docker (or OrbStack, Colima) and start its daemon")
    echo(f"building {tag} from docker/policy/Dockerfile and {vendored_wheel().name} (the first build takes minutes)")
    t = time.perf_counter()
    subprocess.run(
        ["docker", "build", "-q", "--platform", "linux/amd64", "-t", tag, "-"],
        input=_context(vendored_wheel()),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    image = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag], capture_output=True, text=True, check=True
    ).stdout.strip()
    echo(f"  image {image[:19]} in {time.perf_counter() - t:.0f} s")
    return image


def docker_socket() -> str:
    """The Docker daemon's unix socket (``DOCKER_HOST=unix://...``, else /var/run/docker.sock)."""
    host = os.environ.get("DOCKER_HOST", "")
    if host.startswith("unix://"):
        return host[len("unix://") :]
    return "/var/run/docker.sock"


def play(submission_dir: Path, task: str = "small", episodes: int = 1, image: str | None = None, echo=print):
    """Play dev ``episodes`` of ``task`` (small or full) in the container; one dict per episode (module docstring)."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.hosting import metering
    from shockbench_flow.hosting.docker import DockerTransport
    from shockbench_flow.hosting.tasks import scenario, task_generator
    from shockbench_flow.information.runner import play_wire_episode
    from shockbench_flow.information.wire import WireLimits
    from shockbench_flow.marks import compute_marks

    if task not in EPISODE_S:
        raise ValueError(f"the container plays the hosted networks {list(EPISODE_S)}, not {task!r}")
    budget = cpu_budget_s(task)
    image = image or build_image(echo=echo)
    inst, _params = task_generator(task)
    out = []
    for n in range(episodes):
        omega = scenario(task, n)
        work = Path(tempfile.mkdtemp(prefix="sbf-docker-"))
        cid = work / "cid"

        def resolve(cid=cid):
            return metering.cpu_reader("docker_stats", metering.read_cidfile(cid), socket_path=docker_socket())

        meter = metering.WeekCpu(resolve, budget_s=budget, poll_s=metering.POLL_S["docker_stats"])
        transport = DockerTransport(
            image,
            submission_dir,
            deadline_s=DEADLINE_S,
            startup_s=STARTUP_S,
            max_reply_bytes=MAX_REPLY_BYTES,
            episode_timeout_s=EPISODE_S[task],
            stderr=None,  # the agent's stderr (tracebacks, prints) shows in this terminal
            cidfile=cid,
        )
        transport.interrupt, transport.interrupt_poll_s = meter.interrupt, meter.poll_s
        t = time.perf_counter()
        try:
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
                meter=meter,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
        weeks = [w for w in meter.weeks if w is not None]
        row = {
            "episode": n,
            "task": task,
            "weeks": inst.T,
            "ready_s": transport.ready_s,
            "wall_s": time.perf_counter() - t,
            "week1_cpu_s": meter.weeks[0] if meter.weeks else None,
            "median_cpu_s": meter.summary()["median_s"],
            "max_cpu_s": max(weeks) if weeks else None,
            "cpu_budget_s": budget,
            "substitutions": [[int(w), str(c)] for w, c in wired.substitutions],
            "meter_errors": list(meter.errors),
            "watchdog_fired": transport.watchdog_fired,
        }
        out.append(row)
    return out
