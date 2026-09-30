"""The CLI's local commands (evaluate, compare, check, pack, runs, version, fields), the agents and the layout."""

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sbf_starter import agents, container, scoring, search
from sbf_starter import cli as sbf
from tests.conftest import ROOT


TEMPLATE = ROOT / "src" / "sbf_starter" / "agents" / "template"


def write_agent(folder, source: str, **files: str):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agent.py").write_text(textwrap.dedent(source))
    for name, text in files.items():
        (folder / name).write_text(textwrap.dedent(text))
    return folder


SEND_MAX = """
    import numpy as np
    {extra}

    class Agent:
        def __init__(self, config=None):
            u0 = config["static"]["edges"]["u0"]
            self.cap = np.array([u0[e] for e in config["static"]["action_slots"]["edge"]], dtype=float)

        def act(self, observation):
            return {{"flows": self.cap * observation["action_mask"] * {scale}}}
    """


def send_max(extra: str = "", scale: str = "1.0") -> str:
    return SEND_MAX.format(extra=extra, scale=scale)


def test_pack_is_repeatable_and_checked(tmp_path, capsys):
    a = sbf.pack(str(TEMPLATE), out=str(tmp_path / "a.zip"))
    b = sbf.pack(str(TEMPLATE), out=str(tmp_path / "b.zip"))
    assert hashlib.sha256(open(a, "rb").read()).digest() == hashlib.sha256(open(b, "rb").read()).digest()
    assert "OK: the scorer's checks pass" in capsys.readouterr().out


@pytest.mark.parametrize("name", agents.SHIPPED)
def test_every_shipped_agent_passes_check_by_name(name, capsys):
    sbf.check(name)
    out = capsys.readouterr().out
    assert "a timed run in an isolated process" in out and "all checks passed" in out and "WARNING" not in out


def test_pack_by_name_writes_under_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dest = sbf.pack("heuristic")
    assert (tmp_path / "outputs" / "heuristic.zip").is_file() and dest.endswith("heuristic.zip")
    assert "next: uv run sbf check outputs/heuristic.zip" in capsys.readouterr().out


def test_names_resolve_in_agents_first(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "YOURS_DIR", tmp_path)
    assert agents.resolve("template") == agents.SHIPPED_DIR / "template"
    write_agent(tmp_path / "template", send_max(scale="0.5"))  # yours shadows the shipped one of the same name
    write_agent(tmp_path / "mine", send_max())
    assert agents.resolve("template") == tmp_path / "template" and agents.resolve("mine") == tmp_path / "mine"
    assert list(agents.names()) == ["mine", "template", "random", "heuristic"]
    assert agents.resolve(TEMPLATE) == TEMPLATE
    with pytest.raises(FileNotFoundError):
        agents.resolve("no_such_agent")
    assert callable(agents.load("random")) and callable(agents.load(TEMPLATE / "agent.py"))


def test_heuristic_is_the_search_template_rendered():
    assert (agents.SHIPPED_DIR / "heuristic" / "agent.py").read_text() == search.heuristic_source()


def test_check_refuses_a_folder_without_agent(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "README.txt").write_text("no agent here\n")
    with pytest.raises(SystemExit) as exc:
        sbf.check(str(tmp_path / "empty"))
    assert exc.value.code == sbf.REFUSED
    assert "REFUSED" in capsys.readouterr().out


def test_check_fails_an_import_the_image_lacks(tmp_path, capsys):
    folder = write_agent(tmp_path / "imports", send_max("import pandas_that_does_not_exist  # not in the image"))
    with pytest.raises(SystemExit) as exc:
        sbf.check(str(folder))
    out = capsys.readouterr().out
    assert exc.value.code == sbf.FAILED
    assert "WARNING: agent.py imports pandas_that_does_not_exist" in out and "does not import" in out


def test_check_fails_a_helper_module_importing_what_the_image_lacks(tmp_path, capsys):
    """Red team: agent.py imports its own features.py, which imports pandas (installed here, not on the server)."""
    folder = write_agent(
        tmp_path / "helper_pandas", send_max("import features  # noqa: F401"), **{"features.py": "import pandas\n"}
    )
    with pytest.raises(SystemExit) as exc:
        sbf.check(str(folder))
    out = capsys.readouterr().out
    assert exc.value.code == sbf.FAILED
    assert "WARNING: features.py imports pandas" in out and "No module named 'pandas'" in out


def test_check_passes_a_module_named_like_a_kit_package(tmp_path, capsys):
    """Red team: a submission's own helper.py must be the one imported, not this repository's helper package."""
    folder = write_agent(
        tmp_path / "own_helper", send_max("from helper import SCALE"), **{"helper.py": "SCALE = 1.0\n"}
    )
    sbf.check(str(folder))
    out = capsys.readouterr().out
    assert "all checks passed" in out and "WARNING" not in out


def test_check_fails_an_import_of_this_repository(tmp_path, capsys):
    """Red team: agent.py importing sbf_starter (installed here) must fail, not warn and pass."""
    folder = write_agent(tmp_path / "kit_import", send_max("import sbf_starter  # noqa: F401"))
    with pytest.raises(SystemExit) as exc:
        sbf.check(str(folder))
    out = capsys.readouterr().out
    assert exc.value.code == sbf.FAILED
    assert "WARNING: agent.py imports sbf_starter" in out and "does not import" in out


def test_check_without_timing_still_fails_a_fatal_import(tmp_path, capsys):
    folder = write_agent(tmp_path / "kit_import", send_max("import sbf_starter  # noqa: F401"))
    with pytest.raises(SystemExit) as exc:
        sbf.check(str(folder), timing=False)
    assert exc.value.code == sbf.FAILED


def test_check_catches_a_crashing_agent(tmp_path, capsys):
    folder = write_agent(
        tmp_path / "bad",
        """
        import scipy  # in the scoring image

        class Agent:
            def __init__(self, config=None):
                pass

            def act(self, observation):
                return {"flows": observation[0]}
        """,
    )
    with pytest.raises(SystemExit) as exc:
        sbf.check(str(folder))
    out = capsys.readouterr().out
    assert exc.value.code == sbf.FAILED
    assert "WARNING: act indexes observation[0]" in out
    assert "FAILED: 26 of 26 weeks would be played by naive" in out and "KeyError" in out


def test_version_agrees_with_the_vendored_wheel(capsys):
    sbf.version()
    out = capsys.readouterr().out
    assert container.vendored_wheel().name in out and "ok:" in out


def test_evaluate_and_compare_quick(tmp_path, capsys):
    sbf.evaluate("heuristic", quick=True, n_jobs=1, out=str(tmp_path / "score.json"))
    out = capsys.readouterr().out
    assert "heuristic on tiny, 4 episodes" in out and "Score:" in out and "quick:" in out
    assert json.loads((tmp_path / "score.json").read_text())["episodes"] == 4
    sbf.compare("heuristic", str(TEMPLATE / "agent.py"), quick=True, episodes=[0, 1], n_jobs=1)
    out = capsys.readouterr().out
    assert "A: heuristic" in out and "A - B:" in out and "paired interval" in out


def test_quick_means_one_thing():
    es = scoring.episode_set("tiny", "dev", quick=True, n_jobs=1, verbose=False)
    assert es.episodes == tuple(range(scoring.QUICK_EPISODES)) and es.fq_replications == 2 and es.cut_draws == 0
    naive, oracle = [r["J_naive_cents"] for r in es.references], [r["J_oracle_cents"] for r in es.references]
    assert es.rss(naive)["rss"] == 0.0 and es.rss(oracle)["rss"] == 1.0


def test_runs_table(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    sbf.runs()
    assert "no runs under outputs/" in capsys.readouterr().out
    folder = tmp_path / "outputs" / "06_policy_search" / "baseline"
    folder.mkdir(parents=True)
    doc = {"script": "06_policy_search", "run_name": "baseline", "task": "tiny", "started": "2026-10-01T10:00:00"}
    (folder / "meta.json").write_text(json.dumps(doc | {"shockbench_flow": "0.1.1", "results": {"score": 0.5}}))
    sbf.runs()
    out = capsys.readouterr().out
    assert "06_policy_search" in out and "baseline" in out and "score 0.5000" in out


@pytest.mark.parametrize("task", ["tiny", "small", "full"])
def test_field_tables_match_the_installed_wheel(task, tmp_path):
    """docs/fields/<task>.md is what `sbf fields` prints for the vendored wheel (regenerate after a new wheel)."""
    out = tmp_path / f"{task}.md"
    sbf.fields(task, out=str(out))
    assert out.read_text() == (ROOT / "docs" / "fields" / f"{task}.md").read_text()


def test_readme_lists_every_example():
    readme = (ROOT / "README.md").read_text()
    for path in sorted((ROOT / "scripts" / "python").glob("[0-9][0-9]_*.py")):
        assert f"(scripts/python/{path.name})" in readme, path.name


PARTICIPANT_DOCS = ["README.md", "AGENTS.md", "docs/GUIDE.md", "CHANGELOG.md"]
INTERNAL = [
    "§",
    "(55)",
    "(56)",
    "D9",
    "F_Q",
    "RSS_G",
    "docs/evidence",
    "vm_run",
    "SBF_VM_HOST",
    "LOKY",
    "M5 gate",
    "AMD",
]


@pytest.mark.parametrize(
    "path",
    PARTICIPANT_DOCS
    + ["docs/fields/tiny.md"]
    + [
        str(p.relative_to(ROOT))
        for d in ("src/sbf_starter", "scripts/python", "configs")
        for p in (ROOT / d).rglob("*")
        if p.suffix in (".py", ".yaml")
    ],
)
def test_no_organiser_internals(path):
    text = (ROOT / path).read_text()
    assert not [w for w in INTERNAL if w in text], path


@pytest.mark.parametrize(
    "path, ignored",
    [
        ("outputs/x/meta.json", True),
        ("weights/model.pt", True),
        ("data/raw.csv", True),
        ("trash/t.py", True),
        ("agents/mine/weights/model.pt", False),
        ("agents/mine/data/table.npz", False),
        ("agents/mine/policy.onnx", False),
        ("agents/mine/__pycache__/agent.cpython-313.pyc", True),
        ("my/scripts/outputs_note.py", False),
        ("my/data/cache.npz", False),
    ],
)
def test_gitignore_is_anchored(path, ignored):
    res = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT, check=False)
    assert (res.returncode == 0) == ignored, path


@pytest.mark.docker
@pytest.mark.skipif(not container.docker_available(), reason="no Docker daemon")
def test_check_in_the_container(capsys):
    sbf.check(str(TEMPLATE), docker=True)
    out = capsys.readouterr().out
    assert "0 week(s) the scorer would give to naive" in out


@pytest.mark.docker
@pytest.mark.skipif(not container.docker_available(), reason="no Docker daemon")
def test_container_catches_an_import_the_image_lacks(tmp_path):
    folder = write_agent(tmp_path / "pandas_agent", send_max("import pandas  # not in the scoring image"))
    (row,) = container.play(folder, "small", 1, echo=lambda s: None)
    assert len(row["substitutions"]) == row["weeks"]  # every week naive's: agent.py never imported


@pytest.mark.docker
@pytest.mark.skipif(not container.docker_available(), reason="no Docker daemon")
def test_container_meters_cpu(tmp_path):
    folder = write_agent(
        tmp_path / "busy_agent",
        """
        import time

        import numpy as np

        class Agent:
            def __init__(self, config=None):
                self.n = len(config["static"]["action_slots"]["edge"])

            def act(self, observation):
                if int(observation["week"][0]) == 2:  # week 2 burns more than the 2 s budget
                    t = time.process_time()
                    while time.process_time() - t < 3.0:
                        pass
                return {"flows": np.zeros(self.n)}
        """,
    )
    (row,) = container.play(folder, "small", 1, echo=lambda s: None)
    assert [2, "cpu"] in row["substitutions"] and row["meter_errors"] == []


def test_console_script_runs(tmp_path):
    """`uv run sbf ...` is the console script of [project.scripts]; it runs from any working directory."""
    exe = Path(sys.executable).parent / "sbf"
    proc = subprocess.run([str(exe), "version"], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok: up to date" in proc.stdout
