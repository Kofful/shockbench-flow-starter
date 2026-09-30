"""The CLI's local commands (evaluate, compare, check, pack), agent names and the shipped agents."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sbf_starter import agents, container, scoring
from sbf_starter import cli as sbf
from tests.conftest import ROOT, write_agent


TEMPLATE = ROOT / "agents" / "template"
SHIPPED = ("template", "random", "heuristic")

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


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_agent_passes_check_by_name(name, capsys):
    sbf.check(name)
    out = capsys.readouterr().out
    assert "a timed run in an isolated process" in out and "all checks passed" in out and "WARNING" not in out


def test_pack_by_name_writes_under_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dest = sbf.pack("heuristic")
    assert (tmp_path / "outputs" / "heuristic.zip").is_file() and dest.endswith("heuristic.zip")
    assert "next: uv run sbf upload outputs/heuristic.zip" in capsys.readouterr().out


def test_names_resolve_to_agents_folders(tmp_path):
    assert agents.resolve("template") == ROOT / "agents" / "template"
    folder = write_agent(tmp_path / "elsewhere", send_max())
    assert agents.resolve(folder) == folder and agents.resolve(str(folder / "agent.py")) == folder / "agent.py"
    with pytest.raises(FileNotFoundError):
        agents.resolve("no_such_agent")
    assert callable(agents.load("random")) and callable(agents.load(TEMPLATE / "agent.py"))


def test_heuristic_reads_params_json(tmp_path):
    """A params.json beside the heuristic's agent.py replaces its numbers (the policy search writes one)."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
    from shockbench_flow_gym import agent_config_from_reset

    folder = tmp_path / "half"
    folder.mkdir()
    shutil.copy(ROOT / "agents" / "heuristic" / "agent.py", folder / "agent.py")
    (folder / "params.json").write_text(json.dumps({"fraction": 0.5, "closure_power": 0.0}))
    env = gym.make("ShockBench/Tiny-v0")
    obs, info = env.reset(options={"episode": 0})
    config = agent_config_from_reset(env, obs, info)
    half, full = agents.load(folder)(config), agents.load("template")(config)
    np.testing.assert_allclose(half.act(obs)["flows"], 0.5 * full.act(obs)["flows"])


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
    """Red team: agent.py imports its own features.py, which imports numba (installed here, not on the server)."""
    folder = write_agent(
        tmp_path / "helper_pandas", send_max("import features  # noqa: F401"), **{"features.py": "import numba\n"}
    )
    with pytest.raises(SystemExit) as exc:
        sbf.check(str(folder))
    out = capsys.readouterr().out
    assert exc.value.code == sbf.FAILED
    assert "WARNING: features.py imports numba" in out and "No module named 'numba'" in out


def test_check_passes_a_module_of_the_submission_own(tmp_path, capsys):
    """Red team: a submission's own helper.py beside agent.py must be the one imported."""
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


def test_evaluate_and_compare_quick(tmp_path, capsys):
    sbf.evaluate("heuristic", quick=True, n_jobs=1, out=str(tmp_path / "score.json"))
    out = capsys.readouterr().out
    assert "heuristic on tiny, 4 episodes" in out and "Score:" in out and "quick:" in out
    assert json.loads((tmp_path / "score.json").read_text())["episodes"] == 4
    sbf.compare("heuristic", str(TEMPLATE / "agent.py"), quick=True, episodes=[0, 1], n_jobs=1)
    out = capsys.readouterr().out
    assert "A: heuristic" in out and "A - B:" in out and "paired interval" in out


def test_quick_means_one_thing():
    from shockbench_flow_agent import QUICK_EPISODES

    es = scoring.episode_set("tiny", "dev", quick=True, n_jobs=1, verbose=False)
    assert es.episodes == tuple(range(QUICK_EPISODES)) and es.fq_replications == 2 and es.cut_draws == 0
    naive, oracle = [r["J_naive_cents"] for r in es.references], [r["J_oracle_cents"] for r in es.references]
    assert es.rss(naive)["rss"] == 0.0 and es.rss(oracle)["rss"] == 1.0


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
    """The installed `sbf` script finds agents by name from any directory."""
    exe = Path(sys.executable).parent / "sbf"
    args = [str(exe), "check", "template", "--timing=False"]
    proc = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all checks passed" in proc.stdout
