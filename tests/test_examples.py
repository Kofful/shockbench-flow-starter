"""Every Hydra entry point runs on Tiny with small budgets (the smoke version of its default run) in its run folder."""

import importlib.util
import json
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_dir

from tests.conftest import ROOT


SCRIPTS = sorted((ROOT / "scripts" / "python").glob("[0-9][0-9]_*.py"))


def run(args: list[str], env: dict, cwd, timeout: float = 300) -> str:
    """Run an entry point in ``cwd`` (its outputs/ goes there); its output (stdout and the log) when it exits 0."""
    proc = subprocess.run(
        [sys.executable, *args, "run_name=test"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode == 0, f"{args} exited {proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc.stdout + proc.stderr


def script(name: str) -> str:
    return str(ROOT / "scripts" / "python" / name)


def meta(tmp_path, name: str) -> dict:
    folder = tmp_path / "outputs" / name / "test"
    assert (folder / "config.yaml").is_file()
    return json.loads((folder / "meta.json").read_text())


def test_every_entry_point_has_its_config():
    assert len(SCRIPTS) >= 7
    for path in SCRIPTS:
        assert (ROOT / "configs" / f"{path.stem}.yaml").is_file(), path.name


@pytest.mark.parametrize("task", ["tiny", "small", "full"])
@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_configs_compose_with_each_task(path, task):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=path.stem, overrides=[f"task={task}"])
    assert cfg.task.name == task and cfg.task.env_id.startswith("ShockBench/")
    assert cfg.seed == 0 and "run_name" in cfg
    assert not {"fq_replications", "cut_draws", "naive_replications", "out_dir"} & set(cfg)  # plain settings only


def test_the_search_path_holds_both_config_folders(tmp_path):
    """A config dir of one's own composes `main` and the task group (the plugin appends configs/ and my/configs/)."""
    from hydra.core.config_search_path import ConfigSearchPath

    from hydra_plugins.sbf_starter_searchpath import StarterSearchPath

    class Recorder(ConfigSearchPath):
        def __init__(self):
            self.paths = []

        def get_path(self):
            return self.paths

        def append(self, provider, path, anchor=None):
            self.paths.append(path)

        def prepend(self, provider, path, anchor=None):
            self.paths.insert(0, path)

    rec = Recorder()
    StarterSearchPath().manipulate_search_path(rec)
    assert rec.paths == [f"file://{ROOT / 'configs'}", f"file://{ROOT / 'my' / 'configs'}"]
    (tmp_path / "mine.yaml").write_text("defaults:\n  - main\n  - _self_\n\nsteps: 3\n")
    with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
        cfg = compose(config_name="mine", overrides=["task=small"])
    assert cfg.steps == 3 and cfg.task.env_id == "ShockBench/Small-v0"


def test_quickstart(env_with_cache, tmp_path):
    out = run([script("01_quickstart.py")], env_with_cache, tmp_path)
    assert "26 weeks" in out and "options={'episode': 0}" in out
    m = meta(tmp_path, "01_quickstart")
    assert m["task"] == "tiny" and m["shockbench_flow"] and m["results"]["cost_usd"] > 0


def test_play_agents(env_with_cache, tmp_path):
    out = run([script("02_play_agents.py"), "episodes=1"], env_with_cache, tmp_path)
    assert "random" in out and "template" in out and "USD" in out


def test_heuristic_agent(env_with_cache, tmp_path):
    out = run([script("03_heuristic_agent.py"), "episodes=1"], env_with_cache, tmp_path)
    assert "a strait falls below 50% open: [29]" in out and "the rule saves" in out
    assert meta(tmp_path, "03_heuristic_agent")["results"]["saved_usd"] > 0  # the rule acts where the strait closes


def test_evaluate_and_compare(env_with_cache, tmp_path):
    out = run([script("04_evaluate.py"), "quick=true", "n_jobs=1"], env_with_cache, tmp_path)
    assert "Score:" in out and "interval" in out and "quick:" in out
    agent_file = ROOT / "src" / "sbf_starter" / "agents" / "heuristic" / "agent.py"
    args = [f"agent={agent_file}", "against=template", "quick=true", "n_jobs=1"]
    out = run([script("04_evaluate.py"), *args], env_with_cache, tmp_path)
    assert "A - B:" in out and "paired interval" in out
    assert json.loads((tmp_path / "outputs" / "04_evaluate" / "test" / "result.json").read_text())["b"]["rss"]


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="the rl extra is not installed")
def test_train_ppo(env_with_cache, tmp_path):
    args = ["total_timesteps=416", "n_envs=1", "n_scenarios=2", "check_episodes=1", "activation=relu", "net_arch=[16]"]
    out = run([script("05_train_ppo.py"), *args], env_with_cache, tmp_path)
    folder = tmp_path / "outputs" / "05_train_ppo" / "test"
    assert (folder / "submission.zip").is_file() and (folder / "submission" / "policy.pt").is_file()
    assert (
        "max_flow_difference" in out and "POLICY = torch.jit.load" in (folder / "submission" / "agent.py").read_text()
    )


def test_policy_search(env_with_cache, tmp_path):
    args = ["generations=1", "population=2", "train_episodes=2", "quick=true", "n_jobs=1"]
    out = run([script("06_policy_search.py"), *args], env_with_cache, tmp_path)
    assert "held out, on 4 dev episodes" in out and "A - B:" in out
    m = meta(tmp_path, "06_policy_search")
    best = tmp_path / "outputs" / "06_policy_search" / "test" / "best" / "agent.py"
    assert best.is_file() == m["results"]["written"]  # written only when it beats send-the-maximum held out
    if m["results"]["written"]:
        assert m["results"]["holdout_score"] > m["results"]["holdout_start"] and "FRACTION = [" in best.read_text()


def test_dashboard(env_with_cache, tmp_path):
    run([script("07_dashboard.py"), "episode=29", "quick=true", "n_jobs=1"], env_with_cache, tmp_path)
    for name in ("network.png", "dashboard_max.png", "episode_max.gif", "record_random.npz"):
        assert (tmp_path / "outputs" / "07_dashboard" / "test" / name).is_file(), name
