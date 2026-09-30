"""Shared fixtures. Every test uses a private reference cache, never ~/.cache/shockbench-flow."""

import os
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODABENCH_VARS = ("CODABENCH_TOKEN", "CODABENCH_COMPETITION")


def write_agent(folder: Path, source: str, **files: str) -> Path:
    """A submission folder: ``source`` as agent.py, and ``files`` (name -> text) beside it."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agent.py").write_text(textwrap.dedent(source))
    for name, text in files.items():
        (folder / name).write_text(textwrap.dedent(text))
    return folder


def pytest_addoption(parser):
    parser.addoption("--docker", action="store_true", help="also run the tests that build and run the container")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--docker"):
        return
    skip = pytest.mark.skip(reason="builds the scoring container: run with --docker")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> Path:
    """One per session, so each worker computes the references once."""
    return tmp_path_factory.mktemp("sbf-cache")


@pytest.fixture(autouse=True)
def _private_cache(cache_dir, monkeypatch):
    monkeypatch.setenv("SBF_CACHE_DIR", str(cache_dir))
    for var in CODABENCH_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def env_with_cache(cache_dir) -> dict:
    """For an example run in a subprocess: the private cache, one BLAS thread, no Codabench variables."""
    env = dict(os.environ)
    env.update({"SBF_CACHE_DIR": str(cache_dir), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    env["MPLBACKEND"] = "Agg"
    for var in CODABENCH_VARS:
        env.pop(var, None)
    return env
