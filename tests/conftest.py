"""Shared fixtures: a private cache directory, so tests never touch ~/.cache/shockbench-flow, and the repo root."""

import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


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
    """One cache for the whole session (naive's quantiles are computed once per worker at 2 replications)."""
    return tmp_path_factory.mktemp("sbf-cache")


@pytest.fixture(autouse=True)
def _private_cache(cache_dir, monkeypatch):
    monkeypatch.setenv("SBF_CACHE_DIR", str(cache_dir))
    for var in (
        "CODABENCH_TOKEN",
        "CODABENCH_USERNAME",
        "CODABENCH_PASSWORD",
        "CODABENCH_COMPETITION",
        "CODABENCH_URL",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def env_with_cache(cache_dir) -> dict:
    """The environment of an entry point run in a subprocess: the private cache, one BLAS thread, plain logs."""
    env = dict(os.environ)
    env.update(
        {
            "SBF_CACHE_DIR": str(cache_dir),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "COLORIZE": "false",
            "LOG_LEVEL": "INFO",
        }
    )
    for var in ("CODABENCH_TOKEN", "CODABENCH_USERNAME", "CODABENCH_PASSWORD"):
        env.pop(var, None)
    return env


@pytest.fixture
def log_messages():
    """The messages logged through helper.logging's logger while the test runs (INFO and above)."""
    from helper.logging import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="INFO")
    yield messages
    logger.remove(sink)
