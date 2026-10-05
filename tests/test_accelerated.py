"""Fast-evaluation option routing and validation (rollout equivalence is exercised manually/in integration)."""

import os

import pytest

from sbf_starter import scoring
from sbf_starter.accelerated import DEVICE_ENV, check_batch_size, check_device, evaluation_device


class _EpisodeSet:
    def __init__(self):
        self.call = None

    def score(self, agent, **kwargs):
        self.call = (agent, kwargs)
        return "score"


def test_evaluation_jobs_reach_agent_play(monkeypatch):
    episodes = _EpisodeSet()
    agent = object()
    monkeypatch.setattr(scoring, "episode_set", lambda *args, **kwargs: episodes)

    assert scoring.evaluate(agent, n_jobs=7, batch_size=1, device="cpu") == "score"
    assert episodes.call == (agent, {"name": None, "cpu_budget": False, "n_jobs": 7})


def test_device_is_scoped_to_evaluation(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, "cpu")
    with evaluation_device("cuda:2"):
        assert os.environ[DEVICE_ENV] == "cuda:2"
    assert os.environ[DEVICE_ENV] == "cpu"


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "CUDA:3"])
def test_device_values(device):
    assert check_device(device) == device.lower()


@pytest.mark.parametrize("device", ["", "gpu", "cuda:x", 0])
def test_bad_device_values(device):
    with pytest.raises(ValueError, match="device must"):
        check_device(device)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_bad_batch_sizes(size):
    with pytest.raises(ValueError, match="batch_size"):
        check_batch_size(size)
