"""The PPO submission's agent.py (05_train_ppo.py copies it beside policy.pt): the server has torch, not SB3."""

import warnings
from pathlib import Path

import numpy as np
import torch


# loaded at import, which the CPU budget does not meter
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # torch 2.14 marks TorchScript deprecated; it still loads
    POLICY = torch.jit.load(str(Path(__file__).resolve().parent / "policy.pt"), map_location="cpu")
POLICY.eval()
KEYS = list(POLICY.obs_keys)  # the observation fields of the flat vector, in order


class Agent:
    def __init__(self, config):
        u0 = config["static"]["edges"]["u0"]  # each edge's nominal capacity per week
        self.capacity = np.array([u0[e] for e in config["static"]["action_slots"]["edge"]], dtype=float)

    def act(self, observation):
        x = np.concatenate([np.asarray(observation[k], dtype=np.float64).ravel() for k in KEYS])
        with torch.inference_mode():
            fraction = POLICY(torch.from_numpy(x)).numpy()  # of capacity, in [0, 1]
        return {"flows": fraction * self.capacity * observation["action_mask"]}
