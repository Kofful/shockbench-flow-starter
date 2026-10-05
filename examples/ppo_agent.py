"""The PPO submission's agent.py (05_train_ppo.py copies it beside policy.pt): the server has torch, not SB3.

Local evaluation uses CUDA when available and batches live episodes through ``Agent.act_batch``.  The scoring server
has CPU Torch, so the same submission automatically falls back to CPU there.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import torch


def _device() -> torch.device:
    requested = os.environ.get("SBF_EVAL_DEVICE", "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested through SBF_EVAL_DEVICE={requested!r}, but Torch cannot use CUDA")
    return device


# Loaded at import, which the CPU budget does not meter. A CUDA-capable local Torch keeps both weights and batched
# observations on the GPU; the server's CPU build follows the same code with DEVICE=cpu.
DEVICE = _device()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # torch 2.14 marks TorchScript deprecated; it still loads
    POLICY = torch.jit.load(str(Path(__file__).resolve().parent / "policy.pt"), map_location=DEVICE)
POLICY.eval()
KEYS = list(POLICY.obs_keys)  # the observation fields of the flat vector, in order


class Agent:
    def __init__(self, config):
        u0 = config["static"]["edges"]["u0"]  # each edge's nominal capacity per week
        self.capacity = np.array([u0[e] for e in config["static"]["action_slots"]["edge"]], dtype=float)

    def act(self, observation):
        x = np.concatenate([np.asarray(observation[k], dtype=np.float64).ravel() for k in KEYS])
        with torch.inference_mode():
            fraction = POLICY(torch.as_tensor(x, device=DEVICE)).cpu().numpy()  # of capacity, in [0, 1]
        return {"flows": fraction * self.capacity * observation["action_mask"]}

    @staticmethod
    def act_batch(agents, observations):
        """One device forward pass for a lock-step batch of independently seeded episodes."""
        x = np.stack(
            [np.concatenate([np.asarray(obs[k], dtype=np.float64).ravel() for k in KEYS]) for obs in observations]
        )
        with torch.inference_mode():
            fractions = POLICY(torch.as_tensor(x, device=DEVICE)).cpu().numpy()
        return [
            {"flows": fraction * agent.capacity * obs["action_mask"]}
            for agent, obs, fraction in zip(agents, observations, fractions, strict=True)
        ]
