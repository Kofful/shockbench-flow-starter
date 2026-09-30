"""Random flows up to each allowed route's capacity: a lower bar, seeded from ``config["policy_seed"]``."""

import numpy as np


class Agent:
    def __init__(self, config=None):
        u0 = config["static"]["edges"]["u0"]  # each edge's nominal capacity per week
        self.cap = np.array([u0[e] for e in config["static"]["action_slots"]["edge"]], dtype=float)
        self.n_pairs = config["spaces"]["action"]["release_mode"]["shape"]
        self.rng = np.random.default_rng(config["policy_seed"])

    def act(self, observation):
        flows = self.rng.uniform(0.0, 1.0, self.cap.shape) * self.cap * observation["action_mask"]
        return {"flows": flows, "release_mode": np.zeros(self.n_pairs, dtype=np.int64)}  # override_qty may be omitted
