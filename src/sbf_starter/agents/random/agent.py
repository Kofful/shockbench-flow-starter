"""A random agent: uniform random flows up to each route's capacity, on the routes allowed this week (a lower bar).

It shows the submission format with randomness seeded from ``config["policy_seed"]``, as the rules ask: the same
submission on the same episode plays the same actions.
"""

import numpy as np


class Agent:
    def __init__(self, config=None):
        u0 = config["static"]["edges"]["u0"]  # each edge's nominal capacity per week
        self.cap = np.array([u0[e] for e in config["static"]["action_slots"]["edge"]], dtype=float)
        self.n_pairs = config["spaces"]["action"]["release_mode"]["shape"]
        self.rng = np.random.default_rng(config["policy_seed"])  # seed randomness from the policy seed

    def act(self, observation):
        flows = self.rng.uniform(0.0, 1.0, self.cap.shape) * self.cap * observation["action_mask"]
        return {"flows": flows, "release_mode": np.zeros(self.n_pairs, dtype=np.int64)}  # override_qty may be omitted
