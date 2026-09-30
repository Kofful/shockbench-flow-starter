"""Your submission: this file is ``agent.py`` at the root of the zip.

The scorer builds ``Agent(config)`` once per episode and calls ``act(observation)`` once per week. As it stands, this
is the "send the maximum" baseline: every route that is not prohibited this week ships its nominal capacity. Replace
the body of ``act`` (and add what you need to ``__init__``) with your own policy.

The rules that bite (docs/GUIDE.md, "Rules"): only Python's standard library, numpy, scipy and torch (CPU) exist on
the scorer; files you ship beside this one load relative to it (``HERE / "weights.npz"``), best at module level, which
the CPU budget does not meter; read every shape from ``config["spaces"]``; seed randomness from
``config["policy_seed"]``. A week in which ``act`` raises, runs over its CPU budget or returns a malformed action is
played by the naive rule instead of yours.

Check it: ``uv run sbf check <this folder or its name>``.
"""

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent  # load your files relative to this one


class Agent:
    def __init__(self, config=None):
        static = config["static"]  # the network's tables: nodes, edges, lanes, commodities, action slots, ...
        action = config["spaces"]["action"]  # the shape of each part of the action
        u0 = static["edges"]["u0"]  # each edge's nominal capacity per week
        self.capacity = np.array([u0[e] for e in static["action_slots"]["edge"]], dtype=float)  # per action slot
        self.override_qty = np.zeros(action["override_qty"]["shape"])
        self.release_mode = np.zeros(action["release_mode"]["shape"], dtype=np.int64)  # 0: the default release
        self.rng = np.random.default_rng(config["policy_seed"])  # unused here; seed any randomness from it

    def act(self, observation):
        # observation: a dict of numpy arrays keyed by name ("stock.qty", "graph_now.u", "warning.score", ...)
        # action_mask is 1 on every route (action slot) that may carry goods this week, 0 where a sanction forbids it
        flows = self.capacity * observation["action_mask"]
        return {"flows": flows, "override_qty": self.override_qty, "release_mode": self.release_mode}
