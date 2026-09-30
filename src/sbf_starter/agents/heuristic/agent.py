"""A hand-written rule: ship everything that can move, less into a chokepoint that is partly closed.

"Send the maximum" (the template agent) ships every open route at capacity. This rule adds one observation: a sea lane
through a chokepoint (a strait such as Hormuz or Taiwan) ships its capacity times the chokepoint's observed open
fraction (1 open, 0 closed), so less cargo waits in the queue there (queued cargo pays holding costs and arrives
late); its alternative routes (air, or another lane) keep shipping at capacity. Everything it reads is in the
observation and the static tables, the same on every network size.

It is the policy search's template (``sbf_starter.search.TEMPLATE``) at FRACTION 1 and CLOSURE_POWER 1, a starting
point, not a tuned policy: compare it with the template (``uv run sbf compare heuristic template``) before you trust
it, and see docs/GUIDE.md for the other signals (early warnings, announcements, pending sanctions) it ignores.
"""

import numpy as np


FRACTION = 1.0  # of each slot's capacity: one number for every slot, or one per slot
CLOSURE_POWER = 1.0  # a lane through a strait ships (the strait's open fraction) ** CLOSURE_POWER


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        u0 = static["edges"]["u0"]  # each edge's nominal capacity per week
        slots = static["action_slots"]
        self.cap = np.array([u0[e] for e in slots["edge"]], dtype=float) * np.asarray(FRACTION, dtype=float)
        position = {node: i for i, node in enumerate(layout["chokepoints"])}  # strait -> index in graph_now.open
        lane_chokepoints = static["lanes"]["chokepoints"]
        # for every slot, the positions (in graph_now.open) of the straits its lane passes; [] off any lane
        self.through = [
            [position[c] for c in lane_chokepoints[lane]] if lane is not None else [] for lane in slots["lane"]
        ]

    def act(self, observation):
        flows = self.cap * observation["action_mask"]  # the routes not prohibited this week
        open_now = observation["graph_now.open"]  # each strait's open fraction, 1 open .. 0 closed
        seen = observation["graph_now.open.observed"] == 1
        for s, chokepoints in enumerate(self.through):
            for c in chokepoints:
                if seen[c]:
                    flows[s] *= max(float(open_now[c]), 0.0) ** CLOSURE_POWER  # less cargo queues at a closing strait
        return {"flows": flows}
