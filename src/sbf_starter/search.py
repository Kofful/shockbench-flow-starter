"""The pieces of an evolutionary policy search (AlphaEvolve-style), used by ``scripts/python/06_policy_search.py``.

- a **candidate** is the source code of an ``agent.py``; ``render`` fills a parametric template (``TEMPLATE``) with
  numbers, and ``write_candidate`` puts any source in a submission folder, which ``EpisodeSet.score`` scores as it
  scores any submission;
- the **fitness** of a candidate is its score on cached references (the wheel's ``EpisodeSet``);
- the **proposer** makes new candidates from the best ones: ``mutate`` takes Gaussian steps in the template's numbers.

Where an LLM proposer plugs in: replace ``mutate`` (numbers in a fixed template) with a function that takes the best
candidates' source and scores and returns new source code (a language model editing ``act``, say). A candidate is any
``agent.py`` source, so rewritten logic is scored the same way. Nothing here calls a model or needs an API key.

The template's policy: every action slot ships ``fraction`` of its capacity (one number for every slot, or one per
slot; 1 = send the maximum), and a route through a chokepoint is scaled by the chokepoint's observed open fraction
raised to ``closure_power`` (0: ignore closures). The shipped ``heuristic`` agent is this template at fraction 1 and
closure power 1 (``HEURISTIC``; ``tests/test_cli.py`` checks the file is the rendering).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


TEMPLATE = '''"""{doc}"""

import numpy as np


FRACTION = {fraction}  # of each slot's capacity: one number for every slot, or one per slot
CLOSURE_POWER = {closure_power}  # a lane through a strait ships (the strait's open fraction) ** CLOSURE_POWER


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        u0 = static["edges"]["u0"]  # each edge's nominal capacity per week
        slots = static["action_slots"]
        self.cap = np.array([u0[e] for e in slots["edge"]], dtype=float) * np.asarray(FRACTION, dtype=float)
        position = {{node: i for i, node in enumerate(layout["chokepoints"])}}  # strait -> index in graph_now.open
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
        return {{"flows": flows}}
'''
CANDIDATE_DOC = "Evolved by scripts/python/06_policy_search.py: fractions of capacity, scaled at closing straits."
HEURISTIC_DOC = """A hand-written rule: ship everything that can move, less into a chokepoint that is partly closed.

"Send the maximum" (the template agent) ships every open route at capacity. This rule adds one observation: a sea lane
through a chokepoint (a strait such as Hormuz or Taiwan) ships its capacity times the chokepoint's observed open
fraction (1 open, 0 closed), so less cargo waits in the queue there (queued cargo pays holding costs and arrives
late); its alternative routes (air, or another lane) keep shipping at capacity. Everything it reads is in the
observation and the static tables, the same on every network size.

It is the policy search's template (``sbf_starter.search.TEMPLATE``) at FRACTION 1 and CLOSURE_POWER 1, a starting
point, not a tuned policy: compare it with the template (``uv run sbf compare heuristic template``) before you trust
it, and see docs/GUIDE.md for the other signals (early warnings, announcements, pending sanctions) it ignores.
"""
HEURISTIC = {"fraction": 1.0, "closure_power": 1.0}
MAX_CLOSURE_POWER = 3.0  # the range of closure_power is [0, MAX_CLOSURE_POWER]; the fractions live in [0, 1]


def render_source(fraction: float | list[float], closure_power: float, doc: str = CANDIDATE_DOC) -> str:
    """The agent.py source of the template at these numbers (``fraction`` one number or one per slot)."""
    return TEMPLATE.format(doc=doc, fraction=fraction, closure_power=closure_power)


def heuristic_source() -> str:
    """The shipped ``heuristic`` agent's source: the template at ``HEURISTIC``."""
    return render_source(HEURISTIC["fraction"], HEURISTIC["closure_power"], HEURISTIC_DOC)


def start_params(n_slots: int) -> np.ndarray:
    """The parameters of send-the-maximum: every fraction 1, closures ignored (closure_power 0)."""
    return np.r_[np.ones(n_slots), 0.0]


def render(params: np.ndarray) -> str:
    """The agent.py source of a parameter vector: [fraction per slot..., closure_power]."""
    return render_source([round(float(x), 4) for x in params[:-1]], round(float(params[-1]), 4))


def write_candidate(source: str, workdir: Path, name: str) -> Path:
    """A submission folder ``workdir/name`` holding ``source`` as its agent.py (score it with ``EpisodeSet.score``)."""
    folder = Path(workdir) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "agent.py").write_text(source)
    return folder


def mutate(parents: list[np.ndarray], rng: np.random.Generator, n: int, sigma: float = 0.15) -> list[np.ndarray]:
    """The proposer: Gaussian steps around randomly chosen parents, clipped to the parameters' ranges.

    Args:
        parents: parameter vectors to mutate (the elite).
        rng: the search's generator (seed it: a search is then repeatable).
        n: the number of children.
        sigma: the step's scale, relative to each parameter's range.

    """
    out = []
    for _ in range(n):
        p = parents[rng.integers(len(parents))]
        child = p + rng.normal(0.0, sigma, p.shape) * np.r_[np.ones(len(p) - 1), MAX_CLOSURE_POWER]
        child[:-1] = np.clip(child[:-1], 0.0, 1.0)
        child[-1] = np.clip(child[-1], 0.0, MAX_CLOSURE_POWER)
        out.append(child)
    return out


def slot_count(task: str) -> int:
    """The number of action slots of a task (the length of ``flows``)."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments

    from sbf_starter import env_id

    return int(gym.make(env_id(task)).action_space["flows"].shape[0])
