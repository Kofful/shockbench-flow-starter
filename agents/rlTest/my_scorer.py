"""Adapter between tune.py and the ShockBench-Flow starter kit."""
from agent import Agent
from sbf_starter.scoring import episode_set

def score_one(weights: dict, task: str, entropy: int) -> float:
    es = episode_set(task, episodes=[0], quick=False, entropy=entropy, verbose=False)
    
    class TunedAgent(Agent):
        def __init__(self, config):
            super().__init__(config, _weights=weights)

    res = es.score(TunedAgent, name="TunedAgent")
    return float(res.rss)