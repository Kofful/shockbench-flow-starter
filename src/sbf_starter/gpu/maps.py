"""What the GPU trainers (examples/gpu_ppo.py, examples/gpu_es.py) load per map: the tables, a pool of training
episodes with their references and plain MPC's costs, and the held-out episodes of root 9001."""

import json

import numpy as np
import torch
from shockbench_flow.hosting.tasks import task_generator

from sbf_starter import ROOT, scoring
from sbf_starter.gpu import rl, scenarios
from sbf_starter.gpu.net import Batch, compile_net


MPC_CACHE = ROOT.parent / "shockbench-flow-starter" / "outputs" / "train_ppo_residual"  # plain MPC's training costs
HOLDOUT = {"small": 64, "full": 8}  # episodes of root 9001 the trainers evaluate on
MPC_HOLDOUT = {"small": 0.6941, "full": 0.4283}  # plain MPC's pooled score on those (official sbf compare)


class Map:
    """One map's simulator tables, route tables, a pool of episodes on the GPU and their references."""

    def __init__(self, task: str, entropy: int, episodes: int, device: str) -> None:
        self.task = task
        self.inst, _ = task_generator(task)
        self.net = compile_net(self.inst, device)
        self.tables = rl.route_tables(self.inst, self.net)
        self.tb = self.tables.to(device)
        self.base = self.tb["base"].float()
        scenarios.build(task, entropy, episodes)
        self.pool = scenarios.load(task, entropy, episodes, device)
        scenarios.build_info(task, entropy, episodes)
        scenarios.load_info(self.pool, task, entropy, episodes, device)
        self.es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=4, verbose=False)
        refs = self.es.references
        self.gap = torch.tensor([(r["J_naive_cents"] - r["J_oracle_cents"]) / 100 for r in refs], device=device)
        self.naive = torch.tensor([r["J_naive_cents"] / 100 for r in refs], device=device)
        self.excluded = torch.tensor([r.get("excluded") is not None for r in refs], device=device)
        path = MPC_CACHE / f"mpc_{task}_{entropy}.json"
        mpc = json.loads(path.read_text()) if path.is_file() else {}
        self.mpc = torch.tensor([mpc.get(str(n), np.nan) / 100 for n in range(episodes)], device=device)
        # the held-out episodes
        n = HOLDOUT[task]
        self.holdout_es = scoring.episode_set(task, n, entropy=9001, n_jobs=4, verbose=False)
        scenarios.build(task, 9001, n)
        self.holdout = scenarios.load(task, 9001, n, device)
        scenarios.build_info(task, 9001, n)
        scenarios.load_info(self.holdout, task, 9001, n, device)
        self.holdout_gap = torch.tensor(
            [(r["J_naive_cents"] - r["J_oracle_cents"]) / 100 for r in self.holdout_es.references], device=device
        ).clamp(min=1.0)

    def batch(self, idx: torch.Tensor, pool: Batch | None = None) -> Batch:
        """Episodes ``idx`` of the pool (or of ``pool``) as a Batch."""
        p = self.pool if pool is None else pool
        out = Batch(B=len(idx), **{k: getattr(p, k)[idx] for k in Batch.__dataclass_fields__ if k != "B"})
        out.u_now, out.o_now, out.supply_now = p.u_now[idx], p.o_now[idx], p.supply_now[idx]
        out.forecast, out.pending = p.forecast[idx], p.pending[idx]
        return out

    def holdout_score(self, J: torch.Tensor) -> float:
        """The pooled score (the board's RSS) of total costs ``J`` [holdout episodes] in USD."""
        r = self.holdout_es.rss([int(round(j * 100)) for j in J.tolist()])["rss"]
        return float("nan") if r is None else float(r)
