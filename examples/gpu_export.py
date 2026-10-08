"""Export a policy trained by examples/gpu_ppo.py into agents/gpu_rl (policy.npz and one table file per map).

    uv run python examples/gpu_export.py --policy=outputs/gpu_ppo/<run>/policy_best.pt
    uv run python examples/gpu_export.py --policy=outputs/gpu_es/<run>/policy_best.pt --out=agents/gpu_gnn
    uv run sbf check gpu_rl --task=small        # then --task=full

The tables are the training's (``sbf_starter.gpu.rl.route_tables``), with the stock slots of each route's tail and
destination written as (node, commodity), which the agent finds in its observation's layout.
"""

from pathlib import Path

import fire
import numpy as np
import torch
from shockbench_flow.hosting.tasks import task_generator

from sbf_starter import ROOT
from sbf_starter.gpu import rl
from sbf_starter.gpu.net import compile_net


AGENT = ROOT / "agents" / "gpu_rl"


def main(policy: str, out: str | None = None) -> None:
    folder = Path(out) if out else AGENT
    folder.mkdir(parents=True, exist_ok=True)
    if folder != AGENT:  # another agent of the same code (agents/gpu_gnn): the agent file too
        (folder / "agent.py").write_text((AGENT / "agent.py").read_text())
    sd = torch.load(policy, map_location="cpu")
    np.savez(
        folder / "policy.npz",
        enc0_w=sd["enc.0.weight"].numpy(),
        enc0_b=sd["enc.0.bias"].numpy(),
        enc1_w=sd["enc.2.weight"].numpy(),
        enc1_b=sd["enc.2.bias"].numpy(),
        act_w=sd["act.weight"].numpy(),
        act_b=sd["act.bias"].numpy(),
        **(
            {
                "mix_h_w": sd["mix_h.weight"].numpy(),
                "mix_h_b": sd["mix_h.bias"].numpy(),
                "mix_d_w": sd["mix_d.weight"].numpy(),
                "mix_s_w": sd["mix_s.weight"].numpy(),
            }
            if "mix_h.weight" in sd
            else {}
        ),
    )
    for task in ("small", "full"):
        inst, _ = task_generator(task)
        tb = rl.route_tables(inst, compile_net(inst))
        slots = inst.stock_slots
        np.savez(
            folder / f"tables_{inst.instance_id}.npz",
            group=tb.group,
            k=tb.k,
            edge=np.array([e for e, _k, _l in inst.action_slots]),
            lane=np.array([-1 if lane is None else lane for _e, _k, lane in inst.action_slots]),
            route_edges=tb.route_edges,
            route_chks=tb.route_chks,
            tail=np.array([(slots[s].node, slots[s].k) for s in tb.tail]),
            dest=np.array([(slots[s].node, slots[s].k) for s in tb.dest]),
            cap0=tb.cap0,
            scale=tb.scale,
            nominal=tb.nominal,
            base=tb.base,
            base_load=np.array(tb.base_load),
            dest_storage=tb.dest_storage,
        )
    print(f"exported {policy} to {folder}; next: uv run sbf check gpu_rl --task=small (and --task=full)")


if __name__ == "__main__":
    fire.Fire(main)
