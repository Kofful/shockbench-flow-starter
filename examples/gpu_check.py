"""Compare the tensor simulator (sbf_starter.gpu) with the package's, week by week, under the same random flows.

    uv run python examples/gpu_check.py                          # tiny, 4 episodes of root 1003, on the CPU
    uv run python examples/gpu_check.py --task=small --episodes=8 --device=cuda

Every week both simulators get the same requests (a random share, up to 1.3, of each route's capacity, so the clip
binds now and then); the package plays one episode at a time (``shockbench_flow.dynamics.sim.step``), the tensor one
all of them at once. Printed: the largest relative gap of the week's stock and of each cost component over the
episodes, and of the episode's total cost J = sum of the weeks' costs - the terminal credit.
"""

import fire
import numpy as np
import torch
from shockbench_flow.disruption.sampler import sample_omega
from shockbench_flow.dynamics import sim as pkg
from shockbench_flow.hosting.tasks import split_label, task_generator
from shockbench_flow.marks import compute_marks

from sbf_starter.gpu.net import batch_marks, compile_net
from sbf_starter.gpu.sim import COSTS, Sim


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)))


def main(
    task: str = "tiny",
    episodes: int = 4,
    entropy: int = 1003,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    inst, params = task_generator(task)
    marks = [compute_marks(inst, sample_omega(inst, params, entropy, n, split_label(entropy))) for n in range(episodes)]
    net = compile_net(inst, device)
    sim = Sim(net, batch_marks(inst, marks, device))
    sim.reset()
    states = [pkg.initial_state(inst) for _ in range(episodes)]
    rng = np.random.default_rng(seed)
    A = len(inst.action_slots)
    edge = np.array([e for e, _k, _l in inst.action_slots])
    k_of = np.array([k for _e, k, _l in inst.action_slots])
    J_pkg, J_gpu = np.zeros(episodes), np.zeros(episodes)
    worst = {name: 0.0 for name in ("stock", *COSTS, "J")}
    for t in range(1, inst.T + 1):
        flows = np.zeros((episodes, A))
        for b, m in enumerate(marks):
            u = np.where(np.isfinite(m.u[t - 1, edge]), m.u[t - 1, edge], 100.0)
            flows[b] = rng.uniform(0, 1.3, A) * u * (rng.random(A) < 0.8)
        out = sim.step(torch.as_tensor(flows, device=device))
        gpu = {k: v.cpu().numpy() for k, v in out.items()}
        line = []
        for b, (m, st) in enumerate(zip(marks, states)):
            req = {
                s: float(flows[b, s]) for s in range(A) if flows[b, s] > 0 and not m.prohibited[t - 1, edge[s], k_of[s]]
            }
            rec = pkg.step(inst, m, st, req)
            J_pkg[b] += rec.costs.total()
            J_gpu[b] += gpu["total"][b]
            gap = rel(rec.stock, sim.stock[b].cpu().numpy())
            worst["stock"] = max(worst["stock"], gap)
            for name in COSTS:
                worst[name] = max(worst[name], rel(np.array([getattr(rec.costs, name)]), gpu[name][b : b + 1]))
            line.append(gap)
        if verbose or t in (1, inst.T):
            print(f"week {t:3d}: stock gap {max(line):.2e}", flush=True)
    salvage = sim.salvage().cpu().numpy()
    for b, st in enumerate(states):
        J_pkg[b] -= pkg.terminal_salvage(inst, st)
        J_gpu[b] -= salvage[b]
    worst["J"] = rel(J_pkg, J_gpu)
    print(f"\n{task}, {episodes} episodes of root {entropy}, largest relative gap over all weeks:")
    for name, v in worst.items():
        print(f"  {name:14s} {v:.2e}{'   <-- differs' if v > 1e-6 else ''}")
    print("J (package):", np.round(J_pkg / 1e6, 3).tolist(), "M\nJ (tensor): ", np.round(J_gpu / 1e6, 3).tolist(), "M")


if __name__ == "__main__":
    fire.Fire(main)
