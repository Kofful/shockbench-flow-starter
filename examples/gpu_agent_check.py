"""Check agents/gpu_rl against the training: the same features from the real observation as from the tensor simulator.

    uv run python examples/gpu_agent_check.py --task=small --episodes=3
    uv run python examples/gpu_agent_check.py --task=full --episodes=2

Plays episodes of root 9001 in the package's environment (regime standard, as scored) with agents/gpu_rl, and the
same flows in the tensor simulator on the CPU; prints the largest gap of the features over all weeks and routes, of
the flows the network picks from each side's features, and of the episode's total cost.
"""

import sys

import fire
import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
import torch
from shockbench_flow.disruption.sampler import sample_omega
from shockbench_flow.hosting.tasks import split_label, task_generator
from shockbench_flow_agent import agent_config

from sbf_starter import ROOT, env_id
from sbf_starter.gpu import rl, scenarios
from sbf_starter.gpu.net import compile_net
from sbf_starter.gpu.sim import Sim


def main(
    task: str = "small", episodes: int = 3, entropy: int = 9001, agent: str = "gpu_rl", policy: str | None = None
) -> None:
    """``policy``: the training network (policy.pt) the agent was exported from, to compare its flows too."""
    sys.path.insert(0, str(ROOT / "agents" / agent))
    import agent as gpu_rl

    inst, params = task_generator(task)
    net = compile_net(inst, "cpu")
    tb = rl.route_tables(inst, net).to("cpu")
    scenarios.build(task, entropy, episodes)
    worst = {"features": 0.0, "flows": 0.0, "J": 0.0}
    net_t = None
    if policy:
        state = torch.load(policy, map_location="cpu")
        net_t = rl.RoutePolicy(pool=any(k.startswith("mix_") for k in state))
        net_t.load_state_dict(rl.widen(state), strict=False)
    for n in range(episodes):
        env = gym.make(env_id(task), regime="standard")
        env.unwrapped.omega_source = lambda i: sample_omega(inst, params, entropy, i, split_label(entropy))
        obs, info = env.reset(options={"episode": n})
        pol = gpu_rl.Agent(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        batch = scenarios.load(task, entropy, [n], "cpu")
        scenarios.build_info(task, entropy, [n])
        sim = Sim(net, scenarios.load_info(batch, task, entropy, [n], "cpu"))
        sim.reset()
        J_env = J_sim = 0.0
        done, t = False, 0
        while not done:
            t += 1
            x_obs, cap_obs = pol.features(obs)
            x_sim, cap_sim = rl.sim_features(sim, tb, t)
            x_sim = x_sim[0].numpy().astype(np.float64)
            worst["features"] = max(worst["features"], float(np.abs(x_obs - x_sim).max()))
            action = pol.act(obs)
            if net_t is not None:
                with torch.no_grad():
                    z, _ = net_t(torch.as_tensor(x_obs[None], dtype=torch.float32), tb["base"].float(), tb)
                flows_t = torch.sigmoid(z[0]).double().numpy() * np.where(np.isfinite(cap_obs), cap_obs, 0.0)
                ok = action["flows"] > 0
                gap_f = np.abs(flows_t[ok] - action["flows"][ok]) / np.maximum(np.abs(flows_t[ok]), 1.0)
                worst["flows"] = max(worst["flows"], float(gap_f.max()) if ok.any() else 0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            J_env -= reward
            costs = sim.step(torch.as_tensor(action["flows"][None], dtype=torch.float64))
            J_sim += float(costs["total"][0])
            if done:
                J_sim -= float(sim.salvage()[0])
            if t == 1 and n == 0:
                bad = np.abs(x_obs - x_sim).max(0)
                print(
                    "week 1 per-feature gap:",
                    {rl.FEATURES[j]: round(float(bad[j]), 6) for j in np.flatnonzero(bad > 1e-9)},
                )
        gap = abs(J_env - J_sim) / max(abs(J_env), 1.0)
        worst["J"] = max(worst["J"], gap)
        print(f"episode {n}: J env {J_env / 1e6:,.3f}M, J sim {J_sim / 1e6:,.3f}M (relative gap {gap:.1e})", flush=True)
    gaps = f"largest feature gap {worst['features']:.2e}, flow gap {worst['flows']:.2e}, cost gap {worst['J']:.2e}"
    print(f"\n{task}, {episodes} episodes of root {entropy}: {gaps}")


if __name__ == "__main__":
    fire.Fire(main)
