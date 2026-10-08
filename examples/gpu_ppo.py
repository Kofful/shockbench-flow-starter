"""PPO without MPC on the tensor simulator: hundreds of small and full games at once on the GPU.

    uv run python examples/gpu_ppo.py --minutes=5 --small_envs=64 --full_envs=8           # a smoke test
    uv run python examples/gpu_ppo.py --minutes=120                                       # a real run
    uv run python examples/gpu_ppo.py --start=outputs/gpu_ppo/<run>/policy.pt --minutes=240

Every iteration plays ``small_envs`` small and ``full_envs`` full episodes drawn from the cached pools of the training
root (examples/gpu_scenarios.py) with the stochastic policy (``sbf_starter.gpu.rl.RoutePolicy``), then updates it by
PPO on both maps together. A week's reward is its cost divided by the episode's (naive - clairvoyant) cost, the
terminal credit added in the last week, so an episode's return is its score (RSS) minus a constant. The log shows the
training episodes' mean score per map and that minus plain MPC's on the same episodes (exploration noise included);
every ``eval_every`` iterations the deterministic policy plays the held-out episodes of root 9001 and its pooled score
is compared with plain MPC's there (small 0.6941 on 64, full 0.4283 on 8). Everything goes to
outputs/gpu_ppo/<run>/ (log.json for examples/watch_training.py, policy.pt, tables) and MLflow.
"""

import json
import os
import time
from pathlib import Path

import fire
import numpy as np
import torch

import sbf_starter.gpu.sim as gsim
from sbf_starter import ROOT
from sbf_starter.gpu import rl
from sbf_starter.gpu.maps import MPC_HOLDOUT, Map
from sbf_starter.tracking import Tracker


os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")  # one compiler process: WSL has little memory


@torch.no_grad()
def play(mp: Map, batch, policy: rl.RoutePolicy, gap: torch.Tensor, deterministic: bool):
    """One episode of every game in ``batch``; returns the rollout (features, actions, log-probs, values, rewards)
    and each game's total cost J."""
    sim = gsim.Sim(mp.net, batch)
    sim.reset()
    T = mp.inst.T
    xs, zs, logps, vals, rews = [], [], [], [], []
    J = torch.zeros(batch.B, dtype=torch.float64, device=gap.device)
    std = policy.log_std.exp()
    for t in range(1, T + 1):
        x, cap = rl.sim_features(sim, mp.tb, t)
        with torch.no_grad():
            mean, v = policy(x, mp.base, mp.tb)
        z = mean if deterministic else mean + std * torch.randn_like(mean)
        logp = torch.distributions.Normal(mean, std).log_prob(z)  # per route: PPO's ratio and KL are per route
        costs = sim.step(torch.sigmoid(z).double() * torch.where(torch.isfinite(cap), cap, 0.0))
        r = -costs["total"]
        if t == T:
            r = r + sim.salvage()
        J -= r
        xs.append(x), zs.append(z), logps.append(logp), vals.append(v), rews.append((r / gap).float())
    return (torch.stack(xs), torch.stack(zs), torch.stack(logps), torch.stack(vals), torch.stack(rews)), J


def group_advantage(rews: torch.Tensor, group: int, gamma: float) -> torch.Tensor:
    """Advantages [T, B] against the copies of the same episode: each episode is played ``group`` times side by side
    (games b and b + n, b + 2n, ... share it), so a week's discounted reward-to-go minus its copies' mean at that week
    keeps what the actions did and drops what the scenario did (the idea of GRPO)."""
    T, B = rews.shape
    togo = torch.zeros_like(rews)
    run = torch.zeros_like(rews[0])
    for t in reversed(range(T)):
        run = rews[t] + gamma * run
        togo[t] = run
    g = togo.reshape(T, group, B // group)
    return (g - g.mean(1, keepdim=True)).reshape(T, B)


def gae(vals: torch.Tensor, rews: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    T = rews.shape[0]
    adv = torch.zeros_like(rews)
    last = torch.zeros_like(rews[0])
    for t in reversed(range(T)):
        nxt = vals[t + 1] if t + 1 < T else torch.zeros_like(vals[0])
        delta = rews[t] + gamma * nxt - vals[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + vals


def holdout_score(mp: Map, policy) -> float:
    J = play(mp, mp.holdout, policy, mp.holdout_gap, deterministic=True)[1]
    return mp.holdout_score(J)


def main(
    minutes: float = 120,
    small_envs: int = 512,
    full_envs: int = 64,
    entropy: int = 1003,
    pool_small: int = 256,
    pool_full: int = 32,
    epochs: int = 4,
    minibatch: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    target_kl: float = 0.01,
    log_std: float = -1.0,
    min_std: float = 0.05,
    hidden: int = 64,
    pool: bool = False,
    eval_every: int = 10,
    group: int = 8,
    patience: int = 30,
    min_delta: float = 0.005,
    start: str | None = None,
    seed: int = 0,
    device: str = "cuda",
    out: str | None = None,
) -> None:
    """Train ``RoutePolicy`` by PPO for ``minutes`` on small and full, evaluating it on root 9001 as it goes.

    Args:
        minutes: training time (loading the maps and episodes comes first).
        small_envs, full_envs: games per iteration on each map.
        entropy: the training root; pool_small, pool_full: its cached episodes the games draw from (they need
            references: examples/gpu_scenarios.py).
        epochs, minibatch: PPO's passes over an iteration's weeks and the weeks per gradient step.
        lr, gamma, lam, clip, target_kl: Adam's step, the discount, GAE's lambda, PPO's clip, and the KL per route
            that stops an iteration's epochs early (ratios, clipping and KL are per route: an action has hundreds).
        log_std: the initial exploration (standard deviation of the logits, log).
        min_std: the exploration never falls below this (it collapsed to 0.02 overnight and the search slowed).
        hidden: the network's width.
        eval_every: iterations between held-out evaluations.
        group: copies of each training episode played side by side; the advantage is against the copies' mean
            (``group_advantage``; 1: GAE on the value network instead). small_envs and full_envs must be multiples.
        patience, min_delta: stop when the held-out score (the mean of small's and full's) has not beaten its best by
            ``min_delta`` for ``patience`` evaluations in a row (0: never stop early). The best policy is kept as
            policy_best.pt, the last as policy.pt.
        start: a policy.pt to go on from.
    """
    params = dict(locals())
    torch.manual_seed(seed)
    import torch._inductor.config as inductor

    inductor.compile_threads = 1
    gsim._cohort = torch.compile(gsim._cohort, dynamic=False)
    run = Path(out or ROOT / "outputs" / "gpu_ppo" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir(parents=True, exist_ok=True)
    maps = {"small": Map("small", entropy, pool_small, device), "full": Map("full", entropy, pool_full, device)}
    envs = {"small": small_envs, "full": full_envs}
    policy = rl.RoutePolicy(hidden=hidden, log_std=log_std, pool=pool).to(device)
    if start:
        state = rl.widen(torch.load(start, map_location=device))  # 25 inputs: widened
        policy.load_state_dict(state, strict=not pool or any(k.startswith("mix_") for k in state))
    floor = float(np.log(min_std))
    with torch.no_grad():
        policy.log_std.clamp_(min=floor)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    tracker = Tracker("gpu_ppo", run.name, params | {"run_folder": run})
    for task, mp in maps.items():
        np.savez(run / f"tables_{task}.npz", **{k: v for k, v in vars(mp.tables).items()})
    rng = torch.Generator(device=device).manual_seed(seed)
    log, began, it, weeks = [], time.monotonic(), 0, 0
    best, stale = -float("inf"), 0
    print(
        f"training {minutes:g} min; {small_envs} small + {full_envs} full games an iteration; run in {run}", flush=True
    )
    while time.monotonic() - began < 60 * minutes:
        it += 1
        start_it = time.perf_counter()
        data, row = {}, {"iteration": it}
        for task, mp in maps.items():
            ok = (~mp.excluded).nonzero().flatten()
            if group > 1:  # envs[task] // group episodes, each played group times (copies b, b + n, ...)
                idx = ok[torch.randint(len(ok), (envs[task] // group,), generator=rng, device=device)].repeat(group)
            else:
                idx = ok[torch.randint(len(ok), (envs[task],), generator=rng, device=device)]
            (x, z, logp, v, r), J = play(mp, mp.batch(idx), policy, mp.gap[idx], deterministic=False)
            adv, ret = gae(v, r, gamma, lam)
            if group > 1:
                adv = group_advantage(r, group, gamma)
            data[task] = (x, z, logp, adv, ret)
            score = (mp.naive[idx] - J) / mp.gap[idx]
            mpc = (mp.naive[idx] - mp.mpc[idx]) / mp.gap[idx]
            row[task] = {
                "episodes": it * envs[task],
                "rss": float(score.mean()),
                "minus_mpc": float((score - mpc).nanmean()),
            }
            weeks += envs[task] * mp.inst.T
        play_s = time.perf_counter() - start_it
        # ----- PPO update on both maps --------------------------------------------------------------------------
        flat = {}
        for task, (x, z, logp, adv, ret) in data.items():
            A = x.shape[2]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            flat[task] = (
                x.reshape(-1, A, x.shape[3]),
                z.reshape(-1, A),
                logp.reshape(-1, A),
                adv.reshape(-1),
                ret.reshape(-1),
            )
        kl_seen = 0.0
        for _epoch in range(epochs):
            perms = {task: torch.randperm(len(f[3]), device=device) for task, f in flat.items()}
            n_steps = max(len(f[3]) for f in flat.values()) // minibatch + 1
            stop = False
            for i in range(n_steps):
                loss = 0.0
                kls = []
                for task, (x, z, logp_old, adv, ret) in flat.items():
                    n = len(adv)
                    size = max(1, n // n_steps)
                    sel = perms[task][i * size : (i + 1) * size]
                    if len(sel) == 0:
                        continue
                    mean, v = policy(x[sel], maps[task].base, maps[task].tb)
                    logp = torch.distributions.Normal(mean, policy.log_std.exp()).log_prob(z[sel])
                    log_ratio = logp - logp_old[sel]  # [mb, A]: every route its own ratio, clipped on its own
                    ratio = torch.exp(log_ratio)
                    a = adv[sel][:, None]
                    pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                    vl = 0.5 * ((v - ret[sel]) ** 2).mean()
                    loss = loss + pg + vl
                    kls.append(float(((ratio - 1) - log_ratio).mean().detach()))  # KL per route
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    policy.log_std.clamp_(min=floor)
                kl_seen = max(kls) if kls else 0.0
                if kl_seen > target_kl:
                    stop = True
                    break
            if stop:
                break
        row |= {
            "minutes": round((time.monotonic() - began) / 60, 2),
            "weeks": weeks,
            "kl": kl_seen,
            "std": float(policy.log_std.exp()),
            "seconds": round(time.perf_counter() - start_it, 1),
            "play_seconds": round(play_s, 1),
        }
        if it % eval_every == 0 or it == 1:
            policy.eval()
            for task, mp in maps.items():
                h = holdout_score(mp, policy)
                row[f"holdout_{task}"] = h
                row[f"holdout_{task}_minus_mpc"] = h - MPC_HOLDOUT[task]
            policy.train()
            torch.save(policy.state_dict(), run / "policy.pt")
            held_mean = (row["holdout_small"] + row["holdout_full"]) / 2
            if held_mean > best + min_delta:
                best, stale = held_mean, 0
                torch.save(policy.state_dict(), run / "policy_best.pt")
                row["best"] = True
            else:
                stale += 1
        log.append(row)
        tracker.log(
            {
                k: v
                for k, v in {
                    "small_rss": row["small"]["rss"],
                    "small_minus_mpc": row["small"]["minus_mpc"],
                    "full_rss": row["full"]["rss"],
                    "full_minus_mpc": row["full"]["minus_mpc"],
                    "std": row["std"],
                    "kl": row["kl"],
                    **{k: row[k] for k in row if k.startswith("holdout_")},
                }.items()
            },
            step=it,
        )
        held = (
            f" | held out: small {row['holdout_small']:.4f} ({row['holdout_small_minus_mpc']:+.4f} vs mpc),"
            f" full {row['holdout_full']:.4f} ({row['holdout_full_minus_mpc']:+.4f})"
            if "holdout_small" in row
            else ""
        )
        print(
            f"it {it} ({row['minutes']:.1f} min, {weeks / 1e6:.2f}M weeks, {row['seconds']} s): "
            f"small {row['small']['rss']:.3f} ({row['small']['minus_mpc']:+.3f} vs mpc), "
            f"full {row['full']['rss']:.3f} ({row['full']['minus_mpc']:+.3f}), std {row['std']:.3f}" + held,
            flush=True,
        )
        (run / "log.json").write_text(json.dumps(log, indent=1))
        if patience and stale >= patience:
            print(f"early stop: the held-out score has not beaten {best:.4f} for {patience} evaluations", flush=True)
            break
    torch.save(policy.state_dict(), run / "policy.pt")
    tracker.artifact(run / "policy.pt")
    tracker.artifact(run / "policy_best.pt")
    tracker.end()
    print(f"done; the best held-out score (mean of small and full) {best:.4f}; the run is in {run}")


if __name__ == "__main__":
    fire.Fire(main)
