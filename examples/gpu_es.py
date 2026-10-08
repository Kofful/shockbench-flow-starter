"""Evolution strategy for the route policy on the GPU: every candidate network plays the same episodes in one batch.

    uv run python examples/gpu_es.py --minutes=3 --pairs=4 --small_eps=4 --full_eps=1           # a smoke test
    uv run python examples/gpu_es.py --start=outputs/gpu_ppo/<run>/policy_best.pt --minutes=120

OpenAI's evolution strategy on the weights of ``sbf_starter.gpu.rl.RoutePolicy`` (its encoder and action head; the
value head is not used): each generation draws ``pairs`` directions eps, and the mean weights, mean + sigma eps and
mean - sigma eps all play the same ``small_eps`` small and ``full_eps`` full episodes of the training pools,
deterministically (the search's noise is in the weights, not the actions); a candidate's fitness is the mean of its
two maps' mean scores, the fitnesses become centred ranks, and the mean moves along their rank-weighted directions
with Adam. Comparing candidates on the same episodes removes what the scenario did, as in examples/train_residual.py,
but a generation takes seconds instead of minutes. Every ``eval_every`` generations the mean plays the held-out
episodes of root 9001 (as examples/gpu_ppo.py); the best is kept as policy_best.pt.
"""

import json
import os
import time
from pathlib import Path

import fire
import torch

import sbf_starter.gpu.sim as gsim
from sbf_starter import ROOT
from sbf_starter.gpu import rl
from sbf_starter.gpu.maps import MPC_HOLDOUT, Map
from sbf_starter.tracking import Tracker


os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


def searched(policy: rl.RoutePolicy) -> tuple[str, ...]:
    """The weights the search moves, in the order ``logits`` reads them (the value head is not searched)."""
    mix = ("mix_h.weight", "mix_h.bias", "mix_d.weight", "mix_s.weight") if policy.pool else ()
    return ("enc.0.weight", "enc.0.bias", "enc.2.weight", "enc.2.bias", *mix, "act.weight", "act.bias")


def flatten(policy: rl.RoutePolicy) -> torch.Tensor:
    sd = policy.state_dict()
    return torch.cat([sd[k].flatten() for k in searched(policy)]).double()


def unflatten(policy: rl.RoutePolicy, flat: torch.Tensor) -> None:
    sd, i = policy.state_dict(), 0
    for k in searched(policy):
        n = sd[k].numel()
        sd[k].copy_(flat[i : i + n].view_as(sd[k]).to(sd[k].dtype))
        i += n


def logits(policy: rl.RoutePolicy, P: torch.Tensor, x: torch.Tensor, base: torch.Tensor, tb: dict) -> torch.Tensor:
    """Mean logits [C, N, A] of C candidates' flat weights ``P`` [C, D] on their own features ``x`` [C, N, A, F]."""
    C, N, A, F = x.shape
    sd = policy.state_dict()
    parts, i = {}, 0
    for k in searched(policy):
        n = sd[k].numel()
        parts[k] = P[:, i : i + n].float().view(C, *sd[k].shape)
        i += n

    def linear(h, k):  # h [C, M, in] by each candidate's own weight
        out = torch.bmm(h, parts[f"{k}.weight"].transpose(1, 2))
        return out + parts[f"{k}.bias"][:, None, :] if f"{k}.bias" in parts else out

    h = torch.tanh(linear(x.view(C, N * A, F), "enc.0"))
    h = torch.tanh(linear(h, "enc.2"))
    if policy.pool:
        n = int(max(tb["dest"].max(), tb["tail"].max())) + 1
        per_game = h.view(C * N, A, -1)  # neighbours are pooled inside each game
        dest = rl.neighbours(per_game, tb["dest"], n).view(C, N * A, -1)
        tail = rl.neighbours(per_game, tb["tail"], n).view(C, N * A, -1)
        h = h + torch.tanh(linear(h, "mix_h") + linear(dest, "mix_d") + linear(tail, "mix_s"))
    z = linear(h, "act").squeeze(-1)
    return z.view(C, N, A) + base


@torch.no_grad()
def play(mp: Map, batch, policy: rl.RoutePolicy, P: torch.Tensor) -> torch.Tensor:
    """Every game of ``batch`` (candidate-major: games c * N .. c * N + N - 1 are candidate c's) to its end, each
    candidate deterministic with its own weights; returns the games' total costs J [C * N]."""
    C = P.shape[0]
    sim = gsim.Sim(mp.net, batch)
    sim.reset()
    J = torch.zeros(batch.B, dtype=torch.float64, device=P.device)
    for t in range(1, mp.inst.T + 1):
        x, cap = rl.sim_features(sim, mp.tb, t)
        z = logits(policy, P, x.view(C, batch.B // C, *x.shape[1:]), mp.base, mp.tb).view(batch.B, -1)
        costs = sim.step(torch.sigmoid(z).double() * torch.where(torch.isfinite(cap), cap, 0.0))
        J += costs["total"]
        if t == mp.inst.T:
            J -= sim.salvage()
    return J


def centred_ranks(x: torch.Tensor) -> torch.Tensor:
    r = torch.empty_like(x)
    r[torch.argsort(x)] = torch.arange(len(x), dtype=x.dtype, device=x.device)
    return r / (len(x) - 1) - 0.5


def main(
    minutes: float = 120,
    start: str | None = None,
    pairs: int = 32,
    small_eps: int = 16,
    full_eps: int = 4,
    sigma: float = 0.02,
    lr: float = 0.005,
    entropy: int = 1003,
    pool_small: int = 1024,
    pool_full: int = 128,
    hidden: int = 64,
    pool: bool = False,
    eval_every: int = 10,
    patience: int = 30,
    min_delta: float = 0.002,
    seed: int = 0,
    device: str = "cuda",
    out: str | None = None,
) -> None:
    """Search the policy's weights for ``minutes``, evaluating the mean on root 9001 as it goes.

    Args:
        start: a policy.pt to start from (examples/gpu_ppo.py's), else a fresh network (the nominal plan).
        pairs: directions per generation, each played both ways (2 * pairs candidates, plus the mean).
        small_eps, full_eps: episodes every candidate plays per generation (the same for all candidates).
        sigma: the spread of the candidates around the mean, in units of the weights.
        lr: Adam's step on the weights.
        entropy, pool_small, pool_full: the training root and its cached episodes (with references).
        eval_every, patience, min_delta: held-out evaluations, and the early stop as in examples/gpu_ppo.py.
        pool: the routes look at their neighbours (``RoutePolicy`` with pool); from an unpooled start the mix
            starts at zero, so the first generation plays as the start.
    """
    params = dict(locals())
    import torch._inductor.config as inductor

    inductor.compile_threads = 1
    gsim._cohort = torch.compile(gsim._cohort, dynamic=False)
    torch.manual_seed(seed)
    run = Path(out or ROOT / "outputs" / "gpu_es" / time.strftime("%Y-%m-%d_%H-%M-%S"))
    run.mkdir(parents=True, exist_ok=True)
    maps = {"small": Map("small", entropy, pool_small, device), "full": Map("full", entropy, pool_full, device)}
    eps = {"small": small_eps, "full": full_eps}
    policy = rl.RoutePolicy(hidden=hidden, pool=pool).to(device)
    if start:
        state = rl.widen(torch.load(start, map_location=device))  # 25 inputs: widened
        policy.load_state_dict(state, strict=not pool or any(k.startswith("mix_") for k in state))
    mean = flatten(policy)
    D = len(mean)
    m, v = torch.zeros_like(mean), torch.zeros_like(mean)
    tracker = Tracker("gpu_es", run.name, params | {"run_folder": run, "n_params": D})
    rng = torch.Generator(device=device).manual_seed(seed)
    log, began, g = [], time.monotonic(), 0
    best, stale = -float("inf"), 0
    print(
        f"{D} weights, {2 * pairs + 1} candidates x ({small_eps} small + {full_eps} full) games a generation; "
        f"run in {run}",
        flush=True,
    )
    while time.monotonic() - began < 60 * minutes:
        g += 1
        start_g = time.perf_counter()
        noise = torch.randn(pairs, D, generator=rng, device=device, dtype=torch.float64)
        P = torch.cat([mean[None], mean + sigma * noise, mean - sigma * noise])  # [1 + 2 pairs, D]
        C = P.shape[0]
        fit, row = {}, {"generation": g}
        for task, mp in maps.items():
            ok = (~mp.excluded).nonzero().flatten()
            ep = ok[torch.randint(len(ok), (eps[task],), generator=rng, device=device)]
            J = play(mp, mp.batch(ep.repeat(C)), policy, P).view(C, -1)
            score = (mp.naive[ep] - J) / mp.gap[ep]  # [C, episodes]
            fit[task] = score.mean(1)
            mpc = (mp.naive[ep] - mp.mpc[ep]) / mp.gap[ep]
            row[task] = {
                "rss": float(fit[task][0]),
                "minus_mpc": float((score[0] - mpc).nanmean()),
                "episodes": g * eps[task],
            }
        f = (fit["small"] + fit["full"]) / 2
        ranks = centred_ranks(f[1:])
        grad = (ranks[:pairs] - ranks[pairs:]) @ noise / (2 * pairs * sigma)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        mean = mean + lr * (m / (1 - 0.9**g)) / (torch.sqrt(v / (1 - 0.999**g)) + 1e-8)
        row |= {
            "minutes": round((time.monotonic() - began) / 60, 2),
            "best_candidate": float(f[1:].max()),
            "seconds": round(time.perf_counter() - start_g, 1),
        }
        if g % eval_every == 0 or g == 1:
            unflatten(policy, mean)
            for task, mp in maps.items():
                h = mp.holdout_score(play(mp, mp.holdout, policy, mean[None]))
                row[f"holdout_{task}"] = h
                row[f"holdout_{task}_minus_mpc"] = h - MPC_HOLDOUT[task]
            torch.save(policy.state_dict(), run / "policy.pt")
            held = (row["holdout_small"] + row["holdout_full"]) / 2
            if held > best + min_delta:
                best, stale = held, 0
                torch.save(policy.state_dict(), run / "policy_best.pt")
                row["best"] = True
            else:
                stale += 1
        log.append(row)
        tracker.log(
            {
                "small_rss": row["small"]["rss"],
                "full_rss": row["full"]["rss"],
                **{k: row[k] for k in row if k.startswith("holdout_")},
            },
            step=g,
        )
        held_text = (
            f" | held out: small {row['holdout_small']:.4f} ({row['holdout_small_minus_mpc']:+.4f} vs mpc),"
            f" full {row['holdout_full']:.4f} ({row['holdout_full_minus_mpc']:+.4f})"
            if "holdout_small" in row
            else ""
        )
        print(
            f"gen {g} ({row['minutes']:.1f} min, {row['seconds']} s): mean's small {row['small']['rss']:.3f}, "
            f"full {row['full']['rss']:.3f}, best candidate {row['best_candidate']:.3f}" + held_text,
            flush=True,
        )
        (run / "log.json").write_text(json.dumps(log, indent=1))
        if patience and stale >= patience:
            print(f"early stop: the held-out score has not beaten {best:.4f} for {patience} evaluations", flush=True)
            break
    unflatten(policy, mean)
    torch.save(policy.state_dict(), run / "policy.pt")
    tracker.artifact(run / "policy_best.pt")
    tracker.end()
    print(f"done; the best held-out score (mean of small and full) {best:.4f}; the run is in {run}")


if __name__ == "__main__":
    fire.Fire(main)
