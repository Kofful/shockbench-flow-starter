"""A policy without MPC for the tensor simulator: one small network shared by every route, so it plays any map.

Every week each route (action slot) gets ``FEATURES`` (numbers that mean the same on every map) and the network picks
a logit z; the route then requests sigmoid(z) of its capacity this week (the least capacity on its edges). z is the
network's output plus ``base``, the logit of the route's share of capacity in the nominal plan (``nominal_flow``,
what the naive rule ships), so an untrained network plays close to the naive rule.

The features come from what the agent's observation shows at the start of week t (the state at the end of t - 1 and
the instant values at t - 1): ``features`` reads them from the simulator's state here; the submission's agent reads
the same numbers from its observation (``RouteTables`` is saved with the weights, per map).
"""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn


GROUPS = 12
FEATURES = (
    *(f"group {g}" for g in range(GROUPS)),
    "capacity now / nominal capacity",
    "route closure",
    "route prohibited",
    "log1p(tail stock / scale)",
    "log1p(destination stock / scale)",
    "log1p(cargo at sea on the route / scale)",
    "log1p(queued on the route's lane / scale)",
    "nominal flow / scale",
    "tariff rate on the route",
    "log1p(destination headroom / scale)",
    "log1p(work in progress at the tail, out in 4 weeks / scale)",
    "lost share at the route's market",
    "log1p(forecast demand at the route's market / scale)",
    "announced prohibition of the route (1 / (1 + weeks to it))",
    "week / T",
    "T / 104",
    "chokepoints' mean closure",
    "lost share of last week's demand",
    "shed share of base load",
)
BASE_COLUMNS = (*range(20), *range(26, 31))  # the first policy's 25 features (gpu_rl before the 2026-10-08 additions)


def widen(state_dict: dict) -> dict:
    """A RoutePolicy state dict of the first 25 features, widened to ``FEATURES`` with zero weights on the new ones:
    the widened network plays exactly as before and can learn to use the new inputs."""
    w = state_dict["enc.0.weight"]
    if w.shape[1] == len(FEATURES):
        return state_dict
    wide = torch.zeros(w.shape[0], len(FEATURES), dtype=w.dtype, device=w.device)
    wide[:, list(BASE_COLUMNS)] = w
    return {**state_dict, "enc.0.weight": wide}


@dataclass
class RouteTables:
    """Week-invariant route tables of one map (numpy; ``to(device)`` gives torch tensors)."""

    T: int
    group: np.ndarray  # [A] in 0..11 (agents/mpc_route: slot_group)
    k: np.ndarray  # [A] commodity
    route_edges: np.ndarray  # [A, Re] edge indices, padded with E (an edge of infinite capacity)
    route_chks: np.ndarray  # [A, Rc] chokepoint ordinals, padded with C (always open)
    tail: np.ndarray  # [A] stock slot
    dest: np.ndarray  # [A] stock slot of the route's destination
    transit: np.ndarray  # [A, Ch] 1 for the channels whose cargo counts as on the route
    queue: np.ndarray  # [A, L] 1 for the lot keys on the route's lane and commodity
    cap0: np.ndarray  # [A] least nominal capacity on the route's edges
    scale: np.ndarray  # [A] nominal flow, or a quarter of cap0 for a route the nominal plan leaves empty
    nominal: np.ndarray  # [A]
    base: np.ndarray  # [A] logit of nominal / cap0, clipped to [0.02, 0.98]
    base_load: float  # sum of the grids' nominal base load
    dest_storage: np.ndarray  # [A] storage cap of the destination's stock slot (inf: none)
    route_fab: np.ndarray  # [A] fab ordinal whose product the route carries from its tail, -1 for none
    route_pkg: np.ndarray  # [A] OSAT package row whose output the route carries from its tail, -1 for none
    route_demand: np.ndarray  # [A] demand ordinal at the route's destination (a market), -1 for none

    def to(self, device: str) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                out[f.name] = torch.as_tensor(v, device=device)
            else:
                out[f.name] = v
        return out


def slot_group(inst, e: int, lane) -> int:
    """agents/mpc_route's group of an action slot: stage of the chains, mode, through a strait or not."""
    tail = inst.nodes[inst.edges[e].tail]
    mode = inst.edges[e].mode
    strait = lane is not None and len(inst.lanes[lane].chokepoints) > 0
    kind = tail.type
    if kind == "source":
        return 2 if mode == "pipeline" else 1 if strait else 0
    if kind == "terminal":
        return 3
    if kind == "material":
        return 6 if mode == "air" else 5 if strait else 4
    if kind == "fab":
        return 9 if mode == "air" else 8 if strait else 7
    if kind == "osat":
        return 11 if mode == "air" else 10
    raise ValueError(f"an action slot leaves a {kind} node")


def route_tables(inst, net) -> RouteTables:
    """``RouteTables`` of ``inst`` and its compiled ``net`` (``sbf_starter.gpu.net.compile_net``)."""
    from shockbench_flow.instance.nominal import nominal_flow

    A, E, C = len(inst.action_slots), len(inst.edges), len(inst.chokepoints)
    chk = inst.chokepoint_ordinal
    nom = {(r.first_edge, r.k, r.lane): r.flow for r in nominal_flow(inst).routes}
    edges = [[e] if lane is None else list(inst.lanes[lane].edges) for e, _k, lane in inst.action_slots]
    chks = [
        [] if lane is None else [chk[c] for c in inst.lanes[lane].chokepoints] for _e, _k, lane in inst.action_slots
    ]
    Re, Rc = max(len(x) for x in edges), max([len(x) for x in chks] + [1])
    route_edges = np.full((A, Re), E)
    route_chks = np.full((A, Rc), C)
    for i, (es, cs) in enumerate(zip(edges, chks)):
        route_edges[i, : len(es)] = es
        route_chks[i, : len(cs)] = cs
    ch_k, ch_lane = net.ch_k.cpu().numpy(), net.ch_lane.cpu().numpy()
    l_k, l_lane = net.l_k.cpu().numpy(), net.l_lane.cpu().numpy()
    a_channel = net.a_channel.cpu().numpy()
    transit = np.zeros((A, len(ch_k)))
    queue = np.zeros((A, len(l_k)))
    for i, (e, k, lane) in enumerate(inst.action_slots):
        if lane is None:
            transit[i, a_channel[i]] = 1.0
        else:
            transit[i] = (ch_lane == lane) & (ch_k == k)
            queue[i] = (l_lane == lane) & (l_k == k)
    cap0 = np.array([min(inst.edges[x].u0 for x in es) for es in edges], dtype=float)
    nominal = np.array([nom.get((e, k, lane), 0.0) for e, k, lane in inst.action_slots])
    scale = np.where(nominal > 0, nominal, 0.25 * cap0).clip(min=1e-6)
    share = np.clip(nominal / cap0, 0.02, 0.98)
    slot_of = inst.slot_index
    return RouteTables(
        T=inst.T,
        group=np.array([slot_group(inst, e, lane) for e, _k, lane in inst.action_slots]),
        k=np.array([k for _e, k, _l in inst.action_slots]),
        route_edges=route_edges,
        route_chks=route_chks,
        tail=np.array([slot_of[(inst.edges[e].tail, k)] for e, k, _l in inst.action_slots]),
        dest=np.array([slot_of[(inst.edges[es[-1]].head, k)] for es, (_e, k, _l) in zip(edges, inst.action_slots)]),
        transit=transit,
        queue=queue,
        cap0=cap0,
        scale=scale,
        nominal=nominal,
        base=np.log(share / (1 - share)),
        base_load=float(sum(inst.nodes[g].grid.base_load for g in inst.grids)),
        dest_storage=np.array(
            [
                net.storage[slot_of[(inst.edges[es[-1]].head, k)]].item()
                for es, (_e, k, _l) in zip(edges, inst.action_slots)
            ]
        ),
        route_fab=np.array(
            [
                next(
                    (
                        fi
                        for fi, f in enumerate(inst.fabs)
                        if f == inst.edges[e].tail and inst.nodes[f].fab.product == k
                    ),
                    -1,
                )
                for e, k, _l in inst.action_slots
            ]
        ),
        route_pkg=np.array(
            [
                next(
                    (
                        r
                        for r, (o, pk) in enumerate(zip(net.p_osat.tolist(), net.p_k.tolist()))
                        if inst.osats[o] == inst.edges[e].tail and pk == k
                    ),
                    -1,
                )
                for e, k, _l in inst.action_slots
            ]
        ),
        route_demand=np.array(
            [
                next((d for d, dm in enumerate(inst.demands) if dm.node == inst.edges[es[-1]].head and dm.k == k), -1)
                for es, (_e, k, _l) in zip(edges, inst.action_slots)
            ]
        ),
    )


def features(
    tb: dict,
    stock,
    transit_ch,
    queue_key,
    u_now,
    o_now,
    masked,
    lost,
    demand,
    shed,
    week: int,
    tariff,
    wip_tail,
    forecast,
    pending,
):
    """``FEATURES`` [B, A, F] (float32) of every route at the start of ``week``.

    Args:
        tb: ``RouteTables.to(device)``.
        stock: [B, S] stock at the end of last week; transit_ch: [B, Ch] cargo at sea per channel; queue_key: [B, L]
            queued cargo per lot key; u_now: [B, E] capacity per edge (instant); o_now: [B, C] open fraction per
            chokepoint (instant); masked: [B, A] the route is prohibited this week; lost, demand: [B, D] last week's;
            shed: [B, G] last week's shed base load; tariff: [B, E, K] ad valorem rates this week; wip_tail: [B, A]
            work in progress at each route's tail of its commodity, out in the next 4 weeks; forecast: [B, D, H] the
            demand forecast; pending: [B, E, K] the effective week of an announced prohibition, 0 for none.
    """
    B, A = masked.shape
    inf = torch.full((B, 1), float("inf"), dtype=u_now.dtype, device=u_now.device)
    cap = torch.cat([u_now, inf], 1)[:, tb["route_edges"]].min(2).values  # [B, A]
    opened = torch.cat([o_now, torch.ones_like(o_now[:, :1])], 1)[:, tb["route_chks"]].min(2).values
    scale = tb["scale"]
    one_hot = torch.nn.functional.one_hot(tb["group"], GROUPS).to(torch.float32).expand(B, -1, -1)
    lost_share = lost.sum(1) / torch.where(demand.sum(1) > 0, demand.sum(1), 1.0)
    whole = torch.stack(
        [
            torch.full((B,), week / tb["T"], device=u_now.device, dtype=torch.float64),
            torch.full((B,), tb["T"] / 104.0, device=u_now.device, dtype=torch.float64),
            (1 - o_now).mean(1) if o_now.shape[1] else torch.zeros(B, device=u_now.device, dtype=torch.float64),
            lost_share,
            shed.sum(1) / max(tb["base_load"], 1e-9),
        ],
        1,
    )
    local = torch.stack(
        [
            (cap / tb["cap0"]).clamp(0, 2),
            1 - opened,
            masked.to(torch.float64),
            torch.log1p(stock[:, tb["tail"]].clamp(min=0) / scale),
            torch.log1p(stock[:, tb["dest"]].clamp(min=0) / scale),
            torch.log1p((transit_ch @ tb["transit"].T).clamp(min=0) / scale),
            torch.log1p((queue_key @ tb["queue"].T).clamp(min=0) / scale),
            (tb["nominal"] / scale).expand(B, -1),
            *_new_local(tb, stock, lost, demand, week, tariff, wip_tail, forecast, pending, scale),
        ],
        2,
    )
    x = torch.cat([one_hot, local.to(torch.float32), whole[:, None, :].expand(-1, A, -1).to(torch.float32)], 2)
    return x, cap


def _new_local(tb, stock, lost, demand, week, tariff, wip_tail, forecast, pending, scale) -> list:
    """The six per-route features added on 2026-10-08 (``FEATURES`` 20..25)."""
    B = stock.shape[0]
    zero_e = torch.zeros_like(tariff[:, :1])
    rate = torch.cat([tariff, zero_e], 1)[:, tb["route_edges"], tb["k"][:, None]].sum(2).clamp(0, 3)
    headroom = (tb["dest_storage"] - stock[:, tb["dest"]]).clamp(min=0) / scale
    has = tb["route_demand"] >= 0
    d = torch.where(has, tb["route_demand"], 0)
    share = lost / torch.where(demand > 0, demand, 1.0)  # [B, D]
    market_lost = torch.where(has, share[:, d], 0.0)
    ahead = forecast[:, :, :4].mean(2)  # [B, D]
    market_ahead = torch.where(has, ahead[:, d], 0.0)
    eff = torch.cat([pending, torch.zeros_like(pending[:, :1])], 1)[:, tb["route_edges"], tb["k"][:, None]]
    soon = torch.where(eff > 0, 1.0 / (1.0 + (eff.to(torch.float64) - week).clamp(min=0)), 0.0).max(2).values
    return [
        rate.to(torch.float64),
        torch.log1p(headroom.clamp(max=147.0)),  # log1p(147) ~ 5: an unbounded store reads 5
        torch.log1p(wip_tail.clamp(min=0) / scale),
        market_lost,
        torch.log1p(market_ahead.clamp(min=0) / scale),
        soon.expand(B, -1) if soon.dim() == 1 else soon,
    ]


def wip_next(sim, tb: dict, week: int) -> torch.Tensor:
    """[B, A]: work in progress at each route's tail of its commodity, out in weeks week .. week + 3."""
    n = sim.n
    B, A = sim.B, len(tb["route_fab"])
    out = torch.zeros(B, A, dtype=torch.float64, device=sim.stock.device)
    W = sim.fab_wip.shape[2]
    lo = week - n.f_tau + n.pad_fab  # [F]: the start index of WIP out in this week
    idx = torch.arange(W, device=sim.stock.device)
    window = (idx[None, :] >= lo[:, None]) & (idx[None, :] <= lo[:, None] + 3)  # [F, W]
    fab_next = (sim.fab_wip * window[None]).sum(2)  # [B, F]
    has = tb["route_fab"] >= 0
    out = torch.where(has, fab_next[:, torch.where(has, tb["route_fab"], 0)], out)
    if sim.osat_wip.shape[1]:
        pkg_next = sim.osat_wip[:, :, week : week + 4].sum(2)  # [B, P]
        has = tb["route_pkg"] >= 0
        out = torch.where(has, pkg_next[:, torch.where(has, tb["route_pkg"], 0)], out)
    return out


def route_masked(tb: dict, prohibited: torch.Tensor) -> torch.Tensor:
    """[B, A]: an edge of the route is prohibited for its commodity this week (the observation's slot mask)."""
    pad = torch.zeros_like(prohibited[:, :1])
    return torch.cat([prohibited, pad], 1)[:, tb["route_edges"], tb["k"][:, None]].any(2)


def sim_inputs(sim, tb: dict, week: int) -> tuple:
    """``features``' inputs from the tensor simulator before it plays ``week``."""
    n, m, ti = sim.n, sim.m, week - 1
    dev, f64 = sim.stock.device, torch.float64
    if week > 1:
        lost, demand, shed = sim.last["lost"], m.demand[:, ti - 1], sim.last["shed"]
    else:
        lost = demand = torch.zeros(sim.B, len(n.d_slot), dtype=f64, device=dev)
        shed = torch.zeros(sim.B, m.y_bar.shape[2], dtype=f64, device=dev)
    masked = route_masked(tb, m.prohibited[:, ti])
    head = (sim.stock, sim.pipe.sum(2), sim.lots.sum(2), m.u_now[:, ti], m.o_now[:, ti], masked, lost, demand, shed)
    return head, (m.tariff[:, ti], wip_next(sim, tb, week), m.forecast[:, ti], m.pending[:, ti])


def sim_features(sim, tb: dict, week: int):
    """``features`` (and the routes' capacity) read from the tensor simulator before it plays ``week``."""
    head, extra = sim_inputs(sim, tb, week)
    return features(tb, *head, week, *extra)


def neighbours(h: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """For every route, the mean of ``h`` [G, A, H] over the routes sharing its ``index`` [A] (a stock slot)."""
    G, A, H = h.shape
    sums = torch.zeros(G, n, H, dtype=h.dtype, device=h.device).index_add_(1, index, h)
    count = torch.bincount(index, minlength=n).clamp(min=1).to(h.dtype)
    return (sums / count[None, :, None])[:, index]


class RoutePolicy(nn.Module):
    """Per-route encoder; the action head gives each route's logit, the value head reads the pooled routes.

    With ``pool``, the routes look at their neighbours before deciding (a small graph network): each route's
    description h is mixed with the mean description of the routes into the same destination and out of the same
    tail, h + tanh(W_h h + W_d dest + W_s tail + b), whose weights start at zero, so a pooled network loaded from an
    unpooled one plays exactly as it did and learns what to share."""

    def __init__(self, hidden: int = 64, log_std: float = -1.0, pool: bool = False) -> None:
        super().__init__()
        F = len(FEATURES)
        self.enc = nn.Sequential(nn.Linear(F, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pool = pool
        if pool:
            self.mix_h = nn.Linear(hidden, hidden)
            self.mix_d = nn.Linear(hidden, hidden, bias=False)
            self.mix_s = nn.Linear(hidden, hidden, bias=False)
            for layer in (self.mix_h, self.mix_d, self.mix_s):
                nn.init.zeros_(layer.weight)
            nn.init.zeros_(self.mix_h.bias)
        self.act = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.act.weight, gain=0.01)
        nn.init.zeros_(self.act.bias)
        self.value = nn.Sequential(nn.Linear(2 * hidden + 5, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.tensor(float(log_std)))

    def describe(self, x: torch.Tensor, tb: dict | None = None) -> torch.Tensor:
        """Each route's description [B, A, H], after the neighbours' mix when pooled (``tb``: its RouteTables)."""
        h = self.enc(x)
        if self.pool:
            n = int(max(tb["dest"].max(), tb["tail"].max())) + 1
            dest, tail = neighbours(h, tb["dest"], n), neighbours(h, tb["tail"], n)
            h = h + torch.tanh(self.mix_h(h) + self.mix_d(dest) + self.mix_s(tail))
        return h

    def forward(self, x: torch.Tensor, base: torch.Tensor, tb: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean logits [B, A] and values [B] for features ``x`` [B, A, F]."""
        h = self.describe(x, tb)
        mean = self.act(h).squeeze(-1) + base
        pooled = torch.cat([h.mean(1), h.max(1).values, x[:, 0, -5:]], 1)
        return mean, self.value(pooled).squeeze(-1)
