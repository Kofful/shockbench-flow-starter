"""One instance as index tables, and a batch of its episodes as tensors, for ``sbf_starter.gpu.sim``.

The package's simulator (``shockbench_flow.dynamics.sim.step``) keeps lists of shipments and lots; here every
quantity has a fixed place so that many episodes step together:

- a *channel* is a stream of shipments (edge, commodity, lane or -1): every action slot's, every chokepoint release's
  and every initial shipment's; the pipeline holds each channel's cargo by arrival week;
- a *lot key* is (chokepoint, commodity, lane, next edge): the lot book holds each key's cargo by arrival week, so the
  lots that arrived in one week on one key are one number (the default release treats them pro rata, so that is the
  same release);
- the weekly marks of an episode (``shockbench_flow.marks.compute_marks``: capacities, closures, prices, demand,
  outages) are drawn on the CPU by the package and stacked into ``Batch``.

Overrides and holds at chokepoints (``release_mode``, ``override_qty``) are not modelled: the default release only.
"""

from dataclasses import dataclass

import numpy as np
import torch


PRIORITIES = ("base_first", "proportional", "industrial_first")


def _t(x, dtype=torch.float64, device="cpu") -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


@dataclass
class Net:
    """Week-invariant index tables of one instance (every tensor on ``device``)."""

    T: int
    E: int
    S: int
    A: int
    K: int
    C: int
    # edges
    tau: torch.Tensor  # [E] long
    permits: torch.Tensor  # [E, K] bool: k in K_e
    # stock slots
    storage: torch.Tensor  # [S]
    holding: torch.Tensor  # [S], 0 at chokepoint slots
    salvage: torch.Tensor  # [S], 0 at supply slots
    disposal_cost: torch.Tensor  # [S]
    supply_slot: torch.Tensor  # [S] bool
    chk_slot: torch.Tensor  # [S] bool
    disposal_slot: torch.Tensor  # [S] bool: neither chokepoint nor supply
    slot_chk: torch.Tensor  # [S] long: chokepoint ordinal of a chokepoint slot, else 0
    slot_k: torch.Tensor  # [S] long
    # action slots
    a_edge: torch.Tensor  # [A] long
    a_k: torch.Tensor  # [A] long
    a_tail: torch.Tensor  # [A] long: the stock slot (tail, k) it draws on
    a_channel: torch.Tensor  # [A] long
    a_fleet: torch.Tensor  # [A]: sum of Delta tau over the fleet-slack terms it matches
    a_dup: torch.Tensor  # [A] bool: it matches a term
    a_pool: torch.Tensor  # [A] long
    # channels
    ch_edge: torch.Tensor  # [Ch] long
    ch_k: torch.Tensor  # [Ch] long
    ch_lane: torch.Tensor  # [Ch] long, -1 off lanes
    ch_into: torch.Tensor  # [Ch] bool: into a chokepoint
    ch_key: torch.Tensor  # [Ch] long: its lot key (into a chokepoint), else 0
    ch_slot: torch.Tensor  # [Ch] long: the head's stock slot (not into a chokepoint), else 0
    ch_salvage: torch.Tensor  # [Ch]: nu of the head, what cargo in transit is worth at T
    # lot keys
    l_chk: torch.Tensor  # [L] long: chokepoint ordinal
    l_k: torch.Tensor  # [L] long
    l_lane: torch.Tensor  # [L] long
    l_pool: torch.Tensor  # [L] long
    l_next: torch.Tensor  # [L] long: next edge
    l_slot: torch.Tensor  # [L] long: the chokepoint's stock slot (c, k)
    l_channel: torch.Tensor  # [L] long: the channel a release rides (next edge, k, lane)
    l_permit: torch.Tensor  # [L] bool: k in K of the next edge
    l_fleet: torch.Tensor  # [L]
    l_dup: torch.Tensor  # [L] bool
    fleet_cap: torch.Tensor  # [2]
    # grids (fuels padded to the widest grid)
    g_fuel_slot: torch.Tensor  # [G, Fu] long
    g_fuel_share: torch.Tensor  # [G, Fu], 0 on padding
    g_fuel_live: torch.Tensor  # [G, Fu] bool
    g_rationed: torch.Tensor  # [G, Fu] bool
    g_threshold: torch.Tensor  # [G, Fu]: psi I-bar of a rationed fuel
    g_null_share: torch.Tensor  # [G]: the share of the segment without fuel
    g_priority: torch.Tensor  # [G] long, index into PRIORITIES
    g_voll: torch.Tensor  # [G]
    # fabs
    f_input: torch.Tensor  # [F] long
    f_product: torch.Tensor  # [F] long
    f_cap0: torch.Tensor  # [F]
    f_e: torch.Tensor  # [F]
    f_tau: torch.Tensor  # [F] long
    f_wscr: torch.Tensor  # [F] long
    f_grid: torch.Tensor  # [F] long, -1 without a grid
    f_salvage: torch.Tensor  # [F]: nu of the wafer input
    # OSAT packages (one row per raw -> packaged commodity of an OSAT)
    p_osat: torch.Tensor  # [P] long
    p_raw: torch.Tensor  # [P] long
    p_out: torch.Tensor  # [P] long: the packaged commodity's stock slot
    p_k: torch.Tensor  # [P] long: the packaged commodity
    p_tau: torch.Tensor  # [P] long
    p_salvage: torch.Tensor  # [P]: nu of the raw input
    # demands
    d_slot: torch.Tensor  # [D] long
    d_backlog: torch.Tensor  # [D] bool
    d_pi: torch.Tensor  # [D]
    v: torch.Tensor  # [K] customs value
    # initial state
    stock0: torch.Tensor  # [S]
    pipe0: list  # (channel, arrival week, qty)
    fab_wip0: list  # (fab, start week, qty)
    osat_wip0: list  # (package row, out week, qty)
    pad_fab: int  # offset of the fab WIP's week axis (start weeks from 1 - max tau)
    max_tau: int
    device: str


def compile_net(inst, device: str = "cpu") -> Net:
    """The index tables of ``inst`` (a ``shockbench_flow`` Instance)."""
    from shockbench_flow.dynamics.clip import dup_terms, fleet_caps
    from shockbench_flow.dynamics.sim import initial_stock

    E, S, K, C = len(inst.edges), len(inst.stock_slots), len(inst.commodities), len(inst.chokepoints)
    chk, supply = inst.chokepoint_ordinal, set(inst.supply_nodes)
    slot_of = inst.slot_index
    pool = inst.commodity_pool
    terms = dup_terms(inst)

    def fleet(e: int, lane) -> tuple[float, bool]:
        hit = [dtau for ln, dtau in terms.get(e, ()) if ln is None or ln == lane]
        return float(sum(hit)), bool(hit)

    def nu(node: int, k: int) -> float:
        if node in supply:
            return 0.0
        s = slot_of.get((node, k))
        return 0.0 if s is None else inst.stock_slots[s].salvage

    channels: dict[tuple[int, int, int], int] = {}
    keys: dict[tuple[int, int, int, int], int] = {}

    def key_of(c: int, k: int, lane: int, nxt: int) -> int:
        return keys.setdefault((c, k, lane, nxt), len(keys))

    def channel(e: int, k: int, lane) -> int:
        lane = -1 if lane is None else int(lane)
        ch = channels.setdefault((e, k, lane), len(channels))
        head = inst.edges[e].head
        if head in chk:  # its lot key, and the channel its release rides, tandem lanes included
            nxt = inst.lane_next_edge(lane, e)
            if (c_k := (head, k, lane, nxt)) not in keys:
                key_of(*c_k)
                channel(nxt, k, lane)
        return ch

    a_channel = [channel(e, k, lane) for e, k, lane in inst.action_slots]
    pipe0 = [(channel(s.edge, s.k, s.lane), s.arrival_week, s.qty) for s in inst.initial_state.pipeline]
    if inst.initial_state.queue_lots:
        raise NotImplementedError("initial queue lots: none of tiny, small, full has any")

    ch_list = sorted(channels, key=channels.get)
    key_list = sorted(keys, key=keys.get)
    a_fleet = [fleet(e, lane) for e, _k, lane in inst.action_slots]
    l_fleet = [fleet(nxt, lane) for _c, _k, lane, nxt in key_list]

    grids = [inst.nodes[g].grid for g in inst.grids]
    Fu = max(len(g.fuels) for g in grids) if grids else 1
    g_fuel_slot = np.zeros((len(grids), Fu), dtype=np.int64)
    g_fuel_share = np.zeros((len(grids), Fu))
    g_fuel_live = np.zeros((len(grids), Fu), dtype=bool)
    g_rationed = np.zeros((len(grids), Fu), dtype=bool)
    g_threshold = np.zeros((len(grids), Fu))
    for gi, (g, grid) in enumerate(zip(inst.grids, grids)):
        for j, k in enumerate(grid.fuels):
            g_fuel_slot[gi, j] = slot_of[(g, k)]
            g_fuel_share[gi, j] = grid.shares[k]
            g_fuel_live[gi, j] = True
            g_rationed[gi, j] = k == grid.rationed
            g_threshold[gi, j] = inst.params.psi * grid.ibar[k] if k == grid.rationed else 0.0

    fabs = [inst.nodes[f].fab for f in inst.fabs]
    rows = [
        (oi, slot_of[(o, raw)], slot_of[(o, pk)], pk, inst.nodes[o].osat.tau, nu(o, raw))
        for oi, o in enumerate(inst.osats)
        for raw, pk in sorted(inst.nodes[o].osat.packages.items())
    ]
    row_of = {(r[0], r[3]): i for i, r in enumerate(rows)}
    max_fab_tau = max(f.tau for f in fabs)
    fab_wip0 = []
    for w in inst.initial_state.fab_wip:
        fi = inst.fab_ordinal[w.node]
        fab_wip0.append((fi, w.out_week - fabs[fi].tau, w.qty))
    osat_wip0 = [(row_of[(inst.osat_ordinal[w.node], w.k)], w.out_week, w.qty) for w in inst.initial_state.osat_wip]

    dev = device
    L = lambda x: _t(x, torch.long, dev)  # noqa: E731
    Fl = lambda x: _t(x, torch.float64, dev)  # noqa: E731
    B_ = lambda x: _t(x, torch.bool, dev)  # noqa: E731
    slots = inst.stock_slots
    return Net(
        T=inst.T,
        E=E,
        S=S,
        A=len(inst.action_slots),
        K=K,
        C=C,
        tau=L([e.tau for e in inst.edges]),
        permits=B_([[k in e.K for k in range(K)] for e in inst.edges]),
        storage=Fl([np.inf if sl.storage is None else sl.storage for sl in slots]),
        holding=Fl([0.0 if sl.node in chk else sl.holding for sl in slots]),
        salvage=Fl([nu(sl.node, sl.k) for sl in slots]),
        disposal_cost=Fl([inst.commodities[sl.k].disposal_cost for sl in slots]),
        supply_slot=B_([sl.node in supply for sl in slots]),
        chk_slot=B_([sl.node in chk for sl in slots]),
        disposal_slot=B_([sl.node not in chk and sl.node not in supply for sl in slots]),
        slot_chk=L([chk.get(sl.node, 0) for sl in slots]),
        slot_k=L([sl.k for sl in slots]),
        a_edge=L([e for e, _k, _l in inst.action_slots]),
        a_k=L([k for _e, k, _l in inst.action_slots]),
        a_tail=L([slot_of.get((inst.edges[e].tail, k), 0) for e, k, _l in inst.action_slots]),
        a_channel=L(a_channel),
        a_fleet=Fl([f for f, _d in a_fleet]),
        a_dup=B_([d for _f, d in a_fleet]),
        a_pool=L([pool[k] for _e, k, _l in inst.action_slots]),
        ch_edge=L([e for e, _k, _l in ch_list]),
        ch_k=L([k for _e, k, _l in ch_list]),
        ch_lane=L([lane for _e, _k, lane in ch_list]),
        ch_into=B_([inst.edges[e].head in chk for e, _k, _l in ch_list]),
        ch_key=L(
            [
                keys[(inst.edges[e].head, k, lane, inst.lane_next_edge(lane, e))] if inst.edges[e].head in chk else 0
                for e, k, lane in ch_list
            ]
        ),
        ch_slot=L([0 if inst.edges[e].head in chk else slot_of[(inst.edges[e].head, k)] for e, k, _l in ch_list]),
        ch_salvage=Fl([nu(inst.edges[e].head, k) for e, k, _l in ch_list]),
        l_chk=L([chk[c] for c, _k, _l, _n in key_list]),
        l_k=L([k for _c, k, _l, _n in key_list]),
        l_lane=L([lane for _c, _k, lane, _n in key_list]),
        l_pool=L([pool[k] for _c, k, _l, _n in key_list]),
        l_next=L([n for _c, _k, _l, n in key_list]),
        l_slot=L([slot_of[(c, k)] for c, k, _l, _n in key_list]),
        l_channel=L([channels[(n, k, lane)] for _c, k, lane, n in key_list]),
        l_permit=B_([k in inst.edges[n].K for _c, k, _l, n in key_list]),
        l_fleet=Fl([f for f, _d in l_fleet]),
        l_dup=B_([d for _f, d in l_fleet]),
        fleet_cap=Fl(fleet_caps(inst)),
        g_fuel_slot=L(g_fuel_slot),
        g_fuel_share=Fl(g_fuel_share),
        g_fuel_live=B_(g_fuel_live),
        g_rationed=B_(g_rationed),
        g_threshold=Fl(g_threshold),
        g_null_share=Fl([g.shares.get(None, 0.0) for g in grids]),
        g_priority=L([PRIORITIES.index(g.priority) for g in grids]),
        g_voll=Fl([g.voll for g in grids]),
        f_input=L([slot_of[(f, fab.input)] for f, fab in zip(inst.fabs, fabs)]),
        f_product=L([slot_of[(f, fab.product)] for f, fab in zip(inst.fabs, fabs)]),
        f_cap0=Fl([fab.cap0 for fab in fabs]),
        f_e=Fl([fab.e for fab in fabs]),
        f_tau=L([fab.tau for fab in fabs]),
        f_wscr=L([fab.w_scr for fab in fabs]),
        f_grid=L([inst.grid_ordinal[fab.grid] if fab.grid is not None else -1 for fab in fabs]),
        f_salvage=Fl([nu(f, fab.input) for f, fab in zip(inst.fabs, fabs)]),
        p_osat=L([r[0] for r in rows]),
        p_raw=L([r[1] for r in rows]),
        p_out=L([r[2] for r in rows]),
        p_k=L([r[3] for r in rows]),
        p_tau=L([r[4] for r in rows]),
        p_salvage=Fl([r[5] for r in rows]),
        d_slot=L([slot_of[(d.node, d.k)] for d in inst.demands]),
        d_backlog=B_([d.backlog for d in inst.demands]),
        d_pi=Fl([d.pi for d in inst.demands]),
        v=Fl([com.v for com in inst.commodities]),
        stock0=Fl(initial_stock(inst)),
        pipe0=pipe0,
        fab_wip0=fab_wip0,
        osat_wip0=osat_wip0,
        pad_fab=max_fab_tau,
        max_tau=int(max(e.tau for e in inst.edges)),
        device=dev,
    )


@dataclass
class Batch:
    """The weekly marks of B episodes, week index t - 1 on axis 1 (every tensor on the Net's device)."""

    B: int
    u: torch.Tensor  # [B, T, E]
    c: torch.Tensor  # [B, T, E]
    kappa: torch.Tensor  # [B, T, C, 2]
    supply: torch.Tensor  # [B, T, S]
    G_bar: torch.Tensor  # [B, T, G]
    y_bar: torch.Tensor  # [B, T, G]
    R: torch.Tensor  # [B, T, F]
    alpha_bar: torch.Tensor  # [B, T, F]
    osat_thr: torch.Tensor  # [B, T, O]: thr_i R_osat_i(t)
    demand: torch.Tensor  # [B, T, D]
    prohibited: torch.Tensor  # [B, T, E, K] bool
    tariff: torch.Tensor  # [B, T, E, K]
    c_wr: torch.Tensor  # [B, T, E, K]
    h_queue: torch.Tensor  # [B, T, C, K]
    scrap: torch.Tensor  # [B, T, F]: the factor (1 - severity) of the hits whose onset week is t, multiplied


def batch_marks(inst, marks_list, device: str = "cpu") -> Batch:
    """``Batch`` of ``marks_list`` (``WeeklyMarks`` of one instance each)."""
    from shockbench_flow.marks import osat_throughput

    def st(name, dtype=np.float64):
        return torch.as_tensor(np.stack([np.asarray(getattr(m, name), dtype=dtype) for m in marks_list]), device=device)

    T, F = inst.T, len(inst.fabs)
    scrap = np.ones((len(marks_list), T, F))
    for b, m in enumerate(marks_list):
        for hit in m.fab_hits:
            if 1 <= hit.onset_week <= T:
                scrap[b, hit.onset_week - 1, hit.fab] *= 1.0 - hit.severity
    return Batch(
        B=len(marks_list),
        u=st("u"),
        c=st("c"),
        kappa=st("kappa"),
        supply=st("supply"),
        G_bar=st("G_bar"),
        y_bar=st("y_bar"),
        R=st("R"),
        alpha_bar=st("alpha_bar"),
        osat_thr=torch.as_tensor(
            np.stack([osat_throughput(inst, m.R_osat) for m in marks_list]), dtype=torch.float64, device=device
        ),
        demand=st("demand"),
        prohibited=st("prohibited", bool),
        tariff=st("tariff"),
        c_wr=st("c_wr"),
        h_queue=st("h_queue"),
        scrap=torch.as_tensor(scrap, device=device),
    )
