"""The package's weekly simulator (``shockbench_flow.dynamics.sim.step``) on tensors: B episodes step together.

The order of a week is the package's (design §3.2): 3 arrivals join the lot book and the default release at
chokepoints (FIFO across arrival cohorts, pro rata inside one), 4 the clip of the requests (mask, joint edge cap,
shared stock pro rata) and the fleet slack, 5 dispatch, 6 arrivals, 7 supply lift, grids, fabs, OSATs, 8 demand,
9 disposal and the eight cost components. What differs: overrides and holds at chokepoints are left out (the default
release only), the float sums run in another order (the package sums with ``math.fsum`` in a fixed order), and the
clamps of tiny negative stocks are a plain ``max(0, .)``. ``examples/gpu_check.py`` compares it with the package.
"""

import torch

from sbf_starter.gpu.net import Batch, Net


COSTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
LOT_EPS = 1e-12


def _scatter(n: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Sums of ``src`` [B, m] into ``n`` bins by ``index`` [m]: [B, n]."""
    out = torch.zeros(src.shape[0], n, dtype=src.dtype, device=src.device)
    return out.index_add_(1, index, src)


def _ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """min(1, a / b) where b > 0, else 0 (the package's factors)."""
    return torch.where(b > 0, torch.clamp(a / torch.where(b > 0, b, 1.0), max=1.0), 0.0)


def _cohort(lots, open_, ures, kap, to_edge, to_chk):
    """One arrival cohort of one pool at every chokepoint (10): next-edge capacity pro rata, then throughput pro rata.
    Returns its releases per lot key [B, L] and the residual capacity [B, E] and throughput [B, C] left after it."""
    y = torch.where(open_ & (lots > 0), lots, 0.0)
    tot = y @ to_edge
    eta_u = torch.where(tot > 0, torch.clamp(ures / torch.where(tot > 0, tot, 1.0), max=1.0), 0.0)
    yu = (eta_u @ to_edge.T) * y
    tot2 = yu @ to_chk
    eta_k = torch.where(tot2 > 0, torch.clamp(kap / torch.where(tot2 > 0, tot2, 1.0), max=1.0), 0.0)
    q = (eta_k @ to_chk.T) * yu
    return q, ures - q @ to_edge, kap - eta_k * tot2


class Sim:
    """B episodes of one instance: ``reset``, then ``step(flows)`` T times, then ``salvage()``."""

    def __init__(self, net: Net, marks: Batch) -> None:
        self.n, self.m, self.B = net, marks, marks.B

    def reset(self) -> None:
        n, B, dev = self.n, self.B, self.n.device
        f64 = torch.float64
        self.t = 0
        self.stock = n.stock0.expand(B, -1).clone()
        self.pipe = torch.zeros(B, len(n.ch_edge), n.T + n.max_tau + 2, dtype=f64, device=dev)
        for ch, week, q in n.pipe0:
            self.pipe[:, ch, week] += q
        self.lots = torch.zeros(B, len(n.l_chk), n.T + 1, dtype=f64, device=dev)
        self.fab_wip = torch.zeros(B, len(n.f_input), n.T + n.pad_fab + 1, dtype=f64, device=dev)
        for fi, start, q in n.fab_wip0:
            self.fab_wip[:, fi, start + n.pad_fab] += q
        ptau = int(n.p_tau.max()) if len(n.p_tau) else 0
        self.osat_wip = torch.zeros(B, len(n.p_osat), n.T + ptau + 2, dtype=f64, device=dev)
        for row, week, q in n.osat_wip0:
            self.osat_wip[:, row, week] += q
        self.backlog = torch.zeros(B, len(n.d_slot), dtype=f64, device=dev)

    # ----- step 3: the default release at chokepoints -------------------------------------------------------------
    def _release(self, u: torch.Tensor, kappa: torch.Tensor, prohibited: torch.Tensor) -> torch.Tensor:
        """Tentative releases per (lot key, arrival week) [B, L, T + 1], cohorts in increasing arrival week, pools
        (tb, ct) inside one; the residuals of the next-edge capacity and the throughput carry over."""
        n, t = self.n, self.t
        rel = torch.zeros_like(self.lots)
        if len(n.l_chk) == 0:
            return rel
        if not hasattr(self, "_inc"):  # lot key -> next edge and -> chokepoint incidence, per pool
            f64 = torch.float64
            to_edge = torch.nn.functional.one_hot(n.l_next, n.E).to(f64)
            to_chk = torch.nn.functional.one_hot(n.l_chk, n.C).to(f64)
            self._inc = (to_edge, to_chk, [n.l_pool == b for b in (0, 1)])
        to_edge, to_chk, in_pool = self._inc
        open_ = n.l_permit & ~prohibited[:, n.l_next, n.l_k]  # [B, L]
        ures, kap = u.clone(), kappa.clone()
        live = (self.lots[:, :, : t + 1] > 0).any(dim=(0, 1)).nonzero().flatten().tolist()  # one sync per week
        for a in live:
            for b in (0, 1):
                q, ures, kap_b = _cohort(self.lots[:, :, a], open_ & in_pool[b], ures, kap[:, :, b], to_edge, to_chk)
                rel[:, :, a] += q
                kap[:, :, b] = kap_b
        return rel

    def step(self, flows: torch.Tensor) -> dict:
        """Week t + 1 under the requested ``flows`` [B, A]; returns the week's cost components [B] and their total."""
        n, m = self.n, self.m
        self.t += 1
        t, ti = self.t, self.t - 1
        S, E = n.S, n.E
        prev = self.stock
        st = self.stock.clone()
        u, prohibited = m.u[:, ti], m.prohibited[:, ti]

        # ----- 3 arrivals into chokepoints join the lot book; the default release -------------------------------
        into = n.ch_into.nonzero().flatten()
        if len(into):
            self.lots[:, :, t] += _scatter(len(n.l_chk), n.ch_key[into], self.pipe[:, into, t])
            self.pipe[:, into, t] = 0.0
        rel = self._release(u, m.kappa[:, ti], prohibited)
        rel_key = rel.sum(dim=2)  # [B, L]
        released_on = _scatter(E, n.l_next, rel_key)

        # ----- 4 clip (3)-(5), fleet slack (7) -------------------------------------------------------------------
        cap = u - released_on
        ok = n.permits[n.a_edge, n.a_k] & ~prohibited[:, n.a_edge, n.a_k]
        q = torch.where(ok & (flows > 0), flows.to(torch.float64), 0.0)
        f_edge = _ratio(torch.clamp(cap, min=0.0), _scatter(E, n.a_edge, q))
        qf = q * f_edge[:, n.a_edge]
        f_stock = _ratio(prev, _scatter(S, n.a_tail, qf))
        x = qf * f_stock[:, n.a_tail]
        totals = (
            torch.stack(
                [
                    _scatter(2, n.a_pool, x * n.a_fleet).T,
                    _scatter(2, n.l_pool, rel_key * n.l_fleet).T,
                ]
            )
            .sum(0)
            .T
        )  # [B, 2]
        scale = torch.where(totals > n.fleet_cap, n.fleet_cap / torch.where(totals > 0, totals, 1.0), 1.0)
        x = torch.where(n.a_dup, x * scale[:, n.a_pool], x)
        rel_scale = torch.where(n.l_dup, scale[:, n.l_pool], 1.0)  # [B, L]
        rel = rel * rel_scale[:, :, None]
        rel_key = rel_key * rel_scale

        # ----- 5 dispatch: requests leave stock, releases leave their lots -----------------------------------------
        st = torch.clamp(st - _scatter(S, n.a_tail, x), min=0.0)
        arrive_a = t + n.tau[n.a_edge]
        self.pipe.index_put_(
            (torch.arange(self.B, device=st.device)[:, None], n.a_channel, arrive_a), x, accumulate=True
        )
        arrive_l = t + n.tau[n.l_next]
        self.pipe.index_put_(
            (torch.arange(self.B, device=st.device)[:, None], n.l_channel, arrive_l), rel_key, accumulate=True
        )
        self.lots = self.lots - rel
        self.lots = torch.where(self.lots > LOT_EPS, self.lots, 0.0)
        chk_stock = _scatter(S, n.l_slot, self.lots.sum(dim=2))
        st = torch.where(n.chk_slot, chk_stock, st)
        x_ek = torch.zeros(self.B, E, n.K, dtype=torch.float64, device=st.device)
        x_ek.index_put_((torch.arange(self.B, device=st.device)[:, None], n.a_edge, n.a_k), x, accumulate=True)
        x_ek.index_put_((torch.arange(self.B, device=st.device)[:, None], n.l_next, n.l_k), rel_key, accumulate=True)

        # ----- 6 arrivals at every other node ----------------------------------------------------------------------
        other = (~n.ch_into).nonzero().flatten()
        st = st + _scatter(S, n.ch_slot[other], self.pipe[:, other, t])
        self.pipe[:, other, t] = 0.0

        # ----- 7 produce ----------------------------------------------------------------------------------------
        lift = torch.where(n.supply_slot, torch.clamp(torch.minimum(m.supply[:, ti], n.storage - st), min=0.0), 0.0)
        st = st + lift
        alpha, R = m.alpha_bar[:, ti], m.R[:, ti]
        phat = torch.minimum(alpha * R * n.f_cap0, st[:, n.f_input])
        p = phat.clone()
        G_bar, y_bar = m.G_bar[:, ti], m.y_bar[:, ti]
        shed = torch.zeros_like(y_bar)
        if len(n.g_voll):
            fuel_now, fuel_prev = st[:, n.g_fuel_slot], prev[:, n.g_fuel_slot]  # [B, G, Fu]
            ration = torch.where(
                n.g_rationed & (fuel_prev < n.g_threshold),
                fuel_prev / torch.where(n.g_threshold > 0, n.g_threshold, 1.0),
                1.0,
            )
            av = torch.where(n.g_fuel_live, torch.minimum(n.g_fuel_share * G_bar[:, :, None] * ration, fuel_now), 0.0)
            av_null = n.g_null_share * G_bar
            g_av = av.sum(2) + av_null  # [B, G]
            in_grid = n.f_grid >= 0
            gidx = torch.where(in_grid, n.f_grid, 0)
            e_hat = torch.where(in_grid & (R > 0), n.f_e * phat / torch.where(R > 0, R, 1.0), 0.0)  # [B, F]
            tot = _scatter(len(n.g_voll), gidx[in_grid], e_hat[:, in_grid])
            # (18) under each priority, then the grid's own
            y1 = torch.minimum(y_bar, g_av)
            f1 = _ratio(g_av - y1, tot)
            den = y_bar + tot
            eta2 = _ratio(g_av, den)
            y2 = eta2 * y_bar
            f3 = _ratio(g_av, tot)
            y3 = torch.minimum(y_bar, torch.clamp(g_av - f3 * tot, min=0.0))
            pri = n.g_priority
            y = torch.where(pri == 0, y1, torch.where(pri == 1, y2, y3))
            fac = torch.where(pri == 0, f1, torch.where(pri == 1, eta2, f3))  # E_f = fac * e_hat_f
            energy = torch.where(in_grid, fac[:, gidx] * e_hat, 0.0)
            draws = in_grid & (n.f_e > 0)
            p = torch.where(draws, torch.minimum(phat, R * energy / torch.where(n.f_e > 0, n.f_e, 1.0)), phat)
            e_sum = _scatter(len(n.g_voll), gidx[in_grid], energy[:, in_grid])
            load = torch.where(g_av > 0, (e_sum + y) / torch.where(g_av > 0, g_av, 1.0), 0.0)
            burn = av * load[:, :, None]
            live = n.g_fuel_live.flatten()
            st = torch.clamp(st - _scatter(S, n.g_fuel_slot.flatten()[live], burn.flatten(1)[:, live]), min=0.0)
            shed = y_bar - y
        st = st - _scatter(S, n.f_input, p)
        start = t + n.pad_fab
        self.fab_wip[:, :, start] += p
        hit = m.scrap[:, ti]  # [B, F]
        if bool((hit < 1.0).any()):
            weeks = torch.arange(self.fab_wip.shape[2], device=st.device)
            lo = (t - n.f_wscr + n.pad_fab)[:, None]
            window = (weeks[None, :] >= lo) & (weeks[None, :] < start)  # [F, W]
            self.fab_wip = torch.where(window[None], self.fab_wip * hit[:, :, None], self.fab_wip)
        out_idx = t - n.f_tau + n.pad_fab
        fidx = torch.arange(len(n.f_tau), device=st.device)
        st = st + _scatter(S, n.f_product, self.fab_wip[:, fidx, out_idx])
        self.fab_wip[:, fidx, out_idx] = 0.0
        if len(n.p_osat):
            st = st + _scatter(S, n.p_out, self.osat_wip[:, :, t])
            self.osat_wip[:, :, t] = 0.0
            raw = st[:, n.p_raw]  # [B, P]
            thr = m.osat_thr[:, ti][:, n.p_osat]
            raw_tot = _scatter(m.osat_thr.shape[2], n.p_osat, raw)[:, n.p_osat]
            xi = torch.where(raw_tot <= thr, raw, thr * raw / torch.where(raw_tot > 0, raw_tot, 1.0))
            st = torch.clamp(st - _scatter(S, n.p_raw, xi), min=0.0)
            prow = torch.arange(len(n.p_osat), device=st.device)
            self.osat_wip.index_put_(
                (torch.arange(self.B, device=st.device)[:, None], prow, t + n.p_tau), xi, accumulate=True
            )

        # ----- 8 serve ----------------------------------------------------------------------------------------
        demand = m.demand[:, ti]
        want = torch.where(n.d_backlog, demand + self.backlog, demand)
        served = torch.minimum(want, st[:, n.d_slot])
        st = st - _scatter(S, n.d_slot, served)
        backlog = torch.where(n.d_backlog, want - served, 0.0)
        lost = torch.where(n.d_backlog, 0.0, demand - served)

        # ----- 9 disposal and the costs --------------------------------------------------------------------------
        disposal = torch.where(n.disposal_slot & (st > n.storage), st - n.storage, 0.0)
        st = st - disposal
        c, c_wr, tariff, h_queue = m.c[:, ti], m.c_wr[:, ti], m.tariff[:, ti], m.h_queue[:, ti]
        costs = {
            "freight": (c * x_ek.sum(2)).sum(1),
            "war_risk": (c_wr * x_ek).sum((1, 2)),
            "tariff": (tariff * n.v * x_ek).sum((1, 2)),
            "holding": (n.holding * st).sum(1),
            "queue_holding": torch.where(n.chk_slot, h_queue[:, n.slot_chk, n.slot_k] * st, 0.0).sum(1),
            "shortage": (n.d_pi * (lost + backlog)).sum(1),
            "disposal": (n.disposal_cost * disposal).sum(1),
            "shed": (n.g_voll * shed).sum(1) if len(n.g_voll) else torch.zeros_like(st[:, 0]),
        }
        costs["total"] = sum(costs[k] for k in COSTS)
        self.stock, self.backlog = st, backlog
        self.last = {"x": x, "rel_key": rel_key, "lost": lost, "served": served, "shed": shed, "p": p}
        return costs

    def salvage(self) -> torch.Tensor:
        """S_T of (23) [B]: stock (queues at their chokepoint slot), cargo in transit, fab and OSAT WIP."""
        n = self.n
        s = (n.salvage * self.stock).sum(1)
        s = s + (n.ch_salvage[None, :, None] * self.pipe).sum((1, 2))
        s = s + (n.f_salvage[None, :, None] * self.fab_wip).sum((1, 2))
        s = s + (n.p_salvage[None, :, None] * self.osat_wip).sum((1, 2))
        return s
