"""Model-predictive control: every week, plan the next weeks with a linear program and ship the plan's first week.

This is the benchmark package's ``mpc_det`` baseline, made to run on the server: its code is copied into ``sbflow/``
(scripts/vendor_mpc.py; numpy and scipy only) and the LP is solved with scipy's ``linprog`` (HiGHS) instead of highspy.
Each week:

1. the Dict observation is turned back into the package's own observation format (``to_wire``);
2. ``mpc_det`` builds the window LP: the simulator's equations over the next H weeks, on a forecast in which what is
   observed now (closures, capacities, sanctions, tariffs) persists and announced sanctions start on their week;
3. the LP is solved, and its first week becomes the action (flows, and the release of tanker cargo queued at straits).

If the LP fails, mpc_det's own naive rule plays the week. It reads no warnings or messages: that is the next step.
"""

from pathlib import Path

import numpy as np
from sbflow.instance.schema import WAR_RISK_CLASSES
from sbflow.oracle.lp import solve_oracle
from sbflow.policies import lp_common as L
from sbflow.policies.mpc_det import MpcDet
from sbflow.policies.registry import PolicyContext


HERE = Path(__file__).resolve().parent
METHODS = ("highs-ds", "highs-ipm")  # linprog methods tried in order; the second only when the first fails


def _ints(a) -> list:
    return [int(x) for x in a]


def _column(obs: dict, key: str, rows, cast) -> list:
    """Entries ``rows`` of ``obs[key]`` as plain values, None where unobserved."""
    vals, seen = obs[key], obs[f"{key}.observed"]
    return [cast(vals[i]) if seen[i] else None for i in rows]


def _block(obs: dict, block: str, live_key: str, columns: dict) -> dict | None:
    """A padded list block (pipeline, wip, ...) as the package's dict of lists: its live rows only."""
    rows = np.flatnonzero(obs[f"{block}.{live_key}.observed"])
    return {name: _column(obs, f"{block}.{name}", rows, cast) for name, cast in columns.items()}


class Agent:
    def __init__(self, config=None):
        self.static, self.layout = config["static"], config["layout"]
        self.spaces = config["spaces"]["action"]
        self.seed = config["policy_seed"]
        self.policy = MpcDet(context=PolicyContext())  # reset on week 1's observation (its naive fallback reads it)
        self.inst = None
        slots = self.static["override_slots"]
        pairs = {tuple(p): i for i, p in enumerate(self.layout["release_pairs"])}
        self.override_pair = [pairs[(c, k)] for c, k in zip(slots["chokepoint"], slots["k"])]
        self.pair_index = pairs
        # on small and full the queued lots are a dense (lot key, arrival week) block: per key, the lane's edge into it
        lanes = self.static["lanes"]
        heads = self.static["edges"]["head"]
        keys = self.layout.get("lot_keys", [])
        self.lot_entry = [next(e for e in lanes["edges"][lane] if heads[e] == c) for c, _k, lane, _nxt in keys]
        self.tau0 = self.static["edges"]["tau0"]

    # ----- the observation, back in the package's format --------------------------------------------------------------
    def to_wire(self, obs: dict) -> dict:
        lay = self.layout
        t = int(obs["week"][0])
        seen = obs["stock.qty.observed"]
        stock = [(n, k, float(q)) for (n, k), q, s in zip(lay["stock_slots"], obs["stock.qty"], seen) if s]
        seen = obs["backlog.qty.observed"]
        backlog = [(n, k, float(q)) for (n, k), q, s in zip(lay["demands"], obs["backlog.qty"], seen) if s]
        wire = {
            "week": t,
            "stock": {"node": [s[0] for s in stock], "k": [s[1] for s in stock], "qty": [s[2] for s in stock]},
            "backlog": {"node": [b[0] for b in backlog], "k": [b[1] for b in backlog], "qty": [b[2] for b in backlog]},
            "pipeline": _block(
                obs, "pipeline", "qty", {"edge": int, "k": int, "lane": int, "qty": float, "arrival_week": int}
            ),
            "wip": _block(obs, "wip", "qty", {"node": int, "k": int, "qty": float, "out_week": int}),
            "queue_lots": self._lots(obs),
            "graph_now": self._graph(obs),
            "pending_prohibitions": _block(
                obs, "pending_prohibitions", "edge", {"edge": int, "k": int, "effective_week": int}
            ),
            "demand_forecast": self._forecast(obs),
        }
        return wire

    def _lots(self, obs: dict) -> dict:
        if "lot_keys" not in self.layout:  # tiny: the lots' own list
            cols = ("chokepoint", "k", "lane", "next_edge", "arrival_week", "dispatch_week", "entry_edge", "lot_id")
            out = _block(obs, "queue_lots", "qty", {c: int for c in cols})
            out["qty"] = _column(obs, "queue_lots.qty", np.flatnonzero(obs["queue_lots.qty.observed"]), float)
            return out
        rows, weeks = np.nonzero(obs["queue_lots.qty.observed"])  # dense: one lot per observed (key, arrival week)
        out = {c: [] for c in ("chokepoint", "k", "lane", "next_edge", "arrival_week", "dispatch_week", "entry_edge")}
        out["qty"] = []
        for i, w in zip(rows, weeks):
            c, k, lane, nxt = self.layout["lot_keys"][i]
            entry = self.lot_entry[i]
            for key, v in (("chokepoint", c), ("k", k), ("lane", lane), ("next_edge", nxt), ("entry_edge", entry)):
                out[key].append(int(v))
            out["arrival_week"].append(int(w) + 1)
            out["dispatch_week"].append(int(w) + 1 - int(self.tau0[entry]))  # not read by the LP (lp_common)
            out["qty"].append(float(obs["queue_lots.qty"][i, w]))
        return out

    def _graph(self, obs: dict) -> dict | None:
        if not obs["graph_now.open.observed"].any() and not obs["graph_now.c.observed"].any():
            return None  # a blackout week: the memory keeps the last week seen
        lay = self.layout
        E = len(obs["graph_now.u"])
        C = len(lay["chokepoints"])
        pro_e, pro_k = np.nonzero(obs["graph_now.prohibited"] * obs["graph_now.prohibited.observed"])
        tar_e, tar_k = np.nonzero(obs["graph_now.tariff.observed"])
        sup = np.flatnonzero(obs["graph_now.supply.avail.observed"])
        wr = obs["graph_now.war_risk"]
        return {
            "u": _column(obs, "graph_now.u", range(E), float),
            "c": _column(obs, "graph_now.c", range(E), float),
            "tau": _column(obs, "graph_now.tau", range(E), int),
            "open": _column(obs, "graph_now.open", range(C), float),
            "kappa": {p: _column(obs, f"graph_now.kappa.{p}", range(C), float) for p in ("tb", "ct")},
            "war_risk": [
                WAR_RISK_CLASSES[int(wr[i])] if obs["graph_now.war_risk.observed"][i] else None for i in range(C)
            ],
            "supply": {
                "node": [lay["supply_slots"][i][0] for i in sup],
                "k": [lay["supply_slots"][i][1] for i in sup],
                "avail": [float(obs["graph_now.supply.avail"][i]) for i in sup],
            },
            "fab": self._by_node(obs, "graph_now.fab", lay["fabs"], ("R", "alpha_bar", "cap_eff")),
            "grid": self._by_node(obs, "graph_now.grid", lay["grids"], ("G_bar", "y_bar")),
            "osat": self._by_node(obs, "graph_now.osat", lay["osats"], ("R", "thr_eff")),
            "prohibited": {"edge": _ints(pro_e), "k": _ints(pro_k)},
            "tariff": {
                "edge": _ints(tar_e),
                "k": _ints(tar_k),
                "rate": [float(obs["graph_now.tariff"][e, k]) for e, k in zip(tar_e, tar_k)],
            },
        }

    @staticmethod
    def _by_node(obs: dict, prefix: str, nodes: list, fields: tuple) -> dict:
        out = {"node": [int(n) for n in nodes]}
        for f in fields:
            out[f] = _column(obs, f"{prefix}.{f}", range(len(nodes)), float)
        return out

    def _forecast(self, obs: dict) -> dict:
        q, seen = obs["demand_forecast.qty"], obs["demand_forecast.qty.observed"]
        d, h = np.nonzero(seen)
        dem = self.layout["demands"]
        return {
            "node": [dem[i][0] for i in d],
            "k": [dem[i][1] for i in d],
            "h": _ints(h),
            "qty": [float(q[i, j]) for i, j in zip(d, h)],
        }

    # ----- the plan ---------------------------------------------------------------------------------------------------
    def plan(self, wire: dict) -> dict:
        """mpc_det's week (``MpcDet.act``) with the window LP solved by scipy's linprog."""
        p, t = self.policy, int(wire["week"])
        if self.inst is None:
            p.reset(self.static, wire, self.seed)
            self.inst = p._inst
        inst = self.inst
        p._memory.update(inst, wire)
        H_t = L.window_length(p._H, t, inst.T)
        model = L.rolled_lp(inst, wire, p._window_arrays(inst, wire, H_t), H_t, planning_rules=p.params.planning_rules)
        for method in METHODS:
            res = solve_oracle(model, method, fallback=False)
            if res.status == 0 and res.x is not None and np.all(np.isfinite(res.x)):
                return L.week1_action(inst, model, res.x, wire, L.prohibited_now(p._memory, t))
        return p._fallback.act(wire)

    def to_action(self, wire_action: dict) -> dict:
        flows = np.zeros(self.spaces["flows"]["shape"])
        f = wire_action["flows"]
        flows[f["slot"]] = f["qty"]
        override_qty = np.zeros(self.spaces["override_qty"]["shape"])
        release_mode = np.zeros(self.spaces["release_mode"]["shape"], dtype=np.int64)
        ov = wire_action.get("overrides")
        if ov:
            override_qty[ov["slot"]] = ov["qty"]
            for o in ov["slot"]:
                release_mode[self.override_pair[o]] = 1
        hold = wire_action.get("hold")
        if hold:
            for c, k in zip(hold["chokepoint"], hold["k"]):
                release_mode[self.pair_index[(c, k)]] = 2
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}

    def act(self, observation):
        return self.to_action(self.plan(self.to_wire(observation)))
