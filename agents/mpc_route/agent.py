"""agents/mpc's model-predictive control, with a learned correction for every route that follows the week's state.

The plan is agents/mpc's, line for line: the benchmark package's ``mpc_det`` baseline, made to run on the server: its
code is copied into ``sbflow/``
(scripts/vendor_mpc.py; numpy and scipy only) and the LP is solved with scipy's ``linprog`` (HiGHS) instead of highspy.
Each week:

1. the Dict observation is turned back into the package's own observation format (``to_wire``);
2. ``mpc_det`` builds the window LP: the simulator's equations over the next H weeks, on a forecast in which what is
   observed now (closures, capacities, sanctions, tariffs) persists and announced sanctions start on their week;
3. the LP is solved, and its first week becomes the action (flows, and the release of tanker cargo queued at straits).

If the LP fails, mpc_det's own naive rule plays the week.

Then every route's flow is corrected by its own number theta: the route ships (1 + theta) of the plan's flow, at most
its capacity this week (-0.2: 80 % of the plan, 0.2: 120 %; a route the plan leaves empty stays empty). theta comes
from one small network shared by every route (so the same agent fits tiny, small and full): it reads ``FEATURES``
of the route, numbers that mean the same on every map, and theta = THETA_SCALE * tanh(net(x)):

- its group (``GROUPS``: the stage of the chains, the mode, through a strait or not), one-hot;
- route_closed: the closure of the straits on it (1 - the least open; 0 without straits);
- dest_stock: log of the stock at its destination relative to that stock's recent level (an exponential mean,
  ``STOCK_MEMORY``), clipped to [-3, 3];
- queued: its commodity's cargo waiting in its lane's strait queues, as a share of that and what is at sea on it;
- at_sea: log(1 + its cargo at sea / the plan's flow on it), clipped to [0, 5];
- use: the plan's flow / the route's capacity this week;
- the whole network: the chokepoints' mean closure; the share of last week's demand the markets lost; the share of
  all cargo at sea that waits in queues; the week / T; the grids' shed base load / their base load; T / 104.

The network (``HIDDEN`` tanh units, then one output) is ``params.json`` beside this file (examples/train_route.py
writes it); without one every theta is 0 and the agent plays exactly as agents/mpc.
"""

import json
from pathlib import Path

import numpy as np
from sbflow.instance.schema import WAR_RISK_CLASSES
from sbflow.oracle.lp import solve_oracle
from sbflow.policies import lp_common as L
from sbflow.policies.mpc_det import MpcDet
from sbflow.policies.registry import PolicyContext


HERE = Path(__file__).resolve().parent
METHODS = ("highs-ds", "highs-ipm")  # linprog methods tried in order; the second only when the first fails
GROUPS = (
    "fuel by sea, direct",
    "fuel by sea, through a strait",
    "fuel by pipeline from a source",
    "terminal to grid",
    "wafers by sea, direct",
    "wafers by sea, through a strait",
    "wafers by air",
    "raw chips by sea, direct",
    "raw chips by sea, through a strait",
    "raw chips by air",
    "chips to markets by sea",
    "chips to markets by air",
)
FEATURES = tuple(f"group {g}" for g in GROUPS) + (
    "route_closed",
    "dest_stock",
    "queued",
    "at_sea",
    "use",
    "closed",
    "lost",
    "queued_all",
    "week",
    "shed",
    "map",
)
THETA_SCALE = 0.5  # theta = this * tanh(the network's output): at most +-50 % of the plan
STOCK_MEMORY = 0.2  # weight of this week in dest_stock's recent level (about 5 weeks)
HIDDEN, PARAMS = 16, None
if (HERE / "params.json").is_file():
    _doc = json.loads((HERE / "params.json").read_text())
    HIDDEN, PARAMS = int(_doc["hidden"]), np.asarray(_doc["params"], dtype=float)


def n_params(hidden: int, n_in: int = len(FEATURES)) -> int:
    return hidden * n_in + 2 * hidden + 1


def route_theta(params: np.ndarray, hidden: int, X: np.ndarray) -> np.ndarray:
    """theta of every route (a row of ``X``) under the flat ``params``: W1 (hidden x in), b1, w2, b2."""
    n_in = X.shape[1]
    if params.size != n_params(hidden, n_in):
        raise ValueError(f"params.json: {params.size} numbers, not {n_params(hidden, n_in)} for {hidden} hidden units")
    W1 = params[: hidden * n_in].reshape(hidden, n_in)
    b1 = params[hidden * n_in : hidden * n_in + hidden]
    w2 = params[hidden * n_in + hidden : hidden * n_in + 2 * hidden]
    return THETA_SCALE * np.tanh(np.tanh(X @ W1.T + b1) @ w2 + params[-1])


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


def slot_group(static: dict, edge: int, lane: int | None) -> int:
    """The ``GROUPS`` index of an action slot's route: its first edge's tail and mode, through a strait or not."""
    tail = static["nodes"]["type"][static["edges"]["tail"][edge]]
    mode = static["edges"]["mode"][edge]
    strait = lane is not None and len(static["lanes"]["chokepoints"][lane]) > 0
    if tail == "source":
        return 2 if mode == "pipeline" else 1 if strait else 0
    if tail == "terminal":
        return 3
    if tail == "material":
        return 6 if mode == "air" else 5 if strait else 4
    if tail == "fab":
        return 9 if mode == "air" else 8 if strait else 7
    if tail == "osat":
        return 11 if mode == "air" else 10
    raise ValueError(f"an action slot leaves a {tail} node: no group for it")


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
        slots, edges = self.static["action_slots"], self.static["edges"]
        self.group = np.array([slot_group(self.static, e, ln) for e, ln in zip(slots["edge"], slots["lane"])])
        self.route_edges = [[e] if ln is None else lanes["edges"][ln] for e, ln in zip(slots["edge"], slots["lane"])]
        self.u0 = np.array([np.inf if u is None else u for u in edges["u0"]], dtype=float)
        self._features_init()

    # ----- the week's features and corrections ------------------------------------------------------------------------
    def _features_init(self) -> None:
        lay, slots, lanes = self.layout, self.static["action_slots"], self.static["lanes"]
        S, G = len(slots["edge"]), len(GROUPS)
        choke = {int(c): i for i, c in enumerate(lay["chokepoints"])}
        self.slot_chokes = [
            [] if ln is None else [choke[int(c)] for c in lanes["chokepoints"][ln]] for ln in slots["lane"]
        ]
        stock = {(int(n), int(k)): i for i, (n, k) in enumerate(lay["stock_slots"])}
        heads = self.static["edges"]["head"]
        self.dest = np.array(
            [stock.get((int(heads[route[-1]]), int(k)), -1) for route, k in zip(self.route_edges, slots["k"])]
        )
        # a route's cargo: on its lane (for a route through one) or on its edge, of its commodity
        self.route_key = [
            ("lane", int(ln), int(k)) if ln is not None else ("edge", int(e), int(k))
            for e, k, ln in zip(slots["edge"], slots["k"], slots["lane"])
        ]
        self.key_slots = {}
        for i, key in enumerate(self.route_key):
            self.key_slots.setdefault(key, []).append(i)
        self.onehot = np.zeros((S, G))
        self.onehot[np.arange(S), self.group] = 1.0
        self.T = float(self.static["T"])
        self.seen = {}  # the last observed value of every field the features read (blackout weeks keep it)
        self.level = None  # the recent level of every stock slot

    def _seen(self, obs: dict, key: str) -> np.ndarray:
        val, mask = np.asarray(obs[key], dtype=float), np.asarray(obs[f"{key}.observed"], dtype=bool)
        last = self.seen.get(key)
        val = np.where(mask, val, 0.0 if last is None else last)
        self.seen[key] = val
        return val

    def _per_route(self, keys, qty) -> np.ndarray:
        """Sums of ``qty`` over the routes whose key (``route_key``) each entry's key is."""
        out = np.zeros(len(self.route_key))
        for key, q in zip(keys, qty):
            for i in self.key_slots.get(key, ()):
                out[i] += q
        return out

    def _queued(self, obs: dict) -> np.ndarray:
        if "lot_keys" in self.layout:  # small, full: a dense (lot key, arrival week) block
            rows = np.sum(obs["queue_lots.qty"] * obs["queue_lots.qty.observed"], axis=1)
            keys = [("lane", int(lane), int(k)) for _c, k, lane, _nxt in self.layout["lot_keys"]]
            return self._per_route(keys, rows)
        live = np.flatnonzero(obs["queue_lots.qty.observed"])  # tiny: the lots' own list
        keys = [("lane", int(obs["queue_lots.lane"][i]), int(obs["queue_lots.k"][i])) for i in live]
        return self._per_route(keys, obs["queue_lots.qty"][live])

    def _at_sea(self, obs: dict) -> np.ndarray:
        live = np.flatnonzero(obs["pipeline.qty.observed"])
        lane, edge, k = obs["pipeline.lane"], obs["pipeline.edge"], obs["pipeline.k"]
        keys = [("lane", int(lane[i]), int(k[i])) if lane[i] >= 0 else ("edge", int(edge[i]), int(k[i])) for i in live]
        return self._per_route(keys, obs["pipeline.qty"][live])

    def features(self, obs: dict, flows: np.ndarray, cap: np.ndarray) -> np.ndarray:
        """``FEATURES`` of every route this week (a routes x FEATURES array), the plan's ``flows`` given."""
        is_open = np.clip(self._seen(obs, "graph_now.open"), 0.0, 1.0)
        stock = np.maximum(self._seen(obs, "stock.qty"), 0.0)
        self.level = stock if self.level is None else (1 - STOCK_MEMORY) * self.level + STOCK_MEMORY * stock
        lost = self._seen(obs, "last_week.sinks.lost").sum()
        demand = self._seen(obs, "last_week.sinks.demand").sum()
        shed = self._seen(obs, "last_week.shed.qty").sum()
        base = self._seen(obs, "graph_now.grid.y_bar").sum()
        queued, at_sea = self._queued(obs), self._at_sea(obs)
        has = self.dest >= 0
        dest_stock = np.zeros(len(self.dest))
        dest_stock[has] = np.log((stock[self.dest[has]] + 1.0) / (self.level[self.dest[has]] + 1.0))
        local = np.column_stack(
            [
                [1.0 - is_open[c].min() if c else 0.0 for c in self.slot_chokes],
                np.clip(dest_stock, -3.0, 3.0),
                queued / (queued + at_sea + 1.0),
                np.clip(np.log1p(at_sea / (flows + 1.0)), 0.0, 5.0),
                np.where(np.isfinite(cap) & (cap > 0), flows / np.where(cap > 0, cap, 1.0), 0.0).clip(0.0, 1.0),
            ]
        )
        whole = [
            1.0 - is_open.mean() if is_open.size else 0.0,
            lost / demand if demand > 0 else 0.0,
            queued.sum() / (queued.sum() + at_sea.sum() + 1.0),
            float(obs["week"][0]) / self.T,
            shed / base if base > 0 else 0.0,
            self.T / 104.0,
        ]
        return np.hstack([self.onehot, local, np.tile(whole, (len(self.dest), 1))])

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

    def capacity(self, obs: dict) -> np.ndarray:
        u = np.where(obs["graph_now.u.observed"] == 1, obs["graph_now.u"], self.u0)
        return np.array([min(u[e] for e in route) for route in self.route_edges])

    def act_with(self, obs: dict, params: np.ndarray | None, hidden: int) -> dict:
        """The plan's action, its flows corrected by the network ``params`` (None: the plan as it is)."""
        action = self.to_action(self.plan(self.to_wire(obs)))
        cap = self.capacity(obs)
        X = self.features(obs, action["flows"], cap)  # every week, so blackout weeks keep the last values seen
        if params is not None and np.any(params):
            theta = route_theta(params, hidden, X)
            out = np.minimum(action["flows"] * (1.0 + theta), np.maximum(cap, action["flows"]))
            action["flows"] = out * obs["action_mask"]
        return action

    def act(self, observation):
        return self.act_with(observation, PARAMS, HIDDEN)
