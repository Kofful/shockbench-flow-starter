"""History-aware receding-horizon routing heuristic, using public information only.

The local LP approximates automatic production and future queue execution; it
is not the clairvoyant oracle. All dimensions come from config. Replan weekly.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


HERE = Path(__file__).resolve().parent
PARAMS = {
    "fraction": 1.0,
    "closure_power": 1.0,
    "horizon": 24,
    "solver_seconds": 0.65,
    "safety_stock": 0.6,
    "warning_gain": 0.12,
    "terminal_gain": 0.08,
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())


class _LP:
    def __init__(self):
        self.c, self.bounds, self.eq, self.ub = [], [], [], []

    def var(self, cost=0.0, cap=None):
        i = len(self.c)
        self.c.append(float(cost) / 1e6)
        self.bounds.append((0.0, None if cap is None else max(0.0, float(cap))))
        return i

    def row(self, terms, rhs, equality=False):
        (self.eq if equality else self.ub).append((terms, float(rhs)))

    def matrix(self, rows):
        rr, cc, vv = [], [], []
        for i, (terms, _) in enumerate(rows):
            for j, value in terms:
                if value:
                    rr.append(i)
                    cc.append(j)
                    vv.append(value)
        return coo_matrix((vv, (rr, cc)), shape=(len(rows), len(self.c))).tocsr()

    def solve(self):
        return linprog(
            self.c,
            A_ub=self.matrix(self.ub),
            b_ub=[b for _, b in self.ub],
            A_eq=self.matrix(self.eq),
            b_eq=[b for _, b in self.eq],
            bounds=self.bounds,
            method="highs",
            options={"time_limit": float(PARAMS["solver_seconds"]), "presolve": True},
        )


class Agent:
    def __init__(self, config=None):
        self.static, self.layout, self.T = config["static"], config["layout"], int(config["T"])
        self.rng = np.random.default_rng(config["policy_seed"])
        s, layout = self.static, self.layout
        self.instance = s["instance"]
        raw = {n["id"]: n for n in self.instance["nodes"]}
        self.nodes = [raw[name] for name in s["nodes"]["id"]]
        self.names = {name: i for i, name in enumerate(s["nodes"]["id"])}
        self.ks = {name: i for i, name in enumerate(s["commodities"]["id"])}
        self.knames = s["commodities"]["id"]
        self.edges, self.lanes = s["edges"], s["lanes"]
        self.cp = {int(n): i for i, n in enumerate(layout["chokepoints"])}
        self.stock_pairs = [tuple(p) for p in layout["stock_slots"]]
        self.release_pairs = [tuple(p) for p in layout["release_pairs"]]
        self.pairs = self.stock_pairs + [p for p in self.release_pairs if p not in self.stock_pairs]
        self.pindex = {p: i for i, p in enumerate(self.pairs)}
        self.demands = [tuple(p) for p in layout["demands"]]
        self.history, self.actions = [], []
        self.memory, self.count, self.sums = {}, {}, {}
        self.forecasts, self.errors = {}, [[] for _ in self.demands]
        self.loss_totals = np.zeros(len(self.layout["cost_components"]))
        self.service_totals = np.zeros((len(self.demands), 3))
        self.execution_totals = np.zeros((len(self.static["action_slots"]["edge"]), 2))
        self.pending, self.closure_ends = {}, {}
        self.solver_failures, self.last_plan = 0, {}
        action_spaces = config["spaces"]["action"]
        self.nflow = int(action_spaces["flows"]["shape"][0])
        self.noverride = int(action_spaces["override_qty"]["shape"][0])
        if self.nflow != len(s["action_slots"]["edge"]) or self.noverride != len(s["override_slots"]["out_edge"]):
            raise ValueError("action spaces do not match the public slot tables")
        if int(action_spaces["release_mode"]["shape"][0]) != len(self.release_pairs):
            raise ValueError("release-mode space does not match the public release pairs")
        self.fraction = np.broadcast_to(np.asarray(PARAMS["fraction"], float), (self.nflow,)).copy()
        nominal = {}
        for sh in self.instance["initial_state"]["pipeline"]:
            if sh["dispatch_week"] == 0:
                key = (sh["edge"], sh["k"], sh.get("lane"))
                nominal[key] = nominal.get(key, 0.0) + sh["qty"]
        self.routes, self.nominal = [], []
        for table, key in ((s["action_slots"], "edge"), (s["override_slots"], "out_edge")):
            for e, k, lane in zip(table[key], table["k"], table["lane"], strict=True):
                path = [e] if lane is None else list(self.lanes["edges"][lane])
                path = path[path.index(e) :]
                source, dest = (self.edges["tail"][e], k), (self.edges["head"][path[-1]], k)
                self.routes.append((e, k, lane, path, source, dest))
                name = None if lane is None else self.lanes["id"][lane]
                self.nominal.append(nominal.get((self.edges["id"][e], self.knames[k], name), 0.0))
        self.nominal = np.asarray(self.nominal)
        self.supply = set(tuple(p) for p in layout["supply_slots"])
        self.storage, self.holding, self.salvage = [], [], []
        com = {c["id"]: c for c in self.instance["commodities"]}
        self.disposal = [float(com[k]["disposal_cost"]) for k in self.knames]
        for n, k in self.pairs:
            a = self.nodes[n]["stock"].get(self.knames[k], {})
            self.storage.append(math.inf if a.get("storage") is None else float(a["storage"]))
            self.holding.append(float(a.get("holding_cost", 0)))
            self.salvage.append(0.0 if (n, k) in self.supply else float(a.get("salvage", 0)))
        self.mean_demand = np.asarray(
            [self.nodes[n]["sink"]["demand"][self.knames[k]]["dbar"] for n, k in self.demands]
        )
        self.value = self._values()
        self.fleet_terms = self._fleet_terms()

    def _fleet_terms(self):
        """Public extra-transit fleet terms, including divergence at a later strait."""
        terms = set()
        tau = self.edges["tau0"]
        for e, alt in enumerate(self.edges["alt_of"]):
            if alt and self.edges["mode"][e] == "sea":
                replaced = tau[alt["edge"]] if "edge" in alt else sum(tau[p] for p in self.lanes["edges"][alt["lane"]])
                terms.add((e, None, max(0, tau[e] - replaced)))
        for lane, alt in enumerate(self.lanes["alt_of"]):
            path = self.lanes["edges"][lane]
            if not alt or any(self.edges["mode"][e] != "sea" for e in path):
                continue
            if "lane" in alt:
                original = self.lanes["edges"][alt["lane"]]
                off = [e for e in path if e not in original]
                extra = sum(tau[p] for p in path) - sum(tau[p] for p in original)
            else:
                original = alt["edge"]
                off = [e for e in path if self.edges["tail"][e] == self.edges["tail"][original] and e != original]
                extra = sum(tau[p] for p in path[path.index(off[0]) :]) - tau[original] if off else 0
            if off:
                terms.add((off[0], lane if off[0] == path[0] else None, max(0, extra)))
        return sorted(terms, key=lambda item: (item[0], -1 if item[1] is None else item[1]))

    def _throughput(self, c, pool, week, ends):
        if c in ends and week >= ends[c]:
            attrs = self.nodes[c]["chokepoint"]
            return attrs["k_c"] * attrs["mu"].get(pool, 0)
        return float(self.memory["graph_now.kappa." + pool][self.cp[c]])

    @staticmethod
    def _mask(obs, key):
        value = np.asarray(obs[key])
        mask = np.asarray(obs.get(key + ".observed", np.ones_like(value)), bool)
        return np.broadcast_to(mask, value.shape) & np.isfinite(value)

    def _nominal(self, key, shape):
        defaults = {
            "graph_now.u": [0 if x is None else x for x in self.edges["u0"]],
            "graph_now.c": self.edges["c0"],
            "graph_now.tau": self.edges["tau0"],
            "graph_now.open": np.ones(len(self.cp)),
            "graph_now.fab.R": np.ones(len(self.layout["fabs"])),
            "graph_now.fab.alpha_bar": np.ones(len(self.layout["fabs"])),
            "graph_now.osat.R": np.ones(len(self.layout["osats"])),
            "graph_now.fab.cap_eff": [self.nodes[n]["fab"]["cap0"] for n in self.layout["fabs"]],
            "graph_now.osat.thr_eff": [self.nodes[n]["osat"]["thr"] for n in self.layout["osats"]],
            "graph_now.grid.G_bar": [self.nodes[n]["grid"]["deliverable"] for n in self.layout["grids"]],
            "graph_now.grid.y_bar": [self.nodes[n]["grid"]["base_load"] for n in self.layout["grids"]],
        }
        if key.startswith("graph_now.kappa."):
            pool = key.rsplit(".", 1)[-1]
            return np.asarray(
                [
                    self.nodes[n]["chokepoint"]["k_c"] * self.nodes[n]["chokepoint"]["mu"].get(pool, 0)
                    for n in self.layout["chokepoints"]
                ]
            )
        if key == "graph_now.supply.avail":
            return np.asarray(
                [self.nodes[n]["stock"][self.knames[k]].get("supply_rate", 0.0) for n, k in self.layout["supply_slots"]]
            )
        if key == "graph_now.prohibited":
            result = np.zeros(shape)
            for item in self.instance.get("prohibitions_at_reset", []):
                result[self.edges["id"].index(item["edge"]), self.ks[item["k"]]] = 1
            return result
        if key == "stock.qty":
            result = np.zeros(shape)
            for item in self.instance["initial_state"]["stock"]:
                pair = self.names[item["node"]], self.ks[item["k"]]
                if pair in self.stock_pairs:
                    result[self.stock_pairs.index(pair)] += item["qty"]
            return result
        return np.asarray(defaults.get(key, np.zeros(shape)), float).reshape(shape).copy()

    def _remember(self, obs):
        week = int(np.asarray(obs["week"]).item())
        if self.history and week <= int(self.history[-1]["week"].item()):
            raise ValueError("Agent must be called once per week, in increasing order")
        self.history.append({key: np.array(v, copy=True) for key, v in obs.items()})
        for key, value in obs.items():
            if key.endswith(".observed") or not (
                key.startswith("graph_now.") or key in ("warning.score", "stock.qty", "backlog.qty")
            ):
                continue
            value, mask = np.asarray(value, float), self._mask(obs, key)
            if key not in self.memory:
                self.memory[key] = self._nominal(key, value.shape)
                self.sums[key], self.count[key] = np.zeros_like(value), np.zeros_like(value)
            self.memory[key][mask] = value[mask]
            self.sums[key][mask] += value[mask]
            self.count[key][mask] += 1
        prev = self.forecasts.get(week - 1)
        if prev is not None:
            for d in np.flatnonzero(self._mask(obs, "last_week.sinks.demand")):
                self.errors[d].append(float(obs["last_week.sinks.demand"][d] - prev[d]))
        costs = self._mask(obs, "last_week.cost_components")
        self.loss_totals[costs] += np.asarray(obs["last_week.cost_components"])[costs]
        for col, key in enumerate(("demand", "served", "lost")):
            seen = self._mask(obs, "last_week.sinks." + key)
            self.service_totals[seen, col] += np.asarray(obs["last_week.sinks." + key])[seen]
        for col, key in enumerate(("requested", "executed")):
            seen = self._mask(obs, "last_week.clip." + key)
            self.execution_totals[seen, col] += np.asarray(obs["last_week.clip." + key])[seen]
        self.forecasts[week] = self._demand(obs, 1)[:, 0].copy()
        return week

    def _values(self):
        values = {p: float(self.static["sinks"]["pi"][d]) for d, p in enumerate(self.demands)}
        for n in self.layout["grids"]:
            for k in self.nodes[n]["grid"]["shares"]:
                if k in self.ks:
                    values[n, self.ks[k]] = float(self.nodes[n]["grid"]["voll"])
        for _ in range(len(self.nodes)):
            old = values.copy()
            for _, _, _, _, source, dest in self.routes:
                values[source] = max(values.get(source, 0), 0.97 * values.get(dest, 0))
            for n, node in enumerate(self.nodes):
                if "fab" in node:
                    f = node["fab"]
                    values[n, self.ks[f["input"]]] = values.get((n, self.ks[f["product"]]), 0) * 0.85
                if "osat" in node:
                    for raw, prod in node["osat"]["packages"].items():
                        values[n, self.ks[raw]] = values.get((n, self.ks[prod]), 0) * 0.95
            if values == old:
                break
        return values

    def _demand(self, obs, H):
        out = np.repeat(self.mean_demand[:, None], H, axis=1)
        week = int(obs["week"].item())
        for d, (n, k) in enumerate(self.demands):
            seasonal = self.nodes[n]["sink"]["demand"][self.knames[k]].get("seasonal")
            if seasonal:
                out[d] *= [seasonal[(week + h - 1) % len(seasonal)] for h in range(H)]
            bias = sum(self.errors[d]) / (len(self.errors[d]) + 8)
            out[d] += np.clip(bias, -0.2 * self.mean_demand[d], 0.2 * self.mean_demand[d])
        f, seen = np.asarray(obs["demand_forecast.qty"], float), self._mask(obs, "demand_forecast.qty")
        width = min(f.shape[1], H)
        for d in range(len(self.demands)):
            bias = np.clip(
                sum(self.errors[d]) / (len(self.errors[d]) + 8), -0.2 * self.mean_demand[d], 0.2 * self.mean_demand[d]
            )
            out[d, :width] = np.where(seen[d, :width], f[d, :width] + bias, out[d, :width])
        return np.maximum(out, 0)

    def _signals(self, obs, week):
        risk = np.zeros((len(self.edges["id"]), len(self.knames)))
        units = {tuple(unit): i for i, unit in enumerate(self.layout["warning_units"])}
        scores = self.memory.get("warning.score", np.zeros(len(units)))

        def score(kind, i):
            return float(np.clip(scores[units[kind, i]], 0, 1)) if (kind, i) in units else 0.0

        for e, (a, b) in enumerate(zip(self.edges["tail"], self.edges["head"], strict=True)):
            ra, rb = self.static["nodes"]["region"][a], self.static["nodes"]["region"][b]
            r = max(score("region", ra), score("region", rb), score("chokepoint", a), score("chokepoint", b))
            for d, (da, db) in enumerate(zip(self.static["dyads"]["a"], self.static["dyads"]["b"], strict=True)):
                if {ra, rb} == {da, db}:
                    r = max(r, score("dyad", d))
            risk[e] = r
        for i in np.flatnonzero(self._mask(obs, "messages.msg_id")):
            fields = ("channel", "kind", "region", "target_kind", "target", "announced_week")
            if not all(self._mask(obs, "messages." + key)[i] for key in fields):
                continue
            kind, channel = int(obs["messages.kind"][i]), int(obs["messages.channel"][i])
            if kind == 4:
                continue
            effective = (
                int(obs["messages.stated_effective_week"][i])
                if self._mask(obs, "messages.stated_effective_week")[i]
                else week + 4
            )
            age = max(0, week - int(obs["messages.announced_week"][i]))
            confidence = (0.55 if kind in (1, 3) else 0.18) * (0.8 if channel in (1, 4, 5) else 1)
            confidence *= math.exp(-age / 12) / (1 + max(0, effective - week) / 8)
            tk, target = int(obs["messages.target_kind"][i]), int(obs["messages.target"][i])
            message_region = int(obs["messages.region"][i])
            ks = [int(obs["messages.k"][i])] if self._mask(obs, "messages.k")[i] else range(len(self.knames))
            for e, (a, b) in enumerate(zip(self.edges["tail"], self.edges["head"], strict=True)):
                match = (
                    e == target
                    if tk == 1
                    else target in (a, b)
                    if tk in (0, 2)
                    else target in (self.static["nodes"]["region"][a], self.static["nodes"]["region"][b])
                )
                if target < 0:
                    match = message_region in (self.static["nodes"]["region"][a], self.static["nodes"]["region"][b])
                if match:
                    for k in ks:
                        if 0 <= k < len(self.knames):
                            risk[e, k] = max(risk[e, k], confidence)
        pending = self.pending.copy()
        # A later observed lifting supersedes an earlier announcement.
        for pair, effective in list(pending.items()):
            if (
                effective < week
                and self._mask(obs, "graph_now.prohibited")[pair]
                and not obs["graph_now.prohibited"][pair]
            ):
                del pending[pair]
        ends = {c: end for c, end in self.closure_ends.items() if end >= week}
        for i in np.flatnonzero(self._mask(obs, "pending_prohibitions.edge")):
            if all(self._mask(obs, "pending_prohibitions." + key)[i] for key in ("k", "effective_week")):
                pair = int(obs["pending_prohibitions.edge"][i]), int(obs["pending_prohibitions.k"][i])
                pending[pair] = min(pending.get(pair, math.inf), int(obs["pending_prohibitions.effective_week"][i]))
        for i in np.flatnonzero(self._mask(obs, "closure_end.chokepoint")):
            if self._mask(obs, "closure_end.end_week")[i]:
                ends[int(obs["closure_end.chokepoint"][i])] = int(obs["closure_end.end_week"][i])
        self.pending, self.closure_ends = pending.copy(), ends.copy()
        return risk, pending, ends

    def _queue(self, obs):
        lots = []
        if "lot_keys" in self.layout:
            q = np.where(self._mask(obs, "queue_lots.qty"), obs["queue_lots.qty"], 0)
            for row in np.flatnonzero(q.sum(axis=1) > 0):
                c, k, lane, edge = self.layout["lot_keys"][row]
                for cohort in np.flatnonzero(q[row] > 0):
                    lots.append((c, k, lane, edge, cohort + 1, float(q[row, cohort])))
        else:
            for i in np.flatnonzero(self._mask(obs, "queue_lots.qty")):
                lots.append(
                    tuple(
                        int(obs["queue_lots." + key][i])
                        for key in ("chokepoint", "k", "lane", "next_edge", "arrival_week")
                    )
                    + (float(obs["queue_lots.qty"][i]),)
                )
        return lots

    def _route(self, route, week, risk, pending, ends):
        e, k, _, path, _, _ = route
        mem, travel, cost = self.memory, 0, 0.0
        cap = float(mem["graph_now.u"][e])
        for edge in path:
            if mem["graph_now.prohibited"][edge, k] or week + travel >= pending.get((edge, k), math.inf):
                return 0.0, 1, 0.0
            cost += mem["graph_now.c"][edge] + mem["graph_now.tariff"][edge, k] * self.static["commodities"]["v"][k]
            for c in {self.edges["head"][edge], self.edges["tail"][edge]} & self.cp.keys():
                ci = self.cp[c]
                opening = float(mem["graph_now.open"][ci])
                if c in ends and week + travel >= ends[c]:
                    opening = 1.0
                if opening < 0.05:
                    if c not in ends:
                        return 0.0, 1, 0.0
                    travel += max(0, ends[c] - week - travel)
                    opening = 1.0
                cap = min(cap, float(mem["graph_now.u"][edge]) * opening ** float(PARAMS["closure_power"]))
                attrs = self.nodes[c]["chokepoint"]
                wr = int(np.clip(mem["graph_now.war_risk"][ci], 0, 2))
                cost += attrs["war_risk_cost"].get(self.knames[k], [0, 0, 0])[wr]
                cost += (1 - opening) * attrs["queue_holding"].get(self.knames[k], [0, 0, 0])[wr]
            cost += float(PARAMS["warning_gain"]) * risk[edge, k] * max(float(mem["graph_now.c"][edge]), 1)
            travel += max(0, int(mem["graph_now.tau"][edge]))
        return max(cap, 0), max(1, travel), max(cost, 0)

    def _start(self, obs, week, H, risk, pending, ends):
        start, incoming = np.zeros(len(self.pairs)), np.zeros((H, len(self.pairs)))
        start[: len(self.stock_pairs)] = self.memory["stock.qty"]
        lots = self._queue(obs)
        for c, k, lane, edge, cohort, qty in lots:
            if (c, k) in self.pindex:
                start[self.pindex[c, k]] += qty
            else:
                path = list(self.lanes["edges"][lane])
                suffix = path[path.index(edge) :]
                dest = self.edges["head"][suffix[-1]], k
                cap, delay, _ = self._route((edge, k, lane, suffix, (c, k), dest), week, risk, pending, ends)
                if cap <= 0:
                    continue
                pool = self.static["commodities"]["pool"][k]
                throughput = float(self.memory["graph_now.kappa." + pool][self.cp[c]])
                ahead = sum(
                    q
                    for cc, kk, _, _, arrival, q in lots
                    if cc == c and arrival <= cohort and self.static["commodities"]["pool"][kk] == pool
                )
                delay += min(H, int(ahead / max(throughput, 1)))
                if delay < H and dest in self.pindex:
                    incoming[delay, self.pindex[dest]] += qty
        for i in np.flatnonzero(self._mask(obs, "pipeline.qty")):
            if not all(self._mask(obs, "pipeline." + key)[i] for key in ("edge", "k", "arrival_week")):
                continue
            edge, k = int(obs["pipeline.edge"][i]), int(obs["pipeline.k"][i])
            head, arrival = self.edges["head"][edge], int(obs["pipeline.arrival_week"][i]) - week
            if head in self.cp and (head, k) not in self.pindex:
                if not self._mask(obs, "pipeline.lane")[i]:
                    continue
                lane = int(obs["pipeline.lane"][i])
                path = list(self.lanes["edges"][lane])
                suffix = path[path.index(edge) + 1 :]
                if not suffix:
                    continue
                head = self.edges["head"][suffix[-1]]
                dest = head, k
                cap, delay, _ = self._route(
                    (suffix[0], k, lane, suffix, (self.edges["tail"][suffix[0]], k), dest),
                    week + max(0, arrival),
                    risk,
                    pending,
                    ends,
                )
                if cap <= 0:
                    continue
                arrival += delay
            if 0 <= arrival < H and (head, k) in self.pindex:
                incoming[arrival, self.pindex[head, k]] += float(obs["pipeline.qty"][i])
        for i in np.flatnonzero(self._mask(obs, "wip.qty")):
            if all(self._mask(obs, "wip." + key)[i] for key in ("node", "k", "out_week")):
                pair, arrival = (int(obs["wip.node"][i]), int(obs["wip.k"][i])), int(obs["wip.out_week"][i]) - week
                if 0 <= arrival < H and pair in self.pindex:
                    incoming[arrival, self.pindex[pair]] += float(obs["wip.qty"][i])
        return start, incoming

    def _plan(self, obs, week):
        H = min(max(2, int(PARAMS["horizon"])), self.T - week + 1)
        P, R, lp = len(self.pairs), len(self.routes), _LP()
        risk, pending, ends = self._signals(obs, week)
        demand, (start, incoming) = self._demand(obs, H), self._start(obs, week, H, risk, pending, ends)
        balance = [[[] for _ in range(P)] for _ in range(H)]
        outgoing = [[[] for _ in range(P)] for _ in range(H)]
        inventory, x = np.empty((H, P), int), np.empty((H, R), int)
        demand_rhs, backlogs = np.zeros((H, P)), {}
        network_edges, network_pools, fleet_use = {}, {}, {}
        for h in range(H):
            for p, (n, k) in enumerate(self.pairs):
                hold = self.holding[p]
                if n in self.cp:
                    wr = int(np.clip(self.memory["graph_now.war_risk"][self.cp[n]], 0, 2))
                    hold = self.nodes[n]["chokepoint"]["queue_holding"].get(self.knames[k], [0, 0, 0])[wr]
                inventory[h, p] = lp.var(hold - (self.salvage[p] if week + h == self.T else 0), self.storage[p])
                balance[h][p].append((inventory[h, p], 1))
                if h:
                    balance[h][p].append((inventory[h - 1, p], -1))
                if (n, k) not in self.supply:
                    balance[h][p].append((lp.var(self.disposal[k]), 1))
        for h in range(H):
            edge_use, cp_use = {}, {}
            for r, route in enumerate(self.routes):
                e, k, lane, path, source, dest = route
                cap, transit, cost = self._route(route, week + h, risk, pending, ends)
                if r < self.nflow:
                    requested, executed = self.execution_totals[r]
                    # Clipped/stock-starved routes are a weak tie-break, not a fabricated hard capacity mask.
                    cost += 0.005 * max(0.0, 1 - executed / max(requested, 1)) * self.memory["graph_now.c"][e]
                if r < self.nflow:
                    cap *= max(0.0, self.fraction[r])
                    if h == 0 and self._mask(obs, "action_mask")[r] and not obs["action_mask"][r]:
                        cap = 0
                    if h == 0 and self._mask(obs, "slot_mask")[r] and obs["slot_mask"][r]:
                        cap = 0
                elif (
                    h == 0
                    and self._mask(obs, "override_mask")[r - self.nflow]
                    and not obs["override_mask"][r - self.nflow]
                ):
                    cap = 0
                if source not in self.pindex or dest not in self.pindex:
                    cap = 0
                arrival = h + transit
                if arrival >= H:
                    if week + H - 1 < self.T:
                        cap = 0
                    elif dest in self.pindex:
                        cost -= self.salvage[self.pindex[dest]]
                x[h, r] = lp.var(cost, cap)
                if source in self.pindex:
                    p = self.pindex[source]
                    balance[h][p].append((x[h, r], 1))
                    outgoing[h][p].append((x[h, r], 1))
                if arrival < H and dest in self.pindex:
                    balance[arrival][self.pindex[dest]].append((x[h, r], -1))
                edge_use.setdefault(e, []).append((x[h, r], 1))
                offset = 0
                pool = self.static["commodities"]["pool"][k]
                for pe in path:
                    planned_week = h + offset
                    if planned_week < H:
                        network_edges.setdefault((planned_week, pe), []).append((x[h, r], 1))
                        c = self.edges["tail"][pe]
                        if c in self.cp:
                            network_pools.setdefault((planned_week, c, pool), []).append((x[h, r], 1))
                        for de, dlane, extra in self.fleet_terms:
                            if de == pe and (dlane is None or dlane == lane):
                                fleet_use.setdefault((planned_week, pool), []).append((x[h, r], extra))
                    offset += max(0, int(self.memory["graph_now.tau"][pe]))
                if source[0] in self.cp:
                    pool = self.static["commodities"]["pool"][k]
                    cp_use.setdefault((source[0], pool), []).append((x[h, r], 1))
            for e, terms in edge_use.items():
                lp.row(terms, self.memory["graph_now.u"][e])
            for (c, pool), terms in cp_use.items():
                lp.row(terms, self._throughput(c, pool, week + h, ends))
            for p, pair in enumerate(self.pairs):
                terms = outgoing[h][p].copy()
                if h:
                    terms.append((inventory[h - 1, p], -1))
                available = incoming[h, p] if pair[0] in self.cp else 0.0
                lp.row(terms, (start[p] if h == 0 else 0.0) + available)
            for si, pair in enumerate(self.layout["supply_slots"]):
                p = self.pindex[tuple(pair)]
                balance[h][p].append((lp.var(0, self.memory["graph_now.supply.avail"][si]), -1))
            fab_draw = {n: [] for n in self.layout["grids"]}
            for fi, n in enumerate(self.layout["fabs"]):
                fab = self.nodes[n]["fab"]
                restoration = self.memory["graph_now.fab.R"][fi]
                cap = min(
                    self.memory["graph_now.fab.cap_eff"][fi],
                    fab["cap0"] * restoration * self.memory["graph_now.fab.alpha_bar"][fi],
                )
                v = lp.var(0, cap)
                balance[h][self.pindex[n, self.ks[fab["input"]]]].append((v, 1))
                if h + fab["tau"] < H:
                    balance[h + fab["tau"]][self.pindex[n, self.ks[fab["product"]]]].append((v, -1))
                if fab.get("grid") is not None:
                    fab_draw[self.names[fab["grid"]]].append((v, fab["e"] / max(restoration, 1e-6)))
            for oi, n in enumerate(self.layout["osats"]):
                osat, terms = self.nodes[n]["osat"], []
                for raw, product in osat["packages"].items():
                    v = lp.var()
                    terms.append((v, 1))
                    balance[h][self.pindex[n, self.ks[raw]]].append((v, 1))
                    if h + osat["tau"] < H:
                        balance[h + osat["tau"]][self.pindex[n, self.ks[product]]].append((v, -1))
                lp.row(
                    terms,
                    min(self.memory["graph_now.osat.thr_eff"][oi], osat["thr"] * self.memory["graph_now.osat.R"][oi]),
                )
            for gi, n in enumerate(self.layout["grids"]):
                grid = self.nodes[n]["grid"]
                gen, load = self.memory["graph_now.grid.G_bar"][gi], self.memory["graph_now.grid.y_bar"][gi]
                energy = [(lp.var(-grid["voll"], load), 1)] + fab_draw[n]
                for fuel, share in grid["shares"].items():
                    if fuel in self.ks:
                        p = self.pindex[n, self.ks[fuel]]
                        v = lp.var(0, share * gen)
                        balance[h][p].append((v, 1))
                        energy.append((v, -1))
                        if fuel == grid.get("rationed"):
                            threshold = self.instance["params"]["psi"] * grid["ibar"][fuel]
                            if threshold > 0:
                                terms = [(v, threshold)]
                                if h:
                                    terms.append((inventory[h - 1, p], -share * gen))
                                lp.row(terms, share * gen * start[p] if h == 0 else 0)
                lp.row(energy, grid["shares"].get("unmodelled", 0) * gen)
            for d, pair in enumerate(self.demands):
                p, rhs = self.pindex[pair], -demand[d, h]
                shortage = lp.var(self.static["sinks"]["pi"][d])
                balance[h][p].append((shortage, -1))
                if self.static["sinks"]["backlog"][d]:
                    if h:
                        balance[h][p].append((backlogs[d], 1))
                    else:
                        rhs -= float(self.memory["backlog.qty"][d])
                    backlogs[d] = shortage
                demand_rhs[h, p] = rhs
        for (_, e), terms in network_edges.items():
            lp.row(terms, self.memory["graph_now.u"][e])
        for (h, c, pool), terms in network_pools.items():
            lp.row(terms, self._throughput(c, pool, week + h, ends))
        for (_, pool), terms in fleet_use.items():
            params = self.instance["params"]
            lp.row(terms, params["fleet_share"][pool] * params["fleet_measure"][pool])
        for h in range(H):
            for p in range(P):
                lp.row(balance[h][p], incoming[h, p] + (start[p] if h == 0 else 0) + demand_rhs[h, p], True)
        if week + H - 1 < self.T:
            for p, pair in enumerate(self.pairs):
                if pair in self.supply or pair[0] in self.cp:
                    continue
                v = lp.var(-float(PARAMS["terminal_gain"]) * self.value.get(pair, 0), self._buffer(pair, demand, obs))
                lp.row([(v, 1), (inventory[-1, p], -1)], 0)
        result = lp.solve()
        self.last_plan = {
            "week": week,
            "status": int(result.status),
            "message": str(result.message),
            "variables": len(lp.c),
            "horizon": H,
        }
        if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
            self.solver_failures += 1
            return self._fallback(obs, week)
        flows, override = np.maximum(result.x[x[0, : self.nflow]], 0), np.maximum(result.x[x[0, self.nflow :]], 0)
        modes = np.zeros(len(self.release_pairs), np.int64)
        for i, pair in enumerate(self.release_pairs):
            slots = [s for s, route in enumerate(self.routes[self.nflow :]) if route[4] == pair]
            if start[self.pindex[pair]] + incoming[0, self.pindex[pair]] > 1e-8:
                # Mode 1 sends EVERY slot of the pair, even zeros. A zero on a
                # masked slot is still invalid in the flat protocol. Use the
                # legal automatic release if a valid subset cannot be expressed.
                if (
                    slots
                    and np.asarray(obs["override_mask.observed"]).item()
                    and all(obs["override_mask"][s] for s in slots)
                ):
                    modes[i] = 1
                elif not slots or override[slots].sum() <= 1e-8:
                    modes[i] = 2
                else:
                    override[slots] = 0
        return {"flows": flows, "override_qty": override, "release_mode": modes}

    def _buffer(self, pair, demand, obs):
        n, k = pair
        node, name = self.nodes[n], self.knames[k]
        safety = float(PARAMS["safety_stock"])
        components = dict(zip(self.layout["cost_components"], self.loss_totals, strict=True))
        distress_cost = components.get("shortage", 0) + components.get("shed", 0)
        waste_cost = sum(components.get(k, 0) for k in ("holding", "queue_holding", "disposal"))
        waste_cost += 0.05 * sum(components.get(k, 0) for k in ("freight", "war_risk", "tariff"))
        safety += 0.1 * (distress_cost - waste_cost) / max(distress_cost + waste_cost, 1)
        if pair in self.demands:
            d = self.demands.index(pair)
            requested, served, lost = self.service_totals[d]
            unmet = max(lost, requested - served)
            return demand[d].mean() * (1 + safety + min(0.4, unmet / max(requested, 1)))
        if "grid" in node:
            grid = node["grid"]
            rate = grid["deliverable"] * grid["shares"].get(name, 0)
            # All past loss signals contribute to the reserve, with conservative bounds.
            shed = [
                float(o["last_week.shed.qty"][self.layout["grids"].index(n)])
                for o in self.history
                if self._mask(o, "last_week.shed.qty")[self.layout["grids"].index(n)]
            ]
            distress = min(0.5, sum(shed) / max(len(shed) * grid["base_load"], 1))
            return max(rate * (1 + safety + distress), self.instance["params"]["psi"] * grid["ibar"].get(name, 0))
        rate = sum(
            max(self.nominal[r], self.execution_totals[r, 1] / max(len(self.history) - 1, 1))
            for r, route in enumerate(self.routes[: self.nflow])
            if route[4] == pair
        )
        if "fab" in node and name == node["fab"]["input"]:
            rate = node["fab"]["cap0"]
        elif "osat" in node and name in node["osat"]["packages"]:
            rate = node["osat"]["thr"] / len(node["osat"]["packages"])
        return min(self.storage[self.pindex[pair]], rate * (1 + safety))

    def _fallback(self, obs, week):
        flows = np.zeros(self.nflow)
        for r, (e, _, _, path, source, _) in enumerate(self.routes[: self.nflow]):
            cap = self.memory["graph_now.u"][e]
            reserve = 0.75 if self.nodes[source[0]]["type"] in ("source", "material") else 1.0
            flows[r] = min(cap, self.nominal[r] + reserve * max(0, cap - self.nominal[r])) * max(0, self.fraction[r])
            for edge in path:
                c = self.edges["head"][edge]
                if c in self.cp:
                    flows[r] *= max(0, self.memory["graph_now.open"][self.cp[c]]) ** float(PARAMS["closure_power"])
            if week + sum(int(self.memory["graph_now.tau"][e]) for e in path) > self.T:
                flows[r] = 0
        return {
            "flows": flows,
            "override_qty": np.zeros(self.noverride),
            "release_mode": np.zeros(len(self.release_pairs), np.int64),
        }

    def act(self, observation):
        week = self._remember(observation)
        action = self._plan(observation, week)
        for field, mask in (("flows", "action_mask"), ("override_qty", "override_mask")):
            if np.asarray(observation[mask + ".observed"]).item():
                action[field] *= np.asarray(observation[mask], float)
            action[field] = np.nan_to_num(np.maximum(action[field], 0), nan=0, posinf=0, neginf=0)
        self.actions.append({key: value.copy() for key, value in action.items()})
        return action
