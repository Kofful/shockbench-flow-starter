import math
import numpy as np

# ==========================================
# AUTO-UPDATED BY tune.py (DO NOT EDIT MANUALLY)
# ==========================================
_TUNED_OVERRIDES = {
    "use_cap": 0,
    "use_release": 0,
    "planned_source": 0.9087557345615994,
    "planned_terminal": 0.44763076868162593,
    "planned_fab": 0.029287177622229835,
    "planned_osat": 1.0,
    "planned_other": 0.0397085749855638,
    "new_source": 0.5096357614116894,
    "new_terminal": 0.5623361686762799,
    "new_fab": 1.0,
    "new_osat": 0.0,
    "new_other": 0.9497274392652203,
    "distress_gain": 0.0,
    "closure_power": 0.6145057434130496,
    "end_margin": 1
}
# ==========================================

# Original 13 parameters (unchanged defaults for Small/Full).
_WEIGHTS = {
    "planned_source": 0.75, "planned_terminal": 0.50, "planned_fab": 0.00,
    "planned_osat": 1.00, "planned_other": 0.00, "new_source": 0.50,
    "new_terminal": 0.50, "new_fab": 1.00, "new_osat": 0.00, "new_other": 1.00,
    "distress_gain": 0.0, "closure_power": 0.5, "end_margin": 1,
}

_TINY_WEIGHTS = {
    "planned_source": 0.75, "planned_terminal": 1.00, "planned_fab": 1.00,
    "planned_osat": 1.00, "planned_other": 0.00, "new_source": 1.00,
    "new_terminal": 1.00, "new_fab": 1.00, "new_osat": 1.00, "new_other": 1.00,
    "distress_gain": 0.0, "closure_power": 1.0, "end_margin": 0,
}

_NEW_DEFAULTS = {
    "use_cap": 0,         # 1 = cap flows by live graph_now.u (strikes / capacity cuts)
    "cost_gain": 0.0,     # >0 = shrink the surge when live cost c is above nominal c0
    "use_release": 0,     # 1 = return release_mode (and override_qty if possible)
    "hold_below": 0.25,   # open fraction below this -> hold queue (mode 2)
    "flush_above": 0.50,  # open fraction at/above this after a closure -> flush (mode 1)
    "flush_weeks": 3,     # how many weeks to keep flushing after a reopening
    "hold_slack": 3,      # weeks of tolerance before queued cargo is deemed too late
}


class Agent:
    def __init__(self, config, _weights=None):
        is_tiny = config["static"]["instance_id"] == "chokepoint-tiny"
        weights = dict(_NEW_DEFAULTS)
        weights.update(_TINY_WEIGHTS if is_tiny else _WEIGHTS)
        
        # Apply the auto-injected tuned parameters
        weights.update(_TUNED_OVERRIDES)
        
        if _weights is not None:
            weights.update(_weights)
            
        self.weights = weights
        self.T = int(config["T"])

        static = config["static"]
        instance = static["instance"]
        slots = static["action_slots"]
        edges = static["edges"]
        lanes = static["lanes"]
        nodes = static["nodes"]
        commodities = static["commodities"]

        slot_edges = list(slots["edge"])
        slot_commodities = slots["k"]
        slot_lanes = slots["lane"]
        self.n_slots = len(slot_edges)
        self.capacity = np.asarray([edges["u0"][e] for e in slot_edges], dtype=np.float64)

        nominal_by_route = {}
        for shipment in instance["initial_state"]["pipeline"]:
            if shipment["dispatch_week"] == 0:
                key = (shipment["edge"], shipment["k"], shipment.get("lane"))
                nominal_by_route[key] = nominal_by_route.get(key, 0.0) + float(shipment["qty"])

        edge_ids = edges["id"]
        commodity_ids = commodities["id"]
        lane_ids = lanes["id"]
        self.nominal = np.asarray(
            [
                nominal_by_route.get(
                    (edge_ids[e], commodity_ids[k], None if lane is None else lane_ids[lane]),
                    0.0,
                )
                for e, k, lane in zip(slot_edges, slot_commodities, slot_lanes, strict=True)
            ],
            dtype=np.float64,
        )
        self.nominal = np.minimum(self.nominal, self.capacity)

        node_types = nodes["type"]
        reserve = []
        for i, edge in enumerate(slot_edges):
            stage = node_types[edges["tail"][edge]]
            if stage not in {"source", "terminal", "fab", "osat"}:
                stage = "other"
            prefix = "planned" if self.nominal[i] > 0.0 else "new"
            reserve.append(weights[f"{prefix}_{stage}"])
        self.reserve = np.clip(np.asarray(reserve, dtype=np.float64), 0.0, 1.0)

        chokepoint_position = {node: i for i, node in enumerate(config["layout"]["chokepoints"])}
        self.n_cp = len(chokepoint_position)
        self.incidence = np.zeros((self.n_slots, max(self.n_cp, 1)), dtype=bool)
        for slot, lane in enumerate(slot_lanes):
            if lane is not None:
                for c in lanes["chokepoints"][lane]:
                    self.incidence[slot, chokepoint_position[c]] = True

        self.last_useful = self._last_useful_weeks(
            instance, nodes, edges, lanes, commodities, slot_edges, slot_commodities, slot_lanes,
        )

        if self.n_cp:
            lu = np.where(self.incidence, self.last_useful[:, None], -1).max(axis=0)
            lu[~self.incidence.any(axis=0)] = self.T
            self.cp_last_useful = lu[: self.n_cp].astype(np.int64)
        else:
            self.cp_last_useful = np.zeros(0, dtype=np.int64)

        n_edges = len(edge_ids)
        self.n_edges = n_edges
        paths = [
            list(lanes["edges"][lane]) if lane is not None else [e]
            for e, lane in zip(slot_edges, slot_lanes, strict=True)
        ]
        lmax = max(len(p) for p in paths)
        self.path_idx = np.full((self.n_slots, lmax), n_edges, dtype=np.int64) 
        for i, p in enumerate(paths):
            self.path_idx[i, : len(p)] = p

        c0 = edges.get("c0") if hasattr(edges, "get") else None
        if c0 is not None:
            c0 = np.asarray(c0, dtype=np.float64)
            self.slot_c0 = np.maximum(c0[np.asarray(slot_edges)], 1e-9)
        else:
            self.slot_c0 = None
        self.slot_edge_idx = np.asarray(slot_edges, dtype=np.int64)

        raw_node = {node["id"]: node for node in instance["nodes"]}
        self.grid_load = np.asarray(
            [raw_node[nodes["id"][n]]["grid"]["base_load"] for n in config["layout"]["grids"]],
            dtype=np.float64,
        )

        self.is_tiny = is_tiny
        lot_cp = config["layout"].get("lot_chokepoint") if hasattr(config["layout"], "get") else None
        self.lot_cp = None if lot_cp is None else np.asarray(lot_cp, dtype=np.int64)
        self.reset()

    def reset(self):
        self.was_closed = np.zeros(self.n_cp, dtype=bool)
        self.flush_left = np.zeros(self.n_cp, dtype=np.int64)

    def _last_useful_weeks(self, instance, nodes, edges, lanes, commodities, slot_edges, slot_commodities, slot_lanes):
        node_ids = nodes["id"]
        node_types = nodes["type"]
        commodity_ids = commodities["id"]
        commodity_position = {name: i for i, name in enumerate(commodity_ids)}
        raw_node = {node["id"]: node for node in instance["nodes"]}

        routes = {}
        slot_destinations = []
        slot_transit = []
        for edge, commodity, lane in zip(slot_edges, slot_commodities, slot_lanes, strict=True):
            if lane is None:
                destination = edges["head"][edge]
                transit = int(edges["tau0"][edge])
            else:
                path = lanes["edges"][lane]
                destination = edges["head"][path[-1]]
                transit = sum(int(edges["tau0"][x]) for x in path)
            routes.setdefault((edges["tail"][edge], commodity), []).append((destination, commodity, transit))
            slot_destinations.append(destination)
            slot_transit.append(transit)

        memo = {}
        visiting = set()

        def remaining(node, commodity):
            key = (node, commodity)
            if key in memo:
                return memo[key]
            if key in visiting:
                return math.inf
            visiting.add(key)
            raw = raw_node[node_ids[node]]
            commodity_id = commodity_ids[commodity]
            choices = []

            if node_types[node] == "sink" and commodity_id in raw.get("sink", {}).get("demand", {}):
                choices.append(0)
            if node_types[node] == "grid" and commodity_id in raw.get("grid", {}).get("shares", {}):
                choices.append(0)

            fab = raw.get("fab")
            if fab is not None and commodity_id == fab["input"]:
                product = commodity_position[fab["product"]]
                choices.append(int(fab["tau"]) + 1 + remaining(node, product))

            osat = raw.get("osat")
            if osat is not None and commodity_id in osat["packages"]:
                product = commodity_position[osat["packages"][commodity_id]]
                choices.append(int(osat["tau"]) + 1 + remaining(node, product))

            for destination, next_commodity, transit in routes.get(key, ()):
                choices.append(transit + 1 + remaining(destination, next_commodity))

            visiting.remove(key)
            memo[key] = min(choices, default=math.inf)
            return memo[key]

        margin = int(self.weights["end_margin"])
        deadlines = []
        for destination, commodity, transit in zip(slot_destinations, slot_commodities, slot_transit, strict=True):
            after_arrival = remaining(destination, commodity)
            if math.isfinite(after_arrival):
                deadlines.append(self.T - transit - int(after_arrival) + margin)
            else:
                deadlines.append(self.T)
        return np.asarray(deadlines, dtype=np.int64)

    def _distress(self, observation):
        demand = np.asarray(observation["last_week.sinks.demand"], dtype=np.float64)
        lost = np.asarray(observation["last_week.sinks.lost"], dtype=np.float64)
        lost_share = float(lost.sum() / max(demand.sum(), 1.0))
        shed = np.asarray(observation["last_week.shed.qty"], dtype=np.float64)
        shed_share = float(shed.sum() / max(self.grid_load.sum(), 1.0))
        return min(max(lost_share, shed_share, 0.0), 1.0)

    def _lot_totals(self, obs):
        q = obs.get("queue_lots.qty")
        if q is None:
            return None
        q = np.nan_to_num(np.asarray(q, dtype=np.float64))
        if self.is_tiny or "queue_lots.lot_id" in obs:
            q = np.clip(q, 0.0, None)
            return q.reshape(-1) if q.ndim <= 1 else q.sum(axis=tuple(range(1, q.ndim)))
        if q.ndim == 2:
            return np.clip(q, 0.0, None).sum(axis=1)
        return np.clip(q, 0.0, None).reshape(-1)

    def _queue_per_cp(self, totals):
        if totals is None:
            return None
        if self.n_cp == 1:
            return np.array([totals.sum()])
        if self.lot_cp is not None and len(self.lot_cp) == len(totals):
            return np.bincount(self.lot_cp, weights=totals, minlength=self.n_cp)
        return None

    def _release(self, obs, week, open_now, observed):
        w = self.weights
        eff_open = np.where(observed, open_now, 1.0) 
        closed = eff_open < w["hold_below"]

        reopened = self.was_closed & ~closed
        self.flush_left = np.where(reopened, int(w["flush_weeks"]), np.maximum(self.flush_left - 1, 0))
        self.was_closed = closed

        totals = self._lot_totals(obs)
        queue_cp = self._queue_per_cp(totals)

        too_late = week > (self.cp_last_useful + int(w["hold_slack"]))

        hold = closed | too_late
        flush = ~hold & (eff_open >= w["flush_above"]) & (self.flush_left > 0)
        if queue_cp is not None:
            flush &= queue_cp > 0.0

        mode = np.zeros(self.n_cp, dtype=np.int64)
        mode[hold] = 2
        mode[flush] = 1
        out = {"release_mode": mode}

        if totals is not None and (self.n_cp == 1 or self.lot_cp is not None):
            lot_flush = flush[0] if self.n_cp == 1 else flush[self.lot_cp]
            out["override_qty"] = np.where(lot_flush, totals, 0.0)
        return out

    def act(self, observation):
        w = self.weights
        week = int(np.asarray(observation["week"]).item())

        gain = float(w["distress_gain"])
        reserve = self.reserve
        if gain != 0.0:
            reserve = reserve + gain * self._distress(observation)

        if w["cost_gain"] > 0.0 and self.slot_c0 is not None and "graph_now.c" in observation:
            c_now = np.asarray(observation["graph_now.c"], dtype=np.float64)
            if c_now.shape[0] > int(self.slot_edge_idx.max()):
                ratio = c_now[self.slot_edge_idx] / self.slot_c0
                reserve = reserve * np.clip(1.0 - w["cost_gain"] * np.maximum(ratio - 1.0, 0.0), 0.0, 1.0)

        reserve = np.clip(reserve, 0.0, 1.0)
        flows = self.nominal + reserve * (self.capacity - self.nominal)
        flows = np.where(week <= self.last_useful, flows, 0.0)

        if w["use_cap"]:
            u_now = np.asarray(observation["graph_now.u"], dtype=np.float64)
            if u_now.shape[0] == self.n_edges:
                u_pad = np.append(u_now, np.inf)
                flows = np.minimum(flows, u_pad[self.path_idx].min(axis=1))

        open_now = np.asarray(observation["graph_now.open"], dtype=np.float64)[: self.n_cp]
        observed = np.asarray(observation["graph_now.open.observed"], dtype=bool)[: self.n_cp]

        power = float(w["closure_power"])
        if power > 0.0 and self.n_cp:
            factor = np.where(observed, np.maximum(open_now, 0.0) ** power, 1.0)
            flows = flows * np.where(self.incidence[:, : self.n_cp], factor[None, :], 1.0).prod(axis=1)

        action = {"flows": flows * observation["action_mask"]}
        if w["use_release"] and self.n_cp:
            action.update(self._release(observation, week, open_now, observed))
        return action