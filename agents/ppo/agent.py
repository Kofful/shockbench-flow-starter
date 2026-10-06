"""Recurrent residual PPO policy for ShockBench-Flow evaluation.

Training lives in ``train.py``.  The submitted agent only builds compact,
dimensionless route features, loads the exported TorchScript actor once, and
runs deterministic inference.  Each ``Agent`` keeps the previous week's
features and the actor's recurrent state, so decisions use current, previous,
and longer-horizon episode information.

The neural policy predicts bounded corrections around a safe nominal dispatch
rule.  Known prohibitions, closed chokepoints, current first-edge capacity and
end-of-horizon deadlines are projected deterministically after inference.
"""

import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch


FEATURE_VERSION = 1
STATIC_DIM = 12
DYNAMIC_DIM = 28
FEATURE_DIM = STATIC_DIM + 3 * DYNAMIC_DIM
EPS = 1e-4

_TASK_BY_INSTANCE = {
    "chokepoint-tiny": "tiny",
    "chokepoint-small": "small",
    "chokepoint-full": "full",
}

# A strong, deterministic starting controller.  PPO learns residuals around
# these values instead of having to rediscover a viable supply plan through
# destructive random exploration.
_RESERVE = {
    "tiny": {
        "planned_source": 0.75,
        "planned_terminal": 1.00,
        "planned_fab": 1.00,
        "planned_osat": 1.00,
        "planned_other": 0.00,
        "new_source": 1.00,
        "new_terminal": 1.00,
        "new_fab": 1.00,
        "new_osat": 1.00,
        "new_other": 1.00,
        "closure_power": 1.0,
        "end_margin": 0,
    },
    "small": {
        "planned_source": 0.75,
        "planned_terminal": 0.50,
        "planned_fab": 0.00,
        "planned_osat": 1.00,
        "planned_other": 0.00,
        "new_source": 0.50,
        "new_terminal": 0.50,
        "new_fab": 1.00,
        "new_osat": 0.00,
        "new_other": 1.00,
        "closure_power": 0.5,
        "end_margin": 1,
    },
    "full": {
        "planned_source": 0.75,
        "planned_terminal": 0.50,
        "planned_fab": 0.00,
        "planned_osat": 1.00,
        "planned_other": 0.00,
        "new_source": 0.50,
        "new_terminal": 0.50,
        "new_fab": 1.00,
        "new_osat": 0.00,
        "new_other": 1.00,
        "closure_power": 0.5,
        "end_margin": 1,
    },
}


def _device() -> torch.device:
    requested = os.environ.get("SBF_EVAL_DEVICE", "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested through SBF_EVAL_DEVICE={requested!r}, but Torch cannot use CUDA")
    return device


DEVICE = _device()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    POLICY = torch.jit.load(str(Path(__file__).resolve().with_name("policy.pt")), map_location=DEVICE)
POLICY.eval()
if int(POLICY.feature_version) != FEATURE_VERSION or int(POLICY.feature_dim) != FEATURE_DIM:
    raise RuntimeError("policy.pt was exported for a different feature schema; retrain agents/ppo")
HIDDEN_SIZE = int(POLICY.hidden_size)
RESIDUAL_SCALE = float(POLICY.residual_scale)


def _log_ratio(value, scale):
    """Map a nonnegative quantity/scale ratio to a stable [0, 1] feature."""
    return np.clip(np.log1p(np.maximum(value, 0.0) / max(float(scale), EPS)) / math.log(11.0), 0.0, 1.0)


class FeatureBuilder:
    """Convert the variable-size observation into one fixed-width row per action slot."""

    def __init__(self, config):
        self.T = int(config["T"])
        static = config["static"]
        instance = static["instance"]
        task = _TASK_BY_INSTANCE.get(static["instance_id"])
        if task is None:
            raise ValueError(f"unsupported instance {static['instance_id']!r}")
        self.parameters = _RESERVE[task]

        slots = static["action_slots"]
        edges = static["edges"]
        lanes = static["lanes"]
        nodes = static["nodes"]
        commodities = static["commodities"]
        self.slot_edge = np.asarray(slots["edge"], dtype=np.int64)
        self.slot_k = np.asarray(slots["k"], dtype=np.int64)
        self.slot_lane = np.asarray([-1 if lane is None else lane for lane in slots["lane"]], dtype=np.int64)
        self.n_slots = len(self.slot_edge)
        self.edge_tail = np.asarray(edges["tail"], dtype=np.int64)
        self.edge_head = np.asarray(edges["head"], dtype=np.int64)
        self.edge_u0 = np.asarray([np.nan if x is None else x for x in edges["u0"]], dtype=np.float64)
        self.edge_c0 = np.asarray(edges["c0"], dtype=np.float64)
        self.edge_tau0 = np.asarray(edges["tau0"], dtype=np.float64)
        self.capacity = self.edge_u0[self.slot_edge]
        self.capacity_ref = float(np.median(self.capacity[self.capacity > 0.0]))
        self.node_types = list(nodes["type"])

        self.paths = []
        self.route_chokepoints = []
        self.transit = np.empty(self.n_slots, dtype=np.float64)
        for i, (edge, lane) in enumerate(zip(self.slot_edge, self.slot_lane, strict=True)):
            path = [int(edge)] if lane < 0 else [int(x) for x in lanes["edges"][lane]]
            self.paths.append(path)
            self.route_chokepoints.append([] if lane < 0 else [int(x) for x in lanes["chokepoints"][lane]])
            self.transit[i] = float(np.sum(self.edge_tau0[path]))

        edge_ids = list(edges["id"])
        commodity_ids = list(commodities["id"])
        lane_ids = list(lanes["id"])
        nominal_by_route = {}
        for shipment in instance["initial_state"]["pipeline"]:
            if shipment["dispatch_week"] == 0:
                key = (shipment["edge"], shipment["k"], shipment.get("lane"))
                nominal_by_route[key] = nominal_by_route.get(key, 0.0) + float(shipment["qty"])
        self.nominal = np.asarray(
            [
                nominal_by_route.get(
                    (edge_ids[e], commodity_ids[k], None if lane < 0 else lane_ids[lane]),
                    0.0,
                )
                for e, k, lane in zip(self.slot_edge, self.slot_k, self.slot_lane, strict=True)
            ],
            dtype=np.float64,
        )
        self.nominal = np.minimum(self.nominal, self.capacity)

        stages = []
        reserve = []
        for i, edge in enumerate(self.slot_edge):
            stage = self.node_types[self.edge_tail[edge]]
            if stage not in {"source", "terminal", "fab", "osat"}:
                stage = "other"
            stages.append(stage)
            prefix = "planned" if self.nominal[i] > 0.0 else "new"
            reserve.append(self.parameters[f"{prefix}_{stage}"])
        self.reserve = np.asarray(reserve, dtype=np.float64)
        self.last_useful = self._last_useful_weeks(instance, nodes, edges, lanes, commodities)

        stock_positions = {tuple(pair): i for i, pair in enumerate(config["layout"]["stock_slots"])}
        self.stock_index = np.asarray(
            [stock_positions.get((int(self.edge_tail[e]), int(k)), -1) for e, k in zip(self.slot_edge, self.slot_k)],
            dtype=np.int64,
        )
        self.stock_pair_index = stock_positions
        self.chokepoint_position = {int(node): i for i, node in enumerate(config["layout"]["chokepoints"])}
        self.lot_keys = [tuple(int(x) for x in key) for key in config["layout"].get("lot_keys", ())]
        self.warning_chokepoint = {}
        for i, unit in enumerate(config["layout"]["warning_units"]):
            if unit[0] == "chokepoint":
                self.warning_chokepoint[int(unit[1])] = i

        raw_nodes = {node["id"]: node for node in instance["nodes"]}
        grid_load = []
        for node in config["layout"]["grids"]:
            grid_load.append(float(raw_nodes[nodes["id"][node]]["grid"]["base_load"]))
        self.grid_load = max(float(np.sum(grid_load)), 1.0)
        self.total_capacity = max(float(np.sum(self.capacity)), 1.0)

        self.static_features = np.zeros((self.n_slots, STATIC_DIM), dtype=np.float32)
        for i, stage in enumerate(stages):
            self.static_features[i, 0] = 1.0
            self.static_features[i, 1] = float(np.clip(self.nominal[i] / self.capacity[i], 0.0, 1.0))
            self.static_features[i, 2] = float(_log_ratio(self.capacity[i], self.capacity_ref))
            self.static_features[i, 3] = float(np.clip(self.transit[i] / self.T, 0.0, 1.0))
            self.static_features[i, 4] = float(np.clip(self.last_useful[i] / self.T, 0.0, 1.0))
            self.static_features[i, 5] = float(self.slot_lane[i] >= 0)
            self.static_features[i, 6] = float(self.nominal[i] > 0.0)
            self.static_features[i, 7 + ("source", "terminal", "fab", "osat", "other").index(stage)] = 1.0

        self.previous_dynamic = None
        self.last_scale = self.capacity.copy()
        self.last_base = np.zeros(self.n_slots, dtype=np.float64)
        self.last_active = np.zeros(self.n_slots, dtype=np.float64)

    def reset(self):
        self.previous_dynamic = None

    def _last_useful_weeks(self, instance, nodes, edges, lanes, commodities):
        node_ids = list(nodes["id"])
        commodity_ids = list(commodities["id"])
        commodity_position = {name: i for i, name in enumerate(commodity_ids)}
        raw_node = {node["id"]: node for node in instance["nodes"]}
        routes = {}
        destinations = []
        for edge, commodity, path in zip(self.slot_edge, self.slot_k, self.paths, strict=True):
            destination = int(self.edge_head[path[-1]])
            routes.setdefault((int(self.edge_tail[edge]), int(commodity)), []).append(
                (destination, int(commodity), int(np.sum(self.edge_tau0[path])))
            )
            destinations.append(destination)

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
            if self.node_types[node] == "sink" and commodity_id in raw.get("sink", {}).get("demand", {}):
                choices.append(0)
            if self.node_types[node] == "grid" and commodity_id in raw.get("grid", {}).get("shares", {}):
                choices.append(0)
            fab = raw.get("fab")
            if fab is not None and commodity_id == fab["input"]:
                choices.append(int(fab["tau"]) + 1 + remaining(node, commodity_position[fab["product"]]))
            osat = raw.get("osat")
            if osat is not None and commodity_id in osat["packages"]:
                choices.append(
                    int(osat["tau"]) + 1 + remaining(node, commodity_position[osat["packages"][commodity_id]])
                )
            for destination, next_commodity, transit in routes.get(key, ()):
                choices.append(transit + 1 + remaining(destination, next_commodity))
            visiting.remove(key)
            memo[key] = min(choices, default=math.inf)
            return memo[key]

        margin = int(self.parameters["end_margin"])
        deadlines = []
        for destination, commodity, transit in zip(destinations, self.slot_k, self.transit, strict=True):
            tail = remaining(destination, int(commodity))
            deadlines.append(self.T if not math.isfinite(tail) else self.T - int(transit) - int(tail) + margin)
        return np.asarray(deadlines, dtype=np.int64)

    @staticmethod
    def _valid(observation, prefix):
        return np.flatnonzero(np.asarray(observation[f"{prefix}.observed"], dtype=bool).reshape(-1))

    def _dynamic(self, observation):
        week = int(np.asarray(observation["week"]).item())
        dynamic = np.zeros((self.n_slots, DYNAMIC_DIM), dtype=np.float64)

        stock = np.asarray(observation["stock.qty"], dtype=np.float64)
        stock_observed = np.asarray(observation["stock.qty.observed"], dtype=bool)
        tail_stock = np.zeros(self.n_slots, dtype=np.float64)
        for i, position in enumerate(self.stock_index):
            if position >= 0 and stock_observed[position]:
                tail_stock[i] = stock[position]

        inbound = {}
        inbound_soon = {}
        valid = self._valid(observation, "pipeline.edge")
        p_edge = np.asarray(observation["pipeline.edge"], dtype=np.int64)
        p_k = np.asarray(observation["pipeline.k"], dtype=np.int64)
        p_qty = np.asarray(observation["pipeline.qty"], dtype=np.float64)
        p_arrival = np.asarray(observation["pipeline.arrival_week"], dtype=np.int64)
        for j in valid:
            key = (int(self.edge_head[p_edge[j]]), int(p_k[j]))
            inbound[key] = inbound.get(key, 0.0) + max(float(p_qty[j]), 0.0)
            if p_arrival[j] <= week + 4:
                inbound_soon[key] = inbound_soon.get(key, 0.0) + max(float(p_qty[j]), 0.0)

        wip = {}
        wip_soon = {}
        valid = self._valid(observation, "wip.node")
        w_node = np.asarray(observation["wip.node"], dtype=np.int64)
        w_k = np.asarray(observation["wip.k"], dtype=np.int64)
        w_qty = np.asarray(observation["wip.qty"], dtype=np.float64)
        w_out = np.asarray(observation["wip.out_week"], dtype=np.int64)
        for j in valid:
            key = (int(w_node[j]), int(w_k[j]))
            wip[key] = wip.get(key, 0.0) + max(float(w_qty[j]), 0.0)
            if w_out[j] <= week + 4:
                wip_soon[key] = wip_soon.get(key, 0.0) + max(float(w_qty[j]), 0.0)

        queued = np.zeros(self.n_slots, dtype=np.float64)
        queue_map = {}
        if "queue_lots.next_edge" in observation:
            valid = self._valid(observation, "queue_lots.next_edge")
            q_edge = np.asarray(observation["queue_lots.next_edge"], dtype=np.int64)
            q_k = np.asarray(observation["queue_lots.k"], dtype=np.int64)
            q_lane = np.asarray(observation["queue_lots.lane"], dtype=np.int64)
            q_qty = np.asarray(observation["queue_lots.qty"], dtype=np.float64)
            for j in valid:
                key = (int(q_edge[j]), int(q_k[j]), int(q_lane[j]))
                queue_map[key] = queue_map.get(key, 0.0) + max(float(q_qty[j]), 0.0)
        else:
            q_qty = np.asarray(observation["queue_lots.qty"], dtype=np.float64)
            q_observed = np.asarray(observation["queue_lots.qty.observed"], dtype=bool)
            for row, (_chokepoint, commodity, lane, next_edge) in enumerate(self.lot_keys):
                quantity = float(np.sum(np.where(q_observed[row], np.maximum(q_qty[row], 0.0), 0.0)))
                key = (next_edge, commodity, lane)
                queue_map[key] = queue_map.get(key, 0.0) + quantity
        for i, (commodity, lane, path) in enumerate(zip(self.slot_k, self.slot_lane, self.paths, strict=True)):
            for edge in path:
                queued[i] += queue_map.get((edge, int(commodity), int(lane)), 0.0)

        pending_week = np.full(self.n_slots, self.T + 1, dtype=np.int64)
        valid = self._valid(observation, "pending_prohibitions.edge")
        pe = np.asarray(observation["pending_prohibitions.edge"], dtype=np.int64)
        pk = np.asarray(observation["pending_prohibitions.k"], dtype=np.int64)
        pw = np.asarray(observation["pending_prohibitions.effective_week"], dtype=np.int64)
        pending_map = {}
        for j in valid:
            key = (int(pe[j]), int(pk[j]))
            pending_map[key] = min(pending_map.get(key, self.T + 1), int(pw[j]))
        for i, (commodity, path) in enumerate(zip(self.slot_k, self.paths, strict=True)):
            pending_week[i] = min(
                (pending_map.get((edge, int(commodity)), self.T + 1) for edge in path), default=self.T + 1
            )

        closure_end = {}
        valid = self._valid(observation, "closure_end.chokepoint")
        ce_cp = np.asarray(observation["closure_end.chokepoint"], dtype=np.int64)
        ce_week = np.asarray(observation["closure_end.end_week"], dtype=np.int64)
        for j in valid:
            closure_end[int(ce_cp[j])] = max(closure_end.get(int(ce_cp[j]), week), int(ce_week[j]))

        graph_u = np.asarray(observation["graph_now.u"], dtype=np.float64)
        graph_u_obs = np.asarray(observation["graph_now.u.observed"], dtype=bool)
        graph_c = np.asarray(observation["graph_now.c"], dtype=np.float64)
        graph_c_obs = np.asarray(observation["graph_now.c.observed"], dtype=bool)
        graph_tau = np.asarray(observation["graph_now.tau"], dtype=np.float64)
        prohibited = np.asarray(observation["graph_now.prohibited"], dtype=np.float64)
        prohibited_obs = np.asarray(observation["graph_now.prohibited.observed"], dtype=bool)
        cp_open = np.asarray(observation["graph_now.open"], dtype=np.float64)
        cp_open_obs = np.asarray(observation["graph_now.open.observed"], dtype=bool)
        kappa_tb = np.asarray(observation["graph_now.kappa.tb"], dtype=np.float64)
        kappa_ct = np.asarray(observation["graph_now.kappa.ct"], dtype=np.float64)
        war = np.asarray(observation["graph_now.war_risk"], dtype=np.float64)
        tariff = np.asarray(observation["graph_now.tariff"], dtype=np.float64)
        tariff_obs = np.asarray(observation["graph_now.tariff.observed"], dtype=bool)
        warnings_score = np.asarray(observation["warning.score"], dtype=np.float64)
        warnings_obs = np.asarray(observation["warning.score.observed"], dtype=bool)
        warning_global = float(np.max(warnings_score[warnings_obs])) if np.any(warnings_obs) else 0.0

        demand = np.asarray(observation["last_week.sinks.demand"], dtype=np.float64)
        lost = np.asarray(observation["last_week.sinks.lost"], dtype=np.float64)
        lost_share = float(np.sum(lost) / max(float(np.sum(demand)), 1.0))
        shed = np.asarray(observation["last_week.shed.qty"], dtype=np.float64)
        shed_share = float(np.sum(shed) / self.grid_load)
        distress = float(np.clip(max(lost_share, shed_share), 0.0, 1.0))
        forecast = np.asarray(observation["demand_forecast.qty"], dtype=np.float64)
        forecast_pressure = float(_log_ratio(np.mean(np.sum(forecast, axis=0)), self.total_capacity))

        msg_valid = self._valid(observation, "messages.msg_id")
        msg_count = float(np.clip(math.log1p(len(msg_valid)) / math.log(65.0), 0.0, 1.0))
        msg_effective = np.asarray(observation["messages.stated_effective_week"], dtype=np.int64)
        future_messages = [int(msg_effective[j]) for j in msg_valid if msg_effective[j] >= week]
        msg_horizon = min(future_messages, default=self.T + 1)
        msg_nearness = float(np.clip(1.0 - (msg_horizon - week) / max(self.T, 1), 0.0, 1.0))

        costs = np.asarray(observation["last_week.cost_components"], dtype=np.float64)
        cost_scale = self.total_capacity * 1_000.0
        recent_cost = float(_log_ratio(np.sum(np.maximum(costs, 0.0)), cost_scale))
        requested = np.asarray(observation["last_week.clip.requested"], dtype=np.float64)
        executed = np.asarray(observation["last_week.clip.executed"], dtype=np.float64)
        action_mask = np.asarray(observation["action_mask"], dtype=np.float64)

        # The training action wrapper and the server both interpret the actor's
        # fraction against nominal slot capacity.  Current capacity remains a
        # feature and the simulator performs the final shared-edge clip.
        scale = self.capacity.copy()
        base = np.zeros(self.n_slots, dtype=np.float64)
        active = np.ones(self.n_slots, dtype=np.float64)
        for i, (edge, commodity, path, cps) in enumerate(
            zip(self.slot_edge, self.slot_k, self.paths, self.route_chokepoints, strict=True)
        ):
            first = int(edge)
            available = self.capacity[i]
            if graph_u_obs[first] and np.isfinite(graph_u[first]):
                available = max(float(graph_u[first]), 0.0)
            cap_ratio = float(np.clip(available / max(self.capacity[i], EPS), 0.0, 2.0) / 2.0)
            current_cost = 0.0
            nominal_cost = max(float(np.sum(self.edge_c0[path])), EPS)
            observed_parts = [float(graph_u_obs[first]), float(action_mask[i] >= 0.0)]
            current_tau = 0.0
            is_prohibited = 0.0
            for route_edge in path:
                current_cost += float(graph_c[route_edge] if graph_c_obs[route_edge] else self.edge_c0[route_edge])
                current_tau += float(graph_tau[route_edge])
                if prohibited_obs[route_edge, commodity]:
                    is_prohibited = max(is_prohibited, float(prohibited[route_edge, commodity]))
                    observed_parts.append(1.0)
                if tariff_obs[route_edge, commodity]:
                    current_cost += max(float(tariff[route_edge, commodity]), 0.0)
                    observed_parts.append(1.0)

            open_value = 1.0
            tb = ct = war_value = route_warning = 0.0
            end_nearness = 0.0
            for chokepoint in cps:
                position = self.chokepoint_position.get(chokepoint)
                if position is None:
                    continue
                if cp_open_obs[position]:
                    open_value = min(open_value, max(float(cp_open[position]), 0.0))
                    observed_parts.append(1.0)
                tb = max(tb, float(kappa_tb[position]))
                ct = max(ct, float(kappa_ct[position]))
                war_value = max(war_value, float(war[position]))
                wi = self.warning_chokepoint.get(chokepoint)
                if wi is not None and warnings_obs[wi]:
                    route_warning = max(route_warning, float(warnings_score[wi]))
                if chokepoint in closure_end:
                    end_nearness = max(
                        end_nearness,
                        float(np.clip(1.0 - (closure_end[chokepoint] - week) / max(self.T, 1), 0.0, 1.0)),
                    )

            key = (int(self.edge_tail[first]), int(commodity))
            current_nominal = self.nominal[i] / max(scale[i], EPS)
            base[i] = np.clip(current_nominal + self.reserve[i] * (1.0 - current_nominal), 0.0, 1.0)
            base[i] *= open_value ** float(self.parameters["closure_power"])
            active[i] = float(
                week <= self.last_useful[i]
                and available > EPS
                and action_mask[i] > 0.0
                and is_prohibited <= 0.0
                and open_value > 0.0
            )
            base[i] *= active[i]

            dynamic[i] = (
                week / self.T,
                max(self.T - week, 0) / self.T,
                _log_ratio(tail_stock[i], scale[i]),
                _log_ratio(inbound.get(key, 0.0), scale[i]),
                _log_ratio(inbound_soon.get(key, 0.0), scale[i]),
                _log_ratio(wip.get(key, 0.0), scale[i]),
                _log_ratio(wip_soon.get(key, 0.0), scale[i]),
                _log_ratio(queued[i], scale[i]),
                cap_ratio,
                np.clip(current_cost / nominal_cost, 0.0, 5.0) / 5.0,
                np.clip(current_tau / max(self.transit[i], 1.0), 0.0, 3.0) / 3.0,
                np.clip(action_mask[i], 0.0, 1.0),
                np.clip(is_prohibited, 0.0, 1.0),
                np.clip(open_value, 0.0, 1.0),
                np.clip(tb, 0.0, 3.0) / 3.0,
                np.clip(ct, 0.0, 3.0) / 3.0,
                np.clip(war_value, 0.0, 3.0) / 3.0,
                np.clip(requested[i] / max(self.capacity[i], EPS), 0.0, 2.0) / 2.0,
                np.clip(executed[i] / max(self.capacity[i], EPS), 0.0, 2.0) / 2.0,
                float(np.clip(1.0 - (pending_week[i] - week) / 8.0, 0.0, 1.0)),
                end_nearness,
                np.clip(max(route_warning, warning_global), 0.0, 1.0),
                distress,
                forecast_pressure,
                msg_count,
                msg_nearness,
                recent_cost,
                float(np.clip(np.mean(observed_parts), 0.0, 1.0)),
            )

        self.last_scale = scale
        self.last_base = base
        self.last_active = active
        return dynamic.astype(np.float32)

    def encode(self, observation):
        current = self._dynamic(observation)
        previous = current if self.previous_dynamic is None else self.previous_dynamic
        features = np.concatenate((self.static_features, current, previous, current - previous), axis=1)
        self.previous_dynamic = current.copy()
        return features.astype(np.float32, copy=False)


def _fractions(base, active, mean):
    clipped = np.clip(base, EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)
    return (1.0 / (1.0 + np.exp(-(logits + RESIDUAL_SCALE * np.tanh(mean))))) * active


class Agent:
    """Stateful deterministic actor; exploration exists only in ``train.py``."""

    def __init__(self, config):
        self.features = FeatureBuilder(config)
        self.hidden = torch.zeros((1, self.features.n_slots, HIDDEN_SIZE), dtype=torch.float32, device=DEVICE)

    def act(self, observation):
        encoded = self.features.encode(observation)
        x = torch.as_tensor(encoded[None], dtype=torch.float32, device=DEVICE)
        with torch.inference_mode():
            mean, _log_std, _value, self.hidden = POLICY(x, self.hidden)
        fraction = _fractions(self.features.last_base, self.features.last_active, mean[0].cpu().numpy())
        return {"flows": fraction * self.features.last_scale}

    @staticmethod
    def act_batch(agents, observations):
        if not agents:
            return []
        slot_count = agents[0].features.n_slots
        if any(agent.features.n_slots != slot_count for agent in agents):
            return [agent.act(observation) for agent, observation in zip(agents, observations, strict=True)]
        encoded = np.stack([agent.features.encode(obs) for agent, obs in zip(agents, observations, strict=True)])
        hidden = torch.cat([agent.hidden for agent in agents], dim=0)
        with torch.inference_mode():
            mean, _log_std, _value, next_hidden = POLICY(
                torch.as_tensor(encoded, dtype=torch.float32, device=DEVICE), hidden
            )
        means = mean.cpu().numpy()
        actions = []
        for i, agent in enumerate(agents):
            agent.hidden = next_hidden[i : i + 1]
            fraction = _fractions(agent.features.last_base, agent.features.last_active, means[i])
            actions.append({"flows": fraction * agent.features.last_scale})
        return actions
