import os
import math
import numpy as np

# ==========================================
# AUTO-UPDATED BY tune.py
# ==========================================
_TUNED_OVERRIDES = {
    "use_cap": 1,          
    "cost_gain": 0.5,      
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
    "end_margin": 1,
}

# ==========================================
# AUTO-UPDATED BY es_train.py
# ==========================================
# <<POLICY_BEGIN>>
_POLICY_WEIGHTS = [
    0.041998102932846164, -0.010351186180839739, 0.059405030198763684, 0.12599444380976033, -0.10361517953624821, 0.10717115617536677,
    0.1309556440923347, 0.02567876222992108, 0.006436183898343201, -0.03626858166063073, -0.08692892223016807, 0.10800464672344101,
    -0.2303821334263175, -0.1346715008496264, 0.02231644107747502, -0.05795891208489875, -0.03386513364198868, -0.0068887560399795225,
    0.04952081593303933, 0.21193799297050345, 0.04888442529989798, 0.17363866845297912, -0.23864390815649442, -0.0008356391431325803,
    0.0037209036567248414, 0.11479146274100002, -0.047799365591013054, -0.15273477005187241, 0.032690108537962996, -0.12170300724975383,
    -0.1300322292534452, 0.0898695702309717, 0.00785305084468658, 0.08378107161977627, -0.008162545614185907, 0.10357003364439422,
    -0.02142052650249411, -0.037488174151058956, 0.21342354268076677, 0.24518907309600232, -0.04359049160862491, 0.14991574506424504,
    0.13879650947936675, 0.07867460721231541, 0.0486064054278864, -0.06568851292701042, 0.21183327958477574, 0.2060254478680136,
    0.10676739062366519, 0.12333079227308143, 0.014916997128954713, -0.12143738392054633, -0.08806965250467655, 0.043283133848263974,
    -0.12430219921724905, -0.0020479992279784525, 0.07159471296778976, 0.14970231815129795, -0.22545540747312595, -0.03123891447478709,
    -0.20560587698513522, -0.09381507875742612, 0.1671886892118209, -0.1115241616565599, -0.08399723574723242, -0.15480207190333237,
    0.0936781036308955, 0.15523410304328694, 0.11109302568677158, -0.24321023448862902, 0.08620454346733786, 0.07843812520936548,
    0.08493558012154427, -0.0055147053318033695, 0.16610450624128337, -0.17845426663233951, 0.05687985848811457, 0.12464504951828424,
    0.09229455193874712, 0.22287209482310602, -0.05193072140906299, -0.11489671497269804, -0.04696454762382711, -0.13843514611904645,
    -0.19391573828155406, 0.13040896143396913, 0.16123475927501088, 0.2339531985447629, -0.09825324223082788, 0.06798456981684543,
    0.09620885264751675, 0.19114671205504805, -0.04103297513670119, -0.19348765228567455, 0.03245767982931091, -0.0005654066699846339,
    -0.1356576733810562, 0.01211812226611282, -0.18780710202612338, -0.05785607074103415, -0.014559018811315579, 0.19471213005111088,
    0.08969841019303737, -0.052920872457085585, -0.07344551445610809, -0.20879569187427835, -0.041698962563062755, 0.1103771745305716,
    -0.20158356840166747, -0.04691495388346673, -0.10061852828776498, 0.017945764901409086, 0.10256108101239879, 0.019528154462499038,
    0.032355069955640685, -0.07695809829014266, 0.012502264870581065, -0.002243103100707985, 0.1583640245279536, 0.07791870193345546,
    0.24139335914047488, 0.09810141034569898, 0.028051591260198294, -0.15415615716475128, -0.14705915008599751, -0.04663181223476712,
    -0.1853935838199808, 0.09946294194603708, -0.010874927637625752, -0.06081852003479743, -0.09013109174376732, 0.061652245286845354,
    -0.0476580472954144, -0.0014006899913304885, 0.03258194206434649, 0.15679181725312116, 0.1472703682982896, 0.08339440418452389,
    -0.1874841215517987, 0.18818180537416704, 0.06763236412923444, -0.07765329350883268, -0.08163744767118239, 0.10310182969092778,
    0.047957120908338896, -0.09283342365963702, -0.1039736112151075, -0.21906229743023012, -0.07309842256661996, 0.06856873061876076,
    0.03320294902398052, -0.032944302952033766, -0.18545359697629246, 0.0049530125722211825, -0.02470117264556173, -0.06034518023938737,
    0.08278696875392387, -0.03832618907120617, -0.09052861054929638, -0.2205572177786646, -0.10048227213351989, 0.050499136706415315,
    -0.039529842373982306, 0.058250943287431856, -0.012585500333439038, 0.17213551846161962, -0.07150934306930994, -0.036382459460236806,
    0.08660683790241291, 0.07986354231231746, -0.09391133319932057, 0.03660848619316301, -0.05149956210844321, 0.009706906369343434,
    -0.10110121675116254, 0.0361742008478169, 0.09062833564971483, 0.10138404342538004, -0.022444200802641267, 0.07620335363996729,
    -0.058363246456593236, 0.06027914089344997, -0.06906462494861834, -0.05771696289622228, 0.007017466726415277, 0.050652079072012676,
    -0.02058472577235228, -0.03284601890217553, 0.021363623281710768, -0.18505564345088324, 0.06652414618800795, -0.24754684876269473,
    0.1292746756190981, -0.02461247660102732, 0.04510549314059447, -0.14604541617295336, -0.037865275375354245, -0.020945236271121177,
    0.035236103337230686, 0.0473510922612179, -0.039244564367137386, 0.2018373633066662, -0.024442536286190066, 0.12489339891668277,
    0.012664658710371191, 0.07260976844057661, -0.005099627218582326, -0.0921297334555331, 0.05004143929443526, -0.5645105756517604,
]
# <<POLICY_END>>

# --------------------------------------------------------------------------
# CONTINUOUS SPATIAL NEURAL MODULATOR 
# --------------------------------------------------------------------------
N_FEAT = 10  # Added week/T and closure_eta
N_HID = 16
N_OUT = 2
N_WEIGHTS = N_FEAT * N_HID + N_HID + N_HID * N_OUT + N_OUT

_LOG_SCALE = 1.1   # e^1.1 ≈ 3x in either direction (surge > 1, throttle < 1)
_LOG_POW = 0.7     # Multiplier for closure_power

def init_weights(seed=0, scale=0.1):
    rng = np.random.default_rng(seed)
    w = np.zeros(N_WEIGHTS)
    w[: N_FEAT * N_HID] = rng.normal(0.0, scale, N_FEAT * N_HID)
    return w

class Agent:
    def __init__(self, config, _weights=None, _policy=None):
        weights = dict(_TUNED_OVERRIDES)
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
                    (edge_ids[e], commodity_ids[k], None if lane is None else lane_ids[lane]), 0.0
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
            planned = self.nominal[i] > 0.0
            prefix = "planned" if planned else "new"
            reserve.append(weights[f"{prefix}_{stage}"])
        self.reserve = np.clip(np.asarray(reserve, dtype=np.float64), 0.0, 1.0)

        chokepoint_position = {node: i for i, node in enumerate(config["layout"]["chokepoints"])}
        self.n_cp = len(chokepoint_position)

        self.incidence = np.zeros((self.n_slots, max(self.n_cp, 1)), dtype=bool)
        for slot, lane in enumerate(slot_lanes):
            if lane is not None:
                for c in lanes["chokepoints"][lane]:
                    self.incidence[slot, chokepoint_position[c]] = True

        self.lu0, self.lu_free = self._last_useful_weeks(
            instance, nodes, edges, lanes, commodities, slot_edges, slot_commodities, slot_lanes,
        )

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

        self._prev_week = -1
        self._init_modulator(_policy)

    def _init_modulator(self, policy):
        w = None
        if policy is not None:
            w = np.asarray(policy, dtype=np.float64).ravel()
            if w.size != N_WEIGHTS:
                raise ValueError(f"policy needs {N_WEIGHTS} weights, got {w.size}")
        else:
            # THE SMOKING GUN: Restoring the environment hook for SBF subprocesses
            path = os.environ.get("SBF_POLICY_PATH")
            if path:
                try:
                    c = np.asarray(np.load(path), dtype=np.float64).ravel()
                    w = c if c.size == N_WEIGHTS else None
                except Exception:
                    w = None
            if w is None and _POLICY_WEIGHTS is not None and np.size(_POLICY_WEIGHTS) == N_WEIGHTS:
                w = np.asarray(_POLICY_WEIGHTS, dtype=np.float64).ravel()
            if w is None:
                w = init_weights()
        self.set_policy(w)

    def set_policy(self, w):
        w = np.asarray(w, dtype=np.float64).ravel()
        i = 0
        self.W1 = w[i:i + N_FEAT * N_HID].reshape(N_FEAT, N_HID); i += N_FEAT * N_HID
        self.b1 = w[i:i + N_HID]; i += N_HID
        self.W2 = w[i:i + N_HID * N_OUT].reshape(N_HID, N_OUT); i += N_HID * N_OUT
        self.b2 = w[i:i + N_OUT]
        self.policy = w

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
            if key in memo: return memo[key]
            if key in visiting: return math.inf
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

        deadlines = []
        free = []
        for destination, commodity, transit in zip(slot_destinations, slot_commodities, slot_transit, strict=True):
            after_arrival = remaining(destination, commodity)
            if math.isfinite(after_arrival):
                deadlines.append(self.T - transit - int(after_arrival))
                free.append(False)
            else:
                deadlines.append(self.T)
                free.append(True)
        return np.asarray(deadlines, dtype=np.int64), np.asarray(free, dtype=bool)

    def _distress_parts(self, obs):
        def total(key):
            v = obs.get(key)
            return 0.0 if v is None else float(np.nan_to_num(np.asarray(v, dtype=np.float64)).sum())
        lost_share = total("last_week.sinks.lost") / max(total("last_week.sinks.demand"), 1.0)
        shed_share = total("last_week.shed.qty") / max(float(self.grid_load.sum()), 1.0)
        return lost_share, shed_share

    def _modulate(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        return _LOG_SCALE * out[:, 0], _LOG_POW * out[:, 1]

    def act(self, observation):
        w = self.weights
        week = int(np.asarray(observation["week"]).item())
        first = self._prev_week < 0
        self._prev_week = week

        open_now = np.nan_to_num(np.asarray(observation["graph_now.open"], dtype=np.float64))[: max(self.n_cp, 1)]
        observed = np.asarray(observation["graph_now.open.observed"], dtype=bool)[: max(self.n_cp, 1)]
        
        eff = np.where(observed, np.clip(open_now, 0.0, 1.0), 1.0)
        self.ewma_open = eff.copy() if first else 0.2 * eff + 0.8 * self.ewma_open

        # NaN-safe Inventory
        pipe_qty = np.asarray(observation.get("pipeline.qty", []), dtype=np.float64)
        pipe_obs = np.asarray(observation.get("pipeline.qty.observed", []), dtype=bool)
        wip_qty = np.asarray(observation.get("wip.qty", []), dtype=np.float64)
        wip_obs = np.asarray(observation.get("wip.qty.observed", []), dtype=bool)
        
        p_sum = np.sum(np.where(pipe_obs & np.isfinite(pipe_qty), pipe_qty, 0.0))
        w_sum = np.sum(np.where(wip_obs & np.isfinite(wip_qty), wip_qty, 0.0))
        global_inventory = np.log1p((p_sum + w_sum) / max(float(self.capacity.sum()), 1.0))

        # NaN-safe Cost Ratio
        c_ratio = np.ones(self.n_slots)
        if self.slot_c0 is not None and "graph_now.c" in observation:
            c_now = np.asarray(observation["graph_now.c"], dtype=np.float64)
            c_obs = np.asarray(observation["graph_now.c.observed"], dtype=bool)
            if c_now.shape[0] > int(self.slot_edge_idx.max()):
                c_edge = c_now[self.slot_edge_idx]
                c_obs_edge = c_obs[self.slot_edge_idx]
                c_ratio = np.where(c_obs_edge & np.isfinite(c_edge), c_edge / self.slot_c0, 1.0)
        
        c_ratio_cp = np.ones(max(self.n_cp, 1))
        if self.n_cp > 0:
            c_ratio_cp = (c_ratio[:, None] * self.incidence[:, :self.n_cp]).sum(axis=0) / np.maximum(self.incidence[:, :self.n_cp].sum(axis=0), 1)
        c_ratio_cp = np.clip(c_ratio_cp, 0.0, 5.0)

        # NaN-safe Warnings
        warn = np.asarray(observation.get("warning.score", []), dtype=np.float64)
        warn_obs = np.asarray(observation.get("warning.score.observed", []), dtype=bool)
        valid_warn = warn[warn_obs & np.isfinite(warn)]
        max_warn = float(np.max(valid_warn)) if valid_warn.size > 0 else 0.0
        mean_warn = float(np.mean(valid_warn)) if valid_warn.size > 0 else 0.0

        # Closure ETA Feature
        closure_eta = np.zeros(max(self.n_cp, 1))
        c_end_cp = np.asarray(observation.get("closure_end.chokepoint", []), dtype=np.int64)
        c_end_wk = np.asarray(observation.get("closure_end.end_week", []), dtype=np.float64)
        c_end_obs = np.asarray(observation.get("closure_end.end_week.observed", []), dtype=bool)
        if c_end_cp.size > 0:
            valid_end = c_end_obs & np.isfinite(c_end_wk) & (c_end_wk >= week)
            for cp, wk in zip(c_end_cp[valid_end], c_end_wk[valid_end]):
                if cp < self.n_cp:
                    closure_eta[cp] = max(closure_eta[cp], 1.0 / (1.0 + (wk - week)))

        lost_share, shed_share = self._distress_parts(observation)
        distress = min(max(lost_share, shed_share, 0.0), 1.0)

        # ------------------------------------------------------------------
        # NEURAL EVALUATION (10 Features -> log_s, log_p)
        # ------------------------------------------------------------------
        if self.n_cp > 0:
            x = np.column_stack([
                eff, self.ewma_open, eff - self.ewma_open,
                np.full(self.n_cp, global_inventory), c_ratio_cp,
                np.full(self.n_cp, max_warn), np.full(self.n_cp, mean_warn),
                np.full(self.n_cp, distress),
                np.full(self.n_cp, week / max(self.T, 1)),
                closure_eta
            ])
            log_s, log_p = self._modulate(x)
        else:
            log_s, log_p = np.zeros(0), np.zeros(0)

        # ------------------------------------------------------------------
        # APPLY BASELINE AND LOCALIZED RL OVERRIDES
        # ------------------------------------------------------------------
        reserve = np.clip(self.reserve, 0.0, 1.0)
        if float(w["distress_gain"]) != 0.0:
            reserve = reserve + float(w["distress_gain"]) * distress
        
        if float(w["cost_gain"]) > 0.0 and self.slot_c0 is not None:
            reserve = reserve * np.clip(1.0 - float(w["cost_gain"]) * np.maximum(c_ratio - 1.0, 0.0), 0.0, 1.0)

        flows = self.nominal + reserve * (self.capacity - self.nominal)

        # Geometric Mean Lane Scaling for Surge/Throttle
        if self.n_cp > 0:
            inc = self.incidence[:, :self.n_cp]
            slot_scale = np.exp((inc * log_s[None, :]).sum(axis=1) / np.maximum(inc.sum(axis=1), 1))
            flows = flows * slot_scale

        last_useful = np.where(self.lu_free, self.T, self.lu0 + int(w["end_margin"]))
        flows = np.where(week <= last_useful, flows, 0.0)

        if w["use_cap"]:
            u_now = np.asarray(observation["graph_now.u"], dtype=np.float64)
            if u_now.shape[0] == self.n_edges:
                u_pad = np.append(u_now, np.inf)
                flows = np.minimum(flows, u_pad[self.path_idx].min(axis=1))

        # Dynamic Closure Power
        if float(w["closure_power"]) > 0.0 and self.n_cp > 0:
            power_cp = float(w["closure_power"]) * np.exp(log_p)
            factor = np.where(observed, np.maximum(open_now, 0.0) ** power_cp, 1.0)
            slot_factor = np.where(self.incidence[:, :self.n_cp], factor[None, :], 1.0).prod(axis=1)
            flows = flows * slot_factor

        flows = np.clip(np.nan_to_num(flows), 0.0, self.capacity)
        return {"flows": flows * observation["action_mask"]}