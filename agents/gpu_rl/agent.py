"""A policy without MPC, trained by PPO on the tensor simulator (examples/gpu_ppo.py); numpy only.

Every week each route (action slot) gets the numbers of ``sbf_starter.gpu.rl.FEATURES`` (31; a network of the first
25 reads ``BASE_COLUMNS`` of them), read here from the observation, and one small network shared by every route
(so the same agent plays small and full) gives a logit z;
the route requests sigmoid(z) of its capacity this week (the least capacity on its edges). z is the network's output
plus the logit of the route's share of capacity in the nominal plan.

If the network has the neighbours' mix (``mix_*`` arrays: examples/gpu_es.py --pool), each route's description is
mixed with the mean description of the routes into the same destination and out of the same tail before the logit
(``sbf_starter.gpu.rl.RoutePolicy`` with pool).

Files beside this one (examples/gpu_export.py writes them): ``policy.npz`` (the network) and
``tables_<instance_id>.npz`` per map (the route tables: groups, edges, chokepoints, nominal flows and scales, the
stock slots of each route's tail and destination as (node, commodity)).
"""

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
with np.load(HERE / "policy.npz") as _f:
    NET = {k: _f[k].astype(np.float64) for k in _f.files}
GROUPS = 12
BASE_COLUMNS = (*range(20), *range(26, 31))  # the first networks' 25 features


def _load_tables(instance_id: str) -> dict:
    with np.load(HERE / f"tables_{instance_id}.npz") as f:
        return {k: f[k] for k in f.files}


class Agent:
    def __init__(self, config=None):
        st, lay = config["static"], config["layout"]
        tb = _load_tables(st["instance_id"])
        self.T = float(st["T"])
        self.tb = tb
        pos = {(int(n), int(k)): i for i, (n, k) in enumerate(lay["stock_slots"])}
        self.tail = np.array([pos[(int(n), int(k))] for n, k in tb["tail"]])
        self.dest = np.array([pos[(int(n), int(k))] for n, k in tb["dest"]])
        A = len(tb["group"])
        self.onehot = np.zeros((A, GROUPS))
        self.onehot[np.arange(A), tb["group"]] = 1.0
        self.r_lane, self.r_k, self.r_edge = tb["lane"], tb["k"], tb["edge"]
        lot_lane = np.array([lane for _c, _k, lane, _n in lay.get("lot_keys", [])], dtype=int)
        lot_k = np.array([k for _c, k, _l, _n in lay.get("lot_keys", [])], dtype=int)
        self.queue_inc = (self.r_lane[:, None] >= 0) & (lot_lane[None, :] == self.r_lane[:, None])
        self.queue_inc &= lot_k[None, :] == self.r_k[:, None]
        self.queue_inc = self.queue_inc.astype(np.float64)
        self.tail_node = np.array([int(n) for n, _k in tb["tail"]])
        demand_of = {(int(n), int(k)): d for d, (n, k) in enumerate(lay["demands"])}
        self.route_demand = np.array([demand_of.get((int(n), int(k)), -1) for n, k in tb["dest"]])

    def features(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """The 25 features of every route [A, 25] and each route's capacity this week [A]."""
        tb = self.tb
        stock = np.where(obs["stock.qty.observed"] == 1, obs["stock.qty"], 0.0)
        u = np.where(obs["graph_now.u.observed"] == 1, obs["graph_now.u"], np.inf)
        cap = np.append(u, np.inf)[tb["route_edges"]].min(1)
        is_open = np.where(obs["graph_now.open.observed"] == 1, obs["graph_now.open"], 1.0)
        opened = np.append(is_open, 1.0)[tb["route_chks"]].min(1)
        live = obs["pipeline.qty.observed"] == 1
        qty = obs["pipeline.qty"][live]
        lane = np.where(obs["pipeline.lane.observed"][live] == 1, obs["pipeline.lane"][live], -1)
        edge, k = obs["pipeline.edge"][live], obs["pipeline.k"][live]
        on_lane = (self.r_lane[:, None] >= 0) & (lane[None, :] == self.r_lane[:, None])
        on_edge = (self.r_lane[:, None] < 0) & (lane[None, :] < 0) & (edge[None, :] == self.r_edge[:, None])
        transit = ((on_lane | on_edge) & (k[None, :] == self.r_k[:, None])) @ qty
        if self.queue_inc.shape[1]:
            rows = np.sum(np.where(obs["queue_lots.qty.observed"] == 1, obs["queue_lots.qty"], 0.0), axis=1)
            queued = self.queue_inc @ rows
        else:
            queued = np.zeros(len(cap))
        lost = np.where(obs["last_week.sinks.lost.observed"] == 1, obs["last_week.sinks.lost"], 0.0).sum()
        demand = np.where(obs["last_week.sinks.demand.observed"] == 1, obs["last_week.sinks.demand"], 0.0).sum()
        shed = np.where(obs["last_week.shed.qty.observed"] == 1, obs["last_week.shed.qty"], 0.0).sum()
        scale = tb["scale"]
        local = np.column_stack(
            [
                np.clip(cap / tb["cap0"], 0, 2),
                1 - opened,
                np.asarray(obs["slot_mask"], dtype=np.float64),
                np.log1p(np.maximum(stock[self.tail], 0) / scale),
                np.log1p(np.maximum(stock[self.dest], 0) / scale),
                np.log1p(np.maximum(transit, 0) / scale),
                np.log1p(np.maximum(queued, 0) / scale),
                tb["nominal"] / scale,
            ]
        )
        whole = [
            float(obs["week"][0]) / self.T,
            self.T / 104.0,
            float(np.mean(1 - is_open)) if len(is_open) else 0.0,
            lost / demand if demand > 0 else 0.0,
            shed / max(float(tb["base_load"]), 1e-9),
        ]
        x = np.hstack([self.onehot, local, self._new_local(obs, stock, scale), np.tile(whole, (len(cap), 1))])
        return x, cap

    def _new_local(self, obs: dict, stock: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """The six per-route features added on 2026-10-08 (``FEATURES`` 20..25)."""
        tb, week = self.tb, int(obs["week"][0])
        tariff = np.where(obs["graph_now.tariff.observed"] == 1, obs["graph_now.tariff"], 0.0)
        rate = np.clip(
            np.vstack([tariff, np.zeros((1, tariff.shape[1]))])[tb["route_edges"], self.r_k[:, None]].sum(1), 0, 3
        )
        headroom = np.maximum(tb["dest_storage"] - stock[self.dest], 0) / scale
        live = obs["wip.qty.observed"] == 1
        node, k, qty, out = (obs[f"wip.{f}"][live] for f in ("node", "k", "qty", "out_week"))
        near = (out >= week) & (out <= week + 3)
        hit = (node[None, :] == self.tail_node[:, None]) & (k[None, :] == self.r_k[:, None]) & near[None, :]
        wip = hit @ qty
        has = self.route_demand >= 0
        d = np.where(has, self.route_demand, 0)
        lost = np.where(obs["last_week.sinks.lost.observed"] == 1, obs["last_week.sinks.lost"], 0.0)
        demand = np.where(obs["last_week.sinks.demand.observed"] == 1, obs["last_week.sinks.demand"], 0.0)
        share = lost / np.where(demand > 0, demand, 1.0)
        fc = np.where(obs["demand_forecast.qty.observed"] == 1, obs["demand_forecast.qty"], 0.0)[:, :4].mean(1)
        soon = np.zeros(len(scale))
        live = obs["pending_prohibitions.edge.observed"] == 1
        for e, kk, w in zip(
            obs["pending_prohibitions.edge"][live],
            obs["pending_prohibitions.k"][live],
            obs["pending_prohibitions.effective_week"][live],
        ):
            on = (tb["route_edges"] == e).any(1) & (self.r_k == kk)
            soon = np.where(on, np.maximum(soon, 1.0 / (1.0 + max(int(w) - week, 0))), soon)
        return np.column_stack(
            [
                rate,
                np.log1p(np.minimum(headroom, 147.0)),
                np.log1p(np.maximum(wip, 0) / scale),
                np.where(has, share[d], 0.0),
                np.log1p(np.maximum(np.where(has, fc[d], 0.0), 0) / scale),
                soon,
            ]
        )

    @staticmethod
    def _neighbours(h: np.ndarray, index: np.ndarray) -> np.ndarray:
        """For every route, the mean of ``h`` over the routes sharing its ``index`` (a stock slot)."""
        n = int(index.max()) + 1
        sums = np.zeros((n, h.shape[1]))
        np.add.at(sums, index, h)
        return (sums / np.maximum(np.bincount(index, minlength=n), 1)[:, None])[index]

    def act(self, observation):
        x, cap = self.features(observation)
        if NET["enc0_w"].shape[1] != x.shape[1]:
            x = x[:, list(BASE_COLUMNS)]
        h = np.tanh(x @ NET["enc0_w"].T + NET["enc0_b"])
        h = np.tanh(h @ NET["enc1_w"].T + NET["enc1_b"])
        if "mix_h_w" in NET:
            dest, tail = self._neighbours(h, self.dest), self._neighbours(h, self.tail)
            h = h + np.tanh(h @ NET["mix_h_w"].T + NET["mix_h_b"] + dest @ NET["mix_d_w"].T + tail @ NET["mix_s_w"].T)
        z = (h @ NET["act_w"].T).ravel() + NET["act_b"][0] + self.tb["base"]
        flows = np.where(np.isfinite(cap), cap, 0.0) / (1.0 + np.exp(-z))
        prohibited = observation["graph_now.prohibited"] * observation["graph_now.prohibited.observed"]
        flows = np.where(prohibited[self.r_edge, self.r_k] == 1, 0.0, flows)  # the validator drops these anyway
        return {"flows": flows}
