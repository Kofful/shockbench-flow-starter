Generated for instance `chokepoint-full` (T = 104, regime `standard`) by `uv run sbf fields --task=full`.

**Observation** (`act(observation)`): a dict of numpy arrays. Every key below except `action_mask.observed` and `override_mask.observed` is followed by `<key>.observed`, an int8 array of the same shape, 1 where the value is present and 0 where it is unobserved or padding (the value is then 0).

| key | shape | dtype | indexed by | meaning |
| --- | --- | --- | --- | --- |
| `week` | (1,) | int64 | - | t, the week to decide (1-based); every other field is the state at instant t - 1 |
| `stock.qty` | (125,) | float64 | layout.stock_slots | on-hand stock I^{t-1} per (node, k), chokepoints excluded (see queue_lots) |
| `backlog.qty` | (16,) | float64 | layout.demands | unserved demand carried at each backlog sink (0 at lost-sales sinks) |
| `pipeline.edge` | (883,) | int64 | grouped list | shipments in transit, one entry per (edge, k, lane, arrival week): the edge |
| `pipeline.k` | (883,) | int64 | grouped list | commodity index |
| `pipeline.lane` | (883,) | int64 | grouped list | lane index; unobserved for shipments off any lane |
| `pipeline.qty` | (883,) | float64 | grouped list | the group's total quantity; its observed-mask marks the live entries |
| `pipeline.arrival_week` | (883,) | int64 | grouped list | week the group reaches the edge's head |
| `queue_lots.qty` | (206, 104) | float64 | layout.lot_keys x week | quantity waiting at a chokepoint, one row per (chokepoint, k, lane, next edge) of layout.lot_keys, one column per week the lots reached it (column w - 1 is week w; their FIFO cohort): the total of those lots, observed where lots wait |
| `wip.node` | (136,) | int64 | padded list | work in process at fabs and OSATs (gross, Q69): node |
| `wip.k` | (136,) | int64 | padded list | output commodity |
| `wip.qty` | (136,) | float64 | padded list | quantity |
| `wip.out_week` | (136,) | int64 | padded list | week it becomes stock |
| `graph_now.u` | (369,) | float64 | edges | capacity u per edge this week; unobserved on grid couplings |
| `graph_now.c` | (369,) | float64 | edges | freight cost per unit per edge |
| `graph_now.tau` | (369,) | int64 | edges | lead time in weeks per edge |
| `graph_now.prohibited` | (369, 8) | int8 | edges x commodities | 1 where (edge, k) is prohibited (sanctions, export controls) |
| `graph_now.tariff` | (369, 8) | float64 | edges x commodities | tariff rate on (edge, k) |
| `graph_now.open` | (7,) | float64 | layout.chokepoints | open fraction o_c of each chokepoint (1 open, 0 closed) |
| `graph_now.kappa.tb` | (7,) | float64 | layout.chokepoints | throughput kappa_cb of the tanker/bulk pool, (9) |
| `graph_now.kappa.ct` | (7,) | float64 | layout.chokepoints | throughput kappa_cb of the container pool, (9) |
| `graph_now.war_risk` | (7,) | int64 | layout.chokepoints | war-risk class code: 0 none, 1 red_sea, 2 hormuz_2026 |
| `graph_now.supply.avail` | (18,) | float64 | layout.supply_slots | supply available at each source and material slot |
| `graph_now.fab.R` | (16,) | float64 | layout.fabs | restoration factor R_f of each fab, (13) |
| `graph_now.fab.alpha_bar` | (16,) | float64 | layout.fabs | power multiplier alpha-bar_f of each fab, (13) |
| `graph_now.fab.cap_eff` | (16,) | float64 | layout.fabs | effective wafer capacity of each fab |
| `graph_now.grid.G_bar` | (8,) | float64 | layout.grids | deliverable generation G-bar_g of each grid, (15) |
| `graph_now.grid.y_bar` | (8,) | float64 | layout.grids | base load y-bar_g of each grid, (17) |
| `graph_now.osat.R` | (7,) | float64 | layout.osats | restoration factor R^osat of each OSAT, (13) (Q97) |
| `graph_now.osat.thr_eff` | (7,) | float64 | layout.osats | effective throughput thr R^osat of each OSAT, (19) |
| `slot_mask` | (395,) | int8 | action slots | 1 where an edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity this week (the wire's convention; see action_mask) |
| `last_week.clip.requested` | (395,) | float64 | action slots | flow you requested last week |
| `last_week.clip.executed` | (395,) | float64 | action slots | flow executed after the capacity clip |
| `last_week.cost_components` | (8,) | float64 | layout.cost_components | last week's cost by component, USD |
| `last_week.sinks.demand` | (16,) | float64 | layout.demands | last week's demand |
| `last_week.sinks.served` | (16,) | float64 | layout.demands | last week's demand served |
| `last_week.sinks.lost` | (16,) | float64 | layout.demands | last week's demand lost |
| `last_week.shed.qty` | (8,) | float64 | layout.grids | power shed at each grid last week |
| `demand_forecast.qty` | (16, 8) | float64 | layout.demands x h | demand forecast for weeks t + h, h = 0..7, (48) |
| `warning.score` | (22,) | float64 | layout.warning_units | early-warning score S^t per region, dyad and chokepoint, (45) |
| `messages.msg_id` | (2176,) | int64 | padded list | live announcement threads (announced, not effective, not withdrawn): thread id |
| `messages.channel` | (2176,) | int64 | padded list | channel code: 0 tariff_formal, 1 tariff_informal, 2 tariff_final, 3 sanction_legal, 4 ties_threat, 5 mid_threat |
| `messages.kind` | (2176,) | int64 | padded list | message kind code: 0 proposal, 1 final_notice, 2 threat, 3 publication, 4 withdrawal |
| `messages.region` | (2176,) | int64 | padded list | region index |
| `messages.target_kind` | (2176,) | int64 | padded list | target kind code: 0 chokepoint, 1 edge, 2 node, 3 region |
| `messages.target` | (2176,) | int64 | padded list | target index |
| `messages.k` | (2176,) | int64 | padded list | commodity index; unobserved when the message names none |
| `messages.announced_week` | (2176,) | int64 | padded list | week announced |
| `messages.stated_effective_week` | (2176,) | int64 | padded list | stated effective week; unobserved when none is stated |
| `pending_prohibitions.edge` | (4992,) | int64 | padded list | announced prohibitions not yet in force: edge |
| `pending_prohibitions.k` | (4992,) | int64 | padded list | commodity index |
| `pending_prohibitions.effective_week` | (4992,) | int64 | padded list | week it takes effect |
| `closure_end.chokepoint` | (128,) | int64 | padded list | closures acting now: chokepoint |
| `closure_end.end_week` | (128,) | int64 | padded list | announced end week; unobserved when unknown |
| `action_mask` | (395,) | int8 | action slots | 1 where no edge of the slot's route (the edge, or every edge of its lane) is prohibited for its commodity this week (the inverse of slot_mask); capacities and closures are not checked: read graph_now.u and graph_now.open |
| `action_mask.observed` | (1,) | int8 | - | 1 when this week's mask was observed (0 in a blackout week: all slots allowed) |
| `override_mask` | (54,) | int8 | override slots | 1 where the slot's own out edge is not prohibited for its commodity this week; release_mode 1 sends every override slot of its pair, and a slot at 0 is dropped whatever its override_qty (one invalid entry, no cost): the pair's other slots stand, and with none valid the default release stays on |
| `override_mask.observed` | (1,) | int8 | - | 1 when this week's override mask was observed (0 in a blackout week) |

**Action** (the return value of `act`): a dict of numpy arrays (`override_qty` and `release_mode` may be left out: zeros, the default release).

| key | shape | dtype | indexed by | meaning |
| --- | --- | --- | --- | --- |
| `flows` | (395,) | float64 | action slots | quantity to dispatch on each (edge, commodity, lane) slot; 0 sends nothing |
| `override_qty` | (54,) | float64 | override slots | tanker cargo to release on each override slot, read where release_mode is 1 |
| `release_mode` | (14,) | int64 | layout.release_pairs | per (chokepoint, tanker commodity): 0 default release, 1 override, 2 hold |

**Action slots** (`flows`, `action_mask`, `slot_mask`, `last_week.clip.*`):

| slot | edge | from | to | commodity | lane |
| --- | --- | --- | --- | --- | --- |
| 0 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_tw |
| 1 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_kr |
| 2 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_eu |
| 3 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_in |
| 4 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_eu.cape |
| 5 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_tw.lombok |
| 6 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_kr.lombok |
| 7 | sea.tb.src_qa_lng.chk_hormuz | src_qa_lng | chk_hormuz | lng | lane.src_qa_lng.term_kr.east |
| 8 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_kr |
| 9 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_jp |
| 10 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_cn |
| 11 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_in |
| 12 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_kr.lombok |
| 13 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_jp.lombok |
| 14 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_cn.lombok |
| 15 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_kr.east |
| 16 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_jp.east |
| 17 | sea.tb.src_gulf_crude.chk_hormuz | src_gulf_crude | chk_hormuz | crude | lane.src_gulf_crude.term_cn.east |
| 18 | bypass.tb.src_gulf_crude.chk_malacca | src_gulf_crude | chk_malacca | crude | lane.src_gulf_crude.term_kr.bypass |
| 19 | bypass.tb.src_gulf_crude.chk_malacca | src_gulf_crude | chk_malacca | crude | lane.src_gulf_crude.term_jp.bypass |
| 20 | bypass.tb.src_gulf_crude.chk_malacca | src_gulf_crude | chk_malacca | crude | lane.src_gulf_crude.term_cn.bypass |
| 21 | bypass.tb.src_gulf_crude.term_in | src_gulf_crude | term_in | crude | - |
| 22 | sea.tb.src_us_lng.chk_panama | src_us_lng | chk_panama | lng | lane.src_us_lng.term_tw |
| 23 | sea.tb.src_us_lng.chk_panama | src_us_lng | chk_panama | lng | lane.src_us_lng.term_jp |
| 24 | sea.tb.src_us_lng.term_eu | src_us_lng | term_eu | lng | - |
| 25 | pipe.tb.src_us_lng.grid_us | src_us_lng | grid_us | lng | - |
| 26 | sea.tb.src_us_crude.chk_panama | src_us_crude | chk_panama | crude | lane.src_us_crude.term_tw |
| 27 | sea.tb.src_us_crude.chk_panama | src_us_crude | chk_panama | crude | lane.src_us_crude.term_sea |
| 28 | pipe.tb.src_us_crude.term_us | src_us_crude | term_us | crude | - |
| 29 | sea.tb.src_us_crude.term_eu | src_us_crude | term_eu | crude | - |
| 30 | sea.tb.src_au_lng.term_kr | src_au_lng | term_kr | lng | - |
| 31 | sea.tb.src_au_lng.term_jp | src_au_lng | term_jp | lng | - |
| 32 | sea.tb.src_au_lng.term_cn | src_au_lng | term_cn | lng | - |
| 33 | sea.tb.src_au_lng.term_sea | src_au_lng | term_sea | lng | - |
| 34 | sea.tb.src_ru_gas.term_jp | src_ru_gas | term_jp | lng | - |
| 35 | sea.tb.src_ru_gas.term_cn | src_ru_gas | term_cn | lng | - |
| 36 | pipe.tb.src_ru_gas.grid_cn | src_ru_gas | grid_cn | lng | - |
| 37 | pipe.tb.src_ru_gas.grid_eu | src_ru_gas | grid_eu | lng | - |
| 38 | sea.tb.src_ru_crude.chk_turkish | src_ru_crude | chk_turkish | crude | lane.src_ru_crude.term_eu |
| 39 | sea.tb.src_ru_crude.chk_turkish | src_ru_crude | chk_turkish | crude | lane.src_ru_crude.term_in |
| 40 | sea.tb.src_ru_crude.chk_turkish | src_ru_crude | chk_turkish | crude | lane.src_ru_crude.term_in.cape |
| 41 | sea.tb.src_ru_crude.term_cn | src_ru_crude | term_cn | crude | - |
| 42 | pipe.tb.src_no_gas.grid_eu | src_no_gas | grid_eu | lng | - |
| 43 | pipe.tb.src_kz_uranium.grid_kr | src_kz_uranium | grid_kr | nucfuel | - |
| 44 | pipe.tb.src_kz_uranium.grid_jp | src_kz_uranium | grid_jp | nucfuel | - |
| 45 | pipe.tb.src_kz_uranium.grid_cn | src_kz_uranium | grid_cn | nucfuel | - |
| 46 | pipe.tb.src_kz_uranium.grid_us | src_kz_uranium | grid_us | nucfuel | - |
| 47 | pipe.tb.src_kz_uranium.grid_eu | src_kz_uranium | grid_eu | nucfuel | - |
| 48 | pipe.tb.src_kz_uranium.grid_in | src_kz_uranium | grid_in | nucfuel | - |
| 49 | pipe.tb.src_ru_enrichment.grid_kr | src_ru_enrichment | grid_kr | nucfuel | - |
| 50 | pipe.tb.src_ru_enrichment.grid_jp | src_ru_enrichment | grid_jp | nucfuel | - |
| 51 | pipe.tb.src_ru_enrichment.grid_cn | src_ru_enrichment | grid_cn | nucfuel | - |
| 52 | pipe.tb.src_ru_enrichment.grid_us | src_ru_enrichment | grid_us | nucfuel | - |
| 53 | pipe.tb.src_ru_enrichment.grid_eu | src_ru_enrichment | grid_eu | nucfuel | - |
| 54 | pipe.tb.src_ru_enrichment.grid_in | src_ru_enrichment | grid_in | nucfuel | - |
| 55 | tg.tb.term_tw.grid_tw | term_tw | grid_tw | lng | - |
| 56 | tg.tb.term_tw.grid_tw | term_tw | grid_tw | crude | - |
| 57 | tg.tb.term_kr.grid_kr | term_kr | grid_kr | lng | - |
| 58 | tg.tb.term_kr.grid_kr | term_kr | grid_kr | crude | - |
| 59 | tg.tb.term_jp.grid_jp | term_jp | grid_jp | lng | - |
| 60 | tg.tb.term_jp.grid_jp | term_jp | grid_jp | crude | - |
| 61 | tg.tb.term_cn.grid_cn | term_cn | grid_cn | lng | - |
| 62 | tg.tb.term_cn.grid_cn | term_cn | grid_cn | crude | - |
| 63 | tg.tb.term_us.grid_us | term_us | grid_us | crude | - |
| 64 | tg.tb.term_eu.grid_eu | term_eu | grid_eu | lng | - |
| 65 | tg.tb.term_eu.grid_eu | term_eu | grid_eu | crude | - |
| 66 | tg.tb.term_sea.grid_sea | term_sea | grid_sea | lng | - |
| 67 | tg.tb.term_sea.grid_sea | term_sea | grid_sea | crude | - |
| 68 | tg.tb.term_in.grid_in | term_in | grid_in | lng | - |
| 69 | tg.tb.term_in.grid_in | term_in | grid_in | crude | - |
| 70 | sea.ct.mat_jp_wafer.fab_tw_leading_1 | mat_jp_wafer | fab_tw_leading_1 | wafer | - |
| 71 | air.ct.mat_jp_wafer.fab_tw_leading_1 | mat_jp_wafer | fab_tw_leading_1 | wafer | - |
| 72 | sea.ct.mat_jp_wafer.fab_tw_mature_1 | mat_jp_wafer | fab_tw_mature_1 | wafer | - |
| 73 | air.ct.mat_jp_wafer.fab_tw_mature_1 | mat_jp_wafer | fab_tw_mature_1 | wafer | - |
| 74 | sea.ct.mat_jp_wafer.fab_us_leading_1 | mat_jp_wafer | fab_us_leading_1 | wafer | - |
| 75 | air.ct.mat_jp_wafer.fab_us_leading_1 | mat_jp_wafer | fab_us_leading_1 | wafer | - |
| 76 | sea.ct.mat_jp_wafer.fab_kr_leading_1 | mat_jp_wafer | fab_kr_leading_1 | wafer | - |
| 77 | air.ct.mat_jp_wafer.fab_kr_leading_1 | mat_jp_wafer | fab_kr_leading_1 | wafer | - |
| 78 | sea.ct.mat_jp_wafer.fab_kr_memory_1 | mat_jp_wafer | fab_kr_memory_1 | wafer | - |
| 79 | air.ct.mat_jp_wafer.fab_kr_memory_1 | mat_jp_wafer | fab_kr_memory_1 | wafer | - |
| 80 | sea.ct.mat_jp_wafer.fab_jp_memory_1 | mat_jp_wafer | fab_jp_memory_1 | wafer | - |
| 81 | air.ct.mat_jp_wafer.fab_jp_memory_1 | mat_jp_wafer | fab_jp_memory_1 | wafer | - |
| 82 | sea.ct.mat_jp_wafer.fab_us_leading_3 | mat_jp_wafer | fab_us_leading_3 | wafer | - |
| 83 | air.ct.mat_jp_wafer.fab_us_leading_3 | mat_jp_wafer | fab_us_leading_3 | wafer | - |
| 84 | sea.ct.mat_jp_wafer.fab_cn_mature_1 | mat_jp_wafer | fab_cn_mature_1 | wafer | - |
| 85 | air.ct.mat_jp_wafer.fab_cn_mature_1 | mat_jp_wafer | fab_cn_mature_1 | wafer | - |
| 86 | sea.ct.mat_jp_wafer.fab_tw_mature_2 | mat_jp_wafer | fab_tw_mature_2 | wafer | - |
| 87 | air.ct.mat_jp_wafer.fab_tw_mature_2 | mat_jp_wafer | fab_tw_mature_2 | wafer | - |
| 88 | east.ct.mat_jp_resist.chk_malacca | mat_jp_resist | chk_malacca | wafer | lane.mat_jp_resist.fab_eu_leading_1.east |
| 89 | east.ct.mat_jp_resist.chk_malacca | mat_jp_resist | chk_malacca | wafer | lane.mat_jp_resist.fab_row_leading_1.east |
| 90 | sea.ct.mat_jp_resist.chk_taiwan | mat_jp_resist | chk_taiwan | wafer | lane.mat_jp_resist.fab_eu_leading_1 |
| 91 | sea.ct.mat_jp_resist.chk_taiwan | mat_jp_resist | chk_taiwan | wafer | lane.mat_jp_resist.fab_row_leading_1 |
| 92 | sea.ct.mat_jp_resist.chk_taiwan | mat_jp_resist | chk_taiwan | wafer | lane.mat_jp_resist.fab_eu_leading_1.cape |
| 93 | sea.ct.mat_jp_resist.chk_taiwan | mat_jp_resist | chk_taiwan | wafer | lane.mat_jp_resist.fab_row_leading_1.cape |
| 94 | sea.ct.mat_jp_resist.chk_taiwan | mat_jp_resist | chk_taiwan | wafer | lane.mat_jp_resist.fab_eu_leading_1.lombok |
| 95 | sea.ct.mat_jp_resist.chk_taiwan | mat_jp_resist | chk_taiwan | wafer | lane.mat_jp_resist.fab_row_leading_1.lombok |
| 96 | sea.ct.mat_jp_resist.fab_tw_leading_1 | mat_jp_resist | fab_tw_leading_1 | wafer | - |
| 97 | air.ct.mat_jp_resist.fab_tw_leading_1 | mat_jp_resist | fab_tw_leading_1 | wafer | - |
| 98 | sea.ct.mat_jp_resist.fab_us_leading_1 | mat_jp_resist | fab_us_leading_1 | wafer | - |
| 99 | air.ct.mat_jp_resist.fab_us_leading_1 | mat_jp_resist | fab_us_leading_1 | wafer | - |
| 100 | sea.ct.mat_jp_resist.fab_kr_leading_1 | mat_jp_resist | fab_kr_leading_1 | wafer | - |
| 101 | air.ct.mat_jp_resist.fab_kr_leading_1 | mat_jp_resist | fab_kr_leading_1 | wafer | - |
| 102 | sea.ct.mat_jp_resist.fab_kr_memory_1 | mat_jp_resist | fab_kr_memory_1 | wafer | - |
| 103 | air.ct.mat_jp_resist.fab_kr_memory_1 | mat_jp_resist | fab_kr_memory_1 | wafer | - |
| 104 | sea.ct.mat_jp_resist.fab_us_leading_2 | mat_jp_resist | fab_us_leading_2 | wafer | - |
| 105 | air.ct.mat_jp_resist.fab_us_leading_2 | mat_jp_resist | fab_us_leading_2 | wafer | - |
| 106 | sea.ct.mat_jp_resist.fab_jp_memory_1 | mat_jp_resist | fab_jp_memory_1 | wafer | - |
| 107 | air.ct.mat_jp_resist.fab_jp_memory_1 | mat_jp_resist | fab_jp_memory_1 | wafer | - |
| 108 | sea.ct.mat_jp_resist.fab_us_leading_3 | mat_jp_resist | fab_us_leading_3 | wafer | - |
| 109 | air.ct.mat_jp_resist.fab_us_leading_3 | mat_jp_resist | fab_us_leading_3 | wafer | - |
| 110 | air.ct.mat_jp_resist.fab_eu_leading_1 | mat_jp_resist | fab_eu_leading_1 | wafer | - |
| 111 | air.ct.mat_jp_resist.fab_row_leading_1 | mat_jp_resist | fab_row_leading_1 | wafer | - |
| 112 | sea.ct.mat_kr_wafer.fab_kr_leading_1 | mat_kr_wafer | fab_kr_leading_1 | wafer | - |
| 113 | air.ct.mat_kr_wafer.fab_kr_leading_1 | mat_kr_wafer | fab_kr_leading_1 | wafer | - |
| 114 | sea.ct.mat_kr_wafer.fab_kr_memory_1 | mat_kr_wafer | fab_kr_memory_1 | wafer | - |
| 115 | air.ct.mat_kr_wafer.fab_kr_memory_1 | mat_kr_wafer | fab_kr_memory_1 | wafer | - |
| 116 | sea.ct.mat_kr_wafer.fab_us_leading_2 | mat_kr_wafer | fab_us_leading_2 | wafer | - |
| 117 | air.ct.mat_kr_wafer.fab_us_leading_2 | mat_kr_wafer | fab_us_leading_2 | wafer | - |
| 118 | sea.ct.mat_kr_wafer.fab_cn_mature_1 | mat_kr_wafer | fab_cn_mature_1 | wafer | - |
| 119 | air.ct.mat_kr_wafer.fab_cn_mature_1 | mat_kr_wafer | fab_cn_mature_1 | wafer | - |
| 120 | sea.ct.mat_de_wafer.chk_suez | mat_de_wafer | chk_suez | wafer | lane.mat_de_wafer.fab_sea_mature_1 |
| 121 | sea.ct.mat_de_wafer.chk_suez | mat_de_wafer | chk_suez | wafer | lane.mat_de_wafer.fab_tw_mature_2 |
| 122 | sea.ct.mat_de_wafer.chk_suez | mat_de_wafer | chk_suez | wafer | lane.mat_de_wafer.fab_sea_mature_1.lombok |
| 123 | sea.ct.mat_de_wafer.chk_suez | mat_de_wafer | chk_suez | wafer | lane.mat_de_wafer.fab_tw_mature_2.lombok |
| 124 | cape.ct.mat_de_wafer.chk_cape | mat_de_wafer | chk_cape | wafer | lane.mat_de_wafer.fab_sea_mature_1.cape |
| 125 | cape.ct.mat_de_wafer.chk_cape | mat_de_wafer | chk_cape | wafer | lane.mat_de_wafer.fab_tw_mature_2.cape |
| 126 | sea.ct.mat_de_wafer.fab_eu_leading_1 | mat_de_wafer | fab_eu_leading_1 | wafer | - |
| 127 | air.ct.mat_de_wafer.fab_eu_leading_1 | mat_de_wafer | fab_eu_leading_1 | wafer | - |
| 128 | sea.ct.mat_de_wafer.fab_row_leading_1 | mat_de_wafer | fab_row_leading_1 | wafer | - |
| 129 | air.ct.mat_de_wafer.fab_row_leading_1 | mat_de_wafer | fab_row_leading_1 | wafer | - |
| 130 | sea.ct.mat_de_wafer.fab_us_mature_1 | mat_de_wafer | fab_us_mature_1 | wafer | - |
| 131 | air.ct.mat_de_wafer.fab_us_mature_1 | mat_de_wafer | fab_us_mature_1 | wafer | - |
| 132 | sea.ct.mat_de_wafer.fab_eu_mature_1 | mat_de_wafer | fab_eu_mature_1 | wafer | - |
| 133 | air.ct.mat_de_wafer.fab_eu_mature_1 | mat_de_wafer | fab_eu_mature_1 | wafer | - |
| 134 | air.ct.mat_de_wafer.fab_sea_mature_1 | mat_de_wafer | fab_sea_mature_1 | wafer | - |
| 135 | air.ct.mat_de_wafer.fab_tw_mature_2 | mat_de_wafer | fab_tw_mature_2 | wafer | - |
| 136 | sea.ct.mat_de_wafer.fab_us_mature_2 | mat_de_wafer | fab_us_mature_2 | wafer | - |
| 137 | air.ct.mat_de_wafer.fab_us_mature_2 | mat_de_wafer | fab_us_mature_2 | wafer | - |
| 138 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_tw_leading_1 |
| 139 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1 |
| 140 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_eu_leading_1 |
| 141 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_eu_mature_1 |
| 142 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_us_mature_2 |
| 143 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_tw_leading_1.cape |
| 144 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1.cape |
| 145 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_tw_leading_1.lombok |
| 146 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1.lombok |
| 147 | sea.ct.mat_ua_neon.chk_turkish | mat_ua_neon | chk_turkish | wafer | lane.mat_ua_neon.fab_kr_memory_1.east |
| 148 | air.ct.mat_ua_neon.fab_tw_leading_1 | mat_ua_neon | fab_tw_leading_1 | wafer | - |
| 149 | air.ct.mat_ua_neon.fab_kr_memory_1 | mat_ua_neon | fab_kr_memory_1 | wafer | - |
| 150 | air.ct.mat_ua_neon.fab_eu_leading_1 | mat_ua_neon | fab_eu_leading_1 | wafer | - |
| 151 | air.ct.mat_ua_neon.fab_eu_mature_1 | mat_ua_neon | fab_eu_mature_1 | wafer | - |
| 152 | air.ct.mat_ua_neon.fab_us_mature_2 | mat_ua_neon | fab_us_mature_2 | wafer | - |
| 153 | sea.ct.mat_cn_neon.chk_taiwan | mat_cn_neon | chk_taiwan | wafer | lane.mat_cn_neon.fab_sea_mature_1 |
| 154 | sea.ct.mat_cn_neon.fab_tw_mature_1 | mat_cn_neon | fab_tw_mature_1 | wafer | - |
| 155 | air.ct.mat_cn_neon.fab_tw_mature_1 | mat_cn_neon | fab_tw_mature_1 | wafer | - |
| 156 | sea.ct.mat_cn_neon.fab_kr_leading_1 | mat_cn_neon | fab_kr_leading_1 | wafer | - |
| 157 | air.ct.mat_cn_neon.fab_kr_leading_1 | mat_cn_neon | fab_kr_leading_1 | wafer | - |
| 158 | sea.ct.mat_cn_neon.fab_cn_mature_1 | mat_cn_neon | fab_cn_mature_1 | wafer | - |
| 159 | air.ct.mat_cn_neon.fab_cn_mature_1 | mat_cn_neon | fab_cn_mature_1 | wafer | - |
| 160 | east.ct.mat_cn_neon.fab_sea_mature_1 | mat_cn_neon | fab_sea_mature_1 | wafer | - |
| 161 | air.ct.mat_cn_neon.fab_sea_mature_1 | mat_cn_neon | fab_sea_mature_1 | wafer | - |
| 162 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_tw_leading_1 |
| 163 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_kr_leading_1 |
| 164 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_kr_memory_1 |
| 165 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_eu_leading_1 |
| 166 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_sea_mature_1 |
| 167 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_eu_leading_1.cape |
| 168 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_tw_leading_1.lombok |
| 169 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_kr_leading_1.lombok |
| 170 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_kr_memory_1.lombok |
| 171 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_sea_mature_1.lombok |
| 172 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_kr_leading_1.east |
| 173 | sea.ct.mat_helium.chk_hormuz | mat_helium | chk_hormuz | wafer | lane.mat_helium.fab_kr_memory_1.east |
| 174 | air.ct.mat_helium.fab_tw_leading_1 | mat_helium | fab_tw_leading_1 | wafer | - |
| 175 | air.ct.mat_helium.fab_kr_leading_1 | mat_helium | fab_kr_leading_1 | wafer | - |
| 176 | air.ct.mat_helium.fab_kr_memory_1 | mat_helium | fab_kr_memory_1 | wafer | - |
| 177 | air.ct.mat_helium.fab_eu_leading_1 | mat_helium | fab_eu_leading_1 | wafer | - |
| 178 | air.ct.mat_helium.fab_sea_mature_1 | mat_helium | fab_sea_mature_1 | wafer | - |
| 179 | sea.ct.mat_cn_gage.fab_tw_mature_1 | mat_cn_gage | fab_tw_mature_1 | wafer | - |
| 180 | air.ct.mat_cn_gage.fab_tw_mature_1 | mat_cn_gage | fab_tw_mature_1 | wafer | - |
| 181 | sea.ct.mat_cn_gage.fab_jp_memory_1 | mat_cn_gage | fab_jp_memory_1 | wafer | - |
| 182 | air.ct.mat_cn_gage.fab_jp_memory_1 | mat_cn_gage | fab_jp_memory_1 | wafer | - |
| 183 | sea.ct.mat_cn_gage.fab_cn_mature_1 | mat_cn_gage | fab_cn_mature_1 | wafer | - |
| 184 | air.ct.mat_cn_gage.fab_cn_mature_1 | mat_cn_gage | fab_cn_mature_1 | wafer | - |
| 185 | sea.ct.mat_cn_gage.fab_tw_mature_2 | mat_cn_gage | fab_tw_mature_2 | wafer | - |
| 186 | air.ct.mat_cn_gage.fab_tw_mature_2 | mat_cn_gage | fab_tw_mature_2 | wafer | - |
| 187 | sea.ct.mat_cn_gage.fab_us_mature_2 | mat_cn_gage | fab_us_mature_2 | wafer | - |
| 188 | air.ct.mat_cn_gage.fab_us_mature_2 | mat_cn_gage | fab_us_mature_2 | wafer | - |
| 189 | sea.ct.fab_tw_leading_1.osat_my | fab_tw_leading_1 | osat_my | chip_le_raw | - |
| 190 | air.ct.fab_tw_leading_1.osat_my | fab_tw_leading_1 | osat_my | chip_le_raw | - |
| 191 | sea.ct.fab_tw_leading_1.osat_tw | fab_tw_leading_1 | osat_tw | chip_le_raw | - |
| 192 | air.ct.fab_tw_leading_1.osat_tw | fab_tw_leading_1 | osat_tw | chip_le_raw | - |
| 193 | sea.ct.fab_tw_leading_1.osat_sg | fab_tw_leading_1 | osat_sg | chip_le_raw | - |
| 194 | air.ct.fab_tw_leading_1.osat_sg | fab_tw_leading_1 | osat_sg | chip_le_raw | - |
| 195 | sea.ct.fab_tw_mature_1.osat_ph | fab_tw_mature_1 | osat_ph | chip_mat_raw | - |
| 196 | air.ct.fab_tw_mature_1.osat_ph | fab_tw_mature_1 | osat_ph | chip_mat_raw | - |
| 197 | sea.ct.fab_tw_mature_1.osat_cn | fab_tw_mature_1 | osat_cn | chip_mat_raw | - |
| 198 | air.ct.fab_tw_mature_1.osat_cn | fab_tw_mature_1 | osat_cn | chip_mat_raw | - |
| 199 | sea.ct.fab_tw_mature_1.osat_tw | fab_tw_mature_1 | osat_tw | chip_mat_raw | - |
| 200 | air.ct.fab_tw_mature_1.osat_tw | fab_tw_mature_1 | osat_tw | chip_mat_raw | - |
| 201 | sea.ct.fab_us_leading_1.osat_my | fab_us_leading_1 | osat_my | chip_le_raw | - |
| 202 | air.ct.fab_us_leading_1.osat_my | fab_us_leading_1 | osat_my | chip_le_raw | - |
| 203 | sea.ct.fab_us_leading_1.osat_tw | fab_us_leading_1 | osat_tw | chip_le_raw | - |
| 204 | air.ct.fab_us_leading_1.osat_tw | fab_us_leading_1 | osat_tw | chip_le_raw | - |
| 205 | sea.ct.fab_kr_leading_1.chk_taiwan | fab_kr_leading_1 | chk_taiwan | chip_le_raw | lane.fab_kr_leading_1.osat_vn |
| 206 | east.ct.fab_kr_leading_1.osat_vn | fab_kr_leading_1 | osat_vn | chip_le_raw | - |
| 207 | air.ct.fab_kr_leading_1.osat_vn | fab_kr_leading_1 | osat_vn | chip_le_raw | - |
| 208 | sea.ct.fab_kr_leading_1.osat_kr | fab_kr_leading_1 | osat_kr | chip_le_raw | - |
| 209 | air.ct.fab_kr_leading_1.osat_kr | fab_kr_leading_1 | osat_kr | chip_le_raw | - |
| 210 | sea.ct.fab_kr_memory_1.chk_taiwan | fab_kr_memory_1 | chk_taiwan | chip_le_raw | lane.fab_kr_memory_1.osat_my |
| 211 | sea.ct.fab_kr_memory_1.chk_taiwan | fab_kr_memory_1 | chk_taiwan | chip_le_raw | lane.fab_kr_memory_1.osat_vn |
| 212 | east.ct.fab_kr_memory_1.osat_my | fab_kr_memory_1 | osat_my | chip_le_raw | - |
| 213 | air.ct.fab_kr_memory_1.osat_my | fab_kr_memory_1 | osat_my | chip_le_raw | - |
| 214 | east.ct.fab_kr_memory_1.osat_vn | fab_kr_memory_1 | osat_vn | chip_le_raw | - |
| 215 | air.ct.fab_kr_memory_1.osat_vn | fab_kr_memory_1 | osat_vn | chip_le_raw | - |
| 216 | sea.ct.fab_kr_memory_1.osat_kr | fab_kr_memory_1 | osat_kr | chip_le_raw | - |
| 217 | air.ct.fab_kr_memory_1.osat_kr | fab_kr_memory_1 | osat_kr | chip_le_raw | - |
| 218 | sea.ct.fab_us_leading_2.osat_vn | fab_us_leading_2 | osat_vn | chip_le_raw | - |
| 219 | air.ct.fab_us_leading_2.osat_vn | fab_us_leading_2 | osat_vn | chip_le_raw | - |
| 220 | sea.ct.fab_us_leading_2.osat_kr | fab_us_leading_2 | osat_kr | chip_le_raw | - |
| 221 | air.ct.fab_us_leading_2.osat_kr | fab_us_leading_2 | osat_kr | chip_le_raw | - |
| 222 | sea.ct.fab_jp_memory_1.chk_taiwan | fab_jp_memory_1 | chk_taiwan | chip_le_raw | lane.fab_jp_memory_1.osat_my |
| 223 | sea.ct.fab_jp_memory_1.chk_taiwan | fab_jp_memory_1 | chk_taiwan | chip_le_raw | lane.fab_jp_memory_1.osat_ph |
| 224 | east.ct.fab_jp_memory_1.osat_my | fab_jp_memory_1 | osat_my | chip_le_raw | - |
| 225 | air.ct.fab_jp_memory_1.osat_my | fab_jp_memory_1 | osat_my | chip_le_raw | - |
| 226 | east.ct.fab_jp_memory_1.osat_ph | fab_jp_memory_1 | osat_ph | chip_le_raw | - |
| 227 | air.ct.fab_jp_memory_1.osat_ph | fab_jp_memory_1 | osat_ph | chip_le_raw | - |
| 228 | sea.ct.fab_us_leading_3.osat_my | fab_us_leading_3 | osat_my | chip_le_raw | - |
| 229 | air.ct.fab_us_leading_3.osat_my | fab_us_leading_3 | osat_my | chip_le_raw | - |
| 230 | sea.ct.fab_us_leading_3.osat_vn | fab_us_leading_3 | osat_vn | chip_le_raw | - |
| 231 | air.ct.fab_us_leading_3.osat_vn | fab_us_leading_3 | osat_vn | chip_le_raw | - |
| 232 | sea.ct.fab_eu_leading_1.chk_suez | fab_eu_leading_1 | chk_suez | chip_le_raw | lane.fab_eu_leading_1.osat_my |
| 233 | sea.ct.fab_eu_leading_1.chk_suez | fab_eu_leading_1 | chk_suez | chip_le_raw | lane.fab_eu_leading_1.osat_vn |
| 234 | sea.ct.fab_eu_leading_1.chk_suez | fab_eu_leading_1 | chk_suez | chip_le_raw | lane.fab_eu_leading_1.osat_my.lombok |
| 235 | sea.ct.fab_eu_leading_1.chk_suez | fab_eu_leading_1 | chk_suez | chip_le_raw | lane.fab_eu_leading_1.osat_vn.lombok |
| 236 | cape.ct.fab_eu_leading_1.chk_cape | fab_eu_leading_1 | chk_cape | chip_le_raw | lane.fab_eu_leading_1.osat_my.cape |
| 237 | cape.ct.fab_eu_leading_1.chk_cape | fab_eu_leading_1 | chk_cape | chip_le_raw | lane.fab_eu_leading_1.osat_vn.cape |
| 238 | air.ct.fab_eu_leading_1.osat_my | fab_eu_leading_1 | osat_my | chip_le_raw | - |
| 239 | air.ct.fab_eu_leading_1.osat_vn | fab_eu_leading_1 | osat_vn | chip_le_raw | - |
| 240 | sea.ct.fab_row_leading_1.chk_suez | fab_row_leading_1 | chk_suez | chip_le_raw | lane.fab_row_leading_1.osat_my |
| 241 | sea.ct.fab_row_leading_1.chk_suez | fab_row_leading_1 | chk_suez | chip_le_raw | lane.fab_row_leading_1.osat_vn |
| 242 | sea.ct.fab_row_leading_1.chk_suez | fab_row_leading_1 | chk_suez | chip_le_raw | lane.fab_row_leading_1.osat_my.lombok |
| 243 | sea.ct.fab_row_leading_1.chk_suez | fab_row_leading_1 | chk_suez | chip_le_raw | lane.fab_row_leading_1.osat_vn.lombok |
| 244 | cape.ct.fab_row_leading_1.chk_cape | fab_row_leading_1 | chk_cape | chip_le_raw | lane.fab_row_leading_1.osat_my.cape |
| 245 | cape.ct.fab_row_leading_1.chk_cape | fab_row_leading_1 | chk_cape | chip_le_raw | lane.fab_row_leading_1.osat_vn.cape |
| 246 | air.ct.fab_row_leading_1.osat_my | fab_row_leading_1 | osat_my | chip_le_raw | - |
| 247 | air.ct.fab_row_leading_1.osat_vn | fab_row_leading_1 | osat_vn | chip_le_raw | - |
| 248 | sea.ct.fab_cn_mature_1.chk_taiwan | fab_cn_mature_1 | chk_taiwan | chip_mat_raw | lane.fab_cn_mature_1.osat_my |
| 249 | east.ct.fab_cn_mature_1.osat_my | fab_cn_mature_1 | osat_my | chip_mat_raw | - |
| 250 | air.ct.fab_cn_mature_1.osat_my | fab_cn_mature_1 | osat_my | chip_mat_raw | - |
| 251 | sea.ct.fab_cn_mature_1.osat_cn | fab_cn_mature_1 | osat_cn | chip_mat_raw | - |
| 252 | air.ct.fab_cn_mature_1.osat_cn | fab_cn_mature_1 | osat_cn | chip_mat_raw | - |
| 253 | sea.ct.fab_us_mature_1.osat_ph | fab_us_mature_1 | osat_ph | chip_mat_raw | - |
| 254 | air.ct.fab_us_mature_1.osat_ph | fab_us_mature_1 | osat_ph | chip_mat_raw | - |
| 255 | sea.ct.fab_us_mature_1.osat_sg | fab_us_mature_1 | osat_sg | chip_mat_raw | - |
| 256 | air.ct.fab_us_mature_1.osat_sg | fab_us_mature_1 | osat_sg | chip_mat_raw | - |
| 257 | sea.ct.fab_eu_mature_1.chk_suez | fab_eu_mature_1 | chk_suez | chip_mat_raw | lane.fab_eu_mature_1.osat_my |
| 258 | sea.ct.fab_eu_mature_1.chk_suez | fab_eu_mature_1 | chk_suez | chip_mat_raw | lane.fab_eu_mature_1.osat_sg |
| 259 | sea.ct.fab_eu_mature_1.chk_suez | fab_eu_mature_1 | chk_suez | chip_mat_raw | lane.fab_eu_mature_1.osat_my.lombok |
| 260 | sea.ct.fab_eu_mature_1.chk_suez | fab_eu_mature_1 | chk_suez | chip_mat_raw | lane.fab_eu_mature_1.osat_sg.lombok |
| 261 | cape.ct.fab_eu_mature_1.chk_cape | fab_eu_mature_1 | chk_cape | chip_mat_raw | lane.fab_eu_mature_1.osat_my.cape |
| 262 | cape.ct.fab_eu_mature_1.chk_cape | fab_eu_mature_1 | chk_cape | chip_mat_raw | lane.fab_eu_mature_1.osat_sg.cape |
| 263 | air.ct.fab_eu_mature_1.osat_my | fab_eu_mature_1 | osat_my | chip_mat_raw | - |
| 264 | air.ct.fab_eu_mature_1.osat_sg | fab_eu_mature_1 | osat_sg | chip_mat_raw | - |
| 265 | sea.ct.fab_sea_mature_1.osat_my | fab_sea_mature_1 | osat_my | chip_mat_raw | - |
| 266 | air.ct.fab_sea_mature_1.osat_my | fab_sea_mature_1 | osat_my | chip_mat_raw | - |
| 267 | sea.ct.fab_sea_mature_1.osat_sg | fab_sea_mature_1 | osat_sg | chip_mat_raw | - |
| 268 | air.ct.fab_sea_mature_1.osat_sg | fab_sea_mature_1 | osat_sg | chip_mat_raw | - |
| 269 | sea.ct.fab_tw_mature_2.osat_cn | fab_tw_mature_2 | osat_cn | chip_mat_raw | - |
| 270 | air.ct.fab_tw_mature_2.osat_cn | fab_tw_mature_2 | osat_cn | chip_mat_raw | - |
| 271 | sea.ct.fab_tw_mature_2.osat_tw | fab_tw_mature_2 | osat_tw | chip_mat_raw | - |
| 272 | air.ct.fab_tw_mature_2.osat_tw | fab_tw_mature_2 | osat_tw | chip_mat_raw | - |
| 273 | sea.ct.fab_us_mature_2.osat_my | fab_us_mature_2 | osat_my | chip_mat_raw | - |
| 274 | air.ct.fab_us_mature_2.osat_my | fab_us_mature_2 | osat_my | chip_mat_raw | - |
| 275 | sea.ct.fab_us_mature_2.osat_ph | fab_us_mature_2 | osat_ph | chip_mat_raw | - |
| 276 | air.ct.fab_us_mature_2.osat_ph | fab_us_mature_2 | osat_ph | chip_mat_raw | - |
| 277 | sea.ct.fab_us_mature_2.osat_cn | fab_us_mature_2 | osat_cn | chip_mat_raw | - |
| 278 | air.ct.fab_us_mature_2.osat_cn | fab_us_mature_2 | osat_cn | chip_mat_raw | - |
| 279 | sea.ct.osat_my.chk_malacca | osat_my | chk_malacca | chip_mat | lane.osat_my.sink_eu |
| 280 | sea.ct.osat_my.chk_malacca | osat_my | chk_malacca | chip_mat | lane.osat_my.sink_in |
| 281 | sea.ct.osat_my.chk_malacca | osat_my | chk_malacca | chip_mat | lane.osat_my.sink_row |
| 282 | sea.ct.osat_my.chk_taiwan | osat_my | chk_taiwan | chip_mat | lane.osat_my.sink_cn |
| 283 | sea.ct.osat_my.chk_taiwan | osat_my | chk_taiwan | chip_mat | lane.osat_my.sink_jp |
| 284 | sea.ct.osat_my.chk_taiwan | osat_my | chk_taiwan | chip_mat | lane.osat_my.sink_kr |
| 285 | sea.ct.osat_my.sink_us | osat_my | sink_us | chip_mat | - |
| 286 | air.ct.osat_my.sink_us | osat_my | sink_us | chip_le | - |
| 287 | air.ct.osat_my.sink_us | osat_my | sink_us | chip_mat | - |
| 288 | air.ct.osat_my.sink_eu | osat_my | sink_eu | chip_le | - |
| 289 | air.ct.osat_my.sink_eu | osat_my | sink_eu | chip_mat | - |
| 290 | air.ct.osat_my.sink_cn | osat_my | sink_cn | chip_le | - |
| 291 | air.ct.osat_my.sink_jp | osat_my | sink_jp | chip_le | - |
| 292 | air.ct.osat_my.sink_jp | osat_my | sink_jp | chip_mat | - |
| 293 | air.ct.osat_my.sink_kr | osat_my | sink_kr | chip_le | - |
| 294 | air.ct.osat_my.sink_kr | osat_my | sink_kr | chip_mat | - |
| 295 | sea.ct.osat_my.sink_sea | osat_my | sink_sea | chip_mat | - |
| 296 | air.ct.osat_my.sink_sea | osat_my | sink_sea | chip_le | - |
| 297 | air.ct.osat_my.sink_sea | osat_my | sink_sea | chip_mat | - |
| 298 | air.ct.osat_my.sink_in | osat_my | sink_in | chip_le | - |
| 299 | air.ct.osat_my.sink_in | osat_my | sink_in | chip_mat | - |
| 300 | air.ct.osat_my.sink_row | osat_my | sink_row | chip_le | - |
| 301 | air.ct.osat_my.sink_row | osat_my | sink_row | chip_mat | - |
| 302 | air.ct.osat_vn.sink_us | osat_vn | sink_us | chip_le | - |
| 303 | air.ct.osat_vn.sink_eu | osat_vn | sink_eu | chip_le | - |
| 304 | air.ct.osat_vn.sink_cn | osat_vn | sink_cn | chip_le | - |
| 305 | air.ct.osat_vn.sink_jp | osat_vn | sink_jp | chip_le | - |
| 306 | air.ct.osat_vn.sink_kr | osat_vn | sink_kr | chip_le | - |
| 307 | air.ct.osat_vn.sink_sea | osat_vn | sink_sea | chip_le | - |
| 308 | air.ct.osat_vn.sink_in | osat_vn | sink_in | chip_le | - |
| 309 | air.ct.osat_vn.sink_row | osat_vn | sink_row | chip_le | - |
| 310 | sea.ct.osat_ph.chk_malacca | osat_ph | chk_malacca | chip_mat | lane.osat_ph.sink_eu |
| 311 | sea.ct.osat_ph.chk_malacca | osat_ph | chk_malacca | chip_mat | lane.osat_ph.sink_in |
| 312 | sea.ct.osat_ph.chk_malacca | osat_ph | chk_malacca | chip_mat | lane.osat_ph.sink_row |
| 313 | sea.ct.osat_ph.chk_taiwan | osat_ph | chk_taiwan | chip_mat | lane.osat_ph.sink_cn |
| 314 | sea.ct.osat_ph.chk_taiwan | osat_ph | chk_taiwan | chip_mat | lane.osat_ph.sink_jp |
| 315 | sea.ct.osat_ph.chk_taiwan | osat_ph | chk_taiwan | chip_mat | lane.osat_ph.sink_kr |
| 316 | sea.ct.osat_ph.sink_us | osat_ph | sink_us | chip_mat | - |
| 317 | air.ct.osat_ph.sink_us | osat_ph | sink_us | chip_le | - |
| 318 | air.ct.osat_ph.sink_us | osat_ph | sink_us | chip_mat | - |
| 319 | air.ct.osat_ph.sink_eu | osat_ph | sink_eu | chip_le | - |
| 320 | air.ct.osat_ph.sink_eu | osat_ph | sink_eu | chip_mat | - |
| 321 | air.ct.osat_ph.sink_cn | osat_ph | sink_cn | chip_le | - |
| 322 | air.ct.osat_ph.sink_jp | osat_ph | sink_jp | chip_le | - |
| 323 | air.ct.osat_ph.sink_jp | osat_ph | sink_jp | chip_mat | - |
| 324 | air.ct.osat_ph.sink_kr | osat_ph | sink_kr | chip_le | - |
| 325 | air.ct.osat_ph.sink_kr | osat_ph | sink_kr | chip_mat | - |
| 326 | sea.ct.osat_ph.sink_sea | osat_ph | sink_sea | chip_mat | - |
| 327 | air.ct.osat_ph.sink_sea | osat_ph | sink_sea | chip_le | - |
| 328 | air.ct.osat_ph.sink_sea | osat_ph | sink_sea | chip_mat | - |
| 329 | air.ct.osat_ph.sink_in | osat_ph | sink_in | chip_le | - |
| 330 | air.ct.osat_ph.sink_in | osat_ph | sink_in | chip_mat | - |
| 331 | air.ct.osat_ph.sink_row | osat_ph | sink_row | chip_le | - |
| 332 | air.ct.osat_ph.sink_row | osat_ph | sink_row | chip_mat | - |
| 333 | sea.ct.osat_cn.chk_taiwan | osat_cn | chk_taiwan | chip_mat | lane.osat_cn.sink_eu |
| 334 | sea.ct.osat_cn.chk_taiwan | osat_cn | chk_taiwan | chip_mat | lane.osat_cn.sink_sea |
| 335 | sea.ct.osat_cn.chk_taiwan | osat_cn | chk_taiwan | chip_mat | lane.osat_cn.sink_in |
| 336 | sea.ct.osat_cn.chk_taiwan | osat_cn | chk_taiwan | chip_mat | lane.osat_cn.sink_row |
| 337 | sea.ct.osat_cn.sink_us | osat_cn | sink_us | chip_mat | - |
| 338 | sea.ct.osat_cn.sink_cn | osat_cn | sink_cn | chip_mat | - |
| 339 | sea.ct.osat_cn.sink_jp | osat_cn | sink_jp | chip_mat | - |
| 340 | sea.ct.osat_cn.sink_kr | osat_cn | sink_kr | chip_mat | - |
| 341 | sea.ct.osat_tw.chk_malacca | osat_tw | chk_malacca | chip_mat | lane.osat_tw.sink_eu |
| 342 | sea.ct.osat_tw.chk_malacca | osat_tw | chk_malacca | chip_mat | lane.osat_tw.sink_in |
| 343 | sea.ct.osat_tw.chk_malacca | osat_tw | chk_malacca | chip_mat | lane.osat_tw.sink_row |
| 344 | sea.ct.osat_tw.sink_us | osat_tw | sink_us | chip_mat | - |
| 345 | air.ct.osat_tw.sink_us | osat_tw | sink_us | chip_le | - |
| 346 | air.ct.osat_tw.sink_us | osat_tw | sink_us | chip_mat | - |
| 347 | air.ct.osat_tw.sink_eu | osat_tw | sink_eu | chip_le | - |
| 348 | air.ct.osat_tw.sink_eu | osat_tw | sink_eu | chip_mat | - |
| 349 | sea.ct.osat_tw.sink_cn | osat_tw | sink_cn | chip_mat | - |
| 350 | air.ct.osat_tw.sink_cn | osat_tw | sink_cn | chip_le | - |
| 351 | sea.ct.osat_tw.sink_jp | osat_tw | sink_jp | chip_mat | - |
| 352 | air.ct.osat_tw.sink_jp | osat_tw | sink_jp | chip_le | - |
| 353 | air.ct.osat_tw.sink_jp | osat_tw | sink_jp | chip_mat | - |
| 354 | sea.ct.osat_tw.sink_kr | osat_tw | sink_kr | chip_mat | - |
| 355 | air.ct.osat_tw.sink_kr | osat_tw | sink_kr | chip_le | - |
| 356 | air.ct.osat_tw.sink_kr | osat_tw | sink_kr | chip_mat | - |
| 357 | sea.ct.osat_tw.sink_sea | osat_tw | sink_sea | chip_mat | - |
| 358 | air.ct.osat_tw.sink_sea | osat_tw | sink_sea | chip_le | - |
| 359 | air.ct.osat_tw.sink_sea | osat_tw | sink_sea | chip_mat | - |
| 360 | air.ct.osat_tw.sink_in | osat_tw | sink_in | chip_le | - |
| 361 | air.ct.osat_tw.sink_in | osat_tw | sink_in | chip_mat | - |
| 362 | air.ct.osat_tw.sink_row | osat_tw | sink_row | chip_le | - |
| 363 | air.ct.osat_tw.sink_row | osat_tw | sink_row | chip_mat | - |
| 364 | air.ct.osat_kr.sink_us | osat_kr | sink_us | chip_le | - |
| 365 | air.ct.osat_kr.sink_eu | osat_kr | sink_eu | chip_le | - |
| 366 | air.ct.osat_kr.sink_cn | osat_kr | sink_cn | chip_le | - |
| 367 | air.ct.osat_kr.sink_jp | osat_kr | sink_jp | chip_le | - |
| 368 | air.ct.osat_kr.sink_kr | osat_kr | sink_kr | chip_le | - |
| 369 | air.ct.osat_kr.sink_sea | osat_kr | sink_sea | chip_le | - |
| 370 | air.ct.osat_kr.sink_in | osat_kr | sink_in | chip_le | - |
| 371 | air.ct.osat_kr.sink_row | osat_kr | sink_row | chip_le | - |
| 372 | sea.ct.osat_sg.chk_malacca | osat_sg | chk_malacca | chip_mat | lane.osat_sg.sink_eu |
| 373 | sea.ct.osat_sg.chk_malacca | osat_sg | chk_malacca | chip_mat | lane.osat_sg.sink_in |
| 374 | sea.ct.osat_sg.chk_malacca | osat_sg | chk_malacca | chip_mat | lane.osat_sg.sink_row |
| 375 | sea.ct.osat_sg.chk_taiwan | osat_sg | chk_taiwan | chip_mat | lane.osat_sg.sink_cn |
| 376 | sea.ct.osat_sg.chk_taiwan | osat_sg | chk_taiwan | chip_mat | lane.osat_sg.sink_jp |
| 377 | sea.ct.osat_sg.chk_taiwan | osat_sg | chk_taiwan | chip_mat | lane.osat_sg.sink_kr |
| 378 | sea.ct.osat_sg.sink_us | osat_sg | sink_us | chip_mat | - |
| 379 | air.ct.osat_sg.sink_us | osat_sg | sink_us | chip_le | - |
| 380 | air.ct.osat_sg.sink_us | osat_sg | sink_us | chip_mat | - |
| 381 | air.ct.osat_sg.sink_eu | osat_sg | sink_eu | chip_le | - |
| 382 | air.ct.osat_sg.sink_eu | osat_sg | sink_eu | chip_mat | - |
| 383 | air.ct.osat_sg.sink_cn | osat_sg | sink_cn | chip_le | - |
| 384 | air.ct.osat_sg.sink_jp | osat_sg | sink_jp | chip_le | - |
| 385 | air.ct.osat_sg.sink_jp | osat_sg | sink_jp | chip_mat | - |
| 386 | air.ct.osat_sg.sink_kr | osat_sg | sink_kr | chip_le | - |
| 387 | air.ct.osat_sg.sink_kr | osat_sg | sink_kr | chip_mat | - |
| 388 | sea.ct.osat_sg.sink_sea | osat_sg | sink_sea | chip_mat | - |
| 389 | air.ct.osat_sg.sink_sea | osat_sg | sink_sea | chip_le | - |
| 390 | air.ct.osat_sg.sink_sea | osat_sg | sink_sea | chip_mat | - |
| 391 | air.ct.osat_sg.sink_in | osat_sg | sink_in | chip_le | - |
| 392 | air.ct.osat_sg.sink_in | osat_sg | sink_in | chip_mat | - |
| 393 | air.ct.osat_sg.sink_row | osat_sg | sink_row | chip_le | - |
| 394 | air.ct.osat_sg.sink_row | osat_sg | sink_row | chip_mat | - |

**Override slots** (`override_qty`, `override_mask`):

| slot | chokepoint | commodity | out edge | lane |
| --- | --- | --- | --- | --- |
| 0 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_malacca | lane.src_qa_lng.term_tw |
| 1 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_malacca | lane.src_qa_lng.term_kr |
| 2 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_malacca | lane.src_qa_lng.term_kr.east |
| 3 | chk_hormuz | lng | sea.tb.chk_hormuz.chk_suez | lane.src_qa_lng.term_eu |
| 4 | chk_hormuz | lng | cape.tb.chk_hormuz.chk_cape | lane.src_qa_lng.term_eu.cape |
| 5 | chk_hormuz | lng | lombok.tb.chk_hormuz.chk_taiwan | lane.src_qa_lng.term_kr.lombok |
| 6 | chk_hormuz | lng | lombok.tb.chk_hormuz.term_tw | lane.src_qa_lng.term_tw.lombok |
| 7 | chk_hormuz | lng | sea.tb.chk_hormuz.term_in | lane.src_qa_lng.term_in |
| 8 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_kr |
| 9 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_jp |
| 10 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_cn |
| 11 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_kr.east |
| 12 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_jp.east |
| 13 | chk_hormuz | crude | sea.tb.chk_hormuz.chk_malacca | lane.src_gulf_crude.term_cn.east |
| 14 | chk_hormuz | crude | lombok.tb.chk_hormuz.chk_taiwan | lane.src_gulf_crude.term_kr.lombok |
| 15 | chk_hormuz | crude | lombok.tb.chk_hormuz.chk_taiwan | lane.src_gulf_crude.term_jp.lombok |
| 16 | chk_hormuz | crude | lombok.tb.chk_hormuz.chk_taiwan | lane.src_gulf_crude.term_cn.lombok |
| 17 | chk_hormuz | crude | sea.tb.chk_hormuz.term_in | lane.src_gulf_crude.term_in |
| 18 | chk_malacca | lng | sea.tb.chk_malacca.chk_taiwan | lane.src_qa_lng.term_kr |
| 19 | chk_malacca | lng | sea.tb.chk_malacca.term_tw | lane.src_qa_lng.term_tw |
| 20 | chk_malacca | lng | east.tb.chk_malacca.term_kr | lane.src_qa_lng.term_kr.east |
| 21 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_kr |
| 22 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_jp |
| 23 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_cn |
| 24 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_kr.bypass |
| 25 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_jp.bypass |
| 26 | chk_malacca | crude | sea.tb.chk_malacca.chk_taiwan | lane.src_gulf_crude.term_cn.bypass |
| 27 | chk_malacca | crude | east.tb.chk_malacca.term_kr | lane.src_gulf_crude.term_kr.east |
| 28 | chk_malacca | crude | east.tb.chk_malacca.term_jp | lane.src_gulf_crude.term_jp.east |
| 29 | chk_malacca | crude | east.tb.chk_malacca.term_cn | lane.src_gulf_crude.term_cn.east |
| 30 | chk_suez | lng | sea.tb.chk_suez.term_eu | lane.src_qa_lng.term_eu |
| 31 | chk_suez | lng | turnback.tb.chk_suez.term_eu | - |
| 32 | chk_suez | crude | sea.tb.chk_suez.term_in | lane.src_ru_crude.term_in |
| 33 | chk_suez | crude | turnback.tb.chk_suez.term_in | - |
| 34 | chk_cape | lng | cape.tb.chk_cape.term_eu | lane.src_qa_lng.term_eu.cape |
| 35 | chk_cape | crude | cape.tb.chk_cape.term_in | lane.src_ru_crude.term_in.cape |
| 36 | chk_taiwan | lng | sea.tb.chk_taiwan.term_kr | lane.src_qa_lng.term_kr |
| 37 | chk_taiwan | lng | sea.tb.chk_taiwan.term_kr | lane.src_qa_lng.term_kr.lombok |
| 38 | chk_taiwan | crude | sea.tb.chk_taiwan.term_kr | lane.src_gulf_crude.term_kr |
| 39 | chk_taiwan | crude | sea.tb.chk_taiwan.term_kr | lane.src_gulf_crude.term_kr.lombok |
| 40 | chk_taiwan | crude | sea.tb.chk_taiwan.term_kr | lane.src_gulf_crude.term_kr.bypass |
| 41 | chk_taiwan | crude | sea.tb.chk_taiwan.term_jp | lane.src_gulf_crude.term_jp |
| 42 | chk_taiwan | crude | sea.tb.chk_taiwan.term_jp | lane.src_gulf_crude.term_jp.lombok |
| 43 | chk_taiwan | crude | sea.tb.chk_taiwan.term_jp | lane.src_gulf_crude.term_jp.bypass |
| 44 | chk_taiwan | crude | sea.tb.chk_taiwan.term_cn | lane.src_gulf_crude.term_cn |
| 45 | chk_taiwan | crude | sea.tb.chk_taiwan.term_cn | lane.src_gulf_crude.term_cn.lombok |
| 46 | chk_taiwan | crude | sea.tb.chk_taiwan.term_cn | lane.src_gulf_crude.term_cn.bypass |
| 47 | chk_panama | lng | sea.tb.chk_panama.term_tw | lane.src_us_lng.term_tw |
| 48 | chk_panama | lng | sea.tb.chk_panama.term_jp | lane.src_us_lng.term_jp |
| 49 | chk_panama | crude | sea.tb.chk_panama.term_tw | lane.src_us_crude.term_tw |
| 50 | chk_panama | crude | sea.tb.chk_panama.term_sea | lane.src_us_crude.term_sea |
| 51 | chk_turkish | crude | sea.tb.chk_turkish.chk_suez | lane.src_ru_crude.term_in |
| 52 | chk_turkish | crude | cape.tb.chk_turkish.chk_cape | lane.src_ru_crude.term_in.cape |
| 53 | chk_turkish | crude | sea.tb.chk_turkish.term_eu | lane.src_ru_crude.term_eu |

**Layout tables** (`config['layout']`, the positions of the densified blocks):

| table | entries |
| --- | --- |
| `stock_slots` | 0: src_qa_lng/lng, 1: src_gulf_crude/crude, 2: src_us_lng/lng, 3: src_us_crude/crude, 4: src_au_lng/lng, 5: src_ru_gas/lng, 6: src_ru_crude/crude, 7: src_no_gas/lng, 8: src_kz_uranium/nucfuel, 9: src_ru_enrichment/nucfuel, 10: term_tw/lng, 11: term_tw/crude, 12: term_kr/lng, 13: term_kr/crude, 14: term_jp/lng, 15: term_jp/crude, 16: term_cn/lng, 17: term_cn/crude, 18: term_us/crude, 19: term_eu/lng, 20: term_eu/crude, 21: term_sea/lng, 22: term_sea/crude, 23: term_in/lng, 24: term_in/crude, 25: grid_tw/lng, 26: grid_tw/crude, 27: grid_kr/lng, 28: grid_kr/crude, 29: grid_kr/nucfuel, 30: grid_jp/lng, 31: grid_jp/crude, 32: grid_jp/nucfuel, 33: grid_cn/lng, 34: grid_cn/crude, 35: grid_cn/nucfuel, 36: grid_us/lng, 37: grid_us/crude, 38: grid_us/nucfuel, 39: grid_eu/lng, 40: grid_eu/crude, 41: grid_eu/nucfuel, 42: grid_sea/lng, 43: grid_sea/crude, 44: grid_in/lng, 45: grid_in/crude, 46: grid_in/nucfuel, 47: mat_jp_wafer/wafer, 48: mat_jp_resist/wafer, 49: mat_kr_wafer/wafer, 50: mat_de_wafer/wafer, 51: mat_ua_neon/wafer, 52: mat_cn_neon/wafer, 53: mat_helium/wafer, 54: mat_cn_gage/wafer, 55: fab_tw_leading_1/wafer, 56: fab_tw_leading_1/chip_le_raw, 57: fab_tw_mature_1/wafer, 58: fab_tw_mature_1/chip_mat_raw, 59: fab_us_leading_1/wafer, 60: fab_us_leading_1/chip_le_raw, 61: fab_kr_leading_1/wafer, 62: fab_kr_leading_1/chip_le_raw, 63: fab_kr_memory_1/wafer, 64: fab_kr_memory_1/chip_le_raw, 65: fab_us_leading_2/wafer, 66: fab_us_leading_2/chip_le_raw, 67: fab_jp_memory_1/wafer, 68: fab_jp_memory_1/chip_le_raw, 69: fab_us_leading_3/wafer, 70: fab_us_leading_3/chip_le_raw, 71: fab_eu_leading_1/wafer, 72: fab_eu_leading_1/chip_le_raw, 73: fab_row_leading_1/wafer, 74: fab_row_leading_1/chip_le_raw, 75: fab_cn_mature_1/wafer, 76: fab_cn_mature_1/chip_mat_raw, 77: fab_us_mature_1/wafer, 78: fab_us_mature_1/chip_mat_raw, 79: fab_eu_mature_1/wafer, 80: fab_eu_mature_1/chip_mat_raw, 81: fab_sea_mature_1/wafer, 82: fab_sea_mature_1/chip_mat_raw, 83: fab_tw_mature_2/wafer, 84: fab_tw_mature_2/chip_mat_raw, 85: fab_us_mature_2/wafer, 86: fab_us_mature_2/chip_mat_raw, 87: osat_my/chip_le_raw, 88: osat_my/chip_mat_raw, 89: osat_my/chip_le, 90: osat_my/chip_mat, 91: osat_vn/chip_le_raw, 92: osat_vn/chip_le, 93: osat_ph/chip_le_raw, 94: osat_ph/chip_mat_raw, 95: osat_ph/chip_le, 96: osat_ph/chip_mat, 97: osat_cn/chip_mat_raw, 98: osat_cn/chip_mat, 99: osat_tw/chip_le_raw, 100: osat_tw/chip_mat_raw, 101: osat_tw/chip_le, 102: osat_tw/chip_mat, 103: osat_kr/chip_le_raw, 104: osat_kr/chip_le, 105: osat_sg/chip_le_raw, 106: osat_sg/chip_mat_raw, 107: osat_sg/chip_le, 108: osat_sg/chip_mat, 109: sink_us/chip_le, 110: sink_us/chip_mat, 111: sink_eu/chip_le, 112: sink_eu/chip_mat, 113: sink_cn/chip_le, 114: sink_cn/chip_mat, 115: sink_jp/chip_le, 116: sink_jp/chip_mat, 117: sink_kr/chip_le, 118: sink_kr/chip_mat, 119: sink_sea/chip_le, 120: sink_sea/chip_mat, 121: sink_in/chip_le, 122: sink_in/chip_mat, 123: sink_row/chip_le, 124: sink_row/chip_mat |
| `supply_slots` | 0: src_qa_lng/lng, 1: src_gulf_crude/crude, 2: src_us_lng/lng, 3: src_us_crude/crude, 4: src_au_lng/lng, 5: src_ru_gas/lng, 6: src_ru_crude/crude, 7: src_no_gas/lng, 8: src_kz_uranium/nucfuel, 9: src_ru_enrichment/nucfuel, 10: mat_jp_wafer/wafer, 11: mat_jp_resist/wafer, 12: mat_kr_wafer/wafer, 13: mat_de_wafer/wafer, 14: mat_ua_neon/wafer, 15: mat_cn_neon/wafer, 16: mat_helium/wafer, 17: mat_cn_gage/wafer |
| `demands` | 0: sink_us/chip_le, 1: sink_us/chip_mat, 2: sink_eu/chip_le, 3: sink_eu/chip_mat, 4: sink_cn/chip_le, 5: sink_cn/chip_mat, 6: sink_jp/chip_le, 7: sink_jp/chip_mat, 8: sink_kr/chip_le, 9: sink_kr/chip_mat, 10: sink_sea/chip_le, 11: sink_sea/chip_mat, 12: sink_in/chip_le, 13: sink_in/chip_mat, 14: sink_row/chip_le, 15: sink_row/chip_mat |
| `chokepoints` | 0: chk_hormuz, 1: chk_malacca, 2: chk_suez, 3: chk_cape, 4: chk_taiwan, 5: chk_panama, 6: chk_turkish |
| `fabs` | 0: fab_tw_leading_1, 1: fab_tw_mature_1, 2: fab_us_leading_1, 3: fab_kr_leading_1, 4: fab_kr_memory_1, 5: fab_us_leading_2, 6: fab_jp_memory_1, 7: fab_us_leading_3, 8: fab_eu_leading_1, 9: fab_row_leading_1, 10: fab_cn_mature_1, 11: fab_us_mature_1, 12: fab_eu_mature_1, 13: fab_sea_mature_1, 14: fab_tw_mature_2, 15: fab_us_mature_2 |
| `grids` | 0: grid_tw, 1: grid_kr, 2: grid_jp, 3: grid_cn, 4: grid_us, 5: grid_eu, 6: grid_sea, 7: grid_in |
| `osats` | 0: osat_my, 1: osat_vn, 2: osat_ph, 3: osat_cn, 4: osat_tw, 5: osat_kr, 6: osat_sg |
| `warning_units` | 0: region TW, 1: region KR, 2: region JP, 3: region CN, 4: region US, 5: region EU, 6: region GULF, 7: region RU, 8: region AU, 9: region SEA, 10: region IN, 11: region UA, 12: region KZ, 13: region ROW, 14: dyad 0, 15: chokepoint chk_hormuz, 16: chokepoint chk_malacca, 17: chokepoint chk_suez, 18: chokepoint chk_cape, 19: chokepoint chk_taiwan, 20: chokepoint chk_panama, 21: chokepoint chk_turkish |
| `cost_components` | 0: freight, 1: war_risk, 2: tariff, 3: holding, 4: queue_holding, 5: shortage, 6: disposal, 7: shed |
| `release_pairs` | 0: chk_hormuz/lng, 1: chk_hormuz/crude, 2: chk_malacca/lng, 3: chk_malacca/crude, 4: chk_suez/lng, 5: chk_suez/crude, 6: chk_cape/lng, 7: chk_cape/crude, 8: chk_taiwan/lng, 9: chk_taiwan/crude, 10: chk_panama/lng, 11: chk_panama/crude, 12: chk_turkish/lng, 13: chk_turkish/crude |
| `lot_keys` | 0: chk_hormuz/lng/lane.src_qa_lng.term_tw/sea.tb.chk_hormuz.chk_malacca, 1: chk_hormuz/lng/lane.src_qa_lng.term_kr/sea.tb.chk_hormuz.chk_malacca, 2: chk_hormuz/lng/lane.src_qa_lng.term_eu/sea.tb.chk_hormuz.chk_suez, 3: chk_hormuz/lng/lane.src_qa_lng.term_in/sea.tb.chk_hormuz.term_in, 4: chk_hormuz/lng/lane.src_qa_lng.term_eu.cape/cape.tb.chk_hormuz.chk_cape, 5: chk_hormuz/lng/lane.src_qa_lng.term_tw.lombok/lombok.tb.chk_hormuz.term_tw, 6: chk_hormuz/lng/lane.src_qa_lng.term_kr.lombok/lombok.tb.chk_hormuz.chk_taiwan, 7: chk_hormuz/lng/lane.src_qa_lng.term_kr.east/sea.tb.chk_hormuz.chk_malacca, 8: chk_hormuz/crude/lane.src_gulf_crude.term_kr/sea.tb.chk_hormuz.chk_malacca, 9: chk_hormuz/crude/lane.src_gulf_crude.term_jp/sea.tb.chk_hormuz.chk_malacca, 10: chk_hormuz/crude/lane.src_gulf_crude.term_cn/sea.tb.chk_hormuz.chk_malacca, 11: chk_hormuz/crude/lane.src_gulf_crude.term_in/sea.tb.chk_hormuz.term_in, 12: chk_hormuz/crude/lane.src_gulf_crude.term_kr.lombok/lombok.tb.chk_hormuz.chk_taiwan, 13: chk_hormuz/crude/lane.src_gulf_crude.term_jp.lombok/lombok.tb.chk_hormuz.chk_taiwan, 14: chk_hormuz/crude/lane.src_gulf_crude.term_cn.lombok/lombok.tb.chk_hormuz.chk_taiwan, 15: chk_hormuz/crude/lane.src_gulf_crude.term_kr.east/sea.tb.chk_hormuz.chk_malacca, 16: chk_hormuz/crude/lane.src_gulf_crude.term_jp.east/sea.tb.chk_hormuz.chk_malacca, 17: chk_hormuz/crude/lane.src_gulf_crude.term_cn.east/sea.tb.chk_hormuz.chk_malacca, 18: chk_hormuz/wafer/lane.mat_helium.fab_tw_leading_1/sea.ct.chk_hormuz.chk_malacca, 19: chk_hormuz/wafer/lane.mat_helium.fab_kr_leading_1/sea.ct.chk_hormuz.chk_malacca, 20: chk_hormuz/wafer/lane.mat_helium.fab_kr_memory_1/sea.ct.chk_hormuz.chk_malacca, 21: chk_hormuz/wafer/lane.mat_helium.fab_eu_leading_1/sea.ct.chk_hormuz.chk_suez, 22: chk_hormuz/wafer/lane.mat_helium.fab_sea_mature_1/sea.ct.chk_hormuz.chk_malacca, 23: chk_hormuz/wafer/lane.mat_helium.fab_eu_leading_1.cape/cape.ct.chk_hormuz.chk_cape, 24: chk_hormuz/wafer/lane.mat_helium.fab_tw_leading_1.lombok/lombok.ct.chk_hormuz.fab_tw_leading_1, 25: chk_hormuz/wafer/lane.mat_helium.fab_kr_leading_1.lombok/lombok.ct.chk_hormuz.chk_taiwan, 26: chk_hormuz/wafer/lane.mat_helium.fab_kr_memory_1.lombok/lombok.ct.chk_hormuz.chk_taiwan, 27: chk_hormuz/wafer/lane.mat_helium.fab_sea_mature_1.lombok/lombok.ct.chk_hormuz.fab_sea_mature_1, 28: chk_hormuz/wafer/lane.mat_helium.fab_kr_leading_1.east/sea.ct.chk_hormuz.chk_malacca, 29: chk_hormuz/wafer/lane.mat_helium.fab_kr_memory_1.east/sea.ct.chk_hormuz.chk_malacca, 30: chk_malacca/lng/lane.src_qa_lng.term_tw/sea.tb.chk_malacca.term_tw, 31: chk_malacca/lng/lane.src_qa_lng.term_kr/sea.tb.chk_malacca.chk_taiwan, 32: chk_malacca/lng/lane.src_qa_lng.term_kr.east/east.tb.chk_malacca.term_kr, 33: chk_malacca/crude/lane.src_gulf_crude.term_kr/sea.tb.chk_malacca.chk_taiwan, 34: chk_malacca/crude/lane.src_gulf_crude.term_jp/sea.tb.chk_malacca.chk_taiwan, 35: chk_malacca/crude/lane.src_gulf_crude.term_cn/sea.tb.chk_malacca.chk_taiwan, 36: chk_malacca/crude/lane.src_gulf_crude.term_kr.east/east.tb.chk_malacca.term_kr, 37: chk_malacca/crude/lane.src_gulf_crude.term_jp.east/east.tb.chk_malacca.term_jp, 38: chk_malacca/crude/lane.src_gulf_crude.term_cn.east/east.tb.chk_malacca.term_cn, 39: chk_malacca/crude/lane.src_gulf_crude.term_kr.bypass/sea.tb.chk_malacca.chk_taiwan, 40: chk_malacca/crude/lane.src_gulf_crude.term_jp.bypass/sea.tb.chk_malacca.chk_taiwan, 41: chk_malacca/crude/lane.src_gulf_crude.term_cn.bypass/sea.tb.chk_malacca.chk_taiwan, 42: chk_malacca/wafer/lane.mat_jp_resist.fab_eu_leading_1/sea.ct.chk_malacca.chk_suez, 43: chk_malacca/wafer/lane.mat_jp_resist.fab_row_leading_1/sea.ct.chk_malacca.chk_suez, 44: chk_malacca/wafer/lane.mat_de_wafer.fab_sea_mature_1/sea.ct.chk_malacca.fab_sea_mature_1, 45: chk_malacca/wafer/lane.mat_de_wafer.fab_tw_mature_2/sea.ct.chk_malacca.fab_tw_mature_2, 46: chk_malacca/wafer/lane.mat_ua_neon.fab_tw_leading_1/sea.ct.chk_malacca.fab_tw_leading_1, 47: chk_malacca/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_malacca.chk_taiwan, 48: chk_malacca/wafer/lane.mat_helium.fab_tw_leading_1/sea.ct.chk_malacca.fab_tw_leading_1, 49: chk_malacca/wafer/lane.mat_helium.fab_kr_leading_1/sea.ct.chk_malacca.chk_taiwan, 50: chk_malacca/wafer/lane.mat_helium.fab_kr_memory_1/sea.ct.chk_malacca.chk_taiwan, 51: chk_malacca/wafer/lane.mat_helium.fab_sea_mature_1/sea.ct.chk_malacca.fab_sea_mature_1, 52: chk_malacca/wafer/lane.mat_jp_resist.fab_eu_leading_1.cape/cape.ct.chk_malacca.chk_cape, 53: chk_malacca/wafer/lane.mat_jp_resist.fab_row_leading_1.cape/cape.ct.chk_malacca.chk_cape, 54: chk_malacca/wafer/lane.mat_de_wafer.fab_sea_mature_1.cape/sea.ct.chk_malacca.fab_sea_mature_1, 55: chk_malacca/wafer/lane.mat_de_wafer.fab_tw_mature_2.cape/sea.ct.chk_malacca.fab_tw_mature_2, 56: chk_malacca/wafer/lane.mat_ua_neon.fab_tw_leading_1.cape/sea.ct.chk_malacca.fab_tw_leading_1, 57: chk_malacca/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/sea.ct.chk_malacca.chk_taiwan, 58: chk_malacca/wafer/lane.mat_jp_resist.fab_eu_leading_1.east/sea.ct.chk_malacca.chk_suez, 59: chk_malacca/wafer/lane.mat_jp_resist.fab_row_leading_1.east/sea.ct.chk_malacca.chk_suez, 60: chk_malacca/wafer/lane.mat_ua_neon.fab_kr_memory_1.east/east.ct.chk_malacca.fab_kr_memory_1, 61: chk_malacca/wafer/lane.mat_helium.fab_kr_leading_1.east/east.ct.chk_malacca.fab_kr_leading_1, 62: chk_malacca/wafer/lane.mat_helium.fab_kr_memory_1.east/east.ct.chk_malacca.fab_kr_memory_1, 63: chk_malacca/chip_le_raw/lane.fab_eu_leading_1.osat_my/sea.ct.chk_malacca.osat_my, 64: chk_malacca/chip_le_raw/lane.fab_eu_leading_1.osat_vn/sea.ct.chk_malacca.osat_vn, 65: chk_malacca/chip_le_raw/lane.fab_row_leading_1.osat_my/sea.ct.chk_malacca.osat_my, 66: chk_malacca/chip_le_raw/lane.fab_row_leading_1.osat_vn/sea.ct.chk_malacca.osat_vn, 67: chk_malacca/chip_le_raw/lane.fab_eu_leading_1.osat_my.cape/sea.ct.chk_malacca.osat_my, 68: chk_malacca/chip_le_raw/lane.fab_eu_leading_1.osat_vn.cape/sea.ct.chk_malacca.osat_vn, 69: chk_malacca/chip_le_raw/lane.fab_row_leading_1.osat_my.cape/sea.ct.chk_malacca.osat_my, 70: chk_malacca/chip_le_raw/lane.fab_row_leading_1.osat_vn.cape/sea.ct.chk_malacca.osat_vn, 71: chk_malacca/chip_mat_raw/lane.fab_eu_mature_1.osat_my/sea.ct.chk_malacca.osat_my, 72: chk_malacca/chip_mat_raw/lane.fab_eu_mature_1.osat_sg/sea.ct.chk_malacca.osat_sg, 73: chk_malacca/chip_mat_raw/lane.fab_eu_mature_1.osat_my.cape/sea.ct.chk_malacca.osat_my, 74: chk_malacca/chip_mat_raw/lane.fab_eu_mature_1.osat_sg.cape/sea.ct.chk_malacca.osat_sg, 75: chk_malacca/chip_mat/lane.osat_my.sink_eu/sea.ct.chk_malacca.chk_suez, 76: chk_malacca/chip_mat/lane.osat_my.sink_in/sea.ct.chk_malacca.sink_in, 77: chk_malacca/chip_mat/lane.osat_my.sink_row/sea.ct.chk_malacca.chk_suez, 78: chk_malacca/chip_mat/lane.osat_ph.sink_eu/sea.ct.chk_malacca.chk_suez, 79: chk_malacca/chip_mat/lane.osat_ph.sink_in/sea.ct.chk_malacca.sink_in, 80: chk_malacca/chip_mat/lane.osat_ph.sink_row/sea.ct.chk_malacca.chk_suez, 81: chk_malacca/chip_mat/lane.osat_cn.sink_eu/sea.ct.chk_malacca.chk_suez, 82: chk_malacca/chip_mat/lane.osat_cn.sink_in/sea.ct.chk_malacca.sink_in, 83: chk_malacca/chip_mat/lane.osat_cn.sink_row/sea.ct.chk_malacca.chk_suez, 84: chk_malacca/chip_mat/lane.osat_tw.sink_eu/sea.ct.chk_malacca.chk_suez, 85: chk_malacca/chip_mat/lane.osat_tw.sink_in/sea.ct.chk_malacca.sink_in, 86: chk_malacca/chip_mat/lane.osat_tw.sink_row/sea.ct.chk_malacca.chk_suez, 87: chk_malacca/chip_mat/lane.osat_sg.sink_eu/sea.ct.chk_malacca.chk_suez, 88: chk_malacca/chip_mat/lane.osat_sg.sink_in/sea.ct.chk_malacca.sink_in, 89: chk_malacca/chip_mat/lane.osat_sg.sink_row/sea.ct.chk_malacca.chk_suez, 90: chk_suez/lng/lane.src_qa_lng.term_eu/sea.tb.chk_suez.term_eu, 91: chk_suez/crude/lane.src_ru_crude.term_in/sea.tb.chk_suez.term_in, 92: chk_suez/wafer/lane.mat_jp_resist.fab_eu_leading_1/sea.ct.chk_suez.fab_eu_leading_1, 93: chk_suez/wafer/lane.mat_jp_resist.fab_row_leading_1/sea.ct.chk_suez.fab_row_leading_1, 94: chk_suez/wafer/lane.mat_de_wafer.fab_sea_mature_1/sea.ct.chk_suez.chk_malacca, 95: chk_suez/wafer/lane.mat_de_wafer.fab_tw_mature_2/sea.ct.chk_suez.chk_malacca, 96: chk_suez/wafer/lane.mat_ua_neon.fab_tw_leading_1/sea.ct.chk_suez.chk_malacca, 97: chk_suez/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_suez.chk_malacca, 98: chk_suez/wafer/lane.mat_helium.fab_eu_leading_1/sea.ct.chk_suez.fab_eu_leading_1, 99: chk_suez/wafer/lane.mat_jp_resist.fab_eu_leading_1.lombok/sea.ct.chk_suez.fab_eu_leading_1, 100: chk_suez/wafer/lane.mat_jp_resist.fab_row_leading_1.lombok/sea.ct.chk_suez.fab_row_leading_1, 101: chk_suez/wafer/lane.mat_de_wafer.fab_sea_mature_1.lombok/lombok.ct.chk_suez.fab_sea_mature_1, 102: chk_suez/wafer/lane.mat_de_wafer.fab_tw_mature_2.lombok/lombok.ct.chk_suez.fab_tw_mature_2, 103: chk_suez/wafer/lane.mat_ua_neon.fab_tw_leading_1.lombok/lombok.ct.chk_suez.fab_tw_leading_1, 104: chk_suez/wafer/lane.mat_ua_neon.fab_kr_memory_1.lombok/lombok.ct.chk_suez.chk_taiwan, 105: chk_suez/wafer/lane.mat_jp_resist.fab_eu_leading_1.east/sea.ct.chk_suez.fab_eu_leading_1, 106: chk_suez/wafer/lane.mat_jp_resist.fab_row_leading_1.east/sea.ct.chk_suez.fab_row_leading_1, 107: chk_suez/wafer/lane.mat_ua_neon.fab_kr_memory_1.east/sea.ct.chk_suez.chk_malacca, 108: chk_suez/chip_le_raw/lane.fab_eu_leading_1.osat_my/sea.ct.chk_suez.chk_malacca, 109: chk_suez/chip_le_raw/lane.fab_eu_leading_1.osat_vn/sea.ct.chk_suez.chk_malacca, 110: chk_suez/chip_le_raw/lane.fab_row_leading_1.osat_my/sea.ct.chk_suez.chk_malacca, 111: chk_suez/chip_le_raw/lane.fab_row_leading_1.osat_vn/sea.ct.chk_suez.chk_malacca, 112: chk_suez/chip_le_raw/lane.fab_eu_leading_1.osat_my.lombok/lombok.ct.chk_suez.osat_my, 113: chk_suez/chip_le_raw/lane.fab_eu_leading_1.osat_vn.lombok/lombok.ct.chk_suez.osat_vn, 114: chk_suez/chip_le_raw/lane.fab_row_leading_1.osat_my.lombok/lombok.ct.chk_suez.osat_my, 115: chk_suez/chip_le_raw/lane.fab_row_leading_1.osat_vn.lombok/lombok.ct.chk_suez.osat_vn, 116: chk_suez/chip_mat_raw/lane.fab_eu_mature_1.osat_my/sea.ct.chk_suez.chk_malacca, 117: chk_suez/chip_mat_raw/lane.fab_eu_mature_1.osat_sg/sea.ct.chk_suez.chk_malacca, 118: chk_suez/chip_mat_raw/lane.fab_eu_mature_1.osat_my.lombok/lombok.ct.chk_suez.osat_my, 119: chk_suez/chip_mat_raw/lane.fab_eu_mature_1.osat_sg.lombok/lombok.ct.chk_suez.osat_sg, 120: chk_suez/chip_mat/lane.osat_my.sink_eu/sea.ct.chk_suez.sink_eu, 121: chk_suez/chip_mat/lane.osat_my.sink_row/sea.ct.chk_suez.sink_row, 122: chk_suez/chip_mat/lane.osat_ph.sink_eu/sea.ct.chk_suez.sink_eu, 123: chk_suez/chip_mat/lane.osat_ph.sink_row/sea.ct.chk_suez.sink_row, 124: chk_suez/chip_mat/lane.osat_cn.sink_eu/sea.ct.chk_suez.sink_eu, 125: chk_suez/chip_mat/lane.osat_cn.sink_row/sea.ct.chk_suez.sink_row, 126: chk_suez/chip_mat/lane.osat_tw.sink_eu/sea.ct.chk_suez.sink_eu, 127: chk_suez/chip_mat/lane.osat_tw.sink_row/sea.ct.chk_suez.sink_row, 128: chk_suez/chip_mat/lane.osat_sg.sink_eu/sea.ct.chk_suez.sink_eu, 129: chk_suez/chip_mat/lane.osat_sg.sink_row/sea.ct.chk_suez.sink_row, 130: chk_cape/lng/lane.src_qa_lng.term_eu.cape/cape.tb.chk_cape.term_eu, 131: chk_cape/crude/lane.src_ru_crude.term_in.cape/cape.tb.chk_cape.term_in, 132: chk_cape/wafer/lane.mat_jp_resist.fab_eu_leading_1.cape/cape.ct.chk_cape.fab_eu_leading_1, 133: chk_cape/wafer/lane.mat_jp_resist.fab_row_leading_1.cape/cape.ct.chk_cape.fab_row_leading_1, 134: chk_cape/wafer/lane.mat_de_wafer.fab_sea_mature_1.cape/cape.ct.chk_cape.chk_malacca, 135: chk_cape/wafer/lane.mat_de_wafer.fab_tw_mature_2.cape/cape.ct.chk_cape.chk_malacca, 136: chk_cape/wafer/lane.mat_ua_neon.fab_tw_leading_1.cape/cape.ct.chk_cape.chk_malacca, 137: chk_cape/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/cape.ct.chk_cape.chk_malacca, 138: chk_cape/wafer/lane.mat_helium.fab_eu_leading_1.cape/cape.ct.chk_cape.fab_eu_leading_1, 139: chk_cape/chip_le_raw/lane.fab_eu_leading_1.osat_my.cape/cape.ct.chk_cape.chk_malacca, 140: chk_cape/chip_le_raw/lane.fab_eu_leading_1.osat_vn.cape/cape.ct.chk_cape.chk_malacca, 141: chk_cape/chip_le_raw/lane.fab_row_leading_1.osat_my.cape/cape.ct.chk_cape.chk_malacca, 142: chk_cape/chip_le_raw/lane.fab_row_leading_1.osat_vn.cape/cape.ct.chk_cape.chk_malacca, 143: chk_cape/chip_mat_raw/lane.fab_eu_mature_1.osat_my.cape/cape.ct.chk_cape.chk_malacca, 144: chk_cape/chip_mat_raw/lane.fab_eu_mature_1.osat_sg.cape/cape.ct.chk_cape.chk_malacca, 145: chk_taiwan/lng/lane.src_qa_lng.term_kr/sea.tb.chk_taiwan.term_kr, 146: chk_taiwan/lng/lane.src_qa_lng.term_kr.lombok/sea.tb.chk_taiwan.term_kr, 147: chk_taiwan/crude/lane.src_gulf_crude.term_kr/sea.tb.chk_taiwan.term_kr, 148: chk_taiwan/crude/lane.src_gulf_crude.term_jp/sea.tb.chk_taiwan.term_jp, 149: chk_taiwan/crude/lane.src_gulf_crude.term_cn/sea.tb.chk_taiwan.term_cn, 150: chk_taiwan/crude/lane.src_gulf_crude.term_kr.lombok/sea.tb.chk_taiwan.term_kr, 151: chk_taiwan/crude/lane.src_gulf_crude.term_jp.lombok/sea.tb.chk_taiwan.term_jp, 152: chk_taiwan/crude/lane.src_gulf_crude.term_cn.lombok/sea.tb.chk_taiwan.term_cn, 153: chk_taiwan/crude/lane.src_gulf_crude.term_kr.bypass/sea.tb.chk_taiwan.term_kr, 154: chk_taiwan/crude/lane.src_gulf_crude.term_jp.bypass/sea.tb.chk_taiwan.term_jp, 155: chk_taiwan/crude/lane.src_gulf_crude.term_cn.bypass/sea.tb.chk_taiwan.term_cn, 156: chk_taiwan/wafer/lane.mat_jp_resist.fab_eu_leading_1/sea.ct.chk_taiwan.chk_malacca, 157: chk_taiwan/wafer/lane.mat_jp_resist.fab_row_leading_1/sea.ct.chk_taiwan.chk_malacca, 158: chk_taiwan/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_taiwan.fab_kr_memory_1, 159: chk_taiwan/wafer/lane.mat_cn_neon.fab_sea_mature_1/sea.ct.chk_taiwan.fab_sea_mature_1, 160: chk_taiwan/wafer/lane.mat_helium.fab_kr_leading_1/sea.ct.chk_taiwan.fab_kr_leading_1, 161: chk_taiwan/wafer/lane.mat_helium.fab_kr_memory_1/sea.ct.chk_taiwan.fab_kr_memory_1, 162: chk_taiwan/wafer/lane.mat_jp_resist.fab_eu_leading_1.cape/sea.ct.chk_taiwan.chk_malacca, 163: chk_taiwan/wafer/lane.mat_jp_resist.fab_row_leading_1.cape/sea.ct.chk_taiwan.chk_malacca, 164: chk_taiwan/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/sea.ct.chk_taiwan.fab_kr_memory_1, 165: chk_taiwan/wafer/lane.mat_jp_resist.fab_eu_leading_1.lombok/lombok.ct.chk_taiwan.chk_suez, 166: chk_taiwan/wafer/lane.mat_jp_resist.fab_row_leading_1.lombok/lombok.ct.chk_taiwan.chk_suez, 167: chk_taiwan/wafer/lane.mat_ua_neon.fab_kr_memory_1.lombok/sea.ct.chk_taiwan.fab_kr_memory_1, 168: chk_taiwan/wafer/lane.mat_helium.fab_kr_leading_1.lombok/sea.ct.chk_taiwan.fab_kr_leading_1, 169: chk_taiwan/wafer/lane.mat_helium.fab_kr_memory_1.lombok/sea.ct.chk_taiwan.fab_kr_memory_1, 170: chk_taiwan/chip_le_raw/lane.fab_kr_leading_1.osat_vn/sea.ct.chk_taiwan.osat_vn, 171: chk_taiwan/chip_le_raw/lane.fab_kr_memory_1.osat_my/sea.ct.chk_taiwan.osat_my, 172: chk_taiwan/chip_le_raw/lane.fab_kr_memory_1.osat_vn/sea.ct.chk_taiwan.osat_vn, 173: chk_taiwan/chip_le_raw/lane.fab_jp_memory_1.osat_my/sea.ct.chk_taiwan.osat_my, 174: chk_taiwan/chip_le_raw/lane.fab_jp_memory_1.osat_ph/sea.ct.chk_taiwan.osat_ph, 175: chk_taiwan/chip_mat_raw/lane.fab_cn_mature_1.osat_my/sea.ct.chk_taiwan.osat_my, 176: chk_taiwan/chip_mat/lane.osat_my.sink_cn/sea.ct.chk_taiwan.sink_cn, 177: chk_taiwan/chip_mat/lane.osat_my.sink_jp/sea.ct.chk_taiwan.sink_jp, 178: chk_taiwan/chip_mat/lane.osat_my.sink_kr/sea.ct.chk_taiwan.sink_kr, 179: chk_taiwan/chip_mat/lane.osat_ph.sink_cn/sea.ct.chk_taiwan.sink_cn, 180: chk_taiwan/chip_mat/lane.osat_ph.sink_jp/sea.ct.chk_taiwan.sink_jp, 181: chk_taiwan/chip_mat/lane.osat_ph.sink_kr/sea.ct.chk_taiwan.sink_kr, 182: chk_taiwan/chip_mat/lane.osat_cn.sink_eu/sea.ct.chk_taiwan.chk_malacca, 183: chk_taiwan/chip_mat/lane.osat_cn.sink_sea/sea.ct.chk_taiwan.sink_sea, 184: chk_taiwan/chip_mat/lane.osat_cn.sink_in/sea.ct.chk_taiwan.chk_malacca, 185: chk_taiwan/chip_mat/lane.osat_cn.sink_row/sea.ct.chk_taiwan.chk_malacca, 186: chk_taiwan/chip_mat/lane.osat_sg.sink_cn/sea.ct.chk_taiwan.sink_cn, 187: chk_taiwan/chip_mat/lane.osat_sg.sink_jp/sea.ct.chk_taiwan.sink_jp, 188: chk_taiwan/chip_mat/lane.osat_sg.sink_kr/sea.ct.chk_taiwan.sink_kr, 189: chk_panama/lng/lane.src_us_lng.term_tw/sea.tb.chk_panama.term_tw, 190: chk_panama/lng/lane.src_us_lng.term_jp/sea.tb.chk_panama.term_jp, 191: chk_panama/crude/lane.src_us_crude.term_tw/sea.tb.chk_panama.term_tw, 192: chk_panama/crude/lane.src_us_crude.term_sea/sea.tb.chk_panama.term_sea, 193: chk_turkish/crude/lane.src_ru_crude.term_eu/sea.tb.chk_turkish.term_eu, 194: chk_turkish/crude/lane.src_ru_crude.term_in/sea.tb.chk_turkish.chk_suez, 195: chk_turkish/crude/lane.src_ru_crude.term_in.cape/cape.tb.chk_turkish.chk_cape, 196: chk_turkish/wafer/lane.mat_ua_neon.fab_tw_leading_1/sea.ct.chk_turkish.chk_suez, 197: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1/sea.ct.chk_turkish.chk_suez, 198: chk_turkish/wafer/lane.mat_ua_neon.fab_eu_leading_1/sea.ct.chk_turkish.fab_eu_leading_1, 199: chk_turkish/wafer/lane.mat_ua_neon.fab_eu_mature_1/sea.ct.chk_turkish.fab_eu_mature_1, 200: chk_turkish/wafer/lane.mat_ua_neon.fab_us_mature_2/sea.ct.chk_turkish.fab_us_mature_2, 201: chk_turkish/wafer/lane.mat_ua_neon.fab_tw_leading_1.cape/cape.ct.chk_turkish.chk_cape, 202: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1.cape/cape.ct.chk_turkish.chk_cape, 203: chk_turkish/wafer/lane.mat_ua_neon.fab_tw_leading_1.lombok/sea.ct.chk_turkish.chk_suez, 204: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1.lombok/sea.ct.chk_turkish.chk_suez, 205: chk_turkish/wafer/lane.mat_ua_neon.fab_kr_memory_1.east/sea.ct.chk_turkish.chk_suez |

**Static tables** (`config['static']`): a table is a dict of equal-length lists, one entry per node, edge, lane, commodity, slot or sink, and the indices above point into them.

| field | here | meaning |
| --- | --- | --- |
| `instance` | keys `T`, `chokepoint_adjacency`, `commodities`, `compatibility`, `edges`, `initial_state`, `instance_id`, `kind`, `lanes`, `nodes`, `params`, `prohibitions_at_reset`, `provenance`, `region_class`, `regions`, `routing_table`, `schema_version`, `trade_adjacency`, `units`, `use` | the full public instance JSON; `initial_state.pipeline` holds the week-0 shipments of the nominal plan (the nominal flows the heuristic sample reads); edges there call the lead time `tau` |
| `instance_id` | `chokepoint-full` | the instance's name |
| `instance_hash` | `5735cfe5142b...` | SHA-256 of the instance |
| `T` | `104` | the horizon, in weeks |
| `units` | chip_le: wafer-eq 300 mm, chip_le_raw: wafer-eq 300 mm, chip_mat: wafer-eq 300 mm, chip_mat_raw: wafer-eq 300 mm, cost: USD, crude: GWh fuel, lng: GWh fuel, nucfuel: GWh fuel, wafer: wafer-eq 300 mm | the unit of each commodity's quantities, and of costs |
| `regions` | 14 entries | region names: `nodes.region`, `messages.region`, `dyads.*` and the warning's region units index them |
| `nodes.id` | 72 entries | node name |
| `nodes.type` | 72 entries | source, terminal, grid, chokepoint, material, fab, osat or sink |
| `nodes.region` | 72 entries | region index |
| `commodities.id` | 8 entries | commodity name |
| `commodities.v` | 8 entries | customs value v_k, USD per unit |
| `commodities.pool` | 8 entries | chokepoint throughput pool: tb (tanker/bulk, `graph_now.kappa.tb`) or ct (container) |
| `commodities.override` | 8 entries | true for a tanker commodity, whose cargo queued at a chokepoint `release_mode` and `override_qty` steer |
| `edges.id` | 369 entries | edge name |
| `edges.tail` | 369 entries | node index the edge leaves |
| `edges.head` | 369 entries | node index the edge reaches |
| `edges.mode` | 369 entries | sea, air, pipeline or grid |
| `edges.tau0` | 369 entries | nominal lead time in weeks (`graph_now.tau` is this week's) |
| `edges.c0` | 369 entries | nominal freight, USD per unit (`graph_now.c` is this week's) |
| `edges.u0` | 369 entries | nominal capacity per week, null on a grid coupling (`graph_now.u` is this week's) |
| `edges.K` | 369 entries | commodity indices the edge may carry (empty on a grid coupling) |
| `edges.alt_of` | 369 entries | the route the edge duplicates, {'lane': i} or {'edge': i}, or null |
| `edges.pool` | 369 entries | chokepoint pool of its traffic, null on a grid coupling |
| `lanes.id` | 114 entries | lane name (a route through one or more chokepoints) |
| `lanes.edges` | 114 entries | edge indices along the lane, in order |
| `lanes.chokepoints` | 114 entries | chokepoint node indices it passes, in order |
| `lanes.alt_of` | 114 entries | the route the lane duplicates, {'lane': i} or {'edge': i}, or null |
| `action_slots.edge` | 395 entries | each slot's edge (on a lane, the lane's first edge) |
| `action_slots.k` | 395 entries | each slot's commodity index |
| `action_slots.lane` | 395 entries | each slot's lane index, null off any lane |
| `override_slots.chokepoint` | 54 entries | chokepoint node index |
| `override_slots.k` | 54 entries | tanker commodity index |
| `override_slots.out_edge` | 54 entries | edge the released cargo leaves the chokepoint by |
| `override_slots.lane` | 54 entries | lane index, null off any lane |
| `sinks.node` | 16 entries | demand node index (the rows of `layout.demands`) |
| `sinks.k` | 16 entries | commodity demanded |
| `sinks.backlog` | 16 entries | true: unserved demand is carried to later weeks; false: it is lost |
| `sinks.pi` | 16 entries | shortage penalty pi, USD per unit of demand not served |
| `dyads.a` | 1 entry | first region index of each dyad (the warning's dyad units) |
| `dyads.b` | 1 entry | second region index of each dyad |
| `regime` | `standard` | the information regime's published parameters (`name`, `L`, `a`, `phi`, `chi`, `h_cov`, `skill`, `blackout`); null when the runner hides it |

