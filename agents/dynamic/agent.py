import math
import numpy as np

# ==========================================
# AUTO-UPDATED BY tune.py (DO NOT EDIT MANUALLY)
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
    "end_margin": 1
}
# ==========================================

# ==========================================
# AUTO-UPDATED BY es_train.py (DO NOT EDIT MANUALLY)
# Flat weights of the neural modulator. None => zero-output init, which
# reproduces the static tuned policy above exactly.
# ==========================================
# <<POLICY_BEGIN>>
_POLICY_WEIGHTS = [
    0.13856959315178385, -0.06527392613771536, 0.059573466358598864, 0.33309019057021994, 0.09550023609784983, -0.0737328136587734,
    -0.08526653214295103, 0.0006591031031795355, -0.48661560519297126, -0.32585988802052956, 0.027186738360667123, 0.18634584201511603,
    -0.13732053604159533, 0.1846270262980805, -0.15393327374016014, -0.13660039486756267, -0.13909498983223303, -0.0290781596637166,
    0.21048722987141408, -0.2794413326782465, -0.16086801462904632, -0.15768319237367473, -0.3706882458173, -0.39523262156539385,
    -0.10358095955757454, 0.0036806299325656037, 0.25498581808862597, -0.18283487879554658, 0.21472041297407818, 0.3402363769756415,
    -0.005393673506471198, 0.23404946725245782, 0.2855079709884294, 0.04393443745555264, -0.3497981463735509, -0.032454550736271834,
    0.03408221981032843, -0.17545503941455468, 0.08585561883385544, 0.3030144643592953, -0.18427618222493494, 0.11457645699715918,
    0.0644000266085542, 0.6809793045682785, -0.2319462779338638, -0.32786803621650573, 0.028924414810670818, -0.15846887091628545,
    0.29016936042086783, 0.20020630440832565, -0.12641538882610756, -0.14517533619583864, -0.2776154084679468, -0.12754025269756633,
    -0.19226140283248305, -0.038184588783622034, 0.1900958933735243, -0.11766752506396781, -0.2523999419978777, 0.03230665527375498,
    0.15899423772253204, -0.18225262398638487, -0.42714600420744164, 0.058583013746275114, -0.028215491906597983, -0.2450543801433486,
    0.17259004309332543, -0.014598126098482845, -0.016028856346796298, 0.084511006030687, -0.4264537150907323, 0.15430639360399695,
    -0.12982501393811544, 0.24651275954140303, 0.740862250146512, -0.31255018605267015, -0.05732407764909464, -0.06788107287390752,
    -0.08731534555054409, 0.5332600588724161, -0.4284075446877833, 0.271418330602693, -0.23161721375738367, -0.041016748133958,
    0.3068729149387423, 0.01649015434496829, -0.0881295014832617, -0.45024711897517333, -0.2277086284929674, 0.022269881697703345,
    -0.1790318828695372, 0.13300783819223694, 0.35619167634215226, -0.3004913484445834, -0.22941947307125415, 0.06919528361411977,
    0.1796375577611572, -0.0009937787240106902, -0.1448239784648558, -0.023857871233691163, -0.35953233295947346, 0.11349224820519542,
    -0.33013817650792365, -0.002106958836285465, 0.10542615793982688, -0.35095523109391813, 0.029083437770966055, -0.357139701299698,
    -0.6074105762753349, 0.09477977453099592, -0.15684029311828473, 0.2424182178322151, 0.07927180725229656, -0.19740204346655993,
    0.08891210070746201, -0.04976592041594136, 0.46609198019221076, 0.3195770870538283, -0.26560412137579226, 0.35813692219294124,
    -0.03970085233250006, -0.23142294948175243, 0.07630973796406151, -0.27414605300401707, 0.06625515573302085, -0.21496426119452233,
    -0.1002160181227391, -0.14163282093209603, 0.2608365649120476, 0.1537067049903372, -0.34774459924963813, -0.059719082107433916,
    -0.05797869885760264, 0.11386730717862552, -0.27836436900910555, 0.33316538384167155, 0.4902250407236881, 0.14205729172249743,
    -0.19399283768852357, 0.3258572862558016, -0.1331714049008537, -0.0016192105813753696, -0.14809926688893113, -0.0488450298804101,
    -0.07789854764564254, 0.2747553420710826, -0.2833210927235805, -0.3780834635400106, -0.418997273892528, 0.2300430792514941,
    -0.02351532035924198, -0.11978921130083706, -0.1989759092016755, 0.13036014459150558, 0.3387904799493639, -0.22780607158446903,
    0.16567945032376924, 0.2281385137458395, -0.7999597057116545, -1.0409337034116508, -0.060221324046007794, -0.5391138660122528,
    -0.1719933803864419, 0.05112055707972783, 0.28830202074531136, -0.3453649439381174, -0.21084321222251107, -0.38148057438546285,
    0.46134060518709763, 0.12997984039387053, 0.25199380324926957, 0.005381703335430402, -0.1496614944734918, -0.089882186601005,
    0.3728261061194424, -0.10091172821307856, 0.015064425091119993, 0.1638244503537113, -0.18794693967432427, 0.2511627529340525,
    0.030899500501083046, 0.36229687915695175, 0.5549952130376704, 0.3831282022675675, -0.19893180150570464, 0.15551011510392737,
    0.21327861871580342, 0.001994434979732584, 0.07267148814764829, -0.22433302879837397, -0.3458822830345247, 0.40697005885684095,
    0.09749858745398242, 0.2290950278591811, 0.09417229695392135, 0.1151719733460463, -0.6699133014853329, -0.458815808182032,
    0.08143476147333627, 0.04048670790579425, -0.331642301484998, -0.08455396999775112, -0.2891050945123047, -0.1513881454734251,
    -0.041241808520894485, -0.2830509982960255, 0.1723441851182197, -0.21837849342685528, 0.09627780481064194, 0.040920937679696484,
    0.12790662787278767, 0.2240377630269268, -0.12717556322390391, 0.043155865678407915, -0.06770415865213655, 0.3437614501147969,
    0.1367206214371773, -0.08225892710520082, 0.24387442850042354, 0.1516468286317715, -0.08485237351845158, -0.33836150468831333,
    0.12464445957811432, -0.024872283611177184, -0.09777648630261378, 0.7178561678195667, -0.22139385479561616, -0.07772543571596027,
    0.09374532902920805, -0.22327450308812144, 0.024417339579485917, -0.14770548227473485, 0.07257772891406902, 0.6309167426698543,
    -0.18837066799961122, -0.31811272384914685, -0.24523019540211863, -0.19376873586923335, -0.18882988103609505, -0.08143411761212131,
    0.29754396885899226, -0.5118195356058346, 0.3641006678103025, 0.017086942000172835, 0.07211813385677374, 0.4032143240102358,
    0.05263897934787817, 0.2956094922142869, -0.03356179492098673, -0.15695857741737326, 0.1971094760957916, 0.1688434699383667,
    -0.19306100931416495, -0.10688145037337955, -0.006298886028345174, 0.32606915667007536, 0.4144412635358153, 0.25459468931760504,
    0.08345394580335455, -0.5674633596962187, 0.06546741147004317, -0.1936446283452328, -0.06045602457375778, -0.03481452123376577,
    -0.5510673614429071, 0.3185357323296032, -0.2041767930845368, 0.1355512726009633, -0.3777165172764901, -0.002704168641239047,
    -0.4440534044309376, 0.23671397575842984, 0.020109085733639183, -0.2606113926585473, 0.42534630536662865, 0.1005609200400384,
    0.7086581238633639, -0.01920943535743597, 0.027699586203768164, -0.0013347921918872771, 0.10024663973851634, 0.3504288806737089,
    0.040910247876719166, -0.046620907970061805, -0.03053378782282325, -0.0021729934358552666, -0.24293092512329137, -0.07378248190743453,
    -0.5958429379331951, 0.013136429444337882, 0.3458691814815612, 0.2676093501800753, -0.07461276895061962, 0.12369113972980207,
    -0.07252082335191719, -0.22491154487581405, -0.07877774540481355, 0.09831641465342557, -0.2040046593169135, -0.1594801832165641,
    0.3641219689407062, 0.2087380031535839, 0.05626573679382682, -0.3256933333740168, 0.05402421739671327, -0.2295492022971757,
    0.12343105559716334, 0.1331987883711241, 0.09280243822351018, -0.021019538964758753, 0.10685981540492379, -0.3654551938325757,
    0.0822118732156098, -0.2160231116733382, 0.04427634543639901, -0.03100549935325133, -0.032696482833855366, -0.03501100977821077,
    0.31954138259224846, -0.09199141341191672, -0.01222553906526689, -0.21191835078271223, -0.10317535931340492, -0.2103080010618822,
    0.4245072907275532, -0.3912530838495437, -0.06482146692275834, 0.21323835907590644, 0.0584053251077174, -0.20043650681573258,
    -0.20252168627204817, 0.2727336354300832, 0.15125896517127593, 0.02765909345438135, 0.10211802125524262, -0.26295041392050894,
    -0.21088698773279263, 0.054083118477244674, -0.48488444100135586, 0.16803984049771048, -0.060755966195174996, -0.24804733197646583,
    0.14068987397107066, -0.025539232800409508, -0.31110505063548693, -0.02248482707161515, -0.27619650821351754, 0.0029593795831406875,
    -0.30839680164005384, 0.024428639983886273, -0.06632396496851384, -0.08327075250306168, 0.021653416627527688, -0.08072636723234489,
    0.04100339259732214, -0.05001731438270259, -0.2642958633837746, 0.06861150508701767, 0.24842271572807484, 0.15999763727993271,
    -0.32533456377315, -0.22846280860862642, -0.358804329627783, 0.09716245050337041, 0.2583326906317392, 0.15105759871411678,
    -0.0009137027742872541, -0.1309933860301617, -0.033842423394128095, 0.01299679650973454, -0.23413804684056921, -0.29815821462483955,
    -0.22935818190527235, -0.09241258054054269, -0.30396625156643414, -0.030893854043986972, 0.08054809133872493, 0.5406613688710411,
    0.2281961345421837, 0.4529963929469023, 0.159819686630015, -0.21759918860202382, 0.22960611786897836, -0.18212450535713048,
    -0.13494880000362613, -0.11743659341173324, -0.29984100302761735, -0.00045356619776863076, 0.13863834310348455, -0.07114816009077989,
    0.2818627821318022, -0.4069306578664357, 0.018290452169489067, -0.17672863094968638, 0.046805982798787706, 0.0922769446978397,
    -0.0810226432514612, -0.4734056974349876, 0.04692353729338616, 0.11200115970742744, 0.046583080897870985, -0.2023199928789858,
    -0.15816045478016402, 0.22577173654490476, -0.35844374633849696, -0.09043808942498563, -0.019600596814235618, 0.1937503807114383,
    0.11364478293747206, 0.1787992835773686, 0.3659917338677591, 0.18447743690898394, 0.31376986710933596, 0.39813037263705103,
    0.19982159702844235, -0.09929377135290658, -0.13372389994039774, -0.061542832284641726, 0.02672678725649236, -0.18383197113269745,
    -0.5234657170900896, -0.019566059656986236, 0.23133356736757846, 0.19168457589892443, 0.4194708489674891, 0.18985102295908862,
    0.14638644721809665, -0.014680522120190668, -0.2683530020769824, 0.19437111787849926, -0.04802249207314809, 0.14991889522258603,
    0.15750162544981958, -0.27010941754092554, -0.06802120017470463, 0.289979754462385, -0.23051377896523273, 0.16602574608208964,
    -0.20361830512453322, -0.02283563716715734, -0.022866257081574966, -0.07188896717486498, -0.30666917821102285, 0.08420771985983179,
    -0.5695997552421954, -0.07359328315994303, -0.039194766765422065, -0.0336503654116299, 0.1340216146545394, -0.3925706982497282,
    -0.33095533657979914, -0.06197600528531213, 0.20547796354779968, 0.016473376744851866, -0.7602696516713338, 0.4230772085872605,
    0.5345786683139983, 0.021615850573815307, -0.3417684844844457, 0.03516898811692685, -0.22029630068399894, 0.12353016198807269,
    -0.0010200483144501805, -0.29784132324883333, 0.036516768788931016, -0.17474703753456994, -0.5861110954150733, -0.07168015707001407,
    -0.12187420989964709, 0.3322010114094616, 0.030270530481051817, -0.1111402912277681, -0.25373306319921446, 0.19987397473161983,
    0.5857196168948745, 0.05492627326010219, -0.2730697011197565, -0.30187983275960506, 0.03316956079764192, 0.3030756250440021,
    -0.09846732553081813, -0.17182758153943406, -0.03129695052900929, 0.2650736962180326, -0.4374829551904403, -0.20042011389489908,
    0.0024862306758755663, 0.5374152687370556, -0.23588598851330153, 0.12219385128987628, 0.3560917984744148, -0.1488299348882503,
    -0.48842332083034745, -0.026434610159933144, -0.10421566144811467, 0.07129249413345833, 0.04226896385654627, 0.04311604957238608,
    0.3850114241080852, 0.26829714793480064, 0.11073129515001062, 0.1934593042580888, 0.5275220021215498, 0.2213766686964729,
    0.17845525091073047, 0.034108046229701476, -0.1280258108941068, 0.09505164899060754, 0.35607927339099915, 0.2170664170105087,
    0.5318844060011519, -0.1607406339104981, -0.6075872936111036, 0.0536099117163081, 0.27382719358164004, -0.06153118273496921,
    -0.2384425432921213, 0.06459247515423089, 0.48188498903219795, -0.22527397907994862, 0.03259801836114206, 0.10944349059320177,
    0.3920752426364677, 0.015592046247736819, 0.04810429093190736, 0.0894979302640955, -0.30000258907545213, 0.21994826747487456,
    -0.1380568457795432, 0.1425394440190067, 0.13877202889052434, 0.43265792592742475, 0.1243503466665012, 0.383613744146924,
    0.03946907118201479, -0.035117274018856566, -0.13693995208111523, -0.08590290352800542, 0.4657170357035368, 0.21889735724117892,
    -0.1411476910872763, 0.22616904610920155, -0.22327952910278281, 0.05803637930255013, -0.31423361065392064, -0.11577668578633181,
    0.29630222111602683, -0.0335593493894587, -0.43078317079102474, -0.5855380822454015, -0.13830545718939158, -0.3473908971432077,
    0.05012032497036859, -0.06567869964633494, 0.26862332245178633, -0.21774556817558577, 0.10883608606850369, -0.016397717528265417,
    0.0813281940483843, -0.17578776367091495, 0.06776006719524631, -0.28654374095778257, 0.06582172241008342, 0.3612033531479088,
    0.46756206730064254, -0.5262091090513611, 0.4520877941750277, -0.3269553055141486, -0.3752587185213037, 0.07637697217038408,
    0.15237695204857124, -0.12461932243511557, -0.2339404760645948, -0.042219550745783034, 0.15123445099818758, 0.1957154489409934,
    -0.26979028405146555, 0.03131831695995715,
]
# <<POLICY_END>>

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

# --------------------------------------------------------------------------
# Neural modulator specification
#
# Output vector layout (all squashed by tanh, so 0 => "keep the tuned value"):
#   [0:10]  additive deltas on the 10 stage reserves
#           (planned|new) x (source, terminal, fab, osat, other)
#   [10:18] additive deltas on 8 scalar parameters, bounded below
#
# value = clip(base + tanh(out) * scale, lo, hi)   (rounded if integer)
# Bounds are widened to include the tuned base value so an extreme tuned
# value is never silently clipped.
# --------------------------------------------------------------------------
_STAGES = ("source", "terminal", "fab", "osat", "other")
_RESERVE_KEYS = tuple(f"{p}_{s}" for p in ("planned", "new") for s in _STAGES)
_N_RES = len(_RESERVE_KEYS)
_RES_SCALE = 0.5

#            name            scale  lo    hi    is_int
_GLOBALS = (
    ("distress_gain",        1.0,  0.0,  3.0,  False),
    ("closure_power",        1.0,  0.0,  4.0,  False),
    ("cost_gain",            0.5,  0.0,  2.0,  False),
    ("end_margin",           3.0,  0.0,  8.0,  True),
    ("hold_below",           0.25, 0.0,  0.95, False),
    ("flush_above",          0.30, 0.05, 1.0,  False),
    ("flush_weeks",          3.0,  0.0,  8.0,  True),
    ("hold_slack",           3.0,  0.0,  10.0, True),
)
_GI = {g[0]: i for i, g in enumerate(_GLOBALS)}

N_FEAT = 16
N_HID = 16
N_OUT = _N_RES + len(_GLOBALS)
N_WEIGHTS = N_FEAT * N_HID + N_HID + N_HID * N_OUT + N_OUT


def init_weights(seed=0, scale=0.1):
    """Random hidden layer, zero output layer: the policy starts exactly at the
    static tuned parameters, and Evolution Strategies moves it from there."""
    rng = np.random.default_rng(seed)
    w = np.zeros(N_WEIGHTS)
    w[: N_FEAT * N_HID] = rng.normal(0.0, scale, N_FEAT * N_HID)
    return w


class Agent:
    def __init__(self, config, _weights=None, _policy=None):
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
        group = []  # index into the 10 (planned|new) x stage reserve groups
        for i, edge in enumerate(slot_edges):
            stage = node_types[edges["tail"][edge]]
            if stage not in {"source", "terminal", "fab", "osat"}:
                stage = "other"
            planned = self.nominal[i] > 0.0
            prefix = "planned" if planned else "new"
            reserve.append(weights[f"{prefix}_{stage}"])
            group.append((0 if planned else len(_STAGES)) + _STAGES.index(stage))
        self.reserve = np.clip(np.asarray(reserve, dtype=np.float64), 0.0, 1.0)
        self.slot_group = np.asarray(group, dtype=np.int64)

        chokepoint_position = {node: i for i, node in enumerate(config["layout"]["chokepoints"])}
        self.n_cp = len(chokepoint_position)
        self.incidence = np.zeros((self.n_slots, max(self.n_cp, 1)), dtype=bool)
        for slot, lane in enumerate(slot_lanes):
            if lane is not None:
                for c in lanes["chokepoints"][lane]:
                    self.incidence[slot, chokepoint_position[c]] = True

        # Deadlines are computed with margin 0; the (dynamic) margin is added per week.
        self.lu0, self.lu_free = self._last_useful_weeks(
            instance, nodes, edges, lanes, commodities, slot_edges, slot_commodities, slot_lanes,
        )
        self.base_margin = int(weights["end_margin"])
        self.lu_base = np.where(self.lu_free, self.T, self.lu0 + self.base_margin)

        if self.n_cp:
            lu = np.where(self.incidence, self.lu0[:, None], -1).max(axis=0)
            self.cp_free = ~self.incidence.any(axis=0)[: self.n_cp]
            self.cp_lu0 = lu[: self.n_cp].astype(np.int64)
        else:
            self.cp_free = np.zeros(0, dtype=bool)
            self.cp_lu0 = np.zeros(0, dtype=np.int64)

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

        self._init_modulator(_policy)
        self.reset()

    # ------------------------------------------------------------------
    # Modulator setup
    # ------------------------------------------------------------------
    def _init_modulator(self, policy):
        base = np.array([float(self.weights[g[0]]) for g in _GLOBALS], dtype=np.float64)
        self.g_base = base
        self.g_scale = np.array([g[1] for g in _GLOBALS], dtype=np.float64)
        self.g_lo = np.minimum(np.array([g[2] for g in _GLOBALS], dtype=np.float64), base)
        self.g_hi = np.maximum(np.array([g[3] for g in _GLOBALS], dtype=np.float64), base)
        self.g_int = np.array([g[4] for g in _GLOBALS], dtype=bool)

        if policy is not None:
            w = np.asarray(policy, dtype=np.float64).ravel()
            if w.size != N_WEIGHTS:
                raise ValueError(f"policy needs {N_WEIGHTS} weights, got {w.size}")
        elif _POLICY_WEIGHTS is not None and np.size(_POLICY_WEIGHTS) == N_WEIGHTS:
            w = np.asarray(_POLICY_WEIGHTS, dtype=np.float64).ravel()
        else:
            w = init_weights()
        self.set_policy(w)

    def set_policy(self, w):
        """Swap in a new flat weight vector (cheap; used by the ES trainer)."""
        w = np.asarray(w, dtype=np.float64).ravel()
        i = 0
        self.W1 = w[i:i + N_FEAT * N_HID].reshape(N_FEAT, N_HID); i += N_FEAT * N_HID
        self.b1 = w[i:i + N_HID]; i += N_HID
        self.W2 = w[i:i + N_HID * N_OUT].reshape(N_HID, N_OUT); i += N_HID * N_OUT
        self.b2 = w[i:i + N_OUT]
        self.policy = w

    # ------------------------------------------------------------------
    # Static precomputation (identical to the baseline except margin = 0)
    # ------------------------------------------------------------------
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

        deadlines = []
        free = []
        for destination, commodity, transit in zip(slot_destinations, slot_commodities, slot_transit, strict=True):
            after_arrival = remaining(destination, commodity)
            if math.isfinite(after_arrival):
                deadlines.append(self.T - transit - int(after_arrival))
                free.append(False)
            else:
                deadlines.append(self.T)
                free.append(True)  # no margin applies (matches baseline)
        return np.asarray(deadlines, dtype=np.int64), np.asarray(free, dtype=bool)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _distress_parts(self, obs):
        def total(key):
            v = obs.get(key)
            return 0.0 if v is None else float(np.nan_to_num(np.asarray(v, dtype=np.float64)).sum())

        lost_share = total("last_week.sinks.lost") / max(total("last_week.sinks.demand"), 1.0)
        shed_share = total("last_week.shed.qty") / max(float(self.grid_load.sum()), 1.0)
        return lost_share, shed_share

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

    # ------------------------------------------------------------------
    # State features + neural modulator
    # ------------------------------------------------------------------
    def reset(self):
        self.was_closed = np.zeros(self.n_cp, dtype=bool)
        self.flush_left = np.zeros(self.n_cp, dtype=np.int64)
        self._prev_week = -1
        # Replace 3-week array with Long-Term EWMA memory
        self.ewma_om = 1.0
        self.ewma_distress = 0.0

    # ------------------------------------------------------------------
    # State features + neural modulator
    # ------------------------------------------------------------------

    def _state(self, week, open_now, observed, lost_share, shed_share, distress, obs):
        if self.n_cp:
            eff = np.where(observed, np.clip(open_now, 0.0, 1.0), 1.0)
            seen = observed.astype(np.float64)
        else:
            eff = np.ones(1)
            seen = np.ones(1)
        om = float(eff.mean())

        totals = self._lot_totals(obs)
        queue_mass = 0.0 if totals is None else float(np.sum(totals)) / max(float(self.capacity.sum()), 1.0)

        c_ratio = 1.0
        if self.slot_c0 is not None and "graph_now.c" in obs:
            c_now = np.asarray(obs["graph_now.c"], dtype=np.float64)
            if c_now.shape[0] > int(self.slot_edge_idx.max()):
                c_ratio = float(np.mean(c_now[self.slot_edge_idx] / self.slot_c0))

        # --- NEW: Extract Predictive Shock Probabilities ---
        warnings = np.asarray(obs.get("warning.score", [0.0]), dtype=np.float64)
        max_warning = float(np.max(warnings)) if warnings.size > 0 else 0.0
        mean_warning = float(np.mean(warnings)) if warnings.size > 0 else 0.0

        # --- NEW: EWMA Long-Term Memory (Alpha = 0.2) ---
        self.ewma_om = 0.2 * om + 0.8 * self.ewma_om
        self.ewma_distress = 0.2 * distress + 0.8 * self.ewma_distress

        x = np.array([
            om, float(eff.min()), float(seen.mean()), float((eff < 0.25).mean()),
            lost_share, shed_share, distress,
            week / max(self.T, 1), float(np.mean(week <= self.lu_base)),
            queue_mass,
            c_ratio,
            self.ewma_om,        # Long-term open trend
            self.ewma_distress,  # Long-term distress trend
            max_warning,         # Highest probability of an imminent shock
            mean_warning,        # General network tension level
            1.0,
        ])
        return np.clip(np.nan_to_num(x), -5.0, 5.0)
    
    def _modulate(self, x):
        """state -> (10 reserve deltas, 8 bounded scalar parameters). Pure matmuls."""
        h = np.tanh(x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        d_res = out[:_N_RES] * _RES_SCALE
        g = np.clip(self.g_base + out[_N_RES:] * self.g_scale, self.g_lo, self.g_hi)
        g = np.where(self.g_int, np.rint(g), g)
        return d_res, g

    # ------------------------------------------------------------------
    # Release logic (baseline logic, parameters now per-week)
    # ------------------------------------------------------------------
    def _release(self, obs, week, open_now, observed, hold_below, flush_above, flush_weeks, hold_slack, margin):
        eff_open = np.where(observed, open_now, 1.0)
        closed = eff_open < hold_below

        reopened = self.was_closed & ~closed
        self.flush_left = np.where(reopened, flush_weeks, np.maximum(self.flush_left - 1, 0))
        self.was_closed = closed

        totals = self._lot_totals(obs)
        queue_cp = self._queue_per_cp(totals)

        cp_last_useful = np.where(self.cp_free, self.T, self.cp_lu0 + margin)
        too_late = week > (cp_last_useful + hold_slack)

        hold = closed | too_late
        flush = ~hold & (eff_open >= flush_above) & (self.flush_left > 0)
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

    # ------------------------------------------------------------------
    # Weekly decision
    # ------------------------------------------------------------------
    def act(self, observation):
        week = int(np.asarray(observation["week"]).item())
        
        # --- NEW DIAGNOSTIC CODE ---
        # if week == 1:
        #     print("\n--- OBSERVATION KEYS AVAILABLE ---")
        #     for key in observation.keys():
        #         # Print the key and the shape/type of its data
        #         val = observation[key]
        #         if isinstance(val, np.ndarray):
        #             print(f"{key}: array of shape {val.shape}")
        #         elif isinstance(val, dict):
        #             print(f"{key}: dict with keys {list(val.keys())}")
        #         else:
        #             print(f"{key}: {type(val)}")
        #     print("----------------------------------\n")
        # ---------------------------
        w = self.weights
        week = int(np.asarray(observation["week"]).item())
        if week <= self._prev_week:  # new episode without an explicit reset()
            self.reset()
        self._prev_week = week

        open_now = np.nan_to_num(np.asarray(observation["graph_now.open"], dtype=np.float64))[: self.n_cp]
        observed = np.asarray(observation["graph_now.open.observed"], dtype=bool)[: self.n_cp]

        lost_share, shed_share = self._distress_parts(observation)
        distress = min(max(lost_share, shed_share, 0.0), 1.0)

        # ---- neural modulation: this week's parameters ----
        x = self._state(week, open_now, observed, lost_share, shed_share, distress, observation)
        d_res, g = self._modulate(x)
        gain = float(g[_GI["distress_gain"]])
        power = float(g[_GI["closure_power"]])
        cost_gain = float(g[_GI["cost_gain"]])
        margin = int(g[_GI["end_margin"]])

        reserve = np.clip(self.reserve + d_res[self.slot_group], 0.0, 1.0)
        if gain != 0.0:
            reserve = reserve + gain * distress

        if cost_gain > 0.0 and self.slot_c0 is not None and "graph_now.c" in observation:
            c_now = np.asarray(observation["graph_now.c"], dtype=np.float64)
            if c_now.shape[0] > int(self.slot_edge_idx.max()):
                ratio = c_now[self.slot_edge_idx] / self.slot_c0
                reserve = reserve * np.clip(1.0 - cost_gain * np.maximum(ratio - 1.0, 0.0), 0.0, 1.0)

        reserve = np.clip(reserve, 0.0, 1.0)
        flows = self.nominal + reserve * (self.capacity - self.nominal)
        last_useful = np.where(self.lu_free, self.T, self.lu0 + margin)
        flows = np.where(week <= last_useful, flows, 0.0)

        if w["use_cap"]:
            u_now = np.asarray(observation["graph_now.u"], dtype=np.float64)
            if u_now.shape[0] == self.n_edges:
                u_pad = np.append(u_now, np.inf)
                flows = np.minimum(flows, u_pad[self.path_idx].min(axis=1))

        if power > 0.0 and self.n_cp:
            factor = np.where(observed, np.maximum(open_now, 0.0) ** power, 1.0)
            flows = flows * np.where(self.incidence[:, : self.n_cp], factor[None, :], 1.0).prod(axis=1)

        flows = np.clip(np.nan_to_num(flows), 0.0, self.capacity)
        action = {"flows": flows * observation["action_mask"]}
        if w["use_release"] and self.n_cp:
            action.update(self._release(
                observation, week, open_now, observed,
                float(g[_GI["hold_below"]]), float(g[_GI["flush_above"]]),
                int(g[_GI["flush_weeks"]]), int(g[_GI["hold_slack"]]), margin,
            ))
        return action