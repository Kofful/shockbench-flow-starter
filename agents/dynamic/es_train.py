"""Offline Evolution Strategies trainer for agent.py's neural modulator.

Not part of the submission. Usage sketch (adapt `run_episodes` to your runner):

    import agent as A
    from es_train import es_optimize, write_policy_block

    ag = A.Agent(config, _weights={"use_release": 1})   # build once (heavy precompute)

    def evaluate(w):
        ag.set_policy(w)                 # cheap weight swap, no re-init
        return run_episodes(ag)          # mean benchmark score, HIGHER is better,
                                         # use the SAME episode seeds for every call

    w, score = es_optimize(evaluate, iters=150, pop=32)
    write_policy_block("agent.py", w)

Note: with `use_release` = 0 (current tuned value) the 4 release outputs
(hold_below, flush_above, flush_weeks, hold_slack) have no effect on the
score, so ES simply leaves them alone. Set it to 1 to let ES tune them.
"""
import re
import numpy as np

import agent as A


def es_optimize(evaluate, iters=150, pop=32, sigma=0.05, lr=0.03, seed=0,
                w0=None, l2=1e-3, log=print):
    """Antithetic OpenAI-ES with centred-rank fitness shaping.

    Starts from the zero-output policy (== your static tuned agent), so the
    best score can never fall below the baseline: the initial centre is
    evaluated and only replaced when a later centre scores higher.
    """
    rng = np.random.default_rng(seed)
    w = A.init_weights(seed) if w0 is None else np.array(w0, dtype=np.float64)
    best_w, best_f = w.copy(), float(evaluate(w))
    log(f"baseline score {best_f:.4f}")
    half = pop // 2
    for it in range(iters):
        eps = rng.standard_normal((half, w.size))
        eps = np.concatenate([eps, -eps])
        f = np.array([evaluate(w + sigma * e) for e in eps], dtype=np.float64)
        ranks = np.empty(len(f))
        ranks[np.argsort(f)] = np.arange(len(f))
        shaped = ranks / (len(f) - 1) - 0.5
        grad = (shaped @ eps) / (len(f) * sigma)
        w = w + lr * grad - l2 * w
        centre = float(evaluate(w))
        if centre > best_f:
            best_f, best_w = centre, w.copy()
        log(f"iter {it:3d}  pop mean {f.mean():9.4f}  centre {centre:9.4f}  best {best_f:9.4f}")
    return best_w, best_f


def write_policy_block(path, w):
    """Rewrite the <<POLICY_BEGIN>>/<<POLICY_END>> block in agent.py in place."""
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size != A.N_WEIGHTS:
        raise ValueError(f"expected {A.N_WEIGHTS} weights, got {w.size}")
    body = ",\n".join("    " + ", ".join(repr(float(v)) for v in w[i:i + 6])
                      for i in range(0, w.size, 6))
    block = f"# <<POLICY_BEGIN>>\n_POLICY_WEIGHTS = [\n{body},\n]\n# <<POLICY_END>>"
    with open(path) as fh:
        src = fh.read()
    new, n = re.subn(r"# <<POLICY_BEGIN>>.*?# <<POLICY_END>>", lambda _m: block, src, flags=re.S)
    if n != 1:
        raise RuntimeError("policy markers not found exactly once in " + path)
    with open(path, "w") as fh:
        fh.write(new)