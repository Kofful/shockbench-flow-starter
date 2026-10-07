#!/usr/init/env python3
import argparse
import functools
import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.getcwd())
from agent import _NEW_DEFAULTS, _TINY_WEIGHTS, _WEIGHTS 

RESERVE_KEYS = [
    "planned_source", "planned_terminal", "planned_fab", "planned_osat", "planned_other",
    "new_source", "new_terminal", "new_fab", "new_osat", "new_other",
]
BOUNDS = {k: (0.0, 1.0) for k in RESERVE_KEYS}
BOUNDS.update({
    "distress_gain": (0.0, 2.0), "closure_power": (0.0, 2.0), "cost_gain": (0.0, 2.0),
    "hold_below": (0.0, 0.6), "flush_above": (0.2, 1.0),
    "end_margin": (0, 3), "flush_weeks": (1, 6), "hold_slack": (0, 6),
})
INTS = {"end_margin", "flush_weeks", "hold_slack"}

def mock_score(weights, task, entropy):
    rng = np.random.default_rng(entropy)
    target = {k: 0.3 + 0.4 * ((i * 7919) % 10) / 10 for i, k in enumerate(RESERVE_KEYS)}
    err = np.mean([(weights[k] - target[k] + 0.05 * rng.standard_normal()) ** 2 for k in RESERVE_KEYS])
    err += 0.02 * (weights["closure_power"] - 0.8) ** 2
    return 0.6 - 0.5 * err + 0.03 * rng.standard_normal()

@functools.lru_cache(maxsize=None)
def resolve_scorer(spec):
    if spec == "mock":
        return mock_score
    module, func = spec.split(":")
    return getattr(importlib.import_module(module), func)

def _job(job):
    spec, task, wjson, seed = job
    return float(resolve_scorer(spec)(json.loads(wjson), task, int(seed)))

class Evaluator:
    def __init__(self, spec, task, workers):
        self.spec, self.task, self.workers = spec, task, max(1, workers)
        self.cache = {}
        self.n_calls, self.elapsed = 0, 0.0
        self.pool = ProcessPoolExecutor(self.workers) if self.workers > 1 else None

    def run(self, weight_list, seeds):
        wjs = [json.dumps(w, sort_keys=True) for w in weight_list]
        todo = [(wj, int(s)) for wj in dict.fromkeys(wjs) for s in seeds if (wj, int(s)) not in self.cache]
        if todo:
            jobs = [(self.spec, self.task, wj, s) for wj, s in todo]
            t0 = time.time()
            results = self.pool.map(_job, jobs) if self.pool else map(_job, jobs)
            for key, val in zip(todo, results):
                self.cache[key] = val
            self.elapsed += time.time() - t0
            self.n_calls += len(todo)
        return np.array([[self.cache[(wj, int(s))] for s in seeds] for wj in wjs])

    def close(self):
        if self.pool:
            self.pool.shutdown()

def make_space(args):
    base = dict(_NEW_DEFAULTS)
    base.update(_TINY_WEIGHTS if args.task == "tiny" else _WEIGHTS)
    if getattr(args, "init", None):
        base.update(json.load(open(args.init)))
    base["use_cap"] = int(args.use_cap)
    base["use_release"] = int(args.use_release)
    names = RESERVE_KEYS + ["distress_gain", "closure_power", "end_margin"]
    if args.tune_cost:
        names.append("cost_gain")
    if args.use_release:
        names += ["hold_below", "flush_above", "flush_weeks", "hold_slack"]
    return base, names

def encode(base, names):
    return np.array([(base[n] - BOUNDS[n][0]) / (BOUNDS[n][1] - BOUNDS[n][0]) for n in names], dtype=float)

def decode(u, base, names):
    w = dict(base)
    for n, x in zip(names, np.clip(u, 0.0, 1.0)):
        lo, hi = BOUNDS[n]
        v = lo + x * (hi - lo)
        w[n] = int(round(v)) if n in INTS else float(v)
    return w

def robust(scores, lam):
    return float(np.mean(scores) - lam * np.std(scores))

def parse_range(text):
    a, b = text.split("-")
    return list(range(int(a), int(b) + 1))

def inject_weights_into_agent(overrides):
    """Directly rewrite agent.py so the weights are hardcoded inside it."""
    agent_path = "agent.py"
    if not os.path.exists(agent_path):
        print("Warning: agent.py not found in current directory for weight injection.")
        return
    
    with open(agent_path, "r") as f:
        content = f.read()

    # Format dictionary nicely for code insertion
    formatted_dict = "[\n"
    for k, v in sorted(overrides.items()):
        formatted_dict += f"    {repr(k)}: {repr(v)},\n"
    formatted_dict += "]"
    
    # We look for the marker block in agent.py and replace it
    start_marker = "_TUNED_OVERRIDES = {"
    end_marker = "}"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Error: Could not find injection marker in agent.py")
        return
        
    # Find matching closing brace after start marker
    brace_count = 0
    end_idx = -1
    for i in range(start_idx + len(start_marker) - 1, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
                
    if end_idx == -1:
        print("Error: Could not parse block bounds in agent.py")
        return

    new_block = "_TUNED_OVERRIDES = " + json.dumps(overrides, indent=4)
    new_content = content[:start_idx] + new_block + content[end_idx:]
    
    with open(agent_path, "w") as f:
        f.write(new_content)
    print("Successfully injected optimized weights directly into agent.py!")

def cmd_train(args):
    rng = np.random.default_rng(args.seed)
    base, names = make_space(args)
    d = len(names)
    u0 = encode(base, names)
    train_pool = parse_range(args.train_seeds)
    val_seeds = parse_range(args.val_seeds)
    ev = Evaluator(args.scorer, args.task, args.workers)

    def val_obj(u):
        s = ev.run([decode(u, base, names)], val_seeds)[0]
        return robust(s, args.lam), float(s.mean()), float(s.std())

    base_obj, base_mean, base_std = val_obj(u0)
    print(f"[start] {d} tuned params | val mean={base_mean:.4f} std={base_std:.4f} robust={base_obj:.4f}")
    best_u, best_obj, best_gen = u0.copy(), base_obj, 0
    m, sigma, stale = u0.copy(), args.sigma, 0
    history = []

    for g in range(1, args.gens + 1):
        seeds = rng.choice(train_pool, args.batch, replace=False) 
        eps = rng.standard_normal((args.pop // 2, d))
        cand = np.clip(np.vstack([m + sigma * eps, m - sigma * eps]), 0.0, 1.0)
        scores = ev.run([decode(c, base, names) for c in cand], seeds)
        fit = scores.mean(1) - args.lam * scores.std(1) - args.reg * ((cand - u0) ** 2).sum(1)
        ranks = fit.argsort().argsort() / (len(fit) - 1) - 0.5 
        grad = (ranks[:, None] * (cand - m) / sigma).sum(0) / len(fit)
        m = np.clip(m + args.lr * grad, 0.0, 1.0)
        sigma = max(args.sigma_min, sigma * args.sigma_decay)

        line = f"[gen {g:3d}] train mean={scores.mean():.4f} best-cand={fit.max():.4f} sigma={sigma:.3f}"
        if g % args.val_every == 0 or g == args.gens:
            obj, mean, std = val_obj(m)
            line += f" | val mean={mean:.4f} std={std:.4f} robust={obj:.4f}"
            history.append({"gen": g, "val_mean": mean, "val_std": std, "val_robust": obj})
            if obj > best_obj + 1e-9:
                best_u, best_obj, best_gen, stale = m.copy(), obj, g, 0
                line += "  *best*"
            else:
                stale += 1
        print(line)
        if stale >= args.patience:
            print(f"[stop] no validation gain for {args.patience} checks")
            break

    best_w = decode(best_u, base, names)
    obj, mean, std = val_obj(best_u)
    overrides = {k: best_w[k] for k in base if best_w[k] != base.get(k) or k in names or k in ("use_cap", "use_release")}
    
    result = {"task": args.task, "best_gen": best_gen, "val_mean": mean, "val_std": std,
              "val_robust": obj, "start_val_mean": base_mean, "weights": overrides, "history": history}
    
    json.dump(result, open(args.out, "w"), indent=2)
    
    # INJECT DIRECTLY INTO AGENT.PY
    inject_weights_into_agent(overrides)
    
    print(f"\n[done] best gen {best_gen}: val mean {base_mean:.4f} -> {mean:.4f} "
          f"({ev.n_calls} episodes, {ev.elapsed / max(ev.n_calls, 1):.2f}s each)")
    ev.close()

def load_weights(spec, args):
    if spec == "baseline": 
        return {}
    data = json.load(open(spec))
    return data.get("weights", data)

def paired_report(a, b, label_a="A", label_b="B", n_boot=4000, seed=0):
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    verdict = "B better" if lo > 0 else "A better" if hi < 0 else "no clear difference"
    print(f"  {label_a}: mean {np.mean(a):.4f} | {label_b}: mean {np.mean(b):.4f}")
    print(f"  paired diff (B-A): {d.mean():+.4f}  SE {se:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  B wins on {np.mean(d > 0) * 100:.0f}% of seeds, ties {np.mean(d == 0) * 100:.0f}%  ->  {verdict}")

def run_pair(args, seeds, wa, wb):
    ev = Evaluator(args.scorer, args.task, args.workers)
    s = ev.run([wa, wb], seeds)
    ev.close()
    return s[0], s[1]

def full_weights(args, overrides):
    base = dict(_NEW_DEFAULTS)
    base.update(_TINY_WEIGHTS if args.task == "tiny" else _WEIGHTS)
    base.update(overrides)
    return base

def cmd_compare(args):
    seeds = parse_range(args.seeds)
    a, b = run_pair(args, seeds, full_weights(args, load_weights(args.a, args)),
                    full_weights(args, load_weights(args.b, args)))
    print(f"Paired comparison on {len(seeds)} seeds ({args.seeds}), task={args.task}")
    paired_report(a, b, args.a, args.b)

def cmd_final(args):
    seeds = parse_range(args.test_seeds)
    a, b = run_pair(args, seeds, full_weights(args, {}), full_weights(args, load_weights(args.weights, args)))
    print(f"FINAL (test pool {args.test_seeds}, task={args.task}) - look at this once")
    paired_report(a, b, "baseline", "tuned")

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--task", default="small", choices=["tiny", "small", "full"])
        sp.add_argument("--scorer", default="mock", help="module:function or 'mock'")
        sp.add_argument("--workers", type=int, default=1)

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--use-cap", action="store_true")
    t.add_argument("--use-release", action="store_true")
    t.add_argument("--tune-cost", action="store_true")
    t.add_argument("--init", help="JSON of starting overrides")
    t.add_argument("--gens", type=int, default=60)
    t.add_argument("--pop", type=int, default=16, help="even number")
    t.add_argument("--batch", type=int, default=16, help="training seeds per generation")
    t.add_argument("--lam", type=float, default=0.5, help="std penalty in the robust objective")
    t.add_argument("--reg", type=float, default=0.02, help="L2 pull toward the starting weights")
    t.add_argument("--lr", type=float, default=0.5)
    t.add_argument("--sigma", type=float, default=0.10)
    t.add_argument("--sigma-min", type=float, default=0.02)
    t.add_argument("--sigma-decay", type=float, default=0.985)
    t.add_argument("--val-every", type=int, default=3)
    t.add_argument("--patience", type=int, default=4)
    t.add_argument("--train-seeds", default="1000-1999")
    t.add_argument("--val-seeds", default="5000-5049")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="best.json")
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("compare")
    common(c)
    c.add_argument("--a", required=True, help="'baseline' or JSON file")
    c.add_argument("--b", required=True)
    c.add_argument("--seeds", default="5000-5049")
    c.set_defaults(fn=cmd_compare)

    f = sub.add_parser("final")
    common(f)
    f.add_argument("--weights", required=True)
    f.add_argument("--test-seeds", default="9000-9049")
    f.set_defaults(fn=cmd_final)

    args = p.parse_args()
    if args.cmd == "train" and args.pop % 2:
        p.error("--pop must be even")
    args.fn(args)

if __name__ == "__main__":
    main()