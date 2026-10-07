"""Bridge script to train the neural modulator agent using SBF's evaluation engine."""
import argparse
import sys
import os

# Ensure current folder is on path
sys.path.insert(0, os.getcwd())

import agent as A
from es_train import es_optimize, write_policy_block
from sbf_starter.scoring import episode_set

def main():
    parser = argparse.ArgumentParser(description="Train the neural modulator via ES")
    parser.add_argument("--task", default="small", choices=["tiny", "small", "full"], help="Benchmark task")
    parser.add_argument("--iters", type=int, default=50, help="Number of ES iterations")
    parser.add_argument("--pop", type=int, default=16, help="Population size (must be even)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--sigma", type=float, default=0.05, help="ES noise standard deviation")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization penalty")
    
    # Optional arguments to match command-line structure
    parser.add_argument("--cmd", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cwd", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--score-regex", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--minimize", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--resume", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default="best_policy.json", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bake", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-bake", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    print(f"Building evaluation episode set for task={args.task}...")
    
    # MOVED OUTSIDE: We build a larger set of 40 episodes exactly ONCE.
    # This prevents overfitting but keeps training fast because the LP solver only runs once.
    es = episode_set(args.task, episodes=40, quick=True, entropy=42, verbose=True)

    def evaluate(weights):
        class NeuralAgent(A.Agent):
            def __init__(self, config):
                super().__init__(config, _weights={"use_release": 0})
                self.set_policy(weights)

        score_result = es.score(NeuralAgent, name="NeuralAgent")
        return float(score_result.rss)

    print("Starting Evolution Strategies optimization...")
    best_weights, best_score = es_optimize(
        evaluate, 
        iters=args.iters, 
        pop=args.pop, 
        sigma=args.sigma,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed, 
        log=print
    )

    print(f"\nOptimization complete! Best score achieved: {best_score:.4f}")
    print("Writing optimized weights back into agent.py...")
    write_policy_block("agent.py", best_weights)
    print("Done! agent.py is now updated with the trained neural policy.")

if __name__ == "__main__":
    main()