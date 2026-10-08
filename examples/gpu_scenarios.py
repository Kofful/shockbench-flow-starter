"""Draw and cache episodes for the GPU simulator, with their references (naive and clairvoyant costs) for the score.

    uv run python examples/gpu_scenarios.py                                  # training: small 1024, full 128 of 1003
    uv run python examples/gpu_scenarios.py --small=64 --full=8 --entropy=9001   # the held-out episodes

The marks go to ~/.cache/shockbench-flow/gpu/ (``sbf_starter.gpu.scenarios``), the references to the package's
reference cache (``sbf evaluate`` reads the same ones), so this runs once per root.
"""

import time

import fire

from sbf_starter import scoring
from sbf_starter.gpu import scenarios


def main(small: int = 1024, full: int = 128, entropy: int = 1003, n_jobs: int = 6) -> None:
    for task, n in (("small", small), ("full", full)):
        if n <= 0:
            continue
        start = time.perf_counter()
        scenarios.build(task, entropy, n, n_jobs=n_jobs)
        print(f"{task}: {n} episodes' marks ready ({time.perf_counter() - start:.0f} s)", flush=True)
        start = time.perf_counter()
        scoring.episode_set(task, n, entropy=entropy, n_jobs=n_jobs)
        print(f"{task}: {n} episodes' references ready ({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
