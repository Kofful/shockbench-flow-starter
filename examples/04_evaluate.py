"""Score an agent, or compare two, from Python: the functions behind ``sbf evaluate`` and ``sbf compare``.

    uv run python examples/04_evaluate.py
    uv run python examples/04_evaluate.py --agent=heuristic --against=template
    uv run python examples/04_evaluate.py --agent=agents/mine/agent.py    # in this process: a debugger works
    uv run python examples/04_evaluate.py --task=small --quick

The first run on a network computes the reference costs and caches them; later runs play only your agent.
"""

import fire

from sbf_starter import scoring


def main(
    agent: str = "template",
    against: str | None = None,
    task: str = "tiny",
    episodes: str | int | list[int] = "dev",
    quick: bool = False,
    entropy: int = 0,
    cpu_budget: bool = False,
    n_jobs: int = -1,
    batch_size: int = 32,
    device: str = "auto",
) -> None:
    """Print the score of ``agent``, or its paired comparison with ``against``.

    Args:
        agent: an agent's name, a submission folder or zip, or an agent.py.
        against: another agent to compare with, on the same episodes.
        task: tiny, small or full.
        episodes: dev (20 episodes, 5 per harm level), a count k (episodes 0..k-1) or a list.
        quick: seconds, not the leaderboard's numbers.
        entropy: 0 for the public dev episodes; any other integer for scenarios of your own.
        cpu_budget: a week over the task's CPU budget is played by the naive rule, as on the server.
        n_jobs: workers for references and ordinary agent episodes (-1: all cores).
        batch_size: simultaneous episodes when Agent implements act_batch.
        device: local policy inference device: auto, cpu, cuda or cuda:N.

    """
    options = {
        "quick": quick,
        "entropy": entropy,
        "cpu_budget": cpu_budget,
        "n_jobs": n_jobs,
        "batch_size": batch_size,
        "device": device,
    }
    if against is None:
        result = scoring.evaluate(agent, task, episodes, **options)
        print(result)
        print(f"the score alone: {result.rss}")
    else:
        result = scoring.compare(agent, against, task, episodes, **options)
        print(result)
        print(f"the difference alone: {result.diff} (its interval {result.interval})")


if __name__ == "__main__":
    fire.Fire(main)
