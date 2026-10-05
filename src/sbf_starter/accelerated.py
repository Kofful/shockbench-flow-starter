"""Fast local agent play: process-parallel episodes, or lock-step batched inference.

The benchmark simulator is NumPy/SciPy code and cannot use CUDA.  A policy can,
however, expose this optional method on its ``Agent`` class::

    @staticmethod
    def act_batch(agents, observations):
        ...
        return actions

``agents`` contains one initialized Agent per episode and ``observations`` has
the corresponding observations.  This module advances those episodes in lock
step, allowing one Torch forward pass per week instead of one per episode.
Agents without the method use EpisodeSet's process-parallel implementation.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np


DEVICE_ENV = "SBF_EVAL_DEVICE"
DEVICES = ("auto", "cpu", "cuda")


def check_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    return batch_size


def check_device(device: str) -> str:
    if not isinstance(device, str):
        raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {device!r}")
    value = device.strip().lower()
    if value in DEVICES:
        return value
    if value.startswith("cuda:") and value[5:].isdigit():
        return value
    raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {device!r}")


@contextlib.contextmanager
def evaluation_device(device: str) -> Iterator[str]:
    """Publish the requested inference device to agents without changing their scored config."""
    value = check_device(device)
    previous = os.environ.get(DEVICE_ENV)
    os.environ[DEVICE_ENV] = value
    try:
        yield value
    finally:
        if previous is None:
            os.environ.pop(DEVICE_ENV, None)
        else:
            os.environ[DEVICE_ENV] = previous


def _batch_actions(shims, observations) -> list[dict | None]:
    """Convert a class's batched Dict actions to wire actions, preserving shim failure rules."""
    from shockbench_flow_agent.convert import action_to_wire, observation_dict

    actions: list[dict | None] = [None] * len(shims)
    ready = []
    arrays = []
    for i, (shim, obs) in enumerate(zip(shims, observations, strict=True)):
        if shim.agent is None:
            continue
        try:
            arrays.append(observation_dict(shim.layout, obs))
            ready.append(i)
        except (Exception, SystemExit):  # the same per-week failure rule as AgentShim._act
            shim._fail(obs["week"], "the flat view")
    if not ready:
        return actions

    try:
        leader = shims[ready[0]].agent
        produced = list(leader.act_batch([shims[i].agent for i in ready], arrays))
        if len(produced) != len(ready):
            raise ValueError(f"act_batch returned {len(produced)} actions for {len(ready)} episodes")
    except (Exception, SystemExit):
        for i in ready:
            shims[i]._fail(observations[i]["week"], "act_batch")
        return actions

    for i, action in zip(ready, produced, strict=True):
        try:
            actions[i] = action_to_wire(shims[i].layout, observations[i]["week"], action)
        except (Exception, SystemExit):
            shims[i]._fail(observations[i]["week"], "act_batch action")
    return actions


def _rows(factory, sha256: str, episode_set, episodes: Sequence[int], batch_size: int) -> list[dict]:
    from shockbench_flow.dynamics.env import Env, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.scoring import _policy_seed, _world
    from shockbench_flow_agent.shim import AgentShim

    task, entropy, regime, fq_replications, cache = episode_set._spec
    rows = []
    for offset in range(0, len(episodes), batch_size):
        ns = episodes[offset : offset + batch_size]
        envs, shims, observations, worlds = [], [], [], []
        started = time.perf_counter()
        with without_secret_like():
            for n in ns:
                inst, omega, marks, fallback = _world(task, entropy, n, fq_replications, cache)
                seed = _policy_seed(entropy, n, sha256)
                env = Env(fallback=fallback)
                obs, info = env.reset(inst, regime, omega, seed, marks=marks, policy_name="submission")
                shim = AgentShim(factory)
                shim.reset(info["static"], obs, seed)
                envs.append(env)
                shims.append(shim)
                observations.append(obs)
                worlds.append((inst, omega))

            # All scenarios of a task have the same horizon. Keeping them in lock step is what makes policy
            # inference batchable while each simulator state and Agent instance remains independent.
            for _week in range(worlds[0][0].T):
                actions = _batch_actions(shims, observations)
                for i, (env, action) in enumerate(zip(envs, actions, strict=True)):
                    observations[i], _reward, _done, _truncated, _info = env.step(action)

        seconds = (time.perf_counter() - started) / len(ns)
        for n, env, shim, (inst, omega) in zip(ns, envs, shims, worlds, strict=True):
            traj = env.trajectory
            fell = sum(took_fallback(record) for record in traj.records)
            rows.append(
                {
                    "episode": n,
                    "J_policy_cents": traj.J_cents,
                    "fallback_weeks": fell,
                    "cpu_weeks": 0,
                    "invalid_entries": sum(len(r.invalid) for r in traj.records if not took_fallback(r)),
                    "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
                    "weeks": inst.T,
                    "seconds": round(seconds, 3),
                    "omega_hash": omega.hash,
                }
            )
    return rows


def play_batched(episode_set, agent: object, *, batch_size: int) -> list[dict] | None:
    """Play with ``Agent.act_batch``; return None when the agent does not implement it.

    Resolution, validation and seeding are the same operations used by ``EpisodeSet.play``.  Batching deliberately
    does not implement the per-week CPU meter: callers route metered runs through the reference evaluator instead.
    """
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256, agent_file, resolve_agent
    from shockbench_flow_agent.shim import load_agent_class, unload_agent
    from shockbench_flow_agent.submission import Submission

    size = check_batch_size(batch_size)
    with tempfile.TemporaryDirectory(prefix="sbf-batch-") as work:
        file = agent_file(agent)
        resolved = None if file is not None else resolve_agent(agent, Path(work))[0]
        if file is not None:
            target, root, sha = None, str(file), NO_ZIP_SHA256
        elif isinstance(resolved, Submission):
            target, root, sha = None, str(resolved.root), resolved.sha256
        else:
            target, root, sha = resolved, None, NO_ZIP_SHA256
        try:
            factory = target if root is None else load_agent_class(root, f"submission_{Path(root).stem}")
            if not callable(getattr(factory, "act_batch", None)):
                return None
            rows = _rows(factory, sha, episode_set, list(episode_set.episodes), size)
        finally:
            if root is not None:
                unload_agent()

    for ref, row in zip(episode_set.references, rows, strict=True):
        if ref["omega_hash"] != row["omega_hash"]:
            raise RuntimeError(f"episode {ref['episode']}: the cached reference was computed on another scenario")
    return rows


def score_one(
    episode_set,
    agent: object,
    *,
    name: str | None,
    cpu_budget: bool | float | None,
    n_jobs: int,
    batch_size: int,
):
    """Choose batched inference when supported, otherwise EpisodeSet's process batching."""
    from shockbench_flow_agent.local_eval import _agent_name
    from shockbench_flow_agent.scoring import N_BOOT, _boot_stats

    rows = None if cpu_budget or batch_size == 1 else play_batched(episode_set, agent, batch_size=batch_size)
    if rows is None:
        return episode_set.score(agent, name=name, cpu_budget=cpu_budget, n_jobs=n_jobs)

    pooled = episode_set.rss([r["J_policy_cents"] for r in rows])["pooled"]
    (boot,) = _boot_stats(
        episode_set.references,
        [[r["J_policy_cents"] for r in rows]],
        pooled,
        N_BOOT,
        0,
    )
    if name is None:
        name = str(agent) if isinstance(agent, (str, os.PathLike)) else _agent_name(agent)
    return episode_set._score(str(name), rows, episode_set._budget(cpu_budget), 0.90, boot)


def comparison(episode_set, a, b, *, names, cpu_budget, n_jobs, batch_size):
    """The package's paired Comparison, with either score allowed to use batched inference."""
    from shockbench_flow_agent.scoring import N_BOOT, Comparison, _boot_stats, _interval

    na, nb = names
    sa = score_one(episode_set, a, name=na, cpu_budget=cpu_budget, n_jobs=n_jobs, batch_size=batch_size)
    sb = score_one(episode_set, b, name=nb, cpu_budget=cpu_budget, n_jobs=n_jobs, batch_size=batch_size)
    pooled = sa.pooled and sb.pooled
    ba, bb = _boot_stats(episode_set.references, [sa.J_cents, sb.J_cents], pooled, N_BOOT, 0)
    difference = ba - bb
    finite = difference[np.isfinite(difference)]
    diff = None if sa.rss is None or sb.rss is None or sa.pooled != sb.pooled else sa.rss - sb.rss
    probability = float((finite > 0).mean()) if finite.size else None
    return Comparison(sa, sb, diff, _interval(difference, 0.90), 0.90, probability)


__all__ = [
    "DEVICE_ENV",
    "check_batch_size",
    "check_device",
    "comparison",
    "evaluation_device",
    "play_batched",
    "score_one",
]
