"""Train PPO (Stable-Baselines3) on a ShockBench-Flow network and export it as a TorchScript submission.

Install the rl extra first: ``uv sync --extra rl`` (Stable-Baselines3 and torch). Then:

    uv run python scripts/python/05_train_ppo.py                     # Tiny, about a minute on a laptop CPU
    uv run python scripts/python/05_train_ppo.py task=small total_timesteps=2000000 n_envs=8 n_scenarios=512
    uv run python scripts/python/05_train_ppo.py net_arch=[256,256] activation=relu run_name=wide

What it does (``sbf_starter.ppo``, whose docstring has the details; every setting is in configs/05_train_ppo.yaml):
draws training scenarios from your own root, wraps the environment for RL (actions as fractions of capacity, the
reward in naive's weeks, the padded lists dropped, one flat vector), trains PPO with an MLP policy, exports the
deterministic policy with the observation normaliser as ``submission/`` of the run folder (``policy.pt``, loaded by
``torch.jit.load``, which the scoring image can run), and checks that the exported agent acts as the SB3 policy on
every week of ``check_episodes`` episodes.

Then check and score the submission like any other, e.g. ``uv run sbf check outputs/05_train_ppo/<run>/submission``
and ``uv run sbf compare outputs/05_train_ppo/<run>/submission template``.
"""

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from helper.logging import logger
from sbf_starter import runs


@hydra.main(version_base=None, config_path="../../configs", config_name="05_train_ppo")
def main(cfg: DictConfig) -> None:
    """Train, export, write and check the submission in the run folder; log the summary (also summary.json)."""
    try:
        from sbf_starter import ppo
    except ModuleNotFoundError as err:
        raise SystemExit(f"{err.name} is missing: install the rl extra first, uv sync --extra rl") from None

    out = runs.start(cfg)
    summary = ppo.train(OmegaConf.to_container(cfg, resolve=True), out)
    logger.info("summary:\n{}", json.dumps(summary, indent=1))
    logger.info("next: uv run sbf check {} --task={}", runs.shown(summary["submission"]), cfg.task.name)
    runs.finish(out, submission=summary["submission"], timesteps=summary["timesteps"])


if __name__ == "__main__":
    main()
