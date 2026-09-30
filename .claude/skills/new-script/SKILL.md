---
name: new-script
description:
  Scaffold a new Hydra experiment of the participant's under my/ in this
  ShockBench-Flow starter repository, an entry point in my/scripts/ with a
  paired config in my/configs/ and a smoke test in my/tests/. Use when the user
  asks to add a new script, experiment, training run, search, sweep, runner, or
  "new entrypoint".
---

# new-script

Create a participant's Hydra entry point: `my/scripts/<stem>.py`,
`my/configs/<stem>.yaml` and `my/tests/test_<stem>.py`. Everything under `my/`
is the participant's: the organisers never edit it, so `make update` never
conflicts there. Never add or edit files under `scripts/python/`, `configs/` or
`tests/` (the organisers'); to change an example, copy it to `my/scripts/`.

## Steps

1. **Resolve the name.** Use `$ARGUMENTS` if provided, otherwise ask the user
   for a short name (one line). Slugify to `snake_case` for the file stem; it
   must not be the name of an example config (`01_quickstart` ...), since
   `my/configs/` is on the same search path.
2. **Create the config** `my/configs/<stem>.yaml`:

   ```yaml
   # @package _global_
   # my/scripts/<stem>.py: <one line>
   defaults:
     - main
     - _self_
   # Script-specific parameters go here, each with a short comment.
   ```

   `main` (the organisers' `configs/main.yaml`, on the search path) gives
   `cfg.task.name` (`tiny`, `small`, `full`), `cfg.task.env_id`, `cfg.seed` and
   `cfg.run_name`. Only add parameters the user has described; do not invent
   fields. For a score, use `quick: false` (the one switch for rough settings),
   never the package's replication counts.

3. **Create the script** `my/scripts/<stem>.py`:

   ```python
   """<one-line description>.

       uv run python my/scripts/<stem>.py              # Tiny
       uv run python my/scripts/<stem>.py task=small   # Small, the public board's network
   """

   import gymnasium as gym
   import hydra
   import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
   from omegaconf import DictConfig

   from helper.logging import logger
   from sbf_starter import runs


   @hydra.main(version_base=None, config_path="../configs", config_name="<stem>")
   def main(cfg: DictConfig) -> None:
       """Entry point."""
       out = runs.start(cfg)  # outputs/<stem>/<run_name>/ with config.yaml and meta.json
       env = gym.make(cfg.task.env_id)
       logger.info("{}: seed {}", cfg.task.env_id, cfg.seed)
       # TODO: implement; write files under `out`
       runs.finish(out)  # results as keyword arguments: runs.finish(out, score=0.5)


   if __name__ == "__main__":
       main()
   ```

4. **Add a smoke test** `my/tests/test_<stem>.py`: run the script in a
   subprocess on Tiny with the smallest settings that exercise it, in `tmp_path`
   (its `outputs/` goes there), with `run_name=test`, and assert on
   `tmp_path / "outputs" / "<stem>" / "test"`:

   ```python
   import subprocess
   import sys
   from pathlib import Path


   SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "<stem>.py"


   def test_<stem>(tmp_path):
       proc = subprocess.run(
           [sys.executable, str(SCRIPT), "run_name=test"], cwd=tmp_path, capture_output=True, text=True
       )
       assert proc.returncode == 0, proc.stderr[-3000:]
       assert (tmp_path / "outputs" / "<stem>" / "test" / "meta.json").is_file()
   ```

5. **Run it** once with small settings, then `uv run pytest -n 3 my/tests`, and
   report the three files as links with the run command.

## Notes

- Scoring: `sbf_starter.scoring.evaluate(agent, task, episodes, quick=...)` and
  `compare(a, b, ...)`, or the wheel's `EpisodeSet` directly
  (`from shockbench_flow_agent import EpisodeSet`); agents by name resolve in
  `agents/` first.
- Reusable code of the participant's goes in modules under `my/` (importable as
  `my`, e.g. `from my.features import ...`).
- Use the logger of `helper.logging`; do not instantiate `logging.getLogger`.
- Call `main()` directly under `if __name__ == "__main__":`; never wrap a
  `@hydra.main` function in `fire.Fire(...)` (both parse `sys.argv`). Do not add
  `argparse` or `click`.
- Tune on a root of your own (`entropy` other than 0) and keep the dev episodes
  (root 0) for the final check.
- An agent written by the script must follow the scorer's rules (AGENTS.md,
  "Rules that decide a score"): allowed imports only, shapes from `config`,
  randomness from `config["policy_seed"]`; put it in `agents/<name>/` to submit.
