"""Training runs recorded in MLflow (a dev dependency): parameters, metrics as they grow, the run folder's files.

The store is ``outputs/mlflow.db`` and the runs' files ``outputs/mlartifacts/`` (gitignored, like every run folder).
Browse it with

    uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db      # then http://localhost:5000

Without mlflow installed, or with ``SBF_MLFLOW=0``, every call does nothing and training runs as before.
"""

import os
from pathlib import Path

from sbf_starter import ROOT


STORE = ROOT / "outputs" / "mlflow.db"
ARTIFACTS = ROOT / "outputs" / "mlartifacts"  # the runs' files


class Tracker:
    """One MLflow run: ``Tracker(experiment, run_name, params)``, then ``log``, ``artifact``, ``end``."""

    def __init__(self, experiment: str, run_name: str, params: dict) -> None:
        self.mlflow = None
        if os.environ.get("SBF_MLFLOW", "1") == "0":
            return
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        try:
            import mlflow
        except ModuleNotFoundError:
            return
        STORE.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{STORE}")
        if mlflow.get_experiment_by_name(experiment) is None:  # its files beside the store, not in ./mlruns
            mlflow.create_experiment(experiment, artifact_location=(ARTIFACTS / experiment).as_uri())
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        self.mlflow = mlflow

    def log(self, metrics: dict, step: int) -> None:
        if self.mlflow is not None:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None}, step=step)

    def artifact(self, path: Path) -> None:
        if self.mlflow is not None and Path(path).is_file():
            self.mlflow.log_artifact(str(path))

    def end(self) -> None:
        if self.mlflow is not None:
            self.mlflow.end_run()
            where = STORE.relative_to(ROOT) if STORE.is_relative_to(ROOT) else STORE
            print(f"recorded in MLflow: uv run mlflow ui --backend-store-uri sqlite:///{where}")
