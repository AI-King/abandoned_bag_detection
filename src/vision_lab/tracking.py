"""Explicit MLflow runs, including local artifact storage."""

import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

from vision_lab.config import workspace
from vision_lab.provenance import git_metadata


def tracking_uri() -> str:
    database = workspace() / "mlflow.db"
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{database.as_posix()}")


@contextmanager
def tracked_run(kind: str, name: str):
    mlflow.set_tracking_uri(tracking_uri())
    experiment_name = f"vision-lab-{kind}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = None
        if tracking_uri().startswith("sqlite:"):
            artifact_location = (workspace() / "mlartifacts" / kind).as_uri()
        try:
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise
            experiment_id = experiment.experiment_id
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as run:
        mlflow.set_tags({**git_metadata(), "pipeline": kind})
        lock = Path("uv.lock")
        if lock.exists():
            mlflow.log_artifact(str(lock), "environment")
        yield run
