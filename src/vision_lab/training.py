"""YOLO fine tuning and evaluation with explicit MLflow lineage."""

import json
import math
import re
import shutil
import uuid
from pathlib import Path

import mlflow

from vision_lab.config import workspace
from vision_lab.dataset_audit import audit_dataset
from vision_lab.provenance import sha256
from vision_lab.tracking import tracked_run


def metric_name(name: str) -> str:
    """Normalize the parentheses in Ultralytics metric names for MLflow."""
    return re.sub(r"[^\w\-./ ]", "_", name)


def train(
    data: Path,
    model: Path,
    epochs: int = 1,
    image_size: int = 320,
    batch: int = 4,
    device: str = "cpu",
) -> Path:
    from ultralytics import YOLO, settings

    if not data.is_file() or not model.is_file():
        raise FileNotFoundError("Dataset YAML or model missing. Run 'vision prepare' first.")
    if epochs <= 0 or batch <= 0:
        raise ValueError("Epochs and batch size must be positive.")
    dataset_manifest = audit_dataset(data)
    # Avoid a second, implicit MLflow run from the Ultralytics integration.
    settings.update({"mlflow": False})
    with tracked_run("training", f"{data.stem}-{epochs}epochs") as run:
        mlflow.log_params(
            {
                "data": str(data),
                "data_sha256": sha256(data),
                "initial_model_sha256": sha256(model),
                "epochs": epochs,
                "image_size": image_size,
                "batch": batch,
                "device": device,
                "seed": 42,
            }
        )
        mlflow.log_artifact(str(data), "dataset")
        mlflow.log_dict(dataset_manifest, "dataset/content_manifest.json")
        manifest = workspace() / "configs/assets.lock.json"
        if manifest.exists():
            mlflow.log_artifact(str(manifest), "dataset")
        yolo = YOLO(str(model))

        def log_epoch(trainer):
            metrics = {
                metric_name(str(k)): float(v)
                for k, v in trainer.metrics.items()
                if math.isfinite(float(v))
            }
            mlflow.log_metrics(metrics, step=trainer.epoch)

        yolo.add_callback("on_fit_epoch_end", log_epoch)
        yolo.train(
            data=str(data.resolve()),
            epochs=epochs,
            imgsz=image_size,
            batch=batch,
            device=device,
            workers=0,
            seed=42,
            deterministic=True,
            project=str(workspace() / "outputs/training"),
            name=uuid.uuid4().hex[:12],
            plots=True,
            verbose=False,
            amp=False,
        )
        save_dir = Path(yolo.trainer.save_dir)
        mlflow.log_artifacts(str(save_dir), "training")
        promoted = workspace() / "models" / f"{run.info.run_id}-best.pt"
        promoted.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(save_dir / "weights/best.pt", promoted)
        promoted.with_suffix(".json").write_text(
            json.dumps(
                {
                    "mlflow_run_id": run.info.run_id,
                    "model_sha256": sha256(promoted),
                    "dataset": str(data),
                    "epochs": epochs,
                    "note": "Candidate checkpoint; evaluate on held-out scenes before deployment.",
                },
                indent=2,
            )
        )
        return promoted


def evaluate(data: Path, model: Path, image_size: int = 640, device: str = "cpu") -> dict:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})
    if not data.is_file() or not model.is_file():
        raise FileNotFoundError("Dataset YAML or model does not exist.")
    dataset_manifest = audit_dataset(data)
    with tracked_run("evaluation", model.stem):
        mlflow.log_dict(dataset_manifest, "dataset/content_manifest.json")
        mlflow.log_params(
            {
                "data": str(data),
                "model_sha256": sha256(model),
                "image_size": image_size,
                "device": device,
            }
        )
        result = YOLO(str(model)).val(
            data=str(data.resolve()),
            imgsz=image_size,
            device=device,
            workers=0,
            project=str(workspace() / "outputs/evaluation"),
            name=uuid.uuid4().hex[:12],
            plots=True,
        )
        metrics = {
            metric_name(k): float(v)
            for k, v in result.results_dict.items()
            if math.isfinite(float(v))
        }
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(result.save_dir), "evaluation")
        return metrics
