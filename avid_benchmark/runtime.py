from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

from .types import PredictionRecord, RunDirectories


def create_run_directories(train_output_root: Path, model_name: str, run_name: str) -> RunDirectories:
    root = Path(train_output_root) / model_name / run_name
    checkpoints = root / "checkpoints"
    logs = root / "logs"
    predictions = root / "predictions"
    artifacts = root / "artifacts"
    metrics = root / "metrics"
    cache = root / "cache"
    for path in (root, checkpoints, logs, predictions, artifacts, metrics, cache):
        path.mkdir(parents=True, exist_ok=True)
    return RunDirectories(
        root=root,
        checkpoints=checkpoints,
        logs=logs,
        predictions=predictions,
        artifacts=artifacts,
        metrics=metrics,
        cache=cache,
    )


def create_prediction_output_root(predictions_dir: Path, output_tag: str, split: str) -> Path:
    root = predictions_dir / f"{output_tag}_{split}"
    if root.exists() and any(root.iterdir()):
        import uuid
        root = predictions_dir / f"{output_tag}_{split}_{uuid.uuid4().hex[:12]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def save_jsonl(path: Path, rows: Iterable[dict[str, Any] | PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def snapshot_config(run_dirs: RunDirectories, config: dict[str, Any]) -> Path:
    snapshot_path = run_dirs.logs / "config_snapshot.json"
    if snapshot_path.exists():
        import uuid
        snapshot_path = run_dirs.logs / f"config_snapshot_{uuid.uuid4().hex[:12]}.json"
    save_json(snapshot_path, config)
    return snapshot_path


def save_prediction_result(
    predictions_dir: Path,
    prediction_root: Path,
    result: dict[str, Any],
) -> Path:
    tagged_result_path = prediction_root / "prediction_result.json"
    save_json(tagged_result_path, result)
    save_json(predictions_dir / f"{prediction_root.name}_prediction_result.json", result)
    save_json(predictions_dir / "prediction_result.json", result)
    return tagged_result_path
