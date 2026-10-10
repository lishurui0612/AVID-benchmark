from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import PathConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"YAML config requested for {path}, but PyYAML is not installed. "
            "Use JSON or install PyYAML."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _load_config_dict(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix in {".yaml", ".yml"}:
        return _maybe_load_yaml(path)
    raise ValueError(f"Unsupported config file extension for {path}")


def _expand_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is None:
        return path.resolve()
    return (base_dir / path).resolve()


def parse_path_config(data: dict[str, Any], *, base_dir: Path | None = None, repo_root: Path | None = None) -> PathConfig:
    required = [
        "dataset_root",
        "coco_image_root",
        "simple_image_root",
        "used_stimuli_tsv",
        "train_output_root",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required path config keys: {missing}")
    repo_root_path = (repo_root or REPO_ROOT).resolve()
    return PathConfig(
        dataset_root=_expand_path(data["dataset_root"], base_dir),
        coco_image_root=_expand_path(data["coco_image_root"], base_dir),
        simple_image_root=_expand_path(data["simple_image_root"], base_dir),
        used_stimuli_tsv=_expand_path(data["used_stimuli_tsv"], base_dir),
        train_output_root=_expand_path(data["train_output_root"], base_dir),
        asset_root=_expand_path(data["asset_root"], base_dir) if data.get("asset_root") else None,
        taiyi_model_root=_expand_path(data["taiyi_model_root"], base_dir) if data.get("taiyi_model_root") else None,
        repo_root=repo_root_path,
    )


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    path = _expand_path(config_path)
    config = _load_config_dict(path)
    config["config_path"] = str(path)
    config["paths"] = parse_path_config(config.get("paths", {}), base_dir=path.parent, repo_root=REPO_ROOT)
    config.setdefault("train", {})
    config.setdefault("model", {})
    config.setdefault("evaluation", {})
    return config
