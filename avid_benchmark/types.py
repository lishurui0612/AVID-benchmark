from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AVIDShardInfo:
    subject_id: int
    stimulus_type: str
    split: str
    tar_path: Path
    available: bool = True


@dataclass
class AVIDSample:
    subject_id: int
    stimulus_type: str
    split: str
    image_name: str
    fmri_responses: Any
    caption_zh: str
    caption_en: str
    gt_image_path: Path


@dataclass
class PredictionRecord:
    model_name: str
    subject_id: int
    stimulus_type: str
    split: str
    sample_id: str
    image_name: str
    gt_image_path: Path
    generated_image_path: Path
    generated_text: str | None = None
    route: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathConfig:
    dataset_root: Path
    coco_image_root: Path
    simple_image_root: Path
    used_stimuli_tsv: Path
    train_output_root: Path
    asset_root: Path | None = None
    taiyi_model_root: Path | None = None
    repo_root: Path | None = None


@dataclass(frozen=True)
class RunDirectories:
    root: Path
    checkpoints: Path
    logs: Path
    predictions: Path
    artifacts: Path
    metrics: Path
    cache: Path
