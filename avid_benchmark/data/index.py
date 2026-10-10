from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..types import AVIDShardInfo, PathConfig


SHARD_PATTERNS = (
    re.compile(r"sub-(?P<subject>\d+)_(?P<stimulus>complex|simple)_(?P<split>train|val)\.tar$"),
    re.compile(r"S(?P<subject>\d+)_(?P<stimulus>main)_(?P<split>train|val)\.tar$"),
)


class AVIDMetadataIndex:
    def __init__(self, paths: PathConfig):
        self.paths = paths
        self.rows_by_image: dict[str, dict[str, str]] = {}
        with paths.used_stimuli_tsv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                image_name = row["image"]
                gt_root = paths.simple_image_root if image_name.startswith("simple_") else paths.coco_image_root
                payload = dict(row)
                payload["stimulus_type"] = "simple" if image_name.startswith("simple_") else "complex"
                payload["gt_image_path"] = str((gt_root / image_name).resolve())
                self.rows_by_image[image_name] = payload

    def get(self, image_name: str) -> dict[str, str]:
        if image_name not in self.rows_by_image:
            raise KeyError(f"Image {image_name} is missing from {self.paths.used_stimuli_tsv}")
        return self.rows_by_image[image_name]

    def caption_zh(self, image_name: str) -> str:
        return self.get(image_name).get("caption", "")

    def caption_en(self, image_name: str) -> str:
        return self.get(image_name).get("translated_caption", "")

    def stimulus_type(self, image_name: str) -> str:
        return self.get(image_name)["stimulus_type"]

    def gt_image_path(self, image_name: str) -> Path:
        return Path(self.get(image_name)["gt_image_path"])


def discover_shards(dataset_root: Path) -> list[AVIDShardInfo]:
    shards: list[AVIDShardInfo] = []
    seen_paths: set[Path] = set()
    for tar_path in sorted(dataset_root.rglob("*.tar")):
        resolved_path = tar_path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        match = None
        for pattern in SHARD_PATTERNS:
            match = pattern.match(tar_path.name)
            if match:
                break
        if not match:
            continue
        shards.append(
            AVIDShardInfo(
                subject_id=int(match.group("subject")),
                stimulus_type=match.group("stimulus"),
                split=match.group("split"),
                tar_path=resolved_path,
                available=True,
            )
        )
    return shards


def summarize_shards(shards: Iterable[AVIDShardInfo]) -> dict[int, dict[str, set[str]]]:
    summary: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for shard in shards:
        summary[shard.subject_id][shard.stimulus_type].add(shard.split)
    return summary


def filter_shards(
    shards: Iterable[AVIDShardInfo],
    subject_ids: Iterable[int] | None = None,
    split: str | None = None,
    stimulus_type: str | None = None,
) -> list[AVIDShardInfo]:
    subject_id_set = set(subject_ids or [])
    selected: list[AVIDShardInfo] = []
    for shard in shards:
        if subject_id_set and shard.subject_id not in subject_id_set:
            continue
        if split and shard.split != split:
            continue
        if stimulus_type and shard.stimulus_type != stimulus_type:
            continue
        selected.append(shard)
    return selected
