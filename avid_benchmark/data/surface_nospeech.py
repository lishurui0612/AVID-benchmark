"""Audit and deterministic split helpers for AVID Surface-NoSpeech shards."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from typing import Iterable, Sequence

import numpy as np

from .index import discover_shards


DATASET_VARIANT = "surface_nospeech_top15pct"
EXPECTED_COMPLEX_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 10)
EXPECTED_SIMPLE_SUBJECTS = (1, 4, 5, 6, 7, 8, 10)
EXPECTED_SPLIT_COUNTS = {
    ("complex", "train"): 640,
    ("complex", "val"): 160,
    ("simple", "train"): 320,
    ("simple", "val"): 80,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stimulus_names_in_tar(path: str | Path) -> list[str]:
    names: list[str] = []
    with tarfile.open(str(path), "r") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".img_name.npy"):
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            value = np.load(io.BytesIO(extracted.read()), allow_pickle=True)
            names.append(str(value.item()))
    if len(names) != len(set(names)):
        raise ValueError(f"Shard contains duplicate stimulus ids: {path}")
    return names


def deterministic_train_dev_partition(
    urls: Sequence[str | Path],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Partition official train stimuli, stratified by stimulus type.

    Hash ordering makes the result independent of tar member order and Python's
    randomized hash seed.  The split is based only on stimulus identifiers.
    """

    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0, 1).")
    groups: dict[str, set[str]] = {}
    pattern = re.compile(r"_(complex|simple)_train\.tar$")
    for url in urls:
        path = Path(url)
        match = pattern.search(path.name)
        group = match.group(1) if match else "unknown"
        groups.setdefault(group, set()).update(_stimulus_names_in_tar(path))

    train_names: set[str] = set()
    dev_names: set[str] = set()
    for group, names in sorted(groups.items()):
        ordered = sorted(
            names,
            key=lambda value: hashlib.sha256(
                f"{seed}:{group}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        dev_count = int(round(len(ordered) * dev_fraction))
        dev_count = min(max(0, dev_count), len(ordered))
        dev_names.update(ordered[:dev_count])
        train_names.update(ordered[dev_count:])
    if train_names & dev_names:
        raise AssertionError("Deterministic train/dev partition overlaps.")
    return train_names, dev_names


def _read_selection_summary(root: Path) -> list[dict[str, str]]:
    summary_path = root / "selection_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Surface-NoSpeech selection summary is missing: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_surface_nospeech(
    dataset_root: str | Path,
    *,
    strict: bool = True,
) -> dict:
    """Return content-addressed provenance for the Surface-NoSpeech release."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Surface-NoSpeech dataset root is missing: {root}")
    selection_rows = _read_selection_summary(root)
    shards = discover_shards(root)
    if not shards:
        raise ValueError(f"No AVID shards were discovered under {root}")

    shard_rows: list[dict] = []
    for shard in shards:
        names = _stimulus_names_in_tar(shard.tar_path)
        expected = EXPECTED_SPLIT_COUNTS.get((shard.stimulus_type, shard.split))
        if strict and expected is not None and len(names) != expected:
            raise ValueError(
                f"{shard.tar_path}: expected {expected} stimuli, found {len(names)}."
            )
        shard_rows.append(
            {
                "subject_id": shard.subject_id,
                "stimulus_type": shard.stimulus_type,
                "split": shard.split,
                "path": str(shard.tar_path),
                "num_stimuli": len(names),
            }
        )

    subject_rows: list[dict] = []
    for row in selection_rows:
        subject_token = str(row["subject"])
        subject_id = int(subject_token.split("-")[-1])
        task = str(row["task"])
        subject_dir = root / f"sub-{subject_id:02d}"
        indices_path = (
            subject_dir
            / f"sub-{subject_id:02d}_task-{task}_top15pct_selected_vertex_indices.npy"
        )
        mask_path = (
            subject_dir
            / f"sub-{subject_id:02d}_task-{task}_top15pct_nospeech_mask.npy"
        )
        if not indices_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Surface selection artifacts are incomplete for sub-{subject_id:02d} "
                f"task-{task}."
            )
        indices = np.load(indices_path, allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False)
        selected_vertices = int(row["selected_vertices"])
        if int(indices.size) != selected_vertices:
            raise ValueError(
                f"{indices_path}: indices={indices.size}, summary={selected_vertices}."
            )
        if int(np.asarray(mask, dtype=bool).sum()) != selected_vertices:
            raise ValueError(
                f"{mask_path}: true vertices do not match selection summary."
            )
        subject_rows.append(
            {
                "subject_id": subject_id,
                "stimulus_type": task,
                "selected_vertices": selected_vertices,
                "total_vertices": int(row["total_vertices"]),
                "speech_vertices": int(row["speech_vertices"]),
                "non_speech_vertices": int(row["non_speech_vertices"]),
                "selected_fraction_of_non_speech": float(
                    row["selected_fraction_of_non_speech"]
                ),
                "mean_noise_ceiling": float(row["mean_nc"]),
                "indices_path": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
            }
        )

    if strict:
        complex_subjects = {
            row["subject_id"]
            for row in subject_rows
            if row["stimulus_type"] == "complex"
        }
        simple_subjects = {
            row["subject_id"]
            for row in subject_rows
            if row["stimulus_type"] == "simple"
        }
        if complex_subjects != set(EXPECTED_COMPLEX_SUBJECTS):
            raise ValueError(
                "Unexpected Surface-NoSpeech complex subjects: "
                f"{sorted(complex_subjects)}."
            )
        if simple_subjects != set(EXPECTED_SIMPLE_SUBJECTS):
            raise ValueError(
                "Unexpected Surface-NoSpeech simple subjects: "
                f"{sorted(simple_subjects)}."
            )

    payload = {
        "dataset_variant": DATASET_VARIANT,
        "dataset_root": str(root),
        "selection_summary_path": str(root / "selection_summary.csv"),
        "selection_summary_sha256": sha256_file(root / "selection_summary.csv"),
        "subjects": sorted(
            subject_rows,
            key=lambda item: (item["subject_id"], item["stimulus_type"]),
        ),
        "shards": sorted(
            shard_rows,
            key=lambda item: (
                item["subject_id"],
                item["stimulus_type"],
                item["split"],
            ),
        ),
        "selection_label_provenance": (
            "The repository can verify masks, indices, counts, and hashes, but "
            "cannot infer from release artifacts alone whether the upstream "
            "noise-ceiling computation used held-out image labels."
        ),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def subject_surface_provenance(
    audit: dict,
    subject_ids: Iterable[int],
    stimulus_types: Iterable[str],
) -> list[dict]:
    subjects = {int(value) for value in subject_ids}
    stimuli = {str(value) for value in stimulus_types}
    return [
        row
        for row in audit.get("subjects", [])
        if int(row["subject_id"]) in subjects
        and (not stimuli or str(row["stimulus_type"]) in stimuli)
    ]
