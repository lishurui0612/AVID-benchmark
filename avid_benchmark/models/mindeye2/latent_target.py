"""Latent visual target marginalization for AVID imagery reconstruction.

The code in this module deliberately keeps the candidate-bank representation
independent from the MindEye adapter.  This makes the loss testable without
loading any of the large generative-model dependencies and gives checkpoints a
stable, content-addressed target definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


VALID_TARGET_MODES = ("paired", "centroid", "hard_min", "soft_marginal")
VALID_EMBEDDING_REDUCTIONS = ("mean_tokens", "flatten")
DATASET_VARIANT = "surface_nospeech_top15pct"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_caption(value: str) -> str:
    """Return the stable normalization used to identify caption groups."""

    return " ".join(str(value or "").strip().lower().split())


def caption_id(value: str) -> str:
    normalized = normalize_caption(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class LatentTargetConfig:
    """Configuration for candidate-set supervision.

    An absent ``latent_target`` config is represented by ``enabled=False`` and
    preserves the historical paired-image training path exactly.
    """

    enabled: bool = False
    mode: str = "paired"
    manifest_path: str = ""
    temperature: float = 0.006
    include_paired: bool = True
    candidate_count: int = 16
    embedding_reduction: str = "mean_tokens"
    generation_embedding_source: str = "clip_voxels"
    generation_samples: int = 4
    generation_seeds: tuple[int, ...] = (42001, 42002, 42003, 42004)
    dataset_variant: str = DATASET_VARIANT
    dev_fraction: float = 0.1
    split_seed: int = 42001
    validation_split: str = "dev"
    strict_surface_audit: bool = True

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any] | None,
        *,
        config_path: str | Path | None = None,
        repo_root: str | Path | None = None,
    ) -> "LatentTargetConfig":
        if not payload:
            return cls()
        values = cls().__dict__.copy()
        values.update(payload)
        values["enabled"] = bool(payload.get("enabled", True))
        values["generation_seeds"] = tuple(
            int(item) for item in values.get("generation_seeds", ())
        )
        manifest = str(values.get("manifest_path", "")).strip()
        if manifest:
            path = Path(manifest).expanduser()
            if not path.is_absolute():
                if config_path:
                    path = Path(config_path).expanduser().resolve().parent / path
                elif repo_root:
                    path = Path(repo_root).expanduser().resolve() / path
            values["manifest_path"] = str(path.resolve())
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in VALID_TARGET_MODES:
            raise ValueError(
                f"latent_target.mode must be one of {VALID_TARGET_MODES}, got {self.mode!r}"
            )
        if self.embedding_reduction not in VALID_EMBEDDING_REDUCTIONS:
            raise ValueError(
                "latent_target.embedding_reduction must be one of "
                f"{VALID_EMBEDDING_REDUCTIONS}, got {self.embedding_reduction!r}"
            )
        if self.enabled and not self.manifest_path:
            raise ValueError("latent_target.manifest_path is required when LVTM is enabled.")
        if self.temperature <= 0:
            raise ValueError("latent_target.temperature must be positive.")
        if self.candidate_count <= 0:
            raise ValueError("latent_target.candidate_count must be positive.")
        if not 0 <= self.dev_fraction < 1:
            raise ValueError("latent_target.dev_fraction must be in [0, 1).")
        if self.validation_split not in {"dev", "val", "test"}:
            raise ValueError("latent_target.validation_split must be dev, val, or test.")
        if self.generation_samples <= 0:
            raise ValueError("latent_target.generation_samples must be positive.")
        if len(self.generation_seeds) < self.generation_samples:
            raise ValueError(
                "latent_target.generation_seeds must contain at least generation_samples values."
            )
        if self.dataset_variant != DATASET_VARIANT:
            raise ValueError(
                f"LVTM v1 only supports dataset_variant={DATASET_VARIANT!r}; "
                f"got {self.dataset_variant!r}."
            )

    @property
    def view_name(self) -> str:
        return "with_paired" if self.include_paired else "without_paired"


@dataclass(frozen=True)
class CandidateDescriptor:
    candidate_id: str
    source: str
    image_path: Path
    embedding_path: Path
    is_paired: bool
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateSetDescriptor:
    caption_id: str
    caption: str
    stimulus_ids: tuple[str, ...]
    candidates: tuple[CandidateDescriptor, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            rows.append(row)
    return rows


class CandidateBank:
    """Validated, lazy-loading view over a versioned candidate manifest."""

    def __init__(self, config: LatentTargetConfig):
        if not config.enabled:
            raise ValueError("CandidateBank requires an enabled LatentTargetConfig.")
        self.config = config
        self.manifest_path = Path(config.manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"LVTM candidate manifest does not exist: {self.manifest_path}"
            )
        self.manifest_hash = _sha256_file(self.manifest_path)
        self.root = self.manifest_path.parent
        self.sets_by_caption: dict[str, CandidateSetDescriptor] = {}
        self.caption_by_stimulus: dict[str, str] = {}
        self._embedding_cache: dict[Path, torch.Tensor] = {}
        self._load_and_validate()

    def _resolve_path(self, value: str, *, field: str) -> Path:
        path = Path(str(value or "")).expanduser()
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Candidate {field} does not exist: {resolved} "
                f"(manifest={self.manifest_path})"
            )
        return resolved

    def _load_and_validate(self) -> None:
        rows = _read_jsonl(self.manifest_path)
        if not rows:
            raise ValueError(f"Candidate manifest is empty: {self.manifest_path}")

        for row_number, row in enumerate(rows, start=1):
            raw_caption = str(row.get("caption", ""))
            group_id = str(row.get("caption_id") or caption_id(raw_caption))
            if group_id != caption_id(raw_caption):
                raise ValueError(
                    f"Manifest row {row_number}: caption_id does not match normalized caption."
                )
            if group_id in self.sets_by_caption:
                raise ValueError(f"Duplicate caption_id in manifest: {group_id}")
            stimulus_ids = tuple(str(item) for item in row.get("stimulus_ids", ()))
            if not stimulus_ids:
                raise ValueError(f"Manifest row {row_number}: stimulus_ids is empty.")

            candidates_by_id: dict[str, dict[str, Any]] = {}
            for item in row.get("candidates", ()):
                candidate_key = str(item.get("candidate_id", ""))
                if not candidate_key:
                    raise ValueError(
                        f"Manifest row {row_number}: candidate_id is missing."
                    )
                if candidate_key in candidates_by_id:
                    raise ValueError(
                        f"Manifest row {row_number}: duplicate candidate_id={candidate_key}."
                    )
                candidates_by_id[candidate_key] = dict(item)

            views = row.get("views") or {}
            selected_ids = list(views.get(self.config.view_name, ()))
            if not selected_ids:
                selected_ids = [
                    key
                    for key, item in candidates_by_id.items()
                    if self.config.include_paired or not bool(item.get("is_paired"))
                ]
            if len(selected_ids) != self.config.candidate_count:
                raise ValueError(
                    f"Manifest row {row_number} ({group_id}) view={self.config.view_name} "
                    f"contains {len(selected_ids)} candidates; expected "
                    f"{self.config.candidate_count}."
                )
            if len(set(selected_ids)) != len(selected_ids):
                raise ValueError(
                    f"Manifest row {row_number}: view {self.config.view_name} "
                    "contains duplicate candidate ids."
                )

            descriptors: list[CandidateDescriptor] = []
            seen_images: set[Path] = set()
            for candidate_key in selected_ids:
                if candidate_key not in candidates_by_id:
                    raise ValueError(
                        f"Manifest row {row_number}: view references unknown "
                        f"candidate_id={candidate_key}."
                    )
                item = candidates_by_id[candidate_key]
                image_path = self._resolve_path(
                    str(item.get("image_path", "")), field="image_path"
                )
                embedding_path = self._resolve_path(
                    str(item.get("embedding_path", "")), field="embedding_path"
                )
                if image_path in seen_images:
                    raise ValueError(
                        f"Manifest row {row_number}: duplicate image path in selected view: "
                        f"{image_path}"
                    )
                seen_images.add(image_path)
                descriptors.append(
                    CandidateDescriptor(
                        candidate_id=candidate_key,
                        source=str(item.get("source", "unknown")),
                        image_path=image_path,
                        embedding_path=embedding_path,
                        is_paired=bool(item.get("is_paired", False)),
                        metadata={
                            key: value
                            for key, value in item.items()
                            if key
                            not in {
                                "candidate_id",
                                "source",
                                "image_path",
                                "embedding_path",
                                "is_paired",
                            }
                        },
                    )
                )
            if self.config.include_paired and not any(
                item.is_paired for item in descriptors
            ):
                raise ValueError(
                    f"Manifest row {row_number}: with_paired view has no paired candidate."
                )
            if not self.config.include_paired and any(
                item.is_paired for item in descriptors
            ):
                raise ValueError(
                    f"Manifest row {row_number}: without_paired view contains a paired candidate."
                )

            descriptor = CandidateSetDescriptor(
                caption_id=group_id,
                caption=raw_caption,
                stimulus_ids=stimulus_ids,
                candidates=tuple(descriptors),
            )
            self.sets_by_caption[group_id] = descriptor
            for stimulus_id in stimulus_ids:
                previous = self.caption_by_stimulus.get(stimulus_id)
                if previous is not None and previous != group_id:
                    raise ValueError(
                        f"Stimulus {stimulus_id!r} maps to multiple caption groups."
                    )
                self.caption_by_stimulus[stimulus_id] = group_id

    def descriptor_for_stimulus(self, stimulus_id: str) -> CandidateSetDescriptor:
        try:
            group_id = self.caption_by_stimulus[str(stimulus_id)]
        except KeyError as exc:
            raise KeyError(
                f"Stimulus {stimulus_id!r} is absent from candidate bank "
                f"{self.manifest_path}."
            ) from exc
        return self.sets_by_caption[group_id]

    def group_ids(self, stimulus_ids: Iterable[str]) -> list[str]:
        return [
            self.descriptor_for_stimulus(stimulus_id).caption_id
            for stimulus_id in stimulus_ids
        ]

    def candidate_ids(self, stimulus_id: str) -> list[str]:
        return [
            item.candidate_id
            for item in self.descriptor_for_stimulus(stimulus_id).candidates
        ]

    def _load_embedding(self, path: Path) -> torch.Tensor:
        cached = self._embedding_cache.get(path)
        if cached is not None:
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            for key in ("embedding", "openclip_tokens", "tokens"):
                if key in payload:
                    payload = payload[key]
                    break
        tensor = torch.as_tensor(payload).detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim not in {1, 2}:
            raise ValueError(
                f"Candidate embedding must be [D] or [T,D], got "
                f"{tuple(tensor.shape)} from {path}."
            )
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Candidate embedding contains non-finite values: {path}")
        tensor = tensor.to(dtype=torch.float16).contiguous()
        self._embedding_cache[path] = tensor
        return tensor

    @staticmethod
    def _reduce_embedding(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        if reduction == "mean_tokens" and tensor.ndim == 2:
            return tensor.float().mean(dim=0)
        if reduction == "flatten":
            return tensor.float().flatten()
        return tensor.float()

    def load_for_stimuli(
        self,
        stimulus_ids: Sequence[str],
        *,
        device: torch.device | str | None = None,
        reduction: str | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        reduction = reduction or self.config.embedding_reduction
        if reduction not in VALID_EMBEDDING_REDUCTIONS:
            raise ValueError(f"Unsupported embedding reduction: {reduction}")
        rows: list[torch.Tensor] = []
        groups: list[str] = []
        expected_shape: tuple[int, ...] | None = None
        for stimulus_id in stimulus_ids:
            descriptor = self.descriptor_for_stimulus(stimulus_id)
            candidate_tensors = [
                self._reduce_embedding(
                    self._load_embedding(item.embedding_path), reduction
                )
                for item in descriptor.candidates
            ]
            current = torch.stack(candidate_tensors, dim=0)
            if expected_shape is None:
                expected_shape = tuple(current.shape)
            elif tuple(current.shape) != expected_shape:
                raise ValueError(
                    "Candidate embedding shapes are inconsistent: "
                    f"expected {expected_shape}, got {tuple(current.shape)} "
                    f"for stimulus {stimulus_id!r}."
                )
            rows.append(current)
            groups.append(descriptor.caption_id)
        if not rows:
            raise ValueError("Cannot load an empty stimulus batch from candidate bank.")
        stacked = torch.stack(rows, dim=0)
        if device is not None:
            stacked = stacked.to(device)
        return stacked, groups

    def provenance(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_hash,
            "view": self.config.view_name,
            "candidate_count": self.config.candidate_count,
            "num_caption_sets": len(self.sets_by_caption),
            "num_stimuli": len(self.caption_by_stimulus),
        }


def reduce_prediction_embeddings(
    predictions: torch.Tensor, reduction: str
) -> torch.Tensor:
    if predictions.ndim < 2:
        raise ValueError(
            f"Predictions must have a batch dimension, got {tuple(predictions.shape)}."
        )
    if reduction == "mean_tokens" and predictions.ndim == 3:
        return predictions.float().mean(dim=1)
    if reduction == "flatten":
        return predictions.float().flatten(1)
    return predictions.float()


@dataclass
class CandidateLossOutput:
    loss: torch.Tensor
    scores: torch.Tensor
    target_indices: torch.Tensor
    unique_group_ids: list[str]
    matched_similarity: torch.Tensor

    def recall_at(self, k: int) -> torch.Tensor:
        if self.scores.shape[1] == 0:
            return torch.zeros((), device=self.scores.device)
        effective_k = min(max(1, int(k)), self.scores.shape[1])
        top_indices = self.scores.topk(effective_k, dim=1).indices
        return (
            top_indices.eq(self.target_indices[:, None]).any(dim=1).float().mean()
        )


def _unique_group_candidates(
    candidate_embeddings: torch.Tensor,
    group_ids: Sequence[str],
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    if candidate_embeddings.shape[0] != len(group_ids):
        raise ValueError(
            "candidate_embeddings batch and group_ids length differ: "
            f"{candidate_embeddings.shape[0]} != {len(group_ids)}"
        )
    unique_ids: list[str] = []
    first_indices: list[int] = []
    index_by_group: dict[str, int] = {}
    targets: list[int] = []
    for row_index, group_id in enumerate(group_ids):
        group_id = str(group_id)
        if group_id not in index_by_group:
            index_by_group[group_id] = len(unique_ids)
            unique_ids.append(group_id)
            first_indices.append(row_index)
        targets.append(index_by_group[group_id])
    device = candidate_embeddings.device
    selected = candidate_embeddings[
        torch.tensor(first_indices, device=device, dtype=torch.long)
    ]
    target_tensor = torch.tensor(targets, device=device, dtype=torch.long)
    return selected, unique_ids, target_tensor


def candidate_group_loss(
    predictions: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    group_ids: Sequence[str],
    *,
    mode: str,
    temperature: float,
    embedding_reduction: str = "mean_tokens",
) -> CandidateLossOutput:
    """Compute caption-group classification with a set-valued target.

    ``candidate_embeddings`` is shaped ``[B,K,...]``. Duplicate caption groups
    are collapsed before building the denominator, so repeated trials and
    cross-subject copies cannot become false negatives.
    """

    if mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {mode}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if candidate_embeddings.ndim < 3:
        raise ValueError(
            "candidate_embeddings must be [B,K,D] or [B,K,T,D], got "
            f"{tuple(candidate_embeddings.shape)}"
        )

    reduced_predictions = reduce_prediction_embeddings(
        predictions, embedding_reduction
    )
    if candidate_embeddings.ndim == 4:
        if embedding_reduction == "mean_tokens":
            reduced_candidates = candidate_embeddings.float().mean(dim=2)
        else:
            reduced_candidates = candidate_embeddings.float().flatten(2)
    else:
        reduced_candidates = candidate_embeddings.float()
    if reduced_predictions.shape[0] != reduced_candidates.shape[0]:
        raise ValueError(
            "Prediction and candidate batches differ: "
            f"{reduced_predictions.shape[0]} != {reduced_candidates.shape[0]}"
        )
    if reduced_predictions.shape[-1] != reduced_candidates.shape[-1]:
        raise ValueError(
            "Prediction and candidate feature dimensions differ: "
            f"{reduced_predictions.shape[-1]} != {reduced_candidates.shape[-1]}"
        )

    group_candidates, unique_ids, targets = _unique_group_candidates(
        reduced_candidates, group_ids
    )
    query = F.normalize(reduced_predictions, dim=-1)
    candidates = F.normalize(group_candidates, dim=-1)
    similarities = torch.einsum("bd,gkd->bgk", query, candidates)

    if mode == "paired":
        scores = similarities[:, :, 0] / temperature
    elif mode == "centroid":
        centroids = F.normalize(candidates.mean(dim=1), dim=-1)
        scores = torch.einsum("bd,gd->bg", query, centroids) / temperature
    elif mode == "hard_min":
        scores = (similarities / temperature).amax(dim=-1)
    else:
        scores = torch.logsumexp(similarities / temperature, dim=-1)
        scores = scores - math.log(similarities.shape[-1])

    loss = F.cross_entropy(scores, targets)
    matched_group_candidates = group_candidates[targets]
    matched_candidates = F.normalize(matched_group_candidates, dim=-1)
    matched_similarity = torch.einsum(
        "bd,bkd->bk", query, matched_candidates
    ).amax(dim=-1)
    return CandidateLossOutput(
        loss=loss,
        scores=scores,
        target_indices=targets,
        unique_group_ids=unique_ids,
        matched_similarity=matched_similarity,
    )


def matched_candidate_posterior(
    predictions: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    *,
    temperature: float,
    embedding_reduction: str = "mean_tokens",
) -> torch.Tensor:
    """Return the within-caption candidate posterior for each prediction."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    query = reduce_prediction_embeddings(predictions, embedding_reduction)
    if candidate_embeddings.ndim == 4:
        candidates = (
            candidate_embeddings.float().mean(dim=2)
            if embedding_reduction == "mean_tokens"
            else candidate_embeddings.float().flatten(2)
        )
    elif candidate_embeddings.ndim == 3:
        candidates = candidate_embeddings.float()
    else:
        raise ValueError(
            "candidate_embeddings must be [B,K,D] or [B,K,T,D]."
        )
    query = F.normalize(query, dim=-1)
    candidates = F.normalize(candidates, dim=-1)
    logits = torch.einsum("bd,bkd->bk", query, candidates) / temperature
    return logits.softmax(dim=-1)
