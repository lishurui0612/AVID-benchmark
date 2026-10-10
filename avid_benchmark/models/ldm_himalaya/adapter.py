from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm
from torchvision import transforms

from ...data.dataset import aggregate_image_responses, build_loader, count_samples_in_tar, reduce_responses
from ...data.index import AVIDMetadataIndex, discover_shards, filter_shards, summarize_shards
from ...runtime import create_prediction_output_root, create_run_directories, save_json, save_jsonl, save_prediction_result, snapshot_config
from ...types import PredictionRecord
from ...assets import atomic_torch_save
from ..base import ModelAdapter
from ..mindeye2.targets import OpenClipTokenCache
from ..mindeye2.legacy import project_root
from ..registry import register_model_adapter
from .config import LDMHimalayaConfig
from .targets import StableDiffusionV14TargetCache


class TorchStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-12):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps

    def fit(self, X, y=None):
        tensor = torch.as_tensor(X, dtype=torch.float32)
        if tensor.ndim != 2:
            raise ValueError(f"TorchStandardScaler expects 2D input, got shape {tuple(tensor.shape)}")
        if self.with_mean:
            self.mean_ = tensor.mean(dim=0)
        else:
            self.mean_ = torch.zeros(tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
        if self.with_std:
            scale = tensor.std(dim=0, unbiased=False)
            self.scale_ = torch.where(scale > self.eps, scale, torch.ones_like(scale))
        else:
            self.scale_ = torch.ones(tensor.shape[1], device=tensor.device, dtype=tensor.dtype)
        return self

    def transform(self, X):
        tensor = torch.as_tensor(X, dtype=torch.float32, device=self.mean_.device)
        if tensor.ndim != 2:
            raise ValueError(f"TorchStandardScaler expects 2D input, got shape {tuple(tensor.shape)}")
        return (tensor - self.mean_) / self.scale_


@register_model_adapter("ldm_himalaya")
class LDMHimalayaAdapter(ModelAdapter):
    def __init__(self, config: dict[str, Any]):
        self.experiment = config
        self.paths = config["paths"]
        self.train_cfg = config.get("train", {})
        self.eval_cfg = config.get("evaluation", {})
        self.model_cfg = LDMHimalayaConfig.from_dict(config.get("model", {})).resolve_paths(self.paths)
        self.run_name = self.train_cfg.get("run_name", "ldm_himalaya_run")
        self.run_dirs = create_run_directories(self.paths.train_output_root, "ldm_himalaya", self.run_name)
        configured_stimulus_types = self.train_cfg.get("stimulus_types", [])
        self.stimulus_types = [str(item) for item in configured_stimulus_types] if configured_stimulus_types else []
        self.seed = int(self.train_cfg.get("seed", 42))
        import random
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.device("cpu")
        self.metadata = AVIDMetadataIndex(self.paths)
        self.shards = discover_shards(self.paths.dataset_root)
        self.target_cache = StableDiffusionV14TargetCache(self.run_dirs.cache, self.model_cfg, self.device)
        self.selection_clip = OpenClipTokenCache(self.run_dirs.cache, self.model_cfg, self.device)
        self._himalaya_backend: str | None = None
        self._validate_config()

    @classmethod
    def load_from_config(cls, config: dict[str, Any]) -> "LDMHimalayaAdapter":
        adapter = cls(config)
        snapshot_config(adapter.run_dirs, config)
        return adapter

    def capabilities(self) -> dict[str, Any]:
        return {
            "generated_image": self.target_cache.available(),
            "prompt_embedding_target": True,
            "vae_latent_target": True,
            "recon_images_pt": True,
            "asset_status": self._asset_status(),
            "cuda_required": False,
            "cuda_preferred": True,
            "cuda_available": torch.cuda.is_available(),
        }

    def _asset_status(self) -> dict[str, dict[str, str | bool]]:
        status = self.model_cfg.asset_status()
        status["himalaya_backend"]["path"] = self._himalaya_backend or "auto"
        status["himalaya_backend"]["ready"] = self._himalaya_backend in {"torch_cuda", "torch"}
        return status

    def _validate_config(self) -> None:
        if self.world_size > 1:
            raise RuntimeError("ldm_himalaya does not support distributed execution yet.")
        if self.model_cfg.eval_response_reduce != "mean":
            raise ValueError("ldm_himalaya currently requires mean-reduced fMRI responses during prediction.")

    def _is_main_process(self) -> bool:
        return self.rank == 0

    def _log(self, message: str) -> None:
        if self._is_main_process():
            print(message, flush=True)

    def _available_subjects(self) -> list[int]:
        return sorted({shard.subject_id for shard in self.shards})

    def _available_subject_summary(self) -> dict[int, dict[str, list[str]]]:
        summary = summarize_shards(self.shards)
        return {
            subject_id: {stimulus: sorted(list(splits)) for stimulus, splits in stimuli.items()}
            for subject_id, stimuli in summary.items()
        }

    def _resolve_train_subjects(self) -> list[int]:
        subjects = [int(value) for value in self.train_cfg.get("subjects", [])]
        if subjects:
            return subjects
        return self._available_subjects()

    def _subject_urls(self, subject_id: int, split: str) -> list[str]:
        if self.stimulus_types:
            subject_shards = []
            for stimulus_type in self.stimulus_types:
                subject_shards.extend(
                    filter_shards(
                        self.shards,
                        subject_ids=[subject_id],
                        split=split,
                        stimulus_type=stimulus_type,
                    )
                )
        else:
            subject_shards = filter_shards(self.shards, subject_ids=[subject_id], split=split)
        if not subject_shards:
            stimulus_note = f" with stimulus_types={self.stimulus_types}" if self.stimulus_types else ""
            raise FileNotFoundError(f"No {split} shards found for subject {subject_id:02d}{stimulus_note}")
        return [str(shard.tar_path) for shard in subject_shards]

    def _count_subject_samples(self, subject_id: int, split: str) -> int:
        return sum(count_samples_in_tar(url) for url in self._subject_urls(subject_id, split))

    def _subject_loader(self, subject_id: int, split: str, batch_size: int, resampled: bool, drop_last: bool):
        seed = self.seed
        if resampled and split == "train":
            seed = self.seed + subject_id * 1009
        return build_loader(
            url=self._subject_urls(subject_id, split),
            batch_size=batch_size,
            num_workers=self.model_cfg.num_workers,
            resampled=resampled,
            seed=seed,
            drop_last=drop_last,
            distributed=False,
        )

    def _require_himalaya_cuda_backend(self) -> torch.device:
        if self._himalaya_backend in {"torch_cuda", "torch"}:
            return self.device
        try:
            from himalaya.backend import get_backend, set_backend
        except ImportError as exc:
            raise RuntimeError(
                "ldm_himalaya requires himalaya available in the current environment."
            ) from exc
        configured_device = self.train_cfg.get("device", "")
        prefer_cuda = torch.cuda.is_available() and str(configured_device) != "cpu"
        requested_backend = "torch_cuda" if prefer_cuda else "torch"
        try:
            backend = set_backend(requested_backend)
        except Exception as exc:
            if requested_backend == "torch_cuda":
                self._log(
                    "[LDMHimalaya] Falling back to himalaya CPU backend because 'torch_cuda' is unavailable: "
                    f"{exc}"
                )
                backend = set_backend("torch")
            else:
                raise RuntimeError("ldm_himalaya could not initialize himalaya's CPU backend.") from exc
        backend_name = getattr(backend, "name", None) or getattr(get_backend(), "name", "")
        if configured_device:
            requested_device = torch.device(configured_device)
        else:
            requested_device = torch.device(f"cuda:{self.local_rank}") if backend_name == "torch_cuda" else torch.device("cpu")

        if backend_name == "torch_cuda":
            device = requested_device if requested_device.type == "cuda" else torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(device.index or 0)
        elif backend_name == "torch":
            if requested_device.type != "cpu":
                self._log(
                    f"[LDMHimalaya] Requested device {requested_device} is unavailable; using CPU for himalaya."
                )
            device = torch.device("cpu")
        else:
            raise RuntimeError(
                f"ldm_himalaya expected himalaya backend 'torch_cuda' or 'torch', got {backend_name!r}."
            )
        self.device = device
        self.target_cache.set_device(device)
        self.selection_clip.device = device
        self._himalaya_backend = backend_name
        return device

    @staticmethod
    def _normalize_for_loss(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return F.normalize(tensor.float(), dim=-1, eps=eps)

    @staticmethod
    def _rowwise_pearson_r(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        predicted = predicted.float()
        target = target.float()
        predicted = predicted - predicted.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        numerator = (predicted * target).sum(dim=1)
        denominator = torch.sqrt(
            (predicted.square().sum(dim=1) * target.square().sum(dim=1)).clamp_min(eps)
        )
        return numerator / denominator

    def _image_tensor_to_uint8(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return image_tensor.detach().cpu().clamp(0.0, 1.0).mul(255).round().to(torch.uint8).clone()

    def _pil_to_float_tensor(self, image: Image.Image) -> torch.Tensor:
        return transforms.ToTensor()(image.convert("RGB"))

    def _save_candidate_strip(
        self,
        gt_image_path: Path,
        blurry_image: torch.Tensor | None,
        candidate_images: torch.Tensor,
        output_path: Path,
        selected_index: int,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(gt_image_path) as gt_image:
            panels = [gt_image.convert("RGB")]
        if blurry_image is not None:
            panels.append(transforms.ToPILImage()(blurry_image.detach().cpu().clamp(0.0, 1.0)))
        panels.extend(
            transforms.ToPILImage()(candidate.detach().cpu().clamp(0.0, 1.0))
            for candidate in candidate_images
        )

        panel_width = int(self.train_cfg.get("prediction_panel_size", 256))
        panel_height = panel_width
        panels = [panel.resize((panel_width, panel_height), Image.Resampling.BICUBIC) for panel in panels]

        label_height = 36
        canvas = Image.new("RGB", (panel_width * len(panels), panel_height + label_height), color="white")
        labels = ["GT"]
        if blurry_image is not None:
            labels.append("Blurry")
        labels.extend(f"Recon {idx + 1}" for idx in range(candidate_images.shape[0]))
        selected_panel_idx = selected_index + (2 if blurry_image is not None else 1)
        draw = ImageDraw.Draw(canvas)
        for idx, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            x0 = idx * panel_width
            x1 = (idx + 1) * panel_width - 1
            canvas.paste(panel, (x0, label_height))
            fill = (235, 245, 255) if idx == selected_panel_idx else (245, 245, 245)
            outline = (40, 110, 190) if idx == selected_panel_idx else (180, 180, 180)
            draw.rectangle((x0, 0, x1, label_height - 1), fill=fill, outline=outline)
            draw.text((x0 + 12, 9), label, fill=(20, 20, 20))
            if idx == selected_panel_idx:
                draw.rectangle((x0, label_height, x1, panel_height + label_height - 1), outline=outline, width=4)
            if idx < len(panels) - 1:
                draw.line((x1, 0, x1, panel_height + label_height), fill=(180, 180, 180), width=1)
        canvas.save(output_path)

    def _generate_candidate_images(
        self,
        prompt_embeds: torch.Tensor,
        latents: torch.Tensor | None,
        num_candidates: int,
    ) -> torch.Tensor:
        candidates = []
        for _ in range(num_candidates):
            generated_images = self.target_cache.generate_images(prompt_embeds, latents)
            candidates.append(torch.stack([self._pil_to_float_tensor(image) for image in generated_images], dim=0))
        return torch.stack(candidates, dim=1)

    def _select_best_candidate_indices(
        self,
        image_names: list[str],
        candidate_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_embeddings = self.selection_clip.encode_image_names(image_names, self.metadata).to(self.device)
        target_flat = self._normalize_for_loss(target_embeddings.flatten(1))
        batch_scores = []
        best_indices = []
        for batch_idx in range(candidate_images.shape[0]):
            candidate_embeddings = self.selection_clip.encode_image_tensors(candidate_images[batch_idx]).to(self.device)
            candidate_flat = self._normalize_for_loss(candidate_embeddings.flatten(1))
            scores = torch.matmul(candidate_flat, target_flat[batch_idx : batch_idx + 1].T).squeeze(-1)
            batch_scores.append(scores.cpu())
            best_indices.append(int(torch.argmax(scores).item()))
        return torch.stack(batch_scores, dim=0), torch.tensor(best_indices, dtype=torch.long)

    def _aggregate_subject_responses(self, subject_id: int, split: str, max_samples: int = 0) -> tuple[list[str], torch.Tensor]:
        num_samples = self._count_subject_samples(subject_id, split)
        loader = self._subject_loader(
            subject_id=subject_id,
            split=split,
            batch_size=max(1, num_samples),
            resampled=False,
            drop_last=False,
        )
        return aggregate_image_responses(loader, response_reduce=self.model_cfg.eval_response_reduce, max_samples=max_samples)

    def _collect_subject_training_samples(self, subject_id: int, max_samples: int = 0) -> tuple[list[str], torch.Tensor]:
        num_samples = self._count_subject_samples(subject_id, "train")
        loader = self._subject_loader(
            subject_id=subject_id,
            split="train",
            batch_size=max(1, num_samples),
            resampled=False,
            drop_last=False,
        )
        response_mode = str(self.model_cfg.train_response_reduce).lower()
        if response_mode == "mean":
            return aggregate_image_responses(loader, response_reduce="mean", max_samples=max_samples)

        if response_mode in {"first", "random"}:
            image_names_out: list[str] = []
            response_rows: list[torch.Tensor] = []
            for image_names, responses in loader:
                reduced = reduce_responses(responses, response_mode).cpu()
                for idx, image_name in enumerate(image_names):
                    image_names_out.append(image_name)
                    response_rows.append(reduced[idx])
                    if max_samples > 0 and len(image_names_out) >= max_samples:
                        break
                if max_samples > 0 and len(image_names_out) >= max_samples:
                    break
            if not image_names_out:
                return [], torch.empty(0, dtype=torch.float32)
            return image_names_out, torch.stack(response_rows, dim=0)

        if response_mode not in {"none", "expand", "expanded"}:
            raise ValueError(
                "ldm_himalaya train_response_reduce must be one of "
                "'mean', 'none', 'expand', 'expanded', 'first', or 'random'. "
                f"Got {self.model_cfg.train_response_reduce!r}."
            )

        image_names_out: list[str] = []
        response_rows: list[torch.Tensor] = []
        for image_names, responses in loader:
            if responses.ndim == 3:
                expanded = responses.cpu().float()
                for idx, image_name in enumerate(image_names):
                    for repeat_idx in range(expanded.shape[1]):
                        image_names_out.append(image_name)
                        response_rows.append(expanded[idx, repeat_idx])
                        if max_samples > 0 and len(image_names_out) >= max_samples:
                            break
                    if max_samples > 0 and len(image_names_out) >= max_samples:
                        break
            elif responses.ndim == 2:
                expanded = responses.cpu().float()
                for idx, image_name in enumerate(image_names):
                    image_names_out.append(image_name)
                    response_rows.append(expanded[idx])
                    if max_samples > 0 and len(image_names_out) >= max_samples:
                        break
            else:
                raise ValueError(f"Unexpected training responses shape: {tuple(responses.shape)}")
            if max_samples > 0 and len(image_names_out) >= max_samples:
                break
        if not image_names_out:
            return [], torch.empty(0, dtype=torch.float32)
        return image_names_out, torch.stack(response_rows, dim=0)

    def _resolve_ridge_chunk_ranges(self, num_outputs: int) -> list[tuple[int, int]]:
        chunk_size = int(self.model_cfg.ridge_chunk_size)
        if chunk_size <= 0 or chunk_size >= num_outputs:
            return [(0, num_outputs)]
        return [(start, min(start + chunk_size, num_outputs)) for start in range(0, num_outputs, chunk_size)]

    def _fit_ridge_model(self, X: torch.Tensor, Y: torch.Tensor) -> dict[str, Any]:
        from himalaya.kernel_ridge import KernelRidgeCV

        def _build_pipeline():
            ridge = KernelRidgeCV(alphas=list(self.model_cfg.alphas), cv=self.model_cfg.cv, fit_intercept=self.model_cfg.fit_intercept)
            if self.model_cfg.ridge_use_preprocess:
                return make_pipeline(TorchStandardScaler(), ridge), ridge
            return make_pipeline(ridge), ridge

        chunk_ranges = self._resolve_ridge_chunk_ranges(int(Y.shape[1]))
        if len(chunk_ranges) == 1:
            pipeline, ridge = _build_pipeline()
            pipeline.fit(X, Y)
            best_alphas = getattr(ridge, "best_alphas_", None)
            if best_alphas is None:
                best_alpha_tensor = torch.empty(0, device=X.device, dtype=torch.float32)
            else:
                best_alpha_tensor = torch.as_tensor(best_alphas, device=X.device, dtype=torch.float32).reshape(-1)
            return {
                "pipeline": pipeline,
                "best_alphas": best_alpha_tensor.detach().cpu(),
                "chunk_size": int(self.model_cfg.ridge_chunk_size),
                "num_outputs": int(Y.shape[1]),
            }

        chunks: list[dict[str, Any]] = []
        best_alpha_rows: list[torch.Tensor] = []
        for start, end in chunk_ranges:
            pipeline, ridge = _build_pipeline()
            pipeline.fit(X, Y[:, start:end])
            best_alphas = getattr(ridge, "best_alphas_", None)
            if best_alphas is not None:
                best_alpha_rows.append(
                    torch.as_tensor(best_alphas, device=X.device, dtype=torch.float32).reshape(-1).detach().cpu()
                )
            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "pipeline": pipeline,
                }
            )
            if X.device.type == "cuda":
                torch.cuda.empty_cache()
        if best_alpha_rows:
            best_alpha_tensor = torch.cat(best_alpha_rows, dim=0)
        else:
            best_alpha_tensor = torch.empty(0, dtype=torch.float32)
        return {
            "chunks": chunks,
            "best_alphas": best_alpha_tensor,
            "chunk_size": int(self.model_cfg.ridge_chunk_size),
            "num_outputs": int(Y.shape[1]),
        }

    @staticmethod
    def _predict_linear(model_state: dict[str, Any], X: torch.Tensor) -> torch.Tensor:
        chunks = model_state.get("chunks")
        if chunks is not None:
            chunk_predictions = []
            for chunk_state in chunks:
                pipeline = chunk_state.get("pipeline")
                if pipeline is not None:
                    predicted = pipeline.predict(X)
                    chunk_predictions.append(torch.as_tensor(predicted, device=X.device, dtype=torch.float32))
                    continue
                coef = chunk_state["coef"].to(X.device, dtype=torch.float32)
                intercept = chunk_state["intercept"].to(X.device, dtype=torch.float32).reshape(1, -1)
                chunk_predictions.append(X.float().matmul(coef.transpose(0, 1)) + intercept)
            if not chunk_predictions:
                return torch.empty((X.shape[0], 0), device=X.device, dtype=torch.float32)
            return torch.cat(chunk_predictions, dim=1)
        pipeline = model_state.get("pipeline")
        if pipeline is not None:
            predicted = pipeline.predict(X)
            return torch.as_tensor(predicted, device=X.device, dtype=torch.float32)
        coef = model_state["coef"].to(X.device, dtype=torch.float32)
        intercept = model_state["intercept"].to(X.device, dtype=torch.float32).reshape(1, -1)
        return X.float().matmul(coef.transpose(0, 1)) + intercept

    @staticmethod
    def _align_prediction_to_target_stats(
        predicted: torch.Tensor,
        target_mean: torch.Tensor,
        target_var: torch.Tensor,
    ) -> torch.Tensor:
        mean_scores = predicted.mean(dim=0)
        var_scores = predicted.var(dim=0, unbiased=False)
        scale = torch.sqrt(target_var / (var_scores + 1e-8))
        return (predicted - mean_scores) * scale + target_mean

    def _maybe_align_text_predictions(self, model_state: dict[str, Any], predicted: torch.Tensor) -> torch.Tensor:
        if not self.model_cfg.enable_text_variance_alignment:
            return predicted
        if predicted.ndim != 2 or predicted.shape[0] < 2:
            return predicted
        target_mean = model_state.get("target_mean")
        target_var = model_state.get("target_var")
        if target_mean is None or target_var is None:
            return predicted
        return self._align_prediction_to_target_stats(
            predicted,
            torch.as_tensor(target_mean, device=predicted.device, dtype=torch.float32).reshape(1, -1),
            torch.as_tensor(target_var, device=predicted.device, dtype=torch.float32).reshape(1, -1),
        )

    def _log_text_embedding_sample_stats(self, subject_id: int, image_names: list[str], prompt_targets: torch.Tensor) -> None:
        prompt_flat = prompt_targets.flatten(1).float()
        sample_means = prompt_flat.mean(dim=1)
        sample_vars = prompt_flat.var(dim=1, unbiased=False)
        rows = []
        for sample_idx, image_name in enumerate(image_names):
            row = {
                "subject_id": subject_id,
                "sample_index": sample_idx,
                "image_name": image_name,
                "text_embedding_mean": float(sample_means[sample_idx].item()),
                "text_embedding_var": float(sample_vars[sample_idx].item()),
            }
            rows.append(row)
        save_jsonl(self.run_dirs.logs / f"subj{subject_id:02d}_text_embedding_stats.jsonl", rows)

    def _log_prompt_prediction_r(
        self,
        subject_id: int,
        split: str,
        image_names: list[str],
        prompt_flat_before: torch.Tensor,
        prompt_flat_after: torch.Tensor,
    ) -> None:
        target_prompt = self.target_cache.encode_prompt_image_names(image_names, self.metadata).flatten(1).to(
            self.device,
            dtype=torch.float32,
        )
        r_before = self._rowwise_pearson_r(prompt_flat_before, target_prompt)
        r_after = self._rowwise_pearson_r(prompt_flat_after, target_prompt)
        self._log(
            f"[LDMHimalaya] subj{subject_id:02d} {split} prompt-target R before variance adjust: "
            f"mean={r_before.mean().item():.6f} std={r_before.std(unbiased=False).item():.6f}"
        )
        self._log(
            f"[LDMHimalaya] subj{subject_id:02d} {split} prompt-target R after variance adjust: "
            f"mean={r_after.mean().item():.6f} std={r_after.std(unbiased=False).item():.6f}"
        )

    def _subject_bundle_path(self, subject_id: int) -> Path:
        return self.run_dirs.checkpoints / f"subj{subject_id:02d}_bundle.pth"

    def _save_subject_bundle(
        self,
        subject_id: int,
        image_names: list[str],
        averaged_responses: torch.Tensor,
        prompt_targets: torch.Tensor,
        latent_targets: torch.Tensor,
        text_model: dict[str, Any],
        latent_model: dict[str, Any],
    ) -> Path:
        bundle_path = self._subject_bundle_path(subject_id)
        payload = {
            "subject_id": subject_id,
            "backend": "torch_cuda",
            "voxel_dim": int(averaged_responses.shape[1]),
            "text_shape": list(prompt_targets.shape[1:]),
            "latent_shape": list(latent_targets.shape[1:]),
            "num_train_samples": len(image_names),
            "num_unique_train_images": len(set(image_names)),
            "num_train_images": len(image_names),
            "image_names": list(image_names),
            "text_model": text_model,
            "latent_model": latent_model,
            "experiment": self.experiment,
            "model_config": asdict(self.model_cfg),
        }
        atomic_torch_save(payload, bundle_path)
        return bundle_path

    def _save_training_index(self, bundle_paths: dict[int, Path]) -> Path:
        index_path = self.run_dirs.checkpoints / "ldm_himalaya_last.pth"
        payload = {
            "backend": "torch_cuda",
            "subject_order": sorted(bundle_paths),
            "subject_bundles": {str(subject_id): path.name for subject_id, path in bundle_paths.items()},
            "available_subjects": self._available_subject_summary(),
            "experiment": self.experiment,
            "model_config": asdict(self.model_cfg),
        }
        atomic_torch_save(payload, index_path)
        return index_path

    def _load_bundle_index(self, checkpoint_path: str | None) -> tuple[Path, dict[str, Any]]:
        if checkpoint_path:
            path = Path(checkpoint_path).expanduser().resolve()
        else:
            path = (self.run_dirs.checkpoints / "ldm_himalaya_last.pth").resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            payload["_source_path"] = str(path)
        return path, payload

    def _resolve_prediction_bundles(self, checkpoint_payload: dict[str, Any]) -> dict[int, Path]:
        if "subject_bundles" in checkpoint_payload:
            return {
                int(subject_id): (Path(path) if Path(path).is_absolute() else Path(checkpoint_payload["_source_path"]).parent / path).expanduser().resolve()
                for subject_id, path in checkpoint_payload["subject_bundles"].items()
            }
        if "subject_id" in checkpoint_payload and "text_model" in checkpoint_payload:
            subject_id = int(checkpoint_payload["subject_id"])
            return {subject_id: Path(checkpoint_payload["_source_path"]).expanduser().resolve()}
        raise KeyError("Checkpoint payload does not contain subject bundle information.")

    @staticmethod
    def _save_image(image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    def _save_image_tensor(self, image_tensor: torch.Tensor, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        transforms.ToPILImage()(image_tensor.detach().cpu().clamp(0.0, 1.0)).save(path)

    def _save_recon_images_pt(
        self,
        prediction_root: Path,
        image_names: list[str],
        subject_ids: list[int],
        recon_images: list[torch.Tensor],
        selected_indices: list[int],
        clip_score_rows: list[torch.Tensor],
        split: str,
        num_candidates: int,
    ) -> Path:
        prediction_root.mkdir(parents=True, exist_ok=True)
        recon_images_path = prediction_root / "recon_images.pt"
        if recon_images:
            recon_tensor = torch.stack(recon_images, dim=0).to(dtype=torch.uint8)
            clip_scores = torch.stack(clip_score_rows, dim=0)
            selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long)
        else:
            recon_tensor = torch.empty((0, 3, self.model_cfg.image_size, self.model_cfg.image_size), dtype=torch.uint8)
            clip_scores = torch.empty((0, num_candidates), dtype=torch.float32)
            selected_indices_tensor = torch.empty((0,), dtype=torch.long)
        torch.save(
            {
                "image_names": image_names,
                "subject_ids": subject_ids,
                "split": split,
                "recon_images": recon_tensor,
                "selected_indices": selected_indices_tensor,
                "clip_scores": clip_scores,
                "num_candidates_per_response": num_candidates,
                "image_dtype": "uint8",
                "image_value_range": [0, 255],
            },
            recon_images_path,
        )
        return recon_images_path

    def _generate_subject_predictions(
        self,
        subject_id: int,
        image_names: list[str],
        prompt_embeds: torch.Tensor,
        latents: torch.Tensor | None,
        prediction_root: Path,
        requested_subjects: list[int],
        num_candidates: int,
        filename_prefix: str = "",
        output_name_overrides: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str], list[int], list[torch.Tensor], list[int], list[torch.Tensor]]:
        panel_paths: list[str] = []
        final_image_paths: list[str] = []
        output_image_names: list[str] = []
        output_subject_ids: list[int] = []
        selected_recons: list[torch.Tensor] = []
        selected_indices: list[int] = []
        clip_score_rows: list[torch.Tensor] = []
        image_iterator = tqdm(
            range(len(image_names)),
            total=len(image_names),
            desc=f"[LDMHimalaya] Recon subj{subject_id:02d}",
            unit="image",
            dynamic_ncols=True,
            leave=True,
            disable=not self._is_main_process(),
        )
        for idx in image_iterator:
            image_name = image_names[idx]
            single_prompt = prompt_embeds[idx : idx + 1]
            single_latent = latents[idx : idx + 1] if latents is not None else None
            blurry_image = (
                self.target_cache.decode_latents(single_latent)[0]
                if single_latent is not None and self.model_cfg.generation_use_predicted_latent
                else None
            )
            candidate_images = self._generate_candidate_images(single_prompt, single_latent, num_candidates=num_candidates)
            clip_scores, best_indices = self._select_best_candidate_indices([image_name], candidate_images)
            selected_index = int(best_indices[0].item())
            selected_image = candidate_images[0, selected_index]
            if output_name_overrides is not None:
                output_stem = output_name_overrides[idx]
            else:
                image_stem = Path(image_name).stem
                output_stem = image_stem
                output_stem = f"{filename_prefix}{output_stem}"
            subject_root = prediction_root / f"subj{subject_id:02d}"
            final_path = subject_root / "final" / f"{output_stem}.png"
            panel_path = subject_root / "panels" / f"{output_stem}.png"
            self._save_image_tensor(selected_image, final_path)
            self._save_candidate_strip(
                self.metadata.gt_image_path(image_name),
                blurry_image,
                candidate_images[0],
                panel_path,
                selected_index,
            )
            panel_paths.append(str(panel_path))
            final_image_paths.append(str(final_path))
            output_image_names.append(image_name)
            output_subject_ids.append(subject_id)
            selected_recons.append(self._image_tensor_to_uint8(selected_image))
            selected_indices.append(selected_index)
            clip_score_rows.append(clip_scores[0].detach().cpu().clone())
        return (
            panel_paths,
            final_image_paths,
            output_image_names,
            output_subject_ids,
            selected_recons,
            selected_indices,
            clip_score_rows,
        )

    def _predict_train_validation_samples(
        self,
        bundle_paths: dict[int, Path],
        requested_subjects: list[int],
        num_candidates: int,
        prediction_root: Path,
    ) -> dict[str, Any]:
        max_samples = int(self.train_cfg.get("train_validation_num_samples", 1))
        if max_samples <= 0:
            return {"enabled": False, "num_samples": 0, "output_dir": "", "panel_paths": []}

        panel_paths: list[str] = []
        num_samples = 0
        for subject_id in requested_subjects:
            bundle_path = bundle_paths.get(subject_id)
            if bundle_path is None:
                continue
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            all_image_names, averaged_responses = self._aggregate_subject_responses(subject_id, "train")
            if len(all_image_names) == 0:
                continue
            X = averaged_responses.to(self.device, dtype=torch.float32)
            prompt_flat = self._predict_linear(bundle["text_model"], X)
            prompt_flat = self._maybe_align_text_predictions(bundle["text_model"], prompt_flat)
            if max_samples > 0:
                image_names = all_image_names[:max_samples]
                prompt_flat = prompt_flat[:max_samples]
            else:
                image_names = all_image_names
            prompt_embeds = prompt_flat.reshape(len(image_names), *bundle["text_shape"])
            latents = None
            if self.model_cfg.generation_use_predicted_latent:
                latent_flat = self._predict_linear(bundle["latent_model"], X)
                if max_samples > 0:
                    latent_flat = latent_flat[:max_samples]
                latents = latent_flat.reshape(len(image_names), *bundle["latent_shape"])
            if len(requested_subjects) == 1 and len(image_names) == 1:
                output_name_overrides = ["train_sample"]
            else:
                output_name_overrides = [
                    f"train_s{subject_id:02d}_{sample_idx + 1:02d}" for sample_idx in range(len(image_names))
                ]
            subject_panel_paths, _, _, _, _, _, _ = self._generate_subject_predictions(
                subject_id=subject_id,
                image_names=image_names,
                prompt_embeds=prompt_embeds,
                latents=latents,
                prediction_root=prediction_root,
                requested_subjects=requested_subjects,
                num_candidates=num_candidates,
                output_name_overrides=output_name_overrides,
            )
            panel_paths.extend(subject_panel_paths)
            num_samples += len(image_names)
        return {
            "enabled": True,
            "num_samples": num_samples,
            "output_dir": str(prediction_root),
            "panel_paths": panel_paths,
        }

    def train(self) -> dict[str, Any]:
        self._require_himalaya_cuda_backend()
        if not self.target_cache.available():
            raise RuntimeError(
                f"ldm_himalaya training requires {self.model_cfg.stable_diffusion_variant} diffusion assets. "
                f"Current asset status: {self.model_cfg.asset_status()}"
            )
        subject_ids = self._resolve_train_subjects()
        bundle_paths: dict[int, Path] = {}
        summaries: list[dict[str, Any]] = []
        for subject_id in subject_ids:
            self._log(f"[LDMHimalaya] Collecting train responses for subject {subject_id:02d}")
            image_names, averaged_responses = self._collect_subject_training_samples(subject_id)
            if len(image_names) == 0:
                self._log(f"[LDMHimalaya] Skipping subject {subject_id:02d}: no training images found")
                continue
            prompt_targets, latent_targets = self.target_cache.encode_image_names(image_names, self.metadata)
            self._log_text_embedding_sample_stats(subject_id, image_names, prompt_targets)
            X = averaged_responses.to(self.device, dtype=torch.float32)
            Y_prompt = prompt_targets.flatten(1).to(self.device, dtype=torch.float32)
            Y_latent = latent_targets.flatten(1).to(self.device, dtype=torch.float32)
            self._log(
                f"[LDMHimalaya] Cached Stable Diffusion targets for subj{subject_id:02d}: "
                f"X_tr shape={tuple(X.shape)}, "
                f"Y_tr(text_embedding) shape={tuple(Y_prompt.shape)}, "
                f"Y_tr(vae_latent) shape={tuple(Y_latent.shape)}"
            )
            self._log(
                f"[LDMHimalaya] Training response mode for subj{subject_id:02d}: "
                f"{self.model_cfg.train_response_reduce}"
            )
            self._log(
                f"[LDMHimalaya] Ridge regression for subj{subject_id:02d}: "
                f"fMRI -> text embedding | X_tr shape={tuple(X.shape)} | Y_tr shape={tuple(Y_prompt.shape)}"
            )
            text_model = self._fit_ridge_model(X, Y_prompt)
            if self.model_cfg.enable_text_variance_alignment:
                text_model["target_mean"] = Y_prompt.mean(dim=0).detach().cpu()
                text_model["target_var"] = Y_prompt.var(dim=0, unbiased=False).detach().cpu()
            self._log(
                f"[LDMHimalaya] Ridge regression for subj{subject_id:02d}: "
                f"fMRI -> VAE latent | X_tr shape={tuple(X.shape)} | Y_tr shape={tuple(Y_latent.shape)}"
            )
            latent_model = self._fit_ridge_model(X, Y_latent)
            bundle_path = self._save_subject_bundle(
                subject_id=subject_id,
                image_names=image_names,
                averaged_responses=averaged_responses,
                prompt_targets=prompt_targets,
                latent_targets=latent_targets,
                text_model=text_model,
                latent_model=latent_model,
            )
            bundle_paths[subject_id] = bundle_path
            summaries.append(
                {
                    "subject_id": subject_id,
                    "num_train_samples": len(image_names),
                    "num_unique_train_images": len(set(image_names)),
                    "bundle_path": str(bundle_path),
                }
            )
        last_checkpoint = self._save_training_index(bundle_paths)
        result = {
            "last_checkpoint": str(last_checkpoint),
            "subject_bundles": {str(subject_id): path.name for subject_id, path in bundle_paths.items()},
            "subjects": summaries,
            "asset_status": self._asset_status(),
            "backend": "torch_cuda",
        }
        save_json(self.run_dirs.logs / "train_result.json", result)
        return result

    def predict_batch(self, checkpoint_path: str | None = None) -> dict[str, Any]:
        self._require_himalaya_cuda_backend()
        checkpoint_path_resolved, checkpoint_payload = self._load_bundle_index(checkpoint_path)
        bundle_paths = self._resolve_prediction_bundles(checkpoint_payload)
        default_subjects = sorted(bundle_paths)
        requested_subjects = [int(item) for item in self.train_cfg.get("prediction_subjects", default_subjects)]
        split = str(self.train_cfg.get("prediction_split", "val"))
        max_samples = int(self.train_cfg.get("max_prediction_samples", 0))
        output_tag = checkpoint_path_resolved.stem
        prediction_root = create_prediction_output_root(self.run_dirs.predictions, output_tag, split)
        num_candidates = int(self.train_cfg.get("prediction_num_candidates", 5))
        manifest_path = prediction_root / "manifest.jsonl"
        panel_paths: list[str] = []
        final_image_paths: list[str] = []
        output_image_names: list[str] = []
        output_subject_ids: list[int] = []
        selected_recons: list[torch.Tensor] = []
        selected_indices: list[int] = []
        clip_score_rows: list[torch.Tensor] = []
        records: list[PredictionRecord] = []
        train_validation = self._predict_train_validation_samples(
            bundle_paths=bundle_paths,
            requested_subjects=requested_subjects,
            num_candidates=num_candidates,
            prediction_root=prediction_root,
        )
        self._log(
            f"[LDMHimalaya] Finished train sample validation: "
            f"{train_validation.get('num_samples', 0)} sample(s)"
        )
        subject_iterator = tqdm(
            requested_subjects,
            desc="[LDMHimalaya] Subjects",
            dynamic_ncols=True,
            leave=True,
            disable=not self._is_main_process(),
        )
        for subject_id in subject_iterator:
            if self._is_main_process():
                subject_iterator.set_postfix({"subject": f"{subject_id:02d}", "split": split})
            bundle_path = bundle_paths.get(subject_id)
            if bundle_path is None:
                raise KeyError(f"Subject {subject_id} is missing from checkpoint {checkpoint_path_resolved}")
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
            image_names, averaged_responses = self._aggregate_subject_responses(subject_id, split, max_samples=max_samples)
            if len(image_names) == 0:
                continue
            X = averaged_responses.to(self.device, dtype=torch.float32)
            prompt_flat_before = self._predict_linear(bundle["text_model"], X)
            prompt_flat = self._maybe_align_text_predictions(bundle["text_model"], prompt_flat_before)
            self._log_prompt_prediction_r(
                subject_id=subject_id,
                split=split,
                image_names=image_names,
                prompt_flat_before=prompt_flat_before,
                prompt_flat_after=prompt_flat,
            )
            prompt_embeds = prompt_flat.reshape(len(image_names), *bundle["text_shape"])
            latents = None
            if self.model_cfg.generation_use_predicted_latent:
                latent_flat = self._predict_linear(bundle["latent_model"], X)
                latents = latent_flat.reshape(len(image_names), *bundle["latent_shape"])
            (
                subject_panel_paths,
                subject_final_image_paths,
                subject_image_names,
                subject_output_ids,
                subject_selected_recons,
                subject_selected_indices,
                subject_clip_scores,
            ) = self._generate_subject_predictions(
                subject_id=subject_id,
                image_names=image_names,
                prompt_embeds=prompt_embeds,
                latents=latents,
                prediction_root=prediction_root,
                requested_subjects=requested_subjects,
                num_candidates=num_candidates,
            )
            panel_paths.extend(subject_panel_paths)
            final_image_paths.extend(subject_final_image_paths)
            output_image_names.extend(subject_image_names)
            output_subject_ids.extend(subject_output_ids)
            selected_recons.extend(subject_selected_recons)
            selected_indices.extend(subject_selected_indices)
            clip_score_rows.extend(subject_clip_scores)
            route = "ldm_himalaya_img2img" if self.model_cfg.generation_use_predicted_latent else "ldm_himalaya_text_only"
            for image_name, final_path, panel_path, selected_index, clip_scores in zip(
                subject_image_names,
                subject_final_image_paths,
                subject_panel_paths,
                subject_selected_indices,
                subject_clip_scores,
                strict=True,
            ):
                metadata = self.metadata.get(image_name)
                records.append(
                    PredictionRecord(
                        model_name="ldm_himalaya",
                        subject_id=subject_id,
                        stimulus_type=metadata["stimulus_type"],
                        split=split,
                        sample_id=f"subj{subject_id:02d}_{split}_{Path(image_name).stem}",
                        image_name=image_name,
                        gt_image_path=self.metadata.gt_image_path(image_name),
                        generated_image_path=Path(final_path),
                        generated_text=None,
                        route=route,
                        artifact_paths={
                            "final_image": final_path,
                            "candidate_panel_image": panel_path,
                        },
                        notes={
                            "caption_reference": metadata.get("translated_caption", ""),
                            "prompt_field": self.model_cfg.prompt_field,
                            "stable_diffusion_variant": self.model_cfg.stable_diffusion_variant,
                            "image_outputs": {
                                "final": final_path,
                                "candidate_panel": panel_path,
                            },
                            "candidate_selection": "openclip_target_similarity",
                            "selected_index": int(selected_index),
                            "clip_scores": [float(value) for value in clip_scores.tolist()],
                        },
                    )
                )
        recon_images_path = self._save_recon_images_pt(
            prediction_root=prediction_root,
            image_names=output_image_names,
            subject_ids=output_subject_ids,
            recon_images=selected_recons,
            selected_indices=selected_indices,
            clip_score_rows=clip_score_rows,
            split=split,
            num_candidates=num_candidates,
        )
        result = {
            "manifest_path": str(manifest_path),
            "num_predictions": len(output_image_names),
            "prediction_root": str(prediction_root),
            "panel_paths": panel_paths,
            "final_image_paths": final_image_paths,
            "recon_images_path": str(recon_images_path),
            "image_metrics_path": "",
            "image_metrics": None,
            "train_validation": train_validation,
            "prediction_routes": [
                "ldm_himalaya_img2img" if self.model_cfg.generation_use_predicted_latent else "ldm_himalaya_text_only"
            ],
            "num_candidates_per_response": num_candidates,
            "asset_status": self._asset_status(),
            "backend": "torch_cuda",
        }
        save_jsonl(manifest_path, records)
        save_prediction_result(self.run_dirs.predictions, prediction_root, result)
        return result
