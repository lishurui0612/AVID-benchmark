from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import textwrap
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from torchvision import transforms

from ...data.dataset import (
    aggregate_image_responses,
    build_loader,
    count_samples_in_tar,
    expand_response_repeats,
    reduce_responses,
    unpack_loader_batch,
)
from ...data.index import AVIDMetadataIndex, discover_shards, filter_shards, summarize_shards
from ...data.surface_nospeech import (
    audit_surface_nospeech,
    deterministic_train_dev_partition,
    subject_surface_provenance,
)
from ...runtime import (
    RunDirectories,
    create_prediction_output_root,
    create_run_directories,
    save_json,
    save_jsonl,
    save_prediction_result,
    snapshot_config,
)
from ...types import PredictionRecord
from ...assets import atomic_torch_save
from ..base import ModelAdapter
from ..training_utils import offset_mixco_permutation, gather_contrastive_batch
from ..vendor_runtime import recon_tensor_to_uint8
from ..registry import register_model_adapter
from .config import MindEye2Config
from .legacy import legacy_generators_module, legacy_utils_module, project_root
from .latent_target import (
    CandidateBank,
    LatentTargetConfig,
    candidate_group_loss,
    matched_candidate_posterior,
)
from .targets import EnhancedTextImageGenerator, GitCaptionGenerator, LowLevelImageAutoencoder, OpenClipTokenCache, UnclipImageGenerator


class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, voxel_batches, subject_indices, clip_target=None):
        voxel_ridge = torch.cat([
            self.ridge(batch, index)
            for batch, index in zip(voxel_batches, subject_indices)
        ], dim=0)
        backbone, clip_voxels, blurry = self.backbone(voxel_ridge)
        if clip_target is None:
            return backbone, clip_voxels, blurry
        prior_loss, prior_out = self.diffusion_prior(
            text_embed=backbone, image_embed=clip_target)
        return backbone, clip_voxels, blurry, prior_loss, prior_out


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes: list[int], out_features: int):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, out_features) for input_size in input_sizes])

    def forward(self, x: torch.Tensor, subj_idx: int) -> torch.Tensor:
        return self.linears[subj_idx](x[:, 0]).unsqueeze(1)


def normalize_for_loss(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return nn.functional.normalize(tensor.float(), dim=-1, eps=eps)


def safe_cosine_anneal(start: float, end: float, steps: int) -> torch.Tensor:
    if steps <= 1:
        return torch.tensor([end], dtype=torch.float32)
    positions = torch.arange(steps, dtype=torch.float32)
    return end + (start - end) / 2 * (1 + torch.cos(torch.pi * positions / (steps - 1)))


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    lr_scheduler_type: str,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, foreach=False)
    if lr_scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=max(1, total_steps))
    elif lr_scheduler_type == "cycle":
        total_steps = max(1, total_steps)
        # OneCycleLR becomes numerically unstable when the warmup phase collapses
        # to a single step, which can happen in small-step smoke runs or LOO phases.
        if total_steps < 10:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
        else:
            warmup_steps = max(2, int(round(total_steps * 0.1)))
            warmup_steps = min(warmup_steps, total_steps - 2)
            pct_start = warmup_steps / total_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                final_div_factor=1000,
                pct_start=pct_start,
            )
    else:
        scheduler = None
    return optimizer, scheduler


@register_model_adapter("mindeye2")
class MindEye2Adapter(ModelAdapter):
    @staticmethod
    def _image_tensor_to_uint8(image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.ndim != 3:
            raise ValueError('A generated image must be [3,H,W].')
        return recon_tensor_to_uint8(image_tensor.unsqueeze(0))[0]

    def __init__(self, config: dict[str, Any]):
        self.experiment = config
        self.paths = config["paths"]
        self.train_cfg = config.get("train", {})
        self.eval_cfg = config.get("evaluation", {})
        self.model_cfg = MindEye2Config.from_dict(config.get("model", {})).resolve_paths(self.paths)
        self.latent_target_cfg = LatentTargetConfig.from_dict(
            config.get("latent_target"),
            config_path=config.get("config_path"),
            repo_root=self.paths.repo_root,
        )
        self.run_name = self.train_cfg.get("run_name", "mindeye2_run")
        self.run_dirs = create_run_directories(self.paths.train_output_root, "mindeye2", self.run_name)
        configured_stimulus_types = self.train_cfg.get("stimulus_types", [])
        self.stimulus_types = [str(item) for item in configured_stimulus_types] if configured_stimulus_types else []
        self.local_rank = int(self.train_cfg.get("local_rank", -1))
        self.rank = int(self.train_cfg.get("rank", 0))
        self.world_size = int(self.train_cfg.get("world_size", 1))
        self.distributed = False
        self._setup_distributed()
        configured_device = self.train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self._resolve_device(configured_device)
        self.seed = int(self.train_cfg.get("seed", 42))
        import random
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.metadata = AVIDMetadataIndex(self.paths)
        self.shards = discover_shards(self.paths.dataset_root)
        self._logical_split_cache: dict[tuple[int, tuple[str, ...]], tuple[set[str], set[str]]] = {}
        self.surface_audit: dict[str, Any] | None = None
        self.candidate_bank: CandidateBank | None = None
        if self.latent_target_cfg.enabled:
            self._validate_latent_target_training_contract()
            self.surface_audit = audit_surface_nospeech(
                self.paths.dataset_root,
                strict=self.latent_target_cfg.strict_surface_audit,
            )
            self.candidate_bank = CandidateBank(self.latent_target_cfg)
        self.legacy_utils = legacy_utils_module()
        self.target_cache = OpenClipTokenCache(self.run_dirs.cache, self.model_cfg, self.device)
        self.low_level = LowLevelImageAutoencoder(self.run_dirs.cache, self.model_cfg, self.device) if self.model_cfg.blurry_recon else None
        self.caption_generator = GitCaptionGenerator(self.model_cfg, self.device)
        self.image_generator = UnclipImageGenerator(self.model_cfg, self.device)
        self.caption_image_generator = EnhancedTextImageGenerator(self.model_cfg, self.device)
        self.model: MindEyeModule | None = None
        self.subject_order: list[int] = []
        self._fixed_training_log_samples: dict[str, dict[str, Any]] = {}
        self._log(
            f"[MindEye2] Initialized adapter on device={self.device}. "
            f"rank={self.rank}/{self.world_size}. "
            f"Assets={self.model_cfg.asset_status()}. "
            f"latent_target={asdict(self.latent_target_cfg)}"
        )

    def _validate_latent_target_training_contract(self) -> None:
        violations = []
        if self.model_cfg.use_prior:
            violations.append("model.use_prior must be false")
        if self.model_cfg.blurry_recon:
            violations.append("model.blurry_recon must be false")
        if float(self.model_cfg.mixup_pct) != 0:
            violations.append("model.mixup_pct must be 0")
        if violations:
            raise ValueError(
                "LVTM candidate-set supervision cannot use components that require "
                "a unique paired-image target: " + "; ".join(violations)
            )

    def _latent_checkpoint_provenance(
        self, subject_order: list[int]
    ) -> dict[str, Any] | None:
        if not self.latent_target_cfg.enabled:
            return None
        if self.surface_audit is None or self.candidate_bank is None:
            raise RuntimeError("LVTM provenance was requested before initialization.")
        return {
            "dataset_variant": self.latent_target_cfg.dataset_variant,
            "dataset_audit_sha256": self.surface_audit["audit_sha256"],
            "selection_summary_sha256": self.surface_audit[
                "selection_summary_sha256"
            ],
            "surface_selections": subject_surface_provenance(
                self.surface_audit,
                subject_order,
                self.stimulus_types,
            ),
            "candidate_bank": self.candidate_bank.provenance(),
            "latent_target_config": asdict(self.latent_target_cfg),
        }

    def _validate_checkpoint_provenance(
        self, checkpoint: dict[str, Any], subject_order: list[int]
    ) -> None:
        if not self.latent_target_cfg.enabled:
            return
        expected = self._latent_checkpoint_provenance(subject_order)
        observed = checkpoint.get("latent_target_provenance")
        if observed != expected:
            raise ValueError(
                "Checkpoint Surface-NoSpeech/LVTM provenance does not match the "
                "current dataset, selected-vertex masks, latent-target config, or "
                "candidate-bank hash. Refusing to load the checkpoint."
            )

    @classmethod
    def load_from_config(cls, config: dict[str, Any]) -> "MindEye2Adapter":
        adapter = cls(config)
        if adapter._is_main_process():
            snapshot_config(adapter.run_dirs, config)
        return adapter

    def capabilities(self) -> dict[str, Any]:
        return {
            "generated_text": self.caption_generator.available(),
            "image_embedding": True,
            "low_level_embedding": bool(self.low_level),
            "caption_image_refinement": self._caption_image_refinement_available(),
            "unclip_generation": self.image_generator.available(),
            "asset_status": self.model_cfg.asset_status(),
        }

    def _available_subjects(self) -> list[int]:
        return sorted({shard.subject_id for shard in self.shards})

    def _setup_distributed(self) -> None:
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        env_rank = int(os.environ.get("RANK", "0"))
        env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = max(self.world_size, env_world_size)
        if self.world_size > 1:
            self.rank = env_rank
            self.local_rank = env_local_rank
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                dist.init_process_group(backend=backend)
            self.distributed = True

    def _resolve_device(self, configured_device: str) -> torch.device:
        if self.distributed and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device(configured_device)

    def _is_main_process(self) -> bool:
        return self.rank == 0

    def _log(self, message: str) -> None:
        if self._is_main_process():
            print(message, flush=True)

    def _clear_memory(self) -> None:
        gc.collect()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _distributed_barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _epoch_iterator(self, num_epochs: int, desc: str):
        return tqdm(
            range(num_epochs),
            desc=desc,
            dynamic_ncols=True,
            leave=True,
            position=0,
            disable=not self._is_main_process(),
        )

    def _should_run_epoch_eval(self, epoch: int) -> bool:
        interval = max(1, int(self.train_cfg.get("eval_every_n_epochs", 1)))
        is_last_epoch = epoch == self.model_cfg.num_epochs - 1
        return ((epoch + 1) % interval == 0) or is_last_epoch

    def _should_run_epoch_preview(self, epoch: int) -> bool:
        interval = max(1, int(self.train_cfg.get("preview_every_n_epochs", 1)))
        is_last_epoch = epoch == self.model_cfg.num_epochs - 1
        return ((epoch + 1) % interval == 0) or is_last_epoch

    @staticmethod
    def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _set_epoch_postfix(
        iterator,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
        val_rows: list[dict[str, Any]] | None = None,
        best_loss: float | None = None,
    ) -> None:
        val_loss = None
        val_cos = None
        val_fwd = None
        val_bwd = None
        if val_metrics is not None:
            val_loss = val_metrics.get("loss")
            val_cos = val_metrics.get("recon_cossim")
            val_fwd = val_metrics.get("fwd_pct_correct")
            val_bwd = val_metrics.get("bwd_pct_correct")
        elif val_rows is not None:
            val_loss = MindEye2Adapter._mean_metric(val_rows, "loss")
            val_cos = MindEye2Adapter._mean_metric(val_rows, "recon_cossim")
            val_fwd = MindEye2Adapter._mean_metric(val_rows, "fwd_pct_correct")
            val_bwd = MindEye2Adapter._mean_metric(val_rows, "bwd_pct_correct")

        postfix = {
            "epoch": epoch + 1,
            "tr_loss": f"{train_metrics.get('loss', float('nan')):.4f}",
            "val_loss": f"{val_loss:.4f}" if val_loss is not None else "-",
            "tr_cos": f"{train_metrics.get('recon_cossim', float('nan')):.3f}" if "recon_cossim" in train_metrics else "-",
            "val_cos": f"{val_cos:.3f}" if val_cos is not None else "-",
            "tr_fwd": f"{train_metrics.get('fwd_pct_correct', float('nan')):.3f}" if "fwd_pct_correct" in train_metrics else "-",
            "val_fwd": f"{val_fwd:.3f}" if val_fwd is not None else "-",
            "tr_bwd": f"{train_metrics.get('bwd_pct_correct', float('nan')):.3f}" if "bwd_pct_correct" in train_metrics else "-",
            "val_bwd": f"{val_bwd:.3f}" if val_bwd is not None else "-",
        }
        if best_loss is not None:
            postfix["best"] = f"{best_loss:.4f}"
        iterator.set_postfix(postfix)

    def _append_epoch_record(self, log_name: str, record: dict[str, Any]) -> None:
        if not self._is_main_process():
            return
        path = self.run_dirs.logs / f"{log_name}_epoch_metrics.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._log(f"[MindEye2] Warning: failed to write epoch metrics log {path}: {exc}")

    def _log_epoch_record(self, log_name: str, record: dict[str, Any]) -> None:
        if not self._is_main_process():
            return
        train = record.get("train") or {}
        validation = record.get("validation")
        if isinstance(validation, list):
            val_loss = self._mean_metric(validation, "loss")
            val_cos = self._mean_metric(validation, "recon_cossim")
            val_fwd = self._mean_metric(validation, "fwd_pct_correct")
            val_bwd = self._mean_metric(validation, "bwd_pct_correct")
        elif isinstance(validation, dict):
            val_loss = validation.get("loss")
            val_cos = validation.get("recon_cossim")
            val_fwd = validation.get("fwd_pct_correct")
            val_bwd = validation.get("bwd_pct_correct")
        else:
            val_loss = val_cos = val_fwd = val_bwd = None
        summary = {
            "epoch": int(record.get("epoch", -1)) + 1,
            "tr_loss": train.get("loss"),
            "val_loss": val_loss,
            "tr_cos": train.get("recon_cossim"),
            "val_cos": val_cos,
            "tr_fwd": train.get("fwd_pct_correct"),
            "val_fwd": val_fwd,
            "tr_bwd": train.get("bwd_pct_correct"),
            "val_bwd": val_bwd,
            "best_loss": record.get("best_loss"),
            "preview": record.get("preview"),
        }
        self._log(f"[MindEye2][{log_name}] {json.dumps(summary, ensure_ascii=False)}")

    def _reduce_scalar(self, value: float, average: bool = True) -> float:
        if not self.distributed or not dist.is_initialized():
            return value
        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor = tensor / self.world_size
        return float(tensor.item())

    def _reduce_metrics_dict(self, metrics: dict[str, float]) -> dict[str, float]:
        return {key: self._reduce_scalar(value, average=True) for key, value in metrics.items()}

    @staticmethod
    def _unwrap_module(module: nn.Module) -> nn.Module:
        if isinstance(module, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            return module.module
        return module

    def _maybe_wrap_model_for_parallel(self, model: MindEyeModule) -> MindEyeModule:
        if not self.distributed:
            return model
        self._log(f"[MindEye2] Enabling DDP on local_rank={self.local_rank}")
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            output_device=self.local_rank if self.device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    def _normalized_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        normalized = {}
        for key, value in model.state_dict().items():
            clean_key = key[len("module.") :] if key.startswith("module.") else key
            normalized[clean_key.replace(".module.", ".")] = value
        return normalized

    def _load_model_state_dict(self, model: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool) -> tuple[list[str], list[str]]:
        target_keys = list(model.state_dict().keys())
        if set(state_dict.keys()) != set(target_keys):
            remapped: dict[str, torch.Tensor] = {}
            for target_key in target_keys:
                source_candidates = [
                    target_key,
                    target_key[len("module.") :] if target_key.startswith("module.") else f"module.{target_key}",
                    target_key.replace(".module.", "."),
                ]
                for source_key in source_candidates:
                    if source_key in state_dict:
                        remapped[target_key] = state_dict[source_key]
                        break
            state_dict = remapped or state_dict
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        return list(missing), list(unexpected)

    def _get_diffusion_prior(self, model: MindEyeModule):
        core_model = self._unwrap_module(model)
        return self._unwrap_module(core_model.diffusion_prior)

    def _available_subject_summary(self) -> dict[int, dict[str, list[str]]]:
        summary = summarize_shards(self.shards)
        return {
            subject_id: {stimulus: sorted(list(splits)) for stimulus, splits in stimuli.items()}
            for subject_id, stimuli in summary.items()
        }

    def _resolve_train_subjects(self, mode: str) -> list[int]:
        subjects = [int(value) for value in self.train_cfg.get("subjects", [])]
        if subjects:
            return subjects
        available = self._available_subjects()
        if mode == "single_subject":
            if len(available) != 1:
                raise ValueError("single_subject mode requires train.subjects to contain exactly one subject.")
            return available
        return available

    @staticmethod
    def _storage_split(split: str) -> str:
        if split in {"train", "dev"}:
            return "train"
        if split in {"val", "test"}:
            return "val"
        raise ValueError(f"Unsupported logical split: {split!r}")

    def _subject_urls(self, subject_id: int, split: str) -> list[str]:
        storage_split = self._storage_split(split)
        if self.stimulus_types:
            subject_shards = []
            for stimulus_type in self.stimulus_types:
                subject_shards.extend(
                    filter_shards(
                        self.shards,
                        subject_ids=[subject_id],
                        split=storage_split,
                        stimulus_type=stimulus_type,
                    )
                )
        else:
            subject_shards = filter_shards(
                self.shards, subject_ids=[subject_id], split=storage_split
            )
        if not subject_shards:
            stimulus_note = f" with stimulus_types={self.stimulus_types}" if self.stimulus_types else ""
            raise FileNotFoundError(
                f"No {storage_split} shards found for logical split={split} "
                f"subject {subject_id:02d}{stimulus_note}"
            )
        return [str(shard.tar_path) for shard in subject_shards]

    def _allowed_names_for_split(
        self, subject_id: int, split: str
    ) -> set[str] | None:
        if not self.latent_target_cfg.enabled or split not in {"train", "dev"}:
            return None
        urls = self._subject_urls(subject_id, "train")
        cache_key = (
            subject_id,
            tuple(sorted(str(Path(url).resolve()) for url in urls)),
        )
        partition = self._logical_split_cache.get(cache_key)
        if partition is None:
            partition = deterministic_train_dev_partition(
                urls,
                dev_fraction=self.latent_target_cfg.dev_fraction,
                seed=self.latent_target_cfg.split_seed,
            )
            self._logical_split_cache[cache_key] = partition
        return partition[0] if split == "train" else partition[1]

    def _count_subject_samples(self, subject_id: int, split: str) -> int:
        allowed = self._allowed_names_for_split(subject_id, split)
        if allowed is not None:
            return len(allowed)
        return sum(count_samples_in_tar(url) for url in self._subject_urls(subject_id, split))

    def _subject_loader(
        self,
        subject_id: int,
        split: str,
        batch_size: int,
        resampled: bool,
        drop_last: bool,
        distributed_override: bool | None = None,
    ):
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
            distributed=(self.distributed and split == "train") if distributed_override is None else distributed_override,
            include_sample_key=self.latent_target_cfg.enabled,
            allowed_image_names=self._allowed_names_for_split(subject_id, split),
        )

    def _infer_num_voxels(self, subject_id: int) -> int:
        loader = self._subject_loader(
            subject_id,
            "train",
            batch_size=1,
            resampled=False,
            drop_last=False,
            distributed_override=False,
        )
        _, _, responses = unpack_loader_batch(next(iter(loader)))
        return int(reduce_responses(responses, "first").shape[-1])

    def _build_model(self, subject_order: list[int]) -> MindEyeModule:
        self._log(f"[MindEye2] Building model for subjects={subject_order}")
        BrainNetwork, BrainDiffusionPrior, PriorNetwork = legacy_generators_module()
        num_voxels_list = [self._infer_num_voxels(subject_id) for subject_id in subject_order]
        self._log(f"[MindEye2] Inferred num_voxels per subject: {num_voxels_list}")
        model = MindEyeModule()
        model.ridge = RidgeRegression(num_voxels_list, out_features=self.model_cfg.hidden_dim)
        model.backbone = BrainNetwork(
            h=self.model_cfg.hidden_dim,
            in_dim=self.model_cfg.hidden_dim,
            seq_len=1,
            n_blocks=self.model_cfg.n_blocks,
            clip_size=self.model_cfg.clip_emb_dim,
            out_dim=self.model_cfg.clip_emb_dim * self.model_cfg.clip_seq_dim,
            blurry_recon=self.model_cfg.blurry_recon,
            clip_scale=self.model_cfg.clip_scale,
        )
        if self.model_cfg.use_prior:
            prior_heads = self.model_cfg.prior_heads or max(1, self.model_cfg.clip_emb_dim // self.model_cfg.prior_dim_head)
            prior_network = PriorNetwork(
                dim=self.model_cfg.clip_emb_dim,
                depth=self.model_cfg.prior_depth,
                dim_head=self.model_cfg.prior_dim_head,
                heads=prior_heads,
                causal=False,
                num_tokens=self.model_cfg.clip_seq_dim,
                learned_query_mode="pos_emb",
            )
            model.diffusion_prior = BrainDiffusionPrior(
                net=prior_network,
                image_embed_dim=self.model_cfg.clip_emb_dim,
                condition_on_text_encodings=False,
                timesteps=self.model_cfg.prior_timesteps,
                cond_drop_prob=0.2,
                image_embed_scale=None,
            )
        model.to(self.device)
        if self._is_main_process():
            self.legacy_utils.count_params(model)
        return self._maybe_wrap_model_for_parallel(model)

    def _next_batch(self, iterator_map: dict[int, Any], loader_map: dict[int, Any], subject_id: int):
        try:
            return next(iterator_map[subject_id])
        except StopIteration:
            iterator_map[subject_id] = iter(loader_map[subject_id])
            return next(iterator_map[subject_id])

    def _compute_steps_per_epoch(self, subjects: list[int], per_subject_batch_size: int) -> int:
        configured = int(self.model_cfg.steps_per_epoch or self.train_cfg.get("steps_per_epoch", 0))
        if configured > 0:
            return configured
        counts = [self._count_subject_samples(subject_id, "train") for subject_id in subjects]
        global_per_subject_batch_size = max(1, per_subject_batch_size * self.world_size)
        return max(1, min(counts) // global_per_subject_batch_size)

    def _forward_subject_batch(
        self,
        model: MindEyeModule,
        subject_order: list[int],
        phase_subjects: list[int],
        voxel_batches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        return model(voxel_batches, [subject_order.index(s) for s in phase_subjects])

    def _train_epoch(
        self,
        model: MindEyeModule,
        phase_subjects: list[int],
        train_loaders: dict[int, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        epoch: int,
        num_epochs: int,
        steps_per_epoch: int,
        mixup_pct: float | None = None,
    ) -> dict[str, float]:
        iterator_map = {subject_id: iter(loader) for subject_id, loader in train_loaders.items()}
        totals = defaultdict(float)
        effective_mixup_pct = self.model_cfg.mixup_pct if mixup_pct is None else mixup_pct
        mixup_cutoff = int(effective_mixup_pct * num_epochs)
        soft_loss_temps = safe_cosine_anneal(
            0.004, 0.0075, max(1, num_epochs - max(1, mixup_cutoff))
        )
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        model.train()
        step_iterator = tqdm(
            range(steps_per_epoch),
            desc=f"[MindEye2] Epoch {epoch + 1}/{num_epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=not self._is_main_process(),
        )
        for step in step_iterator:
            optimizer.zero_grad()
            clip_target_batches = []
            candidate_image_names: list[str] = []
            voxel_batches = []
            latent_target_batches = []
            perm_batches = []
            betas_batches = []
            select_batches = []

            for subject_id in phase_subjects:
                batch = self._next_batch(iterator_map, train_loaders, subject_id)
                sample_keys, image_names, responses = unpack_loader_batch(batch)
                if self.latent_target_cfg.enabled:
                    _, expanded_names, expanded_responses, _ = expand_response_repeats(
                        sample_keys, image_names, responses
                    )
                    voxel = expanded_responses.unsqueeze(1)
                    candidate_image_names.extend(expanded_names)
                    target_names = expanded_names
                else:
                    voxel = reduce_responses(
                        responses, self.model_cfg.train_response_reduce
                    ).unsqueeze(1)
                    target_names = image_names
                if epoch < mixup_cutoff:
                    voxel, perm, betas, select = self.legacy_utils.mixco(voxel)
                    perm_batches.append(offset_mixco_permutation(perm.to(self.device), voxel_batches))
                    betas_batches.append(betas.to(self.device))
                    select_batches.append(select.to(self.device))
                voxel_batches.append(voxel.to(self.device))
                if not self.latent_target_cfg.enabled:
                    clip_target_batches.append(
                        self.target_cache.encode_image_names(
                            target_names, self.metadata
                        ).to(self.device)
                    )
                if self.low_level is not None:
                    latent_target_batches.append(
                        self.low_level.encode_image_names(target_names, self.metadata)
                    )

            clip_target = (
                torch.cat(clip_target_batches, dim=0).to(self.device)
                if clip_target_batches
                else None
            )
            outputs = model(
                voxel_batches,
                [self.subject_order.index(s) for s in phase_subjects],
                clip_target if self.model_cfg.use_prior else None,
            )
            backbone, clip_voxels, blurry_image_enc_ = outputs[:3]
            loss = torch.zeros((), device=self.device)

            if self.latent_target_cfg.enabled:
                if self.candidate_bank is None:
                    raise RuntimeError("LVTM candidate bank is not initialized.")
                candidate_embeddings, group_ids = self.candidate_bank.load_for_stimuli(
                    candidate_image_names,
                    device=self.device,
                    reduction=self.latent_target_cfg.embedding_reduction,
                )
                set_output = candidate_group_loss(
                    clip_voxels,
                    candidate_embeddings,
                    group_ids,
                    mode=self.latent_target_cfg.mode,
                    temperature=self.latent_target_cfg.temperature,
                    embedding_reduction=self.latent_target_cfg.embedding_reduction,
                )
                loss_clip = set_output.loss
                totals["recon_cossim"] += (
                    set_output.matched_similarity.mean().item()
                )
                totals["set_recall_at_1"] += set_output.recall_at(1).item()
                totals["set_recall_at_5"] += set_output.recall_at(5).item()
                totals["fwd_pct_correct"] += set_output.recall_at(1).item()
                totals["bwd_pct_correct"] += set_output.recall_at(1).item()
            else:
                assert clip_target is not None
                clip_voxels_norm = normalize_for_loss(clip_voxels.flatten(1))
                clip_target_norm = normalize_for_loss(clip_target.flatten(1))
                contrastive_predictions, contrastive_targets, global_perm, global_betas, global_selected = gather_contrastive_batch(
                    clip_voxels_norm, clip_target_norm,
                    torch.cat(perm_batches) if perm_batches else None,
                    torch.cat(betas_batches) if betas_batches else None,
                    torch.cat(select_batches) if select_batches else None,
                )
                if epoch < mixup_cutoff:
                    loss_clip = self.legacy_utils.mixco_nce(
                        contrastive_predictions,
                        contrastive_targets,
                        temp=0.006,
                        perm=global_perm,
                        betas=global_betas,
                        select=global_selected,
                    )
                else:
                    temp_idx = min(
                        len(soft_loss_temps) - 1,
                        max(0, epoch - mixup_cutoff),
                    )
                    loss_clip = self.legacy_utils.soft_clip_loss(
                        contrastive_predictions,
                        contrastive_targets,
                        temp=float(soft_loss_temps[temp_idx]),
                    )
            loss = loss + loss_clip.float() * self.model_cfg.clip_scale
            totals["loss_clip_total"] += loss_clip.item()

            if self.model_cfg.use_prior:
                assert clip_target is not None
                loss_prior, prior_out = outputs[3:5]
                loss = loss + loss_prior.float() * self.model_cfg.prior_scale
                totals["loss_prior"] += loss_prior.item()
                totals["recon_cossim"] += nn.functional.cosine_similarity(
                    prior_out.float().flatten(1), clip_target.float().flatten(1)
                ).mean().item()
                totals["recon_mse"] += mse_loss(prior_out.float(), clip_target.float()).item()

            if self.low_level is not None:
                image_enc_pred, _ = blurry_image_enc_
                image_enc = torch.cat(latent_target_batches, dim=0).to(self.device)
                loss_blurry = l1_loss(image_enc_pred.float(), image_enc.float())
                loss = loss + loss_blurry.float() * self.model_cfg.blur_scale
                totals["loss_blurry_total"] += loss_blurry.item()

            if not self.latent_target_cfg.enabled:
                labels = torch.arange(len(clip_voxels_norm), device=self.device)
                totals["fwd_pct_correct"] += self.legacy_utils.topk(
                    self.legacy_utils.batchwise_cosine_similarity(
                        clip_voxels_norm, clip_target_norm
                    ),
                    labels,
                    k=1,
                ).item()
                totals["bwd_pct_correct"] += self.legacy_utils.topk(
                    self.legacy_utils.batchwise_cosine_similarity(
                        clip_target_norm, clip_voxels_norm
                    ),
                    labels,
                    k=1,
                ).item()

            self.legacy_utils.check_loss(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            totals["loss"] += float(loss.item())

        metrics = {key: value / steps_per_epoch for key, value in totals.items()}
        return self._reduce_metrics_dict(metrics)

    def _evaluate_subject(
        self,
        model: MindEyeModule,
        subject_id: int,
        subject_index: int,
        split: str = "val",
    ) -> dict[str, float]:
        num_samples = self._count_subject_samples(subject_id, split)
        loader = self._subject_loader(
            subject_id,
            split,
            batch_size=num_samples,
            resampled=False,
            drop_last=False,
        )
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        totals = defaultdict(float)
        all_names: list[str] = []
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        model.eval()
        with torch.no_grad():
            for batch in loader:
                _, image_names, responses = unpack_loader_batch(batch)
                clip_target = None
                if not self.latent_target_cfg.enabled:
                    clip_target = self.target_cache.encode_image_names(
                        image_names, self.metadata
                    ).to(self.device)
                voxel = (
                    reduce_responses(responses, self.model_cfg.eval_response_reduce)
                    .to(self.device)
                    .unsqueeze(1)
                )
                core_model = self._unwrap_module(model)
                voxel_ridge = core_model.ridge(voxel, subject_index)
                backbone, clip_voxels, blurry_image_enc_ = core_model.backbone(voxel_ridge)

                loss = torch.zeros((), device=self.device)
                if self.latent_target_cfg.enabled:
                    if self.candidate_bank is None:
                        raise RuntimeError("LVTM candidate bank is not initialized.")
                    candidate_embeddings, group_ids = (
                        self.candidate_bank.load_for_stimuli(
                            image_names,
                            device=self.device,
                            reduction=self.latent_target_cfg.embedding_reduction,
                        )
                    )
                    set_output = candidate_group_loss(
                        clip_voxels,
                        candidate_embeddings,
                        group_ids,
                        mode=self.latent_target_cfg.mode,
                        temperature=self.latent_target_cfg.temperature,
                        embedding_reduction=self.latent_target_cfg.embedding_reduction,
                    )
                    loss_clip = set_output.loss
                    totals["recon_cossim"] += (
                        set_output.matched_similarity.mean().item()
                    )
                    totals["set_recall_at_1"] = set_output.recall_at(1).item()
                    totals["set_recall_at_5"] = set_output.recall_at(5).item()
                    totals["fwd_pct_correct"] = set_output.recall_at(1).item()
                    totals["bwd_pct_correct"] = set_output.recall_at(1).item()
                    all_preds.append(clip_voxels.detach().cpu())
                else:
                    assert clip_target is not None
                    clip_voxels_norm = normalize_for_loss(clip_voxels.flatten(1))
                    clip_target_norm = normalize_for_loss(clip_target.flatten(1))
                    loss_clip = self.legacy_utils.soft_clip_loss(
                        clip_voxels_norm, clip_target_norm, temp=0.006
                    )
                    all_preds.append(clip_voxels_norm.cpu())
                    all_targets.append(clip_target_norm.cpu())
                totals["loss_clip_total"] += loss_clip.item()
                loss = loss + loss_clip.float() * self.model_cfg.clip_scale
                all_names.extend(list(image_names))

                if self.model_cfg.use_prior:
                    assert clip_target is not None
                    loss_prior, prior_out = self._get_diffusion_prior(model)(text_embed=backbone, image_embed=clip_target)
                    totals["loss_prior"] += loss_prior.item()
                    loss = loss + loss_prior.float() * self.model_cfg.prior_scale
                    totals["recon_cossim"] += nn.functional.cosine_similarity(
                        prior_out.float().flatten(1), clip_target.float().flatten(1)
                    ).mean().item()
                    totals["recon_mse"] += mse_loss(prior_out.float(), clip_target.float()).item()

                if self.low_level is not None:
                    image_enc_pred, _ = blurry_image_enc_
                    image_enc = self.low_level.encode_image_names(image_names, self.metadata)
                    loss_blurry = l1_loss(image_enc_pred.float(), image_enc.float())
                    totals["loss_blurry_total"] += loss_blurry.item()
                    loss = loss + loss_blurry.float() * self.model_cfg.blur_scale

                totals["loss"] += float(loss.item())

        unique_names = sorted(set(all_names))
        if not self.latent_target_cfg.enabled:
            clip_preds = torch.cat(all_preds, dim=0)
            clip_targets = torch.cat(all_targets, dim=0)
            grouped_preds = []
            grouped_targets = []
            for image_name in unique_names:
                indices = [
                    idx
                    for idx, current in enumerate(all_names)
                    if current == image_name
                ]
                grouped_preds.append(clip_preds[indices].mean(dim=0))
                grouped_targets.append(clip_targets[indices].mean(dim=0))
            grouped_preds_tensor = torch.stack(grouped_preds, dim=0)
            grouped_targets_tensor = torch.stack(grouped_targets, dim=0)
            labels = torch.arange(len(grouped_preds_tensor))
            totals["fwd_pct_correct"] = self.legacy_utils.topk(
                self.legacy_utils.batchwise_cosine_similarity(
                    grouped_preds_tensor, grouped_targets_tensor
                ),
                labels,
                k=1,
            ).item()
            totals["bwd_pct_correct"] = self.legacy_utils.topk(
                self.legacy_utils.batchwise_cosine_similarity(
                    grouped_targets_tensor, grouped_preds_tensor
                ),
                labels,
                k=1,
            ).item()

        no_average = {
            "fwd_pct_correct",
            "bwd_pct_correct",
            "set_recall_at_1",
            "set_recall_at_5",
        }
        metrics = {
            key: (value / max(1, len(all_preds)) if key not in no_average else value)
            for key, value in totals.items()
        }
        metrics["num_unique_images"] = len(unique_names)
        metrics[f"num_{split}_samples"] = len(all_names)
        return metrics

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        model: MindEyeModule,
        epoch: int,
        mode: str,
        subject_order: list[int],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        if not self._is_main_process():
            return checkpoint_path
        payload = {
            "epoch": epoch,
            "mode": mode,
            "subject_order": subject_order,
            "model_state_dict": self._normalized_state_dict(model),
            "experiment": self.experiment,
            "model_config": asdict(self.model_cfg),
            "dataset_variant": (
                self.latent_target_cfg.dataset_variant
                if self.latent_target_cfg.enabled
                else "legacy_unspecified"
            ),
            "latent_target_provenance": self._latent_checkpoint_provenance(
                subject_order
            ),
            "extra": extra or {},
        }
        atomic_torch_save(payload, checkpoint_path)
        return checkpoint_path

    def _skip_completed_training(self) -> bool:
        return False

    def _checkpoint_metadata(self, checkpoint_path: Path) -> dict[str, Any] | None:
        if not checkpoint_path.exists():
            return None
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            self._log(f"[MindEye2] Existing checkpoint is not readable, will retrain: {checkpoint_path} ({exc})")
            return None
        if not isinstance(checkpoint, dict):
            self._log(f"[MindEye2] Existing checkpoint has unexpected format, will retrain: {checkpoint_path}")
            return None
        return checkpoint

    def _checkpoint_matches(
        self,
        checkpoint_path: Path,
        *,
        expected_mode: str | None = None,
        expected_subject_order: list[int] | None = None,
        min_epoch: int | None = None,
    ) -> bool:
        checkpoint = self._checkpoint_metadata(checkpoint_path)
        if checkpoint is None:
            return False
        if expected_mode is not None and checkpoint.get("mode") != expected_mode:
            return False
        if expected_subject_order is not None:
            checkpoint_subject_order = [int(item) for item in checkpoint.get("subject_order", [])]
            if checkpoint_subject_order != [int(item) for item in expected_subject_order]:
                return False
            if self.latent_target_cfg.enabled:
                try:
                    self._validate_checkpoint_provenance(
                        checkpoint, checkpoint_subject_order
                    )
                except ValueError:
                    return False
        if min_epoch is not None and int(checkpoint.get("epoch", -1)) < min_epoch:
            return False
        return True

    def _maybe_run_prediction_evaluation(self, prediction_result: dict[str, Any]) -> dict[str, Any] | None:
        if not self.eval_cfg.get("run_after_prediction", False):
            return None
        from ...evaluation.runner import evaluate_prediction_outputs

        return evaluate_prediction_outputs(
            prediction_result=prediction_result,
            output_dir=self.run_dirs.metrics,
            experiment_config=self.experiment,
        )

    def _available_prediction_routes(self) -> list[str]:
        routes = []
        if self._caption_image_refinement_available():
            routes.append("unclip_sdxl_refine_lowlevel_blend")
        if self.image_generator.available():
            routes.append("unclip")
        if self.low_level is not None:
            routes.append("blurry_recon")
        return routes

    def _caption_image_refinement_available(self) -> bool:
        return (
            bool(self.model_cfg.caption_image_refinement)
            and self.caption_generator.available()
            and self.caption_image_generator.available()
            and self.image_generator.available()
            and self.low_level is not None
        )

    def _blend_caption_and_low_level_images(
        self,
        caption_images: torch.Tensor,
        low_level_images: torch.Tensor,
    ) -> torch.Tensor:
        weight = float(self.model_cfg.caption_sdxl_blend_weight)
        weight = min(max(weight, 0.0), 1.0)
        if low_level_images.shape[-2:] != caption_images.shape[-2:]:
            low_level_images = nn.functional.interpolate(
                low_level_images.float(),
                size=caption_images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return (caption_images.float() * weight + low_level_images.float() * (1.0 - weight)).clamp(0.0, 1.0)

    def _capture_fixed_log_sample(self, split: str, subject_id: int) -> dict[str, Any]:
        cache_key = f"subj{subject_id:02d}_{split}"
        if cache_key in self._fixed_training_log_samples:
            return self._fixed_training_log_samples[cache_key]
        loader = self._subject_loader(
            subject_id,
            split,
            batch_size=1,
            resampled=False,
            drop_last=False,
            distributed_override=False,
        )
        _, image_names, responses = unpack_loader_batch(next(iter(loader)))
        voxel = reduce_responses(responses, "first").unsqueeze(1).cpu()
        sample = {
            "subject_id": subject_id,
            "split": split,
            "image_name": image_names[0],
            "voxel": voxel,
        }
        self._fixed_training_log_samples[cache_key] = sample
        return sample

    def _resolve_training_log_subject(self, subject_order: list[int], mode: str) -> int:
        requested = self.train_cfg.get("preview_subjects")
        if requested:
            return int(requested[0])
        if mode.startswith("multi_subject_leave_one_out_finetune"):
            return int(subject_order[-1])
        return int(subject_order[0])

    def _generate_from_voxel(
        self,
        model: MindEyeModule,
        subject_order: list[int],
        subject_id: int,
        voxel: torch.Tensor,
        allow_full_generation: bool = True,
    ) -> dict[str, torch.Tensor | list[str] | None | str]:
        subject_index = subject_order.index(subject_id)
        voxel = voxel.to(self.device)
        with torch.no_grad():
            core_model = self._unwrap_module(model)
            voxel_ridge = core_model.ridge(voxel, subject_index)
            backbone, clip_voxels, blurry_image_enc_ = core_model.backbone(voxel_ridge)
            if self.model_cfg.use_prior and allow_full_generation:
                prior_out = self._get_diffusion_prior(model).p_sample_loop(
                    backbone.shape,
                    text_cond=dict(text_embed=backbone),
                    cond_scale=1.0,
                    timesteps=self.model_cfg.eval_prior_timesteps,
                )
            else:
                prior_out = clip_voxels

        generated_texts = (
            self.caption_generator.generate(prior_out)
            if allow_full_generation and self.caption_generator.available()
            else [None] * len(voxel)
        )
        valid_texts = [text or "" for text in generated_texts]
        low_level_images = None
        unclip_images = None
        refined_images = None
        blurry_latents = None

        if not allow_full_generation:
            if self.low_level is None:
                raise RuntimeError("Lightweight training preview needs blurry_recon/low-level VAE.")
            blurry_latents = blurry_image_enc_[0]
            low_level_images = self.low_level.decode_latents(blurry_latents).cpu()
            generated_images = low_level_images
            route = "training_preview_blurry_recon"
        elif self._caption_image_refinement_available() and all(text.strip() for text in valid_texts):
            unclip_images = self.image_generator.generate(prior_out)
            blurry_latents = blurry_image_enc_[0]
            low_level_images = self.low_level.decode_latents(blurry_latents).cpu()
            refined_images = self.caption_image_generator.generate_from_prompts(
                valid_texts,
                blurry_images=unclip_images,
            )
            generated_images = self._blend_caption_and_low_level_images(refined_images, low_level_images)
            route = "unclip_sdxl_refine_lowlevel_blend"
        elif self.image_generator.available():
            unclip_images = self.image_generator.generate(prior_out)
            generated_images = unclip_images
            route = "unclip"
        elif self.low_level is not None:
            blurry_latents = blurry_image_enc_[0]
            low_level_images = self.low_level.decode_latents(blurry_latents).cpu()
            generated_images = low_level_images
            route = "blurry_recon"
        else:
            raise RuntimeError(
                "MindEye2 training image log needs either the full refinement route, unCLIP, or blurry_recon. "
                f"Current asset status: {self.model_cfg.asset_status()}"
            )

        return {
            "backbone": backbone.detach().cpu(),
            "clip_voxels": clip_voxels.detach().cpu(),
            "prior_out": prior_out.detach().cpu(),
            "generated_texts": generated_texts,
            "blurry_latent": blurry_latents.detach().cpu() if blurry_latents is not None else None,
            "low_level_image": low_level_images.cpu() if low_level_images is not None else None,
            "unclip_image": unclip_images.cpu() if unclip_images is not None else None,
            "refined_image": refined_images.cpu() if refined_images is not None else None,
            "generated_image": generated_images.cpu(),
            "route": route,
        }

    def _save_labeled_panels(
        self,
        gt_image_path: Path,
        panels: list[tuple[str, torch.Tensor | None]],
        output_path: Path,
        caption: str | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(gt_image_path) as gt_image:
            pil_panels = [("GT", gt_image.convert("RGB"))]
        for label, tensor in panels:
            if tensor is None:
                continue
            pil_panels.append((label, transforms.ToPILImage()(tensor.detach().cpu().clamp(0.0, 1.0))))

        panel_size = int(self.train_cfg.get("training_image_log_panel_size", self.train_cfg.get("prediction_panel_size", 256)))
        label_height = 36
        caption_text = " ".join(str(caption or "").split())
        caption_lines = textwrap.wrap(caption_text, width=max(40, len(pil_panels) * 32)) if caption_text else []
        caption_height = 24 * len(caption_lines) + 16 if caption_lines else 0
        canvas = Image.new("RGB", (panel_size * len(pil_panels), panel_size + label_height + caption_height), color="white")
        draw = ImageDraw.Draw(canvas)
        if caption_lines:
            draw.rectangle((0, 0, canvas.width - 1, caption_height - 1), fill=(255, 252, 232), outline=(210, 200, 160))
            for line_idx, line in enumerate(caption_lines):
                draw.text((12, 8 + line_idx * 24), f"Caption: {line}" if line_idx == 0 else line, fill=(20, 20, 20))
        for idx, (label, image) in enumerate(pil_panels):
            x0 = idx * panel_size
            x1 = (idx + 1) * panel_size - 1
            image = image.resize((panel_size, panel_size), Image.Resampling.BICUBIC)
            label_y0 = caption_height
            image_y0 = caption_height + label_height
            canvas.paste(image, (x0, image_y0))
            draw.rectangle((x0, label_y0, x1, image_y0 - 1), fill=(245, 245, 245), outline=(180, 180, 180))
            draw.text((x0 + 12, label_y0 + 9), label, fill=(20, 20, 20))
            if idx < len(pil_panels) - 1:
                draw.line((x1, caption_height, x1, panel_size + label_height + caption_height), fill=(180, 180, 180), width=1)
        canvas.save(output_path)

    def _generate_training_image_log(
        self,
        model: MindEyeModule,
        subject_order: list[int],
        epoch: int,
        mode: str,
    ) -> dict[str, Any]:
        subject_id = self._resolve_training_log_subject(subject_order, mode)
        preview_root = self.run_dirs.predictions / "training_image_log"
        if mode.startswith("multi_subject_leave_one_out_finetune_from_"):
            preview_root = preview_root / mode
        allow_full_generation = bool(self.train_cfg.get("training_preview_full_generation", False))
        records = []
        for split in ("train", "val"):
            sample = self._capture_fixed_log_sample(split, subject_id)
            outputs = self._generate_from_voxel(
                model,
                subject_order,
                subject_id,
                sample["voxel"],
                allow_full_generation=allow_full_generation,
            )
            image_name = sample["image_name"]
            output_path = preview_root / f"{split}_epoch_{epoch + 1:03d}.png"
            generated_caption = outputs["generated_texts"][0] if outputs["generated_texts"] else None
            self._save_labeled_panels(
                self.metadata.gt_image_path(image_name),
                [
                    ("Low", outputs["low_level_image"][0] if outputs["low_level_image"] is not None else None),
                    ("unCLIP", outputs["unclip_image"][0] if outputs["unclip_image"] is not None else None),
                    ("SDXL", outputs["refined_image"][0] if outputs["refined_image"] is not None else None),
                    ("Final", outputs["generated_image"][0]),
                ],
                output_path,
                caption=generated_caption,
            )
            records.append(
                {
                    "epoch": epoch + 1,
                    "subject_id": subject_id,
                    "split": split,
                    "image_name": image_name,
                    "route": outputs["route"],
                    "generated_text": generated_caption,
                    "panel_path": str(output_path),
                }
            )
        manifest_path = preview_root / "manifest.json"
        current_records = records
        if manifest_path.exists():
            existing_records = json.loads(manifest_path.read_text())
            records = [
                record
                for record in existing_records
                if not any(
                    record.get("epoch") == new_record["epoch"] and record.get("split") == new_record["split"]
                    for new_record in records
                )
            ] + current_records
        save_json(manifest_path, records)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"output_dir": str(preview_root), "manifest_path": str(manifest_path), "num_samples": len(current_records)}

    def _train_mode(self, mode: str, subject_order: list[int], phase_subjects: list[int], ckpt_prefix: str) -> tuple[Path, Path]:
        best_path = self.run_dirs.checkpoints / f"{ckpt_prefix}_best.pth"
        last_path = self.run_dirs.checkpoints / f"{ckpt_prefix}_last.pth"
        if (
            self._skip_completed_training()
            and self._checkpoint_matches(
                best_path,
                expected_mode=mode,
                expected_subject_order=subject_order,
            )
            and self._checkpoint_matches(
                last_path,
                expected_mode=mode,
                expected_subject_order=subject_order,
                min_epoch=max(0, self.model_cfg.num_epochs - 1),
            )
        ):
            self._log(f"[MindEye2] Found completed training checkpoint. Skipping {ckpt_prefix}: {last_path}")
            return best_path, last_path

        self.subject_order = list(subject_order)
        model = self._build_model(subject_order)
        self.model = model
        configured_per_subject_batch_size = int(self.model_cfg.per_subject_batch_size)
        if configured_per_subject_batch_size > 0:
            per_subject_batch_size = max(1, configured_per_subject_batch_size // max(1, self.world_size))
        else:
            per_subject_batch_size = max(1, self.model_cfg.batch_size // max(1, len(phase_subjects) * self.world_size))
        train_loaders = {
            subject_id: self._subject_loader(
                subject_id,
                "train",
                batch_size=per_subject_batch_size,
                resampled=True,
                drop_last=False,
            )
            for subject_id in phase_subjects
        }
        steps_per_epoch = self._compute_steps_per_epoch(phase_subjects, per_subject_batch_size)
        optimizer, scheduler = create_optimizer_and_scheduler(
            model,
            lr=self.model_cfg.max_lr,
            lr_scheduler_type=self.model_cfg.lr_scheduler_type,
            total_steps=steps_per_epoch * max(1, self.model_cfg.num_epochs),
        )
        print(
            f"[MindEye2] Training mode={mode}, phase_subjects={phase_subjects}, "
            f"per_subject_batch_size_per_rank={per_subject_batch_size}, "
            f"global_per_subject_batch_size={per_subject_batch_size * self.world_size}, "
            f"global_batch_size={per_subject_batch_size * len(phase_subjects) * self.world_size}, "
            f"steps_per_epoch={steps_per_epoch}",
            flush=True,
        )

        best_loss = float("inf")
        history: list[dict[str, Any]] = []
        epoch_iterator = self._epoch_iterator(self.model_cfg.num_epochs, f"[MindEye2] Epochs ({ckpt_prefix})")
        for epoch in epoch_iterator:
            train_metrics = self._train_epoch(
                model=model,
                phase_subjects=phase_subjects,
                train_loaders=train_loaders,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                num_epochs=self.model_cfg.num_epochs,
                steps_per_epoch=steps_per_epoch,
            )
            self._distributed_barrier()
            if self._is_main_process():
                if self._should_run_epoch_eval(epoch):
                    val_rows = []
                    pooled_loss = 0.0
                    validation_split = (
                        self.latent_target_cfg.validation_split
                        if self.latent_target_cfg.enabled
                        else "val"
                    )
                    eval_subjects = phase_subjects if mode != "multi_subject_leave_one_out_finetune" else [phase_subjects[0]]
                    for subject_id in eval_subjects:
                        subject_index = subject_order.index(subject_id)
                        metrics = self._evaluate_subject(
                            model,
                            subject_id,
                            subject_index,
                            split=validation_split,
                        )
                        pooled_loss += metrics["loss"]
                        val_rows.append(
                            {
                                "subject_id": subject_id,
                                "split": validation_split,
                                **metrics,
                            }
                        )
                    pooled_loss /= max(1, len(val_rows))
                    history.append(
                        {
                            "epoch": epoch,
                            "train": train_metrics,
                            "validation": val_rows,
                            "pooled_loss": pooled_loss,
                        }
                    )
                    if pooled_loss < best_loss:
                        best_loss = pooled_loss
                        self._save_checkpoint(best_path, model, epoch, mode, subject_order)
                    if self._should_run_epoch_preview(epoch):
                        preview_result = self._maybe_generate_epoch_preview(model, subject_order, epoch, mode)
                        if preview_result is not None:
                            history[-1]["preview"] = preview_result
                    history[-1]["best_loss"] = best_loss
                    self._append_epoch_record(ckpt_prefix, history[-1])
                    self._log_epoch_record(ckpt_prefix, history[-1])
                    self._set_epoch_postfix(
                        epoch_iterator,
                        epoch=epoch,
                        train_metrics=train_metrics,
                        val_rows=val_rows,
                        best_loss=best_loss,
                    )
                else:
                    history.append({"epoch": epoch, "train": train_metrics, "validation_skipped": True})
                    history[-1]["best_loss"] = best_loss if best_loss < float("inf") else None
                    self._append_epoch_record(ckpt_prefix, history[-1])
                    self._log_epoch_record(ckpt_prefix, history[-1])
                    self._set_epoch_postfix(
                        epoch_iterator,
                        epoch=epoch,
                        train_metrics=train_metrics,
                        best_loss=best_loss if best_loss < float("inf") else None,
                    )
            self._distributed_barrier()
        if self._is_main_process():
            self._save_checkpoint(last_path, model, max(0, self.model_cfg.num_epochs - 1), mode, subject_order)
            save_json(self.run_dirs.logs / f"{ckpt_prefix}_history.json", history)
        self._distributed_barrier()
        return best_path, last_path

    def train(self) -> dict[str, Any]:
        mode = self.train_cfg.get("mode", "single_subject")
        subjects = self._resolve_train_subjects(mode) if mode != "multi_subject_leave_one_out" else self._resolve_train_subjects("multi_subject_joint")
        print(f"[MindEye2] Starting train(): mode={mode}, subjects={subjects}", flush=True)
        results: dict[str, Any] = {
            "mode": mode,
            "available_subjects": self._available_subjects(),
            "available_subject_summary": self._available_subject_summary(),
            "dataset_variant": (
                self.latent_target_cfg.dataset_variant
                if self.latent_target_cfg.enabled
                else "legacy_unspecified"
            ),
            "latent_target": asdict(self.latent_target_cfg),
            "latent_target_provenance": self._latent_checkpoint_provenance(
                subjects
            ),
        }

        if mode == "single_subject":
            if len(subjects) != 1:
                raise ValueError("single_subject mode requires exactly one subject in train.subjects")
            best_path, last_path = self._train_mode(mode, subjects, subjects, "single_subject")
            results.update({"best_checkpoint": str(best_path), "last_checkpoint": str(last_path)})
            if self._is_main_process() and self.train_cfg.get("run_validation_prediction_after_train", True):
                predict_result = self.predict_batch(checkpoint_path=str(best_path))
                results["prediction"] = predict_result
                evaluation_result = self._maybe_run_prediction_evaluation(predict_result)
                if evaluation_result is not None:
                    results["evaluation"] = evaluation_result
            if self._is_main_process():
                save_json(self.run_dirs.logs / "train_result.json", results)
            return results

        if mode == "multi_subject_joint":
            best_path, last_path = self._train_mode(mode, subjects, subjects, "joint")
            results.update({"best_checkpoint": str(best_path), "last_checkpoint": str(last_path)})
            if self._is_main_process() and self.train_cfg.get("run_validation_prediction_after_train", True):
                predict_result = self.predict_batch(checkpoint_path=str(best_path))
                results["prediction"] = predict_result
                evaluation_result = self._maybe_run_prediction_evaluation(predict_result)
                if evaluation_result is not None:
                    results["evaluation"] = evaluation_result
            if self._is_main_process():
                save_json(self.run_dirs.logs / "train_result.json", results)
            return results

        if mode == "multi_subject_leave_one_out":
            held_out_subject = int(self.train_cfg["held_out_subject"])
            if held_out_subject in subjects:
                train_subjects = [subject_id for subject_id in subjects if subject_id != held_out_subject]
            else:
                train_subjects = list(subjects)
            if not train_subjects:
                train_subjects = sorted({shard.subject_id for shard in self.shards if shard.subject_id != held_out_subject})
            pretrain_best = self.run_dirs.checkpoints / "loo_pretrain_best.pth"
            pretrain_last = self.run_dirs.checkpoints / "loo_pretrain_last.pth"
            if (
                self._skip_completed_training()
                and self._checkpoint_matches(
                    pretrain_best,
                    expected_mode="multi_subject_leave_one_out_pretrain",
                    expected_subject_order=train_subjects,
                )
                and self._checkpoint_matches(
                    pretrain_last,
                    expected_mode="multi_subject_leave_one_out_pretrain",
                    expected_subject_order=train_subjects,
                    min_epoch=max(0, self.model_cfg.num_epochs - 1),
                )
            ):
                self._log(
                    f"[MindEye2] Found completed LOO pretrain checkpoints. "
                    f"Skipping pretrain and loading {pretrain_best}"
                )
            else:
                pretrain_best, pretrain_last = self._train_mode(
                    "multi_subject_leave_one_out_pretrain",
                    train_subjects,
                    train_subjects,
                    "loo_pretrain",
                )
            finetune_order = [*train_subjects, held_out_subject]
            configured_per_subject_batch_size = int(self.model_cfg.per_subject_batch_size)
            if configured_per_subject_batch_size > 0:
                finetune_batch = max(1, configured_per_subject_batch_size // max(1, self.world_size))
            else:
                finetune_batch = max(1, self.model_cfg.batch_size // self.world_size)
            steps = self._compute_steps_per_epoch([held_out_subject], finetune_batch)
            finetune_epochs = max(1, int(self.model_cfg.loo_finetune_epochs))
            finetune_results: dict[str, dict[str, str]] = {}
            selected_sources = self.train_cfg.get("loo_finetune_sources", ["pretrain_best"])
            if not selected_sources or set(selected_sources) - {"pretrain_best", "pretrain_last"}:
                raise ValueError("loo_finetune_sources must select pretrain_best and/or pretrain_last.")
            for pretrain_label, pretrain_checkpoint in (
                ("pretrain_best", pretrain_best),
                ("pretrain_last", pretrain_last),
            ):
                if pretrain_label not in selected_sources:
                    continue
                self._log(f"[MindEye2] LOO finetune from {pretrain_label}: {pretrain_checkpoint}")
                finetune_prefix = f"loo_finetune_from_{pretrain_label}"
                finetune_last = self.run_dirs.checkpoints / f"{finetune_prefix}_last.pth"
                if (
                    self._skip_completed_training()
                    and self._checkpoint_matches(
                        finetune_last,
                        expected_mode=mode,
                        expected_subject_order=finetune_order,
                        min_epoch=max(0, finetune_epochs - 1),
                    )
                ):
                    self._log(f"[MindEye2] Found completed LOO finetune checkpoint. Skipping {finetune_prefix}: {finetune_last}")
                    if self._is_main_process():
                        finetune_results[pretrain_label] = {
                            "last_checkpoint": str(finetune_last),
                            "history_path": str(self.run_dirs.logs / f"{finetune_prefix}_history.json"),
                            "skipped_completed_training": True,
                        }
                    continue
                pretrain_state = torch.load(pretrain_checkpoint, map_location="cpu", weights_only=False)
                model = self._build_model(finetune_order)
                missing, unexpected = self._load_model_state_dict(model, pretrain_state["model_state_dict"], strict=False)
                del pretrain_state
                if unexpected:
                    raise RuntimeError(f"Unexpected keys when loading {pretrain_label} weights: {unexpected}")
                new_subject_index = len(finetune_order) - 1
                allowed_missing = {
                    f"ridge.linears.{new_subject_index}.weight",
                    f"ridge.linears.{new_subject_index}.bias",
                    f"module.ridge.linears.{new_subject_index}.weight",
                    f"module.ridge.linears.{new_subject_index}.bias",
                }
                if set(missing) - allowed_missing:
                    raise RuntimeError(f"Unexpected missing keys when loading {pretrain_label} weights: {missing}")
                self.model = model
                self.subject_order = finetune_order
                loader = {
                    held_out_subject: self._subject_loader(
                        held_out_subject,
                        "train",
                        batch_size=finetune_batch,
                        resampled=True,
                        drop_last=False,
                    )
                }
                optimizer, scheduler = create_optimizer_and_scheduler(
                    model,
                    lr=self.model_cfg.loo_finetune_lr,
                    lr_scheduler_type=self.model_cfg.loo_finetune_lr_scheduler_type,
                    total_steps=steps * finetune_epochs,
                )
                best_loss = float("inf")
                history = []
                self._log(
                    f"[MindEye2] LOO finetune from {pretrain_label}: "
                    f"epochs={finetune_epochs}, lr={self.model_cfg.loo_finetune_lr}, "
                    f"scheduler={self.model_cfg.loo_finetune_lr_scheduler_type}, "
                    f"mixup_pct={self.model_cfg.loo_finetune_mixup_pct}, "
                    f"batch_size_per_rank={finetune_batch}, "
                    f"global_batch_size={finetune_batch * self.world_size}"
                )
                epoch_iterator = self._epoch_iterator(
                    finetune_epochs,
                    f"[MindEye2] Epochs ({finetune_prefix})",
                )
                for epoch in epoch_iterator:
                    train_metrics = self._train_epoch(
                        model=model,
                        phase_subjects=[held_out_subject],
                        train_loaders=loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        num_epochs=finetune_epochs,
                        steps_per_epoch=steps,
                        mixup_pct=self.model_cfg.loo_finetune_mixup_pct,
                    )
                    self._distributed_barrier()
                    if self._is_main_process():
                        if self._should_run_epoch_eval(epoch):
                            val_metrics = self._evaluate_subject(model, held_out_subject, finetune_order.index(held_out_subject))
                            pooled_loss = val_metrics["loss"]
                            history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
                            if pooled_loss < best_loss:
                                best_loss = pooled_loss
                            if self._should_run_epoch_preview(epoch) or epoch == finetune_epochs - 1:
                                preview_result = self._maybe_generate_epoch_preview(
                                    model,
                                    finetune_order,
                                    epoch,
                                    f"multi_subject_leave_one_out_finetune_from_{pretrain_label}",
                                )
                                if preview_result is not None:
                                    history[-1]["preview"] = preview_result
                            history[-1]["best_loss"] = best_loss
                            self._append_epoch_record(finetune_prefix, history[-1])
                            self._log_epoch_record(finetune_prefix, history[-1])
                            self._set_epoch_postfix(
                                epoch_iterator,
                                epoch=epoch,
                                train_metrics=train_metrics,
                                val_metrics=val_metrics,
                                best_loss=best_loss,
                            )
                        else:
                            history.append({"epoch": epoch, "train": train_metrics, "validation_skipped": True})
                            history[-1]["best_loss"] = best_loss if best_loss < float("inf") else None
                            self._append_epoch_record(finetune_prefix, history[-1])
                            self._log_epoch_record(finetune_prefix, history[-1])
                            self._set_epoch_postfix(
                                epoch_iterator,
                                epoch=epoch,
                                train_metrics=train_metrics,
                                best_loss=best_loss if best_loss < float("inf") else None,
                            )
                    self._distributed_barrier()
                if self._is_main_process():
                    self._save_checkpoint(
                        finetune_last,
                        model,
                        max(0, finetune_epochs - 1),
                        mode,
                        finetune_order,
                        {"stage": "finetune", "pretrain_source": pretrain_label},
                    )
                    save_json(self.run_dirs.logs / f"{finetune_prefix}_history.json", history)
                    finetune_results[pretrain_label] = {
                        "last_checkpoint": str(finetune_last),
                        "history_path": str(self.run_dirs.logs / f"{finetune_prefix}_history.json"),
                    }
                self._distributed_barrier()
            results.update(
                {
                    "pretrain_best_checkpoint": str(pretrain_best),
                    "pretrain_last_checkpoint": str(pretrain_last),
                    "finetunes": finetune_results,
                }
            )
            if self._is_main_process() and self.train_cfg.get("run_validation_prediction_after_train", True):
                prediction_results = {}
                evaluation_results = {}
                for pretrain_label, finetune_info in finetune_results.items():
                    prediction_name = f"loo_finetune_from_{pretrain_label}_last"
                    checkpoint_path = Path(finetune_info["last_checkpoint"])
                    if not checkpoint_path.exists():
                        self._log(f"[MindEye2] Skipping prediction for missing checkpoint: {checkpoint_path}")
                        continue
                    self._log(f"[MindEye2] Running prediction for {prediction_name}: {checkpoint_path}")
                    predict_result = self.predict_batch(checkpoint_path=str(checkpoint_path))
                    prediction_results[prediction_name] = predict_result
                    evaluation_result = self._maybe_run_prediction_evaluation(predict_result)
                    if evaluation_result is not None:
                        evaluation_results[prediction_name] = evaluation_result
                results["predictions"] = prediction_results
                results["prediction"] = prediction_results.get("loo_finetune_from_pretrain_last_last") or next(iter(prediction_results.values()), None)
                if evaluation_results:
                    results["evaluation"] = evaluation_results
            if self._is_main_process():
                save_json(self.run_dirs.logs / "train_result.json", results)
            return results

        raise ValueError(f"Unsupported training mode {mode}")

    def _load_checkpoint_model(self, checkpoint_path: str | None):
        checkpoint_source = checkpoint_path or self.train_cfg.get("checkpoint_path", "")
        if not checkpoint_source:
            raise ValueError("checkpoint_path is required for prediction.")
        path = Path(checkpoint_source).expanduser().resolve()
        print(f"[MindEye2] Loading checkpoint from {path}", flush=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        subject_order = [int(item) for item in checkpoint["subject_order"]]
        self._validate_checkpoint_provenance(checkpoint, subject_order)
        was_distributed = self.distributed
        try:
            # Post-training prediction runs only on rank 0. Building a DDP-wrapped
            # model here would require all ranks to enter the same collectives.
            self.distributed = False
            model = self._build_model(subject_order)
        finally:
            self.distributed = was_distributed
        self._load_model_state_dict(model, checkpoint["model_state_dict"], strict=True)
        model.eval()
        self.model = model
        self.subject_order = subject_order
        return path, checkpoint, model

    def _save_tensor_artifact(self, artifact_dir: Path, name: str, tensor: torch.Tensor) -> str:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{name}.pt"
        torch.save(tensor.detach().cpu(), path)
        return str(path)

    def _save_text_artifact(self, artifact_dir: Path, name: str, text: str | None) -> str:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{name}.pt"
        torch.save(text, path)
        return str(path)

    def _save_image(self, path: Path, image_tensor: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = transforms.ToPILImage()(image_tensor.cpu().clamp(0.0, 1.0))
        image.save(path)

    def _load_image_tensor(self, path: Path | str) -> torch.Tensor:
        with Image.open(path) as image:
            return transforms.ToTensor()(image.convert("RGB"))

    def _candidate_selection_metadata(
        self,
        clip_voxels: torch.Tensor,
        image_names: list[str],
    ) -> list[dict[str, Any]]:
        if not self.latent_target_cfg.enabled:
            return [{"mode": "none"} for _ in image_names]
        if self.candidate_bank is None:
            raise RuntimeError("LVTM candidate bank is not initialized.")
        candidates, _ = self.candidate_bank.load_for_stimuli(
            image_names,
            device=clip_voxels.device,
            reduction=self.latent_target_cfg.embedding_reduction,
        )
        posterior = matched_candidate_posterior(
            clip_voxels,
            candidates,
            temperature=self.latent_target_cfg.temperature,
            embedding_reduction=self.latent_target_cfg.embedding_reduction,
        ).detach().cpu()
        rows: list[dict[str, Any]] = []
        for index, image_name in enumerate(image_names):
            ids = self.candidate_bank.candidate_ids(image_name)
            probabilities = posterior[index].tolist()
            top_index = int(posterior[index].argmax().item())
            rows.append(
                {
                    "mode": self.latent_target_cfg.mode,
                    "caption_id": self.candidate_bank.group_ids([image_name])[0],
                    "candidate_bank_sha256": self.candidate_bank.manifest_hash,
                    "view": self.latent_target_cfg.view_name,
                    "top_candidate_id": ids[top_index],
                    "top_probability": float(probabilities[top_index]),
                    "candidate_ids": ids,
                    "posterior": [float(value) for value in probabilities],
                }
            )
        return rows

    def _predict_with_model(
        self,
        model: MindEyeModule,
        subject_order: list[int],
        requested_subjects: list[int],
        split: str,
        max_samples: int,
        output_tag: str,
    ) -> dict[str, Any]:
        prediction_root = create_prediction_output_root(self.run_dirs.predictions, output_tag, split)
        manifest_path = prediction_root / "manifest.jsonl"
        recon_images_path = prediction_root / "recon_images.pt"
        records: list[PredictionRecord] = []
        final_image_paths: list[str] = []
        panel_paths: list[str] = []
        output_image_names: list[str] = []
        output_subject_ids: list[int] = []
        selected_recons: list[torch.Tensor] = []
        print(
            f"[MindEye2] Starting prediction for subjects={requested_subjects}, split={split}, "
            f"routes={self._available_prediction_routes()}",
            flush=True,
        )

        for subject_id in requested_subjects:
            print(f"[MindEye2] Preparing aggregated responses for subject {subject_id:02d}", flush=True)
            loader = self._subject_loader(subject_id, split, batch_size=self._count_subject_samples(subject_id, split), resampled=False, drop_last=False)
            image_names, averaged_responses = aggregate_image_responses(
                loader,
                response_reduce=self.model_cfg.eval_response_reduce,
                max_samples=max_samples,
            )
            if len(image_names) == 0:
                continue
            subject_index = subject_order.index(subject_id)
            subject_prediction_dir = prediction_root / f"subj{subject_id:02d}"
            subject_artifact_dir = prediction_root / "intermediates" / f"subj{subject_id:02d}"
            prediction_batch_size = int(self.model_cfg.prediction_batch_size or self.model_cfg.batch_size)
            prediction_batch_size = max(1, prediction_batch_size)

            subject_entries: list[dict[str, Any]] = []
            prior_batches: list[torch.Tensor] = []
            blurry_batches: list[torch.Tensor] = []
            total_batches = (len(image_names) + prediction_batch_size - 1) // prediction_batch_size
            for image_name in image_names:
                metadata = self.metadata.get(image_name)
                sample_id = f"subj{subject_id:02d}_{split}_{Path(image_name).stem}"
                artifact_root = subject_artifact_dir / sample_id
                subject_entries.append(
                    {
                        "metadata": metadata,
                        "image_name": image_name,
                        "sample_id": sample_id,
                        "gt_image_path": self.metadata.gt_image_path(image_name),
                        "artifact_root": artifact_root,
                        "artifact_paths": {},
                        "backbone_path": artifact_root / "backbone.pt",
                        "clip_voxels_path": artifact_root / "clip_voxels.pt",
                        "prior_out_path": artifact_root / "prior_out.pt",
                        "blurry_latent_path": artifact_root / "blurry_latent.pt",
                        "caption_path": artifact_root / "generated_text.pt",
                        "generated_text": None,
                        "candidate_selection": {"mode": "none"},
                        "route": "none",
                        "image_outputs": {},
                        "low_level_path": subject_prediction_dir / "blurry_recon" / f"{sample_id}.png",
                        "unclip_path": subject_prediction_dir / "unclip" / f"{sample_id}.png",
                        "refined_path": subject_prediction_dir / "refined_pre_lowlevel_blend" / f"{sample_id}.png",
                        "final_path": subject_prediction_dir / "final" / f"{sample_id}.png",
                        "five_panel_path": subject_prediction_dir / "five_panel" / f"{sample_id}.png",
                    }
                )

            print(f"[MindEye2] Stage 1/5 subject {subject_id:02d}: MindEye embeddings/latents", flush=True)
            for start in range(0, len(image_names), prediction_batch_size):
                batch_names = image_names[start : start + prediction_batch_size]
                batch_entries = subject_entries[start : start + prediction_batch_size]
                required_pt_paths = [
                    *(entry["backbone_path"] for entry in batch_entries),
                    *(entry["clip_voxels_path"] for entry in batch_entries),
                    *(entry["prior_out_path"] for entry in batch_entries),
                ]
                if self.low_level is not None:
                    required_pt_paths.extend(entry["blurry_latent_path"] for entry in batch_entries)
                if all(path.exists() for path in required_pt_paths):
                    print(
                        f"[MindEye2] MindEye subject {subject_id:02d} batch "
                        f"{start // prediction_batch_size + 1}/{total_batches}: reusing saved .pt artifacts",
                        flush=True,
                    )
                    prior_batches.append(torch.stack([torch.load(entry["prior_out_path"], map_location="cpu", weights_only=False) for entry in batch_entries], dim=0))
                    if self.latent_target_cfg.enabled:
                        cached_clip = torch.stack(
                            [
                                torch.load(
                                    entry["clip_voxels_path"],
                                    map_location="cpu",
                                    weights_only=False,
                                )
                                for entry in batch_entries
                            ],
                            dim=0,
                        ).to(self.device)
                        selections = self._candidate_selection_metadata(
                            cached_clip, list(batch_names)
                        )
                        for entry, selection in zip(batch_entries, selections):
                            entry["candidate_selection"] = selection
                        del cached_clip
                    if self.low_level is not None:
                        blurry_batches.append(torch.stack([torch.load(entry["blurry_latent_path"], map_location="cpu", weights_only=False) for entry in batch_entries], dim=0))
                    for entry in batch_entries:
                        entry["artifact_paths"].update(
                            {
                                "backbone": str(entry["backbone_path"]),
                                "clip_voxels": str(entry["clip_voxels_path"]),
                                "prior_out": str(entry["prior_out_path"]),
                            }
                        )
                        if self.low_level is not None:
                            entry["artifact_paths"]["blurry_latent"] = str(entry["blurry_latent_path"])
                    continue
                print(
                    f"[MindEye2] MindEye subject {subject_id:02d} batch "
                    f"{start // prediction_batch_size + 1}/{total_batches}",
                    flush=True,
                )
                voxel = averaged_responses[start : start + prediction_batch_size].to(self.device).unsqueeze(1)
                with torch.no_grad():
                    core_model = self._unwrap_module(model)
                    voxel_ridge = core_model.ridge(voxel, subject_index)
                    backbone, clip_voxels, blurry_image_enc_ = core_model.backbone(voxel_ridge)
                    if self.model_cfg.use_prior:
                        prior_out = self._get_diffusion_prior(model).p_sample_loop(
                            backbone.shape,
                            text_cond=dict(text_embed=backbone),
                            cond_scale=1.0,
                            timesteps=self.model_cfg.eval_prior_timesteps,
                        )
                    else:
                        prior_out = clip_voxels

                prior_out_cpu = prior_out.detach().cpu()
                prior_batches.append(prior_out_cpu)
                if self.latent_target_cfg.enabled:
                    selections = self._candidate_selection_metadata(
                        clip_voxels, list(batch_names)
                    )
                    for entry, selection in zip(batch_entries, selections):
                        entry["candidate_selection"] = selection
                blurry_cpu = blurry_image_enc_[0].detach().cpu() if self.low_level is not None else None
                if blurry_cpu is not None:
                    blurry_batches.append(blurry_cpu)

                for idx, entry in enumerate(batch_entries):
                    artifact_root = entry["artifact_root"]
                    artifact_paths = {
                        "backbone": self._save_tensor_artifact(artifact_root, "backbone", backbone.detach().cpu()[idx]),
                        "clip_voxels": self._save_tensor_artifact(artifact_root, "clip_voxels", clip_voxels.detach().cpu()[idx]),
                        "prior_out": self._save_tensor_artifact(artifact_root, "prior_out", prior_out_cpu[idx]),
                    }
                    if blurry_cpu is not None:
                        artifact_paths["blurry_latent"] = self._save_tensor_artifact(artifact_root, "blurry_latent", blurry_cpu[idx])
                    entry["artifact_paths"].update(artifact_paths)
                del voxel, voxel_ridge, backbone, clip_voxels, blurry_image_enc_, prior_out
                self._clear_memory()

            self._unwrap_module(model).to("cpu")
            self._clear_memory()

            entry_offset = 0
            if self.low_level is not None:
                print(f"[MindEye2] Stage 2/5 subject {subject_id:02d}: low-level blurry recon", flush=True)
                try:
                    for batch_idx, blurry_cpu in enumerate(blurry_batches):
                        print(f"[MindEye2] Low-level subject {subject_id:02d} batch {batch_idx + 1}/{len(blurry_batches)}", flush=True)
                        batch_entries = subject_entries[entry_offset : entry_offset + len(blurry_cpu)]
                        if all(entry["low_level_path"].exists() for entry in batch_entries):
                            print(f"[MindEye2] Low-level subject {subject_id:02d} batch {batch_idx + 1}: reusing saved images", flush=True)
                            for entry in batch_entries:
                                entry["artifact_paths"]["blurry_recon_image"] = str(entry["low_level_path"].resolve())
                                entry["route"] = "blurry_recon"
                            entry_offset += len(batch_entries)
                            continue
                        low_level_images = self.low_level.decode_latents(blurry_cpu.to(self.device)).cpu()
                        for idx in range(len(low_level_images)):
                            entry = subject_entries[entry_offset + idx]
                            low_level_path = entry["low_level_path"]
                            self._save_image(low_level_path, low_level_images[idx])
                            entry["artifact_paths"]["blurry_recon_image"] = str(low_level_path.resolve())
                            entry["route"] = "blurry_recon"
                        entry_offset += len(low_level_images)
                        del low_level_images
                        self._clear_memory()
                finally:
                    self.low_level.unload()
                    self._clear_memory()

            if self.caption_generator.available():
                print(f"[MindEye2] Stage 3/5 subject {subject_id:02d}: captions", flush=True)
                try:
                    entry_offset = 0
                    for batch_idx, prior_out_cpu in enumerate(prior_batches):
                        print(f"[MindEye2] Caption subject {subject_id:02d} batch {batch_idx + 1}/{len(prior_batches)}", flush=True)
                        batch_entries = subject_entries[entry_offset : entry_offset + len(prior_out_cpu)]
                        if all(entry["caption_path"].exists() for entry in batch_entries):
                            print(f"[MindEye2] Caption subject {subject_id:02d} batch {batch_idx + 1}: reusing saved .pt captions", flush=True)
                            for idx, entry in enumerate(batch_entries):
                                text = torch.load(entry["caption_path"], map_location="cpu", weights_only=False)
                                entry["generated_text"] = text
                                entry["artifact_paths"]["generated_text"] = str(entry["caption_path"])
                            entry_offset += len(batch_entries)
                            continue
                        generated_texts = self.caption_generator.generate(prior_out_cpu)
                        for idx, text in enumerate(generated_texts):
                            entry = subject_entries[entry_offset + idx]
                            entry["generated_text"] = text
                            entry["artifact_paths"]["generated_text"] = self._save_text_artifact(entry["artifact_root"], "generated_text", text)
                        entry_offset += len(generated_texts)
                        self._clear_memory()
                finally:
                    self.caption_generator.unload()
                    self._clear_memory()
            else:
                for entry in subject_entries:
                    if entry["caption_path"].exists():
                        entry["generated_text"] = torch.load(entry["caption_path"], map_location="cpu", weights_only=False)
                        entry["artifact_paths"]["generated_text"] = str(entry["caption_path"])

            if self.image_generator.available():
                print(f"[MindEye2] Stage 4/5 subject {subject_id:02d}: unCLIP images", flush=True)
                try:
                    entry_offset = 0
                    for batch_idx, prior_out_cpu in enumerate(prior_batches):
                        print(f"[MindEye2] unCLIP subject {subject_id:02d} batch {batch_idx + 1}/{len(prior_batches)}", flush=True)
                        batch_entries = subject_entries[entry_offset : entry_offset + len(prior_out_cpu)]
                        if all(entry["unclip_path"].exists() for entry in batch_entries):
                            print(f"[MindEye2] unCLIP subject {subject_id:02d} batch {batch_idx + 1}: reusing saved images", flush=True)
                            for entry in batch_entries:
                                unclip_path = entry["unclip_path"]
                                entry["image_outputs"]["unclip"] = str(unclip_path.resolve())
                                entry["artifact_paths"]["unclip_image"] = str(unclip_path.resolve())
                                entry["route"] = "unclip"
                            entry_offset += len(batch_entries)
                            continue
                        unclip_images = self.image_generator.generate(prior_out_cpu)
                        for idx in range(len(unclip_images)):
                            entry = subject_entries[entry_offset + idx]
                            unclip_path = entry["unclip_path"]
                            self._save_image(unclip_path, unclip_images[idx])
                            entry["image_outputs"]["unclip"] = str(unclip_path.resolve())
                            entry["artifact_paths"]["unclip_image"] = str(unclip_path.resolve())
                            entry["route"] = "unclip"
                        entry_offset += len(unclip_images)
                        del unclip_images
                        self._clear_memory()
                finally:
                    self.image_generator.unload()
                    self._clear_memory()

            print(f"[MindEye2] Stage 5/5 subject {subject_id:02d}: SDXL refinement/final save", flush=True)
            full_refinement_available = self._caption_image_refinement_available()
            try:
                for start in range(0, len(subject_entries), prediction_batch_size):
                    batch_entries = subject_entries[start : start + prediction_batch_size]
                    print(
                        f"[MindEye2] Finalize subject {subject_id:02d} batch "
                        f"{start // prediction_batch_size + 1}/{total_batches}",
                        flush=True,
                    )
                    if all(entry["final_path"].exists() and entry["five_panel_path"].exists() for entry in batch_entries):
                        print(
                            f"[MindEye2] Finalize subject {subject_id:02d} batch "
                            f"{start // prediction_batch_size + 1}: reusing saved final/five-panel images",
                            flush=True,
                        )
                        for entry in batch_entries:
                            if entry["unclip_path"].exists():
                                entry["image_outputs"]["unclip"] = str(entry["unclip_path"].resolve())
                                entry["artifact_paths"]["unclip_image"] = str(entry["unclip_path"].resolve())
                            if entry["refined_path"].exists():
                                entry["image_outputs"]["refined_pre_lowlevel_blend"] = str(entry["refined_path"].resolve())
                                entry["artifact_paths"]["refined_pre_lowlevel_blend_image"] = str(entry["refined_path"].resolve())
                                entry["route"] = "unclip_sdxl_refine_lowlevel_blend"
                            elif entry["unclip_path"].exists():
                                entry["route"] = "unclip"
                            elif entry["low_level_path"].exists():
                                entry["route"] = "blurry_recon"
                            else:
                                entry["route"] = "reused_final"
                            entry["image_outputs"]["final"] = str(entry["final_path"].resolve())
                            entry["artifact_paths"]["final_image"] = str(entry["final_path"].resolve())
                            entry["artifact_paths"]["five_panel_image"] = str(entry["five_panel_path"].resolve())
                            records.append(
                                PredictionRecord(
                                    model_name="mindeye2",
                                    subject_id=subject_id,
                                    stimulus_type=entry["metadata"]["stimulus_type"],
                                    split=split,
                                    sample_id=entry["sample_id"],
                                    image_name=entry["image_name"],
                                    gt_image_path=entry["gt_image_path"],
                                    generated_image_path=entry["final_path"].resolve(),
                                    generated_text=entry.get("generated_text"),
                                    route=entry["route"],
                                    artifact_paths=entry["artifact_paths"],
                                    notes={
                                        "caption_reference": entry["metadata"].get("translated_caption", ""),
                                        "image_outputs": entry["image_outputs"],
                                        "five_panel_path": str(entry["five_panel_path"].resolve()),
                                        "candidate_selection": entry[
                                            "candidate_selection"
                                        ],
                                    },
                                )
                            )
                            final_image_paths.append(str(entry["final_path"].resolve()))
                            panel_paths.append(str(entry["five_panel_path"].resolve()))
                            output_image_names.append(entry["image_name"])
                            output_subject_ids.append(subject_id)
                            selected_recons.append(self._image_tensor_to_uint8(self._load_image_tensor(entry["final_path"])))
                        continue
                    valid_texts = [str(entry.get("generated_text") or "") for entry in batch_entries]
                    has_unclip_batch = all(entry["unclip_path"].exists() for entry in batch_entries)
                    has_low_level_batch = all(entry["low_level_path"].exists() for entry in batch_entries)
                    has_refined_batch = all(entry["refined_path"].exists() for entry in batch_entries)
                    refined_images = None
                    final_images = None
                    unclip_images = None
                    low_level_images = None

                    if has_unclip_batch:
                        unclip_images = torch.stack([self._load_image_tensor(entry["unclip_path"]) for entry in batch_entries], dim=0)
                    if has_low_level_batch:
                        low_level_images = torch.stack([self._load_image_tensor(entry["low_level_path"]) for entry in batch_entries], dim=0)

                    if full_refinement_available and unclip_images is not None and low_level_images is not None and all(text.strip() for text in valid_texts):
                        if has_refined_batch:
                            print(
                                f"[MindEye2] SDXL subject {subject_id:02d} batch "
                                f"{start // prediction_batch_size + 1}: reusing saved refined images",
                                flush=True,
                            )
                            refined_images = torch.stack([self._load_image_tensor(entry["refined_path"]) for entry in batch_entries], dim=0)
                        else:
                            refined_images = self.caption_image_generator.generate_from_prompts(valid_texts, blurry_images=unclip_images)
                        final_images = self._blend_caption_and_low_level_images(refined_images, low_level_images)
                        batch_route = "unclip_sdxl_refine_lowlevel_blend"
                    elif unclip_images is not None:
                        final_images = unclip_images
                        batch_route = "unclip"
                    elif low_level_images is not None:
                        final_images = low_level_images
                        batch_route = "blurry_recon"
                    else:
                        raise RuntimeError(
                            "MindEye2 prediction needs either a ready unCLIP checkpoint or a ready blurry_recon VAE. "
                            f"Current asset status: {self.model_cfg.asset_status()}"
                        )

                    for idx, entry in enumerate(batch_entries):
                        output_image_path = entry["final_path"]
                        self._save_image(output_image_path, final_images[idx])
                        if entry["unclip_path"].exists():
                            entry["image_outputs"]["unclip"] = str(entry["unclip_path"].resolve())
                            entry["artifact_paths"]["unclip_image"] = str(entry["unclip_path"].resolve())
                        entry["image_outputs"]["final"] = str(output_image_path.resolve())
                        entry["artifact_paths"]["final_image"] = str(output_image_path.resolve())
                        entry["route"] = batch_route
                        if refined_images is not None:
                            refined_path = subject_prediction_dir / "refined_pre_lowlevel_blend" / f"{entry['sample_id']}.png"
                            self._save_image(refined_path, refined_images[idx])
                            entry["refined_path"] = refined_path
                            entry["image_outputs"]["refined_pre_lowlevel_blend"] = str(refined_path.resolve())
                            entry["artifact_paths"]["refined_pre_lowlevel_blend_image"] = str(refined_path.resolve())

                        self._save_labeled_panels(
                            entry["gt_image_path"],
                            [
                                ("BlurryRecon", low_level_images[idx] if low_level_images is not None else None),
                                ("unCLIP", unclip_images[idx] if unclip_images is not None else None),
                                ("Refined", refined_images[idx] if refined_images is not None else None),
                                ("Final", final_images[idx]),
                            ],
                            entry["five_panel_path"],
                            caption=entry.get("generated_text"),
                        )
                        entry["artifact_paths"]["five_panel_image"] = str(entry["five_panel_path"].resolve())

                        records.append(
                            PredictionRecord(
                                model_name="mindeye2",
                                subject_id=subject_id,
                                stimulus_type=entry["metadata"]["stimulus_type"],
                                split=split,
                                sample_id=entry["sample_id"],
                                image_name=entry["image_name"],
                                gt_image_path=entry["gt_image_path"],
                                generated_image_path=output_image_path.resolve(),
                                generated_text=entry.get("generated_text"),
                                route=entry["route"],
                                artifact_paths=entry["artifact_paths"],
                                notes={
                                    "caption_reference": entry["metadata"].get("translated_caption", ""),
                                    "image_outputs": entry["image_outputs"],
                                    "five_panel_path": str(entry["five_panel_path"].resolve()),
                                    "candidate_selection": entry[
                                        "candidate_selection"
                                    ],
                                },
                            )
                        )
                        final_image_paths.append(str(output_image_path.resolve()))
                        panel_paths.append(str(entry["five_panel_path"].resolve()))
                        output_image_names.append(entry["image_name"])
                        output_subject_ids.append(subject_id)
                        selected_recons.append(self._image_tensor_to_uint8(final_images[idx]))
                    del refined_images, final_images, unclip_images, low_level_images
                    self._clear_memory()
            finally:
                with contextlib.suppress(Exception):
                    self.caption_image_generator.unload()
                self._clear_memory()
                self._unwrap_module(model).to(self.device)

        save_jsonl(manifest_path, records)
        if selected_recons:
            recon_images_tensor = torch.stack(selected_recons, dim=0)
        else:
            recon_images_tensor = torch.empty((0, 3, 1, 1), dtype=torch.uint8)
        torch.save(
            {
                "image_names": output_image_names,
                "subject_ids": output_subject_ids,
                "split": split,
                "recon_images": recon_images_tensor,
                "image_dtype": "uint8",
                "image_value_range": [0, 255],
            },
            recon_images_path,
        )
        result = {
            "manifest_path": str(manifest_path),
            "num_predictions": len(records),
            "prediction_root": str(prediction_root),
            "final_image_paths": final_image_paths,
            "panel_paths": panel_paths,
            "recon_images_path": str(recon_images_path),
            "prediction_routes": self._available_prediction_routes(),
            "asset_status": self.model_cfg.asset_status(),
            "dataset_variant": (
                self.latent_target_cfg.dataset_variant
                if self.latent_target_cfg.enabled
                else "legacy_unspecified"
            ),
            "latent_target": asdict(self.latent_target_cfg),
            "candidate_bank": (
                self.candidate_bank.provenance()
                if self.candidate_bank is not None
                else None
            ),
        }
        save_prediction_result(self.run_dirs.predictions, prediction_root, result)
        return result

    def _maybe_generate_epoch_preview(
        self,
        model: MindEyeModule,
        subject_order: list[int],
        epoch: int,
        mode: str,
    ) -> dict[str, Any] | None:
        if not self.train_cfg.get("generate_val_preview_every_epoch", False):
            return None
        self._log(f"[MindEye2] Generating training image log for epoch {epoch + 1}")
        return self._generate_training_image_log(model, subject_order, epoch, mode)

    def predict_batch(self, checkpoint_path: str | None = None) -> dict[str, Any]:
        checkpoint_path_resolved, checkpoint, model = self._load_checkpoint_model(checkpoint_path)
        mode = checkpoint.get("mode", self.train_cfg.get("mode", "single_subject"))
        if mode == "multi_subject_leave_one_out":
            default_subjects = [self.subject_order[-1]]
        else:
            default_subjects = list(self.subject_order)
        requested_subjects = [int(item) for item in self.train_cfg.get("prediction_subjects", default_subjects)]
        split = self.train_cfg.get("prediction_split", "val")
        result = self._predict_with_model(
            model=model,
            subject_order=self.subject_order,
            requested_subjects=requested_subjects,
            split=split,
            max_samples=int(self.train_cfg.get("max_prediction_samples", 0)),
            output_tag=checkpoint_path_resolved.stem,
        )
        train_sample_count = int(self.train_cfg.get("prediction_train_samples", 0))
        if train_sample_count > 0 and split != "train":
            self._log(f"[MindEye2] Running extra train-split prediction sample(s): {train_sample_count}")
            result["train_sample_prediction"] = self._predict_with_model(
                model=model,
                subject_order=self.subject_order,
                requested_subjects=requested_subjects,
                split="train",
                max_samples=train_sample_count,
                output_tag=f"{checkpoint_path_resolved.stem}_train_sample",
            )
        save_json(self.run_dirs.predictions / "prediction_result.json", result)
        return result
