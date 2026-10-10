from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...data.index import AVIDMetadataIndex
from ...runtime import (
    create_prediction_output_root,
    create_run_directories,
    save_json,
    save_prediction_result,
    snapshot_config,
)
from ..base import ModelAdapter
from ..registry import register_model_adapter
from ..vendor_runtime import (
    export_prediction_manifest,
    ordered_split_image_names,
    resolve_avid_shard_path,
    resolve_stimulus_image_root,
    run_python_script,
    save_recon_tensor_payload,
)
from .config import MindEye1Config


def _bool_flag(name: str, enabled: bool) -> list[str]:
    return [f"--{name}" if enabled else f"--no-{name}"]


@register_model_adapter("mindeye1")
class MindEye1Adapter(ModelAdapter):
    def __init__(self, config: dict[str, Any]):
        self.experiment = config
        self.paths = config["paths"]
        self.train_cfg = config.get("train", {})
        self.eval_cfg = config.get("evaluation", {})
        self.model_cfg = MindEye1Config.from_dict(config.get("model", {})).resolve_paths(self.paths)
        self.run_name = self.train_cfg.get("run_name", "mindeye1_run")
        self.run_dirs = create_run_directories(self.paths.train_output_root, "mindeye1", self.run_name)
        self.vendor_root = self.paths.repo_root / "avid_benchmark" / "vendor" / "mindeye1" / "src"
        self.prediction_split = str(self.train_cfg.get("prediction_split", "val")).strip() or "val"
        self.subject_id = self._resolve_single_subject()
        self.stimulus_type = self._resolve_single_stimulus()
        self.subject_token = f"{self.subject_id:02d}"
        self.metadata = AVIDMetadataIndex(self.paths)

    @classmethod
    def load_from_config(cls, config: dict[str, Any]) -> "MindEye1Adapter":
        adapter = cls(config)
        snapshot_config(adapter.run_dirs, config)
        return adapter

    def capabilities(self) -> dict[str, Any]:
        return {
            "generated_image": True,
            "recon_images_pt": True,
            "asset_status": self.model_cfg.asset_status(),
        }

    def _resolve_single_subject(self) -> int:
        subjects = [int(value) for value in self.train_cfg.get("subjects", [])]
        if len(subjects) != 1:
            raise ValueError("mindeye1 currently supports exactly one subject per unified run.")
        return subjects[0]

    def _resolve_single_stimulus(self) -> str:
        stimulus_types = [str(value) for value in self.train_cfg.get("stimulus_types", [])]
        if len(stimulus_types) != 1:
            raise ValueError("mindeye1 currently supports exactly one stimulus type per unified run.")
        return stimulus_types[0]

    def _common_paths(self) -> tuple[Path, Path, Path]:
        train_tar_path = resolve_avid_shard_path(
            self.paths.dataset_root,
            subject_id=self.subject_id,
            stimulus_type=self.stimulus_type,
            split="train",
        )
        val_tar_path = resolve_avid_shard_path(
            self.paths.dataset_root,
            subject_id=self.subject_id,
            stimulus_type=self.stimulus_type,
            split=self.prediction_split,
        )
        image_root = resolve_stimulus_image_root(self.paths, self.stimulus_type)
        return train_tar_path, val_tar_path, image_root

    def _train_script_args(self) -> list[str]:
        train_tar_path, val_tar_path, image_root = self._common_paths()
        args = [
            "--gpu",
            str(self.model_cfg.gpu_id),
            "--model_name",
            self.run_name,
            "--sub",
            self.subject_token,
            "--subj",
            str(self.subject_id),
            "--data",
            self.stimulus_type,
            "--train_tar_path",
            str(train_tar_path),
            "--val_tar_path",
            str(val_tar_path),
            "--batch_size",
            str(self.model_cfg.batch_size),
            "--clip_variant",
            self.model_cfg.clip_variant,
            "--num_epochs",
            str(self.model_cfg.num_epochs),
            "--mixup_pct",
            str(self.model_cfg.mixup_pct),
            "--lr_scheduler_type",
            self.model_cfg.lr_scheduler_type,
            "--ckpt_interval",
            str(self.model_cfg.ckpt_interval),
            "--seed",
            str(int(self.train_cfg.get("seed", 42))),
            "--max_lr",
            str(self.model_cfg.max_lr),
            "--n_samples_save",
            str(self.model_cfg.n_samples_save),
            "--img_root",
            str(image_root),
            "--train_outdir",
            str(self.run_dirs.checkpoints),
        ]
        if self.model_cfg.vd_cache_dir:
            args.extend(["--vd_cache_dir", self.model_cfg.vd_cache_dir])
        args.extend(_bool_flag("hidden", self.model_cfg.hidden))
        args.extend(_bool_flag("norm_embs", self.model_cfg.norm_embs))
        args.extend(_bool_flag("use_image_aug", self.model_cfg.use_image_aug))
        args.extend(_bool_flag("prior", self.model_cfg.prior))
        args.extend(_bool_flag("v2c", self.model_cfg.v2c))
        args.extend(_bool_flag("plot_umap", self.model_cfg.plot_umap))
        args.extend(_bool_flag("use_projector", self.model_cfg.use_projector))
        return args

    def train(self) -> dict[str, Any]:
        if self.train_cfg.get('component') == 'lowlevel':
            from .lowlevel_train import train_lowlevel
            return train_lowlevel(self)
        run_python_script(self.vendor_root / "Train_MindEye_1.py", args=self._train_script_args(), cwd=self.vendor_root)
        result = {
            "model_name": "mindeye1",
            "run_name": self.run_name,
            "subject_id": self.subject_id,
            "stimulus_type": self.stimulus_type,
            "run_root": str(self.run_dirs.root),
            "checkpoint_dir": str(self.run_dirs.checkpoints),
            "best_checkpoint": str((self.run_dirs.checkpoints / "best.pth").resolve()),
            "last_checkpoint": str((self.run_dirs.checkpoints / "last.pth").resolve()),
            "asset_status": self.model_cfg.asset_status(),
        }
        save_json(self.run_dirs.logs / "train_result.json", result)
        return result

    def predict_batch(self, checkpoint_path: str | None = None) -> dict[str, Any]:
        if self.model_cfg.img2img_strength < 1 and not self.model_cfg.asset_status()['autoencoder_checkpoint']['ready']:
            raise FileNotFoundError('img2img requires model.autoencoder_checkpoint_path for this AVID participant; set img2img_strength=1.0 for the high-level pipeline.')
        _, val_tar_path, image_root = self._common_paths()
        resolved_checkpoint = Path(checkpoint_path or self.run_dirs.checkpoints / f"{self.model_cfg.checkpoint_tag}.pth").expanduser().resolve()
        prediction_root = create_prediction_output_root(
            self.run_dirs.predictions,
            f"mindeye1_{resolved_checkpoint.stem}",
            self.prediction_split,
        )
        recon_images_path = prediction_root / "recon_images.pt"
        all_images_path = prediction_root / "all_images.pt"
        args = [
            "--gpu",
            str(self.model_cfg.gpu_id),
            "--model_name",
            self.run_name,
            "--sub",
            self.subject_token,
            "--subj",
            str(self.subject_id),
            "--data",
            self.stimulus_type,
            "--val_tar_path",
            str(val_tar_path),
            "--pth",
            self.model_cfg.checkpoint_tag,
            "--img2img_strength",
            str(self.model_cfg.img2img_strength),
            "--recons_per_sample",
            str(self.model_cfg.recons_per_sample),
            "--img_root",
            str(image_root),
            "--train_outdir",
            str(self.run_dirs.checkpoints),
            "--checkpoint_path",
            str(resolved_checkpoint),
            "--recon_outdir",
            str(prediction_root),
            "--save_recon_tensor_path",
            str(recon_images_path),
            "--save_all_images_path",
            str(all_images_path),
            "--num_inference_steps",
            str(self.model_cfg.num_inference_steps),
            "--guidance_scale",
            str(self.model_cfg.guidance_scale),
            "--text_image_ratio",
            str(self.model_cfg.text_image_ratio),
        ]
        if self.model_cfg.vd_cache_dir:
            args.extend(["--vd_cache_dir", self.model_cfg.vd_cache_dir])
        if self.model_cfg.autoencoder_checkpoint_path:
            args.extend(["--autoencoder_ckpt_path", self.model_cfg.autoencoder_checkpoint_path])
        max_samples = int(self.train_cfg.get("max_prediction_samples", 0))
        args.extend(["--max_samples", str(max_samples)])
        args.extend(["--seed", str(int(self.train_cfg.get("seed", 42)))])
        run_python_script(self.vendor_root / "Reconstructions_1.py", args=args, cwd=self.vendor_root)

        image_names = ordered_split_image_names(
            self.paths.dataset_root,
            subject_id=self.subject_id,
            stimulus_type=self.stimulus_type,
            split=self.prediction_split,
        )
        raw_recon_images = torch.load(recon_images_path, map_location="cpu", weights_only=False)
        if max_samples:
            image_names = image_names[:max_samples]
        recon_uint8 = save_recon_tensor_payload(
            recon_images_path,
            raw_recon_images=raw_recon_images,
            image_names=image_names,
            subject_ids=[self.subject_id] * len(image_names),
            split=self.prediction_split,
        )
        manifest_result = export_prediction_manifest(
            prediction_root,
            model_name="mindeye1",
            subject_id=self.subject_id,
            split=self.prediction_split,
            image_names=image_names,
            recon_uint8=recon_uint8,
            metadata_index=self.metadata,
            route="mindeye1_vd_reconstruction",
            generation_protocol={"candidate_selection": "brain_prediction_clip_similarity", "candidates": self.model_cfg.recons_per_sample, "seed": int(self.train_cfg.get("seed", 42)), "img2img_strength": self.model_cfg.img2img_strength},
        )

        result = {
            "model_name": "mindeye1",
            "run_name": self.run_name,
            "subject_id": self.subject_id,
            "stimulus_type": self.stimulus_type,
            "split": self.prediction_split,
            "checkpoint_path": str(resolved_checkpoint),
            "prediction_root": str(prediction_root.resolve()),
            "recon_images_path": str(recon_images_path.resolve()),
            "all_images_path": str(all_images_path.resolve()),
            "manifest_path": str(manifest_result["manifest_path"]),
            "final_image_paths": manifest_result["final_image_paths"],
            "panel_paths": manifest_result["panel_paths"],
            "num_predictions": int(manifest_result["num_predictions"]),
            "prediction_routes": ["mindeye1_vd_reconstruction"],
            "asset_status": self.model_cfg.asset_status(),
        }
        save_prediction_result(self.run_dirs.predictions, prediction_root, result)
        return result
