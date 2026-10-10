from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...types import PathConfig


def _vendored_unclip_config_path(repo_root: Path) -> Path:
    return repo_root / "avid_benchmark" / "vendor" / "mindeye2" / "src" / "generative_models" / "configs" / "unclip6.yaml"


@dataclass(frozen=True)
class MindEye2Config:
    hidden_dim: int = 4096
    n_blocks: int = 4
    clip_seq_dim: int = 256
    clip_emb_dim: int = 1664
    use_prior: bool = True
    clip_scale: float = 1.0
    crossattn_mse_scale: float = 0.0
    crossattn_cos_scale: float = 0.0
    prior_scale: float = 30.0
    blurry_recon: bool = True
    blur_scale: float = 0.5
    max_lr: float = 3e-4
    lr_scheduler_type: str = "cycle"
    batch_size: int = 24
    per_subject_batch_size: int = 0
    prediction_batch_size: int = 0
    num_epochs: int = 150
    loo_finetune_epochs: int = 15
    loo_finetune_lr: float = 3e-4
    loo_finetune_lr_scheduler_type: str = "none"
    loo_finetune_mixup_pct: float = 0.0
    steps_per_epoch: int = 0
    mixup_pct: float = 0.33
    train_response_reduce: str = "random"
    eval_response_reduce: str = "mean"
    prior_depth: int = 6
    prior_dim_head: int = 52
    prior_heads: int = 0
    prior_timesteps: int = 100
    eval_prior_timesteps: int = 20
    num_workers: int = 4
    image_size: int = 224
    use_image_aug: bool = False
    openclip_arch: str = "ViT-bigG-14"
    openclip_version: str = "laion2b_s39b_b160k"
    openclip_cache_dir: str = ""
    openclip_pretrained_path: str = ""
    openai_clip_text_model_path: str = ""
    sdxl_base_ckpt_path: str = ""
    vae_ckpt_path: str = ""
    git_model_path: str = ""
    git_converter_ckpt_path: str = ""
    unclip_config_path: str = ""
    unclip_ckpt_path: str = ""
    enhanced_original_size: int = 768
    enhanced_target_size: int = 1024
    enhanced_img2img_strength: float = 0.0
    enhanced_img2img_timepoint: int = 13
    enhanced_num_steps: int = 38
    caption_image_refinement: bool = True
    caption_sdxl_blend_weight: float = 0.8
    enhanced_negative_prompt: str = (
        "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, "
        "deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, "
        "skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, "
        "ugly face, distorted face, extra legs, anime"
    )
    max_new_tokens: int = 32

    @classmethod
    def from_dict(cls, payload: dict) -> "MindEye2Config":
        defaults = cls()
        values = defaults.__dict__.copy()
        values.update(payload or {})
        return cls(**values)

    @property
    def has_git_caption_head(self) -> bool:
        return self._path_exists(self.git_model_path) and self._path_exists(self.git_converter_ckpt_path)

    @property
    def has_vae(self) -> bool:
        return self._path_exists(self.vae_ckpt_path)

    @property
    def has_unclip(self) -> bool:
        return self._path_exists(self.unclip_config_path) and self._path_exists(self.unclip_ckpt_path)

    @staticmethod
    def _path_exists(value: str) -> bool:
        return bool(value) and Path(value).exists()

    @staticmethod
    def _first_existing(candidates: list[Path]) -> str:
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return ""

    @classmethod
    def _autofill_known_local_paths(cls, values: dict, paths: PathConfig | Path) -> dict:
        repo_root = paths.repo_root if isinstance(paths, PathConfig) and paths.repo_root is not None else Path(paths)
        asset_root = paths.asset_root if isinstance(paths, PathConfig) else None
        def _asset(*parts: str) -> Path | None:
            if asset_root is None:
                return None
            return asset_root.joinpath(*parts)
        default_candidates = {
            "vae_ckpt_path": [_asset("SDXL_VAE")] if _asset("SDXL_VAE") else [],
            "openclip_cache_dir": [_asset("openclip_bigg_laion2b_s39b_b160k")] if _asset("openclip_bigg_laion2b_s39b_b160k") else [],
            "openclip_pretrained_path": (
                [_asset("openclip_bigg_laion2b_s39b_b160k", "open_clip_pytorch_model.bin")]
                if _asset("openclip_bigg_laion2b_s39b_b160k", "open_clip_pytorch_model.bin")
                else []
            ),
            "openai_clip_text_model_path": (
                [_asset("openai_clip_vit_large_patch14", "clip_reranker")]
                if _asset("openai_clip_vit_large_patch14", "clip_reranker")
                else []
            ),
            "sdxl_base_ckpt_path": (
                [_asset("mindeye2_train_logs", "sd_xl_base_1.0.safetensors")]
                if _asset("mindeye2_train_logs", "sd_xl_base_1.0.safetensors")
                else []
            ),
            "git_model_path": [_asset("microsoft_git_large_coco")] if _asset("microsoft_git_large_coco") else [],
            "git_converter_ckpt_path": [_asset("bigG_to_L_epoch8.pth")] if _asset("bigG_to_L_epoch8.pth") else [],
            "unclip_config_path": [
                _vendored_unclip_config_path(repo_root),
            ],
            "unclip_ckpt_path": [_asset("unclip6_epoch0_step110000.ckpt")] if _asset("unclip6_epoch0_step110000.ckpt") else [],
        }
        resolved = dict(values)
        for key, candidates in default_candidates.items():
            if resolved.get(key):
                continue
            resolved_value = cls._first_existing(candidates)
            if resolved_value:
                resolved[key] = resolved_value
        return resolved

    def resolve_paths(self, paths: PathConfig | Path) -> "MindEye2Config":
        repo_root = paths.repo_root if isinstance(paths, PathConfig) and paths.repo_root is not None else Path(paths)

        def _resolve(value: str) -> str:
            if not value:
                return value
            path = Path(value).expanduser()
            return str(path.resolve() if path.is_absolute() else (repo_root / path).resolve())

        values = self._autofill_known_local_paths(self.__dict__, paths)
        return MindEye2Config(
            **{
                **values,
                "openclip_cache_dir": _resolve(values["openclip_cache_dir"]),
                "openclip_pretrained_path": _resolve(values["openclip_pretrained_path"]),
                "openai_clip_text_model_path": _resolve(values["openai_clip_text_model_path"]),
                "sdxl_base_ckpt_path": _resolve(values["sdxl_base_ckpt_path"]),
                "vae_ckpt_path": _resolve(values["vae_ckpt_path"]),
                "git_model_path": _resolve(values["git_model_path"]),
                "git_converter_ckpt_path": _resolve(values["git_converter_ckpt_path"]),
                "unclip_config_path": _resolve(values["unclip_config_path"]),
                "unclip_ckpt_path": _resolve(values["unclip_ckpt_path"]),
            }
        )

    def asset_status(self) -> dict[str, dict[str, str | bool]]:
        return {
            "vae": {
                "path": self.vae_ckpt_path,
                "ready": self.has_vae,
            },
            "openclip_cache": {
                "path": self.openclip_cache_dir,
                "ready": self._path_exists(self.openclip_cache_dir),
            },
            "openclip_pretrained": {
                "path": self.openclip_pretrained_path,
                "ready": self._path_exists(self.openclip_pretrained_path),
            },
            "openai_clip_text_model": {
                "path": self.openai_clip_text_model_path,
                "ready": self._path_exists(self.openai_clip_text_model_path),
            },
            "sdxl_base_ckpt": {
                "path": self.sdxl_base_ckpt_path,
                "ready": self._path_exists(self.sdxl_base_ckpt_path),
            },
            "git_model": {
                "path": self.git_model_path,
                "ready": self._path_exists(self.git_model_path),
            },
            "git_converter": {
                "path": self.git_converter_ckpt_path,
                "ready": self._path_exists(self.git_converter_ckpt_path),
            },
            "git_caption_head": {
                "path": f"{self.git_model_path} :: {self.git_converter_ckpt_path}".strip(),
                "ready": self.has_git_caption_head,
            },
            "unclip_config": {
                "path": self.unclip_config_path,
                "ready": self._path_exists(self.unclip_config_path),
            },
            "unclip_ckpt": {
                "path": self.unclip_ckpt_path,
                "ready": self._path_exists(self.unclip_ckpt_path),
            },
            "unclip": {
                "path": f"{self.unclip_config_path} :: {self.unclip_ckpt_path}".strip(),
                "ready": self.has_unclip,
            },
        }
