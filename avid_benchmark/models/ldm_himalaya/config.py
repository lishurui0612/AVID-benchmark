from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...types import PathConfig

STABLE_DIFFUSION_VARIANTS = {
    "sdv14": {
        "prompt_field": "translated_caption",
    },
    "taiyi_zh": {
        "prompt_field": "caption",
    },
}


@dataclass(frozen=True)
class LDMHimalayaConfig:
    batch_size: int = 4
    num_workers: int = 1
    train_response_reduce: str = "mean"
    eval_response_reduce: str = "mean"
    stable_diffusion_variant: str = "sdv14"
    stable_diffusion_model_path: str = ""
    prompt_field: str = "translated_caption"
    image_size: int = 512
    openclip_arch: str = "ViT-bigG-14"
    openclip_version: str = "laion2b_s39b_b160k"
    openclip_cache_dir: str = ""
    openclip_pretrained_path: str = ""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 0.8
    generation_use_predicted_latent: bool = True
    negative_prompt: str = ""
    enable_text_variance_alignment: bool = False
    ridge_chunk_size: int = 30000
    ridge_use_preprocess: bool = False
    alphas: tuple[float, ...] = (
        1e-10,
        5e-10,
        1e-9,
        5e-9,
        1e-8,
        5e-8,
        1e-7,
        5e-7,
        1e-6,
        5e-6,
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        5e-1,
        1.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
        5e3,
        1e4,
        5e4,
        1e5,
        5e5,
        1e6,
    )
    cv: int = 5
    fit_intercept: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LDMHimalayaConfig":
        defaults = cls()
        values = defaults.__dict__.copy()
        payload = payload or {}
        variant = str(payload.get("stable_diffusion_variant", values["stable_diffusion_variant"])).strip()
        if variant not in STABLE_DIFFUSION_VARIANTS:
            allowed = ", ".join(sorted(STABLE_DIFFUSION_VARIANTS))
            raise ValueError(f"Unknown stable_diffusion_variant {variant!r}; expected one of: {allowed}")
        values.update(STABLE_DIFFUSION_VARIANTS[variant])
        values.update(payload)
        values["stable_diffusion_variant"] = variant
        values["alphas"] = tuple(float(value) for value in values.get("alphas", defaults.alphas))
        return cls(**values)

    @staticmethod
    def _path_exists(value: str) -> bool:
        return bool(value) and Path(value).exists()

    def resolve_paths(self, paths: PathConfig | Path) -> "LDMHimalayaConfig":
        repo_root = paths.repo_root if isinstance(paths, PathConfig) and paths.repo_root is not None else Path(paths)
        asset_root = paths.asset_root if isinstance(paths, PathConfig) else None
        taiyi_model_root = paths.taiyi_model_root if isinstance(paths, PathConfig) else None

        def _first_existing(candidates: list[Path]) -> str:
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate.resolve())
            return ""

        def _resolve(value: str) -> str:
            if not value:
                return value
            path = Path(value).expanduser()
            if path.is_absolute():
                return str(path.resolve())
            return str((repo_root / path).resolve())

        values = dict(self.__dict__)
        if not values.get("openclip_cache_dir"):
            candidates = [asset_root / "openclip_bigg_laion2b_s39b_b160k"] if asset_root is not None else []
            resolved = _first_existing(candidates)
            if resolved:
                values["openclip_cache_dir"] = resolved
        if not values.get("openclip_pretrained_path"):
            candidates = (
                [asset_root / "openclip_bigg_laion2b_s39b_b160k" / "open_clip_pytorch_model.bin"]
                if asset_root is not None
                else []
            )
            resolved = _first_existing(candidates)
            if resolved:
                values["openclip_pretrained_path"] = resolved
        if not values.get("stable_diffusion_model_path"):
            candidates: list[Path] = []
            if self.stable_diffusion_variant == "sdv14" and asset_root is not None:
                candidates.append(asset_root / "stable-diffusion-v1-4")
            if self.stable_diffusion_variant == "taiyi_zh":
                if taiyi_model_root is not None:
                    candidates.append(taiyi_model_root)
                if asset_root is not None:
                    candidates.append(asset_root / "taiyi-stable-diffusion-1b-chinese-en-v0_1")
            resolved = _first_existing(candidates)
            if resolved:
                values["stable_diffusion_model_path"] = resolved

        return LDMHimalayaConfig(
            **{
                **values,
                "stable_diffusion_model_path": _resolve(values["stable_diffusion_model_path"]),
                "openclip_cache_dir": _resolve(values["openclip_cache_dir"]),
                "openclip_pretrained_path": _resolve(values["openclip_pretrained_path"]),
            }
        )

    def asset_status(self) -> dict[str, dict[str, str | bool]]:
        return {
            "stable_diffusion_variant": {
                "path": self.stable_diffusion_variant,
                "ready": self.stable_diffusion_variant in STABLE_DIFFUSION_VARIANTS,
            },
            "stable_diffusion_model": {
                "path": self.stable_diffusion_model_path,
                "ready": self._path_exists(self.stable_diffusion_model_path),
            },
            "openclip_cache": {
                "path": self.openclip_cache_dir,
                "ready": self._path_exists(self.openclip_cache_dir),
            },
            "openclip_pretrained": {
                "path": self.openclip_pretrained_path,
                "ready": self._path_exists(self.openclip_pretrained_path),
            },
            "himalaya_backend": {
                "path": "torch_cuda",
                "ready": False,
            },
        }
