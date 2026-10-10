from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...types import PathConfig


@dataclass(frozen=True)
class MindBridgeConfig:
    batch_size: int = 4
    val_batch_size: int = 4
    num_epochs: int = 600
    eval_interval: int = 10
    ckpt_interval: int = 10
    clip_variant: str = "ViT-L/14"
    norm_embs: bool = True
    use_image_aug: bool = True
    lr_scheduler_type: str = "cycle"
    h_size: int = 2048
    n_blocks: int = 4
    num_workers: int = 1
    repeat_num: int = 2
    max_lr: float = 1.5e-4
    pool_num: int = 8192
    pool_type: str = "max"
    mse_mult: float = 1e4
    rec_mult: float = 1.0
    cyc_mult: float = 1.0
    gpu_id: int = 0
    vd_cache_dir: str = ""
    checkpoint_tag: str = "last"
    autoencoder_checkpoint_path: str = ""
    img2img_strength: float = 0.85
    guidance_scale: float = 3.5
    num_inference_steps: int = 20
    recons_per_sample: int = 16
    text_image_ratio: float = 0.5
    plotting: bool = True
    only_embeddings: bool = False
    verbose: bool = False

    @classmethod
    def from_dict(cls, payload: dict) -> "MindBridgeConfig":
        defaults = cls()
        values = defaults.__dict__.copy()
        values.update(payload or {})
        return cls(**values)

    @staticmethod
    def _first_existing(candidates: list[Path]) -> str:
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return ""

    def resolve_paths(self, paths: PathConfig | Path) -> "MindBridgeConfig":
        repo_root = paths.repo_root if isinstance(paths, PathConfig) and paths.repo_root is not None else Path(paths)
        asset_root = paths.asset_root if isinstance(paths, PathConfig) else None

        def _asset(*parts: str) -> Path | None:
            if asset_root is None:
                return None
            return asset_root.joinpath(*parts)

        def _resolve(value: str) -> str:
            if not value:
                return value
            path = Path(value).expanduser()
            return str(path.resolve() if path.is_absolute() else (repo_root / path).resolve())

        values = dict(self.__dict__)
        if not values.get("vd_cache_dir"):
            values["vd_cache_dir"] = self._first_existing(
                [
                    candidate
                    for candidate in (
                        _asset("models--shi-labs--versatile-diffusion"),
                        _asset("versatile-diffusion"),
                        _asset("vd"),
                    )
                    if candidate is not None
                ]
            )
        return MindBridgeConfig(
            **{
                **values,
                "vd_cache_dir": _resolve(values["vd_cache_dir"]),
                "autoencoder_checkpoint_path": _resolve(values["autoencoder_checkpoint_path"]),
            }
        )

    @staticmethod
    def _path_exists(value: str) -> bool:
        return bool(value) and Path(value).exists()

    def asset_status(self) -> dict[str, dict[str, str | bool]]:
        return {
            "versatile_diffusion_cache": {
                "path": self.vd_cache_dir,
                "ready": self._path_exists(self.vd_cache_dir),
            },
            "autoencoder_checkpoint": {
                "path": self.autoencoder_checkpoint_path,
                "ready": self._path_exists(self.autoencoder_checkpoint_path),
            },
        }
