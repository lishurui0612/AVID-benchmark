from __future__ import annotations

import contextlib
import hashlib
import io
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ...data.index import AVIDMetadataIndex
from .config import MindEye2Config
from .legacy import diffusion_engine_cls, git_caption_model_cls, legacy_utils_module, omegaconf_module, openclip_image_embedder_cls


_ENHANCED_SDXL_ENGINE_CACHE: dict[tuple[str, str, str, str], object] = {}


def _cache_name(path: Path) -> str:
    from ..cache_keys import image_content_key
    return image_content_key(path)


def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    while x.ndim < target_dims:
        x = x[..., None]
    return x


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _is_main_process() -> bool:
    return _distributed_rank() == 0


@contextlib.contextmanager
def _quiet_context():
    if _is_main_process():
        yield
        return
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def _resolved_openclip_source(config: MindEye2Config) -> str:
    if config.openclip_pretrained_path and Path(config.openclip_pretrained_path).exists():
        return config.openclip_pretrained_path
    if config.openclip_cache_dir:
        local_bin = Path(config.openclip_cache_dir) / "open_clip_pytorch_model.bin"
        if local_bin.exists():
            return str(local_bin.resolve())
    raise FileNotFoundError(
        "MindEye2 requires local OpenCLIP weights, but neither model.openclip_pretrained_path "
        "nor model.openclip_cache_dir/open_clip_pytorch_model.bin exists. "
        f"Configured cache_dir={config.openclip_cache_dir!r}, pretrained_path={config.openclip_pretrained_path!r}."
    )


def _patch_conditioner_tree(node: object, config: MindEye2Config) -> None:
    if isinstance(node, dict):
        target = str(node.get("target", ""))
        params = node.setdefault("params", {}) if target else None
        if target.endswith("FrozenOpenCLIPImageEmbedder") and params is not None:
            params["version"] = _resolved_openclip_source(config)
            if config.openclip_cache_dir:
                params["cache_dir"] = config.openclip_cache_dir
        elif target.endswith("FrozenOpenCLIPEmbedder2") and params is not None:
            params["version"] = _resolved_openclip_source(config)
        elif target.endswith("FrozenCLIPEmbedder") and params is not None and config.openai_clip_text_model_path:
            params["version"] = config.openai_clip_text_model_path
        for value in node.values():
            _patch_conditioner_tree(value, config)
    elif isinstance(node, list):
        for value in node:
            _patch_conditioner_tree(value, config)


def _load_patched_diffusion_params(config: MindEye2Config) -> dict:
    OmegaConf = omegaconf_module().OmegaConf
    payload = OmegaConf.load(config.unclip_config_path)
    payload = OmegaConf.to_container(payload, resolve=True)
    params = payload["model"]["params"]
    _patch_conditioner_tree(params.get("conditioner_config", {}), config)
    params["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"
    params["sampler_config"]["params"]["num_steps"] = config.enhanced_num_steps
    if config.sdxl_base_ckpt_path:
        params.setdefault("ckpt_config", {}).setdefault("params", {})["ckpt_path"] = config.sdxl_base_ckpt_path
    return params


def _base_sdxl_config_path(config: MindEye2Config) -> Path:
    return Path(config.unclip_config_path).resolve().parent / "inference" / "sd_xl_base.yaml"


def _load_patched_enhanced_sdxl_params(config: MindEye2Config) -> dict:
    OmegaConf = omegaconf_module().OmegaConf

    unclip_payload = OmegaConf.load(config.unclip_config_path)
    unclip_payload = OmegaConf.to_container(unclip_payload, resolve=True)
    unclip_params = unclip_payload["model"]["params"]
    sampler_config = unclip_params["sampler_config"]
    sampler_config["params"]["num_steps"] = config.enhanced_num_steps

    base_payload = OmegaConf.load(_base_sdxl_config_path(config))
    base_payload = OmegaConf.to_container(base_payload, resolve=True)
    base_params = base_payload["model"]["params"]
    _patch_conditioner_tree(base_params.get("conditioner_config", {}), config)

    return {
        "network_config": base_params["network_config"],
        "denoiser_config": base_params["denoiser_config"],
        "first_stage_config": base_params["first_stage_config"],
        "conditioner_config": base_params["conditioner_config"],
        "sampler_config": sampler_config,
        "scale_factor": base_params["scale_factor"],
        "disable_first_stage_autocast": base_params["disable_first_stage_autocast"],
    }


def _enhanced_sdxl_engine_key(config: MindEye2Config, device: torch.device) -> tuple[str, str, str, str]:
    return (
        str(Path(config.unclip_config_path).resolve()),
        str(Path(config.sdxl_base_ckpt_path).resolve()),
        str(device),
        str(_distributed_rank()),
    )


def _get_or_create_shared_enhanced_sdxl_engine(config: MindEye2Config, device: torch.device):
    key = _enhanced_sdxl_engine_key(config, device)
    cached = _ENHANCED_SDXL_ENGINE_CACHE.get(key)
    if cached is not None:
        return cached

    if _is_main_process():
        print(
            f"[MindEye2Text] Loading shared enhanced SDXL engine from {config.unclip_config_path} "
            f"with base checkpoint {config.sdxl_base_ckpt_path}",
            flush=True,
        )
    DiffusionEngine = diffusion_engine_cls()
    params = _load_patched_enhanced_sdxl_params(config)
    with _quiet_context():
        engine = DiffusionEngine(
            network_config=params["network_config"],
            denoiser_config=params["denoiser_config"],
            first_stage_config=params["first_stage_config"],
            conditioner_config=params["conditioner_config"],
            sampler_config=params["sampler_config"],
            scale_factor=params["scale_factor"],
            disable_first_stage_autocast=params["disable_first_stage_autocast"],
            ckpt_path=config.sdxl_base_ckpt_path,
        )
    engine.eval().requires_grad_(False)
    engine.to(device)
    _ENHANCED_SDXL_ENGINE_CACHE[key] = engine
    return engine


def _release_shared_enhanced_sdxl_engine(config: MindEye2Config, device: torch.device) -> None:
    _ENHANCED_SDXL_ENGINE_CACHE.pop(_enhanced_sdxl_engine_key(config, device), None)


class OpenClipTokenCache:
    def __init__(self, cache_dir: Path, config: MindEye2Config, device: torch.device):
        self.config = config
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
            ]
        )
        self._embedder = None
        self._reported_status = False
        self._memory_cache: dict[str, torch.Tensor] = {}

    def _get_embedder(self):
        if self._embedder is None:
            pretrained = _resolved_openclip_source(self.config)
            if _is_main_process():
                print(
                    f"[MindEye2] Loading OpenCLIP image embedder: arch={self.config.openclip_arch}, "
                    f"weights={pretrained}",
                    flush=True,
                )
            embedder_cls = openclip_image_embedder_cls()
            self._embedder = embedder_cls(
                arch=self.config.openclip_arch,
                version=pretrained,
                output_tokens=True,
                only_tokens=True,
                cache_dir=self.config.openclip_cache_dir or None,
            )
            self._embedder.to(self.device)
            self._embedder.eval()
        return self._embedder

    def _encode_path(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            tensor = tensor * 2.0 - 1.0
        embedder = self._get_embedder()
        with torch.no_grad():
            encoded = embedder(tensor).float().cpu()[0]
        return encoded

    def encode_image_tensors(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected image tensor batch of shape [B,C,H,W], got {tuple(images.shape)}")
        resized = F.interpolate(
            images.to(self.device, dtype=torch.float32),
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized * 2.0 - 1.0
        embedder = self._get_embedder()
        with torch.no_grad():
            encoded = embedder(resized).float().cpu()
        return encoded

    def encode_image_names(self, image_names: Iterable[str], metadata: AVIDMetadataIndex) -> torch.Tensor:
        outputs = []
        missing = 0
        for image_name in image_names:
            image_path = metadata.gt_image_path(image_name)
            cache_key = _cache_name(image_path)
            if cache_key in self._memory_cache:
                outputs.append(self._memory_cache[cache_key])
                continue
            missing += 1
            encoded = self._encode_path(image_path)
            self._memory_cache[cache_key] = encoded
            outputs.append(encoded)
        if not self._reported_status:
            if missing > 0:
                print(f"[MindEye2] Caching {missing} OpenCLIP target embedding(s) in memory", flush=True)
            else:
                print("[MindEye2] Reusing in-memory OpenCLIP target embeddings", flush=True)
            self._reported_status = True
        return torch.stack(outputs, dim=0)

    def encode_paths(self, image_paths: Iterable[str | Path]) -> torch.Tensor:
        """Encode arbitrary candidate images with the training target encoder."""

        outputs = []
        missing = 0
        for value in image_paths:
            image_path = Path(value).expanduser().resolve()
            cache_key = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()
            if cache_key in self._memory_cache:
                outputs.append(self._memory_cache[cache_key])
                continue
            missing += 1
            encoded = self._encode_path(image_path)
            self._memory_cache[cache_key] = encoded
            outputs.append(encoded)
        if not outputs:
            raise ValueError("encode_paths requires at least one image path.")
        if not self._reported_status:
            print(
                f"[MindEye2] Encoded {missing} arbitrary candidate image(s)",
                flush=True,
            )
            self._reported_status = True
        return torch.stack(outputs, dim=0)


class LowLevelImageAutoencoder:
    def __init__(self, cache_dir: Path, config: MindEye2Config, device: torch.device):
        if not config.has_vae:
            raise ValueError("MindEye2 blurry_recon requires model.vae_ckpt_path in the model config.")
        self.config = config
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
            ]
        )
        self._autoenc = None
        self._reported_cache_status = False
        self._memory_cache: dict[str, torch.Tensor] = {}

    def _load_autoencoder(self):
        if self._autoenc is not None:
            return self._autoenc
        if _is_main_process():
            print(f"[MindEye2] Loading VAE from {self.config.vae_ckpt_path}", flush=True)
        from diffusers import AutoencoderKL

        vae_path = Path(self.config.vae_ckpt_path)
        if vae_path.is_dir():
            autoenc = AutoencoderKL.from_pretrained(str(vae_path), variant="fp16", local_files_only=True)
        else:
            checkpoint = torch.load(str(vae_path), map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = checkpoint["state_dict"]
            autoenc = AutoencoderKL(
                down_block_types=(
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ),
                up_block_types=(
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ),
                block_out_channels=(128, 256, 512, 512),
                layers_per_block=2,
                sample_size=max(256, self.config.image_size),
            )
            autoenc.load_state_dict(checkpoint)
        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device=self.device, dtype=torch.float32)
        self._autoenc = autoenc
        return autoenc

    @property
    def scaling_factor(self) -> float:
        autoenc = self._load_autoencoder()
        return float(getattr(autoenc.config, "scaling_factor", 0.18215))

    def encode_image_names(self, image_names: Iterable[str], metadata: AVIDMetadataIndex) -> torch.Tensor:
        autoenc = self._load_autoencoder()
        outputs = []
        missing = 0
        for image_name in image_names:
            image_path = metadata.gt_image_path(image_name)
            cache_key = _cache_name(image_path)
            if cache_key in self._memory_cache:
                outputs.append(self._memory_cache[cache_key])
                continue
            missing += 1
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                latents = autoenc.encode(2 * tensor - 1).latent_dist.mode() * self.scaling_factor
            latent = latents[0].cpu()
            self._memory_cache[cache_key] = latent
            outputs.append(latent)
        if not self._reported_cache_status:
            if missing > 0:
                print(f"[MindEye2] Caching {missing} VAE latent target(s) in memory", flush=True)
            else:
                print("[MindEye2] Reusing in-memory VAE latents", flush=True)
            self._reported_cache_status = True
        return torch.stack(outputs, dim=0).to(self.device, dtype=torch.float32)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        autoenc = self._load_autoencoder()
        latents = latents.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            decoded = autoenc.decode(latents / self.scaling_factor).sample
        return ((decoded / 2) + 0.5).clamp(0.0, 1.0)

    def unload(self) -> None:
        self._autoenc = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class GitCaptionGenerator:
    def __init__(self, config: MindEye2Config, device: torch.device):
        self.config = config
        self.device = device
        self._processor = None
        self._model = None
        self._converter = None

    def available(self) -> bool:
        return self.config.has_git_caption_head

    def _load(self):
        if not self.available():
            raise RuntimeError("GIT caption generation requested but git_model_path/git_converter_ckpt_path are not set.")
        if self._processor is None:
            if _is_main_process():
                print(
                    f"[MindEye2] Loading GIT caption model from {self.config.git_model_path} "
                    f"and converter from {self.config.git_converter_ckpt_path}",
                    flush=True,
                )
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(self.config.git_model_path, local_files_only=True)
            model_cls = git_caption_model_cls()
            model = model_cls.from_pretrained(self.config.git_model_path, local_files_only=True)
            model.to(self.device)
            model.eval().requires_grad_(False)

            class CLIPConverter(torch.nn.Module):
                def __init__(self, clip_seq_dim: int, clip_emb_dim: int):
                    super().__init__()
                    self.linear1 = torch.nn.Linear(clip_seq_dim, 257)
                    self.linear2 = torch.nn.Linear(clip_emb_dim, 1024)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = x.permute(0, 2, 1)
                    x = self.linear1(x)
                    x = self.linear2(x.permute(0, 2, 1))
                    return x

            converter = CLIPConverter(self.config.clip_seq_dim, self.config.clip_emb_dim)
            state = torch.load(self.config.git_converter_ckpt_path, map_location="cpu")
            state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            converter.load_state_dict(state_dict, strict=True)
            converter.to(self.device)
            converter.eval()

            self._processor = processor
            self._model = model
            self._converter = converter
        return self._processor, self._model, self._converter

    def generate(self, embeddings: torch.Tensor) -> list[str]:
        processor, model, converter = self._load()
        with torch.no_grad():
            caption_embeds = converter(embeddings.to(self.device))
            generated_ids = model.generate(pixel_values=caption_embeds, max_length=self.config.max_new_tokens)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [" ".join(text.strip().split()) for text in decoded]

    def unload(self) -> None:
        self._processor = None
        self._model = None
        self._converter = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class EnhancedTextConditionCache:
    def __init__(self, cache_dir: Path, config: MindEye2Config, device: torch.device):
        self.config = config
        self.device = device
        self._engine = None
        self._reported_status = False
        self._shape_cache: tuple[tuple[int, int], int] | None = None
        self._crossattn_memory_cache: dict[str, torch.Tensor] = {}
        self._vector_memory_cache: dict[str, torch.Tensor] = {}

    def available(self) -> bool:
        return (
            bool(self.config.unclip_config_path)
            and bool(self.config.sdxl_base_ckpt_path)
            and Path(self.config.unclip_config_path).exists()
            and Path(self.config.sdxl_base_ckpt_path).exists()
        )

    def _load_engine(self):
        if not self.available():
            raise RuntimeError("mindeye2text target cache requires unclip_config_path and sdxl_base_ckpt_path.")
        if self._engine is None:
            self._engine = _get_or_create_shared_enhanced_sdxl_engine(self.config, self.device)
        return self._engine

    def _conditioning_batch(self, prompts: list[str]) -> dict[str, torch.Tensor | list[str]]:
        batch_size = len(prompts)
        return {
            "txt": prompts,
            "original_size_as_tuple": torch.ones(batch_size, 2, device=self.device) * self.config.enhanced_original_size,
            "crop_coords_top_left": torch.zeros(batch_size, 2, device=self.device),
            "target_size_as_tuple": torch.ones(batch_size, 2, device=self.device) * self.config.enhanced_target_size,
        }

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        engine = self._load_engine()
        with torch.no_grad():
            out = engine.conditioner(self._conditioning_batch(prompts))
        crossattn = out["crossattn"].float().cpu()
        vector = out["vector"].float().cpu()
        return crossattn, vector

    def infer_shapes(self, metadata: AVIDMetadataIndex) -> tuple[tuple[int, int], int]:
        if self._shape_cache is None:
            example_image = next(iter(metadata.rows_by_image))
            crossattn, vector = self.encode_image_names([example_image], metadata)
            self._shape_cache = ((int(crossattn.shape[1]), int(crossattn.shape[2])), int(vector.shape[1]))
        return self._shape_cache

    def encode_image_names(self, image_names: Iterable[str], metadata: AVIDMetadataIndex) -> tuple[torch.Tensor, torch.Tensor]:
        ordered_names = list(image_names)
        missing_names: list[str] = []
        prompts: list[str] = []

        for image_name in ordered_names:
            image_path = metadata.gt_image_path(image_name)
            stem = _cache_name(image_path)
            if stem in self._crossattn_memory_cache and stem in self._vector_memory_cache:
                continue
            prompt = metadata.caption_en(image_name) or metadata.caption_zh(image_name) or image_name
            missing_names.append(image_name)
            prompts.append(prompt)

        if missing_names:
            encoded_crossattn, encoded_vector = self._encode_prompts(prompts)
            for idx, image_name in enumerate(missing_names):
                stem = _cache_name(metadata.gt_image_path(image_name))
                crossattn_tensor = encoded_crossattn[idx]
                vector_tensor = encoded_vector[idx]
                self._crossattn_memory_cache[stem] = crossattn_tensor
                self._vector_memory_cache[stem] = vector_tensor

        crossattn_outputs = [
            self._crossattn_memory_cache[_cache_name(metadata.gt_image_path(image_name))]
            for image_name in ordered_names
        ]
        vector_outputs = [
            self._vector_memory_cache[_cache_name(metadata.gt_image_path(image_name))]
            for image_name in ordered_names
        ]

        if not self._reported_status:
            if missing_names:
                print(f"[MindEye2Text] Caching {len(missing_names)} SDXL text-conditioning target(s) in memory", flush=True)
            else:
                print("[MindEye2Text] Reusing in-memory SDXL text-conditioning targets", flush=True)
            self._reported_status = True

        crossattn = torch.stack(crossattn_outputs, dim=0)
        vector = torch.stack(vector_outputs, dim=0)
        return crossattn, vector


class UnclipImageGenerator:
    def __init__(self, config: MindEye2Config, device: torch.device):
        self.config = config
        self.device = device
        self._engine = None
        self._vector_suffix = None

    def available(self) -> bool:
        return self.config.has_unclip

    def _load(self):
        if not self.available():
            raise RuntimeError("unCLIP generation requested but unclip_config_path/unclip_ckpt_path are not set.")
        if self._engine is None:
            if _is_main_process():
                print(
                    f"[MindEye2] Loading unCLIP config from {self.config.unclip_config_path} "
                    f"and checkpoint from {self.config.unclip_ckpt_path}",
                    flush=True,
                )
            DiffusionEngine = diffusion_engine_cls()
            params = _load_patched_diffusion_params(self.config)
            with _quiet_context():
                engine = DiffusionEngine(
                    network_config=params["network_config"],
                    denoiser_config=params["denoiser_config"],
                    first_stage_config=params["first_stage_config"],
                    conditioner_config=params["conditioner_config"],
                    sampler_config=params["sampler_config"],
                    scale_factor=params["scale_factor"],
                    disable_first_stage_autocast=params["disable_first_stage_autocast"],
                )
            ckpt = torch.load(self.config.unclip_ckpt_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            engine.load_state_dict(state_dict)
            engine.eval().requires_grad_(False)
            engine.to(self.device)

            batch = {
                "jpg": torch.randn(1, 3, 1, 1, device=self.device),
                "original_size_as_tuple": torch.ones(1, 2, device=self.device) * self.config.enhanced_original_size,
                "crop_coords_top_left": torch.zeros(1, 2, device=self.device),
            }
            out = engine.conditioner(batch)
            self._vector_suffix = out["vector"].to(self.device)
            self._engine = engine
        return self._engine, self._vector_suffix

    def generate(self, embeddings: torch.Tensor) -> torch.Tensor:
        engine, vector_suffix = self._load()
        utils = legacy_utils_module()
        samples = []
        for idx in range(len(embeddings)):
            sample = utils.unclip_recon(
                embeddings[idx : idx + 1].to(self.device),
                engine,
                vector_suffix,
                num_samples=1,
            )
            samples.append(sample[0].cpu())
        return torch.stack(samples, dim=0)

    def unload(self) -> None:
        self._engine = None
        self._vector_suffix = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class EnhancedTextImageGenerator:
    def __init__(self, config: MindEye2Config, device: torch.device):
        self.config = config
        self.device = device
        self._engine = None
        self._uc = None

    def available(self) -> bool:
        return (
            bool(self.config.unclip_config_path)
            and bool(self.config.sdxl_base_ckpt_path)
            and Path(self.config.unclip_config_path).exists()
            and Path(self.config.sdxl_base_ckpt_path).exists()
        )

    def _conditioning_batch(self, prompts: list[str]) -> dict[str, torch.Tensor | list[str]]:
        batch_size = len(prompts)
        return {
            "txt": prompts,
            "original_size_as_tuple": torch.ones(batch_size, 2, device=self.device) * self.config.enhanced_original_size,
            "crop_coords_top_left": torch.zeros(batch_size, 2, device=self.device),
            "target_size_as_tuple": torch.ones(batch_size, 2, device=self.device) * self.config.enhanced_target_size,
        }

    def _load(self):
        if not self.available():
            raise RuntimeError("mindeye2text generation requested but unclip_config_path/sdxl_base_ckpt_path are not set.")
        if self._engine is None:
            engine = _get_or_create_shared_enhanced_sdxl_engine(self.config, self.device)
            if hasattr(engine.sampler, "guider") and hasattr(engine.sampler.guider, "scale"):
                engine.sampler.guider.scale = 5
            with torch.no_grad():
                out = engine.conditioner(self._conditioning_batch([self.config.enhanced_negative_prompt]))
            self._uc = {
                "crossattn": out["crossattn"].to(self.device),
                "vector": out["vector"].to(self.device),
            }
            self._engine = engine
        return self._engine, self._uc

    def generate(
        self,
        crossattn: torch.Tensor,
        vector: torch.Tensor,
        blurry_images: torch.Tensor,
        img2img_timepoint_override: int | None = None,
        progress_callback: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        engine, uc = self._load()
        base_num_steps = int(self.config.enhanced_num_steps)
        blurry_images = blurry_images.to(self.device, dtype=torch.float32)
        blurry_images = F.interpolate(
            blurry_images,
            size=(self.config.enhanced_original_size, self.config.enhanced_original_size),
            mode="bilinear",
            align_corners=False,
        )
        denoiser = lambda x, sigma, c: engine.denoiser(engine.model, x, sigma, c)
        outputs = []

        for idx in range(len(crossattn)):
            image = blurry_images[idx : idx + 1]
            original_num_steps = engine.sampler.num_steps
            try:
                engine.sampler.num_steps = base_num_steps
                with torch.no_grad():
                    z = engine.encode_first_stage(image * 2 - 1)
                    noise = torch.randn_like(z)
                    sigmas = engine.sampler.discretization(base_num_steps).to(self.device)
                    if img2img_timepoint_override is not None:
                        requested_timepoint = img2img_timepoint_override
                    elif self.config.enhanced_img2img_strength > 0:
                        requested_timepoint = round(float(self.config.enhanced_img2img_strength) * (len(sigmas) - 1))
                    else:
                        requested_timepoint = self.config.enhanced_img2img_timepoint
                    img2img_timepoint = min(max(1, requested_timepoint), len(sigmas) - 1)
                    init_z = (z + noise * _append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(
                        1.0 + sigmas[0] ** 2.0
                    )
                    sigmas = sigmas[-img2img_timepoint:].repeat(1, 1)

                    cond = {
                        "crossattn": crossattn[idx : idx + 1].to(self.device),
                        "vector": vector[idx : idx + 1].to(self.device),
                    }
                    uc_batch = {
                        "crossattn": uc["crossattn"],
                        "vector": uc["vector"],
                    }
                    sample_num_steps = sigmas.shape[-1] - 1
                    engine.sampler.num_steps = sample_num_steps
                    noised_z, _, _, _, cond, uc_batch = engine.sampler.prepare_sampling_loop(
                        init_z,
                        cond=cond,
                        uc=uc_batch,
                        num_steps=sample_num_steps,
                    )
                    for timestep in range(sample_num_steps):
                        noised_z = engine.sampler.sampler_step(
                            sigmas[:, timestep],
                            sigmas[:, timestep + 1],
                            denoiser,
                            noised_z,
                            cond=cond,
                            uc=uc_batch,
                            gamma=0,
                        )
                    samples_x = engine.decode_first_stage(noised_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            finally:
                engine.sampler.num_steps = original_num_steps
            outputs.append(samples[0].cpu())
            if progress_callback is not None:
                progress_callback()
        return torch.stack(outputs, dim=0)

    def generate_from_prompts(
        self,
        prompts: list[str],
        blurry_images: torch.Tensor | None = None,
        progress_callback: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        engine, _ = self._load()
        with torch.no_grad():
            cond = engine.conditioner(self._conditioning_batch(prompts))
        crossattn = cond["crossattn"].float()
        vector = cond["vector"].float()
        if blurry_images is None:
            return self.generate_text_only(crossattn, vector, progress_callback=progress_callback)
        return self.generate(crossattn, vector, blurry_images, progress_callback=progress_callback)

    def unload(self) -> None:
        self._engine = None
        self._uc = None
        _release_shared_enhanced_sdxl_engine(self.config, self.device)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_text_only(
        self,
        crossattn: torch.Tensor,
        vector: torch.Tensor,
        progress_callback: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        engine, uc = self._load()
        denoiser = lambda x, sigma, c: engine.denoiser(engine.model, x, sigma, c)
        latent_channels = int(getattr(getattr(engine.model, "diffusion_model", None), "in_channels", 4))
        latent_size = max(1, self.config.enhanced_target_size // 8)
        outputs = []

        for idx in range(len(crossattn)):
            with torch.no_grad():
                init_noise = torch.randn(
                    1,
                    latent_channels,
                    latent_size,
                    latent_size,
                    device=self.device,
                    dtype=torch.float32,
                )
                cond = {
                    "crossattn": crossattn[idx : idx + 1].to(self.device),
                    "vector": vector[idx : idx + 1].to(self.device),
                }
                uc_batch = {
                    "crossattn": uc["crossattn"],
                    "vector": uc["vector"],
                }
                x, s_in, sigmas, num_sigmas, cond, uc_batch = engine.sampler.prepare_sampling_loop(
                    init_noise,
                    cond=cond,
                    uc=uc_batch,
                    num_steps=engine.sampler.num_steps,
                )
                for timestep in range(num_sigmas - 1):
                    x = engine.sampler.sampler_step(
                        s_in * sigmas[timestep],
                        s_in * sigmas[timestep + 1],
                        denoiser,
                        x,
                        cond=cond,
                        uc=uc_batch,
                        gamma=0,
                    )
                samples_x = engine.decode_first_stage(x)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            outputs.append(samples[0].cpu())
            if progress_callback is not None:
                progress_callback()
        return torch.stack(outputs, dim=0)
