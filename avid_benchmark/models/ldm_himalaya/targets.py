from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

from ...data.index import AVIDMetadataIndex
from .config import LDMHimalayaConfig


def _cache_name(path: Path) -> str:
    from ..cache_keys import image_content_key
    return image_content_key(path)


class StableDiffusionV14TargetCache:
    def __init__(self, cache_dir: Path, config: LDMHimalayaConfig, device: torch.device):
        self.cache_dir = cache_dir
        self.config = config
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
            ]
        )
        self._pipe = None
        self._reported_status = False
        self._prompt_cache: dict[str, torch.Tensor] = {}
        self._latent_cache: dict[str, torch.Tensor] = {}
        self._negative_prompt_cache: dict[str, torch.Tensor] = {}
        self._fixed_init_image: Image.Image | None = None

    def available(self) -> bool:
        return bool(self.config.stable_diffusion_model_path) and Path(self.config.stable_diffusion_model_path).exists()

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self._pipe is not None:
            self._pipe.to(device)

    def _dtype(self) -> torch.dtype:
        return torch.float16 if self.device.type == "cuda" else torch.float32

    def _load_pipe(self):
        if not self.available():
            raise RuntimeError(
                f"{self.config.stable_diffusion_variant} diffusion assets are required for ldm_himalaya. "
                f"Missing path: {self.config.stable_diffusion_model_path}"
            )
        if self._pipe is None:
            from diffusers import StableDiffusionImg2ImgPipeline

            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config.stable_diffusion_model_path,
                torch_dtype=self._dtype(),
                use_safetensors=False,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=True,
            )
            pipe.set_progress_bar_config(disable=True)
            pipe = pipe.to(self.device)
            pipe.vae.eval().requires_grad_(False)
            pipe.text_encoder.eval().requires_grad_(False)
            pipe.unet.eval().requires_grad_(False)
            self._pipe = pipe
        return self._pipe

    def _prompt_for_image(self, image_name: str, metadata: AVIDMetadataIndex) -> str:
        row = metadata.get(image_name)
        prompt_field = str(self.config.prompt_field).strip() or "translated_caption"
        if prompt_field == "caption":
            return row.get("caption") or row.get("translated_caption") or image_name
        if prompt_field == "translated_caption":
            return row.get("translated_caption") or row.get("caption") or image_name
        return row.get(prompt_field) or row.get("translated_caption") or row.get("caption") or image_name

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        pipe = self._load_pipe()
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = getattr(tokens, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            encoded = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        return encoded[0].detach().float().cpu()

    def _encode_latent(self, image_path: Path) -> torch.Tensor:
        pipe = self._load_pipe()
        vae_dtype = next(pipe.vae.parameters()).dtype
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device, dtype=vae_dtype)
        scaling_factor = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
        with torch.no_grad():
            latent = pipe.vae.encode(2 * tensor - 1).latent_dist.mode() * scaling_factor
        return latent[0].detach().float().cpu()

    def encode_prompt_image_names(self, image_names: Iterable[str], metadata: AVIDMetadataIndex) -> torch.Tensor:
        ordered_names = list(image_names)
        prompt_outputs: list[torch.Tensor] = []
        missing = 0
        iterator = tqdm(
            ordered_names,
            total=len(ordered_names),
            desc="[LDMHimalaya] Prompt targets",
            unit="image",
            dynamic_ncols=True,
            leave=True,
        )
        for image_name in iterator:
            image_path = metadata.gt_image_path(image_name)
            cache_key = _cache_name(image_path) + ':' + self._prompt_for_image(image_name, metadata)
            if cache_key not in self._prompt_cache:
                missing += 1
                self._prompt_cache[cache_key] = self._encode_prompt(self._prompt_for_image(image_name, metadata))
            prompt_outputs.append(self._prompt_cache[cache_key])
        if not self._reported_status:
            if missing > 0:
                print(f"[LDMHimalaya] Caching {missing} Stable Diffusion prompt target(s) in memory", flush=True)
            else:
                print("[LDMHimalaya] Reusing in-memory Stable Diffusion prompt targets", flush=True)
            self._reported_status = True
        return torch.stack(prompt_outputs, dim=0)

    def encode_image_names(self, image_names: Iterable[str], metadata: AVIDMetadataIndex) -> tuple[torch.Tensor, torch.Tensor]:
        ordered_names = list(image_names)
        prompt_outputs: list[torch.Tensor] = []
        latent_outputs: list[torch.Tensor] = []
        missing = 0
        iterator = tqdm(
            ordered_names,
            total=len(ordered_names),
            desc="[LDMHimalaya] Target cache",
            unit="image",
            dynamic_ncols=True,
            leave=True,
        )
        for image_name in iterator:
            image_path = metadata.gt_image_path(image_name)
            cache_key = _cache_name(image_path) + ':' + self._prompt_for_image(image_name, metadata)
            if cache_key not in self._prompt_cache:
                missing += 1
                self._prompt_cache[cache_key] = self._encode_prompt(self._prompt_for_image(image_name, metadata))
            if cache_key not in self._latent_cache:
                self._latent_cache[cache_key] = self._encode_latent(image_path)
            prompt_outputs.append(self._prompt_cache[cache_key])
            latent_outputs.append(self._latent_cache[cache_key])
        if not self._reported_status:
            if missing > 0:
                print(f"[LDMHimalaya] Caching {missing} Stable Diffusion prompt/latent target(s) in memory", flush=True)
            else:
                print("[LDMHimalaya] Reusing in-memory Stable Diffusion prompt/latent targets", flush=True)
            self._reported_status = True
        return torch.stack(prompt_outputs, dim=0), torch.stack(latent_outputs, dim=0)

    def negative_prompt_embeds(self, batch_size: int) -> torch.Tensor:
        prompt = self.config.negative_prompt
        if prompt not in self._negative_prompt_cache:
            self._negative_prompt_cache[prompt] = self._encode_prompt(prompt)
        base = self._negative_prompt_cache[prompt].unsqueeze(0)
        return base.repeat(batch_size, 1, 1).to(self.device, dtype=self._dtype())

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        pipe = self._load_pipe()
        scaling_factor = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
        latents = latents.to(self.device, dtype=self._dtype())
        with torch.no_grad():
            decoded = pipe.vae.decode(latents / scaling_factor).sample
        images = torch.clamp((decoded.float() + 1.0) / 2.0, min=0.0, max=1.0)
        return images.detach().cpu()

    def generate_images(self, prompt_embeds: torch.Tensor, latents: torch.Tensor | None = None) -> list[Image.Image]:
        pipe = self._load_pipe()
        prompt_embeds = prompt_embeds.to(self.device, dtype=self._dtype())
        negative_prompt_embeds = self.negative_prompt_embeds(len(prompt_embeds))
        images: list[Image.Image] = []
        init_image: Image.Image | None = None
        if latents is None:
            if self._fixed_init_image is None:
                self._fixed_init_image = Image.new(
                    "RGB",
                    (self.config.image_size, self.config.image_size),
                    color=(127, 127, 127),
                )
            init_image = self._fixed_init_image
        else:
            latents = latents.to(self.device, dtype=self._dtype())
        for idx in range(len(prompt_embeds)):
            image_input = init_image if init_image is not None else latents[idx : idx + 1]
            result = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds[idx : idx + 1],
                negative_prompt_embeds=negative_prompt_embeds[idx : idx + 1],
                image=image_input,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                strength=self.config.strength,
                output_type="pil",
            )
            images.append(result.images[0].copy())
        return images


def pil_to_uint8_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
