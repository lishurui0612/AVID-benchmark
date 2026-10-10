from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
    resnet50,
)
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import CLIPModel

from .vmi_metrics import two_way_identification_i2i as vmi_two_way_identification_i2i


@dataclass(frozen=True)
class ImageMetricConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    feature_batch_size: int = 32
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_local_files_only: bool = True
    swav_source: str = "facebookresearch/swav:main"
    swav_source_type: str = "github"
    skip_swav: bool = False
    swav_weights_path: str = ""
    alexnet_weights_path: str = ""
    inception_weights_path: str = ""
    efficientnet_b1_weights_path: str = ""

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ImageMetricConfig":
        values = cls().__dict__.copy()
        values.update(payload or {})
        return cls(**values)


def load_image_batch(image_paths: list[str | Path]) -> torch.Tensor:
    images = []
    to_tensor = transforms.ToTensor()
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(to_tensor(image.convert("RGB")))
    if not images:
        return torch.empty(0, 3, 224, 224, dtype=torch.float32)
    return torch.stack(images, dim=0)


def resize_batch(images: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def rowwise_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).sum(dim=1)
    denominator = torch.sqrt((x.square().sum(dim=1) * y.square().sum(dim=1)).clamp_min(eps))
    return numerator / denominator


def compute_pixcorr(recons: torch.Tensor, originals: torch.Tensor) -> float:
    recons_resized = resize_batch(recons, 425).reshape(recons.shape[0], -1)
    originals_resized = resize_batch(originals, 425).reshape(originals.shape[0], -1)
    return rowwise_correlation(recons_resized, originals_resized).mean().item()


def compute_ssim(recons: torch.Tensor, originals: torch.Tensor) -> float:
    recons_resized = resize_batch(recons, 425).permute(0, 2, 3, 1).cpu().numpy()
    originals_resized = resize_batch(originals, 425).permute(0, 2, 3, 1).cpu().numpy()
    recons_gray = rgb2gray(recons_resized)
    originals_gray = rgb2gray(originals_resized)
    scores = []
    for recon_gray, original_gray in zip(recons_gray, originals_gray):
        scores.append(
            structural_similarity(
                recon_gray,
                original_gray,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
        )
    return float(np.mean(scores))


def build_normalize(mean: list[float], std: list[float], device: str) -> Callable[[torch.Tensor], torch.Tensor]:
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def normalize(batch: torch.Tensor) -> torch.Tensor:
        return (batch - mean_tensor) / std_tensor

    return normalize


def extract_features(
    images: torch.Tensor,
    model_fn: Callable[[torch.Tensor], torch.Tensor | dict[str, torch.Tensor]],
    resize_size: int,
    normalize_fn: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    device: str,
) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            batch = resize_batch(batch, resize_size)
            batch = normalize_fn(batch)
            feats = model_fn(batch)
            if isinstance(feats, dict):
                if len(feats) != 1:
                    raise ValueError("Feature extractor returned multiple nodes; provide a single tensor output.")
                feats = next(iter(feats.values()))
            outputs.append(feats.float().flatten(1).detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def corrcoef_matrix(reals: np.ndarray, preds: np.ndarray) -> np.ndarray:
    matrix = np.corrcoef(reals, preds)
    return matrix[: len(reals), len(reals) :]


def two_way_identification_score(pred_features: np.ndarray, real_features: np.ndarray) -> float:
    corr = corrcoef_matrix(real_features, pred_features)
    congruents = np.diag(corr)
    success = corr < congruents
    success_count = np.sum(success, axis=0)
    return float(np.mean(success_count) / (len(real_features) - 1))


def correlation_distance_mean(pred_features: np.ndarray, real_features: np.ndarray) -> float:
    pred_centered = pred_features - pred_features.mean(axis=1, keepdims=True)
    real_centered = real_features - real_features.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * real_centered, axis=1)
    denominator = np.sqrt(
        np.clip(np.sum(pred_centered**2, axis=1) * np.sum(real_centered**2, axis=1), a_min=1e-12, a_max=None)
    )
    corr = numerator / denominator
    return float(np.mean(1.0 - corr))


def maybe_clear_cuda_cache(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_alexnet_extractor(config: ImageMetricConfig, device: str):
    weights_path = Path(config.alexnet_weights_path)
    if config.alexnet_weights_path and weights_path.exists():
        model = alexnet(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    else:
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    extractor = create_feature_extractor(model, return_nodes={"features.4": "alex2", "features.11": "alex5"}).to(device)
    extractor.eval().requires_grad_(False)
    return extractor


def load_inception_extractor(config: ImageMetricConfig, device: str):
    weights_path = Path(config.inception_weights_path)
    if config.inception_weights_path and weights_path.exists():
        model = inception_v3(weights=None, init_weights=False)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    else:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "avgpool"}).to(device)
    extractor.eval().requires_grad_(False)
    return extractor


def load_efficientnet_b1_extractor(config: ImageMetricConfig, device: str):
    weights_path = Path(config.efficientnet_b1_weights_path)
    if config.efficientnet_b1_weights_path and weights_path.exists():
        model = efficientnet_b1(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    else:
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "avgpool"}).to(device)
    extractor.eval().requires_grad_(False)
    return extractor


def load_swav_extractor(config: ImageMetricConfig, device: str):
    weights_path = Path(config.swav_weights_path)
    if config.swav_weights_path and weights_path.exists():
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        backbone_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            if normalized_key.startswith("projection_head.") or normalized_key.startswith("prototypes."):
                continue
            backbone_state_dict[normalized_key] = value
        model = resnet50(weights=None)
        missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
        allowed_missing = {"fc.weight", "fc.bias"}
        if set(missing_keys) - allowed_missing:
            raise RuntimeError(f"Unexpected missing SwAV backbone keys: {missing_keys}")
        if unexpected_keys:
            raise RuntimeError(f"Unexpected SwAV checkpoint keys: {unexpected_keys}")
        extractor = create_feature_extractor(model, return_nodes={"avgpool": "avgpool"}).to(device)
        extractor.eval().requires_grad_(False)
        return model, extractor

    hub_kwargs = {"source": "local"} if config.swav_source_type == "local" else {}
    model = torch.hub.load(config.swav_source, "resnet50", **hub_kwargs)
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "avgpool"}).to(device)
    extractor.eval().requires_grad_(False)
    return model, extractor


def evaluate_image_metrics(recons: torch.Tensor, originals: torch.Tensor, config: ImageMetricConfig | dict | None = None) -> dict[str, float]:
    resolved = config if isinstance(config, ImageMetricConfig) else ImageMetricConfig.from_dict(config)
    if recons.ndim != 4 or originals.ndim != 4:
        raise ValueError(f"Expected image batches shaped [N, C, H, W], got recon {tuple(recons.shape)} vs original {tuple(originals.shape)}")
    if recons.shape[0] != originals.shape[0] or recons.shape[1] != originals.shape[1]:
        raise ValueError(f"Batch/channel mismatch: recon {tuple(recons.shape)} vs original {tuple(originals.shape)}")
    if len(recons) < 2:
        raise ValueError("MindEye2 image metrics require at least two samples for identification-based scores.")

    device = resolved.device
    normalize_imagenet = build_normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], device)
    normalize_clip = build_normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], device)

    metrics = {
        "PixCorr": compute_pixcorr(recons, originals),
        "SSIM": compute_ssim(recons, originals),
    }

    alex_model = load_alexnet_extractor(resolved, device)
    alex2_real = extract_features(originals, lambda batch: alex_model(batch)["alex2"], 256, normalize_imagenet, resolved.feature_batch_size, device)
    alex2_pred = extract_features(recons, lambda batch: alex_model(batch)["alex2"], 256, normalize_imagenet, resolved.feature_batch_size, device)
    alex5_real = extract_features(originals, lambda batch: alex_model(batch)["alex5"], 256, normalize_imagenet, resolved.feature_batch_size, device)
    alex5_pred = extract_features(recons, lambda batch: alex_model(batch)["alex5"], 256, normalize_imagenet, resolved.feature_batch_size, device)
    metrics["AlexNet(2)"] = two_way_identification_score(alex2_pred, alex2_real)
    metrics["AlexNet(5)"] = two_way_identification_score(alex5_pred, alex5_real)
    del alex_model
    maybe_clear_cuda_cache(device)

    inception_model = load_inception_extractor(resolved, device)
    inception_real = extract_features(originals, lambda batch: inception_model(batch)["avgpool"], 342, normalize_imagenet, resolved.feature_batch_size, device)
    inception_pred = extract_features(recons, lambda batch: inception_model(batch)["avgpool"], 342, normalize_imagenet, resolved.feature_batch_size, device)
    metrics["InceptionV3"] = vmi_two_way_identification_i2i(inception_real, inception_pred)
    del inception_model
    maybe_clear_cuda_cache(device)

    clip_model = CLIPModel.from_pretrained(resolved.clip_model_name, local_files_only=resolved.clip_local_files_only).to(device)
    clip_model.eval()
    clip_real = extract_features(originals, lambda batch: clip_model.get_image_features(pixel_values=batch), 224, normalize_clip, resolved.feature_batch_size, device)
    clip_pred = extract_features(recons, lambda batch: clip_model.get_image_features(pixel_values=batch), 224, normalize_clip, resolved.feature_batch_size, device)
    metrics["CLIP"] = vmi_two_way_identification_i2i(clip_real, clip_pred)
    del clip_model
    maybe_clear_cuda_cache(device)

    eff_model = load_efficientnet_b1_extractor(resolved, device)
    eff_real = extract_features(originals, lambda batch: eff_model(batch)["avgpool"], 255, normalize_imagenet, resolved.feature_batch_size, device)
    eff_pred = extract_features(recons, lambda batch: eff_model(batch)["avgpool"], 255, normalize_imagenet, resolved.feature_batch_size, device)
    metrics["EffNet-B"] = correlation_distance_mean(eff_pred, eff_real)
    del eff_model
    maybe_clear_cuda_cache(device)

    if resolved.skip_swav:
        metrics["SwAV"] = float("nan")
    else:
        try:
            swav_base, swav_model = load_swav_extractor(resolved, device)
            swav_real = extract_features(originals, lambda batch: swav_model(batch)["avgpool"], 224, normalize_imagenet, resolved.feature_batch_size, device)
            swav_pred = extract_features(recons, lambda batch: swav_model(batch)["avgpool"], 224, normalize_imagenet, resolved.feature_batch_size, device)
            metrics["SwAV"] = correlation_distance_mean(swav_pred, swav_real)
            del swav_base, swav_model
            maybe_clear_cuda_cache(device)
        except Exception:
            metrics["SwAV"] = float("nan")

    return metrics
