from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModel, AutoProcessor, CLIPModel, SiglipModel


@dataclass(frozen=True)
class VMITextMetricConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_local_files_only: bool = True
    siglip_model_name: str = ""
    siglip_local_files_only: bool = True
    altclip_model_name: str = ""
    altclip_local_files_only: bool = True
    altclip_trust_remote_code: bool = True
    tifa_question_bank_dir: str = ""
    tifa_questions_json: str = ""
    tifa_max_samples: int | None = None
    tifa_mplug_model_path: str = ""
    tifa_sbert_model_path: str = ""
    skip_siglip: bool = False
    skip_altclip: bool = False
    skip_tifa: bool = False

    @classmethod
    def from_dict(cls, payload: dict | None) -> "VMITextMetricConfig":
        values = cls().__dict__.copy()
        values.update(payload or {})
        return cls(**values)


def iter_slices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)


def _to_pil_images(images: torch.Tensor) -> list[Any]:
    to_pil = transforms.ToPILImage()
    return [to_pil(image.cpu()) for image in images]


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_matrix(query_feats: torch.Tensor, target_feats: torch.Tensor) -> np.ndarray:
    query_norm = l2_normalize(query_feats.float())
    target_norm = l2_normalize(target_feats.float())
    return (query_norm @ target_norm.T).detach().cpu().numpy()


def recall_at_k(sim: np.ndarray, k: int) -> float:
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"Recall@k requires a square matrix, got {sim.shape}")
    if sim.size == 0:
        return 0.0
    k = min(k, sim.shape[0])
    topk = np.argsort(sim, axis=1)[:, ::-1][:, :k]
    labels = np.arange(sim.shape[0])[:, None]
    return float(np.any(topk == labels, axis=1).mean())


def two_way_identification_i2t(sim: np.ndarray) -> float:
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"2WC requires a square matrix, got {sim.shape}")
    n = sim.shape[0]
    if n <= 1:
        return 0.0
    diagonal = np.diag(sim)
    success_count = np.sum(sim < diagonal[:, None], axis=1)
    return float(np.mean(success_count) / (n - 1))


def two_way_identification_i2i(
    real_features: torch.Tensor | np.ndarray,
    pred_features: torch.Tensor | np.ndarray,
) -> float:
    """VMI-style image-image 2-way identification using Pearson correlation."""
    real_array = real_features.detach().cpu().float().numpy() if isinstance(real_features, torch.Tensor) else np.asarray(real_features, dtype=np.float32)
    pred_array = pred_features.detach().cpu().float().numpy() if isinstance(pred_features, torch.Tensor) else np.asarray(pred_features, dtype=np.float32)

    if real_array.ndim != 2 or pred_array.ndim != 2:
        raise ValueError("two_way_identification_i2i expects [N, D] features.")
    if real_array.shape != pred_array.shape:
        raise ValueError(f"Feature shape mismatch: real={real_array.shape} pred={pred_array.shape}")

    n = real_array.shape[0]
    if n <= 1:
        return 0.0

    corr = np.corrcoef(real_array, pred_array)
    corr = corr[:n, n:]
    diagonal = np.diag(corr)
    success_count = np.sum(corr < diagonal, axis=0)
    return float(np.mean(success_count) / (n - 1))


@torch.no_grad()
def _encode_hf_images(
    images: torch.Tensor,
    processor: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats = []
    total = (len(images) + batch_size - 1) // batch_size
    for start, end in iter_slices(len(images), batch_size):
        pil_images = _to_pil_images(images[start:end])
        inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("Processor output has no pixel_values.")
        pixel_values = pixel_values.to(device)
        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(pixel_values=pixel_values)
        else:
            raw_outputs = model(pixel_values=pixel_values)
            outputs = getattr(raw_outputs, "image_embeds", None)
            if outputs is None:
                raise RuntimeError("Model has no get_image_features and no image_embeds.")
        feats.append(outputs.float())
    return torch.cat(feats, dim=0) if total > 0 else torch.empty((0, 0), dtype=torch.float32)


@torch.no_grad()
def _encode_hf_texts(
    texts: Sequence[str],
    processor: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats = []
    total = (len(texts) + batch_size - 1) // batch_size
    for start, end in iter_slices(len(texts), batch_size):
        chunk = list(texts[start:end])
        inputs = processor(text=chunk, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
        if hasattr(model, "get_text_features"):
            outputs = model.get_text_features(**inputs)
        else:
            raw_outputs = model(**inputs)
            outputs = getattr(raw_outputs, "text_embeds", None)
            if outputs is None:
                raise RuntimeError("Model has no get_text_features and no text_embeds.")
        feats.append(outputs.float())
    return torch.cat(feats, dim=0) if total > 0 else torch.empty((0, 0), dtype=torch.float32)


def _resolve_tifa_question_bank(config: VMITextMetricConfig, stimulus_type: str) -> Path:
    if config.tifa_questions_json:
        return Path(config.tifa_questions_json).expanduser().resolve()
    if config.tifa_question_bank_dir:
        return (Path(config.tifa_question_bank_dir).expanduser().resolve() / f"tifa_bank_{stimulus_type}.json").resolve()
    return (
        Path(__file__).resolve().parents[2] / "vendor" / "vmi_metrics" / f"tifa_bank_{stimulus_type}.json"
    ).resolve()


def _load_question_bank(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Question bank JSON must contain an 'entries' list")
    return sorted(entries, key=lambda entry: int(entry.get("sample_index", 0)))


def _load_question_bank_subset_for_captions(path: Path, captions: Sequence[str]) -> list[dict[str, Any]]:
    entries = _load_question_bank(path)
    caption_lookup = {str(entry.get("caption", "")): entry for entry in entries}
    selected_entries = []
    missing = []
    for caption in captions:
        key = str(caption)
        if key not in caption_lookup:
            missing.append(key)
            continue
        selected_entries.append(caption_lookup[key])
    if missing:
        raise ValueError(f"Question bank is missing captions; first few: {missing[:5]}")
    return selected_entries


def _resolve_tifa_sbert_path(config: VMITextMetricConfig) -> Path:
    if config.tifa_sbert_model_path:
        return Path(config.tifa_sbert_model_path).expanduser().resolve()
    raise FileNotFoundError("TIFA SBERT model path is not configured.")


def _resolve_tifa_mplug_path(config: VMITextMetricConfig) -> Path:
    if config.tifa_mplug_model_path:
        return Path(config.tifa_mplug_model_path).expanduser().resolve()
    raise FileNotFoundError("TIFA mPLUG model path is not configured.")


def _metric_tifa_from_question_bank(
    images: torch.Tensor,
    question_bank_entries: Sequence[dict[str, Any]],
    config: VMITextMetricConfig,
) -> float:
    try:
        from tifascore.mc_sbert import SBERTModel  # type: ignore
        from tifascore.vqa_models import MPLUG  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"TIFA import failed: {exc}") from exc

    mplug_path = _resolve_tifa_mplug_path(config)
    sbert_path = _resolve_tifa_sbert_path(config)
    if not mplug_path.exists():
        raise FileNotFoundError(f"Missing TIFA mPLUG model: {mplug_path}")
    if not sbert_path.exists():
        raise FileNotFoundError(f"Missing TIFA SBERT model: {sbert_path}")

    vqa_model = MPLUG(str(mplug_path))
    sbert_model = SBERTModel(str(sbert_path))
    to_pil = transforms.ToPILImage()

    scores = []
    sample_count = len(images)
    if config.tifa_max_samples is not None:
        sample_count = min(sample_count, int(config.tifa_max_samples))

    for idx in range(sample_count):
        entry = question_bank_entries[idx]
        qas = entry.get("qas")
        if not isinstance(qas, list) or not qas:
            raise ValueError(f"Invalid TIFA QA entry at index {idx}")
        pil_image = to_pil(images[idx].cpu())
        vqa_answers = vqa_model.multiple_choice_vqa(pil_image, qas)
        result = sbert_model.compute_score(qas, vqa_answers)
        scores.append(float(result["tifa_score"]))

    if not scores:
        raise ValueError("No TIFA samples were evaluated.")
    return float(np.mean(scores))


def _evaluate_clip_text_metrics(
    images: torch.Tensor,
    captions: Sequence[str],
    config: VMITextMetricConfig,
) -> dict[str, Any]:
    device = torch.device(config.device)
    processor = AutoProcessor.from_pretrained(config.clip_model_name, local_files_only=config.clip_local_files_only)
    model = CLIPModel.from_pretrained(config.clip_model_name, local_files_only=config.clip_local_files_only).to(device)
    model.eval()
    image_feats = _encode_hf_images(images, processor, model, device, config.batch_size)
    text_feats = _encode_hf_texts(captions, processor, model, device, config.batch_size)
    sim = cosine_matrix(image_feats, text_feats)
    return {
        "CLIP-Text-R@1": recall_at_k(sim, k=1),
        "CLIP-Text-R@5": recall_at_k(sim, k=5),
        "CLIP-Text-2WC": two_way_identification_i2t(sim),
    }


def _evaluate_siglip_text_metric(
    images: torch.Tensor,
    captions: Sequence[str],
    config: VMITextMetricConfig,
) -> dict[str, Any]:
    if config.skip_siglip or not config.siglip_model_name:
        return {}
    device = torch.device(config.device)
    processor = AutoProcessor.from_pretrained(config.siglip_model_name, local_files_only=config.siglip_local_files_only)
    model = SiglipModel.from_pretrained(config.siglip_model_name, local_files_only=config.siglip_local_files_only).to(device)
    model.eval()
    image_feats = _encode_hf_images(images, processor, model, device, config.batch_size)
    text_feats = _encode_hf_texts(captions, processor, model, device, config.batch_size)
    sim = cosine_matrix(image_feats, text_feats)
    return {"SigLIP-Text-2WC": two_way_identification_i2t(sim)}


def _evaluate_altclip_text_metric(
    images: torch.Tensor,
    captions: Sequence[str],
    config: VMITextMetricConfig,
) -> dict[str, Any]:
    if config.skip_altclip or not config.altclip_model_name:
        return {}
    device = torch.device(config.device)
    processor = AutoProcessor.from_pretrained(
        config.altclip_model_name,
        local_files_only=config.altclip_local_files_only,
        trust_remote_code=config.altclip_trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        config.altclip_model_name,
        local_files_only=config.altclip_local_files_only,
        trust_remote_code=config.altclip_trust_remote_code,
    ).to(device)
    model.eval()
    image_feats = _encode_hf_images(images, processor, model, device, config.batch_size)
    text_feats = _encode_hf_texts(captions, processor, model, device, config.batch_size)
    sim = cosine_matrix(image_feats, text_feats)
    return {"AltCLIP-Text-2WC": two_way_identification_i2t(sim)}


def _evaluate_tifa_metric(
    images: torch.Tensor,
    captions: Sequence[str],
    stimulus_type: str,
    config: VMITextMetricConfig,
) -> dict[str, Any]:
    if config.skip_tifa:
        return {}
    question_bank_path = _resolve_tifa_question_bank(config, stimulus_type)
    entries = _load_question_bank_subset_for_captions(question_bank_path, captions)
    return {"TIFA": _metric_tifa_from_question_bank(images, entries, config)}


def evaluate_vmi_text_metrics(
    images: torch.Tensor,
    captions: Sequence[str],
    stimulus_type: str,
    config: VMITextMetricConfig | dict | None = None,
) -> dict[str, Any]:
    resolved = config if isinstance(config, VMITextMetricConfig) else VMITextMetricConfig.from_dict(config)
    if images.ndim != 4:
        raise ValueError(f"Expected image batch shaped [N, C, H, W], got {tuple(images.shape)}")
    if len(images) != len(captions):
        raise ValueError(f"Image/caption length mismatch: {len(images)} vs {len(captions)}")
    if len(images) < 2:
        return {}

    metrics: dict[str, Any] = {}
    metric_functions = (
        _evaluate_clip_text_metrics,
        _evaluate_siglip_text_metric,
        _evaluate_altclip_text_metric,
    )
    for metric_function in metric_functions:
        try:
            metrics.update(metric_function(images, captions, resolved))
        except Exception as exc:
            metric_name_prefix = metric_function.__name__.replace("_evaluate_", "").replace("_metric", "").replace("_metrics", "")
            metrics[f"{metric_name_prefix}_status"] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        metrics.update(_evaluate_tifa_metric(images, captions, stimulus_type, resolved))
    except Exception as exc:
        metrics["TIFA_status"] = f"unavailable: {type(exc).__name__}: {exc}"
    return metrics
