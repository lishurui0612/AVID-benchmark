from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from ..data.dataset import build_loader, count_samples_in_tar
from ..runtime import save_jsonl
from ..types import PathConfig, PredictionRecord


def subject_token(subject_id: int) -> str:
    return f"{int(subject_id):02d}"


def resolve_stimulus_image_root(paths: PathConfig, stimulus_type: str) -> Path:
    normalized = str(stimulus_type).strip().lower()
    if normalized == "complex":
        return paths.coco_image_root.resolve()
    if normalized == "simple":
        return paths.simple_image_root.resolve()
    raise ValueError(f"Unsupported stimulus_type {stimulus_type!r}")


def resolve_avid_shard_path(
    dataset_root: Path,
    *,
    subject_id: int,
    stimulus_type: str,
    split: str,
) -> Path:
    token = subject_token(subject_id)
    filename = f"sub-{token}_{stimulus_type}_{split}.tar"
    candidates = [
        Path(dataset_root) / filename,
        Path(dataset_root) / f"sub-{token}" / filename,
        Path(dataset_root) / f"sub-{int(subject_id)}" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = sorted(Path(dataset_root).rglob(filename))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find {filename} under {Path(dataset_root).resolve()}"
    )


def prepare_vendor_dataset_root(
    dataset_root: Path,
    cache_root: Path,
    *,
    subject_id: int,
    stimulus_type: str,
) -> Path:
    compat_root = cache_root / "vendor_dataset"
    compat_subject_dir = compat_root / f"sub-{subject_token(subject_id)}"
    compat_subject_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        source = resolve_avid_shard_path(
            dataset_root,
            subject_id=subject_id,
            stimulus_type=stimulus_type,
            split=split,
        )
        target = compat_subject_dir / source.name
        if target.is_symlink() or target.exists():
            try:
                if os.path.samefile(target, source):
                    continue
            except OSError:
                pass
            try:
                source_stat = source.stat()
                target_stat = target.stat()
                if (
                    not target.is_symlink()
                    and target_stat.st_size == source_stat.st_size
                    and int(target_stat.st_mtime) == int(source_stat.st_mtime)
                ):
                    continue
            except OSError:
                pass
            target.unlink()
        _materialize_vendor_shard(source, target)
    return compat_root.resolve()


def _materialize_vendor_shard(source: Path, target: Path) -> None:
    try:
        target.symlink_to(source)
        return
    except OSError:
        pass

    try:
        os.link(source, target)
        return
    except OSError:
        pass

    shutil.copy2(source, target)


def run_python_script(
    script_path: Path,
    *,
    args: Sequence[str],
    cwd: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update({key: str(value) for key, value in extra_env.items()})
    subprocess.run(
        [sys.executable, str(script_path), *[str(item) for item in args]],
        check=True,
        cwd=str(cwd or script_path.parent),
        env=env,
    )


def ordered_split_image_names(
    dataset_root: Path,
    *,
    subject_id: int,
    stimulus_type: str,
    split: str,
) -> list[str]:
    tar_path = resolve_avid_shard_path(
        dataset_root,
        subject_id=subject_id,
        stimulus_type=stimulus_type,
        split=split,
    )
    loader = build_loader(
        url=str(tar_path),
        batch_size=min(256, count_samples_in_tar(tar_path)),
        num_workers=0,
        resampled=False,
        seed=0,
        drop_last=False,
        distributed=False,
    )
    image_names: list[str] = []
    for batch_names, _ in loader:
        for image_name in batch_names:
            if isinstance(image_name, bytes):
                normalized = image_name.decode("utf-8")
            else:
                normalized = str(image_name)
            image_names.append(os.path.basename(normalized.strip()))
    return image_names


def recon_tensor_to_uint8(raw_recon_images: torch.Tensor) -> torch.Tensor:
    recon_images = torch.as_tensor(raw_recon_images)
    if recon_images.ndim == 5:
        recon_images = recon_images.view(-1, recon_images.shape[-3], recon_images.shape[-2], recon_images.shape[-1])
    if recon_images.ndim != 4:
        raise ValueError(f"Expected recon image tensor shaped [N, C, H, W], got {tuple(recon_images.shape)}")
    recon_images = recon_images.detach().cpu()
    if recon_images.dtype == torch.uint8:
        return recon_images.clone()
    recon_images = recon_images.float()
    if not torch.isfinite(recon_images).all():
        raise ValueError("Reconstruction tensors contain nonfinite values.")
    if recon_images.numel() > 0 and (recon_images.min().item() < -1e-6 or recon_images.max().item() > 1.0 + 1e-6):
        raise ValueError("Floating reconstructions must be explicitly scaled to [0,1]; range guessing is disabled.")
    return recon_images.clamp(0.0, 1.0).mul(255).round().to(torch.uint8).clone()


def save_recon_tensor_payload(
    recon_images_path: Path,
    *,
    raw_recon_images: torch.Tensor,
    image_names: list[str],
    subject_ids: list[int],
    split: str,
) -> torch.Tensor:
    recon_uint8 = recon_tensor_to_uint8(raw_recon_images)
    torch.save(
        {
            "image_names": image_names,
            "subject_ids": subject_ids,
            "split": split,
            "recon_images": recon_uint8,
            "image_dtype": "uint8",
            "image_value_range": [0, 255],
        },
        recon_images_path,
    )
    return recon_uint8


def _save_pair_panel(gt_image_path: str | Path, recon_uint8: torch.Tensor, output_path: Path) -> None:
    with Image.open(gt_image_path) as gt_image:
        gt_pil = gt_image.convert("RGB")
    recon_pil = transforms.ToPILImage()(recon_uint8)
    recon_pil = recon_pil.resize(gt_pil.size, Image.Resampling.BICUBIC)
    label_height = 36
    width, height = gt_pil.size
    canvas = Image.new("RGB", (width * 2, height + label_height), color="white")
    canvas.paste(gt_pil, (0, label_height))
    canvas.paste(recon_pil, (width, label_height))
    draw = ImageDraw.Draw(canvas)
    for idx, label in enumerate(("GT", "Recon")):
        x0 = idx * width
        x1 = (idx + 1) * width - 1
        draw.rectangle((x0, 0, x1, label_height - 1), fill=(245, 245, 245), outline=(180, 180, 180))
        draw.text((x0 + 12, 9), label, fill=(20, 20, 20))
        if idx == 0:
            draw.line((x1, 0, x1, height + label_height), fill=(180, 180, 180), width=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def export_prediction_manifest(
    prediction_root: Path,
    *,
    model_name: str,
    subject_id: int,
    split: str,
    image_names: list[str],
    recon_uint8: torch.Tensor,
    metadata_index,
    route: str,
    generation_protocol: dict | None = None,
) -> dict[str, object]:
    if len(image_names) != recon_uint8.shape[0]:
        raise ValueError(
            f"image_names length {len(image_names)} does not match recon_images batch size {recon_uint8.shape[0]}"
        )

    manifest_path = prediction_root / "manifest.jsonl"
    subject_root = prediction_root / f"subj{subject_token(subject_id)}"
    final_image_paths: list[str] = []
    panel_paths: list[str] = []
    records: list[PredictionRecord] = []

    for idx, image_name in enumerate(image_names):
        metadata = metadata_index.get(image_name)
        image_stem = Path(image_name).stem
        output_stem = f"subj{subject_token(subject_id)}_{split}_{image_stem}"
        final_path = subject_root / "final" / f"{output_stem}.png"
        panel_path = subject_root / "panels" / f"{output_stem}.png"

        final_path.parent.mkdir(parents=True, exist_ok=True)
        transforms.ToPILImage()(recon_uint8[idx]).save(final_path)
        _save_pair_panel(metadata["gt_image_path"], recon_uint8[idx], panel_path)

        final_image_paths.append(str(final_path.resolve()))
        panel_paths.append(str(panel_path.resolve()))
        records.append(
            PredictionRecord(
                model_name=model_name,
                subject_id=subject_id,
                stimulus_type=metadata["stimulus_type"],
                split=split,
                sample_id=f"subj{subject_token(subject_id)}_{split}_{image_stem}",
                image_name=image_name,
                gt_image_path=Path(metadata["gt_image_path"]).resolve(),
                generated_image_path=final_path.resolve(),
                generated_text=None,
                route=route,
                artifact_paths={
                    "final_image": str(final_path.resolve()),
                    "candidate_panel_image": str(panel_path.resolve()),
                },
                notes={
                    "generation_protocol": generation_protocol or {},
                    "image_outputs": {
                        "final": str(final_path.resolve()),
                        "candidate_panel": str(panel_path.resolve()),
                    }
                },
            )
        )

    save_jsonl(manifest_path, records)
    return {
        "manifest_path": str(manifest_path.resolve()),
        "final_image_paths": final_image_paths,
        "panel_paths": panel_paths,
        "num_predictions": len(records),
    }
