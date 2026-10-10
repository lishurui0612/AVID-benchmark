from __future__ import annotations

import random
import hashlib
import functools
import os
import tarfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import webdataset as wds


def passthrough_nodesplitter(urls):
    return urls


def distributed_nodesplitter(src):
    return wds.split_by_node(src)


def split_samples(src, rank: int, world_size: int):
    """Split within a participant tar; URL splitting cannot split one shard."""
    for sample in src:
        digest=hashlib.blake2b(str(sample['image_name']).encode(),digest_size=8).digest()
        if int.from_bytes(digest,'big') % world_size == rank:
            yield sample


def _normalize_urls(url: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(url, (str, Path)):
        return [str(url)]
    return [str(item) for item in url]


def build_webdataset(
    url: str | Path | Sequence[str | Path],
    resampled: bool,
    seed: int,
    distributed: bool = False,
    include_sample_key: bool = False,
    allowed_image_names: set[str] | None = None,
):
    kwargs = {
        "resampled": resampled,
        "nodesplitter": passthrough_nodesplitter,
    }
    try:
        dataset = wds.WebDataset(
            _normalize_urls(url),
            empty_check=False,
            **kwargs,
        )
    except TypeError:
        dataset = wds.WebDataset(
            _normalize_urls(url),
            **kwargs,
        )
    dataset = (
        dataset.decode("torch")
        .rename(image_name="img_name.npy", responses="responses.npy")
        .map_dict(image_name=lambda x: x.item() if isinstance(x, np.ndarray) else x)
    )
    if allowed_image_names is not None:
        allowed = frozenset(str(item) for item in allowed_image_names)
        dataset = dataset.select(lambda sample: str(sample["image_name"]) in allowed)
    if distributed:
        if torch.distributed.is_initialized():
            rank=torch.distributed.get_rank()
            world_size=torch.distributed.get_world_size()
        else:
            rank=int(os.environ.get('RANK',0))
            world_size=int(os.environ.get('WORLD_SIZE',1))
        dataset=dataset.compose(functools.partial(split_samples,rank=rank,world_size=world_size))
    if resampled:
        dataset = dataset.shuffle(750, initial=1500, rng=random.Random(seed))
    if include_sample_key:
        dataset = dataset.to_tuple("__key__", "image_name", "responses")
    else:
        dataset = dataset.to_tuple("image_name", "responses")
    return dataset


def build_loader(
    url: str | Path | Sequence[str | Path],
    batch_size: int,
    num_workers: int,
    resampled: bool,
    seed: int,
    drop_last: bool,
    distributed: bool = False,
    include_sample_key: bool = False,
    allowed_image_names: set[str] | None = None,
):
    effective_num_workers = num_workers
    if num_workers > 1:
        effective_num_workers = 1
    dataset = build_webdataset(
        url=url,
        resampled=resampled,
        seed=seed,
        distributed=distributed,
        include_sample_key=include_sample_key,
        allowed_image_names=allowed_image_names,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        num_workers=effective_num_workers,
    )


def count_samples_in_tar(url: str | Path) -> int:
    count = 0
    with tarfile.open(str(url), "r") as handle:
        for member in handle:
            if member.name.endswith(".img_name.npy"):
                count += 1
    if count <= 0:
        raise RuntimeError(f"No samples found in {url}")
    return count


def reduce_responses(responses: torch.Tensor | np.ndarray, mode: str) -> torch.Tensor:
    if isinstance(responses, np.ndarray):
        responses = torch.from_numpy(responses)
    if responses.ndim == 3:
        if mode == "mean":
            responses = responses.mean(dim=1)
        elif mode == "random":
            row_index = torch.arange(responses.shape[0], device=responses.device)
            repeat_index = torch.randint(0, responses.shape[1], (responses.shape[0],), device=responses.device)
            responses = responses[row_index, repeat_index]
        else:
            responses = responses[:, 0]
    elif responses.ndim != 2:
        raise ValueError(f"Unexpected responses shape: {tuple(responses.shape)}")
    return responses.float()


def unpack_loader_batch(
    batch: tuple,
) -> tuple[list[str], list[str], torch.Tensor]:
    """Normalize legacy and key-preserving WebDataset tuples."""

    if len(batch) == 3:
        sample_keys, image_names, responses = batch
        return list(sample_keys), list(image_names), responses
    if len(batch) == 2:
        image_names, responses = batch
        synthesized = [str(image_name) for image_name in image_names]
        return synthesized, list(image_names), responses
    raise ValueError(f"Unexpected loader batch with {len(batch)} fields.")


def expand_response_repeats(
    sample_keys: Sequence[str],
    image_names: Sequence[str],
    responses: torch.Tensor | np.ndarray,
) -> tuple[list[str], list[str], torch.Tensor, list[int]]:
    """Expand ``[B,R,V]`` responses into repeat-addressable ``[B*R,V]`` rows."""

    if isinstance(responses, np.ndarray):
        responses = torch.from_numpy(responses)
    if responses.ndim == 2:
        return (
            [str(item) for item in sample_keys],
            [str(item) for item in image_names],
            responses.float(),
            [0] * len(image_names),
        )
    if responses.ndim != 3:
        raise ValueError(
            f"Expected response tensor [B,V] or [B,R,V], got {tuple(responses.shape)}"
        )
    if responses.shape[0] != len(image_names) or len(sample_keys) != len(image_names):
        raise ValueError("Sample keys, image names, and response batch differ in length.")
    expanded_keys: list[str] = []
    expanded_names: list[str] = []
    repeat_indices: list[int] = []
    rows: list[torch.Tensor] = []
    for sample_index, (sample_key, image_name) in enumerate(
        zip(sample_keys, image_names)
    ):
        for repeat_index in range(responses.shape[1]):
            expanded_keys.append(f"{sample_key}:repeat-{repeat_index}")
            expanded_names.append(str(image_name))
            repeat_indices.append(repeat_index)
            rows.append(responses[sample_index, repeat_index])
    return (
        expanded_keys,
        expanded_names,
        torch.stack(rows, dim=0).float(),
        repeat_indices,
    )


def aggregate_image_responses(
    loader: Iterable[tuple],
    response_reduce: str,
    max_samples: int = 0,
) -> tuple[list[str], torch.Tensor]:
    response_lists: dict[str, list[torch.Tensor]] = {}
    image_order: list[str] = []
    for batch in loader:
        _, image_names, responses = unpack_loader_batch(batch)
        reduced = reduce_responses(responses, response_reduce).cpu()
        for idx, image_name in enumerate(image_names):
            if image_name not in response_lists:
                response_lists[image_name] = []
                image_order.append(image_name)
            response_lists[image_name].append(reduced[idx])
    if max_samples > 0:
        image_order = image_order[:max_samples]
    if not image_order:
        return [], torch.empty(0, dtype=torch.float32)
    averaged_responses = [
        torch.stack(response_lists[image_name], dim=0).mean(dim=0) for image_name in image_order
    ]
    return image_order, torch.stack(averaged_responses, dim=0)
