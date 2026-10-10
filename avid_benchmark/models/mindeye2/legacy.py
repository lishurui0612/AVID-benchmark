from __future__ import annotations

import functools
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def _force_offline_hf_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _ensure_path(path: Path) -> None:
    _force_offline_hf_mode()
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _avid_code_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _avid_benchmark_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return _avid_code_root()


@functools.lru_cache(maxsize=1)
def legacy_src_dir() -> Path:
    return (_avid_benchmark_root() / "vendor" / "mindeye2" / "src").resolve()


@functools.lru_cache(maxsize=1)
def legacy_generative_models_dir() -> Path:
    return (legacy_src_dir() / "generative_models").resolve()


@functools.lru_cache(maxsize=1)
def legacy_models_module() -> ModuleType:
    _ensure_path(legacy_src_dir())
    return importlib.import_module("models")


@functools.lru_cache(maxsize=1)
def legacy_utils_module() -> ModuleType:
    _ensure_path(legacy_src_dir())
    return importlib.import_module("utils")


@functools.lru_cache(maxsize=1)
def legacy_generators_module() -> tuple[type, type, type]:
    module = legacy_models_module()
    return module.BrainNetwork, module.BrainDiffusionPrior, module.PriorNetwork


@functools.lru_cache(maxsize=1)
def openclip_image_embedder_cls():
    _ensure_path(legacy_src_dir())
    _ensure_path(legacy_generative_models_dir())
    module = importlib.import_module("generative_models.sgm.modules.encoders.modules")
    return module.FrozenOpenCLIPImageEmbedder


@functools.lru_cache(maxsize=1)
def git_caption_model_cls():
    _ensure_path(legacy_src_dir())
    module = importlib.import_module("modeling_git")
    return module.GitForCausalLMClipEmb


@functools.lru_cache(maxsize=1)
def omegaconf_module():
    return importlib.import_module("omegaconf")


@functools.lru_cache(maxsize=1)
def diffusion_engine_cls():
    _ensure_path(legacy_src_dir())
    _ensure_path(legacy_generative_models_dir())
    module = importlib.import_module("generative_models.sgm.models.diffusion")
    return module.DiffusionEngine


def project_root() -> Path:
    return _project_root()
