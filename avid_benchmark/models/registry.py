from __future__ import annotations

import importlib
from typing import Any, Callable

from .base import ModelAdapter


MODEL_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register_model_adapter(name: str) -> Callable[[type[ModelAdapter]], type[ModelAdapter]]:
    def decorator(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_model_adapter(model_name: str, config: dict[str, Any]) -> ModelAdapter:
    if model_name not in MODEL_REGISTRY:
        try:
            importlib.import_module(f"{__package__}.{model_name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}.{model_name}":
                raise
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model adapter {model_name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name].load_from_config(config)
