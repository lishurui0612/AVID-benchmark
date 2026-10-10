from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    @classmethod
    @abstractmethod
    def load_from_config(cls, config: dict[str, Any]) -> "ModelAdapter":
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, checkpoint_path: str | None = None) -> dict[str, Any]:
        raise NotImplementedError
