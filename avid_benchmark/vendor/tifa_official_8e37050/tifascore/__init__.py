"""Minimal TIFA runtime pinned from upstream commit 8e37050."""

from .mc_sbert import SBERTModel
from .vqa_models import MPLUG

__all__ = ["MPLUG", "SBERTModel"]

