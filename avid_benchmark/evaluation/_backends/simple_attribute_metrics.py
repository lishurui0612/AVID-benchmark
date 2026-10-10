"""Split-independent Simple-Attribute-VQA evaluation.

This module deliberately does not modify the existing TIFA metric.  It builds a
frozen, deterministic question bank for the 400 ``simple_*.jpg`` stimuli and
reuses TIFA's mPLUG + SBERT answer mapping at question granularity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np


BANK_VERSION = "v1"
METRIC_NAME = "Simple-Attribute-VQA"
EXPECTED_FAMILY_COUNTS = {
    "letter": 66,
    "single_digit": 30,
    "digit_sequence": 30,
    "shape": 225,
    "object": 49,
}
ATTRIBUTE_ORDER = ("color", "position", "visual_identity")
LETTER_CHOICES = list("CFGHJKQSWXY")
DIGIT_CHOICES = [str(value) for value in range(10)]
DIGIT_SEQUENCE_CHOICES = [
    "194",
    "286",
    "372",
    "431",
    "566",
    "653",
    "740",
    "829",
    "918",
    "103",
]
COLOR_CHOICES = ["red", "yellow", "blue", "natural or multicolored"]
SHAPE_CHOICES = [
    "triangle",
    "ring",
    "circle",
    "cross",
    "diamond",
    "five-pointed star",
    "heart",
    "square",
    "three-by-three dot array",
    "upward-facing semicircle",
    "downward-facing semicircle",
    "left-facing semicircle",
    "right-facing semicircle",
    "vertical rectangle",
    "horizontal rectangle",
]
POSITION_CHOICES = ["top", "bottom", "left", "right", "center"]


class AnswerBackend(Protocol):
    """Minimal per-question interface implemented by TIFA and test doubles."""

    def answer(self, image: Any, question: str, choices: Sequence[str]) -> tuple[str, str]:
        """Return ``(free_form_answer, mapped_answer)``."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _simple_id(image_name: str) -> int:
    match = re.fullmatch(r"simple_(\d{3})\.jpg", image_name)
    if not match:
        raise ValueError(f"Invalid simple image name: {image_name!r}")
    return int(match.group(1))


def _load_simple_rows(used_stimuli_tsv: str | Path) -> list[dict[str, str]]:
    path = Path(used_stimuli_tsv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle, delimiter="\t")
            if re.fullmatch(r"simple_\d{3}\.jpg", str(row.get("image", "")))
        ]
    rows.sort(key=lambda row: _simple_id(row["image"]))
    names = [row["image"] for row in rows]
    expected = [f"simple_{index:03d}.jpg" for index in range(400)]
    if names != expected:
        missing = sorted(set(expected) - set(names))[:10]
        extra = sorted(set(names) - set(expected))[:10]
        raise ValueError(
            "used_stimuli.tsv must contain exactly simple_000.jpg through "
            f"simple_399.jpg; missing={missing}, extra={extra}, count={len(names)}"
        )
    return rows


def _extract_object_label(caption: str) -> str:
    prefix = "In the center, there is "
    if not caption.startswith(prefix):
        raise ValueError(f"Unexpected object caption: {caption!r}")
    label = caption[len(prefix) :].strip().rstrip(".")
    for article in ("a pair of ", "an ", "a "):
        if label.startswith(article):
            label = label[len(article) :]
            break
    return label


def _qa(
    image_name: str,
    attribute: str,
    subtype: str,
    question: str,
    choices: Sequence[str],
    answer: str,
    *,
    counterfactual_eligible: bool = True,
) -> dict[str, Any]:
    if answer not in choices:
        raise ValueError(f"{image_name}: answer {answer!r} is not in choices")
    return {
        "question_id": f"{Path(image_name).stem}.{attribute}.{BANK_VERSION}",
        "attribute": attribute,
        "subtype": subtype,
        "question": question,
        "choices": list(choices),
        "answer": answer,
        "counterfactual_eligible": counterfactual_eligible,
    }


def build_question_bank(used_stimuli_tsv: str | Path) -> dict[str, Any]:
    """Build the deterministic 400-image v1 question bank."""

    rows = _load_simple_rows(used_stimuli_tsv)
    object_choices = [
        _extract_object_label(row["translated_caption"]) for row in rows[351:400]
    ]
    if len(set(object_choices)) != 49:
        duplicates = [
            value for value, count in Counter(object_choices).items() if count > 1
        ]
        raise ValueError(f"Object labels are not unique: {duplicates}")

    visual_identity_choices = (
        LETTER_CHOICES
        + DIGIT_CHOICES
        + DIGIT_SEQUENCE_CHOICES
        + SHAPE_CHOICES
        + object_choices
    )
    if len(visual_identity_choices) != 95 or len(set(visual_identity_choices)) != 95:
        raise ValueError("The global visual-identity vocabulary must contain 95 unique labels")

    entries: list[dict[str, Any]] = []
    for row in rows:
        image_name = row["image"]
        index = _simple_id(image_name)
        caption = row["translated_caption"]
        labels: dict[str, str] = {}
        qas: list[dict[str, Any]] = []

        if index < 66:
            family = "letter"
            letter = LETTER_CHOICES[index // 6]
            within = index % 6
            color = COLOR_CHOICES[within // 2]
            labels.update(
                color=color,
                position="center",
                visual_identity=letter,
            )
            qas.extend(
                [
                    _qa(
                        image_name,
                        "color",
                        "controlled_solid",
                        "Which color category best describes the displayed content?",
                        COLOR_CHOICES,
                        color,
                    ),
                    _qa(
                        image_name,
                        "visual_identity",
                        "letter",
                        "What is the identity of the displayed content?",
                        visual_identity_choices,
                        letter,
                    ),
                    _qa(
                        image_name,
                        "position",
                        "fixed_center",
                        "Where is the main displayed item located?",
                        POSITION_CHOICES,
                        "center",
                        counterfactual_eligible=False,
                    ),
                ]
            )
        elif index < 96:
            family = "single_digit"
            digit = DIGIT_CHOICES[(index - 66) // 3]
            color = COLOR_CHOICES[(index - 66) % 3]
            labels.update(
                color=color,
                position="center",
                visual_identity=digit,
            )
            qas.extend(
                [
                    _qa(
                        image_name,
                        "color",
                        "controlled_solid",
                        "Which color category best describes the displayed content?",
                        COLOR_CHOICES,
                        color,
                    ),
                    _qa(
                        image_name,
                        "visual_identity",
                        "single_digit",
                        "What is the identity of the displayed content?",
                        visual_identity_choices,
                        digit,
                    ),
                    _qa(
                        image_name,
                        "position",
                        "fixed_center",
                        "Where is the main displayed item located?",
                        POSITION_CHOICES,
                        "center",
                        counterfactual_eligible=False,
                    ),
                ]
            )
        elif index < 126:
            family = "digit_sequence"
            sequence = DIGIT_SEQUENCE_CHOICES[(index - 96) // 3]
            color = COLOR_CHOICES[(index - 96) % 3]
            labels.update(
                color=color,
                position="center",
                visual_identity=sequence,
            )
            qas.extend(
                [
                    _qa(
                        image_name,
                        "color",
                        "controlled_solid",
                        "Which color category best describes the displayed content?",
                        COLOR_CHOICES,
                        color,
                    ),
                    _qa(
                        image_name,
                        "visual_identity",
                        "digit_sequence",
                        "What is the identity of the displayed content?",
                        visual_identity_choices,
                        sequence,
                    ),
                    _qa(
                        image_name,
                        "position",
                        "fixed_center",
                        "Where is the main displayed item located?",
                        POSITION_CHOICES,
                        "center",
                        counterfactual_eligible=False,
                    ),
                ]
            )
        elif index < 351:
            family = "shape"
            within = index - 126
            position = POSITION_CHOICES[within // 45]
            color = COLOR_CHOICES[(within % 45) // 15]
            shape = SHAPE_CHOICES[within % 15]
            labels.update(color=color, position=position, visual_identity=shape)
            qas.extend(
                [
                    _qa(
                        image_name,
                        "color",
                        "controlled_solid",
                        "Which color category best describes the displayed content?",
                        COLOR_CHOICES,
                        color,
                    ),
                    _qa(
                        image_name,
                        "visual_identity",
                        "shape",
                        "What is the identity of the displayed content?",
                        visual_identity_choices,
                        shape,
                    ),
                    _qa(
                        image_name,
                        "position",
                        "controlled_shape",
                        "Where is the main displayed item located?",
                        POSITION_CHOICES,
                        position,
                    ),
                ]
            )
        else:
            family = "object"
            object_label = object_choices[index - 351]
            labels.update(
                color="natural or multicolored",
                position="center",
                visual_identity=object_label,
            )
            qas.extend(
                [
                    _qa(
                        image_name,
                        "color",
                        "natural_or_multicolored",
                        "Which color category best describes the displayed content?",
                        COLOR_CHOICES,
                        "natural or multicolored",
                        counterfactual_eligible=False,
                    ),
                    _qa(
                        image_name,
                        "visual_identity",
                        "object",
                        "What is the identity of the displayed content?",
                        visual_identity_choices,
                        object_label,
                    ),
                    _qa(
                        image_name,
                        "position",
                        "fixed_center",
                        "Where is the main displayed item located?",
                        POSITION_CHOICES,
                        "center",
                        counterfactual_eligible=False,
                    ),
                ]
            )

        entries.append(
            {
                "image_name": image_name,
                "simple_id": index,
                "caption": caption,
                "family": family,
                "labels": labels,
                "qas": qas,
            }
        )

    bank = {
        "metric_name": METRIC_NAME,
        "version": BANK_VERSION,
        "metadata": {
            "source_tsv_sha256": sha256_file(used_stimuli_tsv),
            "image_count": len(entries),
            "family_counts": dict(Counter(entry["family"] for entry in entries)),
            "attribute_question_counts": dict(
                Counter(qa["attribute"] for entry in entries for qa in entry["qas"])
            ),
        },
        "entries": entries,
    }
    validate_question_bank(bank)
    bank["metadata"]["bank_content_sha256"] = canonical_json_hash(entries)
    return bank


def validate_question_bank(bank: Mapping[str, Any]) -> dict[str, Any]:
    entries = bank.get("entries")
    if not isinstance(entries, list) or len(entries) != 400:
        raise ValueError(f"Question bank must have 400 entries, got {len(entries or [])}")
    names = [str(entry["image_name"]) for entry in entries]
    expected = [f"simple_{index:03d}.jpg" for index in range(400)]
    if names != expected:
        raise ValueError("Question bank image order is not simple_000..simple_399")

    family_counts = Counter(str(entry["family"]) for entry in entries)
    if dict(family_counts) != EXPECTED_FAMILY_COUNTS:
        raise ValueError(
            f"Unexpected family counts: {dict(family_counts)} != {EXPECTED_FAMILY_COUNTS}"
        )

    question_ids: list[str] = []
    attribute_counts: Counter[str] = Counter()
    for entry in entries:
        if "train" in entry or "val" in entry or "split" in entry:
            raise ValueError("Question bank must be split-independent")
        qas = entry.get("qas")
        if not isinstance(qas, list) or not qas:
            raise ValueError(f"{entry['image_name']} has no questions")
        seen_attributes: set[str] = set()
        for qa in qas:
            attribute = str(qa["attribute"])
            if attribute in seen_attributes:
                raise ValueError(
                    f"{entry['image_name']} has duplicate attribute {attribute}"
                )
            seen_attributes.add(attribute)
            if qa["answer"] not in qa["choices"]:
                raise ValueError(f"{qa['question_id']}: answer is not in choices")
            acceptable_answers = list(
                qa.get("acceptable_answers", [qa["answer"]])
            )
            if (
                not acceptable_answers
                or qa["answer"] not in acceptable_answers
                or len(acceptable_answers) != len(set(acceptable_answers))
                or any(answer not in qa["choices"] for answer in acceptable_answers)
            ):
                raise ValueError(
                    f"{qa['question_id']}: acceptable_answers must be a unique, "
                    "non-empty subset of choices containing the canonical answer"
                )
            normalized = {str(choice).strip().lower() for choice in qa["choices"]}
            if {"yes", "no"} & normalized:
                raise ValueError(f"{qa['question_id']}: Yes/No choices are forbidden")
            question_ids.append(str(qa["question_id"]))
            attribute_counts[attribute] += 1
    duplicates = [key for key, count in Counter(question_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {duplicates[:5]}")

    expected_attribute_counts = {"color": 400, "position": 400, "visual_identity": 400}
    if dict(attribute_counts) != expected_attribute_counts:
        raise ValueError(
            f"Unexpected attribute counts: {dict(attribute_counts)} "
            f"!= {expected_attribute_counts}"
        )
    return {
        "image_count": len(entries),
        "question_count": len(question_ids),
        "family_counts": dict(family_counts),
        "attribute_question_counts": dict(attribute_counts),
    }


def load_question_bank(path: str | Path) -> dict[str, Any]:
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_question_bank(bank)
    return bank


def bank_lookup(bank: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry["image_name"]): dict(entry) for entry in bank["entries"]}


def _tar_image_names(path: str | Path) -> list[str]:
    with tarfile.open(path, "r:*") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    result: list[str] = []
    seen: set[str] = set()
    for member in members:
        basename = Path(member).name
        match = re.fullmatch(r"(simple_\d{3})\.img_name\.npy", basename)
        if match:
            image_name = f"{match.group(1)}.jpg"
            if image_name not in seen:
                result.append(image_name)
                seen.add(image_name)
    if result:
        return result

    # Some exports store the actual JPEG names rather than img_name arrays.
    for member in members:
        basename = Path(member).name
        if re.fullmatch(r"simple_\d{3}\.jpg", basename) and basename not in seen:
            result.append(basename)
            seen.add(basename)
    return result


def build_manifest_rows(
    bank: Mapping[str, Any],
    *,
    image_names: Sequence[str] | None = None,
    tar_path: str | Path | None = None,
    subset_name: str = "all",
) -> list[dict[str, Any]]:
    if image_names is not None and tar_path is not None:
        raise ValueError("Specify either image_names or tar_path, not both")
    if tar_path is not None:
        image_names = _tar_image_names(tar_path)
        if not image_names:
            raise ValueError(f"No simple image names found in tar: {tar_path}")
    if image_names is None:
        image_names = [str(entry["image_name"]) for entry in bank["entries"]]

    lookup = bank_lookup(bank)
    names = [str(name) for name in image_names]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    unknown = [name for name in names if name not in lookup]
    if duplicates or unknown:
        raise ValueError(f"Invalid manifest names: duplicates={duplicates[:5]}, unknown={unknown[:5]}")
    return [
        {"order_index": index, "image_name": name, "subset_name": subset_name}
        for index, name in enumerate(names)
    ]


def write_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["order_index", "image_name", "subset_name"]
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["order_index"]))
    names = [row["image_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Manifest contains duplicate image names")
    return rows


def choose_counterfactuals(bank: Mapping[str, Any]) -> dict[str, str]:
    """Select one deterministic negative control image for every question.

    Candidates have the same generic question/subtype and a different target
    answer.  Among them, the selector maximizes agreement on all non-target
    labels and then uses image name as a deterministic tie breaker.
    """

    records: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (entry, qa) for entry in bank["entries"] for qa in entry["qas"]
    ]
    result: dict[str, str] = {}
    for entry, qa in records:
        if not bool(qa.get("counterfactual_eligible", True)):
            continue
        candidates: list[tuple[int, str]] = []
        target_attribute = str(qa["attribute"])
        labels = entry["labels"]
        for other, other_qa in records:
            if other["image_name"] == entry["image_name"]:
                continue
            if other_qa["attribute"] != target_attribute:
                continue
            if other_qa["subtype"] != qa["subtype"]:
                continue
            if other_qa["question"] != qa["question"]:
                continue
            if other_qa["answer"] == qa["answer"]:
                continue
            shared = sum(
                other["labels"].get(key) == value
                for key, value in labels.items()
                if key != target_attribute and key in other["labels"]
            )
            candidates.append((-shared, str(other["image_name"])))
        if not candidates:
            raise ValueError(f"No counterfactual candidate for {qa['question_id']}")
        candidates.sort()
        result[str(qa["question_id"])] = candidates[0][1]
    return result


class TifaQuestionBackend:
    """Compatibility wrapper around the locally used TIFA fork."""

    def __init__(self, mplug_model_path: str | Path, sbert_model_path: str | Path):
        try:
            from tifascore.mc_sbert import SBERTModel  # type: ignore
            from tifascore.vqa_models import MPLUG  # type: ignore
        except Exception as exc:  # pragma: no cover - optional GPU dependency
            raise RuntimeError(f"TIFA import failed: {exc}") from exc

        self.mplug_model_path = str(Path(mplug_model_path).expanduser().resolve())
        self.sbert_model_path = str(Path(sbert_model_path).expanduser().resolve())
        for label, path in (
            ("mPLUG", self.mplug_model_path),
            ("SBERT", self.sbert_model_path),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing {label} model: {path}")
        self.vqa_model = MPLUG(self.mplug_model_path)
        self.sbert_model = SBERTModel(self.sbert_model_path)

    def answer(self, image: Any, question: str, choices: Sequence[str]) -> tuple[str, str]:
        # The established local TIFA fork exposes vqa(path, question).  Saving a
        # temporary PNG avoids relying on undocumented PIL support.
        with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
            image.save(temporary.name)
            free_form = str(self.vqa_model.vqa(temporary.name, question)).strip()
        if free_form in choices:
            mapped = free_form
        else:
            mapped_raw = self.sbert_model.multiple_choice(free_form, list(choices))
            if isinstance(mapped_raw, (tuple, list)):
                mapped_raw = mapped_raw[0]
            mapped = str(mapped_raw).strip()
        return free_form, mapped


class AtomicResultCache:
    def __init__(self, root: str | Path, run_hash: str, resume: bool):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        metadata_path = self.root / "run_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("run_hash") != run_hash:
                raise ValueError(
                    "Existing cache belongs to a different run. Use a new output "
                    "directory or remove the cache explicitly."
                )
            if not resume and any(self.root.glob("*.json")):
                raise FileExistsError(
                    "Cache already contains results; pass --resume or use a new output directory"
                )
        else:
            atomic_write_json(metadata_path, {"run_hash": run_hash})

    @staticmethod
    def _key(pair_type: str, image_name: str, question_id: str) -> str:
        return hashlib.sha256(
            f"{pair_type}\0{image_name}\0{question_id}".encode("utf-8")
        ).hexdigest()

    def path_for(self, pair_type: str, image_name: str, question_id: str) -> Path:
        return self.root / f"{self._key(pair_type, image_name, question_id)}.json"

    def get(
        self, pair_type: str, image_name: str, question_id: str
    ) -> dict[str, Any] | None:
        path = self.path_for(pair_type, image_name, question_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, row: Mapping[str, Any]) -> None:
        path = self.path_for(
            str(row["pair_type"]), str(row["image_name"]), str(row["question_id"])
        )
        atomic_write_json(path, dict(row))

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for path in self.root.glob("*.json"):
            if path.name == "run_metadata.json":
                continue
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return sorted(
            rows,
            key=lambda row: (
                int(row["image_order"]),
                0 if row["pair_type"] == "factual" else 1,
                ATTRIBUTE_ORDER.index(row["attribute"]),
            ),
        )


def _pil_from_tensor(tensor: Any) -> Any:
    from PIL import Image

    if hasattr(tensor, "detach"):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = array[..., 0]
    if np.issubdtype(array.dtype, np.floating):
        finite_min = float(np.nanmin(array))
        finite_max = float(np.nanmax(array))
        if finite_min < 0:
            array = (array + 1.0) * 127.5
        elif finite_max <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def load_reconstruction_images(
    recon_path: str | Path, manifest_names: Sequence[str]
) -> dict[str, Any]:
    import torch

    payload = torch.load(Path(recon_path), map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        tensor_key = next(
            (
                key
                for key in ("recon_images", "images", "reconstructions")
                if key in payload
            ),
            None,
        )
        if tensor_key is None:
            raise ValueError("Reconstruction dict has no image tensor key")
        tensors = payload[tensor_key]
        raw_names = payload.get("image_names")
        if raw_names is None:
            raise ValueError("Reconstruction dict has no image_names")
        names = [
            Path(value.decode() if isinstance(value, bytes) else str(value)).name
            for value in raw_names
        ]
    elif hasattr(payload, "shape"):
        tensors = payload
        names = list(manifest_names)
    else:
        raise TypeError(f"Unsupported reconstruction payload: {type(payload).__name__}")

    if len(tensors) != len(names):
        raise ValueError(
            f"Reconstruction tensor/name count mismatch: {len(tensors)} != {len(names)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("Reconstruction image_names contain duplicates")
    expected = list(manifest_names)
    missing = sorted(set(expected) - set(names))
    extra = sorted(set(names) - set(expected))
    if missing or extra or len(expected) != len(names):
        raise ValueError(
            f"Reconstruction/manifest mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}, counts={len(names)}/{len(expected)}"
        )
    by_name = {name: tensors[index] for index, name in enumerate(names)}
    return {name: _pil_from_tensor(by_name[name]) for name in expected}


def load_gt_images(image_root: str | Path, image_names: Sequence[str]) -> dict[str, Any]:
    from PIL import Image

    root = Path(image_root)
    images: dict[str, Any] = {}
    missing: list[str] = []
    for name in image_names:
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        with Image.open(path) as source:
            images[name] = source.convert("RGB").copy()
    if missing:
        raise FileNotFoundError(
            f"GT image root is missing {len(missing)} files; first={missing[:5]}"
        )
    return images


def _build_result_row(
    *,
    metric_name: str,
    bank_version: str,
    image_order: int,
    target_entry: Mapping[str, Any],
    qa: Mapping[str, Any],
    evaluated_image_name: str,
    pair_type: str,
    free_form: str,
    mapped: str,
) -> dict[str, Any]:
    acceptable_answers = list(qa.get("acceptable_answers", [qa["answer"]]))
    return {
        "metric_name": metric_name,
        "bank_version": bank_version,
        "image_order": image_order,
        "image_name": target_entry["image_name"],
        "evaluated_image_name": evaluated_image_name,
        "pair_type": pair_type,
        "question_id": qa["question_id"],
        "attribute": qa["attribute"],
        "subtype": qa["subtype"],
        "question": qa["question"],
        "choices": list(qa["choices"]),
        "answer": qa["answer"],
        "acceptable_answers": acceptable_answers,
        "free_form_answer": free_form,
        "mapped_answer": mapped,
        "correct": int(mapped in acceptable_answers),
        "chance": len(acceptable_answers) / len(qa["choices"]),
    }


def evaluate_questions(
    *,
    bank: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    images: Mapping[str, Any],
    backend: AnswerBackend,
    output_dir: str | Path,
    result_stem: str,
    resume: bool = False,
    include_counterfactual: bool = False,
    max_images: int | None = None,
    backend_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    lookup = bank_lookup(bank)
    selected = list(manifest_rows)
    if max_images is not None:
        selected = selected[: max(0, int(max_images))]
    names = [str(row["image_name"]) for row in selected]
    unknown = [name for name in names if name not in lookup]
    missing_images = [name for name in names if name not in images]
    if unknown or missing_images:
        raise ValueError(
            f"Evaluation inputs invalid: unknown={unknown[:5]}, "
            f"missing_images={missing_images[:5]}"
        )

    counterfactuals = choose_counterfactuals(bank) if include_counterfactual else {}
    metric_name = str(bank.get("metric_name", METRIC_NAME))
    bank_version = str(bank.get("version", BANK_VERSION))
    required_cf_names = (
        {
            counterfactuals[str(qa["question_id"])]
            for name in names
            for qa in lookup[name]["qas"]
            if str(qa["question_id"]) in counterfactuals
        }
        if include_counterfactual
        else set()
    )
    absent_cf = sorted(required_cf_names - set(images))
    if absent_cf:
        raise ValueError(
            "Counterfactual evaluation requires the full candidate image pool; "
            f"missing={absent_cf[:5]}"
        )

    run_payload = {
        "bank_hash": bank["metadata"].get("bank_content_sha256")
        or canonical_json_hash(bank["entries"]),
        "manifest_names": names,
        "counterfactual": include_counterfactual,
        "backend": dict(backend_identity or {}),
    }
    output = Path(output_dir)
    cache = AtomicResultCache(
        output / f"{result_stem}_cache", canonical_json_hash(run_payload), resume
    )

    total_questions = sum(
        1
        + int(
            include_counterfactual
            and str(qa["question_id"]) in counterfactuals
        )
        for name in names
        for qa in lookup[name]["qas"]
    )
    completed_questions = len(cache.rows())

    from tqdm import tqdm

    with tqdm(
        total=total_questions,
        initial=completed_questions,
        desc=metric_name,
        unit="question",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress:
        for image_order, name in enumerate(names):
            entry = lookup[name]
            for qa in entry["qas"]:
                pairs = [("factual", name)]
                if (
                    include_counterfactual
                    and str(qa["question_id"]) in counterfactuals
                ):
                    pairs.append(
                        ("counterfactual", counterfactuals[str(qa["question_id"])])
                    )
                for pair_type, evaluated_name in pairs:
                    if (
                        cache.get(pair_type, name, str(qa["question_id"]))
                        is not None
                    ):
                        continue
                    free_form, mapped = backend.answer(
                        images[evaluated_name],
                        str(qa["question"]),
                        list(qa["choices"]),
                    )
                    cache.put(
                        _build_result_row(
                            metric_name=metric_name,
                            bank_version=bank_version,
                            image_order=image_order,
                            target_entry=entry,
                            qa=qa,
                            evaluated_image_name=evaluated_name,
                            pair_type=pair_type,
                            free_form=str(free_form),
                            mapped=str(mapped),
                        )
                    )
                    progress.update(1)

    rows = cache.rows()
    write_question_results(output, result_stem, rows)
    return rows


QUESTION_RESULT_FIELDS = [
    "metric_name",
    "bank_version",
    "image_order",
    "image_name",
    "evaluated_image_name",
    "pair_type",
    "question_id",
    "attribute",
    "subtype",
    "question",
    "choices",
    "answer",
    "acceptable_answers",
    "free_form_answer",
    "mapped_answer",
    "correct",
    "chance",
]


def write_question_results(
    output_dir: str | Path, result_stem: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    atomic_write_text(output / f"{result_stem}.jsonl", jsonl)

    destination = output / f"{result_stem}.csv"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["choices"] = json.dumps(
                serialized["choices"], ensure_ascii=False
            )
            serialized["acceptable_answers"] = json.dumps(
                serialized["acceptable_answers"], ensure_ascii=False
            )
            writer.writerow(serialized)
    os.replace(temporary, destination)


def load_question_results(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["choices"] = json.loads(row["choices"])
        raw_acceptable = row.get("acceptable_answers")
        row["acceptable_answers"] = (
            json.loads(raw_acceptable)
            if raw_acceptable
            else [row["answer"]]
        )
        row["correct"] = int(row["correct"])
        row["chance"] = float(row["chance"])
        row["image_order"] = int(row["image_order"])
    return rows


def _accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([int(row["correct"]) for row in rows])) if rows else math.nan


def _macro_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    classes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        classes[str(row["answer"])].append(row)
    return float(np.mean([_accuracy(group) for group in classes.values()])) if classes else math.nan


def _random_guess_accuracy_sd(rows: Sequence[Mapping[str, Any]]) -> float:
    """SD of one participant's mean accuracy under independent uniform guesses."""

    if not rows:
        return math.nan
    probabilities = np.asarray([float(row["chance"]) for row in rows], dtype=float)
    return float(np.sqrt(np.sum(probabilities * (1.0 - probabilities))) / len(rows))


def _balanced_random_guess_stats(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    """Mean and SD of balanced accuracy under independent uniform guesses."""

    classes: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        classes[str(row["answer"])].append(float(row["chance"]))
    if not classes:
        return math.nan, math.nan
    class_count = len(classes)
    mean = float(
        np.mean([np.mean(probabilities) for probabilities in classes.values()])
    )
    variance = sum(
        sum(probability * (1.0 - probability) for probability in probabilities)
        / (len(probabilities) ** 2)
        for probabilities in classes.values()
    ) / (class_count**2)
    return mean, float(math.sqrt(variance))


def _majority_baseline_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    """Best accuracy obtainable by returning one fixed valid option."""

    if not rows:
        return math.nan
    choices = {
        str(choice)
        for row in rows
        for choice in row.get("choices", [])
    }
    if not choices:
        return math.nan
    return max(
        float(
            np.mean(
                [
                    choice
                    in {
                        str(answer)
                        for answer in row.get(
                            "acceptable_answers", [row["answer"]]
                        )
                    }
                    for row in rows
                ]
            )
        )
        for choice in choices
    )


def paired_bootstrap_difference(
    factual_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    factual = {
        (str(row["image_name"]), str(row["question_id"])): int(row["correct"])
        for row in factual_rows
    }
    counterfactual = {
        (str(row["image_name"]), str(row["question_id"])): int(row["correct"])
        for row in counterfactual_rows
    }
    if set(factual) != set(counterfactual):
        raise ValueError("Factual and counterfactual result keys do not match")
    differences = np.asarray(
        [factual[key] - counterfactual[key] for key in sorted(factual)],
        dtype=np.float64,
    )
    if not len(differences):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        size = min(1000, iterations - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        bootstrap[start : start + size] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def summarize_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    factual = [row for row in rows if row["pair_type"] == "factual"]
    counterfactual = [
        row for row in rows if row["pair_type"] == "counterfactual"
    ]
    by_attribute: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cf_by_attribute: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in factual:
        by_attribute[str(row["attribute"])].append(row)
    for row in counterfactual:
        cf_by_attribute[str(row["attribute"])].append(row)

    attribute_summary: list[dict[str, Any]] = []
    counterfactual_summary: list[dict[str, Any]] = []
    for attribute in ATTRIBUTE_ORDER:
        group = by_attribute.get(attribute, [])
        if not group:
            continue
        overall = _accuracy(group)
        macro = _macro_accuracy(group)
        chance = float(np.mean([float(row["chance"]) for row in group]))
        chance_sd = _random_guess_accuracy_sd(group)
        balanced_chance, balanced_chance_sd = _balanced_random_guess_stats(group)
        majority_baseline = _majority_baseline_accuracy(group)
        cf_group = cf_by_attribute.get(attribute, [])
        if cf_group:
            cf_keys = {
                (str(row["image_name"]), str(row["question_id"])) for row in cf_group
            }
            paired_factual = [
                row
                for row in group
                if (str(row["image_name"]), str(row["question_id"])) in cf_keys
            ]
            difference, ci_low, ci_high = paired_bootstrap_difference(
                paired_factual,
                cf_group,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
            cf_accuracy = _accuracy(cf_group)
            reliable = overall >= 0.9 and macro >= 0.9 and ci_low > 0
            counterfactual_summary.append(
                {
                    "attribute": attribute,
                    "n_pairs": len(paired_factual),
                    "factual_accuracy": _accuracy(paired_factual),
                    "counterfactual_accuracy": cf_accuracy,
                    "paired_difference": difference,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_iterations": bootstrap_iterations,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
        else:
            cf_accuracy = math.nan
            ci_low = math.nan
            reliable = None
        attribute_summary.append(
            {
                "attribute": attribute,
                "n_questions": len(group),
                "overall_accuracy": overall,
                "macro_accuracy": macro,
                "theoretical_chance": chance,
                "theoretical_chance_sd": chance_sd,
                "theoretical_balanced_chance": balanced_chance,
                "theoretical_balanced_chance_sd": balanced_chance_sd,
                "majority_baseline_accuracy": majority_baseline,
                "counterfactual_accuracy": cf_accuracy,
                "n_counterfactual_pairs": len(cf_group),
                "counterfactual_ci95_low": ci_low,
                "passes_gt_reliability": reliable,
            }
        )

    subtype_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in factual:
        subtype_groups[(str(row["attribute"]), str(row["subtype"]))].append(row)
    subtype_summary = [
        {
            "attribute": attribute,
            "subtype": subtype,
            "n_questions": len(group),
            "overall_accuracy": _accuracy(group),
            "macro_accuracy": _macro_accuracy(group),
            "theoretical_chance": float(
                np.mean([float(row["chance"]) for row in group])
            ),
        }
        for (attribute, subtype), group in sorted(subtype_groups.items())
    ]

    confusion = Counter(
        (
            str(row["attribute"]),
            str(row["subtype"]),
            str(row["answer"]),
            str(row["mapped_answer"]),
        )
        for row in factual
    )
    confusion_rows = [
        {
            "attribute": attribute,
            "subtype": subtype,
            "answer": answer,
            "mapped_answer": mapped,
            "count": count,
        }
        for (attribute, subtype, answer, mapped), count in sorted(confusion.items())
    ]
    return {
        "attribute_summary": attribute_summary,
        "subtype_summary": subtype_summary,
        "confusion_matrices": confusion_rows,
        "counterfactual_summary": counterfactual_summary,
    }


def write_summary_tables(
    output_dir: str | Path,
    summary: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    prefix: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for key, rows in summary.items():
        destination = output / f"{prefix}_{key}.csv"
        rows = list(rows)
        fields = list(rows[0].keys()) if rows else ["attribute"]
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, destination)
