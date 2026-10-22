#!/usr/bin/env python3
"""Generate the AVID simple visual-identity-disjoint v1 split.

The split is derived only from the frozen Qwen v3 ground-truth attribute bank.
No reconstruction or model-performance result is used during selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = (
    REPO_ROOT
    / "avid_benchmark/vendor/vmi_metrics/simple_attribute_bank_qwen_v3.json"
)
ORIGINAL_VAL_PATH = (
    REPO_ROOT
    / "avid_benchmark/vendor/vmi_metrics/simple_sub01_current_val80_manifest.csv"
)
OUTPUT_DIR = REPO_ROOT / "avid_benchmark/vendor/vmi_metrics"
TRAIN_PATH = OUTPUT_DIR / "simple_identity_disjoint_v1_train.csv"
VAL_PATH = OUTPUT_DIR / "simple_identity_disjoint_v1_val.csv"
COMPARISON_PATH = (
    OUTPUT_DIR / "simple_identity_disjoint_v1_attribute_comparison.md"
)

SEED = "AVID-simple-identity-disjoint-v1"
# SHA-256 of the released bank's UTF-8 text with LF line endings.
BANK_SHA256 = "cfab4aac69f418d6c5755af1f069aa0c3e37398e831dc53ff3c3322390f8002a"
FAMILY_IDENTITY_QUOTAS = {
    "letter": 2,
    "single_digit": 2,
    "digit_sequence": 2,
    "shape": 3,
}
OBJECT_COLOR_QUOTAS = {
    "red": 1,
    "yellow": 1,
    "blue": 1,
    "other or multicolored": 8,
}
EXPECTED_VAL_FAMILIES = {
    "letter": 12,
    "single_digit": 6,
    "digit_sequence": 6,
    "shape": 45,
    "object": 11,
}
EXPECTED_VAL_COLORS = {
    "red": 24,
    "yellow": 24,
    "blue": 24,
    "other or multicolored": 8,
}
EXPECTED_VAL_POSITIONS = {
    "top": 9,
    "bottom": 9,
    "left": 9,
    "right": 9,
    "center": 44,
}
CSV_FIELDS = [
    "order_index",
    "image_name",
    "subset_name",
    "simple_id",
    "family",
    "color",
    "position",
    "visual_identity",
    "caption",
]


def _hash_key(family: str, identity: str) -> str:
    value = f"{SEED}|{family}|{identity}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load_entries() -> list[dict]:
    text = BANK_PATH.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != BANK_SHA256:
        raise ValueError("The attribute bank differs from the frozen identity-disjoint v1 source")
    bank = json.loads(text)
    if bank.get("version") != "qwen_v3":
        raise ValueError("Expected the frozen qwen_v3 attribute bank")
    entries = sorted(bank["entries"], key=lambda entry: int(entry["simple_id"]))
    if len(entries) != 400:
        raise ValueError(f"Expected 400 simple stimuli, found {len(entries)}")
    names = [entry["image_name"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("The attribute bank contains duplicate image names")
    return entries


def _identity_rows(entries: Iterable[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        grouped[entry["family"]][entry["labels"]["visual_identity"]].append(entry)
    return grouped


def _select_shape_identities(identities: Iterable[str], quota: int) -> list[str]:
    """Hash-select diverse shapes with at most one semicircle and one rectangle."""

    selected: list[str] = []
    semicircles = 0
    rectangles = 0
    for identity in sorted(identities, key=lambda value: _hash_key("shape", value)):
        is_semicircle = identity.startswith("semicircle (")
        is_rectangle = "rectangle" in identity
        if is_semicircle and semicircles >= 1:
            continue
        if is_rectangle and rectangles >= 1:
            continue
        selected.append(identity)
        semicircles += int(is_semicircle)
        rectangles += int(is_rectangle)
        if len(selected) == quota:
            return selected
    raise ValueError("Unable to satisfy the shape identity quota")


def _select_val_identities(entries: list[dict]) -> dict[str, list[str]]:
    grouped = _identity_rows(entries)
    selected: dict[str, list[str]] = {}

    for family, quota in FAMILY_IDENTITY_QUOTAS.items():
        identities = grouped[family].keys()
        if family == "shape":
            selected[family] = _select_shape_identities(identities, quota)
        else:
            selected[family] = sorted(
                identities, key=lambda value: _hash_key(family, value)
            )[:quota]

    objects_by_color: dict[str, list[str]] = defaultdict(list)
    for identity, rows in grouped["object"].items():
        colors = {row["labels"]["color"] for row in rows}
        if len(rows) != 1 or len(colors) != 1:
            raise ValueError(
                f"Expected singleton object identity, found {identity}: "
                f"rows={len(rows)}, colors={sorted(colors)}"
            )
        objects_by_color[next(iter(colors))].append(identity)

    selected_objects: list[str] = []
    for color, quota in OBJECT_COLOR_QUOTAS.items():
        candidates = sorted(
            objects_by_color[color],
            key=lambda value: _hash_key("object", value),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"Object color {color!r} has {len(candidates)} candidates "
                f"but requires {quota}"
            )
        selected_objects.extend(candidates[:quota])
    selected["object"] = selected_objects
    return selected


def _partition(
    entries: list[dict], selected: dict[str, list[str]]
) -> tuple[list[dict], list[dict]]:
    selected_pairs = {
        (family, identity)
        for family, identities in selected.items()
        for identity in identities
    }
    val = [
        entry
        for entry in entries
        if (entry["family"], entry["labels"]["visual_identity"]) in selected_pairs
    ]
    train = [entry for entry in entries if entry not in val]
    return train, val


def _counter(entries: Iterable[dict], attribute: str) -> Counter:
    if attribute == "family":
        return Counter(entry["family"] for entry in entries)
    return Counter(entry["labels"][attribute] for entry in entries)


def _validate(train: list[dict], val: list[dict]) -> None:
    train_names = {entry["image_name"] for entry in train}
    val_names = {entry["image_name"] for entry in val}
    train_identities = {entry["labels"]["visual_identity"] for entry in train}
    val_identities = {entry["labels"]["visual_identity"] for entry in val}

    if len(train) != 320 or len(val) != 80:
        raise ValueError(f"Unexpected split sizes: train={len(train)}, val={len(val)}")
    if train_names & val_names:
        raise ValueError("Train and validation image names overlap")
    if len(train_names | val_names) != 400:
        raise ValueError("Train and validation do not cover all 400 stimuli")
    if train_identities & val_identities:
        raise ValueError(
            "Visual-identity leakage: "
            f"{sorted(train_identities & val_identities)}"
        )
    if (len(train_identities), len(val_identities)) != (75, 20):
        raise ValueError("Expected 75 training and 20 validation identities")
    if dict(_counter(val, "family")) != EXPECTED_VAL_FAMILIES:
        raise ValueError(f"Unexpected validation family counts: {_counter(val, 'family')}")
    if dict(_counter(val, "color")) != EXPECTED_VAL_COLORS:
        raise ValueError(f"Unexpected validation color counts: {_counter(val, 'color')}")
    if dict(_counter(val, "position")) != EXPECTED_VAL_POSITIONS:
        raise ValueError(
            f"Unexpected validation position counts: {_counter(val, 'position')}"
        )


def _csv_row(entry: dict, order_index: int, subset_name: str) -> dict:
    labels = entry["labels"]
    return {
        "order_index": order_index,
        "image_name": entry["image_name"],
        "subset_name": subset_name,
        "simple_id": entry["simple_id"],
        "family": entry["family"],
        "color": labels["color"],
        "position": labels["position"],
        "visual_identity": labels["visual_identity"],
        "caption": entry["caption"],
    }


def _write_csv(path: Path, entries: list[dict], subset_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for order_index, entry in enumerate(entries):
            writer.writerow(_csv_row(entry, order_index, subset_name))


def _load_original_val_names() -> set[str]:
    with ORIGINAL_VAL_PATH.open("r", encoding="utf-8", newline="") as handle:
        names = {row["image_name"] for row in csv.DictReader(handle)}
    if len(names) != 80:
        raise ValueError(
            f"Expected 80 original validation stimuli, found {len(names)}"
        )
    return names


def _format_count(count: int, total: int) -> str:
    return f"{count} ({100.0 * count / total:.2f}%)"


def _write_comparison(
    entries: list[dict],
    old_train: list[dict],
    old_val: list[dict],
    new_train: list[dict],
    new_val: list[dict],
    selected: dict[str, list[str]],
    path: Path,
) -> None:
    attribute_values = {
        "family": [
            "letter",
            "single_digit",
            "digit_sequence",
            "shape",
            "object",
        ],
        "color": ["red", "yellow", "blue", "other or multicolored"],
        "position": ["top", "bottom", "left", "right", "center"],
    }
    partitions = [old_train, old_val, new_train, new_val]
    lines = [
        "# Simple split attribute-distribution comparison",
        "",
        f"- Source attribute bank: `{BANK_PATH.relative_to(REPO_ROOT).as_posix()}`",
        f"- Original validation manifest: `{ORIGINAL_VAL_PATH.relative_to(REPO_ROOT).as_posix()}`",
        f"- New split seed: `{SEED}`",
        "- New split selection is independent of reconstruction performance.",
        "",
        "| Attribute | Value | Original train | Original val | "
        "Identity-disjoint train | Identity-disjoint val |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for attribute, values in attribute_values.items():
        counters = [_counter(partition, attribute) for partition in partitions]
        for value in values:
            lines.append(
                f"| {attribute} | {value} | "
                f"{_format_count(counters[0][value], len(old_train))} | "
                f"{_format_count(counters[1][value], len(old_val))} | "
                f"{_format_count(counters[2][value], len(new_train))} | "
                f"{_format_count(counters[3][value], len(new_val))} |"
            )

    def identity_stats(train: list[dict], val: list[dict]) -> tuple[int, int, int, int]:
        train_ids = {entry["labels"]["visual_identity"] for entry in train}
        val_ids = {entry["labels"]["visual_identity"] for entry in val}
        leaked_stimuli = sum(
            entry["labels"]["visual_identity"] in train_ids for entry in val
        )
        return len(train_ids), len(val_ids), len(train_ids & val_ids), leaked_stimuli

    old_stats = identity_stats(old_train, old_val)
    new_stats = identity_stats(new_train, new_val)
    lines.extend(
        [
            "",
            "## Visual-identity separation",
            "",
            "| Split | Unique train identities | Unique val identities | "
            "Overlapping identities | Val stimuli whose identity occurs in train |",
            "|---|---:|---:|---:|---:|",
            f"| Original | {old_stats[0]} | {old_stats[1]} | "
            f"{old_stats[2]} | {old_stats[3]}/80 |",
            f"| Identity-disjoint v1 | {new_stats[0]} | {new_stats[1]} | "
            f"{new_stats[2]} | {new_stats[3]}/80 |",
            "",
            "## Held-out validation identities",
            "",
        ]
    )
    for family in [
        "letter",
        "single_digit",
        "digit_sequence",
        "shape",
        "object",
    ]:
        values = ", ".join(f"`{value}`" for value in selected[family])
        lines.append(f"- **{family}:** {values}")

    full_names = {entry["image_name"] for entry in entries}
    if len(full_names) != 400:
        raise ValueError("Comparison input does not contain 400 unique stimuli")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def generate(output_dir: Path) -> None:
    entries = _load_entries()
    selected = _select_val_identities(entries)
    new_train, new_val = _partition(entries, selected)
    _validate(new_train, new_val)

    original_val_names = _load_original_val_names()
    if not original_val_names <= {entry["image_name"] for entry in entries}:
        raise ValueError("Original validation manifest contains unknown stimuli")
    old_val = [entry for entry in entries if entry["image_name"] in original_val_names]
    old_train = [
        entry for entry in entries if entry["image_name"] not in original_val_names
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / TRAIN_PATH.name, new_train, "identity_disjoint_v1_train")
    _write_csv(output_dir / VAL_PATH.name, new_val, "identity_disjoint_v1_val")
    _write_comparison(
        entries,
        old_train,
        old_val,
        new_train,
        new_val,
        selected,
        output_dir / COMPARISON_PATH.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory; defaults to the bundled manifest directory.")
    parser.add_argument("--check", action="store_true",
                        help="Verify existing outputs without modifying them (ignores line endings).")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    filenames = (TRAIN_PATH.name, VAL_PATH.name, COMPARISON_PATH.name)
    if args.check:
        with tempfile.TemporaryDirectory() as temporary:
            expected_dir = Path(temporary)
            generate(expected_dir)
            for name in filenames:
                path = output_dir / name
                if not path.is_file() or path.read_text(encoding="utf-8") != (
                    expected_dir / name
                ).read_text(encoding="utf-8"):
                    raise SystemExit(f"Missing or inconsistent split artifact: {path}")
        print("Verified identity-disjoint v1: 320/80 stimuli, 75/20 identities, zero overlap.")
    else:
        generate(output_dir)
        for name in filenames:
            print(f"Wrote {output_dir / name}")


if __name__ == "__main__":
    main()
