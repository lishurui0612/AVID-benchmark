import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = ROOT / "avid_benchmark/vendor/vmi_metrics"
SCRIPT = ROOT / "scripts/generate_simple_identity_disjoint_split.py"
TRAIN = "simple_identity_disjoint_v1_train.csv"
VAL = "simple_identity_disjoint_v1_val.csv"
COMPARISON = "simple_identity_disjoint_v1_attribute_comparison.md"


def read_rows(name):
    with (BANK_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_released_identity_disjoint_protocol():
    train, val = read_rows(TRAIN), read_rows(VAL)
    assert (len(train), len(val)) == (320, 80)
    train_names = {row["image_name"] for row in train}
    val_names = {row["image_name"] for row in val}
    assert not train_names & val_names
    bank = json.loads((BANK_DIR / "simple_attribute_bank_qwen_v3.json").read_text(encoding="utf-8"))
    entries = {entry["image_name"]: entry for entry in bank["entries"]}
    assert train_names | val_names == set(entries)
    for rows, subset in [(train, "train"), (val, "val")]:
        assert [int(row["order_index"]) for row in rows] == list(range(len(rows)))
        assert [int(row["simple_id"]) for row in rows] == sorted(int(row["simple_id"]) for row in rows)
        for row in rows:
            entry = entries[row["image_name"]]
            assert row["subset_name"] == f"identity_disjoint_v1_{subset}"
            assert int(row["simple_id"]) == entry["simple_id"]
            assert row["family"] == entry["family"]
            assert row["caption"] == entry["caption"]
            assert all(row[key] == entry["labels"][key] for key in ("color", "position", "visual_identity"))
    train_ids = {row["visual_identity"] for row in train}
    val_ids = {row["visual_identity"] for row in val}
    assert (len(train_ids), len(val_ids)) == (75, 20)
    assert not train_ids & val_ids
    assert Counter(row["family"] for row in val) == {
        "letter": 12, "single_digit": 6, "digit_sequence": 6, "shape": 45, "object": 11,
    }
    assert Counter(row["color"] for row in val) == {
        "red": 24, "yellow": 24, "blue": 24, "other or multicolored": 8,
    }
    assert Counter(row["position"] for row in val) == {
        "top": 9, "bottom": 9, "left": 9, "right": 9, "center": 44,
    }
    # Freeze every field and the order of the previously prepared v1 manifests.
    for name, expected in {
        TRAIN: "7848d2b0a1781486f2b547b905920ed00eff4c07da997ed6a11787760f0e07c0",
        VAL: "ac1ce7f715458305327ec4e2fa2dadb6c5c90355f74af684552b12998c222baa",
    }.items():
        text = (BANK_DIR / name).read_text(encoding="utf-8")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected
    old_val = {row["image_name"] for row in read_rows("simple_sub01_current_val80_manifest.csv")}
    assert len(old_val) == 80
    assert old_val <= set(entries)
    old_train_ids = {entry["labels"]["visual_identity"] for name, entry in entries.items() if name not in old_val}
    old_val_ids = {entries[name]["labels"]["visual_identity"] for name in old_val}
    assert len(old_train_ids & old_val_ids) == 35
    assert sum(entries[name]["labels"]["visual_identity"] in old_train_ids for name in old_val) == 66


def test_generator_and_read_only_check_from_other_directory(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp_path,
                              capture_output=True, text=True)

    assert run("--check").returncode == 0
    output = tmp_path / "regenerated"
    generated = run("--output-dir", str(output))
    assert generated.returncode == 0, generated.stderr
    for name in (TRAIN, VAL, COMPARISON):
        assert (output / name).read_text(encoding="utf-8") == (BANK_DIR / name).read_text(encoding="utf-8")
    assert run("--output-dir", str(output), "--check").returncode == 0
    damaged = output / VAL
    damaged.write_text("invalid split\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    checked = run("--output-dir", str(output), "--check")
    assert checked.returncode != 0
    assert "inconsistent split artifact" in checked.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
