# Simple visual-identity-disjoint split

The additional `simple_identity_disjoint_v1` protocol was prepared for the rebuttal to test generalization to unseen visual identities. It partitions the 400 simple stimuli into **320 training and 80 validation stimuli**, with **75 training identities and 20 validation identities and zero identity overlap**. The original stimulus-level split remains the default for the baseline commands and the original benchmark results.

## Released files

The split is defined by these fixed, ordered manifests, joined to stimuli and fMRI samples by `image_name`:

- [Training manifest (320 rows)](../avid_benchmark/vendor/vmi_metrics/simple_identity_disjoint_v1_train.csv)
- [Validation manifest (80 rows)](../avid_benchmark/vendor/vmi_metrics/simple_identity_disjoint_v1_val.csv)
- [Attribute distributions, overlap checks, and held-out identities](../avid_benchmark/vendor/vmi_metrics/simple_identity_disjoint_v1_attribute_comparison.md)
- [Frozen Qwen v3 annotations for all 400 stimuli](../avid_benchmark/vendor/vmi_metrics/simple_attribute_bank_qwen_v3.json)
- [Original subject-01 validation manifest](../avid_benchmark/vendor/vmi_metrics/simple_sub01_current_val80_manifest.csv), used only for the original-versus-new comparison

Each new manifest includes `order_index`, `image_name`, `subset_name`, `simple_id`, `family`, `color`, `position`, `visual_identity`, and `caption`. The two manifests cover all 400 stimuli exactly once. Their membership and order match the previously prepared rebuttal artifacts.

## Definition and selection

Canonical identity is the `labels.visual_identity` field in the frozen Qwen v3 bank. All variants of an identity stay in one partition, including different colors, positions, and letter cases. Orientation-specific shape labels remain distinct identities as defined in that bank. Color and position values deliberately remain shared across partitions: this is **visual-identity-disjoint**, rather than fully semantic-disjoint, color-disjoint, or position-disjoint.

The deterministic seed is `AVID-simple-identity-disjoint-v1`. Within each family, identities are ranked by the hexadecimal SHA-256 of `seed|family|identity`; the lowest-ranked eligible identities are selected for validation. The quotas are two letters, two single digits, two digit sequences, three shapes, and eleven natural objects. Shape selection allows at most one semicircle identity and one rectangle identity. Each natural-object identity is a singleton; objects are selected within color strata (one red, one yellow, one blue, and eight other or multicolored). Selection uses annotations only, independently of reconstruction performance.

| Validation attribute | Counts |
|---|---|
| Family | 12 letter / 6 single digit / 6 digit sequence / 45 shape / 11 object |
| Color | 24 red / 24 yellow / 24 blue / 8 other or multicolored |
| Position | 9 top / 9 bottom / 9 left / 9 right / 44 center |

For the subject-01 original split used in the comparison, 35 identities occur in both partitions and 66 of the 80 validation stimuli have an identity seen in training. The new split has zero for both overlap measures. The comparison manifest is a reference for that subject; it is not an instruction to infer every participant's original storage partition.

## Reproduce or verify

From a repository checkout, use Python 3.10 or later; the generator uses only the standard library and needs no fMRI data, model weights, or network access:

```bash
# Verify all three published outputs without modifying them.
python scripts/generate_simple_identity_disjoint_split.py --check

# Regenerate into a separate directory.
python scripts/generate_simple_identity_disjoint_split.py --output-dir outputs/identity_disjoint_v1
python scripts/generate_simple_identity_disjoint_split.py --output-dir outputs/identity_disjoint_v1 --check

# Run the split regressions.
python -m pytest tests/test_identity_disjoint_split.py -q
```

Without arguments, the generator regenerates the bundled train/validation CSVs and comparison report. Source paths are resolved from the script location; `--output-dir` is relative to the working directory. The generator validates split sizes, stimulus coverage, identity separation, and validation attribute counts. `--check` also detects missing or modified artifacts and exits with a nonzero status on a mismatch. Line-ending differences are ignored when checking.

The v1 generator pins the source bank by SHA-256. For provenance, the following hashes use UTF-8 text normalized to LF line endings:

| File | SHA-256 |
|---|---|
| `simple_attribute_bank_qwen_v3.json` | `cfab4aac69f418d6c5755af1f069aa0c3e37398e831dc53ff3c3322390f8002a` |
| `simple_identity_disjoint_v1_train.csv` | `7848d2b0a1781486f2b547b905920ed00eff4c07da997ed6a11787760f0e07c0` |
| `simple_identity_disjoint_v1_val.csv` | `ac1ce7f715458305327ec4e2fa2dadb6c5c90355f74af684552b12998c222baa` |

## Using the manifests with fMRI data

For each participant, read the **union of the original simple train and validation shards**, then assign samples to the new partition using `image_name`. Filtering only the original training shard cannot recover the full new training set. Keep every repeat of the same stimulus in the same new partition, retain participant IDs and response/vertex ordering, and verify coverage against that participant's data. Use the same identity assignment for every participant in multi-subject training. Keep validation identities out of training across all participants, including pretraining and fine-tuning, and document the checkpoint-selection policy and any separate development split.

For a custom data loader, the membership sets can be read as follows:

```python
import csv
from pathlib import Path

root = Path("avid_benchmark/vendor/vmi_metrics")
def image_names(partition):
    with (root / f"simple_identity_disjoint_v1_{partition}.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return {row["image_name"] for row in csv.DictReader(handle)}

train_names = image_names("train")
validation_names = image_names("val")
assert train_names.isdisjoint(validation_names)
```

The current `train.py` / `predict.py` commands read the original shard partitions and do **not** yet expose a manifest-based split selector. Merely copying these CSVs into the repository does not change those commands' behavior. This release provides the additional protocol, annotations, generator, and checks; it does not report baseline performance under the new split. Experiments under this protocol require a loader that repartitions the original shard union and models trained using that assignment. Report the split version alongside results and evaluate each participant separately using the corresponding validation candidate set.
