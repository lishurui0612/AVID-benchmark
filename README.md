# AVID Benchmark

Evaluation tools and reconstruction baselines for **AVID: A 5T fMRI Dataset for Benchmarking Auditory-induced Visual Mental Imagery Decoding**.

This repository provides the benchmark evaluation code and AVID adaptations of **LDM, MindEye1, MindEye2, and MindBridge**.

[Introduction](#introduction) · [Data](#data-and-configuration) · [Asset downloads](docs/ASSETS.md) · [Evaluation](#evaluation) · [Baselines](#baseline-training-and-prediction) · [Sources](#sources) · [Citation](#citation) · [Acknowledgements](#acknowledgements)

## Introduction

Visual mental imagery allows people to construct visual scenes without seeing them. Decoding these internally generated representations from brain activity requires data and evaluation protocols tailored to imagery. Many existing paradigms ask participants to view or memorize images before imagining them, making it difficult to distinguish imagery-related signals from visual perception or memory of a recently presented image.

**AVID (Auditory-induced Visual Imagery Dataset)** is a 5T fMRI resource for studying imagery elicited by spoken descriptions. It contains approximately **67 hours of recordings from nine participants** and two complementary stimulus sets: **complex** scenes described in naturalistic Mandarin and **simple** stimuli with controlled visual attributes. Participants construct the described images mentally, providing a setting for studying reconstruction from verbally specified content.

The AVID benchmark distinguishes **reference-image consistency** from **semantic alignment** with the descriptions, and evaluates controlled visual attributes and generated captions. A reference image is a stimulus-associated comparison target; it is not a direct observation of what a participant imagined. Reporting these aspects separately makes the scope of reconstruction scores explicit.

This repository has two main purposes: provide independently callable benchmark metrics, and support baseline training and reconstruction through separate commands. Users can evaluate their own models with the same scoring functions and adapt the supplied baseline recipes to the downloaded AVID data.

## Overview

The benchmark evaluates both consistency with reference images and alignment with the described content. These measure different aspects of imagery reconstruction and should be reported separately.

| Component | What it provides |
|---|---|
| Evaluation | One callable function per metric, with per-sample scores, sample IDs, and protocol information |
| Baselines | Separate training and prediction commands for four reconstruction methods |
| Configuration | Editable paths to downloaded AVID data and existing pretrained assets |

Training, prediction, and evaluation are separate stages. You can evaluate predictions from your own model without training any bundled baseline. This repository assumes that the released, preprocessed data are already available locally.

## Installation

Use **Python 3.10** for the pinned baseline environment. CUDA GPUs are needed for baseline neural-network training and diffusion generation; scoring precomputed arrays or saved answers can run on CPU.

```bash
git clone https://github.com/lishurui0612/AVID-benchmark.git
cd AVID-benchmark
python -m pip install -e .
```

Choose the additional dependencies for your workflow:

| Workflow | Installation |
|---|---|
| Image and text feature extraction | `python -m pip install -e ".[evaluation]"` |
| Baseline training and prediction | `python -m pip install -e ".[models]"` |
| Unit tests | `python -m pip install -e ".[test]"` |

The baseline extra pins PyTorch, Transformers, Diffusers, and related packages; see [pyproject.toml](pyproject.toml). Use a separate environment for Qwen3-VL attribute extraction: it requires a Transformers version that provides `Qwen3VLForConditionalGeneration`, whereas the baseline environment uses an older pinned version. TIFA inference additionally requires the ModelScope/mPLUG backend. A clean, portable TIFA installation remains to be verified; the baseline extra alone does not install that backend. Scoring saved attribute answers or precomputed TIFA question scores does not load either inference backend.

## Data and configuration

The AVID manuscript provides [full data](https://www.scidb.cn/en/anonymous/dXltWWJp) and [tutorial data](https://www.scidb.cn/en/anonymous/em1peTZi) through ScienceDB. Follow the data provider's access conditions, download the supplied files, and point the configuration to their local locations. Data and pretrained weights are distributed separately from this repository. Download links are collected in [Asset downloads](docs/ASSETS.md).

The reader accepts the released tar shards, either directly under the dataset root or in subject subdirectories:

```text
<dataset_root>/
└── sub-01/
    ├── sub-01_complex_train.tar
    ├── sub-01_complex_val.tar
    ├── sub-01_simple_train.tar
    └── sub-01_simple_val.tar
```

Each tar sample contains `<key>.img_name.npy` and `<key>.responses.npy`. Image names must match the supplied stimulus images and the `image` column in `used_stimuli.tsv`. English semantic references use its `translated_caption` column. Keep the released splits and vertex order intact. The held-out prediction split is named `val` in the files and manifests.

Copy the template and replace its placeholder paths:

```bash
cp config.yaml config.local.yaml
```

| Configuration section | Set these paths |
|---|---|
| `paths` | Tar dataset root, complex/simple stimulus images, `used_stimuli.tsv`, output directory, pretrained asset root |
| `assets` | Shared CLIP weights and MindEye1's frozen VAE/ConvNeXt teacher weights, as required |
| `model_overrides` | Pretrained generators and subject-specific reconstruction checkpoints for the chosen baseline |
| `evaluation` | Local weights for the requested encoders or VQA models, and the relevant question bank |

Only the assets needed by your selected workflow have to be available. Existing data folders and weight files can be used in place; their directory names do not have to change. Model loading uses local assets with Hugging Face offline mode enabled.

For attribute evaluation, use the bundled [qwen_v3 bank](avid_benchmark/vendor/vmi_metrics/simple_attribute_bank_qwen_v3.json). TIFA uses the fixed [complex](avid_benchmark/vendor/vmi_metrics/tifa_bank_complex.json) or [simple](avid_benchmark/vendor/vmi_metrics/tifa_bank_simple.json) bank. Set the corresponding configuration entries to the bank for your stimulus type.

## Evaluation

### Evaluate reconstructed images

Pass the JSONL manifest produced by a baseline, or a compatible manifest from your own model:

```bash
python evaluate.py \
  --predictions /path/to/predictions/manifest.jsonl \
  --config config.local.yaml \
  --metrics pixcorr ssim inception_2way clip_i_2way clip_t_2way \
  --output scores.json
```

Each manifest row identifies one reconstruction:

```json
{"subject_id": 1, "stimulus_type": "complex", "split": "val", "sample_id": "sub01_example", "image_name": "example.jpg", "generated_image_path": "images/example.png", "gt_image_path": "references/example.jpg", "reference_caption": "A description of the reference scene."}
```

Image paths may be absolute or relative to the manifest. Semantic evaluation requires a reference caption for every stimulus, supplied in the manifest or looked up in `used_stimuli.tsv`. Caption metrics additionally read `generated_text`; an empty prediction scores zero.

### Supported metrics

| Group | Metrics | CLI names |
|---|---|---|
| Reference-image consistency | Pixel correlation, SSIM | `pixcorr`, `ssim` |
| Image identification | Inception, CLIP-I, AlexNet layers 2/5 | `inception_2way`, `clip_i_2way`, `alexnet_2_2way`, `alexnet_5_2way` |
| Feature distance | EfficientNet, SwAV correlation distance | `efficientnet_distance`, `swav_distance` |
| Semantic alignment | CLIP-T, SigLIP, AltCLIP identification; CLIP-T retrieval | `clip_t_2way`, `siglip_2way`, `altclip_2way`, `clip_t_recall_at_1`, `clip_t_recall_at_5` |
| Question answering | TIFA with a fixed question bank | `tifa` |
| Visual attributes | Color, position, visual identity: raw accuracy, balanced accuracy, balanced chance | `color_*`, `position_*`, `identity_*` |
| Captions | BLEU-1/4, ROUGE-L F1, token F1, exact match | `bleu1`, `bleu4`, `rouge_l_f1`, `token_f1`, `exact_match` |

For attributes, `*` denotes `accuracy`, `balanced_accuracy`, or `balanced_chance`. Expand these names explicitly on the command line. Full signatures, preprocessing, tie handling, and score definitions are documented in [METRICS.md](METRICS.md).

### Use individual functions

Scoring functions accept arrays or saved answers independently of feature extraction:

```python
import numpy as np
from avid_benchmark.evaluation import compute_clip_t_2way, compute_token_f1

# Replace these toy embeddings with aligned features for your candidate set.
sample_ids = ["image_1", "image_2", "image_3"]
image_embeddings = np.eye(3)
text_embeddings = np.eye(3)

result = compute_clip_t_2way(
    image_embeddings, text_embeddings, sample_ids=sample_ids
)
print(result.score, result.n, result.values)

caption_result = compute_token_f1(
    ["a red circle"], ["a red circle"], sample_ids=["simple_1"]
)
```

Every function returns a `MetricResult` containing the aggregate score, per-sample values, IDs, denominator, and protocol. Importing the scoring functions does not load baseline models or encoders.

Precomputed arrays can also be evaluated with `--inputs features.npz`; NPZ keys must match the selected function's arguments. Saved Qwen answers can be rescored without Qwen inference:

```bash
python evaluate.py \
  --answers /path/to/answers.jsonl \
  --metrics color_accuracy position_accuracy identity_accuracy \
  --output attribute_scores.json
```

**Evaluation protocol.** Compute identification and retrieval on each subject's complete, fixed candidate set. Do not pool repeated stimuli across subjects as distractors. Keep the stimulus split, encoder weights, candidate count, and reconstruction route consistent when comparing results. Caption BLEU follows the AVID implementation's add-one smoothing; it is not interchangeable with unsmoothed BLEU. Attribute scores use the frozen `qwen_v3` annotation bank.

## Baseline training and prediction

The bundled recipes implement the AVID training settings for each method:

| Method | Training setting | CLI model |
|---|---|---|
| LDM | Single-subject ridge decoding of diffusion targets | `ldm` |
| MindEye1 | Single-subject high-level model; independently trained low-level model | `mindeye1` |
| MindEye2 | Leave-one-subject-out pretraining, followed by target-subject fine-tuning | `mindeye2` |
| MindBridge | Joint multi-subject training | `mindbridge` |

For example, train MindEye1's high-level model, then generate predictions in a separate command:

```bash
python train.py \
  --model mindeye1 --subject 1 --stimulus complex \
  --config config.local.yaml

python predict.py \
  --model mindeye1 --subject 1 --stimulus complex \
  --checkpoint /path/to/highlevel/best.pth \
  --config config.local.yaml
```

With the default img2img setting, also configure a compatible low-level checkpoint. The prediction command prints the output manifest path; pass that path to `evaluate.py`. Training can compute validation loss to select checkpoints, but does not generate final reconstructions or run the benchmark. Prediction does not run evaluation.

See [Baseline guide](docs/BASELINES.md) for all four training commands, MindEye1 low-level training, MindEye2 distributed training and full generation, required pretrained assets, and candidate-selection protocols. Default hyperparameters are in [recipes](avid_benchmark/recipes); `--recipe` accepts an edited JSON or YAML recipe.

## Repository layout

```text
avid_benchmark/
├── data/          # Released-data readers and metadata lookup
├── evaluation/    # Independent metrics and feature/VQA backends
├── models/        # AVID baseline adapters and training code
├── recipes/       # Per-model hyperparameters
└── vendor/        # Adapted upstream code and fixed question banks
docs/              # Baseline usage
tests/             # Metric, model, and distributed-gradient checks
config*.yaml       # Local-path templates
train.py           # Training entry point
predict.py         # Prediction entry point
evaluate.py        # Evaluation entry point
```

For development, run `python -m pytest tests -q`. The optional two-process CPU gradient check is `torchrun --standalone --nproc_per_node=2 tests/ddp_gradient_check.py`.

## Citation

If you use the AVID dataset or benchmark in your research, please cite our accompanying manuscript:

**AVID: A 5T fMRI Dataset for Benchmarking Auditory-induced Visual Mental Imagery Decoding.**

### Paper citation

The fields below are placeholders for the final manuscript metadata. Replace them with the released author list, publication details, and paper URL before citing.

```bibtex
@misc{avid_dataset,
  title = {{AVID}: A 5T fMRI Dataset for Benchmarking Auditory-induced Visual Mental Imagery Decoding},
  author = {AUTHOR_LIST},
  year = {YEAR},
  howpublished = {VENUE_OR_PREPRINT},
  url = {PAPER_URL}
}
```

### Repository citation

For the software, include the repository URL and the commit used. The following entry identifies this code repository:

```bibtex
@misc{avid_benchmark,
  title = {{AVID Benchmark}: Evaluation Tools and Reconstruction Baselines},
  howpublished = {\url{https://github.com/lishurui0612/AVID-benchmark}},
  note = {Source code accompanying the AVID dataset and benchmark}
}
```

## Sources

We thank the authors and contributors of the following projects for making their methods and implementations available. The AVID baselines adapt the original reconstruction methods to the released imagery data; their original datasets and settings are documented in the upstream repositories.

| Method or tool | Original work | Upstream implementation |
|---|---|---|
| LDM reconstruction | [Takagi and Nishimoto, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html) | [StableDiffusionReconstruction](https://github.com/yu-takagi/StableDiffusionReconstruction); the AVID adapter uses [Himalaya](https://github.com/gallantlab/himalaya) and [Diffusers](https://github.com/huggingface/diffusers) |
| MindEye1 | [Scotti et al., NeurIPS 2023](https://arxiv.org/abs/2305.18274) | [fMRI-reconstruction-NSD](https://github.com/MedARC-AI/fMRI-reconstruction-NSD) |
| MindEye2 | [Scotti et al., ICML 2024](https://arxiv.org/abs/2403.11207) | [MindEyeV2](https://github.com/MedARC-AI/MindEyeV2) |
| MindBridge | [Wang et al., CVPR 2024](https://arxiv.org/abs/2404.07850) | [MindBridge](https://github.com/littlepure2333/MindBridge) |
| TIFA | [Hu et al., 2023](https://arxiv.org/abs/2303.11897) | [TIFA](https://github.com/Yushi-Hu/tifa) |

We also acknowledge [Hugging Face Transformers](https://github.com/huggingface/transformers) and [Diffusers](https://github.com/huggingface/diffusers), [Himalaya](https://github.com/gallantlab/himalaya), [OpenCLIP](https://github.com/mlfoundations/open_clip), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), and [ModelScope](https://github.com/modelscope/modelscope) for the model, regression, feature-extraction, and VQA components used in this repository.

Third-party source notices and licenses are preserved in their respective directories. Source provenance, local modifications, and weight-license boundaries are described in [THIRD_PARTY.md](THIRD_PARTY.md); no repository-wide license is assigned to copied third-party code. Data, weights, private configurations, experiment records, and generated results are not included in this repository.

Please also cite the original baseline methods and evaluation tools used in your experiments. Their BibTeX entries are provided in [REFERENCES.bib](REFERENCES.bib). MindEye's upstream authors additionally request citation of the [Natural Scenes Dataset](https://doi.org/10.1038/s41593-021-00962-x), on which their original methods were developed.

## Acknowledgements

**Funding:** [Funding agencies and grant numbers to be added.]

**Computing and institutional support:** [Computing platforms and institutional support to be added.]

For questions or reproducible bug reports, please [open an issue](https://github.com/lishurui0612/AVID-benchmark/issues) with the command, software versions, relevant configuration fields, and traceback. Remove private paths and credentials before posting.
