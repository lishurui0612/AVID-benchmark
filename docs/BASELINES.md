# Baseline guide

The commands below assume that the [baseline dependencies](../README.md#installation), AVID data, and the assets required by the chosen method are ready. Asset links are listed in [Asset downloads](ASSETS.md). Edit `config.local.yaml` before running them. Paths shown here are placeholders.

## Training settings

| Model | Training participants | Meaning of `--subject` |
|---|---|---|
| LDM | Selected participant | Training and prediction participant |
| MindEye1 | Selected participant | Training and prediction participant |
| MindEye2 | Other participants for pretraining; selected participant for fine-tuning | Held-out pretraining participant and prediction participant |
| MindBridge | All configured participants jointly | Prediction participant |

The multi-subject recipes use participants **1, 2, 3, 4, 5, 6, 7, 8, 10** for complex stimuli and **1, 4, 5, 6, 7, 8, 10** for simple stimuli. A single-subject prediction target does not make the multi-subject training recipes single-subject runs.

## LDM

```bash
python train.py --model ldm --subject 1 --stimulus complex --config config.local.yaml
python predict.py --model ldm --subject 1 --stimulus complex --config config.local.yaml
```

Training fits ridge decoders with Himalaya; `--epochs` does not apply. Configure the Stable Diffusion v1.4 asset in `model_overrides.ldm.model.stable_diffusion_model_path`. The candidate-selection encoder uses OpenCLIP; its existing weight file can be set with `model_overrides.ldm.model.openclip_pretrained_path`.

## MindEye1

Train the high-level and low-level models independently:

```bash
python train.py --model mindeye1 --subject 1 --stimulus complex --config config.local.yaml
python train.py --model mindeye1 --component lowlevel --subject 1 --stimulus complex --config config.local.yaml
```

The low-level recipe uses frozen VAE and ConvNeXt teachers. Configure `assets.mindeye1_vae` and `assets.mindeye1_convnext`, along with the shared CLIP assets used by the high-level model.

Set `model_overrides.mindeye1.model.autoencoder_checkpoint_path` to the trained low-level checkpoint. Then load the high-level checkpoint for prediction:

```bash
python predict.py \
  --model mindeye1 --subject 1 --stimulus complex \
  --checkpoint /path/to/highlevel/best.pth \
  --config config.local.yaml
```

The img2img route also requires Versatile Diffusion weights via `vd_cache_dir`. The low-level checkpoint must match the subject, stimulus type, vertex selection, and input dimension of the data. Input vertices are not silently truncated or pooled to fit an incompatible checkpoint. To explicitly use high-level generation alone, set `img2img_strength: 1.0`.

## MindEye2

Prepare a configuration for the full route:

```bash
cp config.mindeye2_full.yaml config.local.mindeye2_full.yaml
```

Fill in the local paths for OpenCLIP bigG, VAE, unCLIP, GIT, the bigG-to-GIT converter, the CLIP text encoder, and SDXL. Train one leave-one-subject-out fold:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --model mindeye2 --subject 1 --stimulus complex \
  --config config.local.mindeye2_full.yaml
```

Set `--nproc_per_node` to the number of GPUs available. `per_subject_batch_size` is a global batch size per participant; choose a GPU count that divides it. The recipe performs source-subject pretraining and target-subject fine-tuning, including the high-level, prior, and blurry branches. The target participant's data are excluded from pretraining.

Run prediction separately with the target-subject fine-tuned checkpoint:

```bash
python predict.py \
  --model mindeye2 --subject 1 --stimulus complex \
  --checkpoint /path/to/target_subject_finetune.pth \
  --config config.local.mindeye2_full.yaml
```

The full route generates captions, refines unCLIP images with SDXL, and blends `0.8 * refined + 0.2 * blurry` images. Its manifest records `unclip_sdxl_refine_lowlevel_blend` and retains the component image paths. Set `caption_image_refinement: false` to use the high-level unCLIP route with blurry images saved separately.

MixCo subject/rank offsets and self-permutation probabilities are corrected in this implementation. Distributed contrastive candidates retain gradients across ranks.

## MindBridge

```bash
python train.py --model mindbridge --subject 1 --stimulus complex --config config.local.yaml
python predict.py --model mindbridge --subject 1 --stimulus complex --config config.local.yaml
```

The recipe jointly trains on the configured participants, then predicts the selected participant. Configure the shared CLIP assets, Versatile Diffusion weights, and a compatible low-level checkpoint under `model_overrides.mindbridge.model`. The high-level branch pools vertices; the low-level branch uses the original vertex input.

## Outputs and evaluation

Outputs are written under `paths.train_output_root/<model>/<run_name>/`; LDM uses the model directory `ldm_himalaya`. Each stage prints its result paths. Training saves a checkpoint summary in `logs/train_result.json`; prediction saves a JSONL manifest and reconstructed images. If `--checkpoint` is omitted, prediction resolves a checkpoint from the matching run's training summary. MindEye2 selects a target-subject fine-tuned checkpoint.

Use the same `--run-name` for matching stages, or provide an explicit checkpoint. A completed training run requires a new run name for a fresh training attempt. Prediction creates a new output directory when an existing prediction directory is occupied.

```bash
python evaluate.py \
  --predictions /path/to/predictions/manifest.jsonl \
  --config config.local.yaml \
  --metrics pixcorr ssim inception_2way clip_i_2way clip_t_2way \
  --output scores.json
```

For a small execution check, use `--epochs 1` with neural-network training or `--max-samples 8` with prediction. These settings do not provide a complete benchmark result.

## Reconstruction protocol

The AVID recipes preserve method-specific candidate selection: LDM selects among five candidates using the reference image, MindBridge among sixteen using the reference image, and MindEye1 uses the brain-predicted embedding. Report the candidate count, selection rule, and generation route when comparing methods. Scores from different selection rules should be identified as different protocols.
