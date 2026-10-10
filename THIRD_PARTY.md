# Third-party source and pretrained assets

The `avid_benchmark/vendor` directory retains upstream model code and the authors' AVID adaptations. Preserved upstream LICENSE files apply to their respective directories. MindEye2 README records upstream commit `c184718`; its upstream MIT LICENSE is retained along with the generative-model source notices.

The LDM adapter implements the AVID adaptation of Takagi and Nishimoto's reconstruction method using Himalaya ridge regression and Diffusers. Original-method references and official repositories are linked in [README.md](README.md); bibliographic entries are in [REFERENCES.bib](REFERENCES.bib). Method attribution does not establish an exact source revision for a local adaptation.

The TIFA runtime is adapted from [Yushi-Hu/tifa](https://github.com/Yushi-Hu/tifa), commit `8e37050f44ab4ee681e4251f3c8a9d4cc86e901e`. Its Apache-2.0 license and [adaptation record](avid_benchmark/vendor/tifa_official_8e37050/UPSTREAM.md) are included. The fixed question banks and `qwen_v3` attribute annotations define the AVID scoring inputs; they are distinct from upstream pretrained model weights.

MindEye1's MIT LICENSE was restored from https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/LICENSE (Git blob ca8c64c7e99a07a6101a61de78b56b4928897869; retrieved 2026-10-09). This identifies the license text, not the unknown originating commit of the locally adapted snapshot.

MindBridge's adapted snapshot has no LICENSE, and the root file listing of https://github.com/littlepure2333/MindBridge also contained no LICENSE when inspected on 2026-10-09. Its upstream revision and redistribution permission remain unresolved. Do not treat public availability as a license grant. No blanket repository license is assigned to copied third-party code. Pretrained-weight licenses are independent of source-code licenses.

Refactor changes to vendored code: configure CLIP through existing asset paths, enforce offline pretrained loading where used, derive process count from launched workers, and propagate checkpoint-write failures using atomic saves. MindBridge training receives the requested dataset root. Historical LDM/MindBridge candidate-selection behavior is preserved at the user's request.

Unused MindEye1/MindBridge training variants, evaluation scripts and shell/config launchers are excluded. Retained entry points are called only by the public model adapters. Low-level checkpoint loading is strict. MixCo target probability is added when a permutation maps a sample to itself, avoiding replacement of its diagonal mass; MindEye2 also fixes participant/rank index offsets and gathers global contrastive candidates with gradients.

All pretrained weights and scientific data remain external. Public configurations are templates; machine-specific validation configs are kept outside the public package. No cluster credentials or rjob bootstrap scripts belong in this repository.

MindEye1 low-level training restores the existing adapted `convnext.py` teacher implementation from the authors' source snapshot. Its Meta copyright notice is retained; the ConvNeXt source license notice refers to its upstream repository (https://github.com/facebookresearch/ConvNeXt). Teacher feature weights must load completely; classification heads are unused. The low-level model/loss follows the AVID `train_autoencoder_1.py` adaptation, with finite full-stimulus epochs, frozen-target caching and mathematically equivalent chunked contrastive scoring to limit memory.
