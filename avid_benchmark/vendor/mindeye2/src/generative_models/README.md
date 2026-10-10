# Stability AI generative-model runtime sources

This directory retains the subset of [Stability AI's generative-models project](https://github.com/Stability-AI/generative-models) used by the AVID MindEye2 adaptation. It came through the locally adapted MindEyeV2 snapshot; see the [MindEye2 source note](../../README.md) for provenance.

The bundled `sgm` implementations and [unCLIP configuration](configs/unclip6.yaml) support high-level reconstruction. The [SDXL base configuration](configs/inference/sd_xl_base.yaml) supports the full caption/refinement/blurry-blend route. Some temporal modules remain because the shared model implementations import them. Standalone diffusion training, toy datasets, demo applications, generic inference APIs, and video-generation configurations are excluded.

Use the root [installation instructions](../../../../../README.md#installation) and [MindEye2 guide](../../../../../docs/BASELINES.md#mindeye2). Pretrained weights are external and configured through local asset paths. The code [LICENSE-CODE](LICENSE-CODE), model-license notices in `model_licenses/`, and source copyright notices are retained; source and pretrained-model licenses apply separately.

For the complete upstream workflows, model descriptions, and original documentation, consult the [upstream README](https://github.com/Stability-AI/generative-models#readme). The SDXL paper citation is included in [REFERENCES.bib](../../../../../REFERENCES.bib).
