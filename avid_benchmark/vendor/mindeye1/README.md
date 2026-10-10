# MindEye1 runtime sources

This directory contains the AVID adaptation of [MindEye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD), by Scotti et al., *Reconstructing the Mind's Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors* (NeurIPS 2023; [paper](https://arxiv.org/abs/2305.18274)). The originating upstream commit of the locally adapted snapshot is unknown.

The retained sources provide the high-level training and reconstruction entry points, their model/data utilities, and the ConvNeXt teacher used by AVID low-level training. The public low-level trainer is `avid_benchmark.models.mindeye1.lowlevel_train`. Use the separate commands in the [AVID baseline guide](../../../docs/BASELINES.md#mindeye1) and the dependencies in the root [pyproject.toml](../../../pyproject.toml).

The upstream MIT [LICENSE](LICENSE) and source copyright notices are retained. Citation records for MindEye1, MindEye2, and the Natural Scenes Dataset are in [REFERENCES.bib](../../../REFERENCES.bib); cite the original methods and any datasets you use.

For the original NSD notebooks, retrieval tools, installation instructions, and acknowledgements, consult the [upstream README](https://github.com/MedARC-AI/fMRI-reconstruction-NSD#readme). This runtime subset does not include those workflows or pretrained weights.
