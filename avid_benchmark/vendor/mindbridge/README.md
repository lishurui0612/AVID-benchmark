# MindBridge runtime sources

This directory contains the AVID adaptation of [MindBridge](https://github.com/littlepure2333/MindBridge), by Shizun Wang, Songhua Liu, Zhenxiong Tan, and Xinchao Wang, *MindBridge: A Cross-Subject Brain Decoding Framework* (CVPR 2024; [paper](https://arxiv.org/abs/2404.07850)). The originating upstream commit of the locally adapted snapshot is unknown.

The retained training and reconstruction entry points and their dependencies are called by the AVID adapter. Use the separate commands in the [AVID baseline guide](../../../docs/BASELINES.md#mindbridge) and the root [pyproject.toml](../../../pyproject.toml) for installation. Upstream NSD launch scripts and their conflicting requirements file are outside this runtime subset.

MindBridge's redistribution license remains unresolved. Public availability does not establish a license grant.

The original project acknowledges MindEye, nsd_access, the Natural Scenes Dataset, Accelerate, DeepSpeed, Xingyi Yang, and Yifan Zhang. Consult the [upstream README](https://github.com/littlepure2333/MindBridge#readme) for its complete acknowledgements, original installation guide, and NSD workflows.
