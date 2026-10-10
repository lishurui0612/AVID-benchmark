# Vendored TIFA runtime

- Upstream: https://github.com/Yushi-Hu/tifa
- Commit: `8e37050f44ab4ee681e4251f3c8a9d4cc86e901e`
- License: Apache-2.0 (see `LICENSE`)
- Upstream package version: 1.1.0

Only the mPLUG and SBERT classes required by Simple-Attribute-VQA are included.
The implementations are derived from the upstream `tifascore/vqa_models.py`
and `tifascore/mc_sbert.py`. Imports for unrelated optional VQA backends were
removed so that using mPLUG does not require PromptCap, LAVIS, OFA, or Fairseq.

