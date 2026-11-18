# CanonTalk training code

The static VISR implementation is in `train_static_claude/`. The dynamic
VISR-conditioned OmniAvatar implementation is in `train_dynamic_claude/`.
Run their scripts from the repository root. The dynamic stage imports the
vendored `vendor/omniavatar/OmniAvatar/` package and reads `model/config.json`.

The vendored OmniAvatar code is based on
`https://github.com/Omni-Avatar/OmniAvatar` at commit
`1536bf31abaec74364fb7d5883470d5b23ffa7f8` (Apache-2.0; see
`vendor/omniavatar/LICENSE.txt`).
`vendor/omniavatar/OmniAvatar/models/wan_video_dit.py`
contains local changes for batched audio conditioning and optional
camera/memory hooks. Keep this patched version for dynamic training.

Weights and datasets are intentionally not tracked:

- `model/pytorch_model.pt`: OmniAvatar audio LoRA weights, loaded with
  `model/config.json`.
- Wan2.1-1.3B and wav2vec2-base-960h weights: defaults in
  `train_dynamic_claude/omni.py`; pass `--wan`, `--wav2vec`, and `--omni`
  to `train_dynamic_claude.train` to use other locations.
- DINOv2, SegFormer, and LPIPS weights: dependencies of the static stage;
  see `train_static_claude/README.md` and `requirements.txt`.
- Ava-256, EmoTalk3D, CelebV-HQ, MultiDialog, the generated `data_cache/`,
  and training checkpoints in `*/runs/`.
- `train/manifests/{train,val}.jsonl`: local EmoTalk3D clip manifests used
  by the `prepare` scripts; supply equivalent manifests before preparing
  data. `train_dynamic_claude/prepare.py` also has a machine-specific
  `DATA` path to update for your dataset location.
- External InsightFace and SyncNet models and code for
  `train_dynamic_claude/evaluate.py` (see its `EVAL_MODELS` setting).

The run scripts default to the local `env/.venv`; set `PY` and `TORCHRUN`
for another environment. The OmniAvatar Python dependencies are listed in
`vendor/omniavatar/requirements.txt` in addition to the static-stage requirements.
