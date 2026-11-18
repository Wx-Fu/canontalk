# VISR — View-Invariant Slot Representation (static stage)

## v4.1 (current defaults) — clean copy path

v4 diagnosis (step 15k): slots moved to a median 10 mm from the surface (was 38 mm) and
the copy path is used, but copy-only renders were blocky (3 mm texels) with radial
streaks where rays hit no patch (background, hair edges), which leaked into renders.

| change | flag |
|---|---|
| copied colour only where the pixel ray hits a slot patch in front of the camera (geometric hit); elsewhere the free prediction | `--no-copy-coverage` to disable |
| 8×8 texels per slot (≈1.6 mm, was 4×4 ≈3 mm); warm start upsamples the texture-dynamics weights | `--tex-size 8` |

Warm start: `runs/visr_v4_a/step_0015000.pt`, 30k steps, output `runs/visr_v41_a`.

## v4 (current defaults) — slots must be a surface

v3 diagnosis (`best.pt`, step 12k): no gain over v2; the copy path alone rendered a
flat colour (the learned gate had closed) and forcing it open gave radial streaks.
Cause: slots were a median **37–40 mm** from the GT Ava surface (≈5 % within 5 mm,
half inside the head), so copied colours reprojected to the wrong place.

| v4 change | flag (default) |
|---|---|
| surface loss: each source slot → nearest of (model's own depth surface from 4 t1 views, detached) ∪ (Ava GT registration); slots farther than 6 cm from any evidence ignored | `--w-surf 1 --surf-gate 0.6 --surf-points 2048` |
| hard copy gate = in image × z-buffer visible × facing camera (cannot be learned closed) | `--no-copy-hard` to disable |
| decoder colour correction bounded to ±1.5 logit around the copied colour | `--corr-bound 1.5` |
| stricter visibility tolerance 12 mm | `--vis-tol 0.12` |
| v3 NaN fix: pose pooling in float32 with pre-norm + guard | – |

Logs: `surf` (dm), `surf_5mm` (fraction of slots within 5 mm). Diagnose adds
`slot_surface_mm_median` / `slot_within_5mm` (Ava). Gate for v4: `self_copygate1` > 25 dB.

## v3 (current defaults) — from the v2 diagnosis

`diagnose.py` on v2 `best.pt` (step 26k): re-rendering the **source view itself**
reached only 23.5 dB (Ava) / 17.0 dB (Emo), and was as blurry as novel views; the GT
source pose gave +2.1 dB; fusing 4–5 views gave nothing. So detail was lost when
*writing* new people into the slots, plus ~2 dB from the 20 mm pose error.

| v3 change | how | flag |
|---|---|---|
| colour-copy path | each texel also stores a colour = gate·(observed RGB, anti-aliased) + (1−gate)·(learned colour); the decoder outputs sigmoid(logit(copied colour) + correction), the correction starting at 0. Observed appearance is carried by geometry, not decoded per identity | on by default; `--no-color-copy` |
| pose refinement | after each lift stage, 4 learned queries attention-pool the slots lifted with the current pose and predict a bounded update (≤0.1 rad, ≤5 % image shift, ≤5 % depth); every iterate is supervised with the Ava GT pose; the refined pose also places the texels | `--pose-iters 2` |
| warm start | `--init v2/best.pt`: 604/624 tensors copied, 3 widened layers get the v2 weights in the matching slices, only the pose-refinement layers are new (zero-init). At step 0 v3 = v2 slots/pose/textures + a clean copy path | `--init` |

The decoder still never sees the source image: copied colours are stored inside the
canonical slots, so the representation stays view-independent.

## v2 (current defaults) — what changed after the v1 run and why

v1 (`runs/visr_a`, 2048 slots, no textures): validation stopped improving at
~4k steps (Ava frontal 21.7 dB, Emo 15.6 dB) while the training error kept
falling (val/train rgb ≈ 3×). In the panels, pose and silhouette were right
but faces were blurred, expressions were weak, and new Emo identities drifted
towards training people.

| v1 problem | v2 change | flag (default) |
|---|---|---|
| blur: one vector per slot, ~1.3 cm apart | **texture slots**: each slot also carries a 4×4×16 feature patch on its tangent plane (~3 mm texels), written from full-res image features; each pixel ray reads the patches of its 8 nearest slots where it hits them (frame-invariant, test enforced). Slots 2048 → 4096 | `--tex-size 4 --tex-dim 16 --tex-k 8 --slots 4096` |
| weak expression | dynamics code 64 → 128; `e` also drives texture residuals; extra L1 weight on projected mouth/eyes (canonical landmarks, fades for back views) | `--dyn-dim 128 --w-detail 2` |
| overfitting / identity drift | per-sample colour/exposure augmentation, consistent across all views of one sample; Ava packs 128 → 256 frames per person; best checkpoint kept by held-out `rgb` (`best.pt`) | `--no-color-aug` to disable |

The decoder still never sees the source image: the textures are written into
the slots by the encoder, so all detail passes through the view-invariant
representation. v1 checkpoints still load (missing v2 fields default to v1).

One unified feed-forward model (no per-person fitting, no UV atlas, no
Gaussians) that **writes** a single posed image into a learned,
perspective-independent space, and **reads** that space out from any camera.
The reasoning is in `train_stage_static/VIEW_INVARIANT_LATENT_THINKING.md`
(P1–P4 below refer to it). This directory is the implementation.

```
            write (encoder)                        read (decoder)
 I_s, K_s ─► DINOv2 ─► Q̂_s=(R,t) ─► N canonical slots ─►  rays of target camera in the
                       (moving frame)   {p_j, h_j, w_j}, e    canonical frame, ray↔slot
                                        identity ⟂ dynamics    geometric attention ─► RGB, α, depth
```

## 1. The representation

| symbol | meaning | invariance |
|---|---|---|
| `p_j = p0_j + Δp_id_j + Δp_dyn_j(e)` | 3D position of slot *j* in the head-canonical frame (dm) | camera- and pose-invariant |
| `h_j = h_id_j + h_dyn_j(e)` | 256-d slot feature (appearance/identity) | camera- and pose-invariant |
| `w_j` | precision: how well *this* view observed slot *j* | view-dependent **by design** (P1) |
| `e` | 64-d dynamics code (expression, lips) | view-invariant, time-varying |
| `Q = (R, t)` | head→camera pose | *not* part of z |

- **Slots are shared by every person.** There are `N=2048` slots, initialized
  inside a head-sized ellipsoid, and their layout `p0` is learned. Slot *j* is
  the same anatomical "address" for everyone, which gives correspondence
  across identities, views and time without a template mesh (P3, R3).
- **Writing** happens through the learned moving frame (Cartan, Kaba et al.
  2023). The pose head predicts `Q̂_s`. Each slot is posed into the source
  camera, projected, and gathers pixel-aligned ViT features, a raw 4×4 RGB
  patch and global context. Two lift/refine stages move the slots onto the
  subject. All outputs are expressed in the canonical frame, and the source
  camera is never passed on.
- **Reading** uses only *relative ray–slot geometry*: perpendicular distance and
  depth along the ray, through a fixed multi-scale bank of attention biases.
  The attention is **invariant to any global rigid motion of the frame**
  (`tests/test_geometry.py::test_decoder_bias_invariant_to_global_rigid_motion`),
  so changing viewpoint is exact geometry rather than something learned per
  camera pair (P3). The decoder's signature has no source image or source
  camera (test enforced), which closes the "image code + learned warp"
  shortcut (P2).
- **Multi-view fusion** is a precision-weighted product of experts in slot
  space (P1): one code path for 1 or *k* views.
- **Stochasticity lives in the invariant space** (P4). Stage B learns
  `p(z_id^multi | z_id^single)` with a rectified-flow transformer over the slot
  set. One sample is drawn per identity and rendered deterministically at every
  view and frame, so the completed hidden side is the same in all views.
- **Dynamics:** `e` → per-slot residuals through a shared MLP. Stage 2 (audio)
  only has to predict `e_t` (and head pose `P_t`). Because `e` is
  view-invariant, one audio sequence maps to one code, which can be displayed
  from any camera.

## 2. Supervision — each term enforces one property

| term | what it enforces | data |
|---|---|---|
| `rgb` (head-weighted L1), `perc` (LPIPS-VGG), `alpha` (BCE vs. parser mask) | write/read carries the information | all |
| `depth` (decoder depth vs. GT registration at z-buffer-visible vertices) | read-out geometry is metric and correct; removes the σ_z blur that photometric losses can't remove (aperture problem) | Ava |
| `pose_abs` (geodesic + L1 to GT `E_c·H_t`) | the moving frame **is** the head frame (fixes the gauge) | Ava |
| `pose_rel` (`R_b R_aᵀ = R_rel`, `t_b = R_rel t_a + t_rel`) | two synchronized views agree where the head is in the calibrated rig | Ava + Emo |
| `exchange` (slots from view s vs. view a, weighted by detached co-observation) | invariance: *which camera wrote the slot doesn't matter* (P2) | synchronized pairs |
| `dyn_inv` | `e` is view-invariant | synchronized pairs |
| cross-reenactment render (identity from t1 + `e` from t2) | expression can't hide in the identity slots | same person, 2 times |
| fused render (fuse(s, a) → targets) | trains the precisions `w` | synchronized pairs |
| GT-pose teacher forcing (Ava, decays 1→0 over 20k steps) | renders are aligned early; at the end, training equals deployment | Ava |

Render plan per sample: 10 decodes at 256² (`train.py::Forward`). Target
poses are always `Q_t = (E_t E_s⁻¹) ∘ Q̂_s`, the same formula used at
inference, so pose errors are trained through the renders too.

Background handling:
- Encoder inputs are composited on neutral gray using a frozen human parser
  (SegFormer-B2, ATR labels). This removes the studio-light pattern, which
  identifies the camera on Ava, and the black/white domain gap.
- Decoder targets are compared only on the foreground. Foreground and
  head-weight masks come from the same parser and were checked visually on
  both datasets with `check_data.py`.

## 3. Dataset roles

| | Ava-256 (`data/ava256_rgb_4TB`) | EmoTalk3D (`data_ori/emotalk3d/...`) |
|---|---|---|
| identities | 181 usable → hash split 161 train / 10 val / 10 test (`data_ava.split_captures`) | 27 train / 3 val (0046–0048, historical) |
| views | ~50 cams, 0–72° incl. pitch, head moves → continuous relative poses | 11 yaw-only cams |
| used for | **all** terms: render, depth, absolute + relative pose, exchange, dynamics, reenactment | render, relative pose, exchange, dynamics, reenactment (no 3D labels) |
| role in the theory | teaches the moving frame, equivariance of the write, metric geometry | target domain + speech-related dynamics; can't teach the frame alone |
| sampling | `--ava-prob 0.7`; capture → frame t1, frame t2 from another segment | person → clip t1, another clip t2 |
| crop rule | metric: square of 360 mm at the head center's depth, ±8% jitter | dataset 512² head crop, 88–100% random sub-crop |

Views per sample: source `s`; auxiliary `a` (15–60° from s); targets stratified
as [0,25), [25,50), [50,180) deg; a driver `d` and two targets at t2.

## 4. 8-GPU recipe (from `tmm/`)

Install `requirements.txt` into the training env. Pre-download weights once:
`facebook/dinov2-with-registers-base`, `mattmdjaga/segformer_b2_clothes` (NVIDIA
SegFormer license, research use) and the LPIPS-VGG weights.

```bash
# 0) pre-fetch weights (needs hub access; or set HF_HOME to a shared cache)
env/.venv/bin/python -c "from transformers import AutoModel, SegformerForSemanticSegmentation as S; \
AutoModel.from_pretrained('facebook/dinov2-with-registers-base'); \
S.from_pretrained('mattmdjaga/segformer_b2_clothes'); import lpips; lpips.LPIPS(net='vgg')"

# 1) data caches (CPU nodes are fine; both are resumable)
bash train_static_claude/run_8gpu.sh prep_emo   # EmoTalk3D -> JPEG frames (6 synced frames/clip)
bash train_static_claude/run_8gpu.sh prep_ava   # Ava -> one pack per frame (all cams+pose+mesh)
bash train_static_claude/run_8gpu.sh check      # inspect runs/data_check/*.png

# 2) one-GPU smoke: watch mem_gb, sec_per_step, data_sec, finite losses
bash train_static_claude/run_8gpu.sh smoke

# 3) stage A on 8 GPUs (resume: RESUME=runs/visr_a/step_XXXXXXX.pt bash ... stage_a)
bash train_static_claude/run_8gpu.sh stage_a
bash train_static_claude/run_8gpu.sh eval_a

# 4) stage B (completion prior in slot space) on the frozen stage-A model
CK=train_static_claude/runs/visr_a/step_0150000.pt bash train_static_claude/run_8gpu.sh stage_b
CK=... PRIOR=train_static_claude/runs/visr_b/prior_0060000.pt bash train_static_claude/run_8gpu.sh eval_b

# 5) any image -> yaw/pitch orbit (optionally --driver for expression)
CK=... IMG=face.png OUT=orbit.png bash train_static_claude/run_8gpu.sh infer
```

The stage A command expands to:

```bash
env/.venv/bin/torchrun --standalone --nproc_per_node=8 -m train_static_claude.train \
  --ava-root data/ava256_rgb_4TB --ava-pack data_cache/ava_packs \
  --emo-index data_cache/emotalk3d_frames/index.jsonl \
  --batch 4 --workers 12 --steps 150000 --lr 3e-4 --lr-vit 3e-5 \
  --gt-pose-warmup 20000 --val-every 2000 --save-every 5000 \
  --out train_static_claude/runs/visr_a
```

- Global batch: 32 samples, i.e. 96 encoder images and 320 decoder renders per
  step.
- Model: 146 M parameters, 116 M trainable (DINOv2 blocks 5–12 at
  `--lr-vit`, new encoder 25 M, decoder 34 M).
- **If memory is short:** use `--grad-ckpt` first, then `--batch 2`, then
  `--slots 1024`.
- **Logs:** `log.jsonl`, plus `val_XXXXXXX.png` panels (source | per target:
  GT, render, α).
- **Validation reports:** foreground PSNR / LPIPS per domain and per angle
  bin, and Ava pose error in degrees and mm.

### What to read in `evaluate.py` output

- `*/psnr_fg/50-75`, `*/lpips/50-75`: large-angle quality, the main number
  (F→F-like bins are reported separately).
- `ava/pose_rot_deg`, `ava/pose_trans_mm`: moving-frame accuracy. Geometry
  limits transport: 1 cm of error is ≈15 px at 512 for a 70° view change.
- `*/exchange_gap` vs `*/error_to_gt`: invariance. The gap should be well
  below the error.
- `probe/*/slots` vs `probe/*/vit_cls` vs `majority`: camera leakage. The ViT
  token should be decodable; the slots should be close to `majority`.
- `prior/sample_std_fg`: diversity of the completed regions. Views of one
  sample are consistent by construction.

## 5. What was verified here, and what was not

Verified on this CPU node, with real data, by these commands and tests:
- Ava camera/pose/crop conventions: GT registered vertices project onto the
  face in all views and at both time instants (`check_data.py`).
- Pixel-exact crop intrinsics, 6D rotations, rig relative poses, z-buffer
  visibility, and rigid-motion invariance of the decoder attention
  (`tests/test_geometry.py`).
- Every trainable parameter receives a gradient (DDP with
  `find_unused_parameters=False`), and the decoder cannot receive the source
  image (`tests/test_model.py`).
- End-to-end train → stage-B prior → evaluate with a tiny ViT on real Ava packs
  and EmoTalk3D frames, with bf16 autocast forced on (CUDA dtype path). Camera
  geometry stays in float32 (`geometry.exact`); bf16 is too coarse for it.
- A full-size forward pass with pretrained DINOv2 weights: 10 renders at
  256², initial pose = frontal, crop-prior depth.
- Mask quality on both datasets.
- Data speed: packed Ava sample 0.1–0.25 s, versus ~5 s from the release ZIPs.

**Not** verified: no GPU on this node.
- Throughput and memory on the target GPUs (run `smoke` first).
- NCCL/DDP at 8 ranks.
- Any training result. Nothing here claims image quality yet.

## 6. Known limitations / next knobs

- Ava has no supervision beyond ~72° (rear cameras weren't downloaded); hidden
  regions there are left to stage B.
- Emo has no 3D labels; its frame is anchored only through the shared pose head
  and the rig-relative constraint. `EmoTalk3D_mesh` would add pose/depth
  terms there directly.
- Lens distortion is ignored (same assumption as the verified overlays).
- Per-camera color differences are not modeled. A per-target affine color fit
  inside the loss would be the next refinement if `rgb` plateaus on Ava.
- The decoder is deterministic (sharpness bounded by LPIPS). The intended
  final pipeline renders VISR views and lets the video generator
  (Wan/OmniAvatar) refine them, conditioned on renders and slots, in the
  dynamic stage.

## Files

`geometry.py` SE(3)/pinhole/rays · `data_ava.py` `data_emo.py` `data_common.py` `dataset.py` data ·
`prepare_ava.py` `prepare_emo.py` caches · `model.py` encoder/fusion/decoder · `losses.py` ·
`train.py` stage A · `prior.py` `train_prior.py` stage B · `evaluate.py` · `infer.py` ·
`check_data.py` · `run_8gpu.sh` · `tests/`
