# VISR-conditioned OmniAvatar (dynamic stage)

## v2 (current defaults)

Probe of v1 (`runs/vco_a`, 30k): the generator follows VISR geometry (output pose error
2.8°, follows 24 of a requested extra 25° yaw; without VISR the view is wrong, 48°),
but **audio is almost unused** (no-audio ≈ full), and the hazy render can leak into outputs.

| v2 change | flag (default) |
|---|---|
| reference expression in the render more often, so lips must come from audio | `--p-static 0.6` (was 0.3) |
| blur the projected mouth region of the render | `--p-mouth-blur 0.5` |
| corrupt the render (blur, down/up-sample, silhouette ±3 px, noise): coarse guidance only | `--p-degrade 0.5` |
| pick the reference candidate (of 4) with the largest head-pose gap to the target | `--p-pick-gap 0.7` |
| Ava-256 1-frame cross-camera pairs (~160 multi-view ids) every 6th step | `--ava-root/--ava-pack --ava-every 6 --ava-batch 16` |
| batch 4 per GPU (v1 used 15.8 GB at batch 1) | `--batch 4` |
| warm start from v1 step 30k | `--init` |

Goal: **one reference image (any view) + audio + target camera / head motion →
talking video at that view.** The view-independent representation (VISR, static
stage, frozen) decides *where* everything is; the pretrained video generator
(OmniAvatar = Wan2.1-1.3B + audio LoRA) decides *what it looks like* and how it moves
over time.

## 1. Why this split

Five static-stage versions showed: VISR gets pose (~2.4°), geometry, identity
separation and expression right for unseen people, but its renders stay at
~22 dB because ~190 training identities cannot teach photoreal detail. A generator
pretrained on large video data has that prior, but on its own it cannot change
the viewpoint (the yaw-modulated experiments in `train/` turned the face but lost
the person). Conditioning the generator on VISR renders gives it the
geometry it lacks, and it adds the realism VISR lacks.

## 2. Conditioning design

```
reference image ──► VISR encoder ──► z_id (slots, textures), Q_ref, e_ref
target frame i  ──► (training: VISR encoder on the GT frame) ──► Q_i, e_i
                    (inference: Q_i = requested camera∘head motion, e_i = driver/audio→e)
render_i = VISR decoder( compose(z_id, e_i), Q_i, K ) ── rgb, alpha, depth
                           │
    VAE(render video) 16ch ┤   alpha 1ch, relative depth 1ch (latent grid, 4-frame pooled)
                           ▼
DiT input = [ noisy target latent 16 | ref latent 16 | gen-mask 1 | render 16 | α 1 | depth 1 ]
            └────── original OmniAvatar 33 channels ─────┘ └──── new 18 channels ────┘
```

- **Channel concat, zero-initialised.** The patch embedding grows from 33 to 51
  input channels. The new weights start at 0, so at step 0 the model *is* OmniAvatar.
  This needs no change to Wan's blocks, is cheap, and is the most robust way to
  add a pixel-aligned control signal.
- **Reference ≠ first frame.** OmniAvatar normally pins latent frame 0 to the
  reference. Here the reference may come from another view or another time, so
  **every frame is generated**. The reference latent stays in `y` (identity and
  appearance), with the generate mask set to 1 everywhere.
- **Motion is in VISR space.** Per-frame `(Q_i, e_i)` drive the render, so head
  pose and expression come from the view-independent representation. In
  training they are read off the GT target frames by the frozen VISR encoder: a
  low-dimensional code, so no appearance leaks through. This is exactly the
  interface for the next stage (audio → `e_i`, head-motion model → `P_i`).
- **Audio stays.** OmniAvatar's audio path is kept and trained, and supplies
  fine lip detail and sync. With probability `p_static` the render uses the
  *reference* expression for every frame, so the generator must take mouth motion
  from audio. That way both driving modes (VISR motion or audio only) work at inference.
- **Classifier-free dropout:** audio 10 %, the whole VISR condition 10 % (zeros),
  which enables guidance on either.

## 3. Data roles

| dataset | what it teaches | sampling |
|---|---|---|
| EmoTalk3D (11 synced cams, 27 train ids) | **novel view**: reference from camera s, target video from camera t ≠ s | 40 % |
| CelebV-HQ (35 k in-the-wild clips) | identity diversity, realism, head motion as self-supervised view change; reference from another time of the clip | 30 % |
| MultiDialog (7 k conversations, studio) | clean speech/lip sync, long talking turns | 30 % |

All three go through the same pipeline: (reference image, target window, audio).
For monocular clips the target camera equals the reference camera and the view
change comes from head motion, which VISR measures per frame. Monocular crops use
an assumed focal length (`fx = 4.8·size`, EmoTalk3D-like) and are cropped so the
head occupies the same fraction as in EmoTalk3D, so VISR sees its training framing.

Splits: EmoTalk3D 0046–0048 held out; CelebV-HQ by YouTube id, MultiDialog by
conversation (1 % held out each, stable hash).

## 4. Training

- **Frozen:** VAE, wav2vec, T5 (prompt encoded once), VISR (`runs/visr_v41_a/best.pt`).
- **Trained:** OmniAvatar LoRA (continued), patch embedding (incl. the new 18
  channels), audio projections.
- **Loss:** flow matching (`noise − target`) on all latent frames.
- **Default size:** 512², 33 frames (1.32 s at 25 fps), batch 1/GPU, gradient
  checkpointing.

## 5. Inference

`infer.py`: reference image + audio + camera path (yaw/pitch per frame around the
head) [+ optional driving video for `e_i`] → video. Sampling uses the Wan flow
scheduler with CFG on audio. The VISR render can also be used as an SDEdit-style
start (`--init-strength`), which is the basis for few-step distillation and real
time later.

## 6. Run

See `run.sh` (prepare → smoke → 8-GPU training → validation renders → inference).

## Evaluation (`evaluate.py`)

Tasks: `emo_novel` (EmoTalk3D val, reference and target from different cameras),
`emo_same`, `celebv`, `multidialog` (val splits); 50 clips each by default, 3 s per clip,
audio-driven (reference expression, lips from audio) with GT head pose.

| metric | meaning |
|---|---|
| PSNR / SSIM / LPIPS | vs GT video |
| CSIM_ref / CSIM_gt | ArcFace (InsightFace buffalo_l) identity vs reference image / vs GT frame |
| LSE-C ↑ / LSE-D ↓ | SyncNet lip sync (Wav2Lip protocol); `*_gt` columns = real-video ceiling |
| av_offset | SyncNet audio-video offset in frames (0 = in sync) |
| FID | torchvision Inception-v3 features, generated vs GT frames |
| FVD-R3D | Fréchet distance of R3D-18 clip features (proxy; no I3D weights available) |
| pose_err | VISR-measured head rotation error of the generated video (deg) |

Extra packages (once): `uv pip install --python env/.venv/bin/python insightface onnxruntime-gpu python_speech_features`
