"""Inference: reference image + audio (+ camera path, + driving video) -> talking video.

  python -m train_dynamic_claude.infer --checkpoint RUN/step_0030000.pt \
      --visr train_static_claude/runs/visr_v41_a/best.pt --image face.png --audio speech.wav \
      --yaw "0:0,50:35,100:-35" --pitch "0:0" --out out.mp4

Camera path keys are "frame:degrees" (linear interpolation; the head rotates about
its own centre, which is equivalent to orbiting the camera). Long audio is
generated in windows; each window starts from the last generated frame (clean
first latent), the continuation mode used during training (--p-prefix).
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from . import omni
from .data import FPS, assumed_k
from .utils import save_comparison
from .visr_cond import VISRCondition


def keyframes(spec, n):
    pts = sorted((int(a), float(b)) for a, b in (kv.split(":") for kv in spec.split(",")))
    xs, ys = zip(*pts)
    return np.interp(np.arange(n), xs, ys)


def rotation(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    ry = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = torch.tensor([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return ry @ rx


def load_image(path, size, device):
    """Parser-based head crop (same rule as prepare.py), resized to size."""
    import cv2
    from train_static_claude.losses import Masker
    from .prepare import head_crop
    im = cv2.imread(str(path))[..., ::-1]
    h, w = im.shape[:2]
    small = cv2.resize(np.ascontiguousarray(im), (384, 384))
    crop = head_crop(Masker(size=384).to(device), [small], w, h, device)
    if crop is not None:
        x0, y0, side = (int(round(v)) for v in crop)
        im = im[y0:y0 + side, x0:x0 + side]
    im = cv2.resize(np.ascontiguousarray(im), (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im.copy()).permute(2, 0, 1).float().div(255)[None].to(device)


def load_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // g, sr // g).astype(np.float32)
    return torch.from_numpy(wave)


@torch.no_grad()
def generate_long(pipe, speech, extractor, visr, ref_state, ref, rots, trans, e, k, wave,
                  context, negative, window=33, steps=25, cfg=4.5, seed=0, size=512, log=True):
    """Chunked generation of len(rots[0]) frames: each window of `window` frames
    starts from the last generated frame (clean prefix latent, as in training).
    rots [1,N,3,3], trans [1,N,3], e [1,N,E] per output frame; wave 16 kHz [S].
    Returns [1,3,N,H,W] in [0,1] (cpu)."""
    dev = pipe.device
    n = rots.shape[1]
    step = window - 1
    frames, prefix, start = [], None, 0
    while start < n - 1:
        end = min(start + window, n)
        t_win = end - start
        idx = list(range(start, end)) + [end - 1] * (window - t_win)   # pad the last window
        render, alpha, depth = visr.render(ref_state, rots[:, idx], trans[:, idx], e[:, idx], k, size)
        y = omni.build_y(pipe, ref, render, alpha, depth)
        w0, wn = int(start / FPS * 16000), int(window / FPS * 16000)
        w = torch.nn.functional.pad(wave[w0:w0 + wn], (0, max(0, wn - len(wave[w0:w0 + wn]))))
        audio = omni.audio_features(speech, extractor, w[None], window, dev)
        latent = omni.sample(pipe, y, audio, context, negative, steps=steps, cfg_audio=cfg,
                             seed=seed, prefix_latent=prefix)
        video = pipe.decode_video(latent, tiled=False).float().add(1).div(2).clamp(0, 1)
        frames.append(video[:, :, (1 if start else 0):t_win].cpu())
        prefix = pipe.encode_video(omni.to_vae(video[:, :, t_win - 1:t_win], dev), tiled=False).to(dev)
        start += step
        if log:
            print(f"frames {min(start, n)}/{n}", flush=True)
    return torch.cat(frames, 2)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--visr", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--driver", default=None, help="optional video: expressions per frame")
    p.add_argument("--yaw", default="0:0")
    p.add_argument("--pitch", default="0:0")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--window", type=int, default=33)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    pipe, speech, extractor = omni.load_models(dev)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["dit"]
    missing, unexpected = pipe.dit.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    pipe.dit.eval()
    context, negative = omni.encode_prompt(pipe)
    visr = VISRCondition(a.visr, dev)

    ref = load_image(a.image, a.size, dev)
    k = assumed_k(a.size)[None].to(dev)
    wave = load_audio(a.audio)[: int(a.max_seconds * 16000)]
    n = max(a.window, int(len(wave) / 16000 * FPS))
    ref_state = visr.reference(ref, k)
    r0, t0, e0 = ref_state["R"].float(), ref_state["t"].float(), ref_state["e"].float()
    yaw, pitch = keyframes(a.yaw, n), keyframes(a.pitch, n)
    rots = torch.stack([r0[0] @ rotation(y, q).to(dev) for y, q in zip(yaw, pitch)])[None]
    trans = t0[:, None].expand(1, n, 3)
    if a.driver:
        from .data import read_frames
        import cv2
        cap = cv2.VideoCapture(a.driver)
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idx = [min(int(round(i / FPS * fps)), total - 1) for i in range(n)]
        drive = read_frames(a.driver, idx, None, a.size)[None].to(dev)
        _, _, e = visr.motion(drive, k)
    else:
        e = e0[:, None].expand(1, n, -1)

    out = generate_long(pipe, speech, extractor, visr, ref_state, ref, rots, trans, e, k, wave,
                        context, negative, a.window, a.steps, a.cfg, a.seed, a.size)[0]
    save_comparison(a.out, [out], wave[: int(out.shape[1] / FPS * 16000)])
    print("saved", a.out)


if __name__ == "__main__":
    main()
