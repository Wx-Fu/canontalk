"""Does the generator follow the VISR condition? (held-out validation clips)

For each clip, generate under several conditions and measure:
  psnr / lpips     vs the GT target video
  pose_err_deg     head rotation of the *generated* video (measured with the frozen
                   VISR encoder) vs the head rotation of the GT target video
Conditions:
  full       reference + audio + VISR render (training setting)
  no_cond    VISR channels zeroed (generator alone: reference + audio)
  no_audio   audio zeroed
  yaw+X      VISR render with the head turned by an extra X deg; a generator that
             follows geometry shows ~X deg of extra rotation (follow_deg)
  static*    reference expression in the render (audio-driven deployment): lips can
             only come from audio. lip_corr: generated vs GT mouth-motion correlation;
             static vs static_shift (time-shifted audio) shows real sync, not just motion;
             audio_effect_*: mean mouth-region change caused by switching audio off

  python -m train_dynamic_claude.probe --checkpoint RUN/step_0030000.pt \
      --visr train_static_claude/runs/visr_v41_a/best.pt --index data_cache/dyn_index \
      --samples 24 --out RUN/probe
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import omni
from .data import TalkingWindows, collate
from .infer import rotation
from .utils import save_comparison
from .visr_cond import VISRCondition, mouth_mask


def mouth_stats(gen, gt, m):
    """Mouth-region PSNR and lip-motion agreement: Pearson correlation of per-frame
    mouth motion energy |x_t - x_{t-1}| between generated and GT (crude sync proxy)."""
    w = m[:, :, 1:]
    energy = lambda x: ((x[:, :, 1:] - x[:, :, :-1]).abs().mean(1, keepdim=True) * w).sum((1, 3, 4)) \
        / w.sum((1, 3, 4)).clamp_min(1)
    eg, et = energy(gen)[0], energy(gt)[0]
    eg, et = eg - eg.mean(), et - et.mean()
    corr = float((eg * et).sum() / (eg.norm() * et.norm()).clamp_min(1e-8))
    mse = (((gen - gt) ** 2).mean(1, keepdim=True) * m).sum() / m.sum().clamp_min(1)
    return float(-10 * torch.log10(mse.clamp_min(1e-8))), corr


def geodesic_deg(a, b):
    m = a.transpose(-1, -2) @ b
    c = ((m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


def yaw_deg(r):
    """Head yaw seen from the camera (head forward axis = 3rd column)."""
    f = r[..., :, 2]
    return torch.rad2deg(torch.atan2(f[..., 0], -f[..., 2]))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--visr", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--yaw-offset", type=float, default=25.)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--save-videos", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    pipe, speech, extractor = omni.load_models(dev)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["dit"]
    _, unexpected = pipe.dit.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    pipe.dit.eval()
    context, negative = omni.encode_prompt(pipe)
    visr = VISRCondition(a.visr, dev)
    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    data = TalkingWindows(Path(a.index) / "val.jsonl", "val", 33, a.size,
                          {"emotalk3d": 0.6, "celebv": 0.2, "multidialog": 0.2}, seed=11)
    rows = []
    for i in range(a.samples):
        batch = collate([data[i]])
        ref, video = batch["ref"][:, 0].to(dev), batch["video"].to(dev)
        k_ref, k_tgt = batch["K_ref"].to(dev), batch["K_tgt"].to(dev)
        st = visr.reference(ref, k_ref)
        r, t, e = visr.motion(video, k_tgt)
        audio = omni.audio_features(speech, extractor, batch["wave"], 33, dev)
        turn = rotation(a.yaw_offset, 0.).to(dev)
        e_ref = st["e"].float()[:, None].expand_as(e)
        shifted = torch.roll(audio, audio.shape[1] // 2, dims=1)   # same voice, wrong timing
        # (pose, audio, use VISR, expression): the "static*" rows are the audio-driven
        # deployment case: reference expression in the render, lips only from audio.
        conds = {"full": (r, audio, True, e), "no_cond": (r, audio, False, e),
                 "no_audio": (r, torch.zeros_like(audio), True, e),
                 f"yaw+{a.yaw_offset:g}": (r @ turn, audio, True, e),
                 "static": (r, audio, True, e_ref),
                 "static_shift": (r, shifted, True, e_ref),
                 "static_noaudio": (r, torch.zeros_like(audio), True, e_ref)}
        mouth = mouth_mask(r, t, k_tgt, a.size)
        row = {"dataset": batch["dataset"][0], "view": batch["view"][0]}
        outs = {}
        for name, (rr, au, use_cond, ee) in conds.items():
            render, alpha, depth = visr.render(st, rr, t, ee, k_tgt, a.size)
            y = omni.build_y(pipe, ref, render, alpha, depth)
            if not use_cond:
                y = torch.cat((y[:, :17], torch.zeros_like(y[:, 17:])), 1)
            lat = omni.sample(pipe, y, au, context, negative, steps=a.steps, seed=i)
            gen = pipe.decode_video(lat, tiled=False).float().add(1).div(2).clamp(0, 1)
            g_r, _, _ = visr.motion(gen, k_tgt)
            mse = ((gen - video) ** 2).mean().clamp_min(1e-8)
            frames = lambda x: x[0].permute(1, 0, 2, 3) * 2 - 1
            row[f"{name}/psnr"] = float(-10 * torch.log10(mse))
            row[f"{name}/lpips"] = float(lp(frames(gen), frames(video)).mean())
            row[f"{name}/pose_err_deg"] = float(geodesic_deg(g_r, r).mean())
            row[f"{name}/yaw_vs_gt_deg"] = float((yaw_deg(g_r) - yaw_deg(r)).mean())
            row[f"{name}/mouth_psnr"], row[f"{name}/lip_corr"] = mouth_stats(gen, video, mouth)
            outs[name] = (gen[0].cpu(), render[0].cpu())
        diff = lambda x, y: float(((outs[x][0] - outs[y][0]).abs().mean(0, keepdim=True)
                                   * mouth[0].cpu()).sum() / mouth[0].cpu().sum().clamp_min(1))
        row["audio_effect_static"] = diff("static", "static_noaudio")   # >0: audio moves the mouth
        row["audio_effect_full"] = diff("full", "no_audio")
        row["follow_deg"] = row[f"yaw+{a.yaw_offset:g}/yaw_vs_gt_deg"] - row["full/yaw_vs_gt_deg"]
        row["ref_vs_tgt_deg"] = float(geodesic_deg(st["R"].float()[:, None], r).mean())
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
        if i < a.save_videos:
            panels = [batch["ref"][0, 0], outs["full"][1], outs["full"][0], outs["no_cond"][0],
                      outs[f"yaw+{a.yaw_offset:g}"][0], batch["video"][0]]
            save_comparison(a.out / f"probe_{i:02d}_{row['dataset']}.mp4", panels, batch["wave"][0])
    summary = {}
    for d in sorted({r["dataset"] for r in rows}):
        sel = [r for r in rows if r["dataset"] == d]
        keys = [k for k in sel[0] if k not in ("dataset", "view")]
        summary[d] = {k: round(float(np.mean([r[k] for r in sel])), 3) for k in keys}
        summary[d]["n"] = len(sel)
    summary["video_panels"] = "ref | VISR render | full | no_cond | yaw+offset | GT"
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
