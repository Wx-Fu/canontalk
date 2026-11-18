"""Locate the quality bottleneck of a trained stage-A model (no training).

For held-out samples, renders the same targets under controlled conditions:

  self        slots from source s -> source camera itself     (write bottleneck)
  novel       slots from s        -> other cameras T1..T3      (deployment setting)
  novel_gtpose  same, GT source head pose (Ava only)           (pose contribution)
  novel_notex   same, learned texel features zeroed            (are textures used?)
  self_copyonly   v3: source view from the transported colour alone (no correction)
  self_copygate1  v3: same with the observation gate forced open
  novel_copygate1 v3: novel views, gate forced open (is copy usable off-view?)
  fused       fuse(s, a, extra views) -> T1..T3                (more observation;
              extras may include a target camera: an upper bound, not a test)

Per condition: foreground PSNR and a sharpness ratio = mean |grad pred| /
mean |grad GT| on the foreground (1 = as sharp as the photo, <1 = blurred).

  python -m train_static_claude.diagnose --checkpoint RUN/best.pt \
      --ava-root data/ava256_rgb_4TB --ava-pack data_cache/ava_packs \
      --emo-index data_cache/emotalk3d_frames/index.jsonl --samples 200 --out RUN/diagnose
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import losses as L
from .dataset import MultiViewHeads
from .model import fuse
from .dataset import collate
from .train import build_from_checkpoint, take, target_pose, to_device


def sharpness(img, mask):
    gx = (img[..., :, 1:] - img[..., :, :-1]).abs().mean(1, keepdim=True)[..., :-1, :]
    gy = (img[..., 1:, :] - img[..., :-1, :]).abs().mean(1, keepdim=True)[..., :, :-1]
    m = mask[..., :-1, :-1]
    return ((gx + gy) * m).sum((1, 2, 3)) / (m.sum((1, 2, 3)) + 1)


def psnr(pred, gt, m):
    mse = (((pred - gt) ** 2) * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)) + 1)
    return -10 * torch.log10(mse.clamp_min(1e-8))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ava-root")
    p.add_argument("--ava-pack")
    p.add_argument("--emo-index")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--fusion-views", type=int, default=4)
    p.add_argument("--mask-size", type=int, default=512)
    p.add_argument("--panels", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    net, saved = build_from_checkpoint(args.checkpoint, device)
    data = MultiViewHeads(args.ava_root, args.emo_index, "val", 0.5, saved.enc_size,
                          saved.render_size, length=args.samples, seed=99,
                          ava_pack=args.ava_pack, fusion_views=args.fusion_views)
    masker = L.Masker(size=args.mask_size).to(device)
    enc_net = net.encoder
    has_tex = bool(getattr(enc_net, "tex_size", 0))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for i in range(args.samples):
        batch = to_device(collate([data[i]]), device)
        enc, tgt = batch["enc"], batch["tgt"]
        if not getattr(saved, "keep_source_background", False):
            enc["image"] = L.neutral_background(masker, enc["image"][0])[None]
        v = enc["image"].shape[1]
        with torch.autocast(**amp):
            views = [enc_net(enc["image"][:, j], enc["K"][:, j]) for j in [0, 1] + list(range(3, v))]
            s = views[0]
            cam_s = take(enc, 0)
            gt_pose = None
            if batch["has_pose"][0] > 0:
                gt_pose = enc_net(enc["image"][:, 0], enc["K"][:, 0],
                                  (enc["Q_R"][:, 0], enc["Q_t"][:, 0]), torch.ones(1, device=device))

            def render(z, view, idx):
                outs = []
                for j in idx:
                    cam = take(tgt, j)
                    r, t = target_pose({"R_used": view["R_used"], "t_used": view["t_used"]},
                                       cam_s, cam)
                    outs.append(net.decoder(z, r, t, cam["K"])["rgb"].float())
                return torch.cat(outs)

            z_s = enc_net.compose(s, s["e"])
            cond = {"self": render(z_s, s, [4]), "novel": render(z_s, s, [1, 2, 3])}
            if getattr(net.decoder, "color_copy", False):
                # Is the transported colour usable, and does the model use it?
                net.decoder.copy_only = True
                cond["self_copyonly"] = render(z_s, s, [4])
                enc_net.force_gate = 1.0
                s1 = enc_net(enc["image"][:, 0], enc["K"][:, 0])
                cond["self_copygate1"] = render(enc_net.compose(s1, s1["e"]), s1, [4])
                cond["novel_copygate1"] = render(enc_net.compose(s1, s1["e"]), s1, [1, 2, 3])
                enc_net.force_gate = None
                net.decoder.copy_only = False
            if gt_pose is not None:
                cond["novel_gtpose"] = render(enc_net.compose(gt_pose, gt_pose["e"]), gt_pose,
                                              [1, 2, 3])
            if has_tex:
                z0 = dict(z_s)
                # zero the learned texel features only; copied colours (v3) stay
                z0["tex"] = z_s["tex"].clone()
                z0["tex"][..., :enc_net.tex_dim] = 0
                cond["novel_notex"] = render(z0, s, [1, 2, 3])
            f = fuse(views)
            cond["fused"] = render(enc_net.compose(f, f["e"]), s, [1, 2, 3])
        gts = {"self": tgt["image"][0, 4:5], "novel": tgt["image"][0, 1:4]}
        valids = {"self": tgt["valid"][0, 4:5], "novel": tgt["valid"][0, 1:4]}
        row = {"domain": "ava" if batch["domain"][0] == 0 else "emo"}
        if batch["has_pose"][0] > 0:  # are the slots on the (face/scalp) surface?
            slots = (enc_net.p0[None] + s["dp_id"]).float()
            dist = torch.cdist(slots, batch["verts"][:, 0].float()).min(-1).values * 100
            row["slot_surface_mm_median"] = dist.median().item()
            row["slot_within_5mm"] = (dist < 5).float().mean().item()
        for name, pred in cond.items():
            key = "self" if name.startswith("self") else "novel"
            gt, valid = gts[key], valids[key]
            fg, _ = masker(gt)
            m = fg * valid
            row[f"{name}/psnr"] = psnr(pred, gt, m).mean().item()
            row[f"{name}/sharp"] = (sharpness(pred, m) / sharpness(gt, m).clamp_min(1e-6)).mean().item()
        row["gt_sharp_abs"] = sharpness(gts["novel"], masker(gts["novel"])[0]).mean().item()
        rows.append(row)
        if i < args.panels:
            src = F.interpolate(enc["image"][0, :1], size=gts["novel"].shape[-2:], mode="bilinear",
                                align_corners=False)
            line = [torch.cat([src[0], gts["self"][0]] +
                              [cond[n][0] for n in cond if n.startswith("self")], -1)]
            for j in range(3):
                tiles = [gts["novel"][j]] + [cond[n][j] for n in cond if not n.startswith("self")]
                line.append(torch.cat(tiles, -1))
            width = max(x.shape[-1] for x in line)
            line = [F.pad(x, (0, width - x.shape[-1]), value=1.) for x in line]
            grid = torch.cat(line, -2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            Image.fromarray((grid * 255).astype(np.uint8)).save(args.out / f"panel_{i:02d}.png")
        print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()}),
              flush=True)
    summary = {}
    for dom in ("ava", "emo"):
        sel = [r for r in rows if r["domain"] == dom]
        keys = sorted({k for r in sel for k in r if k != "domain"})
        summary[dom] = {k: float(np.mean([r[k] for r in sel if k in r])) for k in keys}
        summary[dom]["n"] = len(sel)
    summary["panel_layout"] = ("row 1: source | GT source view | " +
                               " | ".join(n for n in cond if n.startswith("self")) +
                               "; rows 2-4: GT target | " +
                               " | ".join(n for n in cond if not n.startswith("self")))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
