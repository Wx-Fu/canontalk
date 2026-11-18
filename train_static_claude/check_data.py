"""Visual data check before training: crops, GT-vertex projections and masks.

Writes <out>/sample_XX.png with rows = [encoder views | t1 targets | t2 targets];
Ava views show projected registered vertices (green) when the GT pose is
available, and every target shows foreground (red) / head-weight (blue) masks.

  python -m train_static_claude.check_data --ava-root data/ava256_rgb_4TB \
      --ava-pack data_cache/ava_packs --emo-index data_cache/emotalk3d_frames/index.jsonl \
      --out train_static_claude/runs/data_check --samples 8
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import losses as L
from .dataset import MultiViewHeads
from .geometry import project, transform


def overlay_vertices(img, verts, q_r, q_t, k):
    x = transform(q_r[None], q_t[None], verts[None])
    uv, z = project(x, k[None])
    img = img.clone()
    h, w = img.shape[-2:]
    for u, v in uv[0, ::9].long().tolist():
        if 0 <= u < w and 0 <= v < h:
            img[:, v, u] = torch.tensor([0., 1., 0.])
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ava-root")
    p.add_argument("--ava-pack")
    p.add_argument("--emo-index")
    p.add_argument("--split", default="train")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--mask-size", type=int, default=512)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    data = MultiViewHeads(args.ava_root, args.emo_index, args.split, 0.5, 448, 256,
                          length=args.samples, seed=5, ava_pack=args.ava_pack)
    masker = L.Masker(size=args.mask_size).to(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(args.samples):
        x = data[i]
        rows = []
        for key, vi in (("enc", 0), ("tgt", 0), ("tgt2", 1)):
            v = x[key]
            imgs = v["image"]
            if key == "enc":
                imgs = F.interpolate(imgs, size=(256, 256), mode="bilinear", align_corners=False)
                scale = 256 / v["image"].shape[-1]
                ks = v["K"].clone()
                ks[:, :2] *= scale
            else:
                ks = v["K"]
            tiles = []
            fg, head = masker(imgs.to(args.device))
            for j in range(imgs.shape[0]):
                img = imgs[j]
                if x["has_pose"] > 0:
                    img = overlay_vertices(img, x["verts"][vi], v["Q_R"][j], v["Q_t"][j], ks[j])
                tint = img.clone()
                tint[0] = torch.maximum(tint[0], 0.6 * fg[j, 0].cpu() * (1 - head[j, 0].cpu()))
                tint[2] = torch.maximum(tint[2], 0.6 * head[j, 0].cpu())
                tiles.append(torch.cat((img, 0.5 * img + 0.5 * tint), -2))
            rows.append(torch.cat(tiles, -1))
        width = max(r.shape[-1] for r in rows)
        rows = [F.pad(r, (0, width - r.shape[-1]), value=1.) for r in rows]
        grid = torch.cat(rows, -2).clamp(0, 1).permute(1, 2, 0).numpy()
        Image.fromarray((grid * 255).astype(np.uint8)).save(args.out / f"sample_{i:02d}.png")
        print(f"sample {i}: {x['person']} angles {x['angles'].round().tolist()}")


if __name__ == "__main__":
    main()
