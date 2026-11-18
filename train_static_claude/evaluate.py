"""Evaluation of a stage-A model (optionally with the stage-B prior).

Reports, on identity-disjoint data (Ava val/test captures, Emo 0046-0048):

  quality      single-source novel views, PSNR / L1 / LPIPS on the target
               foreground, per domain and per source-target angle bin
  pose         Ava: source head-pose rotation (deg) / translation (mm) error
  exchange     invariance test (P2): render the same target from slots written
               by view s and by view a; mean |difference| on the foreground.
               Reported next to the error to GT, as a ratio.
  probe        leakage test: logistic regression predicting the *source camera*
               (Ava: head-relative yaw bin; Emo: camera letter) from the pooled
               identity slots, vs. the same probe on the ViT CLS token and the
               majority-class rate. An invariant space should be near chance.
  prior        (with --prior) the same quality metrics for K samples, and the
               per-pixel std across samples (diversity of completed content)

  python -m train_static_claude.evaluate --checkpoint RUN/step_0150000.pt \
      --ava-root data/ava256_rgb_4TB --ava-pack data_cache/ava_packs \
      --emo-index data_cache/emotalk3d_frames/index.jsonl --split val --out RUN/eval_val.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import losses as L
from .dataset import MultiViewHeads, collate
from .geometry import geodesic
from .train import ANGLE_EDGES, build_from_checkpoint, take, target_pose, to_device


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prior", type=Path, default=None)
    p.add_argument("--prior-samples", type=int, default=2)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--ava-root", default=None)
    p.add_argument("--ava-pack", default=None)
    p.add_argument("--emo-index", default=None)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--perceptual", choices=("lpips", "none"), default="lpips")
    p.add_argument("--masker", choices=("segformer", "threshold"), default="segformer")
    p.add_argument("--mask-size", type=int, default=512)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def yaw_bin(r_q, width=20.):
    """Head-relative yaw of the camera from head->camera rotation (Ava)."""
    f = r_q[:, :, 2]  # head forward axis in camera coordinates
    yaw = torch.rad2deg(torch.atan2(f[:, 0], -f[:, 2]))
    return torch.floor((yaw + 90) / width).clamp(0, 180 / width - 1).long()


@torch.enable_grad()
def probe(features, labels, steps=300):
    """Even/odd split logistic regression; returns (accuracy, majority rate)."""
    if len(features) < 20:
        return None
    x = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(labels)
    classes = y.unique()
    if len(classes) < 2 or len(y) < 20:
        return None
    y = torch.searchsorted(classes, y)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    tr, te = torch.arange(len(y)) % 2 == 0, torch.arange(len(y)) % 2 == 1
    w = torch.zeros(x.shape[1], len(classes), requires_grad=True)
    b = torch.zeros(len(classes), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x[tr] @ w + b, y[tr]) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    acc = ((x[te] @ w + b).argmax(1) == y[te]).float().mean().item()
    majority = (y[te] == torch.mode(y[tr]).values).float().mean().item()
    return {"accuracy": acc, "majority": majority, "classes": int(len(classes)), "n": int(len(y))}


class Acc:
    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, key, values):
        values = values.detach().float().flatten().cpu()
        self.s[key] = self.s.get(key, 0.) + float(values.sum())
        self.n[key] = self.n.get(key, 0) + int(values.numel())

    def result(self):
        return {k: self.s[k] / max(1, self.n[k]) for k in sorted(self.s)}


def render(net, z, view, cam_enc, batch, idx):
    rs, ts, ks = [], [], []
    for j in idx:
        cam = take(batch["tgt"], j)
        r, t = target_pose(view, cam_enc, cam)
        rs.append(r); ts.append(t); ks.append(cam["K"])
    rep = lambda x: torch.cat([x] * len(idx), 0)
    return net.decoder({key: rep(v) for key, v in z.items()}, torch.cat(rs), torch.cat(ts),
                       torch.cat(ks))


@torch.no_grad()
def main(argv=None):
    args = parse(argv)
    device = torch.device(args.device)
    net, saved = build_from_checkpoint(args.checkpoint, device)
    prior = None
    if args.prior:
        from .prior import SlotPrior
        ck = torch.load(args.prior, map_location="cpu", weights_only=False)
        prior = SlotPrior(**ck["prior_kwargs"]).to(device).eval()
        prior.load_state_dict(ck["prior"])
    data = MultiViewHeads(args.ava_root, args.emo_index, args.split, 0.5, saved.enc_size,
                          saved.render_size, length=args.samples, seed=2027,
                          ava_pack=args.ava_pack)
    loader = DataLoader(data, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
    masker = (L.Masker(size=args.mask_size) if args.masker == "segformer" else L.ThresholdMasker()).to(device)
    perceptual = L.Perceptual(args.perceptual).to(device)
    acc = Acc()
    probe_feats = {"ava": ([], [], []), "emo": ([], [], [])}  # slots, vit-cls, label
    targets = [0, 1, 2, 3]  # [a, T1, T2, T3]
    for batch in loader:
        batch = to_device(batch, device)
        b = batch["has_pose"].shape[0]
        if not getattr(saved, "keep_source_background", False):
            e = batch["enc"]
            e["image"] = L.neutral_background(masker, e["image"].flatten(0, 1)).view_as(e["image"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            enc = batch["enc"]
            s = net.encoder(enc["image"][:, 0], enc["K"][:, 0])
            a = net.encoder(enc["image"][:, 1], enc["K"][:, 1])
            cam_s, cam_a = take(enc, 0), take(enc, 1)
            z_s, z_a = net.encoder.compose(s, s["e"]), net.encoder.compose(a, a["e"])
            pred = render(net, z_s, {"R_used": s["R"], "t_used": s["t"]}, cam_s, batch, targets)
            pred_a = render(net, z_a, {"R_used": a["R"], "t_used": a["t"]}, cam_a, batch, [1, 2, 3])
            _, _, cls = net.encoder.features(enc["image"][:, 0], enc["K"][:, 0])
        image = torch.cat([batch["tgt"]["image"][:, j] for j in targets])
        valid = torch.cat([batch["tgt"]["valid"][:, j] for j in targets])
        fg, wmap = masker(image)
        m = fg * valid
        terms = L.render_terms(pred, image, valid, fg, wmap, perceptual)
        mse = (((pred["rgb"] - image) ** 2) * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)) + 1)
        psnr = -10 * torch.log10(mse.clamp_min(1e-8))
        dom = batch["domain"].repeat(len(targets))
        ang = torch.cat([batch["angles"][:, j] for j in targets])
        for name, d in (("ava", 0), ("emo", 1)):
            for lo, hi in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:]):
                sel = (dom == d) & (ang >= lo) & (ang < hi)
                if sel.any():
                    acc.add(f"{name}/psnr_fg/{lo}-{hi}", psnr[sel])
                    acc.add(f"{name}/l1_fg/{lo}-{hi}", terms["rgb"][sel])
                    acc.add(f"{name}/lpips/{lo}-{hi}", terms["perc"][sel])
        # exchangeability: same targets T1..T3, slots written by s vs by a
        sl = slice(b, 4 * b)
        m3 = m[sl]
        gap = ((pred["rgb"][sl] - pred_a["rgb"]).abs() * m3).sum((1, 2, 3)) / (3 * m3.sum((1, 2, 3)) + 1)
        err = ((pred["rgb"][sl] - image[sl]).abs() * m3).sum((1, 2, 3)) / (3 * m3.sum((1, 2, 3)) + 1)
        for name, d in (("ava", 0), ("emo", 1)):
            sel = batch["domain"].repeat(3) == d
            if sel.any():
                acc.add(f"{name}/exchange_gap", gap[sel])
                acc.add(f"{name}/error_to_gt", err[sel])
        hp = batch["has_pose"] > 0
        if hp.any():
            acc.add("ava/pose_rot_deg", torch.rad2deg(geodesic(s["R"].float(), cam_s["Q_R"]))[hp])
            acc.add("ava/pose_trans_mm", ((s["t"].float() - cam_s["Q_t"]).norm(dim=-1) * 100)[hp])
        pooled = s["h_id"].float().mean(1).cpu().numpy()
        cls = cls.float().cpu().numpy()
        ybin = yaw_bin(cam_s["Q_R"]).cpu().numpy()
        for i in range(b):
            key = "ava" if batch["domain"][i] == 0 else "emo"
            label = int(ybin[i]) if key == "ava" else ord(batch["src_cam"][i][0])
            probe_feats[key][0].append(pooled[i])
            probe_feats[key][1].append(cls[i])
            probe_feats[key][2].append(label)
        if prior is not None:
            outs = []
            for k in range(args.prior_samples):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    z_id = prior.sample(s, net.encoder.slot_embed, guidance=args.guidance)
                    z = net.encoder.compose(z_id, s["e"])
                    outs.append(render(net, z, {"R_used": s["R"], "t_used": s["t"]}, cam_s,
                                       batch, targets)["rgb"].float())
            for k, rgb in enumerate(outs):
                mse = (((rgb - image) ** 2) * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)) + 1)
                ps = -10 * torch.log10(mse.clamp_min(1e-8))
                for name, d in (("ava", 0), ("emo", 1)):
                    for lo, hi in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:]):
                        sel = (dom == d) & (ang >= lo) & (ang < hi)
                        if sel.any():
                            acc.add(f"prior/{name}/psnr_fg/{lo}-{hi}", ps[sel])
            if len(outs) > 1:
                std = torch.stack(outs).std(0).mean(1, keepdim=True)
                acc.add("prior/sample_std_fg", (std * m).sum((1, 2, 3)) / (m.sum((1, 2, 3)) + 1))
    result = {"checkpoint": str(args.checkpoint), "split": args.split, "metrics": acc.result()}
    for key, (zf, cf, lab) in probe_feats.items():
        result[f"probe/{key}/slots"] = probe(zf, lab)
        result[f"probe/{key}/vit_cls"] = probe(cf, lab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
