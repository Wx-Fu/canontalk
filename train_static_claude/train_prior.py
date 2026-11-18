"""Stage B training: slot-space completion prior on a frozen stage-A model.

  torchrun --standalone --nproc_per_node=8 -m train_static_claude.train_prior \
      --stage-a train_static_claude/runs/visr_a/step_0150000.pt \
      --ava-root data/ava256_rgb_4TB --ava-pack data_cache/ava_packs \
      --emo-index data_cache/emotalk3d_frames/index.jsonl \
      --out train_static_claude/runs/visr_b
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from . import losses as L
from .dataset import MultiViewHeads, collate
from .model import fuse
from .prior import SlotPrior
from .train import build_from_checkpoint, setup, split_views, take, target_pose, to_device


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--ava-root", default=None)
    p.add_argument("--ava-pack", default=None)
    p.add_argument("--emo-index", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ava-prob", type=float, default=0.7)
    p.add_argument("--fusion-views", type=int, default=4)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--w-render", type=float, default=0.2,
                   help="weight of rendering the one-step estimate (keeps samples decodable)")
    p.add_argument("--perceptual", choices=("lpips", "none"), default="lpips")
    p.add_argument("--masker", choices=("segformer", "threshold"), default="segformer")
    p.add_argument("--mask-size", type=int, default=512)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


@torch.no_grad()
def encode_targets(net, batch):
    """Condition: single source view with *predicted* pose (deployment setting).
    Target: fusion of every t1 view; Ava uses GT pose here (privileged, target side only)."""
    enc = batch["enc"]
    b, v = enc["image"].shape[:2]
    flat = lambda x: x.flatten(0, 1)
    force = batch["has_pose"][:, None].expand(b, v).flatten()
    out = net.encoder(flat(enc["image"]), flat(enc["K"]), (flat(enc["Q_R"]), flat(enc["Q_t"])),
                      force)
    views = split_views(out, b, v)
    t1_views = [views[0], views[1]] + views[3:]
    single = net.encoder(enc["image"][:, 0], enc["K"][:, 0])
    return single, fuse(t1_views)


def main(argv=None):
    args = parse(argv)
    rank, world, local = setup()
    device = torch.device(args.device, local) if args.device == "cuda" else torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    net, saved = build_from_checkpoint(args.stage_a, device)
    for p in net.parameters():
        p.requires_grad_(False)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    data = MultiViewHeads(args.ava_root, args.emo_index, "train", args.ava_prob, saved.enc_size,
                          saved.render_size, length=10 ** 7, seed=777 + rank,
                          ava_pack=args.ava_pack, fusion_views=args.fusion_views)
    loader = DataLoader(data, batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                        pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=args.workers > 0)
    enc = net.encoder
    tex_shape = ((enc.tex_size ** 2, getattr(enc, "tex_channels", enc.tex_dim))
                 if getattr(enc, "tex_size", 0) else None)
    prior = SlotPrior(enc.p0.shape[0], saved.token_dim, enc.slot_embed.shape[1], args.dim,
                      args.heads, args.layers, tex_shape=tex_shape).to(device)
    masker = (L.Masker(size=args.mask_size) if args.masker == "segformer"
              else L.ThresholdMasker()).to(device)
    perceptual = L.Perceptual(args.perceptual).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior.load_state_dict(ckpt["prior"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
    model = DDP(prior, device_ids=[local] if device.type == "cuda" else None) if world > 1 else prior
    log = (args.out / "log.jsonl").open("a") if rank == 0 else None
    it, t0 = iter(loader), time.time()
    while step < args.steps:
        batch = to_device(next(it), device)
        if not getattr(saved, "keep_source_background", False):
            e = batch["enc"]
            e["image"] = L.neutral_background(masker, e["image"].flatten(0, 1)).view_as(e["image"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            single, multi = encode_targets(net, batch)
            if not bool(prior.stats_ready):
                prior.update_stats(multi)
                if world > 1:
                    for buf in (prior.mean, prior.std):
                        dist.all_reduce(buf)
                        buf /= world
            fm, x1_hat, t = model(single, multi, enc.slot_embed)
            logs = {"flow": fm.mean()}
            loss = logs["flow"]
            if args.w_render > 0:
                # Decode the one-step estimate at held-out targets (frozen decoder).
                z_hat = enc.compose(x1_hat, single["e"])
                cam_s = take(batch["enc"], 0)
                rs, ts, ks, ims, vals = [], [], [], [], []
                for j in (1, 2, 3):
                    cam = take(batch["tgt"], j)
                    r_q, t_q = target_pose({"R_used": single["R"], "t_used": single["t"]}, cam_s, cam)
                    rs.append(r_q); ts.append(t_q); ks.append(cam["K"])
                    ims.append(cam["image"]); vals.append(cam["valid"])
                rep = lambda x: torch.cat([x] * 3, 0)
                pred = net.decoder({k: rep(v) for k, v in z_hat.items()},
                                   torch.cat(rs), torch.cat(ts), torch.cat(ks))
                image, valid = torch.cat(ims), torch.cat(vals)
                fg, wmap = masker(image)
                terms = L.render_terms(pred, image, valid, fg, wmap, perceptual)
                tw = rep(t).float()  # trust the one-step estimate more at late t
                logs["render"] = ((terms["rgb"] + 0.5 * terms["perc"]) * tw).sum() / tw.sum()
                loss = loss + args.w_render * logs["render"]
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1., (step + 1) / args.warmup) * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        opt.step()
        step += 1
        if rank == 0 and step % args.log_every == 0:
            rec = {"step": step, **{k: round(float(v), 5) for k, v in logs.items()},
                   "sec_per_step": round((time.time() - t0) / args.log_every, 3)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t0 = time.time()
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save({"prior": prior.state_dict(), "optimizer": opt.state_dict(), "step": step,
                        "args": vars(args),
                        "prior_kwargs": {"slots": enc.p0.shape[0], "token_dim": saved.token_dim,
                                         "tex_shape": tex_shape,
                                         "slot_dim": enc.slot_embed.shape[1], "dim": args.dim,
                                         "heads": args.heads, "layers": args.layers}},
                       args.out / f"prior_{step:07d}.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
