"""Static-stage training of the view-invariant slot representation (VISR).

Launch (8 GPUs, one node):
  torchrun --standalone --nproc_per_node=8 -m train_static_claude.train \
      --ava-root data/ava256_rgb_4TB --emo-index data_cache/emotalk3d_frames/index.jsonl \
      --out train_static_claude/runs/visr_a
See README.md for the full recipe.
"""
import argparse
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from . import losses as L
from .dataset import MultiViewHeads, collate
from .model import VISR, fuse, relative_extrinsic
from .geometry import compose, rays_in_head_frame

ANGLE_EDGES = (0, 25, 50, 75, 181)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ava-root", type=str, default=None)
    p.add_argument("--ava-pack", type=str, default=None,
                   help="packed Ava frames from prepare_ava.py (strongly recommended)")
    p.add_argument("--emo-index", type=str, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ava-prob", type=float, default=0.7)
    p.add_argument("--ava-val-count", type=int, default=10)
    p.add_argument("--ava-test-count", type=int, default=10)
    # model
    p.add_argument("--backbone", default="facebook/dinov2-with-registers-base")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--freeze-blocks", type=int, default=4)
    p.add_argument("--slots", type=int, default=4096)
    p.add_argument("--tex-size", type=int, default=8,
                   help="v2 texture patch side per slot (0 = v1 model without textures)")
    p.add_argument("--tex-dim", type=int, default=16)
    p.add_argument("--tex-k", type=int, default=8, help="candidate slots per pixel")
    p.add_argument("--tex-chunk", type=int, default=4, help="renders per checkpointed readout")
    p.add_argument("--dyn-dim", type=int, default=128)
    p.add_argument("--no-color-copy", action="store_true",
                   help="v3 off: texels carry learned features only (v2 behaviour)")
    p.add_argument("--no-copy-hard", action="store_true",
                   help="v4 off: learned copy gate instead of geometric visibility")
    p.add_argument("--no-copy-coverage", action="store_true",
                   help="v4.1 off: copy colour also where rays miss every patch")
    p.add_argument("--corr-bound", type=float, default=1.5,
                   help="v4: max |decoder colour correction| in logit units (0 = free)")
    p.add_argument("--vis-tol", type=float, default=0.12, help="v4 z-buffer tolerance (dm)")
    p.add_argument("--w-surf", type=float, default=1.0, help="v4: slots on the surface")
    p.add_argument("--surf-gate", type=float, default=0.6, help="ignore slots farther (dm)")
    p.add_argument("--surf-points", type=int, default=2048)
    p.add_argument("--pose-iters", type=int, default=2,
                   help="v3 iterative pose refinement steps (0 = v2 behaviour)")
    p.add_argument("--init", type=Path, default=None,
                   help="warm start: load matching weights from a checkpoint (new optimizer, step 0)")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--token-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--enc-blocks", type=int, nargs=2, default=(2, 3))
    p.add_argument("--decoder-layers", type=int, default=8)
    p.add_argument("--enc-size", type=int, default=448)
    p.add_argument("--render-size", type=int, default=256)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--tiny", action="store_true", help="tiny random ViT (CPU tests)")
    # optimization
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--batch", type=int, default=4, help="samples per GPU")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-vit", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=3000)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--gt-pose-warmup", type=int, default=20000,
                   help="Ava: steps over which GT-pose teacher forcing decays 1 -> 0")
    # losses
    p.add_argument("--perceptual", choices=("lpips", "none"), default="lpips")
    p.add_argument("--masker", choices=("segformer", "threshold"), default="segformer")
    p.add_argument("--mask-size", type=int, default=512)
    p.add_argument("--keep-source-background", action="store_true",
                   help="feed raw source images (default: subject on neutral gray)")
    p.add_argument("--no-color-aug", action="store_true",
                   help="disable the per-sample photometric augmentation")
    p.add_argument("--w-detail", type=float, default=2.0,
                   help="extra L1 weight on projected eyes/mouth (0 = off)")
    p.add_argument("--w-rgb", type=float, default=1.0)
    p.add_argument("--w-perc", type=float, default=0.5)
    p.add_argument("--w-alpha", type=float, default=0.2)
    p.add_argument("--w-depth", type=float, default=1.0)
    p.add_argument("--w-pose-abs", type=float, default=1.0)
    p.add_argument("--w-pose-rel", type=float, default=1.0)
    p.add_argument("--w-exchange", type=float, default=0.1)
    p.add_argument("--w-dyn", type=float, default=0.1)
    p.add_argument("--w-id", type=float, default=0.1, help="anti-collapse: identity contrast")
    p.add_argument("--w-e-var", type=float, default=0.1, help="anti-collapse: spread of e")
    p.add_argument("--id-slots", type=int, default=256, help="slots used by the contrast")
    p.add_argument("--id-tau", type=float, default=0.1)
    p.add_argument("--w-groups", type=float, nargs=4, default=(1.0, 0.5, 0.5, 0.5),
                   help="render groups: single s->{a,T}, single a->s, fused->T, reenact->t2")
    # bookkeeping
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-batches", type=int, default=25)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def model_kwargs(args):
    kw = dict(backbone=args.backbone, pretrained=not args.no_pretrained,
              freeze_blocks=args.freeze_blocks, slots=args.slots, dim=args.dim,
              token_dim=args.token_dim, heads=args.heads, blocks=tuple(args.enc_blocks),
              decoder_layers=args.decoder_layers, render_size=args.render_size,
              image_size=args.enc_size, grad_checkpoint=args.grad_ckpt,
              # defaults reproduce v1 when loading checkpoints saved before v2
              dyn_dim=getattr(args, "dyn_dim", 64), tex_size=getattr(args, "tex_size", 0),
              tex_dim=getattr(args, "tex_dim", 16), tex_k=getattr(args, "tex_k", 8),
              tex_chunk=getattr(args, "tex_chunk", 4),
              color_copy=not getattr(args, "no_color_copy", True),
              pose_iters=getattr(args, "pose_iters", 0),
              # v4 (old checkpoints: learned gate, free correction, old tolerance)
              copy_hard=not getattr(args, "no_copy_hard", True),
              corr_bound=getattr(args, "corr_bound", 0.),
              copy_coverage=not getattr(args, "no_copy_coverage", True),
              vis_tol=getattr(args, "vis_tol", 0.25))
    if args.tiny:
        from transformers import Dinov2WithRegistersConfig
        kw["backbone_config"] = Dinov2WithRegistersConfig(
            hidden_size=48, num_hidden_layers=3, num_attention_heads=4, intermediate_size=96,
            patch_size=14, image_size=args.enc_size, num_register_tokens=2)
        kw["feature_layers"] = (1, 2, 3)
        kw["freeze_blocks"] = 1
    return kw


def build_from_checkpoint(path, device="cpu"):
    """Rebuild VISR from a stage-A checkpoint (weights from the file, not the hub)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**ckpt["args"])
    saved.no_pretrained = True
    net = VISR(**model_kwargs(saved))
    missing = load_model_state(net, ckpt["model"])
    # Evaluate older checkpoints exactly as they were trained.
    net.encoder.legacy_pose_pool = any(k.startswith("encoder.pose_norm.") for k in missing)
    return net.to(device).eval(), saved


# ---------------------------------------------------------------------------
def setup():
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    return rank, world, local


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def take(views, idx):
    """Select view ``idx`` (int) from a stacked view dict [B, V, ...]."""
    return {k: v[:, idx] for k, v in views.items()}


def split_views(enc_out, b, v):
    def one(x, i):
        if isinstance(x, (list, tuple)):
            return type(x)(one(y, i) for y in x)
        return x.view(b, v, *x.shape[1:])[:, i]
    return [{k: one(x, i) for k, x in enc_out.items()} for i in range(v)]


def target_pose(enc_view, enc_cam, tgt_cam):
    """Q_target = (E_t E_s^-1) o Q_used(source)."""
    r_rel, t_rel = relative_extrinsic(tgt_cam["E_R"], tgt_cam["E_t"], enc_cam["E_R"], enc_cam["E_t"])
    return compose(r_rel, t_rel, enc_view["R_used"], enc_view["t_used"])


class Forward(torch.nn.Module):
    """The whole differentiable pass of one iteration (a single DDP forward).

    Render plan (each target pose is (E_t E_s^-1) o Q_used of the writing view):
      g0  z(s)             -> tgt[a, T1, T2, T3]   single-source novel views
      g1  z(a)             -> tgt[s]               reverse direction
      g2  fuse(z(s), z(a)) -> tgt[T1, T2, T3]      multi-view fusion
      g3  id(s) + e(d)     -> tgt2[U1, U2]         cross-reenactment at t2
    """

    def __init__(self, net, w_groups):
        super().__init__()
        self.net, self.w_groups = net, w_groups

    def forward(self, batch, p_tf):
        enc_net = self.net.encoder
        enc, has_pose = batch["enc"], batch["has_pose"]
        b, v = enc["image"].shape[:2]
        force = (has_pose[:, None].expand(b, v) *
                 (torch.rand(b, v, device=has_pose.device) < p_tf).float()).flatten()
        flat = lambda x: x.flatten(0, 1)
        out = enc_net(flat(enc["image"]), flat(enc["K"]), (flat(enc["Q_R"]), flat(enc["Q_t"])),
                      force)
        s, a, d = split_views(out, b, v)[:3]
        cams = [take(enc, i) for i in range(v)]
        f = fuse([s, a])
        plan = [(enc_net.compose(s, s["e"]), s, 0, "tgt", [0, 1, 2, 3], 0),
                (enc_net.compose(a, a["e"]), a, 1, "tgt", [4], 0),
                (enc_net.compose(f, f["e"]), s, 0, "tgt", [1, 2, 3], 0),
                (enc_net.compose(s, d["e"]), d, 2, "tgt2", [0, 1], 1)]
        cols = {k: [] for k in ("p", "h", "R", "t", "K", "image", "valid", "w", "QR", "Qt",
                                "verts", "group")}
        for gi, (z, view, ei, key, idx, t_index) in enumerate(plan):
            for j in idx:
                cam = take(batch[key], j)
                r_q, t_q = target_pose(view, cams[ei], cam)
                if "tex" in z:
                    cols.setdefault("tex", []).append(z["tex"])
                for k, x in (("p", z["p"]), ("h", z["h"]), ("R", r_q), ("t", t_q),
                             ("K", cam["K"]), ("image", cam["image"]), ("valid", cam["valid"]),
                             ("QR", cam["Q_R"]), ("Qt", cam["Q_t"]),
                             ("verts", batch["verts"][:, t_index])):
                    cols[k].append(x)
                cols["w"].append(torch.full((b,), self.w_groups[gi], device=r_q.device))
                cols["group"].append(torch.full((b,), gi, device=r_q.device))
        r = {k: torch.cat(x, 0) for k, x in cols.items()}
        pred = self.net.decoder({k: r[k] for k in ("p", "h", "tex") if k in r},
                                r["R"], r["t"], r["K"])
        return pred, r, (s, a, d), cams


class Step:
    """Forward + losses for one batch; shared by train and validation."""

    @torch.no_grad()
    def _surface_points(self, pred, r, fg, n):
        """Canonical surface points from the first n renders' own depth (detached),
        valid where both the render and the target say foreground."""
        size = pred["depth"].shape[-1]
        origin, dirs = rays_in_head_frame(r["R"][:n], r["t"][:n], r["K"][:n], size, size, 1)
        pts = origin[:, None] + pred["depth"][:n].float().flatten(1)[..., None] * dirs
        valid = ((pred["alpha"][:n] > 0.5) & (fg[:n] > 0.5)).flatten(1)
        return pts, valid

    def surface(self, pred, r, fg, batch, b):
        """v4: source slots (render group 0) must lie on the head surface: the
        model's own depth from 4 views at t1, plus GT registration on Ava."""
        args = self.args
        pts, valid = self._surface_points(pred, r, fg, 4 * b)
        pts = pts.view(4, b, -1, 3).transpose(0, 1).flatten(1, 2)
        valid = valid.view(4, b, -1).transpose(0, 1).flatten(1, 2)
        gt = batch["verts"][:, 0].float()
        gt_ok = (batch["has_pose"][:, None] > 0).expand(-1, gt.shape[1])
        loss, near = L.surface_term(r["p"][:b].float(), torch.cat((pts, gt), 1),
                                    torch.cat((valid, gt_ok), 1), args.surf_gate,
                                    args.surf_points)
        return loss.mean(), near.mean()

    def __init__(self, model, masker, perceptual, args):
        self.model, self.masker, self.perceptual, self.args = model, masker, perceptual, args

    def __call__(self, batch, step, train=True):
        args = self.args
        has_pose = batch["has_pose"]
        p_tf = max(0., 1 - step / max(1, args.gt_pose_warmup)) if train else 0.
        if train and not getattr(args, "no_color_aug", False):
            batch["enc"]["image"], batch["tgt"]["image"], batch["tgt2"]["image"] = \
                L.color_augment([batch["enc"]["image"], batch["tgt"]["image"],
                                 batch["tgt2"]["image"]])
        if not args.keep_source_background:
            enc = batch["enc"]
            enc["image"] = L.neutral_background(self.masker, enc["image"].flatten(0, 1)).view_as(
                enc["image"])
        pred, r, (s, a, d), cams = self.model(batch, p_tf)
        b = has_pose.shape[0]
        with torch.no_grad():
            fg, wmap = self.masker(r["image"])
        detail = None
        if getattr(args, "w_detail", 0) > 0:
            # GT target pose where known (Ava), otherwise the predicted one.
            hpr = has_pose.repeat(r["w"].shape[0] // has_pose.shape[0]).view(-1, 1, 1) > 0
            q_r = torch.where(hpr, r["QR"], r["R"].detach())
            q_t = torch.where(hpr[:, :, 0], r["Qt"], r["t"].detach())
            detail = args.w_detail * L.detail_weight(q_r, q_t, r["K"], r["image"].shape[-1])
        terms = L.render_terms(pred, r["image"], r["valid"], fg, wmap, self.perceptual, detail)
        rw = r["w"]
        hp = has_pose.repeat(r["w"].shape[0] // b)
        depth, _ = L.depth_term(pred["depth"], r["verts"], r["QR"], r["Qt"], r["K"])
        logs = {k: (t * rw).sum() / rw.sum() for k, t in terms.items()}
        logs["depth"] = (depth * hp * rw).sum() / ((hp * rw).sum() + 1e-6)
        # Initial and every refined pose; refinements count only where not teacher-forced.
        pose_abs = 0.
        for i, x in enumerate((s, a, d)):
            gt = (cams[i]["Q_R"], cams[i]["Q_t"])
            terms_i = [L.pose_abs_term(*x["pose_iters"][0], *gt)]
            terms_i += [L.pose_abs_term(*it, *gt) * x["pose_free"]
                        for it in x["pose_iters"][1:]]
            pose_abs = pose_abs + sum(terms_i) / len(terms_i)
        pose_abs = pose_abs / 3
        logs["pose_abs"] = (pose_abs * has_pose).sum() / (has_pose.sum() + 1e-6)
        if getattr(args, "w_surf", 0) > 0:
            logs["surf"], logs["surf_5mm"] = self.surface(pred, r, fg, batch, b)
        logs["pose_rel"] = L.pose_rel_term(s, a, (cams[0]["E_R"], cams[0]["E_t"]),
                                           (cams[1]["E_R"], cams[1]["E_t"])).mean()
        logs["exchange"] = L.exchange_term(s, a).mean()
        logs["dyn_inv"] = L.dyn_inv_term(s, a).mean()
        # Anti-collapse. Same slot subset on every rank (seeded by step).
        ids = torch.tensor([zlib.crc32(p.encode()) for p in batch["person"]],
                           device=has_pose.device)
        all_ids = L.gather(ids)
        slots = s["h_id"].shape[1]
        idx = torch.randperm(slots, generator=torch.Generator().manual_seed(step))
        idx = idx[:min(args.id_slots, slots)].to(has_pose.device)
        logs["id_nce"], logs["id_acc"], logs["id_chance"] = L.identity_nce(
            s["h_id"], s["conf"],
            torch.cat((L.gather(a["h_id"]), L.gather(d["h_id"]))),
            torch.cat((L.gather(a["conf"]), L.gather(d["conf"]))),
            ids, torch.cat((all_ids, all_ids)), idx, args.id_tau)
        logs["e_var"], logs["e_std"] = L.variance_hinge(
            torch.cat((L.gather(s["e"]), L.gather(d["e"]))))
        loss = (args.w_rgb * logs["rgb"] + args.w_perc * logs["perc"] +
                args.w_alpha * logs["alpha"] + args.w_depth * logs["depth"] +
                args.w_pose_abs * logs["pose_abs"] + args.w_pose_rel * logs["pose_rel"] +
                args.w_exchange * logs["exchange"] + args.w_dyn * logs["dyn_inv"] +
                args.w_id * logs["id_nce"] + args.w_e_var * logs["e_var"] +
                getattr(args, "w_surf", 0) * logs.get("surf", 0.))
        logs["loss"] = loss
        logs["teacher_forcing"] = torch.tensor(p_tf)
        extra = {"pred": pred, "image": r["image"], "fg": fg, "valid": r["valid"],
                 "n_per_group": b, "angles": batch["angles"], "has_pose": has_pose,
                 "views": (s, a, d), "enc_cams": cams, "terms": terms, "detail": detail}
        return loss, logs, extra


def param_groups(net, args):
    vit, other_decay, other_no = [], [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.vit."):
            vit.append(p)
        elif p.ndim < 2 or name.endswith(("p0", "slot_embed")):
            other_no.append(p)
        else:
            other_decay.append(p)
    return [{"params": vit, "lr": args.lr_vit, "base_lr": args.lr_vit,
             "weight_decay": args.weight_decay},
            {"params": other_decay, "lr": args.lr, "base_lr": args.lr,
             "weight_decay": args.weight_decay},
            {"params": other_no, "lr": args.lr, "base_lr": args.lr, "weight_decay": 0.}]


def lr_factor(step, args):
    if step < args.warmup:
        return (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1., progress)))


# ---------------------------------------------------------------------------
def psnr(pred, gt, mask):
    mse = (((pred - gt) ** 2) * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3)) + 1)
    return -10 * torch.log10(mse.clamp_min(1e-8))


@torch.no_grad()
def validate(step_fn, loader, device, step, out_dir, args, rank, world):
    step_fn.model.eval()
    sums, counts = {}, {}

    def add(key, values):
        sums[key] = sums.get(key, 0.) + float(values.sum())
        counts[key] = counts.get(key, 0) + int(values.numel())

    for i, batch in enumerate(loader):
        if i >= args.val_batches:
            break
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, logs, ex = step_fn(batch, step, train=False)
        for k, v in logs.items():
            add(f"loss/{k}", v.float().reshape(1))
        b = ex["n_per_group"]
        pred, image, fg, valid = ex["pred"], ex["image"], ex["fg"], ex["valid"]
        # single-source group: renders 0..4b-1 correspond to tgt[:, 0:4]
        m = fg * valid
        ps = psnr(pred["rgb"], image, m)
        lp = ex["terms"]["perc"]
        domains = batch["domain"]
        for j in range(4):
            sl = slice(j * b, (j + 1) * b)
            ang = ex["angles"][:, j]
            for lo, hi in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:]):
                sel = (ang >= lo) & (ang < hi)
                for dom, name in ((0, "ava"), (1, "emo")):
                    s2 = sel & (domains == dom)
                    if s2.any():
                        add(f"{name}/psnr_fg_{lo}-{hi}", ps[sl][s2])
                        add(f"{name}/lpips_{lo}-{hi}", lp[sl][s2])
        s, a, d = ex["views"]
        hp = ex["has_pose"] > 0
        if hp.any():
            from .geometry import geodesic
            cam = ex["enc_cams"][0]
            add("ava/pose_rot_deg", torch.rad2deg(geodesic(s["R"], cam["Q_R"]))[hp])
            add("ava/pose_trans_mm", ((s["t"] - cam["Q_t"]).norm(dim=-1) * 100)[hp])
        if i == 0 and rank == 0:
            save_panel(out_dir / f"val_{step:07d}.png", batch, ex)
    if world > 1:
        keys = sorted(sums)
        t = torch.tensor([[sums[k], counts[k]] for k in keys], device=device, dtype=torch.float64)
        dist.all_reduce(t)
        sums = {k: t[i, 0].item() for i, k in enumerate(keys)}
        counts = {k: t[i, 1].item() for i, k in enumerate(keys)}
    step_fn.model.train()
    return {k: sums[k] / max(1, counts[k]) for k in sums}


def save_panel(path, batch, ex, rows=4):
    from PIL import Image
    b = ex["n_per_group"]
    pred = ex["pred"]
    src = F.interpolate(batch["enc"]["image"][:, 0].float(), size=pred["rgb"].shape[-2:],
                        mode="bilinear", align_corners=False)
    lines = []
    for r in range(min(rows, b)):
        tiles = [src[r]]
        for j in range(4):
            k = j * b + r
            tiles += [ex["image"][k], pred["rgb"][k] * pred["alpha"][k] + (1 - pred["alpha"][k]),
                      pred["alpha"][k].expand(3, -1, -1)]
        lines.append(torch.cat(tiles, -1))
    grid = torch.cat(lines, -2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
@torch.no_grad()
def warm_start(net, state):
    """Load a v2 checkpoint into v3: identical tensors are copied; the three layers
    that only gained channels (texel writer, dynamics head, decoder input conv) get
    the old weights in the matching slices, new channels keep their fresh init."""
    own = net.state_dict()
    enc = net.encoder
    head = 3 + enc.to_h.out_features
    texels, c_new = enc.tex_size ** 2, getattr(enc, "tex_channels", enc.tex_dim)
    loaded, sliced = 0, []
    for key, new in own.items():
        old = state.get(key)
        if old is None:
            continue
        if old.shape == new.shape:
            new.copy_(old)
            loaded += 1
        elif key.startswith("encoder.tex_write.") and old.shape[0] < new.shape[0]:
            new[:old.shape[0]] = old           # features first; colour/gate rows new
            sliced.append(key)
        elif key.startswith("encoder.dyn_mlp.2.") and texels:
            new[:head] = old[:head]
            rest = old.shape[0] - head
            # old texel grid: same channels (texel count changed) or same texels
            # (channels changed); upsample the grid with nearest neighbours.
            if rest % c_new == 0 and rest // c_new != texels:
                t_old, c_old = rest // c_new, c_new
            else:
                t_old, c_old = texels, rest // texels
            s_old, s_new = round(t_old ** 0.5), enc.tex_size
            grid = old[head:].view(s_old, s_old, c_old, *old.shape[1:])
            rep = s_new // s_old
            grid = grid.repeat_interleave(rep, 0).repeat_interleave(rep, 1)
            new[head:].view(s_new, s_new, c_new, *new.shape[1:])[:, :, :c_old] = grid
            sliced.append(key)
        elif key == "decoder.refine.0.weight" and old.shape[1] < new.shape[1]:
            keep = old.shape[1] - 1             # [pixel feat, tex feat] | coverage last
            new[:, :keep] = old[:, :keep]
            new[:, -1] = old[:, -1]
            new[:, keep:-1] = 0.               # new colour channels start unused
            sliced.append(key)
    if getattr(net.decoder, "color_copy", False):
        # Start from a clean copy path: the old RGB head would add to the copied colour.
        net.decoder.refine[-1].weight[:3].zero_()
        net.decoder.refine[-1].bias[:3].zero_()
    fresh = sorted(k for k in own if k not in state)
    return {"loaded": loaded, "sliced": sliced, "fresh": fresh}


def load_model_state(net, state):
    """Strict load, except layers added after the checkpoint was written
    (currently only encoder.pose_norm, whose default init is the identity)."""
    missing, unexpected = net.load_state_dict(state, strict=False)
    allowed = ("encoder.pose_norm.",)
    bad = [k for k in missing if not k.startswith(allowed)] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:10]}")
    return missing


def checkpoint_dict(net, opt, step, args):
    return {"model": net.state_dict(), "optimizer": opt.state_dict(), "step": step,
            "args": vars(args), "model_kwargs": {k: v for k, v in model_kwargs(args).items()
                                                 if k != "backbone_config"}}


def main(argv=None):
    args = parse(argv)
    rank, world, local = setup()
    device = torch.device(args.device, local) if args.device == "cuda" else torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed + rank)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))

    train_set = MultiViewHeads(args.ava_root, args.emo_index, "train", args.ava_prob,
                               args.enc_size, args.render_size, length=10 ** 7,
                               seed=args.seed * 1000 + rank, ava_val_count=args.ava_val_count,
                               ava_test_count=args.ava_test_count, ava_pack=args.ava_pack)
    val_set = MultiViewHeads(args.ava_root, args.emo_index, "val", 0.5, args.enc_size,
                             args.render_size, length=args.val_batches * args.batch * world,
                             seed=12345, ava_val_count=args.ava_val_count,
                             ava_test_count=args.ava_test_count, ava_pack=args.ava_pack)
    val_index = list(range(rank, len(val_set), world))
    loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                        collate_fn=collate, pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None, shuffle=False)
    val_loader = DataLoader(torch.utils.data.Subset(val_set, val_index), batch_size=args.batch,
                            num_workers=min(4, args.workers), collate_fn=collate)

    net = VISR(**model_kwargs(args)).to(device)
    wrapped = Forward(net, tuple(args.w_groups))
    model = DDP(wrapped, device_ids=[local] if device.type == "cuda" else None,
                find_unused_parameters=False) if world > 1 else wrapped
    masker = (L.Masker(size=args.mask_size) if args.masker == "segformer" else L.ThresholdMasker()).to(device)
    perceptual = L.Perceptual(args.perceptual).to(device)
    opt = torch.optim.AdamW(param_groups(net, args), betas=(0.9, 0.95))
    step = 0
    if args.init and not args.resume:
        state = torch.load(args.init, map_location="cpu", weights_only=False)["model"]
        report = warm_start(net, state)
        if rank == 0:
            print(json.dumps({"init": str(args.init), **report}), flush=True)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing = load_model_state(net, ckpt["model"])
        if missing:  # older checkpoint: parameter groups no longer line up
            if rank == 0:
                print(json.dumps({"resume_warning": "fresh optimizer state",
                                  "new_params": missing}), flush=True)
        else:
            opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
    step_fn = Step(model, masker, perceptual, args)
    best = float("inf")
    if args.resume and (args.out / "best.json").is_file():
        best = json.loads((args.out / "best.json").read_text())["val_rgb"]
    if rank == 0:
        n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(json.dumps({"trainable_params_M": n_train / 1e6, "world": world,
                          "ava_train": len(train_set.ava), "emo_train_people": len(train_set.emo),
                          "ava_val": len(val_set.ava), "emo_val_people": len(val_set.emo)}),
              flush=True)
    log_file = (args.out / "log.jsonl").open("a") if rank == 0 else None
    t_data, t0 = 0., time.time()
    skipped, dumped = 0, 0
    it = iter(loader)
    model.train()
    while step < args.steps:
        td = time.time()
        batch = to_device(next(it), device)
        t_data += time.time() - td
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_factor(step, args)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, logs, _ = step_fn(batch, step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        if not torch.isfinite(grad):
            opt.zero_grad(set_to_none=True)  # skip a non-finite step (same on all ranks)
            skipped += 1
            if not torch.isfinite(loss) and dumped < 3:
                # Save the offending batch (this rank only) for offline replay.
                dumped += 1
                torch.save({"batch": {k: ({kk: vv.cpu() for kk, vv in v.items()} if isinstance(v, dict)
                                          else v.cpu() if torch.is_tensor(v) else v)
                                      for k, v in batch.items()},
                            "step": step, "rank": rank,
                            "logs": {k: float(v) for k, v in logs.items()}},
                           args.out / f"nan_batch_step{step}_rank{rank}.pt")
        else:
            opt.step()
        step += 1
        if step % args.log_every == 0 and rank == 0:
            rec = {"step": step, **{k: round(float(v), 5) for k, v in logs.items()},
                   "grad": round(float(grad), 3), "skipped_total": skipped, "lr": opt.param_groups[1]["lr"],
                   "sec_per_step": round((time.time() - t0) / args.log_every, 3),
                   "data_sec": round(t_data / args.log_every, 3),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)
                   if device.type == "cuda" else 0.,
                   "ava_frac": float((batch["domain"] == 0).float().mean())}
            print(json.dumps(rec), flush=True)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            t_data, t0 = 0., time.time()
        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(step_fn, val_loader, device, step, args.out, args, rank, world)
            if rank == 0:
                rec = {"step": step, "val": metrics}
                print(json.dumps(rec), flush=True)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                # Keep the checkpoint with the lowest held-out-identity image error.
                score = metrics.get("loss/rgb", float("inf"))
                if score < best:
                    best = score
                    torch.save(checkpoint_dict(net, opt, step, args), args.out / "best.pt")
                    (args.out / "best.json").write_text(json.dumps({"step": step,
                                                                    "val_rgb": score}))
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save(checkpoint_dict(net, opt, step, args), args.out / f"step_{step:07d}.pt")
            (args.out / "latest.json").write_text(json.dumps({"step": step,
                                                              "checkpoint": f"step_{step:07d}.pt"}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
