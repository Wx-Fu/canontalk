"""Supervision. Each term is tied to one property of the representation:

  render      write/read carries the information            (all data)
  depth       readout geometry is metric and correct         (Ava registrations)
  pose_abs    the moving frame is the head frame (gauge)     (Ava head_pose)
  pose_rel    the moving frame is consistent across cameras  (all calibrated rigs)
  exchange    slots are the same whichever camera wrote them (synchronized pairs)
  dyn_inv     the dynamics code is view-invariant            (synchronized pairs)
  id_nce      slots still tell people apart (anti-collapse)  (all, negatives from all GPUs)
  e_var       e keeps unit spread, so dyn_inv can't shrink it (all)
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .geometry import exact, geodesic, project, sample_points, transform, zbuffer_visible
from .model import relative_extrinsic


class Masker(nn.Module):
    """Frozen human parsing (SegFormer, ATR labels) -> soft foreground + head weight.

    Foreground = 1 - P(background). Head = hat + hair + sunglasses + face. The
    weight map emphasizes the head (1.0) over neck/clothing (``body_weight``).
    0.5 because the parser is unsure on shaved scalps (hair vs face); checked
    with check_data.py on Ava and EmoTalk3D crops.
    """

    HEAD = (1, 2, 3, 11)

    def __init__(self, name="mattmdjaga/segformer_b2_clothes", size=512, body_weight=0.5,
                 chunk=16):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(name).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.size, self.body_weight, self.chunk = size, body_weight, chunk
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), False)

    @torch.no_grad()
    def forward(self, image):
        outs = [self._masks(x) for x in image.split(self.chunk)]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    def _masks(self, image):
        h, w = image.shape[-2:]
        x = F.interpolate(image.float(), size=(self.size, self.size), mode="bilinear",
                          align_corners=False)
        logits = self.net(pixel_values=(x - self.mean) / self.std).logits
        prob = F.interpolate(logits.float(), size=(h, w), mode="bilinear",
                             align_corners=False).softmax(1)
        fg = 1 - prob[:, :1]
        head = prob[:, list(self.HEAD)].sum(1, keepdim=True)
        weight = self.body_weight * fg + (1 - self.body_weight) * head
        return fg.clamp(0, 1), weight.clamp(0, 1)


@torch.no_grad()
def neutral_background(masker, images, gray=0.5):
    """Composite the subject on neutral gray before encoding: removes the
    studio background (a camera-identity cue on Ava, and the black/white domain
    gap between Ava and EmoTalk3D). images: [N,3,H,W]."""
    fg, _ = masker(images)
    return images * fg + gray * (1 - fg)


class ThresholdMasker(nn.Module):
    """Fallback for white-background data only (EmoTalk3D)."""

    def forward(self, image):
        fg = ((image < 0.85).any(1, keepdim=True)).float()
        return fg, fg


class Perceptual(nn.Module):
    def __init__(self, kind="lpips"):
        super().__init__()
        self.kind = kind
        if kind == "lpips":
            import lpips
            self.net = lpips.LPIPS(net="vgg", verbose=False).eval()
            for p in self.net.parameters():
                p.requires_grad_(False)

    def forward(self, a, b):
        if self.kind == "none":
            return a.new_zeros(a.shape[0])
        return self.net(a.float() * 2 - 1, b.float() * 2 - 1).flatten()


def render_terms(pred, image, valid, fg, weight, perceptual, detail=None):
    """Per-sample losses [B] for one batch of renders. ``detail`` (optional,
    >= 0) up-weights the L1 term on eyes/mouth; the normalizer ignores it so the
    loss scale of the rest of the face is unchanged."""
    w = valid * weight
    wd = w if detail is None else w * (1 + detail)
    rgb = ((pred["rgb"] - image).abs() * wd).sum((1, 2, 3)) / (3 * w.sum((1, 2, 3)) + 1)
    bce = F.binary_cross_entropy_with_logits(pred["alpha_logit"].float(), fg.float(),
                                             reduction="none")
    alpha = (bce * valid).sum((1, 2, 3)) / (valid.sum((1, 2, 3)) + 1)
    # Composite both on white with the *target* mask; invalid pixels white.
    white = torch.ones_like(image)
    m = fg * valid
    perc = perceptual(pred["rgb"] * m + white * (1 - m), image * m + white * (1 - m))
    return {"rgb": rgb, "alpha": alpha, "perc": perc}


# Canonical (Ava head frame) landmark centers in dm and Gaussian radii (dm):
# mouth and both eyes, measured on registered meshes (nose tip at z = +69 mm).
DETAIL_POINTS = ((0.0, -0.42, 0.46, 0.35), (-0.32, 0.25, 0.33, 0.25), (0.32, 0.25, 0.33, 0.25))


@exact
def detail_weight(r_q, t_q, k, size):
    """[B,1,S,S] in [0,1]: Gaussians at the projected mouth/eyes, faded out when
    the face turns away (facing = head forward axis vs. direction to camera)."""
    pts = r_q.new_tensor([p[:3] for p in DETAIL_POINTS])
    rad = r_q.new_tensor([p[3] for p in DETAIL_POINTS])
    x = transform(r_q, t_q, pts[None].expand(r_q.shape[0], -1, -1))
    uv, z = project(x, k)
    sigma = k[:, None, 0, 0] * rad[None] / z.clamp_min(1e-3) / 2           # pixels
    forward = r_q[:, :, 2]                                                 # head +z in camera
    head_dir = F.normalize(t_q, dim=-1)
    facing = (-(forward * head_dir).sum(-1) * 2).clamp(0, 1)               # [B]
    ys = torch.arange(size, device=k.device, dtype=torch.float32) + 0.5
    gy = (ys[None, None, :] - uv[..., 1:2]) ** 2                           # [B,3,S]
    gx = (ys[None, None, :] - uv[..., 0:1]) ** 2
    g = torch.exp(-(gy[..., :, None] + gx[..., None, :]) / (2 * sigma[..., None, None] ** 2))
    return (g.amax(1, keepdim=True) * facing[:, None, None, None]).clamp(0, 1)


@exact
def surface_term(slots, targets, target_valid, gate=0.6, points=1024, generator=None):
    """One-sided distance (dm) from each slot to the nearest surface point.
    slots [B,N,3]; targets [B,M,3] (detached) with validity [B,M]. Slots farther
    than ``gate`` from every target are ignored (e.g. back of the head with no
    surface evidence). Returns per-sample mean distance [B] and the
    fraction of slots within 5 mm."""
    b, m = target_valid.shape
    score = torch.rand(b, m, device=slots.device, generator=generator) + target_valid.float()
    idx = score.topk(min(points, m), dim=1).indices
    tgt = torch.gather(targets, 1, idx[..., None].expand(-1, -1, 3))
    ok = torch.gather(target_valid, 1, idx)
    d = torch.cdist(slots, tgt)                                           # [B,N,P]
    d = d.masked_fill(~ok[:, None, :], float("inf")).min(-1).values       # [B,N]
    use = d < gate
    loss = torch.where(use, d, torch.zeros_like(d)).sum(1) / use.sum(1).clamp_min(1)
    return loss, (d < 0.05).float().mean(1)


@torch.no_grad()
def color_augment(views, generator=None):
    """One random photometric transform per sample applied to all its views
    (a different 'studio'): channel gains, brightness, saturation, gamma.
    views: list of [B, V, 3, H, W] tensors in [0,1], modified consistently."""
    b, dev = views[0].shape[0], views[0].device
    r = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(b, *shape, device=dev)
    gain = r(0.85, 1.15, 3).view(b, 1, 3, 1, 1)
    bright = r(0.85, 1.15).view(b, 1, 1, 1, 1)
    sat = r(0.7, 1.3).view(b, 1, 1, 1, 1)
    gamma = r(0.8, 1.25).view(b, 1, 1, 1, 1)
    out = []
    for x in views:
        y = x.float().clamp(0, 1) ** gamma
        gray = y.mean(2, keepdim=True)
        y = (gray + sat * (y - gray)) * gain * bright
        out.append(y.clamp(0, 1).to(x.dtype))
    return out


@exact
def depth_term(pred_depth, verts, q_r, q_t, k, tolerance=0.08, buffer=64):
    """L1 (dm) between predicted along-ray depth and GT registered surface at
    vertices visible in the target view. verts: head frame [B,V,3]; (q_r,q_t)
    the *ground-truth* head->target-camera pose; k the target crop intrinsics."""
    size = pred_depth.shape[-1]
    x = transform(q_r, q_t, verts)
    uv, z = project(x, k)
    dist = x.norm(dim=-1)
    vis = zbuffer_visible(uv * (buffer / size), z, buffer, buffer, tolerance)
    sampled = sample_points(pred_depth, uv)[..., 0]
    err = ((sampled - dist).abs().clamp(max=1.0) * vis).sum(1)
    return err / vis.sum(1).clamp_min(1), vis.float().mean(1)


@exact
def pose_abs_term(r_hat, t_hat, r_gt, t_gt):
    return geodesic(r_hat.float(), r_gt.float()) + (t_hat - t_gt).abs().sum(-1)


def pose_rel_term(enc_a, enc_b, cam_a, cam_b):
    """Two synchronized views must agree on the head's pose in the rig:
    R_b R_a^T = R_rel (orientation) and t_b = R_rel t_a + t_rel (the head
    center seen from b equals a's estimate moved by the calibrated extrinsic).
    The translation check uses the calibrated R_rel, so rotation errors are not
    amplified by the ~1-2 m camera distance."""
    return _pose_rel(enc_a["R"], enc_a["t"], enc_b["R"], enc_b["t"], *cam_a, *cam_b)


@exact
def _pose_rel(r_a, t_a, r_b, t_b, e_r_a, e_t_a, e_r_b, e_t_b):
    r_rel, t_rel = relative_extrinsic(e_r_b, e_t_b, e_r_a, e_t_a)
    r_pred = r_b @ r_a.transpose(-1, -2)
    t_from_a = (r_rel @ t_a[..., None])[..., 0] + t_rel
    return geodesic(r_pred, r_rel) + (t_b - t_from_a).abs().sum(-1)


def exchange_term(a, b):
    """Slots written from two synchronized views must agree where *both*
    observed them. Weights are detached: a view cannot lower its confidence to
    escape the constraint; confidence is learned only through fusion."""
    w = (torch.sigmoid(a["conf"]) * torch.sigmoid(b["conf"])).detach()
    cos = 1 - F.cosine_similarity(a["h_id"].float(), b["h_id"].float(), dim=-1)
    pos = (a["dp_id"] - b["dp_id"]).abs().sum(-1)
    if "tex_id" in a:
        cos = cos + 1 - F.cosine_similarity(a["tex_id"].float().flatten(2),
                                            b["tex_id"].float().flatten(2), dim=-1)
    return ((cos + pos) * w).sum(1) / w.sum(1).clamp_min(1e-3)


def dyn_inv_term(a, b):
    return (a["e"].float() - b["e"].float()).pow(2).mean(-1)


def gather(x):
    """Concatenate x from all ranks; only the local part keeps gradients."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    # NCCL requires contiguous tensors; slot tensors are slices of a batched view.
    local = x.detach().contiguous()
    parts = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, local)
    parts[dist.get_rank()] = x
    return torch.cat(parts, 0)


@exact
def identity_nce(h_a, c_a, h_c, c_c, ids_a, ids_c, idx, tau=0.1):
    """Anchor slots (b samples) must match the same person's candidates (C slot
    sets, other views/times) better than other people's. Features are centered
    per slot, since the part shared by all people says nothing about identity.
    Same-person candidates are positives, never negatives.
    Returns (loss, top-1 retrieval accuracy, chance level of that accuracy)."""
    h_a, h_c = h_a[:, idx], h_c[:, idx]
    center = h_c.mean(0, keepdim=True).detach()
    za = F.normalize(h_a - center, dim=-1)
    zc = F.normalize(h_c - center, dim=-1)
    w = (torch.sigmoid(c_a[:, idx])[:, None] * torch.sigmoid(c_c[:, idx])[None]).detach()
    cos = (za[:, None] * zc[None]).sum(-1)
    sim = (cos * w).sum(-1) / w.sum(-1).clamp_min(1e-3)
    pos = ids_a[:, None] == ids_c[None]
    logp = F.log_softmax(sim / tau, dim=1)
    loss = -(logp * pos).sum(1) / pos.sum(1).clamp_min(1)
    hit = pos[torch.arange(len(sim), device=sim.device), sim.argmax(1)].float()
    return loss.mean(), hit.mean(), pos.float().mean()


@exact
def variance_hinge(x, gamma=1.0):
    """VICReg-style: per-dimension std across samples should reach gamma."""
    std = (x.var(0) + 1e-4).sqrt()
    return F.relu(gamma - std).mean(), std.mean()
