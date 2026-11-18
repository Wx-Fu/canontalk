"""View-invariant slot representation (VISR): encoder, fusion and ray decoder.

Representation (all in a learned head-canonical frame, decimeters):

  z = { p_j, h_j, w_j }_{j=1..N}   N canonical *slots* shared by every person
      p_j  position  = p0_j + dp_id_j + dp_dyn_j(e)
      h_j  feature   = h_id_j + h_dyn_j(e)
      T_j  texture   = T_id_j + T_dyn_j(e)   (v2, optional) P x P x C feature patch on
                       the slot's tangent plane: sub-slot detail (~3 mm texels)
      w_j  precision (how strongly this view observed slot j); used only for fusion
  e      low-dimensional dynamics code (expression / lips), view-invariant
  Q      head->camera pose, *not* part of z

Write (encoder, one image): ViT features -> Q_hat = (R, t) (the learned moving
frame) -> every slot is posed into the source camera with Q_hat, projected,
and gathers pixel-aligned ViT features, a raw RGB patch and global context
-> outputs are expressed back in the canonical frame. The source camera is
never passed on.

Read (decoder, any camera): rays of the target pixels are expressed in the
canonical frame with the *target* pose; ray queries cross-attend to slots with
an attention bias that depends only on ray-slot geometry (perpendicular
distance and relative depth), which is invariant to the global rigid frame.
v2: each pixel ray additionally reads the texture patches of the K slots
closest to its patch ray at the ray's local tangent-plane coordinates (also
invariant to the global frame), which gives detail below the slot spacing.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .geometry import (exact, fourier, fourier_dim, invert, matrix_to_rot6d, pixel_rays, project,
                       rays_in_head_frame, plucker, rot6d_to_matrix, sample_points, transform,
                       zbuffer_visible)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_EXTENT_DM = 3.6  # must match the datasets' metric crop rule
# Frontal head in an OpenCV camera: head x -> cam x, head y (up) -> cam -y,
# head z (forward) -> cam -z.
FRONTAL_6D = (1., 0., 0., 0., -1., 0.)


def amp_cast(x):
    """Cast once to the autocast dtype so attention layers share one bias copy."""
    if torch.is_autocast_enabled(x.device.type):
        return x.to(torch.get_autocast_dtype(x.device.type))
    return x


def mlp(i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth - 1):
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Attention(nn.Module):
    """Multi-head attention with an optional additive float bias [B, H, Nq, Nk]."""

    def __init__(self, dim, heads, kv_dim=None):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim or dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context=None, bias=None):
        b, n, d = x.shape
        context = x if context is None else context
        q = self.q(x).view(b, n, self.heads, -1).transpose(1, 2)
        k, v = self.kv(context).view(b, context.shape[1], 2, self.heads, -1).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None
                                           else bias.to(q.dtype))
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class CrossBlock(nn.Module):
    """self-attn over queries, geometry-biased cross-attn to context, MLP."""

    def __init__(self, dim, heads, ctx_dim):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.nc = nn.LayerNorm(ctx_dim)
        self.self_attn = Attention(dim, heads)
        self.cross = Attention(dim, heads, ctx_dim)
        self.ff = mlp(dim, 4 * dim, dim)

    def forward(self, x, ctx, bias):
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross(self.n2(x), self.nc(ctx), bias)
        return x + self.ff(self.n3(x))


def ellipsoid_points(n, generator):
    """Initial canonical slot layout: a head-sized ellipsoid volume (dm).

    Only an initialization; positions are learned. Head frame: x lateral,
    y up, z forward (Ava head_pose convention).
    """
    pts = []
    while sum(len(p) for p in pts) < n:
        x = torch.rand(4 * n, 3, generator=generator) * 2 - 1
        x = x[(x ** 2).sum(-1) <= 1]
        x = x * torch.sign(x) * (x.abs() ** 0.5)  # bias towards the shell
        pts.append(x)
    x = torch.cat(pts)[:n]
    return x * torch.tensor([0.85, 1.25, 1.1]) + torch.tensor([0., 0.0, -0.6])


ELLIPSOID_CENTER = (0., 0., -0.6)
ELLIPSOID_AXES = (0.85, 1.25, 1.1)


def slot_frames(p0):
    """Outward normal of the initial ellipsoid at each slot plus a tangent
    basis (t1 ~ head x, t2 ~ head y). Fixed convention, identical for all
    people, so a slot's texture patch has the same orientation everywhere."""
    c = p0.new_tensor(ELLIPSOID_CENTER)
    axes = p0.new_tensor(ELLIPSOID_AXES)
    n = F.normalize((p0 - c) / axes ** 2, dim=-1)
    up = p0.new_tensor((0., 1., 0.)).expand_as(n)
    side = p0.new_tensor((1., 0., 0.)).expand_as(n)
    ref = torch.where((n * up).sum(-1, keepdim=True).abs() > 0.9, side, up)
    t1 = F.normalize(torch.cross(ref, n, dim=-1), dim=-1)
    t2 = torch.cross(n, t1, dim=-1)
    return t1, t2, n


@exact
def axis_angle_to_matrix(w):
    """Rodrigues: rotation vector [B,3] -> [B,3,3]."""
    theta = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / theta
    zero = torch.zeros_like(k[:, 0])
    kx = torch.stack((zero, -k[:, 2], k[:, 1], k[:, 2], zero, -k[:, 0],
                      -k[:, 1], k[:, 0], zero), -1).view(-1, 3, 3)
    eye = torch.eye(3, device=w.device).expand_as(kx)
    th = theta[..., None]
    return eye + torch.sin(th) * kx + (1 - torch.cos(th)) * kx @ kx


def texel_grid(size, device):
    """Texel centers of a size x size patch in [-0.5, 0.5]^2, row-major (v, u)."""
    a = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size - 0.5
    v, u = torch.meshgrid(a, a, indexing="ij")
    return torch.stack((u, v), -1).reshape(-1, 2)


class Encoder(nn.Module):
    def __init__(self, backbone="facebook/dinov2-with-registers-base", backbone_config=None,
                 pretrained=True, feature_layers=(4, 8, 12), freeze_blocks=4, slots=2048,
                 dim=512, token_dim=256, heads=8, blocks=(2, 2), dyn_dim=64, image_size=448,
                 tex_size=0, tex_dim=16, color_copy=False, pose_iters=0,
                 copy_hard=False, vis_tol=0.25):
        super().__init__()
        # v4: copy observed colour wherever the texel is geometrically visible
        # (no learned gate the model can close); stricter z-buffer tolerance (dm).
        self.copy_hard, self.vis_tol = bool(copy_hard and color_copy), vis_tol
        self.tex_size, self.tex_dim = tex_size, tex_dim
        # v3: texels also store an observed colour (3 extra channels).
        self.color_copy = bool(color_copy and tex_size)
        self.tex_channels = tex_dim + (3 if self.color_copy else 0)
        self.pose_iters = min(pose_iters, len(blocks))
        from transformers import AutoConfig, AutoModel
        if backbone_config is not None:
            self.vit = AutoModel.from_config(backbone_config)
        elif pretrained:
            self.vit = AutoModel.from_pretrained(backbone)
        else:
            self.vit = AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        cfg = self.vit.config
        self.patch = cfg.patch_size
        self.registers = getattr(cfg, "num_register_tokens", 0)
        self.feature_layers = feature_layers
        for p in self.vit.embeddings.parameters():
            p.requires_grad_(False)
        for layer in self.vit.encoder.layer[:freeze_blocks]:
            for p in layer.parameters():
                p.requires_grad_(False)
        # hidden_states are taken before the final norm/pooler: freeze them so
        # DDP (find_unused_parameters=False) sees no unused trainable weights.
        for name in ("layernorm", "pooler"):
            module = getattr(self.vit, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)
        c = cfg.hidden_size
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), False)
        self.fuse = nn.Linear(c * len(feature_layers), dim)
        self.ray_embed = nn.Linear(fourier_dim(3, 4), dim)
        nn.init.zeros_(self.ray_embed.weight)
        nn.init.zeros_(self.ray_embed.bias)
        # pose head: global ViT token + pooled features + intrinsics summary
        self.pose_head = mlp(c + dim + 3, dim, 6 + 3, depth=3)
        nn.init.zeros_(self.pose_head[-1].weight)
        with torch.no_grad():
            self.pose_head[-1].bias.copy_(torch.tensor(FRONTAL_6D + (0., 0., 0.)))
        # canonical slots
        g = torch.Generator().manual_seed(0)
        self.p0 = nn.Parameter(ellipsoid_points(slots, g))
        self.slot_embed = nn.Parameter(torch.randn(slots, dim) * 0.02)
        lift_in = dim + 48 + fourier_dim(3, 6) + fourier_dim(3, 4) + 2
        self.lift = nn.Linear(lift_in, dim)
        self.stages = nn.ModuleList(
            nn.ModuleList(CrossBlock(dim, heads, dim) for _ in range(n)) for n in blocks)
        # Fixed multi-scale bank (ALiBi-style, not learned): keeps the attention
        # bias gradient-free so SDPA can use its memory-efficient kernel.
        self.register_buffer("log_sigma", torch.linspace(math.log(0.02), math.log(0.5), heads))
        self.offset_heads = nn.ModuleList(nn.Linear(dim, 3) for _ in blocks)
        for head in self.offset_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.out_norm = nn.LayerNorm(dim)
        self.to_h = nn.Linear(dim, token_dim)
        self.to_conf = nn.Linear(dim, 1)
        self.to_dyn = mlp(dim, dim, dyn_dim)
        # dynamics: e -> per-slot residuals (shared MLP over slot embeddings)
        self.dyn_slot = nn.Linear(dim, 128)
        tex_flat = tex_size * tex_size * self.tex_channels
        self.dyn_mlp = mlp(dyn_dim + 128, 256 + (256 if tex_size else 0), 3 + token_dim + tex_flat)
        nn.init.zeros_(self.dyn_mlp[-1].weight)
        nn.init.zeros_(self.dyn_mlp[-1].bias)
        self.heads = heads
        if self.pose_iters:
            # v3: iterative pose refinement from slots lifted with the current pose
            self.pose_queries = nn.Parameter(torch.randn(4, dim) * 0.02)
            self.pose_pool = Attention(dim, heads)
            self.pose_norm = nn.LayerNorm(dim)
            self.pose_refine = nn.ModuleList(mlp(4 * dim, dim, 6) for _ in range(self.pose_iters))
            for head in self.pose_refine:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        if tex_size:
            # Full-resolution image features (stride 2) for the texture write.
            self.hires = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(48, 48, 3, padding=1))
            self.tex_ctx = nn.Linear(dim, 64)
            # features (+ v3: prior-colour logits and an observation gate)
            self.tex_write = mlp(48 + 3 + 3 + 64 + 2, 128,
                                 tex_dim + (4 if self.color_copy else 0), depth=3)
            if self.color_copy:
                with torch.no_grad():  # observation gate starts mostly open (sigmoid(2) = 0.88)
                    self.tex_write[-1].bias[-1].fill_(2.)
            # Patch side from the expected *surface* slot spacing: about half the
            # slots end up on the head surface (ellipsoid area, Thomsen's formula).
            # Nearest-neighbour distances of the initial cloud underestimate it.
            a, b_, c = (100 * x for x in ELLIPSOID_AXES)                    # mm
            q = 1.6075
            area = 4 * math.pi * (((a * b_) ** q + (a * c) ** q + (b_ * c) ** q) / 3) ** (1 / q)
            spacing = math.sqrt(area / (0.5 * slots)) / 100                 # dm
            # Patch side 1.5x the spacing so neighbouring patches overlap.
            self.register_buffer("tex_extent", torch.tensor([1.5 * spacing]))

    # -------------------------------------------------------------- backbone
    def features(self, image, k):
        b, _, s, _ = image.shape
        x = (image - self.mean) / self.std
        out = self.vit(pixel_values=x, output_hidden_states=True)
        start = 1 + self.registers
        grid = s // self.patch
        hidden = [out.hidden_states[i][:, start:] for i in self.feature_layers]
        tokens = self.fuse(torch.cat(hidden, -1))
        rays = pixel_rays(k, s, s, self.patch)
        tokens = tokens + self.ray_embed(fourier(rays, 4, 4))
        global_token = out.hidden_states[-1][:, 0]
        fmap = tokens.transpose(1, 2).reshape(b, -1, grid, grid)
        return tokens, fmap, global_token

    def pose(self, tokens, global_token, k, size):
        intr = torch.stack((k[:, 0, 0] / size, k[:, 0, 2] / size - 0.5,
                            k[:, 1, 2] / size - 0.5), -1)
        y = self.pose_head(torch.cat((global_token, tokens.mean(1), intr), -1).float()).float()
        return self._pose_from(y, k, size)

    @exact
    def _pose_from(self, y, k, size):
        r = rot6d_to_matrix(y[:, :6])
        # Head-origin projection (u, v) relative to the crop center, and depth
        # relative to the metric crop prior z0 = fx * extent / size.
        u = size / 2 * (1 + y[:, 6])
        v = size / 2 * (1 + y[:, 7])
        z = k[:, 0, 0] * CROP_EXTENT_DM / size * torch.exp(y[:, 8])
        pix = torch.stack((u, v, torch.ones_like(u)), -1)
        t = z[:, None] * (pix[:, None] @ torch.linalg.inv(k.float()).transpose(-1, -2))[:, 0]
        return r, t

    @exact
    def _pose_update(self, r, t, y, k, size):
        """Small update: rotation vector (<= 0.1 rad / axis) applied in the camera
        frame; head-origin image shift (<= 5% of the crop) and depth scale (<= 5%)."""
        r_new = axis_angle_to_matrix(0.1 * torch.tanh(y[:, :3])) @ r
        uvw = (k @ t[..., None])[..., 0]
        z = uvw[:, 2:3]
        uv = uvw[:, :2] / z + 0.05 * size * torch.tanh(y[:, 3:5])
        z = z * torch.exp(0.05 * torch.tanh(y[:, 5:6]))
        pix = torch.cat((uv, torch.ones_like(z)), -1)
        t_new = z * (pix[:, None] @ torch.linalg.inv(k).transpose(-1, -2))[:, 0]
        return r_new, t_new

    # ---------------------------------------------------------------- lifting
    @exact
    def _lift_inputs(self, p, r, t, k, fmap, image, size):
        x_cam = transform(r, t, p)
        uv, z = project(x_cam, k)
        inside = ((uv >= 0) & (uv < size)).all(-1) & (z > 0)
        grid = fmap.shape[-1]
        f_loc = sample_points(fmap, uv * (grid / size))
        offsets = torch.stack(torch.meshgrid(
            torch.arange(4, device=uv.device) - 1.5, torch.arange(4, device=uv.device) - 1.5,
            indexing="xy"), -1).reshape(-1, 2).float() * 3.0
        patch = sample_points(image, (uv[:, :, None, :] + offsets).flatten(1, 2))
        patch = patch.reshape(uv.shape[0], uv.shape[1], -1)
        view_dir = F.normalize(x_cam, dim=-1) @ r  # camera->point direction, canonical frame
        depth_rel = (z - t[:, None, 2])[..., None]
        inp = torch.cat((f_loc, patch.to(f_loc.dtype), fourier(p, 6, 4).to(f_loc.dtype),
                         fourier(view_dir, 4, 4).to(f_loc.dtype), depth_rel.to(f_loc.dtype),
                         inside[..., None].to(f_loc.dtype)), -1)
        return inp, uv

    @exact
    def _bias(self, uv, size, grid):
        centers = (torch.arange(grid, device=uv.device, dtype=torch.float32) + 0.5) / grid
        cy, cx = torch.meshgrid(centers, centers, indexing="ij")
        c = torch.stack((cx, cy), -1).reshape(1, 1, -1, 2)
        d2 = ((uv[:, :, None, :].float() / size - c) ** 2).sum(-1)  # [B, N, M]
        sigma2 = torch.exp(2 * self.log_sigma).view(1, -1, 1, 1)
        bias = -d2[:, None] / (2 * sigma2)
        # the last two heads keep global, unbiased context
        local = torch.ones(self.heads, device=uv.device)
        local[-2:] = 0.
        return bias.clamp(min=-30.) * local.view(1, -1, 1, 1)

    @exact
    def _texel_inputs(self, p, r, t, k, hires, image, size):
        """Project every texel of every slot into the source image and gather
        evidence: hi-res features, RGB, facing (n . to-camera) and z-buffer
        visibility of the slot center. Returns [B, N, P*P, C_in]."""
        b, n, _ = p.shape
        t1, t2, nrm = slot_frames(self.p0.detach())
        grid = texel_grid(self.tex_size, p.device)                         # [T,2]
        offs = self.tex_extent * (grid[None, :, 0:1] * t1[:, None, :] +
                                  grid[None, :, 1:2] * t2[:, None, :])     # [N,T,3]
        pts = p[:, :, None, :] + offs[None]                                # [B,N,T,3]
        x_cam = transform(r, t, pts.flatten(1, 2))
        uv, z = project(x_cam, k)
        feat = sample_points(hires, uv * (hires.shape[-1] / size))
        rgb = sample_points(F.avg_pool2d(image, 3, stride=1, padding=1,
                                         count_include_pad=False), uv)
        n_cam = nrm @ r.transpose(-1, -2)                                  # [B,N,3]
        c_cam = transform(r, t, p)
        uvc, zc = project(c_cam, k)
        facing = -(n_cam * F.normalize(c_cam, dim=-1)).sum(-1)             # >0: faces camera
        vis = zbuffer_visible(uvc * (64 / size), zc, 64, 64, self.vis_tol).float()
        inside = (((uv >= 0) & (uv < size)).all(-1) & (z > 0)).float()
        tsz = pts.shape[2]
        per_slot = torch.stack((facing, vis), -1)[:, :, None].expand(-1, -1, tsz, -1)
        texel = torch.cat((feat.view(b, n, tsz, -1), rgb.view(b, n, tsz, 3),
                           inside.view(b, n, tsz, 1), grid[None, None].expand(b, n, -1, -1)), -1)
        return torch.cat((texel, per_slot), -1)

    def forward(self, image, k, pose_override=None, override_mask=None):
        """image [B,3,S,S] in [0,1]; k [B,3,3]. Returns a dict (all canonical)."""
        b, _, size, _ = image.shape
        tokens, fmap, global_token = self.features(image, k)
        r_hat, t_hat = self.pose(tokens, global_token, k, size)
        r, t = r_hat, t_hat
        if pose_override is not None:
            m = override_mask.view(-1, 1, 1).to(r.dtype)
            r = m * pose_override[0] + (1 - m) * r
            t = m[:, :, 0] * pose_override[1] + (1 - m[:, :, 0]) * t
        free = torch.ones(b, device=image.device) if pose_override is None else 1 - override_mask
        iters = [(r_hat, t_hat)]
        p = self.p0[None].expand(b, -1, -1)
        x = self.slot_embed[None].expand(b, -1, -1)
        for i, (stage, offset_head) in enumerate(zip(self.stages, self.offset_heads)):
            inp, uv = self._lift_inputs(p, r, t, k, fmap, image, size)
            x = x + self.lift(inp)
            bias = amp_cast(self._bias(uv, size, fmap.shape[-1]))
            for block in stage:
                x = block(x, tokens, bias)
            p = p + 0.3 * torch.tanh(offset_head(x).float())
            if i < self.pose_iters:
                # Pre-norm and float32 (4 queries, cheap): the raw residual stream
                # under bf16 occasionally produced inf here, and 0-init weights
                # times inf gave NaN poses (~5% of v3 steps were skipped).
                with torch.autocast(x.device.type, enabled=False):
                    q = self.pose_queries[None].expand(b, -1, -1).float()
                    xs = x.float() if getattr(self, "legacy_pose_pool", False) else \
                        self.pose_norm(x.float())  # legacy: checkpoints trained without it
                    pooled = self.pose_pool(q, xs).flatten(1)
                    y = self.pose_refine[i](pooled)
                y = torch.nan_to_num(y, nan=0., posinf=0., neginf=0.)
                r_new, t_new = self._pose_update(r, t, y, k, size)
                f = free.view(-1, 1, 1).to(r.dtype)  # forced (GT) samples keep GT
                r = f * r_new + (1 - f) * r
                t = f[:, :, 0] * t_new + (1 - f[:, :, 0]) * t
                r_hat, t_hat = torch.where(f > 0, r_new, r_hat), torch.where(f[:, :, 0] > 0, t_new, t_hat)
                iters.append((r_new, t_new))
        x = self.out_norm(x)
        e = self.to_dyn(x.mean(1))
        out = {"R": r_hat, "t": t_hat, "R_used": r, "t_used": t, "pose_iters": iters,
               "pose_free": free, "dp_id": p - self.p0[None], "h_id": self.to_h(x),
               "conf": self.to_conf(x)[..., 0], "e": e}
        if self.tex_size:
            hires = self.hires(image)
            texel = self._texel_inputs(p, r, t, k, hires, image, size)
            ctx = self.tex_ctx(x)[:, :, None].expand(-1, -1, texel.shape[2], -1)
            y = self.tex_write(torch.cat((texel.to(ctx.dtype), ctx), -1))
            if self.color_copy:
                # Observed colour where the texel is seen, learned colour elsewhere.
                rgb = texel[..., 48:51].float()
                gate = torch.sigmoid(y[..., -1:].float())
                if self.copy_hard:  # in image, unoccluded, facing the camera
                    gate = (texel[..., 51:52] * texel[..., 55:56] *
                            (texel[..., 54:55] > 0.1)).float()
                if getattr(self, "force_gate", None) is not None:  # diagnostic only
                    gate = torch.full_like(gate, self.force_gate)
                color = gate * rgb + (1 - gate) * torch.sigmoid(y[..., -4:-1].float())
                y = torch.cat((y[..., :self.tex_dim], color.to(y.dtype)), -1)
            out["tex_id"] = y
        return out

    def dynamics(self, e):
        b = e.shape[0]
        slot = self.dyn_slot(self.slot_embed)[None].expand(b, -1, -1)
        y = self.dyn_mlp(torch.cat((e[:, None].expand(-1, slot.shape[1], -1), slot), -1))
        c = self.to_h.out_features
        return 0.3 * torch.tanh(y[..., :3].float()), y[..., 3:3 + c], y[..., 3 + c:]

    def compose(self, z_id, e):
        """Identity slots + dynamics code -> renderable slot set."""
        dp, dh, dt = self.dynamics(e)
        z = {"p": self.p0[None] + z_id["dp_id"] + dp, "h": z_id["h_id"] + dh}
        if self.tex_size:
            z["tex"] = z_id["tex_id"] + dt.view_as(z_id["tex_id"])
        return z


def fuse(views):
    """Precision-weighted (product-of-experts style) fusion of per-view slots."""
    w = torch.stack([F.softplus(v["conf"].float()) + 1e-3 for v in views])  # [V,B,N]
    norm = w / w.sum(0, keepdim=True)
    out = {"dp_id": sum(n[..., None] * v["dp_id"] for n, v in zip(norm, views)),
           "h_id": sum(n[..., None].to(v["h_id"].dtype) * v["h_id"] for n, v in zip(norm, views)),
           "conf": torch.log(torch.expm1(w.sum(0)).clamp_min(1e-6)),
           "e": sum(v["e"] for v in views) / len(views)}
    if "tex_id" in views[0]:
        out["tex_id"] = sum(n[..., None, None].to(v["tex_id"].dtype) * v["tex_id"]
                            for n, v in zip(norm, views))
    return out


class Decoder(nn.Module):
    def __init__(self, token_dim=256, dim=512, heads=8, layers=8, patch=8, render_size=256,
                 grad_checkpoint=False, tex_size=0, tex_dim=16, tex_k=8, tex_chunk=4,
                 color_copy=False, corr_bound=0., copy_coverage=False):
        super().__init__()
        # v4.1: blend copy only where the pixel ray actually hits a slot patch
        self.copy_coverage = bool(copy_coverage and color_copy)
        self.corr_bound = corr_bound  # v4: |correction| <= bound (logit units); 0 = free
        self.color_copy = bool(color_copy and tex_size)
        tex_ch = tex_dim + (3 if self.color_copy else 0)
        self.tex_size, self.tex_dim, self.tex_k, self.tex_chunk = tex_size, tex_dim, tex_k, tex_chunk
        self.encoder_ref = []  # set by VISR: slot frames / texel extent live in the encoder
        self.patch, self.render_size, self.heads = patch, render_size, heads
        self.grad_checkpoint = grad_checkpoint
        self.query_in = nn.Linear(fourier_dim(6, 4), dim)
        self.token_in = nn.Linear(token_dim, dim)
        self.token_pos = nn.Linear(fourier_dim(3, 6), dim)
        self.blocks = nn.ModuleList(CrossBlock(dim, heads, dim) for _ in range(layers))
        # Fixed geometric bank: per-head ray-distance scale sigma = 8 mm .. 8 cm
        # (slot spacing is ~1.3 cm), and depth slopes 0..3 per dm (later heads
        # prefer slots nearer to the camera, i.e. occlusion ordering).
        self.register_buffer("log_sigma", torch.linspace(math.log(0.08), math.log(0.8), heads))
        self.register_buffer("depth_slope", torch.linspace(0., 3., heads))
        self.norm = nn.LayerNorm(dim)
        feat = 16
        self.to_pixels = nn.Linear(dim, patch * patch * feat)
        if tex_size:
            self.tex_opacity = nn.Linear(token_dim, 1)
            self.tex_beta = nn.Parameter(torch.tensor(1.87))  # softplus -> 2 per dm
            width = 64
            self.refine = nn.Sequential(
                nn.Conv2d(feat + tex_ch + 1, width, 3, padding=1), nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                nn.Conv2d(width, 5, 3, padding=1))
            if self.color_copy:
                # RGB starts as the copied colour; the network learns a correction.
                with torch.no_grad():
                    self.refine[-1].weight[:3].zero_()
                    self.refine[-1].bias[:3].zero_()
        else:
            self.refine = nn.Sequential(
                nn.Conv2d(feat, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 5, 3, padding=1))

    @exact
    def _bias(self, origin, dirs, p):
        """Ray-slot geometry only: squared perpendicular distance r^2 and depth
        along the ray relative to the ray's closest approach c to the head
        origin. Both are invariant to a global rigid motion applied jointly to
        rays and slots (see tests). Writing them relative to c keeps all
        magnitudes ~1 dm, avoiding float32 cancellation at camera distance."""
        s0 = -(origin[:, None, :] * dirs).sum(-1)                    # [B,M]
        c = origin[:, None, :] + s0[..., None] * dirs                # [B,M,3], c.d = 0
        pd = dirs @ p.transpose(1, 2)                                # [B,M,N] depth vs c
        rel2 = ((p ** 2).sum(-1)[:, None, :] - 2 * (c @ p.transpose(1, 2))
                + (c ** 2).sum(-1)[..., None])                       # |p - c|^2
        r2 = (rel2 - pd ** 2).clamp_min(0)
        sigma2 = torch.exp(2 * self.log_sigma).view(1, -1, 1, 1)
        bias = -r2[:, None] / (2 * sigma2) - self.depth_slope.view(1, -1, 1, 1) * pd[:, None]
        # Relative to each ray's best slot: softmax ignores the shift, and the
        # clamp then stays rigid-invariant even for rays that miss every slot.
        return (bias - bias.amax(-1, keepdim=True)).clamp(min=-30.)

    @exact
    def _candidates(self, origin, dirs, p, extent):
        """K slots per patch ray: close to the ray, preferring the camera side."""
        s0 = -(origin[:, None, :] * dirs).sum(-1)
        c = origin[:, None, :] + s0[..., None] * dirs
        pd = dirs @ p.transpose(1, 2)
        rel2 = ((p ** 2).sum(-1)[:, None, :] - 2 * (c @ p.transpose(1, 2))
                + (c ** 2).sum(-1)[..., None])
        r2 = (rel2 - pd ** 2).clamp_min(0)
        score = -r2 / (2 * (1.5 * extent) ** 2) - 2.0 * pd
        return score.topk(min(self.tex_k, p.shape[1]), dim=-1).indices

    @exact
    def _tex_weights(self, p_k, opa_k, origin, d, t1_k, t2_k, extent, beta):
        """Per pixel ray and candidate slot: local patch coords (u, v) of the
        ray's closest approach and the blending logit. Only relative ray/slot
        geometry enters, so this is invariant to the global frame."""
        rel = p_k - origin[:, None, None, :]                              # [b,S2,K,3]
        sd = (rel * d[:, :, None, :]).sum(-1)                              # depth along ray
        delta = sd[..., None] * d[:, :, None, :] - rel                     # closest point - p
        u = (delta * t1_k).sum(-1) / extent
        v = (delta * t2_k).sum(-1) / extent
        dist2 = (delta ** 2).sum(-1) / extent ** 2
        outside = F.relu(u.abs() - 0.5) ** 2 + F.relu(v.abs() - 0.5) ** 2
        s_front = sd - sd.min(-1, keepdim=True).values.detach()
        logit = -2.0 * dist2 - 20.0 * outside - beta * s_front + opa_k
        # Geometric hit (v4.1): the ray passes through the patch square and close
        # to its plane. Used as the copy coverage instead of the opacity-dependent
        # softmax mass.
        self._hit = (torch.exp(-(20.0 * outside + 4.0 * F.relu(dist2 - 0.25))) *
                     (sd > 0).float()).amax(-1)                            # in front only
        return logit, u, v

    def _read_chunk(self, tex, p, opa, origin, d, idx, t1, t2, extent):
        b, n, tsz, c = tex.shape
        m, kk = idx.shape[1], idx.shape[2]
        flat = idx.reshape(b, -1)
        p_k = torch.gather(p.float(), 1, flat[..., None].expand(-1, -1, 3)).view(b, m, kk, 3)
        opa_k = torch.gather(opa, 1, flat).view(b, m, kk)
        beta = F.softplus(self.tex_beta)
        logit, u, v = self._tex_weights(p_k, opa_k, origin, d, t1[idx], t2[idx], extent, beta)
        # A non-finite coordinate cast to long becomes a wild index (CUDA assert):
        # sanitise here; the step itself is skipped later if its gradients are not finite.
        u = torch.nan_to_num(u, nan=0., posinf=1., neginf=-1.)
        v = torch.nan_to_num(v, nan=0., posinf=1., neginf=-1.)
        logit = torch.nan_to_num(logit, nan=-1e4, posinf=1e4, neginf=-1e4)
        w = torch.softmax(logit, -1)
        coverage = (self._hit if self.copy_coverage
                    else torch.tanh(torch.logsumexp(logit, -1) / 4))
        size = self.tex_size
        gx = ((u + 0.5) * size - 0.5).clamp(0, size - 1)
        gy = ((v + 0.5) * size - 0.5).clamp(0, size - 1)
        x0 = gx.floor().clamp(max=size - 2)
        y0 = gy.floor().clamp(max=size - 2)
        fx, fy = gx - x0, gy - y0
        base = idx * tsz + y0.long() * size + x0.long()                    # [b,m,K]
        src = tex.reshape(b, n * tsz, c)
        out = 0.
        for dy, dx, cw in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                           (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            ind = (base + dy * size + dx).reshape(b, -1).clamp(0, n * tsz - 1)
            g = torch.gather(src, 1, ind[..., None].expand(-1, -1, c)).view(b, m, kk, c)
            out = out + (w * cw).to(g.dtype)[..., None] * g
        return torch.cat((out.sum(2), coverage[..., None].to(out.dtype)), -1)

    def _texture(self, z, origin, dirs, r_q, t_q, k):
        enc = self.encoder_ref[0]
        size, patch = self.render_size, self.patch
        t1, t2, _ = slot_frames(enc.p0.detach().float())
        extent = enc.tex_extent.float()
        with torch.no_grad():
            idx = self._candidates(origin, dirs, z["p"].detach(), extent)  # [b,M,K]
        _, d = rays_in_head_frame(r_q, t_q, k, size, size, 1)              # [b,S2,3]
        g = size // patch
        rows = torch.arange(size, device=d.device) // patch
        pm = (rows[:, None] * g + rows[None, :]).reshape(-1)
        idx = idx[:, pm]                                                    # [b,S2,K]
        opa = self.tex_opacity(z["h"]).float()[..., 0]
        outs = []
        for i in range(0, k.shape[0], self.tex_chunk):
            sl = slice(i, i + self.tex_chunk)
            args = (z["tex"][sl], z["p"][sl], opa[sl], origin[sl].float(), d[sl], idx[sl],
                    t1, t2, extent)
            if self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self._read_chunk, *args, use_reentrant=False))
            else:
                outs.append(self._read_chunk(*args))
        feat = torch.cat(outs, 0)                                           # [b,S2,C+1]
        return feat.transpose(1, 2).reshape(k.shape[0], -1, size, size)

    def forward(self, z, r_q, t_q, k):
        """z: dict p [B,N,3], h [B,N,C] (+ tex [B,N,P*P,C_t]) canonical; target Q, K."""
        size, patch = self.render_size, self.patch
        b = k.shape[0]
        origin, dirs = rays_in_head_frame(r_q, t_q, k, size, size, patch)
        q = self.query_in(fourier(plucker(origin, dirs), 4, 4))
        ctx = self.token_in(z["h"]) + self.token_pos(fourier(z["p"].float(), 6, 4).to(z["h"].dtype))
        bias = amp_cast(self._bias(origin.float(), dirs.float(), z["p"].float()))
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                q = checkpoint(block, q, ctx, bias, use_reentrant=False)
            else:
                q = block(q, ctx, bias)
        y = self.to_pixels(self.norm(q))                              # [B, M, p*p*f]
        g = size // patch
        y = y.view(b, g, g, patch, patch, -1).permute(0, 5, 1, 3, 2, 4).reshape(b, -1, size, size)
        copy = None
        if self.tex_size:
            tex = self._texture(z, origin, dirs, r_q, t_q, k)
            if self.color_copy:
                c = self.encoder_ref[0].tex_dim
                copy = tex[:, c:c + 3].float().clamp(2e-3, 1 - 2e-3)
            y = torch.cat((y, tex.to(y.dtype)), 1)
        y = self.refine(y).float()
        _, full_dirs = rays_in_head_frame(r_q, t_q, k, size, size, 1)
        s0 = -(origin[:, None, :].float() * full_dirs).sum(-1).view(b, 1, size, size)
        if copy is not None and getattr(self, "copy_only", False):
            y = y.clone()
            y[:, :3] = 0  # diagnostic: render the transported colour alone
        corr = y[:, :3]
        if copy is not None and self.corr_bound > 0:
            corr = self.corr_bound * torch.tanh(corr / self.corr_bound)
        logit = corr if copy is None else torch.logit(copy) + corr
        if copy is not None and self.copy_coverage:
            # Off the surface (background, hair edges) the copied colour of the
            # nearest patch is meaningless: use the free (unbounded) prediction.
            hit = tex[:, -1:].float()
            logit = hit * logit + (1 - hit) * y[:, :3]
        return {"rgb": torch.sigmoid(logit), "alpha": torch.sigmoid(y[:, 3:4]),
                "alpha_logit": y[:, 3:4], "depth": s0 + y[:, 4:5]}


class VISR(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        enc_keys = ("backbone", "backbone_config", "pretrained", "feature_layers",
                    "freeze_blocks", "slots", "dim", "token_dim", "heads", "blocks",
                    "dyn_dim", "image_size", "tex_size", "tex_dim", "color_copy", "pose_iters", "copy_hard",
                    "vis_tol")
        dec_keys = ("token_dim", "dim", "heads", "decoder_layers", "patch", "render_size",
                    "grad_checkpoint", "tex_size", "tex_dim", "tex_k", "tex_chunk", "color_copy",
                    "corr_bound", "copy_coverage")
        self.encoder = Encoder(**{k: v for k, v in kw.items() if k in enc_keys})
        dec = {k: v for k, v in kw.items() if k in dec_keys}
        if "decoder_layers" in dec:
            dec["layers"] = dec.pop("decoder_layers")
        self.decoder = Decoder(**dec)
        self.decoder.encoder_ref = [self.encoder]  # plain list: not a second submodule


@exact
def relative_extrinsic(r_t, t_t, r_s, t_s):
    """E_t o E_s^-1 : source camera -> target camera."""
    r_inv, t_inv = invert(r_s, t_s)
    return (r_t @ r_inv, (r_t @ t_inv[..., None])[..., 0] + t_t)


__all__ = ["VISR", "Encoder", "Decoder", "fuse", "relative_extrinsic", "matrix_to_rot6d"]
