"""Frozen VISR as the geometry/expression conditioner of the generator.

ref image  -> z_id, Q_ref, e_ref        (identity slots, head pose, expression)
frame i    -> Q_i, e_i                  (training: read from the GT frame)
render_i   =  VISR.decode(compose(z_id, e_i), Q_i, K)  -> rgb (on gray), alpha, depth
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_static_claude.losses import Masker, neutral_background  # noqa: E402
from train_static_claude.train import build_from_checkpoint  # noqa: E402

VISR_SIZE = 448


def scale_k(k, src, dst):
    k = k.clone().float()
    k[..., :2, :] *= dst / src
    return k


def smooth(x, width=3):
    """Temporal moving average over dim 1 (reduces per-frame estimation jitter)."""
    if width <= 1 or x.shape[1] < width:
        return x
    pad = width // 2
    xp = torch.cat([x[:, :1].expand(-1, pad, *x.shape[2:]), x,
                    x[:, -1:].expand(-1, pad, *x.shape[2:])], 1)
    return torch.stack([xp[:, i:i + x.shape[1]] for i in range(width)]).mean(0)


# Ava head frame (VISR canonical frame): mouth centre and radius in dm.
MOUTH = (0.0, -0.42, 0.46, 0.35)


def mouth_mask(r, t, k, size):
    """[B,1,T,S,S] soft mask around the projected mouth, faded when the face turns away."""
    b, n = r.shape[:2]
    p = r.new_tensor(MOUTH[:3])
    x = (r @ p) + t                                                   # [B,T,3]
    uv = x @ k.transpose(-1, -2)                                      # [B,T,3]
    uv = uv[..., :2] / uv[..., 2:].clamp_min(1e-3)
    sigma = k[:, None, 0, 0] * MOUTH[3] / x[..., 2].clamp_min(1e-3) / 2
    facing = (-(r[..., :, 2] * F.normalize(t, dim=-1)).sum(-1) * 2).clamp(0, 1)
    g = torch.arange(size, device=r.device, dtype=torch.float32) + 0.5
    dy = (g[None, None, :] - uv[..., 1:2]) ** 2
    dx = (g[None, None, :] - uv[..., 0:1]) ** 2
    m = torch.exp(-(dy[..., :, None] + dx[..., None, :]) / (2 * sigma[..., None, None] ** 2))
    return (m * facing[..., None, None])[:, None]


def blur_video(x, sigma):
    """Separable Gaussian blur of [B,C,T,H,W] (per frame); sigma in pixels (float or [B])."""
    b, c, t, h, w = x.shape
    s = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1)
    s = s.expand(b) if s.numel() == 1 else s
    r = int(torch.ceil(3 * s.max()).item())
    if r < 1:
        return x
    k = torch.exp(-(torch.arange(-r, r + 1, device=x.device).float()[None] ** 2) /
                  (2 * s[:, None].clamp_min(1e-3) ** 2))
    k = k / k.sum(1, keepdim=True)                                    # [B,2r+1]
    y = x.permute(0, 2, 1, 3, 4).reshape(1, b * t * c, h, w)
    kk = k.repeat_interleave(t * c, 0)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), kk[:, None, None, :], groups=b * t * c)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), kk[:, None, :, None], groups=b * t * c)
    return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def degrade_condition(render, alpha, depth, which):
    """Treat the VISR render as coarse guidance: random blur (0-3 px), down/up-sampling
    (1-4x), silhouette erosion/dilation (+-3 px), colour noise. Only rows in `which`."""
    b, _, t, h, w = render.shape
    dev = render.device
    sel = which.view(-1, 1, 1, 1, 1).float()
    sig = torch.rand(b, device=dev) * 3
    out = blur_video(render, sig)
    f = int(torch.randint(1, 5, (1,)).item())
    if f > 1:
        flat = out.flatten(1, 2)
        flat = F.interpolate(F.interpolate(flat, scale_factor=1 / f, mode="area"),
                             size=(h, w), mode="bilinear", align_corners=False)
        out = flat.view_as(render)
    out = (out + 0.03 * torch.randn_like(out)).clamp(0, 1)
    rad = int(torch.randint(1, 4, (1,)).item())
    a = alpha.flatten(1, 2)
    a = F.max_pool2d(a, 2 * rad + 1, 1, rad) if torch.rand(1).item() < 0.5 else \
        -F.max_pool2d(-a, 2 * rad + 1, 1, rad)
    a = a.view_as(alpha)
    d = blur_video(depth, sig)
    return (render * (1 - sel) + out * sel, alpha * (1 - sel) + a * sel,
            depth * (1 - sel) + d * sel)


class VISRCondition(nn.Module):
    def __init__(self, checkpoint, device, chunk=8):
        super().__init__()
        net, saved = build_from_checkpoint(checkpoint, device)
        self.net = net.eval().requires_grad_(False)
        self.render_size = saved.render_size
        self.masker = Masker(size=384).to(device).eval()
        self.chunk = chunk

    def _encode(self, images, k):
        """images [N,3,S,S] in [0,1], k at size S -> encoder outputs (chunked)."""
        s = images.shape[-1]
        x = F.interpolate(images.float(), size=(VISR_SIZE, VISR_SIZE), mode="bilinear",
                          align_corners=False, antialias=True)
        x = neutral_background(self.masker, x)
        k = scale_k(k, s, VISR_SIZE)
        outs = [self.net.encoder(x[i:i + self.chunk], k[i:i + self.chunk])
                for i in range(0, x.shape[0], self.chunk)]
        return {key: torch.cat([o[key] for o in outs]) for key in outs[0]
                if torch.is_tensor(outs[0][key])}

    def reference(self, ref, k_ref):
        """Identity slots and expression of the reference image."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._encode(ref, k_ref)

    def motion(self, video, k_tgt):
        """Per-frame head pose (head->target camera) and expression from frames.
        video [B,3,T,S,S] -> R [B,T,3,3], t [B,T,3], e [B,T,E]."""
        b, _, t, s, _ = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, 3, s, s)
        k = k_tgt[:, None].expand(b, t, 3, 3).reshape(b * t, 3, 3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = self._encode(frames, k)
        r = o["R"].float().view(b, t, 3, 3)
        tr = smooth(o["t"].float().view(b, t, 3))
        # re-orthonormalise the smoothed rotation (SVD projection)
        u, _, vh = torch.linalg.svd(smooth(r.reshape(b, t, 9)).view(b, t, 3, 3))
        r = u @ vh
        return r, tr, smooth(o["e"].float().view(b, t, -1))

    def render(self, ref_state, r, t, e, k_tgt, size_out):
        """Render frames for every (R_i, t_i, e_i). Returns rgb [B,3,T,H,W] in [0,1] on
        gray, alpha [B,1,T,H,W], depth [B,1,T,H,W] relative to the head origin (dm/1.5)."""
        b, n = r.shape[:2]
        rs = self.render_size
        k = scale_k(k_tgt, size_out, rs)[:, None].expand(b, n, 3, 3).reshape(b * n, 3, 3)
        idx = torch.arange(b, device=r.device).repeat_interleave(n)
        enc = self.net.encoder
        rgb, alpha, depth = [], [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, b * n, self.chunk):
                sl = slice(i, i + self.chunk)
                state = {key: v[idx[sl]] for key, v in ref_state.items()
                         if key in ("dp_id", "h_id", "tex_id")}
                z = enc.compose(state, e.reshape(b * n, -1)[sl].to(ref_state["h_id"].dtype))
                out = self.net.decoder(z, r.reshape(b * n, 3, 3)[sl], t.reshape(b * n, 3)[sl], k[sl])
                a = out["alpha"].float()
                rgb.append(out["rgb"].float() * a + 0.5 * (1 - a))
                alpha.append(a)
                dist = t.reshape(b * n, 3)[sl].norm(dim=-1).view(-1, 1, 1, 1)
                depth.append(((out["depth"].float() - dist) / 1.5).clamp(-1, 1) * a)
        def video(xs):
            x = torch.cat(xs).view(b, n, -1, rs, rs).permute(0, 2, 1, 3, 4)
            if size_out != rs:
                x = F.interpolate(x.flatten(1, 2), size=(size_out, size_out), mode="bilinear",
                                  align_corners=False).view(b, -1, n, size_out, size_out)
            return x
        return video(rgb), video(alpha), video(depth)

    @torch.no_grad()
    def forward(self, ref, k_ref, video, k_tgt, static=None, pick_gap=None, mouth_blur=None,
                degrade=None):
        """Training-time condition: motion read from the GT target frames.
        ref [B,C,3,S,S] candidates (or [B,3,S,S]); pick_gap [B] bool: choose the
        candidate with the largest head-rotation gap to the target (else candidate 0);
        static [B] bool: reference expression for every frame;
        mouth_blur [B] bool: blur the projected mouth region of the render;
        degrade [B] bool: random blur / resolution / silhouette / noise corruption.
        Returns render, alpha, depth, the chosen reference [B,3,S,S] and the mouth mask."""
        size = video.shape[-1]
        r, t, e = self.motion(video, k_tgt)
        if ref.dim() == 5:
            b, c = ref.shape[:2]
            choice = torch.zeros(b, dtype=torch.long, device=ref.device)
            if pick_gap is not None and pick_gap.any() and c > 1:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cand = self._encode(ref.flatten(0, 1), k_ref.repeat_interleave(c, 0))
                rc = cand["R"].float().view(b, c, 3, 3)
                mid = r[:, r.shape[1] // 2]
                cos = ((rc.transpose(-1, -2) @ mid[:, None]).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
                choice = torch.where(pick_gap, cos.argmin(1), choice)  # largest angle
            ref = ref[torch.arange(b, device=ref.device), choice]
        ref_state = self.reference(ref, k_ref)
        if static is not None and static.any():
            e = torch.where(static.view(-1, 1, 1), ref_state["e"].float()[:, None].expand_as(e), e)
        render, alpha, depth = self.render(ref_state, r, t, e, k_tgt, size)
        mouth = mouth_mask(r, t, k_tgt, size)
        if mouth_blur is not None and mouth_blur.any():
            blurred = blur_video(render, 0.06 * size)
            m = mouth * mouth_blur.view(-1, 1, 1, 1, 1).float()
            render = render * (1 - m) + blurred * m
        if degrade is not None and degrade.any():
            render, alpha, depth = degrade_condition(render, alpha, depth, degrade)
        return render, alpha, depth, ref, mouth
