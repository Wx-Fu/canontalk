"""Stage B: generative completion *in the invariant space* (P4).

A single view determines only part of the head. Instead of sampling in image
space per view and per frame (inconsistent), we learn the conditional
distribution of the *identity slots* given the single-view slots:

    p( z_id^multi | z_id^single )      z_id = (dp_id, h_id) per canonical slot

Target ``z_id^multi`` = precision-weighted fusion of 5-7 synchronized views
from the frozen stage-A encoder (the best available "observed" identity).
Model: rectified-flow transformer over the slot set, conditioned per slot on
the single-view slot, its precision, and the slot embedding. One sample is
drawn per identity and then rendered at every view and frame with the frozen
deterministic decoder, so the hallucinated side is shared by all views.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .model import mlp


class AdaLNBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = mlp(dim, 4 * dim, dim)
        self.mod = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.mod(c)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.ff(h)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * 1000 * freqs[None]
    return torch.cat((torch.cos(a), torch.sin(a)), -1)


class SlotPrior(nn.Module):
    def __init__(self, slots, token_dim, slot_dim, dim=512, heads=8, layers=8, cond_drop=0.1,
                 tex_shape=None):
        super().__init__()
        self.token_dim, self.cond_drop = token_dim, cond_drop
        self.tex_shape = tuple(tex_shape) if tex_shape else None   # (P*P, C_t) or None
        x_dim = token_dim + 3 + (self.tex_shape[0] * self.tex_shape[1] if self.tex_shape else 0)
        self.x_in = nn.Linear(x_dim, dim)
        self.c_in = nn.Linear(x_dim + 1 + slot_dim, dim)
        self.t_in = mlp(256, dim, dim)
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out = nn.Linear(dim, x_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # per-channel normalization of (h_id, dp_id), estimated from data
        self.register_buffer("mean", torch.zeros(x_dim))
        self.register_buffer("std", torch.ones(x_dim))
        self.register_buffer("stats_ready", torch.tensor(False))

    # -------------------------------------------------------------- packing
    def _raw(self, z):
        parts = [z["h_id"].float(), z["dp_id"].float()]
        if self.tex_shape:
            parts.append(z["tex_id"].float().flatten(2))
        return torch.cat(parts, -1)

    def pack(self, z):
        return (self._raw(z) - self.mean) / self.std

    def unpack(self, x):
        x = x * self.std + self.mean
        c = self.token_dim
        out = {"h_id": x[..., :c], "dp_id": x[..., c:c + 3]}
        if self.tex_shape:
            out["tex_id"] = x[..., c + 3:].reshape(*x.shape[:2], *self.tex_shape)
        return out

    @torch.no_grad()
    def update_stats(self, z):
        x = self._raw(z).flatten(0, 1)
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-3))
        self.stats_ready.fill_(True)

    # -------------------------------------------------------------- network
    def velocity(self, x_t, t, cond, conf, slot_embed, drop=None):
        c = torch.cat((cond, torch.sigmoid(conf)[..., None].float(),
                       slot_embed[None].expand(x_t.shape[0], -1, -1).float()), -1)
        if drop is not None:  # classifier-free guidance training
            c = c * (1 - drop.view(-1, 1, 1).float())
        h = self.x_in(x_t) + self.c_in(c)
        temb = self.t_in(timestep_embedding(t, 256))
        for block in self.blocks:
            h = block(h, temb)
        return self.out(self.out_norm(h))

    def forward(self, z_single, z_multi, slot_embed):
        return self.loss(z_single, z_multi, slot_embed)

    def loss(self, z_single, z_multi, slot_embed):
        x1 = self.pack(z_multi)
        cond = self.pack(z_single)
        b = x1.shape[0]
        t = torch.rand(b, device=x1.device)
        x0 = torch.randn_like(x1)
        x_t = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
        drop = (torch.rand(b, device=x1.device) < self.cond_drop)
        v = self.velocity(x_t, t, cond, z_single["conf"], slot_embed, drop)
        loss = (v - (x1 - x0)).pow(2).mean((1, 2))
        x1_hat = x_t + (1 - t.view(-1, 1, 1)) * v
        return loss, self.unpack(x1_hat), t

    @torch.no_grad()
    def sample(self, z_single, slot_embed, steps=32, guidance=1.5, generator=None):
        cond = self.pack(z_single)
        x = torch.randn(cond.shape, device=cond.device, generator=generator)
        b = x.shape[0]
        for i in range(steps):
            t = torch.full((b,), i / steps, device=x.device)
            v = self.velocity(x, t, cond, z_single["conf"], slot_embed)
            if guidance != 1:
                v0 = self.velocity(x, t, cond, z_single["conf"], slot_embed,
                                   torch.ones(b, device=x.device))
                v = v0 + guidance * (v - v0)
            x = x + v / steps
        return self.unpack(x)
