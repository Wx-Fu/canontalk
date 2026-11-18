"""CPU tests (no OmniAvatar weights needed): VISR conditioner + generator plumbing.
Run from tmm/:  python -m pytest -q train_dynamic_claude/tests"""
import os

import torch
from torch import nn

from train_dynamic_claude import omni

TINY_VISR = os.environ.get("TINY_VISR", "/tmp/tsc/v41a/best.pt")


class FakePipe:
    """Wan VAE stand-in with the same causal latent grid (T -> 1+(T-1)/4, /8)."""
    device = torch.device("cpu")

    def encode_video(self, x, tiled=False):
        b, c, t, h, w = x.shape
        l = 1 + (t - 1) // 4
        pooled = torch.nn.functional.adaptive_avg_pool3d(x.float(), (l, h // 8, w // 8))
        return pooled.repeat(1, 6, 1, 1, 1)[:, :16].to(torch.bfloat16)


class FakeDiT(nn.Module):
    """Takes cat([x, y]) like WanModel and predicts a velocity."""

    def __init__(self, in_ch=51):
        super().__init__()
        self.patch_embedding = nn.Conv3d(in_ch, 16, 1)

    def forward(self, x, timestep, context, y, audio_emb, use_gradient_checkpointing=False):
        return self.patch_embedding(torch.cat([x, y], 1).float())


def test_extend_patch_embedding_preserves_pretrained_output():
    dit = FakeDiT(33)
    x = torch.randn(1, 33, 3, 4, 4)
    before = dit.patch_embedding(x)
    omni.extend_patch_embedding(dit, omni.EXTRA)
    after = dit.patch_embedding(torch.cat([x, torch.randn(1, omni.EXTRA, 3, 4, 4)], 1))
    assert dit.patch_embedding.in_channels == 51
    assert torch.allclose(before, after, atol=1e-6)


def test_latent_pool_matches_wan_causal_grid():
    x = torch.zeros(1, 1, 9, 16, 16)
    x[:, :, 0] = 1.          # frame 0 alone -> latent 0
    x[:, :, 5:9] = 2.        # frames 5..8 -> latent 2
    p = omni.latent_pool(x, 3)
    assert p.shape == (1, 1, 3, 2, 2)
    assert torch.allclose(p[:, :, 0], torch.ones(1)) and torch.allclose(p[:, :, 1], torch.zeros(1))
    assert torch.allclose(p[:, :, 2], torch.full((1,), 2.))


def test_build_y_and_flow_loss_with_prefix():
    pipe = FakePipe()
    ref = torch.rand(2, 3, 64, 64)
    video = torch.rand(2, 3, 9, 64, 64)
    render, alpha, depth = torch.rand_like(video), torch.rand(2, 1, 9, 64, 64), torch.rand(2, 1, 9, 64, 64)
    target, y = omni.encode_conditions(pipe, ref, video, render, alpha, depth)
    assert target.shape == (2, 16, 3, 8, 8) and y.shape == (2, 35, 3, 8, 8)
    assert torch.all(y[:, 16] == 1)                       # every frame generated
    dit = FakeDiT()
    loss = omni.flow_loss(dit, target.float(), y.float(), torch.zeros(2, 9, 8), torch.zeros(1, 4, 8),
                          prefix=torch.tensor([True, False]), checkpoint=False)
    loss.backward()
    assert torch.isfinite(loss) and dit.patch_embedding.weight.grad is not None


def test_sampler_keeps_prefix_latent():
    from OmniAvatar.schedulers.flow_match import FlowMatchScheduler

    class P(FakePipe):
        scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
    pipe = P()
    pipe.dit = FakeDiT()
    y = torch.rand(1, 35, 3, 8, 8)
    prefix = torch.full((1, 16, 1, 8, 8), 0.7)
    orig = torch.autocast
    torch.autocast = lambda *a, **k: orig("cpu", enabled=False)
    try:
        lat = omni.sample(pipe, y, torch.zeros(1, 9, 8), torch.zeros(1, 4, 8), steps=3,
                          cfg_audio=2.0, prefix_latent=prefix)
    finally:
        torch.autocast = orig
    assert lat.shape == (1, 16, 3, 8, 8) and torch.allclose(lat[:, :, :1].float(), prefix, atol=1e-2)


def test_visr_condition_shapes_and_static_expression():
    if not os.path.isfile(TINY_VISR):
        return
    from train_dynamic_claude.visr_cond import VISRCondition
    visr = VISRCondition(TINY_VISR, torch.device("cpu"), chunk=4)
    ref = torch.rand(1, 3, 64, 64)
    video = torch.rand(1, 3, 5, 64, 64)
    k = torch.tensor([[[300., 0, 32], [0, 300., 32], [0, 0, 1]]])
    rgb, alpha, depth, _, _ = visr(ref, k, video, k, static=torch.tensor([False]))
    assert rgb.shape == (1, 3, 5, 64, 64) and alpha.shape == depth.shape == (1, 1, 5, 64, 64)
    assert torch.isfinite(rgb).all() and depth.abs().max() <= 1
    st = visr.reference(ref, k)
    r, t, e = visr.motion(video, k)
    assert torch.allclose(r @ r.transpose(-1, -2), torch.eye(3).expand_as(r), atol=1e-4)
    a, _, _ = visr.render(st, r, t, st["e"].float()[:, None].expand_as(e), k, 64)
    b, _, _, _, _ = visr(ref, k, video, k, static=torch.tensor([True]))
    assert torch.allclose(a, b, atol=1e-4)


# ------------------------------------------------------------------ v2
def test_v2_candidates_mouth_blur_and_degrade():
    if not os.path.isfile(TINY_VISR):
        return
    from train_dynamic_claude.visr_cond import VISRCondition
    visr = VISRCondition(TINY_VISR, torch.device("cpu"), chunk=4)
    refs = torch.rand(2, 3, 3, 64, 64)                      # 3 candidates per sample
    video = torch.rand(2, 3, 5, 64, 64)
    k = torch.tensor([[300., 0, 32], [0, 300., 32], [0, 0, 1]]).expand(2, 3, 3)
    yes, no = torch.tensor([True, True]), torch.tensor([False, False])
    rgb, a, d, ref, mouth = visr(refs, k, video, k, no, yes, yes, yes)
    assert ref.shape == (2, 3, 64, 64) and rgb.shape == (2, 3, 5, 64, 64)
    assert mouth.shape == (2, 1, 5, 64, 64) and torch.isfinite(rgb).all()
    rgb0, *_ = visr(refs, k, video, k, no, no, no, no)
    assert not torch.allclose(rgb, rgb0)                    # augmentations changed the render


def test_mouth_mask_lands_on_projected_mouth():
    from train_dynamic_claude.visr_cond import MOUTH, mouth_mask
    r = torch.diag(torch.tensor([1., -1., -1.]))[None, None]
    t = torch.tensor([[[0., 0., 9.]]])
    k = torch.tensor([[[400., 0, 64], [0, 400., 64], [0, 0, 1]]])
    m = mouth_mask(r, t, k, 128)
    x = r[0, 0] @ torch.tensor(MOUTH[:3]) + t[0, 0]
    u, v = (k[0] @ x)[:2] / x[2]
    assert m[0, 0, 0, int(v), int(u)] > 0.9 and m[0, 0, 0, 2, 2] < 1e-3
    back = torch.diag(torch.tensor([-1., -1., 1.]))[None, None]
    assert mouth_mask(back, t, k, 128).max() < 1e-6


def test_ava_pairs_sample():
    root, packs = "data/ava256_rgb_4TB", "/tmp/tsc/ava_packs"
    if not os.path.isdir(packs):
        return
    from train_dynamic_claude.data import AvaPairs, collate
    ds = AvaPairs(root, packs, "train", size=64, ref_cands=3)
    if not ds.caps:
        return
    b = collate([ds[0], ds[1]])
    assert b["ref"].shape == (2, 3, 3, 64, 64) and b["video"].shape == (2, 3, 1, 64, 64)
    assert b["K_ref"].shape == (2, 3, 3) and b["wave"].shape[0] == 2


def test_single_frame_latents():
    pipe = FakePipe()
    video = torch.rand(2, 3, 1, 64, 64)
    target, y = omni.encode_conditions(pipe, torch.rand(2, 3, 64, 64), video, torch.rand_like(video),
                                       torch.rand(2, 1, 1, 64, 64), torch.rand(2, 1, 1, 64, 64))
    assert target.shape == (2, 16, 1, 8, 8) and y.shape == (2, 35, 1, 8, 8)
