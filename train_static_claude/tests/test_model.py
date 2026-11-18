"""Tiny-model CPU tests: shapes, gradient coverage (DDP-critical), structure."""
import argparse

import torch
import torch.nn.functional as F

from train_static_claude import losses as L
from train_static_claude.train import Forward, Step, model_kwargs, parse
from train_static_claude.model import VISR


def tiny_args():
    return parse(["--out", "/tmp/unused", "--tiny", "--no-pretrained", "--enc-size", "56",
                  "--render-size", "32", "--slots", "32", "--dim", "32", "--token-dim", "16",
                  "--heads", "4", "--enc-blocks", "1", "1", "--decoder-layers", "1",
                  "--perceptual", "none", "--masker", "threshold", "--device", "cpu"])


def fake_views(b, v, size, dist=9.0):
    k = torch.tensor([[4.8 * size, 0, size / 2], [0, 4.8 * size, size / 2], [0, 0, 1.]])
    q_r = torch.diag(torch.tensor([1., -1., -1.]))
    return {"image": torch.rand(b, v, 3, size, size), "valid": torch.ones(b, v, 1, size, size),
            "K": k.expand(b, v, 3, 3).clone(), "E_R": torch.eye(3).expand(b, v, 3, 3).clone(),
            "E_t": torch.zeros(b, v, 3), "Q_R": q_r.expand(b, v, 3, 3).clone(),
            "Q_t": torch.tensor([0., 0., dist]).expand(b, v, 3).clone()}


def fake_batch(b=2):
    return {"enc": fake_views(b, 3, 56), "tgt": fake_views(b, 5, 32), "tgt2": fake_views(b, 2, 32),
            "verts": torch.randn(b, 2, 100, 3) * 0.5, "has_pose": torch.tensor([1., 0.]), "person": ["p0", "p1"],
            "domain": torch.tensor([0, 1]), "angles": torch.rand(b, 5) * 90}


def test_forward_backward_and_every_trainable_param_gets_grad():
    args = tiny_args()
    net = VISR(**model_kwargs(args))
    step = Step(Forward(net, tuple(args.w_groups)), L.ThresholdMasker(), L.Perceptual("none"), args)
    loss, logs, extra = step(fake_batch(), step=100000)
    assert torch.isfinite(loss)
    assert extra["pred"]["rgb"].shape == (2 * 10, 3, 32, 32)
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_decoder_does_not_take_source_image():
    import inspect
    from train_static_claude.model import Decoder
    params = list(inspect.signature(Decoder.forward).parameters)
    assert params == ["self", "z", "r_q", "t_q", "k"]


def test_bf16_autocast_full_step_keeps_geometry_in_float32():
    """Same dtype path as CUDA training (bf16 autocast), including backward."""
    args = tiny_args()
    net = VISR(**model_kwargs(args))
    step = Step(Forward(net, tuple(args.w_groups)), L.ThresholdMasker(), L.Perceptual("none"), args)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, logs, extra = step(fake_batch(), step=0)
        s = extra["views"][0]
        for name in ("R", "t", "R_used", "t_used", "dp_id"):
            assert s[name].dtype == torch.float32, name
        assert extra["pred"]["depth"].dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert not [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]


def test_projection_is_exact_under_autocast():
    from train_static_claude import geometry as G
    x = torch.tensor([[[0.123456, -0.654321, 9.87654]]])
    k = torch.tensor([[[833.3, 0, 224.], [0, 833.3, 224.], [0, 0, 1.]]])
    plain, _ = G.project(x, k)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp, _ = G.project(x, k)
    assert amp.dtype == torch.float32
    assert (plain - amp).abs().max() < 1e-4


def test_identity_nce_punishes_collapse():
    b, n, c = 8, 32, 16
    ids = torch.arange(b)
    idx = torch.arange(n)
    conf = torch.zeros(b, n)
    people = torch.randn(b, n, c)
    good = L.identity_nce(people, conf, people + 0.05 * torch.randn(b, n, c), conf, ids, ids, idx)
    same = torch.randn(1, n, c).expand(b, -1, -1)
    bad = L.identity_nce(same, conf, same, conf, ids, ids, idx)
    assert good[0] < 0.1 < bad[0] and good[1] == 1.0
    assert abs(bad[0] - torch.log(torch.tensor(float(b)))) < 0.01
    assert abs(good[2] - 1 / b) < 1e-6


def test_same_person_is_never_a_negative():
    n, c = 16, 8
    conf = torch.zeros(2, n)
    h = torch.randn(1, n, c).expand(2, -1, -1) + 0.01 * torch.randn(2, n, c)
    other = torch.randn(1, n, c)
    cands = torch.cat((h, other))
    ids_c = torch.tensor([7, 7, 9])
    cc = torch.zeros(3, n)
    loss, acc, _ = L.identity_nce(h, conf, cands, cc, torch.tensor([7, 7]), ids_c,
                                  torch.arange(n))
    assert acc == 1.0 and loss < 0.8


def test_variance_hinge_punishes_shrinking():
    big, std = L.variance_hinge(torch.randn(64, 32) * 2)
    small, _ = L.variance_hinge(torch.randn(64, 32) * 1e-3)
    assert big < 0.05 and small > 0.9 and std > 1.5


def test_gather_passes_contiguous_tensors_like_nccl_requires():
    from unittest import mock
    x = torch.rand(8, 3, 16, 4).requires_grad_()[:, 1]  # same slicing as split_views
    assert not x.is_contiguous()

    def fake_all_gather(parts, tensor):
        assert tensor.is_contiguous() and all(p.is_contiguous() for p in parts)
        for p in parts:
            p.copy_(tensor)

    with mock.patch.object(L.dist, "is_initialized", return_value=True), \
            mock.patch.object(L.dist, "get_world_size", return_value=2), \
            mock.patch.object(L.dist, "get_rank", return_value=0), \
            mock.patch.object(L.dist, "all_gather", fake_all_gather):
        out = L.gather(x)
    assert out.shape == (16, 16, 4)
    out.sum().backward()  # gradient flows through the local part


# ----------------------------------------------------------------------- v2
def _readout_decoder(tex_size=4, tex_dim=3, k=1):
    from train_static_claude.model import Decoder
    return Decoder(token_dim=8, dim=16, heads=2, layers=1, patch=8, render_size=32,
                   tex_size=tex_size, tex_dim=tex_dim, tex_k=k)


def test_texture_readout_returns_the_texel_the_ray_hits():
    """One slot at the origin with tangent plane z=0; a ray along +z through
    the center of texel (row 1, col 2) must read exactly that texel."""
    dec = _readout_decoder()
    size = 4
    tex = torch.zeros(1, 1, size * size, 3)
    tex[0, 0, 1 * size + 2] = torch.tensor([1., 2., 3.])
    u = (2 + 0.5) / size - 0.5
    v = (1 + 0.5) / size - 0.5
    origin = torch.tensor([[u, v, -5.]])
    d = torch.tensor([[[0., 0., 1.]]])
    idx = torch.zeros(1, 1, 1, dtype=torch.long)
    out = dec._read_chunk(tex, torch.zeros(1, 1, 3), torch.zeros(1, 1), origin, d, idx,
                          torch.tensor([[1., 0, 0]]), torch.tensor([[0., 1, 0]]),
                          torch.tensor([1.]))
    assert torch.allclose(out[0, 0, :3], torch.tensor([1., 2., 3.]), atol=1e-5), out


def test_texture_weights_invariant_to_global_rigid_motion():
    dec = _readout_decoder(k=6)
    g = torch.Generator().manual_seed(1)
    p = torch.randn(1, 7, 6, 3, generator=g) * 0.3
    t1 = F.normalize(torch.randn(1, 7, 6, 3, generator=g), dim=-1)
    t2 = F.normalize(torch.cross(t1, torch.randn(1, 7, 6, 3, generator=g), dim=-1), dim=-1)
    origin = torch.tensor([[0.2, -0.1, -9.]])
    d = F.normalize(torch.randn(1, 7, 3, generator=g) * 0.05 + torch.tensor([0., 0, 1.]), dim=-1)
    opa = torch.randn(1, 7, 6, generator=g)
    ext, beta = torch.tensor([0.15]), torch.tensor(2.0)
    from train_static_claude.tests.test_geometry import random_rotation
    rot, shift = random_rotation(1, 5)[0], torch.tensor([0.3, -0.7, 0.4])
    move = lambda x: x @ rot.T
    l1, u1, v1 = dec._tex_weights(p, opa, origin, d, t1, t2, ext, beta)
    l2, u2, v2 = dec._tex_weights(move(p) + shift, opa, move(origin) + shift, move(d),
                                  move(t1), move(t2), ext, beta)
    for a, b in ((l1, l2), (u1, u2), (v1, v2)):
        assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_detail_weight_peaks_on_projected_mouth_and_fades_for_back_views():
    k = torch.tensor([[[400., 0, 64], [0, 400., 64], [0, 0, 1]]])
    front = torch.diag(torch.tensor([1., -1., -1.]))[None]      # frontal (FRONTAL_6D)
    t = torch.tensor([[0., 0., 9.]])
    w = L.detail_weight(front, t, k, 128)
    mouth = torch.tensor([[[0.0, -0.42, 0.46]]]) @ front.transpose(1, 2) + t
    uv = (mouth @ k.transpose(1, 2))[0, 0]
    col, row = int(uv[0] / uv[2]), int(uv[1] / uv[2])
    assert w[0, 0, row, col] > 0.9 and w[0, 0, 5, 5] < 1e-3
    back = torch.diag(torch.tensor([-1., -1., 1.]))[None]       # head turned 180 deg
    assert L.detail_weight(back, t, k, 128).max() < 1e-6


def test_v1_checkpoint_args_build_v1_model():
    import argparse
    from train_static_claude.train import model_kwargs
    v1 = vars(tiny_args())
    for key in ("tex_size", "tex_dim", "tex_k", "tex_chunk", "dyn_dim"):
        v1.pop(key)
    kw = model_kwargs(argparse.Namespace(**v1))
    assert kw["tex_size"] == 0 and kw["dyn_dim"] == 64
    net = VISR(**kw)
    assert not hasattr(net.decoder, "tex_opacity")


def test_color_augment_is_consistent_within_a_sample():
    x = torch.rand(2, 1, 3, 8, 8)
    a, b = L.color_augment([x, x.clone()])
    assert torch.allclose(a, b) and not torch.allclose(a, x)


# ----------------------------------------------------------------------- v3
def test_v3_copy_path_renders_the_stored_colour_at_init():
    """With colour copy, a fresh decoder outputs exactly the blended texel colour."""
    from train_static_claude.model import Decoder
    args = tiny_args()
    net = VISR(**model_kwargs(args))
    dec = net.decoder
    assert dec.color_copy and net.encoder.tex_channels == net.encoder.tex_dim + 3
    b, n = 1, net.encoder.p0.shape[0]
    t = net.encoder.tex_size ** 2
    tex = torch.zeros(b, n, t, net.encoder.tex_channels)
    tex[..., -3:] = torch.tensor([0.2, 0.5, 0.8])  # every texel the same colour
    z = {"p": net.encoder.p0[None].detach(), "h": torch.zeros(b, n, 16), "tex": tex}
    k = torch.tensor([[[160., 0, 16], [0, 160., 16], [0, 0, 1]]])
    r = torch.diag(torch.tensor([1., -1., -1.]))[None]
    out = dec(z, r, torch.tensor([[0., 0., 9.]]), k)
    assert torch.allclose(out["rgb"][0, :, 16, 16], torch.tensor([0.2, 0.5, 0.8]), atol=1e-3)


def test_v3_pose_update_is_small_and_identity_at_zero():
    from train_static_claude.model import Encoder
    enc = VISR(**model_kwargs(tiny_args())).encoder
    k = torch.tensor([[[400., 0, 28], [0, 400., 28], [0, 0, 1]]])
    r = torch.diag(torch.tensor([1., -1., -1.]))[None]
    t = torch.tensor([[0.1, -0.2, 9.]])
    r0, t0 = enc._pose_update(r, t, torch.zeros(1, 6), k, 56)
    assert torch.allclose(r0, r, atol=1e-6) and torch.allclose(t0, t, atol=1e-5)
    r1, t1 = enc._pose_update(r, t, torch.full((1, 6), 50.), k, 56)
    from train_static_claude.geometry import geodesic
    assert geodesic(r, r1) < 0.18 and (t1 - t).norm() < 0.6
    assert torch.allclose(r1 @ r1.transpose(1, 2), torch.eye(3)[None], atol=1e-5)


def test_v3_teacher_forced_samples_keep_gt_pose():
    net = VISR(**model_kwargs(tiny_args()))
    with torch.no_grad():
        for head in net.encoder.pose_refine:
            head[-1].bias.fill_(1.)
    img = torch.rand(2, 3, 56, 56)
    k = torch.tensor([[400., 0, 28], [0, 400., 28], [0, 0, 1]]).expand(2, 3, 3)
    gt_r = torch.diag(torch.tensor([1., -1., -1.])).expand(2, 3, 3)
    gt_t = torch.tensor([0., 0., 9.]).expand(2, 3)
    out = net.encoder(img, k, (gt_r, gt_t), torch.tensor([1., 0.]))
    assert torch.allclose(out["R_used"][0], gt_r[0]) and torch.allclose(out["t_used"][0], gt_t[0])
    assert not torch.allclose(out["R_used"][1], out["pose_iters"][0][0][1])
    assert len(out["pose_iters"]) == 1 + net.encoder.pose_iters


def test_texture_readout_survives_nan_coordinates():
    dec = _readout_decoder(k=2)
    tex = torch.rand(1, 3, 16, 3)
    p = torch.rand(1, 3, 3)
    origin = torch.tensor([[float("nan"), 0., -5.]])
    d = torch.tensor([[[0., 0., 1.], [0., 0., 1.]]])
    idx = torch.tensor([[[0, 2], [1, 2]]])
    out = dec._read_chunk(tex, p, torch.zeros(1, 3), origin, d, idx,
                          torch.tensor([[1., 0, 0]] * 3), torch.tensor([[0., 1, 0]] * 3),
                          torch.tensor([1.]))
    assert out.shape == (1, 2, 4) and torch.isfinite(out).all()


# ----------------------------------------------------------------------- v4
def test_surface_term_pulls_slots_and_ignores_far_ones():
    slots = torch.tensor([[[0., 0., 0.1], [0., 0., 5.]]], requires_grad=True)
    tgt = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
    loss, near = L.surface_term(slots, tgt, torch.tensor([[True, True]]), gate=0.6, points=8)
    loss.sum().backward()
    assert abs(loss.item() - 0.1) < 1e-5 and near.item() == 0.0
    assert slots.grad[0, 0, 2] > 0 and slots.grad[0, 1].abs().sum() == 0


def test_v4_hard_copy_gate_cannot_be_closed():
    net = VISR(**model_kwargs(tiny_args()))
    enc = net.encoder
    assert enc.copy_hard and net.decoder.corr_bound == 1.5
    with torch.no_grad():
        enc.tex_write[-1].bias[-1].fill_(-30.)   # a learned gate would now be closed
    img = torch.full((1, 3, 56, 56), 0.3)
    k = torch.tensor([[[400., 0, 28], [0, 400., 28], [0, 0, 1]]])
    out = enc(img, k)
    col = out["tex_id"][..., -3:]
    seen = (col - 0.3).abs().amax(-1) < 1e-3
    assert seen.any(), "some texels must copy the observed colour"


# --------------------------------------------------------------------- v4.1
def test_v41_copy_used_only_where_rays_hit_a_patch():
    net = VISR(**model_kwargs(tiny_args()))
    dec, enc = net.decoder, net.encoder
    assert dec.copy_coverage and enc.tex_size == 8
    n, t = enc.p0.shape[0], enc.tex_size ** 2
    tex = torch.zeros(1, n, t, enc.tex_channels)
    tex[..., -3:] = 0.9                                   # copied colour: light
    p = torch.zeros(1, n, 3)
    p[0, :, 0] = 5.0                                       # every slot 50 cm to the side
    z = {"p": p, "h": torch.zeros(1, n, 16), "tex": tex}
    k = torch.tensor([[[160., 0, 16], [0, 160., 16], [0, 0, 1]]])
    r = torch.diag(torch.tensor([1., -1., -1.]))[None]
    with torch.no_grad():
        dec.refine[-1].bias[:3].fill_(-3.)                 # free prediction: dark
    out = dec(z, r, torch.tensor([[0., 0., 9.]]), k)
    # rays far from every patch must not take the copied colour
    assert out["rgb"][0, :, 0, 0].max() < 0.2, out["rgb"][0, :, 0, 0]


def test_warm_start_upsamples_texture_dynamics():
    from train_static_claude.train import warm_start
    old_args = tiny_args(); old_args.tex_size = 4
    old = VISR(**model_kwargs(old_args))
    with torch.no_grad():
        old.encoder.dyn_mlp[2].weight.normal_()
    new = VISR(**model_kwargs(tiny_args()))
    rep = warm_start(new, old.state_dict())
    assert "encoder.dyn_mlp.2.weight" in rep["sliced"]
    head = 3 + new.encoder.to_h.out_features
    c = new.encoder.tex_channels
    w_old = old.encoder.dyn_mlp[2].weight[head:].view(4, 4, c, -1)
    w_new = new.encoder.dyn_mlp[2].weight[head:].view(8, 8, c, -1)
    assert torch.allclose(w_new[2:4, 6:8], w_old[1:2, 3:4].expand(2, 2, c, -1))
