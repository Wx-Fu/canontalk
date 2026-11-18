"""CPU unit tests. Run from tmm/:  python -m pytest -q train_static_claude/tests"""
import math

import numpy as np
import torch

from train_static_claude import geometry as G
from train_static_claude.data_common import crop_square
from train_static_claude.model import Decoder, relative_extrinsic


def random_rotation(n=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return G.rot6d_to_matrix(torch.randn(n, 6, generator=g))


def test_rot6d_roundtrip_and_orthonormal():
    r = random_rotation(16)
    assert torch.allclose(r @ r.transpose(1, 2), torch.eye(3).expand(16, 3, 3), atol=1e-5)
    assert torch.allclose(torch.det(r), torch.ones(16), atol=1e-5)
    assert torch.allclose(G.rot6d_to_matrix(G.matrix_to_rot6d(r)), r, atol=1e-5)


def test_relative_extrinsic_maps_source_to_target_camera():
    rs, rt = random_rotation(4, 1), random_rotation(4, 2)
    ts, tt = torch.randn(4, 3), torch.randn(4, 3)
    x_world = torch.randn(4, 10, 3)
    x_s = G.transform(rs, ts, x_world)
    x_t = G.transform(rt, tt, x_world)
    r_rel, t_rel = relative_extrinsic(rt, tt, rs, ts)
    assert torch.allclose(G.transform(r_rel, t_rel, x_s), x_t, atol=1e-4)


def test_crop_intrinsics_consistent_with_pixels():
    """A bright dot projected with K lands at the same place in the crop."""
    h, w = 300, 200
    k = np.array([[400., 0, 100.], [0, 400., 150.], [0, 0, 1]])
    point = np.array([0.03, -0.05, 1.0])
    uv = (k @ point)[:2] / point[2]
    image = np.zeros((h, w, 3), np.uint8)
    image[int(uv[1]), int(uv[0])] = 255
    crop, valid, k_crop = crop_square(image, k, uv[0] - 37.3, uv[1] - 52.1, 120.0, 240)
    center_px = np.floor(uv) + 0.5  # the dot's true location is its pixel center
    a = k_crop @ np.linalg.inv(k)
    uv_c = (a @ np.array([*center_px, 1.]))[:2]
    ys, xs = np.nonzero(crop[..., 0] > 0)
    center = np.array([np.average(xs + 0.5, weights=crop[ys, xs, 0]),
                       np.average(ys + 0.5, weights=crop[ys, xs, 0])])
    assert np.abs(center - uv_c).max() < 1.0, (center, uv_c)
    assert valid.min() == 1.0


def test_zbuffer_hides_back_points():
    k = torch.tensor([[[50., 0, 32], [0, 50., 32], [0, 0, 1]]])
    front = torch.tensor([[[0., 0, 1.0]]])
    back = torch.tensor([[[0., 0, 2.0]]])
    pts = torch.cat((front, back), 1)
    uv, z = G.project(pts, k)
    vis = G.zbuffer_visible(uv, z, 64, 64, 0.05)
    assert vis.tolist() == [[True, False]]


def test_decoder_bias_invariant_to_global_rigid_motion():
    """Ray-slot attention must not change if the canonical frame is moved
    rigidly together with the camera (it only sees relative geometry)."""
    dec = Decoder(token_dim=8, dim=16, heads=2, layers=1, patch=8, render_size=32)
    k = torch.tensor([[[40., 0, 16], [0, 40., 16], [0, 0, 1]]])
    r_q, t_q = random_rotation(1, 3), torch.tensor([[0.1, -0.2, 9.0]])
    p = torch.randn(1, 20, 3)
    o, d = G.rays_in_head_frame(r_q, t_q, k, 32, 32, 8)
    b1 = dec._bias(o, d, p)
    g_r, g_t = random_rotation(1, 4), torch.tensor([[0.5, 0.3, -0.2]])
    # Move the canonical frame: p' = g p ; Q' = Q g^-1
    p2 = G.transform(g_r, g_t, p)
    gi_r, gi_t = G.invert(g_r, g_t)
    r2, t2 = G.compose(r_q, t_q, gi_r, gi_t)
    o2, d2 = G.rays_in_head_frame(r2, t2, k, 32, 32, 8)
    with torch.no_grad():
        dec.depth_slope.uniform_(-1, 1)
    # The depth term may shift by a per-ray constant (frame translation), which
    # softmax over slots ignores: compare attention distributions.
    a1 = dec._bias(o, d, p).softmax(-1)
    a2 = dec._bias(o2, d2, p2).softmax(-1)
    assert torch.allclose(a1, a2, atol=5e-4), (a1 - a2).abs().max()
    assert b1.shape == (1, 2, 16, 20)


def test_fourier_shapes():
    x = torch.randn(2, 5, 3)
    assert G.fourier(x, 4, 4).shape[-1] == G.fourier_dim(3, 4)
