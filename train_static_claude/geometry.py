"""Rigid-body and pinhole utilities shared by data, model and losses.

Conventions (used everywhere in this package):
- A pose is a pair (R, t) mapping x_b = R x_a + t. ``Q`` always denotes the
  head->camera pose; ``E`` the world->camera extrinsic; ``H`` head->world.
- Lengths are in **decimeters** (1 dm = 100 mm) inside the model; datasets
  convert from millimeters with ``MM``.
- Intrinsics use *continuous* pixel coordinates: pixel (row i, col j) covers
  [j, j+1) x [i, i+1) and its center is (j+0.5, i+0.5). Calibrations that use
  the OpenCV integer-center convention are converted with ``opencv_to_continuous``.
  ``grid_sample(..., align_corners=False)`` then uses ``2 * u / W - 1``.
"""
import functools
import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

MM = 0.01  # millimeters -> decimeters


def exact(fn):
    """Run ``fn`` in float32 with autocast off.

    Autocast casts matmul to bf16, whose 8-bit mantissa has a step of ~6 mm at
    10 dm and ~1 px at u=256, so all camera geometry must stay out of it.
    Floating tensor arguments are cast to float32; everything else passes through.
    """
    def cast(x):
        return x.float() if torch.is_tensor(x) and x.is_floating_point() else x

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        tensors = (x for x in itertools.chain(args, kwargs.values()) if torch.is_tensor(x))
        device = next(tensors).device.type
        with torch.autocast(device, enabled=False):
            return fn(*[cast(x) for x in args], **{k: cast(v) for k, v in kwargs.items()})
    return wrapper


# ----------------------------------------------------------------------------
# numpy helpers (data side)
# ----------------------------------------------------------------------------
def opencv_to_continuous(k):
    k = np.asarray(k, np.float64).copy()
    k[0, 2] += 0.5
    k[1, 2] += 0.5
    return k


def crop_intrinsics(k, x0, y0, side, size):
    """Intrinsics after cropping the square [x0, x0+side) and resizing to ``size``."""
    s = size / side
    a = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1.]])
    return a @ k


def compose_np(r1, t1, r2, t2):
    """(r1,t1) o (r2,t2): x -> r1 (r2 x + t2) + t1."""
    return r1 @ r2, r1 @ t2 + t1


def invert_np(r, t):
    return r.T, -r.T @ t


# ----------------------------------------------------------------------------
# torch helpers (model side)
# ----------------------------------------------------------------------------
@exact
def rot6d_to_matrix(x):
    """Continuous 6D rotation (Zhou et al. 2019). Columns are b1, b2, b3."""
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(r):
    return torch.cat((r[..., :, 0], r[..., :, 1]), dim=-1)


@exact
def compose(r1, t1, r2, t2):
    return r1 @ r2, (r1 @ t2[..., None])[..., 0] + t1


@exact
def invert(r, t):
    rt = r.transpose(-1, -2)
    return rt, -(rt @ t[..., None])[..., 0]


@exact
def transform(r, t, x):
    """Apply (R, t) to points x[..., N, 3] with R[..., 3, 3], t[..., 3]."""
    return x @ r.transpose(-1, -2) + t[..., None, :]


@exact
def geodesic(r1, r2):
    """Rotation angle (radians) of r1^T r2, numerically safe."""
    m = r1.transpose(-1, -2) @ r2
    cos = ((m.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)


@exact
def project(x_cam, k):
    """Camera-frame points [..., N, 3] -> continuous pixels [..., N, 2], depth z."""
    z = x_cam[..., 2]
    safe = torch.where(z.abs() < 1e-4, torch.full_like(z, 1e-4), z)
    uvw = x_cam @ k.transpose(-1, -2)
    return uvw[..., :2] / safe[..., None], z


@exact
def pixel_rays(k, height, width, stride=1):
    """Unit ray directions (camera frame) through pixel / patch centers.

    With ``stride`` > 1, rays go through the centers of stride x stride patches.
    Returns [B, (H/stride)*(W/stride), 3].
    """
    b = k.shape[0]
    hs, ws = height // stride, width // stride
    ys = (torch.arange(hs, device=k.device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(ws, device=k.device, dtype=torch.float32) + 0.5) * stride
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack((x, y, torch.ones_like(x)), -1).reshape(1, -1, 3).expand(b, -1, -1)
    d = pix @ torch.linalg.inv(k.float()).transpose(-1, -2)
    return F.normalize(d, dim=-1)


@exact
def rays_in_head_frame(r_q, t_q, k, height, width, stride=1):
    """Camera rays expressed in the head (canonical) frame for pose Q=(r_q,t_q).

    Returns origins [B, 3] (camera center in head frame) and unit directions
    [B, N, 3].
    """
    d_cam = pixel_rays(k, height, width, stride)
    r_inv, t_inv = invert(r_q.float(), t_q.float())
    return t_inv, d_cam @ r_inv.transpose(-1, -2)


@exact
def plucker(origin, direction):
    """Plücker coordinates (d, o x d) for rays sharing origin [B,3]."""
    o = origin[:, None, :].expand_as(direction)
    return torch.cat((direction, torch.cross(o, direction, dim=-1)), dim=-1)


def to_grid(uv, height, width):
    return torch.stack((2 * uv[..., 0] / width - 1, 2 * uv[..., 1] / height - 1), -1)


@exact
def sample_points(feature, uv):
    """Bilinear features at continuous pixel coords uv [B, N, 2] -> [B, N, C]."""
    h, w = feature.shape[-2:]
    grid = to_grid(uv, h, w)[:, :, None, :]
    out = F.grid_sample(feature, grid.to(feature.dtype), mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[..., 0].transpose(1, 2)


@exact
def fourier(x, bands, max_freq):
    """NeRF-style encoding of x[..., D] with log-spaced frequencies up to max_freq."""
    freqs = 2 ** torch.linspace(0, math.log2(max_freq), bands, device=x.device,
                                dtype=torch.float32)
    xf = x.float()[..., None] * freqs * math.pi
    return torch.cat((x.float(), torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)), -1)


def fourier_dim(d, bands):
    return d * (1 + 2 * bands)


@exact
def zbuffer_visible(uv, depth, height, width, tolerance):
    """Visibility of points by splatting their depth into a coarse min-buffer.

    uv: [B, N, 2] continuous pixels in an image of size (height, width) at the
    buffer resolution; depth: [B, N]. A point is visible if it is in front of
    the camera, inside the image and within ``tolerance`` of the buffer.
    """
    b, n = depth.shape
    col = uv[..., 0].floor().long()
    row = uv[..., 1].floor().long()
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height) & (depth > 0)
    index = (row.clamp(0, height - 1) * width + col.clamp(0, width - 1))
    index = index + torch.arange(b, device=uv.device)[:, None] * height * width
    big = torch.full((b * height * width,), float("inf"), device=uv.device)
    values = torch.where(inside, depth, torch.full_like(depth, float("inf")))
    big.scatter_reduce_(0, index.flatten(), values.flatten().float(), reduce="amin")
    front = big[index.flatten()].reshape(b, n)
    return inside & (depth <= front + tolerance)
