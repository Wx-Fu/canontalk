"""EmoTalk3D synchronized multi-view frames from the JPEG cache (prepare_emo.py).

calib.json: K_<cam> (512x512 crop intrinsics, principal point at 256 -> treated
as continuous-pixel convention), Rt_<cam> world->camera in mm with world = camera F,
near_F / far_F. No head pose or geometry is available: the model's head pose is
supervised only through *relative* camera transforms on this dataset.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data_common import crop_square
from .geometry import MM

VAL_PEOPLE = ("0046", "0047", "0048")  # historical validation identities


@lru_cache(maxsize=512)
def load_calib(path):
    return json.loads(Path(path).read_text())


def read_index(path, split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if split == "train":
        return [r for r in rows if r["person"] not in VAL_PEOPLE]
    return [r for r in rows if r["person"] in VAL_PEOPLE]


def camera(calib, cam):
    k = np.asarray(calib[f"K_{cam}"], np.float64)
    rt = np.asarray(calib[f"Rt_{cam}"], np.float64)
    return k, rt[:, :3], rt[:, 3]


def valid_cameras(calib, cams):
    """Drop cameras with implausible calibration. EmoTalk3D subject 0020 camera D
    has fx = 1.6e8 px and a centre ~100 km away in every clip, which turned the
    pose / rays of any sample using it into NaN (all v3-v4.1 NaN steps)."""
    center = head_center_world(calib)
    ok = []
    for c in cams:
        k, r, t = camera(calib, c)
        fx_ratio = k[0, 0] / calib[f"width_{c}"]
        dist = np.linalg.norm(-r.T @ t - center)
        if (np.isfinite(k).all() and np.isfinite(t).all() and 1 < fx_ratio < 20
                and 300 < dist < 5000):
            ok.append(c)
    return ok


def head_center_world(calib):
    # F is the world frame and looks along +z at the subject.
    return np.array([0., 0., (calib["near_F"] + calib["far_F"]) / 2])


def view_angles(calib, source, cams):
    center = head_center_world(calib)
    def direction(c):
        _, r, t = camera(calib, c)
        d = -r.T @ t - center
        return d / np.linalg.norm(d)
    ds = direction(source)
    return {c: float(np.degrees(np.arccos(np.clip(direction(c) @ ds, -1, 1)))) for c in cams}


def view(record, cam, frame, size, rng, jitter):
    import cv2
    calib = load_calib(record["calib"])
    path = Path(record["dir"]) / f"{cam}_{frame:04d}.jpg"
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = bgr[..., ::-1]
    k, r_e, t_e = camera(calib, cam)
    height, width = rgb.shape[:2]
    if (width, height) != (calib[f"width_{cam}"], calib[f"height_{cam}"]):
        raise ValueError(f"image/calib size mismatch at {path}")
    side = width * (1 - rng.uniform(0, 0.12)) if jitter else width
    side *= 1 + (rng.uniform(-jitter, jitter) / 2 if jitter else 0)
    u = width / 2 + (rng.uniform(-jitter, jitter) * side / 2 if jitter else 0)
    v = height / 2 + (rng.uniform(-jitter, jitter) * side / 2 if jitter else 0)
    crop, valid, k_crop = crop_square(np.ascontiguousarray(rgb), k, u - side / 2,
                                      v - side / 2, side, size)
    return {"image": crop.astype(np.float32) / 255., "valid": valid, "K": k_crop,
            "E_R": r_e.astype(np.float32), "E_t": (t_e * MM).astype(np.float32)}
