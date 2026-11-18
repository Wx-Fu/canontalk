"""Ava-256 multi-view loader (images straight from the release ZIPs).

Per capture this module reads:
  decoder/camera_calibration.json  K (transposed), T (4x4 transposed) world->cam
  decoder/frame_list.csv           frame ids (+ segment names)
  decoder/head_pose/head_pose.zip  <frame>.txt  3x4 head->world (mm)
  decoder/kinematic_tracking/registration_vertices.zip  <frame>.ply  7306 verts, head space (mm)
  decoder/image/cam<id>.zip        cam<id>/<frame>.avif  667x1024 RGB

Conventions follow ``experiment/static_upper_bound/common.py`` (verified there
by mesh overlays): K = KRT.K^T scaled by (w/2668, h/4096); world->cam = T[:4,:3]^T;
registration vertices live in head space and head_pose maps head->world.
Lens distortion is ignored (same assumption as the verified overlays).
"""
import csv
import hashlib
import io
import json
import threading
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

try:  # Pillow < 11.3 needs the plugin for AVIF
    import pillow_avif  # noqa: F401
except ImportError:
    pass

from .data_common import crop_square
from .geometry import MM, opencv_to_continuous

CALIB_SIZE = (2668, 4096)  # (width, height) of the calibration
HEAD_CROP_CENTER_MM = np.array([0., -10., -40.])  # head-frame point at crop center
CROP_EXTENT_MM = 360.  # metric side length of the head crop at the head depth
NUM_VERTS = 7306


class ZipCache:
    """Per-process LRU of open ZipFile handles plus a thread pool.

    On the shared filesystem a random member read costs ~0.5-1.5 s of
    *latency*, so all members of one sample are fetched concurrently
    (``read_many``). ZipFile serializes seeks on a shared handle with a lock,
    so concurrent reads are safe. Handles are opened lazily, after fork.
    """

    def __init__(self, capacity=128, threads=16):
        self.capacity, self.threads = capacity, threads
        self.handles = OrderedDict()
        self.lock = threading.Lock()
        self.pool = None

    def _handle(self, path):
        path = str(path)
        with self.lock:
            handle = self.handles.pop(path, None)
            if handle is not None:
                self.handles[path] = handle
                return handle
        handle = zipfile.ZipFile(path)
        with self.lock:
            existing = self.handles.pop(path, None)
            if existing is not None:
                handle.close()
                handle = existing
            self.handles[path] = handle
            while len(self.handles) > self.capacity:
                _, old = self.handles.popitem(last=False)
                old.close()
        return handle

    def read(self, path, member):
        return self._handle(path).read(member)

    def read_many(self, requests):
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.threads)
        return list(self.pool.map(lambda r: self.read(*r), requests))


def read_ply_xyz(data):
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", "replace")
    if "binary_little_endian" not in header or f"element vertex {NUM_VERTS}" not in header:
        raise ValueError("Unexpected registration PLY header")
    return np.frombuffer(data[end:end + NUM_VERTS * 12], dtype="<f4").reshape(-1, 3)


def load_cameras(decoder):
    cameras = {}
    for item in json.loads((decoder / "camera_calibration.json").read_text())["KRT"]:
        k = np.asarray(item["K"], np.float64).T
        rt = np.asarray(item["T"], np.float64)[:4, :3].T
        cameras[str(item["cameraId"])] = (k, rt[:, :3], rt[:, 3])
    return cameras


def split_captures(root, val_count=10, test_count=10, seed="ava-static-v1"):
    """Identity-disjoint split by a stable hash of the capture name."""
    root = Path(root)
    captures = []
    for path in sorted(root.iterdir()):
        decoder = path / "decoder"
        images = decoder / "image"
        if not (images.is_dir() and (decoder / "head_pose/head_pose.zip").is_file()
                and (decoder / "kinematic_tracking/registration_vertices.zip").is_file()):
            continue
        if len(list(images.glob("cam*.zip"))) < 8:
            continue
        captures.append(path.name)
    ranked = sorted(captures, key=lambda c: hashlib.sha1((seed + c).encode()).hexdigest())
    return {"test": sorted(ranked[:test_count]),
            "val": sorted(ranked[test_count:test_count + val_count]),
            "train": sorted(ranked[test_count + val_count:])}


class AvaCapture:
    def __init__(self, root, name):
        self.name = name
        self.decoder = Path(root) / name / "decoder"
        self.cameras = load_cameras(self.decoder)
        available = {p.stem[3:] for p in (self.decoder / "image").glob("cam*.zip")}
        self.camera_ids = sorted(c for c in self.cameras if c in available)
        with (self.decoder / "frame_list.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.frames = [int(r["frame_id"]) for r in rows]
        self.segments = [r["seg_id"] for r in rows]
        # Camera centers in world coordinates, for angle-stratified sampling.
        self.centers = {c: -r.T @ t for c, (_, r, t) in self.cameras.items()}


class AvaReader:
    """Decodes one (capture, frame, camera) into a crop with exact intrinsics."""

    def __init__(self, root):
        self.root = Path(root)
        self.zips = ZipCache()

    @staticmethod
    def pose_member(capture, frame):
        return capture.decoder / "head_pose/head_pose.zip", f"{frame:06d}.txt"

    @staticmethod
    def verts_member(capture, frame):
        return (capture.decoder / "kinematic_tracking/registration_vertices.zip",
                f"{frame:06d}.ply")

    @staticmethod
    def rgb_member(capture, camera, frame):
        return capture.decoder / f"image/cam{camera}.zip", f"cam{camera}/{frame:06d}.avif"

    @staticmethod
    def decode_pose(data):
        pose = np.loadtxt(io.BytesIO(data)).reshape(-1, 4)[:3]
        return pose[:, :3], pose[:, 3]

    @staticmethod
    def decode_rgb(data):
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))

    def head_pose(self, capture, frame):
        return self.decode_pose(self.zips.read(*self.pose_member(capture, frame)))

    def vertices(self, capture, frame):
        data = self.zips.read(*self.verts_member(capture, frame))
        return read_ply_xyz(data).astype(np.float32)

    def rgb(self, capture, camera, frame):
        return self.decode_rgb(self.zips.read(*self.rgb_member(capture, camera, frame)))

    def view(self, capture, camera, frame, head, size, rng, jitter, rgb=None):
        """Return crop [size,size,3] float, valid mask, K (continuous), E, Q (head->cam).

        The crop is a square centered on the projection of a fixed head-frame
        point, with metric side CROP_EXTENT_MM at that point's depth. ``jitter``
        randomizes scale (+-jitter) and shift (+-jitter/2 of the side) so that the
        crop center is never an exact pose label.
        """
        if rgb is None:
            rgb = self.rgb(capture, camera, frame)
        height, width = rgb.shape[:2]
        k_raw, r_e, t_e = capture.cameras[camera]
        k = opencv_to_continuous(k_raw)
        k = np.diag([width / CALIB_SIZE[0], height / CALIB_SIZE[1], 1.]) @ k
        r_h, t_h = head
        r_q, t_q = r_e @ r_h, r_e @ t_h + t_e  # head->camera, mm
        center_cam = r_q @ HEAD_CROP_CENTER_MM + t_q
        if center_cam[2] <= 0:
            raise ValueError("head behind camera")
        uvw = k @ center_cam
        u, v = uvw[:2] / uvw[2]
        side = k[0, 0] * CROP_EXTENT_MM / center_cam[2]
        if jitter:
            side *= 1 + rng.uniform(-jitter, jitter)
            u += rng.uniform(-jitter, jitter) * side / 2
            v += rng.uniform(-jitter, jitter) * side / 2
        x0, y0 = u - side / 2, v - side / 2
        crop, valid, k_crop = crop_square(rgb, k, x0, y0, side, size)
        return {"image": crop.astype(np.float32) / 255., "valid": valid,
                "K": k_crop,
                "E_R": r_e.astype(np.float32), "E_t": (t_e * MM).astype(np.float32),
                "Q_R": r_q.astype(np.float32), "Q_t": (t_q * MM).astype(np.float32)}


def view_angles(capture, head, source, cameras):
    """Angle (deg) between source and each camera as seen from the head center."""
    r_h, t_h = head
    center = r_h @ HEAD_CROP_CENTER_MM + t_h
    def direction(c):
        d = capture.centers[c] - center
        return d / np.linalg.norm(d)
    ds = direction(source)
    return {c: float(np.degrees(np.arccos(np.clip(direction(c) @ ds, -1, 1)))) for c in cameras}
