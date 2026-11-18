"""Pack selected Ava-256 frames into one file per frame (run once).

Why: on the shared filesystem each random ZIP-member read costs ~0.5-3 s of
latency, and a training sample needs ~12 of them. A pack holds *all* cameras of
one frame plus its head pose and registration (~3.5 MB), so one sample costs
two sequential reads. Image bytes are copied unchanged (no re-encoding).

Layout:  <out>/<capture>/<frame:06d>.pack   <out>/<capture>/index.json
Pack format: 8-byte little-endian header length | JSON header {member: [offset, size]} | blobs

  python -m train_static_claude.prepare_ava --ava-root data/ava256_rgb_4TB \
      --out data_cache/ava_packs --frames-per-capture 128 --processes 8 --threads 32
"""
import argparse
import hashlib
import json
import os
import struct
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import data_ava


def select_frames(capture, count, seed):
    """Round-robin over segments (speech, expressions, gaze) for coverage."""
    digest = hashlib.sha1(f"{seed}/{capture.name}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))
    by_segment = {}
    for frame, seg in zip(capture.frames, capture.segments):
        by_segment.setdefault(seg, []).append(frame)
    pools = [list(rng.permutation(v)) for _, v in sorted(by_segment.items())]
    rng.shuffle(pools)
    chosen = []
    while len(chosen) < count and any(pools):
        for pool in pools:
            if pool and len(chosen) < count:
                chosen.append(int(pool.pop()))
    return sorted(chosen)


def write_pack(path, members):
    header, offset, blobs = {}, 0, []
    for name, data in members.items():
        header[name] = [offset, len(data)]
        offset += len(data)
        blobs.append(data)
    head = json.dumps(header).encode()
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<Q", len(head)))
        handle.write(head)
        for b in blobs:
            handle.write(b)
    os.replace(tmp, path)


def read_pack(path):
    with open(path, "rb") as handle:
        data = handle.read()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    base = 8 + n
    return {k: data[base + o:base + o + s] for k, (o, s) in header.items()}


def pack_capture(job):
    root, name, out, count, threads, seed = job
    out_dir = Path(out) / name
    index_path = out_dir / "index.json"
    if index_path.is_file():
        record = json.loads(index_path.read_text())
        if record.get("requested", len(record["frames"])) >= count:
            return record
        # Growing an existing capture: selection is prefix-stable, old packs are reused.
    capture = data_ava.AvaCapture(root, name)
    reader = data_ava.AvaReader(root)
    reader.zips = data_ava.ZipCache(capacity=len(capture.camera_ids) + 4, threads=threads)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for frame in select_frames(capture, count, seed):
        path = out_dir / f"{frame:06d}.pack"
        if path.is_file():
            done.append(frame)
            continue
        requests = [reader.pose_member(capture, frame), reader.verts_member(capture, frame)]
        requests += [reader.rgb_member(capture, c, frame) for c in capture.camera_ids]
        try:
            blobs = reader.zips.read_many(requests)
        except (KeyError, OSError):
            continue  # frame missing in some archive: skip it entirely
        names = ["head_pose.txt", "vertices.ply"] + [f"cam{c}.avif" for c in capture.camera_ids]
        write_pack(path, dict(zip(names, blobs)))
        done.append(frame)
    record = {"capture": name, "frames": done, "cameras": capture.camera_ids, "requested": count}
    index_path.write_text(json.dumps(record))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ava-root", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames-per-capture", type=int, default=128)
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--seed", default="ava-pack-v1")
    p.add_argument("--captures", nargs="*", default=None, help="subset (debug)")
    args = p.parse_args()
    splits = data_ava.split_captures(args.ava_root)
    names = args.captures or sorted(n for v in splits.values() for n in v)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "splits.json").write_text(json.dumps(splits, indent=1))
    jobs = [(args.ava_root, n, str(args.out), args.frames_per_capture, args.threads, args.seed)
            for n in names]
    with Pool(args.processes) as pool:
        for i, rec in enumerate(pool.imap_unordered(pack_capture, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {rec['capture']}: {len(rec['frames'])} frames", flush=True)


if __name__ == "__main__":
    main()
