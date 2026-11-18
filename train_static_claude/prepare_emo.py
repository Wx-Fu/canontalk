"""Extract synchronized EmoTalk3D frames into a JPEG cache (run once, CPU).

Random seeking in the HEVC .mov files is unreliable (see train_stage_static
notes), so each video is decoded *sequentially* and the same frame indices are
kept for all 11 cameras of a clip. Output layout:

  <out>/<person>/<clip>/<cam>_<frame:04d>.jpg
  <out>/index.jsonl     one line per clip: person, clip, calib, cameras, frames

Example (32 processes, 6 synchronized frames per clip):
  python -m train_static_claude.prepare_emo \
      --manifests train/manifests/train.jsonl train/manifests/val.jsonl \
      --out data_cache/emotalk3d_frames --frames-per-clip 6 --workers 32
"""
import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path


def frame_count(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def decode_indices(path, indices):
    import cv2
    wanted = set(indices)
    out = {}
    cap = cv2.VideoCapture(str(path))
    i = 0
    while len(out) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = frame
        i += 1
    cap.release()
    return out


def process(job):
    import cv2
    row, out_root, count, seed, quality = job
    clip_dir = Path(out_root) / row["person"] / row["clip"]
    done = clip_dir / "done.json"
    if done.is_file():
        return json.loads(done.read_text())
    cams = sorted(row["cameras"])
    counts = [frame_count(row["cameras"][c]["path"]) for c in cams]
    usable = min(counts)
    if usable < 10:
        return None
    rng = random.Random(f"{seed}/{row['person']}/{row['clip']}")
    # Evenly spaced with a random phase; avoid the first/last few frames.
    lo, hi = 3, usable - 4
    step = (hi - lo) / count
    indices = sorted({int(lo + step * (i + rng.random())) for i in range(count)})
    clip_dir.mkdir(parents=True, exist_ok=True)
    for c in cams:
        frames = decode_indices(row["cameras"][c]["path"], indices)
        if len(frames) != len(indices):
            return None
        for i, image in frames.items():
            cv2.imwrite(str(clip_dir / f"{c}_{i:04d}.jpg"), image,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
    record = {"person": row["person"], "clip": row["clip"], "calib": row["calib"],
              "cameras": cams, "frames": indices, "frame_counts": counts,
              "dir": str(clip_dir)}
    done.write_text(json.dumps(record))
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames-per-clip", type=int, default=6)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--seed", default="emo-cache-v1")
    p.add_argument("--limit", type=int, default=0, help="debug: only first N clips")
    args = p.parse_args()
    rows = [json.loads(line) for m in args.manifests for line in m.read_text().splitlines() if line]
    if args.limit:
        rows = rows[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(r, str(args.out), args.frames_per_clip, args.seed, args.quality) for r in rows]
    records = []
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(process, jobs, chunksize=2)):
            if rec is not None:
                records.append(rec)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} clips, {len(records)} ok", flush=True)
    records.sort(key=lambda r: (r["person"], r["clip"]))
    with (args.out / "index.jsonl").open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} clips to {args.out / 'index.jsonl'}")


if __name__ == "__main__":
    main()
