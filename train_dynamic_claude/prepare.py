"""Build the unified clip index for the dynamic stage (run once).

For every clip: video path(s), 16 kHz mono wav (cached), fps / frame count,
person id, split, and for monocular data a fixed square head crop.

Monocular crops: head (hat+hair+face) box from the frozen human parser on 3
frames, then a square of side = head_height / HEAD_FRAC centred like the
EmoTalk3D crops (HEAD_FRAC, HEAD_CY measured on EmoTalk3D), so VISR sees the
framing it was trained on. EmoTalk3D clips are already head crops (no crop).

Sharded: run one process per GPU, then `--merge`.
  python -m train_dynamic_claude.prepare --out data_cache/dyn_index --shard 0 --num-shards 8
  python -m train_dynamic_claude.prepare --out data_cache/dyn_index --merge
"""
import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path("/apdcephfs_nj8/share_305278643/wenxinfu/data_ori")
CELEBV = DATA / "CelebV-HQ-data/35666"
MULTIDIALOG = DATA / "MultiDialog/video/train"
EMO_MANIFESTS = [Path("train/manifests/train.jsonl"), Path("train/manifests/val.jsonl")]
EMO_VAL = {"0046", "0047", "0048"}
HEAD_FRAC = 0.876  # parser head-box height / crop side in EmoTalk3D crops (median of 18)
HEAD_CY = 0.56     # parser head-box centre / crop side (vertical), same measurement


def split_of(key, val_percent=1.0):
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 10000
    return "val" if h < val_percent * 100 else "train"


def probe(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    if fps > 0 and n > 10:
        for idx in (n // 5, n // 2, 4 * n // 5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, im = cap.read()
            if ok:
                frames.append(cv2.resize(im, (384, 384))[..., ::-1].copy())
    cap.release()
    return fps, n, w, h, frames


def extract_wav(src, dst):
    if dst.is_file() and dst.stat().st_size > 1000:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                        "-f", "wav", str(tmp)], capture_output=True)
    if r.returncode != 0 or not tmp.is_file():
        return False
    os.replace(tmp, dst)
    return True


def head_crop(masker, frames, w, h, device):
    """Square crop (x0, y0, side) in source pixels from parser head masks."""
    import torch
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255).to(device)
    with torch.no_grad():
        _, head = masker(x)
    m = (head[:, 0] > 0.5).any(0).cpu().numpy()
    ys, xs = np.nonzero(m)
    if len(ys) < 200:
        return None
    sx, sy = w / m.shape[1], h / m.shape[0]
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    cx = (xs.min() + xs.max() + 1) / 2 * sx
    side = min((y1 - y0) / HEAD_FRAC, w, h)
    cy = (y0 + y1) / 2
    x0 = float(np.clip(cx - side / 2, 0, w - side))
    y0 = float(np.clip(cy - HEAD_CY * side, 0, h - side))
    return [round(x0, 1), round(y0, 1), round(side, 1)]


def monocular_jobs():
    jobs = []
    for f in sorted(os.listdir(CELEBV)):
        if f.endswith(".mp4"):
            yt = f.rsplit(".", 1)[0].split("_")[0]
            jobs.append(("celebv", CELEBV / f, yt, f[:-4]))
    for conv in sorted(os.listdir(MULTIDIALOG)):
        for f in sorted(os.listdir(MULTIDIALOG / conv)):
            if f.endswith(".mp4"):
                jobs.append(("multidialog", MULTIDIALOG / conv / f, conv, f[:-4]))
    return jobs


def run_shard(args):
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_static_claude.losses import Masker

    device = torch.device(f"cuda:{args.shard % torch.cuda.device_count()}"
                          if torch.cuda.is_available() else "cpu")
    masker = Masker(size=384).to(device)
    out = Path(args.out)
    audio_root = out / "audio"
    jobs = [] if args.no_mono else monocular_jobs()[args.shard::args.num_shards]
    if args.limit:
        rng = np.random.default_rng(0)  # spread a debug subset over both datasets
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), min(args.limit, len(jobs)), replace=False))]
    records, stats = [], {"ok": 0, "no_audio": 0, "short": 0, "no_head": 0}

    def cpu_part(job):
        ds, path, person, clip = job
        fps, n, w, h, frames = probe(path)
        wav = audio_root / ds / f"{clip}.wav"
        has_audio = fps > 0 and extract_wav(path, wav)
        return job, fps, n, w, h, frames, wav, has_audio

    with ThreadPoolExecutor(args.threads) as pool:
        for i, (job, fps, n, w, h, frames, wav, has_audio) in enumerate(pool.map(cpu_part, jobs)):
            ds, path, person, clip = job
            if not has_audio:
                stats["no_audio"] += 1
                continue
            if fps <= 0 or n / fps < args.min_seconds or len(frames) < 2:
                stats["short"] += 1
                continue
            crop = head_crop(masker, frames, w, h, device)
            if crop is None:
                stats["no_head"] += 1
                continue
            records.append({"dataset": ds, "person": person, "clip": clip, "video": str(path),
                            "audio": str(wav), "fps": fps, "frames": n, "size": [w, h],
                            "crop": crop, "split": split_of(person)})
            stats["ok"] += 1
            if (i + 1) % 500 == 0:
                print(json.dumps({"shard": args.shard, "done": i + 1, "of": len(jobs), **stats}),
                      flush=True)
    if args.shard == 0 and args.emo_limit != 0:
        records += emo_records(audio_root, args.emo_limit)
    with (out / f"shard_{args.shard:02d}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"shard": args.shard, "final": stats, "records": len(records)}), flush=True)


def emo_records(audio_root, limit=-1):
    import cv2
    rows = [json.loads(l) for m in EMO_MANIFESTS for l in m.read_text().splitlines() if l]
    if limit > 0:
        rows = rows[:limit // 2] + rows[-(limit - limit // 2):]  # some train, some val
    out = []
    for r in rows:
        wav = audio_root / "emotalk3d" / f"{r['person']}_{r['clip']}.wav"
        if not extract_wav(r["audio"], wav):
            continue
        cams = {c: v["path"] for c, v in r["cameras"].items()}
        cap = cv2.VideoCapture(cams["F"])
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        out.append({"dataset": "emotalk3d", "person": r["person"], "clip": r["clip"],
                    "video": cams, "calib": r["calib"], "audio": str(wav), "fps": fps,
                    "frames": n, "size": [512, 512], "crop": None,
                    "split": "val" if r["person"] in EMO_VAL else "train"})
    return out


def merge(args):
    out = Path(args.out)
    rows = [l for f in sorted(out.glob("shard_*.jsonl")) for l in f.read_text().splitlines() if l]
    for split in ("train", "val"):
        sel = [l for l in rows if json.loads(l)["split"] == split]
        (out / f"{split}.jsonl").write_text("\n".join(sel) + "\n")
    from collections import Counter
    c = Counter((json.loads(l)["dataset"], json.loads(l)["split"]) for l in rows)
    print(json.dumps({f"{d}/{s}": n for (d, s), n in sorted(c.items())}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--min-seconds", type=float, default=1.6)
    p.add_argument("--limit", type=int, default=0, help="debug: monocular clips per shard")
    p.add_argument("--emo-limit", type=int, default=-1, help="debug: EmoTalk3D clips (-1 all)")
    p.add_argument("--no-mono", action="store_true")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    main()
