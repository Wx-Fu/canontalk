"""Mixed talking-head windows: (reference image, target video window, audio).

Every sample, whatever the source:
  ref      [3,S,S]   in [0,1]   reference image (another time and, for EmoTalk3D,
                                usually another camera)
  video    [3,T,S,S] in [0,1]   target window, T frames resampled to 25 fps
  K_ref, K_tgt [3,3]            intrinsics at size S (continuous pixels)
  wave     [T/25 s * 16000]     16 kHz audio aligned with the window
Monocular clips use an assumed focal length (fx = FX_RATIO * S) and the
EmoTalk3D-like head crop from the index; EmoTalk3D uses its calibration.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FPS = 25
FX_RATIO = 4.8   # EmoTalk3D 512-crops: fx/size ~ 4.5-5.3


def read_frames(path, indices, crop, size):
    """Decode the given (sorted) frame indices; crop (x0, y0, side) and resize to size."""
    import cv2
    want = sorted(set(indices))
    cap = cv2.VideoCapture(str(path))
    got = {}
    seek = str(path).endswith(".mp4")
    if seek and want[0] > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
        i = want[0]
    else:
        i = 0
    while i <= want[-1]:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = im
        i += 1
    cap.release()
    if len(got) != len(want):
        raise RuntimeError(f"decoded {len(got)}/{len(want)} frames from {path}")
    out = []
    for idx in indices:
        im = got[idx]
        if crop is not None:
            x0, y0, side = crop
            im = im[int(round(y0)):int(round(y0 + side)), int(round(x0)):int(round(x0 + side))]
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        out.append(im[..., ::-1])
    return torch.from_numpy(np.stack(out)).permute(3, 0, 1, 2).float().div(255)


def read_wave(path, start, seconds):
    import soundfile as sf
    info = sf.info(str(path))
    first = int(round(start * info.samplerate))
    n = int(round(seconds * info.samplerate))
    wave, sr = sf.read(str(path), start=first, frames=n, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(1)
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio (prepare.py), got {sr}")
    wave = np.pad(wave, (0, max(0, n - len(wave))))
    return torch.from_numpy(wave[:n].copy())


def assumed_k(size):
    f = FX_RATIO * size
    return torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.]])


class TalkingWindows(Dataset):
    def __init__(self, index, split="train", frames=33, size=512, probs=None, length=10 ** 7,
                 seed=0, same_cam=0.2, ref_cands=4):
        rows = [json.loads(l) for l in Path(index).read_text().splitlines() if l]
        self.by_ds = {}
        for r in rows:
            if r["split"] == split:
                self.by_ds.setdefault(r["dataset"], []).append(r)
        probs = probs or {"emotalk3d": 0.4, "celebv": 0.3, "multidialog": 0.3}
        self.names = [d for d in probs if self.by_ds.get(d)]
        p = np.array([probs[d] for d in self.names], np.float64)
        self.p = p / p.sum()
        self.frames, self.size, self.length = frames, size, length
        self.seed, self.split, self.same_cam = seed, split, same_cam
        self.ref_cands = ref_cands  # v2: candidate references; training picks a large pose gap

    def __len__(self):
        return self.length

    def counts(self):
        return {d: len(v) for d, v in self.by_ds.items()}

    def __getitem__(self, i):
        key = (f"{self.seed}/{i}/{torch.initial_seed()}" if self.split == "train"
               else f"{self.seed}/{i}")
        rng = random.Random(key)
        for _ in range(10):
            ds = self.names[np.searchsorted(np.cumsum(self.p), rng.random())]
            row = self.by_ds[ds][rng.randrange(len(self.by_ds[ds]))]
            try:
                return self._emo(row, rng) if ds == "emotalk3d" else self._mono(row, rng)
            except (RuntimeError, ValueError, OSError, KeyError):
                continue
        raise RuntimeError("10 consecutive unreadable samples")

    def _times(self, row, rng):
        seconds = row["frames"] / row["fps"]
        window = (self.frames - 1) / FPS
        if seconds < window + 0.1:
            raise ValueError("clip too short")
        start = rng.uniform(0, seconds - window - 0.05)
        times = start + np.arange(self.frames) / FPS
        # reference from another moment when possible (avoid copying frame 0)
        outside = [t for t in np.arange(0.05, seconds - 0.05, 0.2)
                   if t < start - 0.3 or t > times[-1] + 0.3]
        pool = outside or list(np.arange(0.05, seconds - 0.05, 0.2))
        refs = [rng.choice(pool) for _ in range(self.ref_cands)]
        idx = lambda t: min(int(round(t * row["fps"])), row["frames"] - 1)
        return start, [idx(t) for t in times], [idx(t) for t in refs]

    def _mono(self, row, rng):
        start, idx, ref_idx = self._times(row, rng)
        video = read_frames(row["video"], idx, row["crop"], self.size)
        ref = read_frames(row["video"], ref_idx, row["crop"], self.size).permute(1, 0, 2, 3)
        k = assumed_k(self.size)
        return self._pack(row, ref, video, k, k, start, "same")

    def _emo(self, row, rng):
        from train_static_claude import data_emo
        calib = data_emo.load_calib(row["calib"])
        cams = data_emo.valid_cameras(calib, sorted(row["video"]))
        if len(cams) < 2:
            raise ValueError("no valid camera pair")
        src = rng.choice(cams)
        tgt = src if rng.random() < self.same_cam else rng.choice([c for c in cams if c != src])
        start, idx, ref_idx = self._times(row, rng)
        video = read_frames(row["video"][tgt], idx, None, self.size)
        ref = read_frames(row["video"][src], ref_idx, None, self.size).permute(1, 0, 2, 3)
        ks = []
        for c in (src, tgt):
            k = torch.tensor(calib[f"K_{c}"], dtype=torch.float32).clone()
            k[:2] *= self.size / calib[f"width_{c}"]
            ks.append(k)
        return self._pack(row, ref, video, ks[0], ks[1], start, f"{src}->{tgt}")

    def _pack(self, row, ref, video, k_ref, k_tgt, start, view):
        """ref: [C,3,S,S] candidate references (ref[0] is the default)."""
        wave = read_wave(row["audio"], start, self.frames / FPS)
        return {"ref": ref, "video": video, "K_ref": k_ref, "K_tgt": k_tgt, "wave": wave,
                "dataset": row["dataset"], "person": row["person"], "clip": row["clip"],
                "view": view}


def collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        out[k] = torch.stack(v) if torch.is_tensor(v[0]) else v
    return out


class AvaPairs(Dataset):
    """Ava-256 synchronized camera pairs as 1-frame samples (v2): reference from
    camera s, target image from camera t at the same instant, no audio. Gives
    ~160 extra multi-view identities for view change (EmoTalk3D has 27)."""

    def __init__(self, ava_root, pack_root, split="train", size=512, length=10 ** 7, seed=0,
                 ref_cands=4):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from train_static_claude import data_ava
        self.data_ava = data_ava
        names = data_ava.split_captures(ava_root)[split]
        self.pack_root = Path(pack_root)
        self.caps = []
        for n in names:
            idx = self.pack_root / n / "index.json"
            if idx.is_file():
                cap = data_ava.AvaCapture(ava_root, n)
                rec = json.loads(idx.read_text())
                cap.frames = rec["frames"]
                cap.camera_ids = [c for c in cap.camera_ids if c in set(rec["cameras"])]
                if len(cap.camera_ids) >= 4 and cap.frames:
                    self.caps.append(cap)
        self.reader = data_ava.AvaReader(ava_root)
        self.size, self.length, self.seed, self.split = size, length, seed, split
        self.ref_cands = ref_cands

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        from train_static_claude.prepare_ava import read_pack
        key = (f"{self.seed}/{i}/{torch.initial_seed()}" if self.split == "train"
               else f"{self.seed}/{i}")
        rng = random.Random(key)
        cap = self.caps[rng.randrange(len(self.caps))]
        frame = cap.frames[rng.randrange(len(cap.frames))]
        pack = read_pack(self.pack_root / cap.name / f"{frame:06d}.pack")
        head = self.reader.decode_pose(pack["head_pose.txt"])
        cams = rng.sample(cap.camera_ids, min(len(cap.camera_ids), self.ref_cands + 1))
        np_rng = np.random.default_rng(rng.randrange(2 ** 31))
        views = [self.reader.view(cap, c, frame, head, self.size, np_rng, 0.0,
                                  self.reader.decode_rgb(pack[f"cam{c}.avif"])) for c in cams]
        img = lambda v: torch.from_numpy(v["image"]).permute(2, 0, 1)
        refs = torch.stack([img(v) for v in views[1:]])
        while refs.shape[0] < self.ref_cands:
            refs = torch.cat((refs, refs[:1]))
        return {"ref": refs, "video": img(views[0])[:, None],
                "K_ref": torch.from_numpy(views[1]["K"]), "K_tgt": torch.from_numpy(views[0]["K"]),
                "wave": torch.zeros(int(16000 / FPS)), "dataset": "ava", "person": cap.name,
                "clip": str(frame), "view": f"{cams[1]}->{cams[0]}"}
