"""Mixed Ava-256 / EmoTalk3D multi-view sampler.

One sample = one person, two time instants t1, t2:

  enc      [3 + F] images at ``enc_size`` fed to the encoder: source s (t1),
               auxiliary a (t1, 15-60 deg from s), driver d (t2), then F optional
               extra t1 views (``fusion_views``, stage-B completion targets)
  tgt      [5] images at ``render_size`` at t1: [a, T1, T2, T3, s]
               T1..T3 stratified by angle from s: [0,25), [25,50), [50,180) deg
  tgt2     [2] images at ``render_size`` at t2 (cross-reenactment targets)

Every image comes with K (continuous pixels), world->camera E and a validity
mask (pixels inside the original frame). Ava additionally provides the
head->camera pose Q of every image and the head-space registered vertices at
t1 / t2 (decimeters); EmoTalk3D has ``has_pose = 0`` and zero placeholders.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import json
from pathlib import Path

from . import data_ava, data_emo
from .geometry import MM
from .prepare_ava import read_pack

ANGLE_BINS = ((0, 25), (25, 50), (50, 181))


def _pick_targets(angles, exclude, rng, count_bins=ANGLE_BINS):
    chosen = []
    for lo, hi in count_bins:
        cands = [c for c, a in angles.items() if lo <= a < hi and c not in exclude + chosen]
        if not cands:
            cands = [c for c in angles if c not in exclude + chosen]
        chosen.append(cands[rng.integers(len(cands))])
    return chosen


def _pick_aux(angles, source, rng, lo=15, hi=60):
    cands = [c for c, a in angles.items() if lo <= a <= hi and c != source]
    if not cands:
        cands = [c for c in angles if c != source]
    return cands[rng.integers(len(cands))]


def _pick_extra(cams, exclude, count, rng):
    pool = [c for c in cams if c not in exclude]
    if count <= 0:
        return []
    return [pool[j] for j in rng.choice(len(pool), min(count, len(pool)), replace=False)]


class MultiViewHeads(Dataset):
    def __init__(self, ava_root=None, emo_index=None, split="train", ava_prob=0.7,
                 enc_size=448, render_size=256, length=100000, jitter=0.08, seed=0,
                 ava_val_count=10, ava_test_count=10, ava_pack=None, fusion_views=0):
        self.split, self.ava_prob, self.length = split, ava_prob, length
        self.fusion_views = fusion_views  # extra encoder views at t1 (stage-B targets)
        self.enc_size, self.render_size = enc_size, render_size
        self.jitter = jitter if split == "train" else 0.
        self.seed = seed
        self.ava, self.ava_pack = [], Path(ava_pack) if ava_pack else None
        if ava_root:
            names = data_ava.split_captures(ava_root, ava_val_count, ava_test_count)[split]
            if self.ava_pack:
                names = [n for n in names if (self.ava_pack / n / "index.json").is_file()]
            self.ava = [data_ava.AvaCapture(ava_root, n) for n in names]
            if self.ava_pack:
                for cap in self.ava:
                    rec = json.loads((self.ava_pack / cap.name / "index.json").read_text())
                    seg = dict(zip(cap.frames, cap.segments))
                    cap.frames = list(rec["frames"])
                    cap.segments = [seg[f] for f in cap.frames]
                    cap.camera_ids = [c for c in cap.camera_ids if c in set(rec["cameras"])]
            self.ava = [c for c in self.ava if len(c.camera_ids) >= 8 and len(c.frames) >= 2]
            self.ava_reader = data_ava.AvaReader(ava_root)
        self.emo = {}
        if emo_index:
            for rec in data_emo.read_index(emo_index, "train" if split == "train" else "val"):
                self.emo.setdefault(rec["person"], []).append(rec)
            self.emo = {p: r for p, r in self.emo.items() if len(r) >= 2}
        self.emo_people = sorted(self.emo)
        if not self.ava and not self.emo:
            raise ValueError(f"No data for split {split}")
        if not self.ava:
            self.ava_prob = 0.
        if not self.emo:
            self.ava_prob = 1.
        self.epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _rng(self, index):
        if self.split == "train":
            return np.random.default_rng([self.seed, self.epoch, index,
                                          torch.initial_seed() % (2 ** 31)])
        return np.random.default_rng([self.seed, index])

    def __getitem__(self, index):
        rng = self._rng(index)
        last = None
        for attempt in range(8):
            try:
                if rng.random() < self.ava_prob:
                    return self._ava(rng)
                return self._emo(rng)
            except (KeyError, FileNotFoundError, ValueError, OSError) as exc:
                last = exc
        raise RuntimeError(f"8 failed sampling attempts; last: {last!r}")

    # ------------------------------------------------------------------ Ava
    def _ava(self, rng):
        cap = self.ava[rng.integers(len(self.ava))]
        reader = self.ava_reader
        i1 = rng.integers(len(cap.frames))
        stride = 1 if self.ava_pack else 7
        others = [i for i in range(0, len(cap.frames), stride)
                  if cap.segments[i] != cap.segments[i1]]
        i2 = others[rng.integers(len(others))] if others else rng.integers(len(cap.frames))
        f1, f2 = cap.frames[i1], cap.frames[i2]
        if self.ava_pack:
            # One sequential read per frame: every camera + pose + registration.
            packs = {f: read_pack(self.ava_pack / cap.name / f"{f:06d}.pack") for f in (f1, f2)}
            raw = [packs[f1]["head_pose.txt"], packs[f2]["head_pose.txt"],
                   packs[f1]["vertices.ply"], packs[f2]["vertices.ply"]]
        else:
            # Round 1 (concurrent): head poses and registrations of both frames.
            raw = reader.zips.read_many([reader.pose_member(cap, f1), reader.pose_member(cap, f2),
                                         reader.verts_member(cap, f1), reader.verts_member(cap, f2)])
        h1, h2 = reader.decode_pose(raw[0]), reader.decode_pose(raw[1])
        verts = np.stack([data_ava.read_ply_xyz(raw[2]), data_ava.read_ply_xyz(raw[3])]) * MM
        cams = cap.camera_ids
        source = cams[rng.integers(len(cams))]
        ang1 = data_ava.view_angles(cap, h1, source, cams)
        aux = _pick_aux(ang1, source, rng)
        targets = _pick_targets(ang1, [source, aux], rng)
        driver = cams[rng.integers(len(cams))]
        rest = [c for c in cams if c != driver]
        t2 = [rest[j] for j in rng.choice(len(rest), 2, replace=False)]
        extra = _pick_extra(cams, [source, aux], self.fusion_views, rng)
        # Round 2 (concurrent): every distinct image of the sample.
        keys = sorted({(c, f1) for c in [source, aux] + targets + extra} |
                      {(c, f2) for c in [driver] + t2})
        if self.ava_pack:
            blobs = [packs[f][f"cam{c}.avif"] for c, f in keys]
        else:
            blobs = reader.zips.read_many([reader.rgb_member(cap, c, f) for c, f in keys])
        decoded = {k: reader.decode_rgb(b) for k, b in zip(keys, blobs)}

        def get(cam, frame, head, size):
            return reader.view(cap, cam, frame, head, size, rng, self.jitter, decoded[(cam, frame)])

        enc = [get(source, f1, h1, self.enc_size), get(aux, f1, h1, self.enc_size),
               get(driver, f2, h2, self.enc_size)]
        enc += [get(c, f1, h1, self.enc_size) for c in extra]
        tgt = [get(c, f1, h1, self.render_size) for c in [aux] + targets + [source]]
        tgt2 = [get(c, f2, h2, self.render_size) for c in t2]
        out = self._collate(enc, tgt, tgt2, domain=0)
        out["verts"] = torch.from_numpy(verts.astype(np.float32))
        out["has_pose"] = torch.tensor(1.)
        out["angles"] = torch.tensor([ang1[c] for c in [aux] + targets + [source]])
        out["person"] = cap.name
        out["src_cam"] = source
        return out

    # ------------------------------------------------------------------ Emo
    def _emo(self, rng):
        person = self.emo_people[rng.integers(len(self.emo_people))]
        records = self.emo[person]
        j1, j2 = rng.choice(len(records), 2, replace=False)
        r1, r2 = records[j1], records[j2]
        f1 = r1["frames"][rng.integers(len(r1["frames"]))]
        f2 = r2["frames"][rng.integers(len(r2["frames"]))]
        c1, c2 = data_emo.load_calib(r1["calib"]), data_emo.load_calib(r2["calib"])
        cams = data_emo.valid_cameras(c1, r1["cameras"])
        if len(cams) < 6:
            raise ValueError(f"too few valid cameras in {r1['calib']}")
        source = cams[rng.integers(len(cams))]
        ang1 = data_emo.view_angles(c1, source, cams)
        aux = _pick_aux(ang1, source, rng)
        targets = _pick_targets(ang1, [source, aux], rng)
        cams2 = data_emo.valid_cameras(c2, r2["cameras"])
        driver = cams2[rng.integers(len(cams2))]
        rest = [c for c in cams2 if c != driver]
        t2 = [rest[j] for j in rng.choice(len(rest), 2, replace=False)]
        v = lambda rec, cam, f, size: data_emo.view(rec, cam, f, size, rng, self.jitter)
        enc = [v(r1, source, f1, self.enc_size), v(r1, aux, f1, self.enc_size),
               v(r2, driver, f2, self.enc_size)]
        enc += [v(r1, c, f1, self.enc_size) for c in
                _pick_extra(cams, [source, aux], self.fusion_views, rng)]
        tgt = [v(r1, c, f1, self.render_size) for c in [aux] + targets + [source]]
        tgt2 = [v(r2, c, f2, self.render_size) for c in t2]
        out = self._collate(enc, tgt, tgt2, domain=1)
        out["verts"] = torch.zeros(2, data_ava.NUM_VERTS, 3)
        out["has_pose"] = torch.tensor(0.)
        out["angles"] = torch.tensor([ang1[c] for c in [aux] + targets + [source]])
        out["person"] = f"emo_{person}"
        out["src_cam"] = source
        return out

    # ------------------------------------------------------------------ util
    @staticmethod
    def _stack(views):
        out = {
            "image": torch.from_numpy(np.stack([v["image"] for v in views])).permute(0, 3, 1, 2),
            "valid": torch.from_numpy(np.stack([v["valid"] for v in views]))[:, None],
            "K": torch.from_numpy(np.stack([v["K"] for v in views])),
            "E_R": torch.from_numpy(np.stack([v["E_R"] for v in views])),
            "E_t": torch.from_numpy(np.stack([v["E_t"] for v in views])),
        }
        if "Q_R" in views[0]:
            out["Q_R"] = torch.from_numpy(np.stack([v["Q_R"] for v in views]))
            out["Q_t"] = torch.from_numpy(np.stack([v["Q_t"] for v in views]))
        else:
            out["Q_R"] = torch.eye(3).expand(len(views), 3, 3).clone()
            out["Q_t"] = torch.zeros(len(views), 3)
        return out

    def _collate(self, enc, tgt, tgt2, domain):
        return {"enc": self._stack(enc), "tgt": self._stack(tgt), "tgt2": self._stack(tgt2),
                "domain": torch.tensor(domain)}


def collate(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], dict):
            out[key] = {k: torch.stack([b[key][k] for b in batch]) for k in batch[0][key]}
        elif torch.is_tensor(batch[0][key]):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out
