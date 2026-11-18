"""Standard talking-head evaluation of the VISR-conditioned generator.

Two phases (so the expensive generation is parallel and metrics can be re-run):

  generate  (torchrun, sharded)  for every eval clip: reference image + GT audio +
            GT head pose (read by VISR from the GT video) -> long video (default
            3 s, chunked with clean-prefix continuation). Default motion mode
            "audio": reference expression, lips only from audio (deployment case).
            Saves gen.mp4 / gt.mp4 (with audio), ref.png, meta.json (+ PSNR /
            SSIM / LPIPS / pose error, computed on the fly).
  score     (1 GPU)  ArcFace identity (CSIM), SyncNet LSE-C / LSE-D, FID and an
            FVD proxy, aggregated per task -> summary.json, summary.md.

Tasks: emo_novel (EmoTalk3D val, reference and target from different cameras),
emo_same (same camera), celebv, multidialog (val splits, same camera).

Metric notes:
  CSIM_ref  ArcFace cosine(generated frames, reference image)
  CSIM_gt   ArcFace cosine(generated frame t, GT frame t)  (identity at the target view)
  LSE-C/D   SyncNet confidence / min distance (Wav2Lip protocol: face track crop with
            crop_scale 0.4, offset search +-15); also reported for GT videos (ceiling)
  av_offset |audio-video offset| found by SyncNet, frames (0 = in sync)
  FID       torchvision Inception-v3 pool features, generated vs GT frames
  FVD-R3D   Frechet distance of torchvision R3D-18 (Kinetics) clip features:
            a proxy, NOT the I3D-based FVD of the literature (no I3D weights here)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EVAL_MODELS = Path("/apdcephfs_nj8/share_305278643/wenxinfu/models/eval")
TASKS = ("emo_novel", "emo_same", "celebv", "multidialog")


# =============================================================== generate
def ssim(a, b):
    """Gaussian-window SSIM on [N,3,H,W] in [0,1]."""
    g = torch.exp(-(torch.arange(11, device=a.device) - 5.).pow(2) / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None]) / g.sum() ** 2
    w = w.expand(3, 1, 11, 11)
    mu = lambda x: F.conv2d(x, w, padding=5, groups=3)
    ma, mb = mu(a), mu(b)
    va, vb, cov = mu(a * a) - ma ** 2, mu(b * b) - mb ** 2, mu(a * b) - ma * mb
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return s.mean((1, 2, 3))


def eval_items(index, per_task, frames, size):
    from .data import TalkingWindows
    items = []
    for task in TASKS:
        ds_name = {"emo_novel": "emotalk3d", "emo_same": "emotalk3d"}.get(task, task)
        same = 1.0 if task == "emo_same" else 0.0
        ds = TalkingWindows(index, "val", frames, size, {ds_name: 1.0}, seed=2024, same_cam=same,
                            ref_cands=1)
        if ds.by_ds.get(ds_name):
            items += [(task, ds, i) for i in range(per_task)]
    return items


@torch.no_grad()
def generate(a):
    from . import omni
    from .data import collate
    from .infer import generate_long
    from .probe import geodesic_deg
    from .utils import save_comparison
    from .visr_cond import VISRCondition
    import lpips

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    frames = int(round(a.seconds * 25)) + 1
    items = eval_items(Path(a.index) / "val.jsonl", a.per_task, frames, a.size)[rank::world]
    pipe, speech, extractor = omni.load_models(dev)
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["dit"]
    _, unexpected = pipe.dit.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    pipe.dit.eval()
    context, negative = omni.encode_prompt(pipe)
    pipe.text_encoder = None
    visr = VISRCondition(a.visr, dev)
    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    clips = Path(a.out) / "clips"
    for task, ds, i in items:
        out = clips / f"{task}_{i:03d}"
        if (out / "meta.json").is_file():
            continue
        try:
            batch = collate([ds[i]])
        except RuntimeError as exc:
            print(json.dumps({"skip": f"{task}_{i}", "error": str(exc)}), flush=True)
            continue
        ref = batch["ref"][:, 0].to(dev)
        gt = batch["video"].to(dev)
        k_ref, k_tgt = batch["K_ref"].to(dev), batch["K_tgt"].to(dev)
        st = visr.reference(ref, k_ref)
        r, t, e = visr.motion(gt, k_tgt)
        if a.motion == "audio":
            e = st["e"].float()[:, None].expand_as(e)
        gen = generate_long(pipe, speech, extractor, visr, st, ref, r, t, e, k_tgt,
                            batch["wave"][0], context, negative, a.window, a.steps, a.cfg,
                            seed=i, size=a.size, log=False)
        n = min(gen.shape[2], gt.shape[2])
        g, y = gen[0, :, :n].to(dev), gt[0, :, :n]
        gf, yf = g.permute(1, 0, 2, 3), y.permute(1, 0, 2, 3)
        g_r, _, _ = visr.motion(g[None], k_tgt)
        meta = {"task": task, "dataset": batch["dataset"][0], "person": batch["person"][0],
                "clip": batch["clip"][0], "view": batch["view"][0], "frames": n,
                "psnr": float(-10 * torch.log10(((gf - yf) ** 2).mean().clamp_min(1e-8))),
                "ssim": float(ssim(gf, yf).mean()),
                "lpips": float(torch.cat([lp(gf[j:j + 8] * 2 - 1, yf[j:j + 8] * 2 - 1).flatten()
                                          for j in range(0, n, 8)]).mean()),
                "pose_err_deg": float(geodesic_deg(g_r[0], r[0, :n]).mean()),
                "ref_vs_tgt_deg": float(geodesic_deg(st["R"].float()[:, None], r).mean())}
        out.mkdir(parents=True, exist_ok=True)
        wave = batch["wave"][0][: int(n / 25 * 16000)]
        save_comparison(out / "gen.mp4", [g.cpu()], wave)
        save_comparison(out / "gt.mp4", [y.cpu()], wave)
        import cv2
        cv2.imwrite(str(out / "ref.png"),
                    (batch["ref"][0, 0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1])
        (out / "meta.json").write_text(json.dumps(meta))
        print(json.dumps({"rank": rank, **meta}), flush=True)


# ================================================================== score
def read_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    cap.release()
    return out


class Faces:
    """InsightFace buffalo_l: SCRFD detection + ArcFace (w600k_r50) embeddings."""

    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", root=str(EVAL_MODELS / "insightface"),
                                allowed_modules=["detection", "recognition"],
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(512, 512))

    def __call__(self, bgr):
        faces = self.app.get(bgr)
        if not faces:
            return None, None
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return f.normed_embedding, f.bbox


class Sync:
    """SyncNet LSE-C / LSE-D (Wav2Lip evaluation protocol)."""

    def __init__(self, device):
        sys.path.insert(0, str(EVAL_MODELS / "syncnet_python"))
        from SyncNetInstance import SyncNetInstance
        self.net = SyncNetInstance(device=device)
        self.net.loadParameters(str(EVAL_MODELS / "syncnet/syncnet_v2.model"))
        self.net.eval()

    @staticmethod
    def face_track_video(frames, boxes, wav, path, crop_scale=0.4):
        """Crop like syncnet_python run_pipeline (median-smoothed box, crop_scale)."""
        import cv2
        from scipy.signal import medfilt
        valid = [b for b in boxes if b is not None]
        if len(valid) < len(boxes) * 0.5:
            return False
        last = valid[0]
        filled = []
        for b in boxes:
            last = b if b is not None else last
            filled.append(last)
        b = np.array(filled, np.float64)
        k = min(13, len(b) // 2 * 2 - 1) if len(b) > 2 else 1
        s = medfilt(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) / 2, k)
        mx = medfilt((b[:, 0] + b[:, 2]) / 2, k)
        my = medfilt((b[:, 1] + b[:, 3]) / 2, k)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "v.mp4"
            vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 25, (224, 224))
            for im, bs, x, y in zip(frames, s, mx, my):
                bsi = int(bs * (1 + 2 * crop_scale))
                pad = cv2.copyMakeBorder(im, bsi, bsi, bsi, bsi, cv2.BORDER_CONSTANT, value=(110, 110, 110))
                y, x = y + bsi, x + bsi
                face = pad[int(y - bs):int(y + bs * (1 + 2 * crop_scale)),
                           int(x - bs * (1 + crop_scale)):int(x + bs * (1 + crop_scale))]
                vw.write(cv2.resize(face, (224, 224)))
            vw.release()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(wav),
                            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)], check=True)
        return True

    def score(self, video_path):
        opt = argparse.Namespace(tmp_dir=tempfile.mkdtemp(), reference="s", batch_size=20, vshift=15)
        _, _, dists = self.net.evaluate(opt, str(video_path))
        mdist = dists.mean(0)
        offset = opt.vshift - int(mdist.argmin())        # frames; 0 = in sync
        return float(np.median(mdist) - mdist.min()), float(mdist.min()), offset  # LSE-C, LSE-D


class Frechet:
    """Inception-v3 frame features (FID) and R3D-18 clip features (FVD proxy)."""

    def __init__(self, device):
        import torchvision
        os.environ.setdefault("TORCH_HOME", str(EVAL_MODELS / "torch"))
        self.dev = device
        inc = torchvision.models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        inc.fc = torch.nn.Identity()
        self.inc = inc.eval().to(device)
        r3d = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
        r3d.fc = torch.nn.Identity()
        self.r3d = r3d.eval().to(device)

    @torch.no_grad()
    def frame_feats(self, frames):
        x = torch.from_numpy(np.stack([f[..., ::-1] for f in frames]).copy()).permute(0, 3, 1, 2)
        x = F.interpolate(x.float().div(255).to(self.dev), size=(299, 299), mode="bilinear",
                          align_corners=False)
        m = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        return self.inc((x - m) / s).cpu().numpy()

    @torch.no_grad()
    def clip_feats(self, frames, length=16):
        out = []
        for start in range(0, len(frames) - length + 1, length):
            x = torch.from_numpy(np.stack([f[..., ::-1] for f in frames[start:start + length]]).copy())
            x = x.permute(3, 0, 1, 2).float().div(255)[None].to(self.dev)
            x = F.interpolate(x[0].permute(1, 0, 2, 3), size=(112, 112), mode="bilinear",
                              align_corners=False).permute(1, 0, 2, 3)[None]
            m = torch.tensor([0.43216, 0.394666, 0.37645], device=self.dev).view(1, 3, 1, 1, 1)
            s = torch.tensor([0.22803, 0.22145, 0.216989], device=self.dev).view(1, 3, 1, 1, 1)
            out.append(self.r3d((x - m) / s).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512))

    @staticmethod
    def distance(a, b):
        from scipy import linalg
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        mu1, mu2 = a.mean(0), b.mean(0)
        s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
        covmean = covmean.real
        return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def score(a):
    import cv2
    dev = torch.device("cuda")
    faces, sync, fre = Faces(), Sync(dev), Frechet(dev)
    rows, feats = [], {}
    clip_dirs = sorted(p for p in (Path(a.out) / "clips").iterdir() if (p / "meta.json").is_file())
    for d in clip_dirs:
        meta = json.loads((d / "meta.json").read_text())
        gen, gt = read_video(d / "gen.mp4"), read_video(d / "gt.mp4")
        n = min(len(gen), len(gt))
        gen, gt = gen[:n], gt[:n]
        ref_emb, _ = faces(cv2.imread(str(d / "ref.png")))
        g_emb, g_box, t_emb, t_box = [], [], [], []
        for fg, ft in zip(gen, gt):
            e1, b1 = faces(fg)
            e2, b2 = faces(ft)
            g_emb.append(e1); g_box.append(b1); t_emb.append(e2); t_box.append(b2)
        cs_ref = [float(e @ ref_emb) for e in g_emb if e is not None and ref_emb is not None]
        cs_gt = [float(e1 @ e2) for e1, e2 in zip(g_emb, t_emb) if e1 is not None and e2 is not None]
        meta["csim_ref"] = float(np.mean(cs_ref)) if cs_ref else float("nan")
        meta["csim_gt"] = float(np.mean(cs_gt)) if cs_gt else float("nan")
        meta["face_det_rate"] = float(np.mean([e is not None for e in g_emb]))
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "gt.mp4"), "-vn", "-ac", "1",
                            "-ar", "16000", str(wav)], check=True)
            for name, frames, boxes in (("gen", gen, g_box), ("gt", gt, t_box)):
                path = Path(tmp) / f"{name}_face.mp4"
                ok = sync.face_track_video(frames, boxes, wav, path)
                c, dd, off = sync.score(path) if ok else (float("nan"), float("nan"), float("nan"))
                meta[f"lse_c_{name}"], meta[f"lse_d_{name}"] = c, dd
                meta[f"av_offset_abs_{name}"] = abs(off)
        f = feats.setdefault(meta["task"], {"fid_gen": [], "fid_gt": [], "fvd_gen": [], "fvd_gt": []})
        f["fid_gen"].append(fre.frame_feats(gen[::4])); f["fid_gt"].append(fre.frame_feats(gt[::4]))
        f["fvd_gen"].append(fre.clip_feats(gen)); f["fvd_gt"].append(fre.clip_feats(gt))
        rows.append(meta)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()}),
              flush=True)
    keys = ["psnr", "ssim", "lpips", "csim_ref", "csim_gt", "lse_c_gen", "lse_d_gen",
            "lse_c_gt", "lse_d_gt", "av_offset_abs_gen", "av_offset_abs_gt", "pose_err_deg", "ref_vs_tgt_deg", "face_det_rate"]
    summary = {}
    for task in TASKS:
        sel = [r for r in rows if r["task"] == task]
        if not sel:
            continue
        s = {k: float(np.nanmean([r[k] for r in sel])) for k in keys}
        f = {k: np.concatenate(v) for k, v in feats[task].items()}
        s["fid"] = Frechet.distance(f["fid_gen"], f["fid_gt"])
        s["fvd_r3d"] = Frechet.distance(f["fvd_gen"], f["fvd_gt"])
        s["n"] = len(sel)
        summary[task] = s
    (Path(a.out) / "summary.json").write_text(json.dumps(summary, indent=1))
    (Path(a.out) / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cols = ["n", "psnr", "ssim", "lpips", "csim_ref", "csim_gt", "lse_c_gen", "lse_d_gen",
            "lse_c_gt", "lse_d_gt", "av_offset_abs_gen", "fid", "fvd_r3d", "pose_err_deg"]
    md = ["| task | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for task, s in summary.items():
        md.append(f"| {task} | " + " | ".join(f"{s[c]:.3f}" if c != "n" else str(s[c]) for c in cols) + " |")
    (Path(a.out) / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("generate", "score"))
    p.add_argument("--checkpoint")
    p.add_argument("--visr")
    p.add_argument("--index", default="data_cache/dyn_index")
    p.add_argument("--out", required=True)
    p.add_argument("--per-task", type=int, default=50)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--window", type=int, default=33)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--motion", choices=("audio", "gt"), default="audio",
                   help="audio: reference expression, lips from audio; gt: VISR expression from GT")
    a = p.parse_args()
    if a.phase == "generate":
        generate(a)
    else:
        score(a)


if __name__ == "__main__":
    main()
