"""Small helpers: side-by-side comparison videos with audio."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def save_comparison(path, videos, wave, fps=25):
    """videos: list of [3,T,H,W] or [3,H,W] (still) tensors in [0,1]; same H.
    Writes an mp4 with the panels side by side and the 16 kHz audio."""
    import cv2
    import soundfile as sf
    t = max(v.shape[1] for v in videos if v.dim() == 4)
    h = videos[0].shape[-2]
    frames = []
    for i in range(t):
        tiles = []
        for v in videos:
            x = v[:, min(i, v.shape[1] - 1)] if v.dim() == 4 else v
            x = x.float().clamp(0, 1).permute(1, 2, 0).numpy()
            if x.shape[0] != h:
                x = cv2.resize(x, (h, h))
            tiles.append(x)
        frames.append((np.concatenate(tiles, 1) * 255).astype(np.uint8)[..., ::-1])
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        raw, wav = Path(tmp) / "v.mp4", Path(tmp) / "a.wav"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (frames[0].shape[1], frames[0].shape[0]))
        for f in frames:
            writer.write(f)
        writer.release()
        sf.write(str(wav), wave.float().numpy(), 16000)
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(wav),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                            "-shortest", str(path)], capture_output=True)
        if r.returncode != 0:  # no ffmpeg/libx264: keep the silent video
            path.write_bytes(raw.read_bytes())
    return str(path)
