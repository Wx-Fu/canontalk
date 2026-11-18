"""Real OmniAvatar WanModel (tiny, random): batched forward == per-sample forward."""
import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "OmniAvatar"))

try:  # sequence-parallel helpers are unused at sp_size=1; stub them if xfuser is absent
    import xfuser  # noqa: F401
except ImportError:
    import types
    for name in ("xfuser", "xfuser.core", "xfuser.core.distributed"):
        sys.modules.setdefault(name, types.ModuleType(name))
    d = sys.modules["xfuser.core.distributed"]
    d.get_sequence_parallel_rank = lambda: 0
    d.get_sequence_parallel_world_size = lambda: 1
    d.get_sp_group = lambda: None


def tiny_model(module):
    torch.manual_seed(0)
    return module.WanModel(dim=96, in_dim=51, ffn_dim=192, out_dim=16, text_dim=32, freq_dim=32,
                           eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=8,
                           has_image_input=False, audio_hidden_size=8).eval()


def inputs(b):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, 16, 3, 8, 8, generator=g)
    y = torch.randn(b, 35, 3, 8, 8, generator=g)
    ctx = torch.randn(b, 5, 32, generator=g)
    audio = torch.randn(b, 9, 10752, generator=g)        # 9 frames -> 3 latents
    t = torch.tensor([300., 700.])[:b]
    return x, t, ctx, y, audio


def load(name):
    from OmniAvatar.utils import args_config
    args_config.args = argparse.Namespace(use_audio=True, sp_size=1, model_config={"in_dim": 51})
    return importlib.import_module(name)


def test_batched_forward_matches_per_sample():
    m = load("OmniAvatar.models.wan_video_dit")
    model = tiny_model(m)
    x, t, ctx, y, audio = inputs(2)
    with torch.no_grad():
        both = model(x, timestep=t, context=ctx, y=y, audio_emb=audio)
        one = [model(x[i:i + 1], timestep=t[i:i + 1], context=ctx[i:i + 1], y=y[i:i + 1],
                     audio_emb=audio[i:i + 1]) for i in range(2)]
    assert torch.allclose(both, torch.cat(one), atol=1e-4), (both - torch.cat(one)).abs().max()


def test_batch1_identical_to_original_code():
    orig = Path("/tmp/tsc/wan_video_dit.orig.py")
    if not orig.is_file():
        return
    m = load("OmniAvatar.models.wan_video_dit")
    spec = importlib.util.spec_from_file_location("OmniAvatar.models.wan_video_dit_orig", orig)
    o = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o)
    new, old = tiny_model(m), tiny_model(o)
    old.load_state_dict(new.state_dict())
    x, t, ctx, y, audio = inputs(1)
    with torch.no_grad():
        a = new(x, timestep=t, context=ctx, y=y, audio_emb=audio)
        b = old(x, timestep=t, context=ctx, y=y, audio_emb=audio)
    assert torch.equal(a, b)
