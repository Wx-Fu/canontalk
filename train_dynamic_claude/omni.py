"""OmniAvatar (Wan2.1-1.3B + audio LoRA) with an extra VISR condition.

DiT input channels: [noisy 16 | ref latent 16 | gen-mask 1 | VISR render latent 16 |
alpha 1 | depth 1] = 51. The 18 new patch-embedding input channels start at zero,
so at step 0 the network is exactly the pretrained OmniAvatar.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "omniavatar"))
WAN = "/apdcephfs_nj8/share_305278643/wenxinfu/models/omniavatar/Wan2.1-T2V-1.3B"
WAV2VEC = "/apdcephfs_nj8/share_305278643/wenxinfu/models/interactive/wav2vec2-base-960h"
OMNI = str(ROOT / "model")
PROMPT = "A person's head speaking, natural lighting, realistic face, sharp details."
EXTRA = 18
TRAINABLE = ("lora_", "patch_embedding.", "audio_proj.", "audio_cond_projs.")


def load_models(device, wan=WAN, wav2vec=WAV2VEC, omni=OMNI):
    from OmniAvatar.utils import args_config
    args_config.args = argparse.Namespace(use_audio=True, sp_size=1, model_config={"in_dim": 33})
    from OmniAvatar.models.model_manager import ModelManager
    from OmniAvatar.wan_video import WanVideoPipeline
    from OmniAvatar.models.wav2vec import Wav2VecModel
    from OmniAvatar.utils.io_utils import load_state_dict
    from transformers import Wav2Vec2FeatureExtractor
    from peft import LoraConfig, inject_adapter_in_model

    manager = ModelManager(device="cpu", infer=True)
    manager.load_models([[str(Path(wan) / "diffusion_pytorch_model.safetensors")],
                         str(Path(wan) / "models_t5_umt5-xxl-enc-bf16.pth"),
                         str(Path(wan) / "Wan2.1_VAE.pth")], torch_dtype=torch.bfloat16, device="cpu")
    pipe = WanVideoPipeline.from_model_manager(manager, torch_dtype=torch.bfloat16, device=str(device))
    cfg = json.loads((Path(omni) / "config.json").read_text())
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                      target_modules=cfg["lora_target_modules"].split(","))
    pipe.dit = inject_adapter_in_model(lora, pipe.dit)
    weights = load_state_dict(str(Path(omni) / "pytorch_model.pt"))
    missing, unexpected = pipe.dit.load_state_dict(weights, strict=False)
    if unexpected or any(k in missing for k in ("patch_embedding.weight", "audio_proj.proj.weight")):
        raise RuntimeError(f"OmniAvatar load mismatch: {missing[:6]} {unexpected[:6]}")
    del weights, manager
    extend_patch_embedding(pipe.dit, EXTRA)
    pipe.dit.to(device)
    set_trainable(pipe.dit)
    pipe.vae.to(device).eval().requires_grad_(False)
    pipe.text_encoder.to(device).eval().requires_grad_(False)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    speech = Wav2VecModel.from_pretrained(wav2vec, local_files_only=True).to(device).eval()
    speech.requires_grad_(False)
    return pipe, speech, extractor


@torch.no_grad()
def extend_patch_embedding(dit, extra):
    old = dit.patch_embedding
    new = nn.Conv3d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride,
                    old.padding, bias=old.bias is not None).to(old.weight.dtype)
    new.weight.zero_()
    new.weight[:, :old.in_channels] = old.weight
    if old.bias is not None:
        new.bias.copy_(old.bias)
    dit.patch_embedding = new


def set_trainable(dit):
    for name, p in dit.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE) or "lora_" in name)
        if p.requires_grad:
            p.data = p.data.float()
    dit.train()


def trainable_state(dit):
    return {k: v.detach().cpu() for k, v in dit.state_dict().items()
            if k.startswith(TRAINABLE) or "lora_" in k}


@torch.no_grad()
def encode_prompt(pipe, prompt=PROMPT, negative=None):
    ctx = pipe.encode_prompt(prompt)["context"].detach()
    neg = pipe.encode_prompt(negative, positive=False)["context"].detach() if negative else None
    return ctx, neg


@torch.no_grad()
def audio_features(speech, extractor, wave, frames, device):
    """wav2vec features for every generated frame (no pinned reference frame)."""
    x = extractor(list(wave.cpu().numpy()), sampling_rate=16000,
                  return_tensors="pt").input_values.to(device)
    states = speech(x, seq_len=frames, output_hidden_states=True)
    return torch.cat((states.last_hidden_state, *states.hidden_states), -1).to(torch.bfloat16)


def latent_pool(x, latent_frames):
    """[B,C,T,H,W] pixel-rate maps -> [B,C,L,H/8,W/8] on the Wan causal latent grid
    (latent 0 = frame 0, latent j = frames 4j-3..4j)."""
    b, c, t, h, w = x.shape
    x = F.interpolate(x.flatten(1, 2), size=(h // 8, w // 8), mode="area").view(b, c, t, h // 8, w // 8)
    first = x[:, :, :1]
    rest = x[:, :, 1:].view(b, c, latent_frames - 1, 4, h // 8, w // 8).mean(3)
    return torch.cat((first, rest), 2)


def to_vae(x, device):
    return x.to(device, torch.bfloat16) * 2 - 1


@torch.no_grad()
def build_y(pipe, ref, render, alpha, depth):
    """Condition y [B,35,L,h,w] from the reference image [B,3,H,W] and the VISR
    render video [B,3,T,H,W] (+ alpha/depth [B,1,T,H,W]); all in [0,1]."""
    dev = pipe.device
    ref_lat = pipe.encode_video(to_vae(ref[:, :, None], dev), tiled=False).to(dev)
    cond = pipe.encode_video(to_vae(render, dev), tiled=False).to(dev)
    l = cond.shape[2]
    geo = latent_pool(torch.cat((alpha, depth), 1).float(), l).to(cond.dtype)
    return torch.cat((ref_lat.expand(-1, -1, l, -1, -1), torch.ones_like(cond[:, :1]), cond, geo), 1)


@torch.no_grad()
def encode_conditions(pipe, ref, video, render, alpha, depth):
    """Target latent [B,16,L,h,w] and condition y [B,35,L,h,w]."""
    target = pipe.encode_video(to_vae(video, pipe.device), tiled=False).to(pipe.device)
    return target, build_y(pipe, ref, render, alpha, depth)


def flow_loss(dit, target, y, audio, context, prefix=None, checkpoint=True):
    """Flow matching on all latent frames. prefix [B] bool: latent 0 given clean
    (continuation mode for chunked long videos) and excluded from the loss."""
    b = target.shape[0]
    sigma = torch.rand(b, 1, 1, 1, 1, device=target.device, dtype=target.dtype)
    noise = torch.randn_like(target)
    noisy = (1 - sigma) * target + sigma * noise
    weight = torch.ones_like(target[:, :1])
    if prefix is not None and prefix.any():
        p = prefix.view(-1, 1, 1, 1, 1)
        noisy[:, :, :1] = torch.where(p, target[:, :, :1], noisy[:, :, :1])
        y = y.clone()
        y[:, 16:17, :1] = torch.where(p, 0., y[:, 16:17, :1])
        weight[:, :, :1] = torch.where(p, 0., weight[:, :, :1])
    pred = dit(noisy, timestep=sigma.flatten() * 1000, context=context.expand(b, -1, -1),
               y=y, audio_emb=audio, use_gradient_checkpointing=checkpoint)
    err = (pred.float() - (noise - target).float()).square().mean(1, keepdim=True)
    return (err * weight).sum() / weight.expand_as(err).sum().clamp_min(1)


@torch.no_grad()
def sample(pipe, y, audio, context, negative=None, steps=25, cfg_audio=4.5, seed=0,
           prefix_latent=None, init_latent=None, init_strength=1.0, shift=5.0):
    """Euler sampling with the Wan flow scheduler; CFG drops audio and uses the
    negative prompt (upstream convention). Optional clean prefix latent (chunk
    continuation) and SDEdit-style start from an init latent (e.g. the VISR render)."""
    dit, dev = pipe.dit, pipe.device
    shape = (y.shape[0], 16, *y.shape[2:])
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.randn(shape, device=dev, dtype=torch.bfloat16, generator=g)
    pipe.scheduler.set_timesteps(steps, denoising_strength=init_strength, shift=shift)
    ts = pipe.scheduler.timesteps
    latent = noise if init_latent is None else pipe.scheduler.add_noise(init_latent, noise, ts[0])
    if prefix_latent is not None:
        y = y.clone()
        y[:, 16:17, :1] = 0
    dit.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for t in ts:
            if prefix_latent is not None:
                latent[:, :, :1] = prefix_latent
            tt = t.reshape(1).to(dev, torch.bfloat16).expand(shape[0])
            v = dit(latent, timestep=tt, context=context.expand(shape[0], -1, -1), y=y, audio_emb=audio)
            if cfg_audio != 1:
                neg = context if negative is None else negative
                v0 = dit(latent, timestep=tt, context=neg.expand(shape[0], -1, -1), y=y,
                         audio_emb=torch.zeros_like(audio))
                v = v0 + cfg_audio * (v - v0)
            latent = pipe.scheduler.step(v, t, latent)
    if prefix_latent is not None:
        latent[:, :, :1] = prefix_latent
    return latent
