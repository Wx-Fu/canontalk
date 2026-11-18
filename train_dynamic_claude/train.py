"""Train OmniAvatar conditioned on frozen VISR renders (dynamic stage).

  torchrun --standalone --nproc_per_node=8 -m train_dynamic_claude.train \
      --index data_cache/dyn_index --visr train_static_claude/runs/visr_v41_a/best.pt \
      --out train_dynamic_claude/runs/vco_a
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from . import omni
from .data import TalkingWindows, collate
from .utils import save_comparison
from .visr_cond import VISRCondition


def options(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="prepare.py output directory")
    p.add_argument("--visr", required=True, help="frozen VISR checkpoint")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wan", default=omni.WAN)
    p.add_argument("--wav2vec", default=omni.WAV2VEC)
    p.add_argument("--omni", default=omni.OMNI)
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=5e-5, help="LoRA / audio layers")
    p.add_argument("--lr-cond", type=float, default=2e-4, help="patch embedding (new channels)")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--probs", type=float, nargs=3, default=(0.4, 0.3, 0.3),
                   help="sampling weights: emotalk3d celebv multidialog")
    p.add_argument("--batch", type=int, default=4, help="clips per GPU")
    p.add_argument("--init", type=Path, default=None,
                   help="warm start from a previous run's checkpoint (new optimizer, step 0)")
    p.add_argument("--ava-root", default=None, help="Ava-256 root (adds 1-frame view-change pairs)")
    p.add_argument("--ava-pack", default=None)
    p.add_argument("--ava-every", type=int, default=6, help="every N-th step is an Ava pair step")
    p.add_argument("--ava-batch", type=int, default=16, help="Ava 1-frame pairs per GPU")
    p.add_argument("--p-pick-gap", type=float, default=0.7,
                   help="choose the reference candidate with the largest pose gap")
    p.add_argument("--p-mouth-blur", type=float, default=0.5,
                   help="blur the mouth in the render (lips must come from audio)")
    p.add_argument("--p-degrade", type=float, default=0.5,
                   help="corrupt the render (it is coarse guidance, not ground truth)")
    p.add_argument("--p-static", type=float, default=0.6,
                   help="render with the reference expression (audio must drive the mouth)")
    p.add_argument("--p-drop-audio", type=float, default=0.1)
    p.add_argument("--p-drop-cond", type=float, default=0.1)
    p.add_argument("--p-prefix", type=float, default=0.2,
                   help="clean first latent given (chunk continuation)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--render-count", type=int, default=2)
    p.add_argument("--render-steps", type=int, default=25)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def setup():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda", 0)
    dist.init_process_group("nccl")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)


class Step:
    def __init__(self, pipe, speech, extractor, visr, context, args, device):
        self.pipe, self.speech, self.extractor, self.visr = pipe, speech, extractor, visr
        self.context, self.args, self.device = context, args, device

    @torch.no_grad()
    def prepare(self, batch, train=True, gen=None):
        a, dev = self.args, self.device
        b = batch["ref"].shape[0]
        ref = batch["ref"].to(dev)
        video = batch["video"].to(dev)
        k_ref, k_tgt = batch["K_ref"].to(dev), batch["K_tgt"].to(dev)
        rand = lambda: torch.rand(b, device=dev, generator=gen)
        no = torch.zeros(b, dtype=torch.bool, device=dev)
        single = video.shape[2] == 1                        # Ava pair step: no audio
        static = (rand() < a.p_static) if train and not single else no
        pick = (rand() < a.p_pick_gap) if train else no
        mouth = (rand() < a.p_mouth_blur) if train and not single else no
        degrade = (rand() < a.p_degrade) if train else no
        render, alpha, depth, ref, _ = self.visr(ref, k_ref, video, k_tgt, static, pick, mouth,
                                                 degrade)
        target, y = omni.encode_conditions(self.pipe, ref, video, render, alpha, depth)
        if single:
            audio = torch.zeros(b, 1, 10752, device=dev, dtype=torch.bfloat16)
        else:
            audio = omni.audio_features(self.speech, self.extractor, batch["wave"], a.frames, dev)
        prefix = None
        if train:
            drop_c = (rand() < a.p_drop_cond).view(-1, 1, 1, 1, 1)
            y = torch.cat((y[:, :17], torch.where(drop_c, 0., y[:, 17:])), 1)
            drop_a = (rand() < a.p_drop_audio).view(-1, 1, 1)
            audio = torch.where(drop_a, 0., audio)
            prefix = (rand() < a.p_prefix) if not single else None
        return target, y, audio, prefix, render, ref

    def loss(self, model, batch, train=True, gen=None):
        target, y, audio, prefix, _, _ = self.prepare(batch, train, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return omni.flow_loss(model, target, y, audio, self.context, prefix)


def lr_lambda(step, args):
    if step < args.warmup:
        return (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1., t)))


@torch.no_grad()
def validate(step_fn, val_set, rank, world, args, step):
    dit = step_fn.pipe.dit
    dit.eval()
    losses = {}
    for i in range(rank, args.val_batches, world):
        batch = collate([val_set[i]])
        torch.manual_seed(1000 + i)  # same noise/sigma each validation
        loss = step_fn.loss(dit, batch, train=False).item()
        losses.setdefault(batch["dataset"][0], []).append(loss)
    sums = torch.zeros(3, 2, device=step_fn.device, dtype=torch.float64)
    for j, d in enumerate(("emotalk3d", "celebv", "multidialog")):
        sums[j] = torch.tensor([sum(losses.get(d, [])), len(losses.get(d, []))])
    if world > 1:
        dist.all_reduce(sums)
    out = {f"val_flow/{d}": (sums[j, 0] / sums[j, 1]).item()
           for j, d in enumerate(("emotalk3d", "celebv", "multidialog")) if sums[j, 1] > 0}
    if rank == 0:
        for i in range(args.render_count):
            batch = collate([val_set[i]])
            target, y, audio, _, render, ref = step_fn.prepare(batch, train=False)
            neg = step_fn.context
            latent = omni.sample(step_fn.pipe, y, audio, step_fn.context, neg,
                                 steps=args.render_steps, seed=i)
            gen = step_fn.pipe.decode_video(latent, tiled=False).float().add(1).div(2).clamp(0, 1)
            gt = batch["video"].to(gen.device)
            out["val_psnr/" + batch["dataset"][0]] = float(
                -10 * torch.log10(((gen - gt) ** 2).mean().clamp_min(1e-8)))
            save_comparison(args.out / f"val_{step:07d}_{i}_{batch['dataset'][0]}.mp4",
                            [ref[0].cpu(), render[0].cpu(), gen[0].cpu(), batch["video"][0]],
                            batch["wave"][0])
    dit.train()
    return out


def main(argv=None):
    args = options(argv)
    rank, world, device = setup()
    torch.manual_seed(args.seed + rank)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))
    probs = dict(zip(("emotalk3d", "celebv", "multidialog"), args.probs))
    index = Path(args.index)
    train_set = TalkingWindows(index / "train.jsonl", "train", args.frames, args.size, probs,
                               seed=args.seed * 100 + rank)
    val_set = TalkingWindows(index / "val.jsonl", "val", args.frames, args.size,
                             {"emotalk3d": 0.5, "celebv": 0.25, "multidialog": 0.25}, seed=7)
    loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                        collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=4)
    ava_it = None
    if args.ava_root and args.ava_pack:
        from .data import AvaPairs
        ava = AvaPairs(args.ava_root, args.ava_pack, "train", args.size, seed=args.seed * 100 + rank)
        ava_it = iter(DataLoader(ava, batch_size=args.ava_batch, num_workers=max(2, args.workers // 2),
                                 collate_fn=collate, pin_memory=True, persistent_workers=True))
        if rank == 0:
            print(json.dumps({"ava_pair_captures": len(ava.caps)}), flush=True)
    pipe, speech, extractor = omni.load_models(device, args.wan, args.wav2vec, args.omni)
    context, _ = omni.encode_prompt(pipe)
    pipe.text_encoder.cpu()
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    visr = VISRCondition(args.visr, device)
    dit = pipe.dit
    groups = [{"params": [p for n, p in dit.named_parameters() if p.requires_grad
                          and not n.startswith("patch_embedding.")], "lr": args.lr},
              {"params": list(dit.patch_embedding.parameters()), "lr": args.lr_cond}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    step = 0
    if args.init and not args.resume:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        _, unexpected = dit.load_state_dict(state["dit"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys {unexpected[:5]}")
        if rank == 0:
            print(json.dumps({"init": str(args.init), "from_step": state["step"]}), flush=True)
        del state
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = dit.load_state_dict(state["dit"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys {unexpected[:5]}")
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
    model = DDP(dit, device_ids=[device.index], broadcast_buffers=False) if world > 1 else dit
    step_fn = Step(pipe, speech, extractor, visr, context, args, device)
    if rank == 0:
        n = sum(p.numel() for p in dit.parameters() if p.requires_grad)
        print(json.dumps({"world": world, "trainable_M": round(n / 1e6, 1),
                          "train_clips": train_set.counts(), "val_clips": val_set.counts()}),
              flush=True)
    log = (args.out / "log.jsonl").open("a") if rank == 0 else None
    it, t0, t_data = iter(loader), time.time(), 0.
    while step < args.steps:
        td = time.time()
        ava_step = ava_it is not None and step % args.ava_every == args.ava_every - 1
        batch = next(ava_it) if ava_step else next(it)
        t_data += time.time() - td
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * lr_lambda(step, args)
        loss = step_fn.loss(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in dit.parameters() if p.requires_grad], 1.0)
        if torch.isfinite(grad):
            opt.step()
        step += 1
        if rank == 0 and step % args.log_every == 0:
            rec = {"step": step, "flow": round(loss.item(), 5), "grad": round(float(grad), 3),
                   "lr": opt.param_groups[0]["lr"], "dataset": batch["dataset"][0],
                   "sec_per_step": round((time.time() - t0) / args.log_every, 3),
                   "data_sec": round(t_data / args.log_every, 3),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            t0, t_data = time.time(), 0.
        if step % args.val_every == 0 or step == args.steps:
            metrics = validate(step_fn, val_set, rank, world, args, step)
            if rank == 0:
                rec = {"step": step, "val": metrics}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save({"dit": omni.trainable_state(dit), "optimizer": opt.state_dict(),
                        "step": step, "args": vars(args)}, args.out / f"step_{step:07d}.pt")
            (args.out / "latest.json").write_text(json.dumps({"step": step}))
        if world > 1 and step % args.val_every == 0:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
