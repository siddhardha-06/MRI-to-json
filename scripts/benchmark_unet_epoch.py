r"""Measure 3D U-Net training throughput to estimate the cost of the Phase 2 retraining proposal.

Runs a few optimisation steps exactly as training/train_unet3d.train does (same model, loss,
AdamW, AMP, grad clipping, batch size 1, real Train DataLoader with its flip augmentation),
plus validation inference, and extrapolates to one epoch (403 train + 58 val subjects).
It never saves weights: the model starts from random initialisation and is discarded.

Output: results/phase2_planning/unet3d_epoch_benchmark.json
Usage: .\.venv\Scripts\python scripts\benchmark_unet_epoch.py [--steps 30] [--val 10]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosphere.provenance import provenance  # noqa: E402
from training.train_unet3d import Loss, UNet3D, dice_per_class, make_dataloaders  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--val", type=int, default=10)
ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--recipe", choices=["v1", "v2"], default="v1", help="v2 = Option A augmentation (swap flip, shift 5, TorchIO intensity, class-weighted CE)")
a = ap.parse_args()

torch.manual_seed(0)
dev = torch.device("cuda")
aug = dict(flip_mode="swap", shift_vox=5.0, intensity=True) if a.recipe == "v2" else {}
tdl, vdl, _ = make_dataloaders(1, workers=a.workers, **aug)
model = UNet3D(base=16).to(dev)
if a.recipe == "v2":
    from training.train_unet3d import class_weights
    crit = Loss(ce_weight=class_weights(sorted((ROOT / "IXI_data" / "Train").glob("*.pkl")), ROOT / "data" / "unet_class_weights.json").to(dev))
else:
    crit = Loss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.amp.GradScaler("cuda")
torch.cuda.reset_peak_memory_stats()

load_t, step_t = [], []
it = iter(tdl)
model.train()
for i in range(a.warmup + a.steps):
    t0 = time.perf_counter()
    img, seg = next(it)
    t1 = time.perf_counter()
    img, seg = img.to(dev), seg.to(dev)
    opt.zero_grad()
    with torch.amp.autocast("cuda"):
        loss = crit(model(img), seg)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    if i >= a.warmup:
        load_t.append(t1 - t0); step_t.append(t2 - t1)

val_t = []
model.eval()
with torch.no_grad():
    for k, (img, seg) in enumerate(vdl):
        if k >= a.val:
            break
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            pred = model(img.to(dev)).argmax(1).cpu().numpy()[0]
        dice_per_class(pred, seg.numpy()[0])
        val_t.append(time.perf_counter() - t0)

n_train = len(tdl.dataset); n_val = len(vdl.dataset)
per_train_iter = float(np.mean(load_t) + np.mean(step_t))
epoch_s = per_train_iter * n_train + float(np.mean(val_t)) * n_val
res = {
    "what": "throughput of the current 3D U-Net training loop (batch 1, AMP), extrapolated to one epoch",
    "n_train_subjects": n_train, "n_val_subjects": n_val,
    "measured_train_steps": a.steps, "warmup_steps_excluded": a.warmup, "measured_val_subjects": len(val_t),
    "data_load_s_per_iter": {"mean": float(np.mean(load_t)), "median": float(np.median(load_t))},
    "gpu_step_s_per_iter": {"mean": float(np.mean(step_t)), "median": float(np.median(step_t))},
    "val_s_per_subject": {"mean": float(np.mean(val_t))},
    "estimated_epoch_s": epoch_s, "estimated_epoch_min": epoch_s / 60,
    "estimated_hours_for_epochs": {str(e): round(epoch_s * e / 3600, 2) for e in (50, 100, 150)},
    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    "device": torch.cuda.get_device_name(0),
    "recipe": a.recipe, "train_loader_workers": a.workers,
    "provenance": provenance(seed=0, data_subset=f"IXI_data/Train first {a.warmup + a.steps} shuffled, IXI_data/Val first {len(val_t)}"),
}
out = ROOT / "results" / "phase2_planning" / ("unet3d_epoch_benchmark.json" if a.recipe == "v1" else f"unet3d_epoch_benchmark_recipe_v2_workers{a.workers}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2))
