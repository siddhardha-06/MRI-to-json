"""
train_unet3d.py  --  3D U-Net for brain MRI subcortical segmentation.

Data  : FreeSurfer-labeled IXI (576 subjects, 30 labels) from TransMorph repo.
        Each .pkl = (T1 volume, label map), shape 160 x 192 x 224.
Labels: 30 FreeSurfer aseg structures (hippocampus, thalamus, caudate, etc.)

Usage
-----
  python train_unet3d.py --download-only   # one-time download (~1.8 GB)
  python train_unet3d.py --epochs 50       # train
  python train_unet3d.py --eval-only       # evaluate saved checkpoint
"""
import os, sys, time, pickle, argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── FreeSurfer label map (30 structures) ────────────────────────────────────
ALL_LABELS = [
    0, 2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 26, 28, 41, 42, 43, 46, 47, 49, 50, 51,
    52, 53, 54, 58, 60
]
NUM_CLASSES  = len(ALL_LABELS)   # 30
LABEL_TO_IDX = {l: i for i, l in enumerate(ALL_LABELS)}

REPORT_LABELS = {
    "L.Hippocampus":17,"R.Hippocampus":53,
    "L.Thalamus":10,   "R.Thalamus":49,
    "L.Caudate":11,    "R.Caudate":50,
    "L.Putamen":12,    "R.Putamen":51,
    "L.Pallidum":13,   "R.Pallidum":52,
    "L.Amygdala":18,   "R.Amygdala":54,
    "L.Ventricle":4,   "R.Ventricle":43,
}

ROOT      = Path(__file__).resolve().parents[1]   # repo root (this file lives in training/)
DATA_DIR  = ROOT / "IXI_data"
CKPT_DIR  = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results" / "e1_unet_in_domain"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GDRIVE_ID = "1-VQewCVNj5eTtc3eQGhTM2yXBQmgm8Ol"

# ── Dataset ──────────────────────────────────────────────────────────────────
def pkload(path):
    with open(path, "rb") as f:
        return pickle.load(f)

LR_PAIRS = [(2, 41), (3, 42), (4, 43), (7, 46), (8, 47), (10, 49), (11, 50), (12, 51), (13, 52), (17, 53), (18, 54), (26, 58), (28, 60)]
SWAP_IDX = np.arange(NUM_CLASSES)
for _l, _r in LR_PAIRS:
    SWAP_IDX[LABEL_TO_IDX[_l]], SWAP_IDX[LABEL_TO_IDX[_r]] = LABEL_TO_IDX[_r], LABEL_TO_IDX[_l]


def _torchio_transforms(shift_vox, intensity):
    import torchio as tio
    t = []
    if shift_vox:
        # small affine incl. translation: removes field-of-view position as a left/right cue
        t.append(tio.RandomAffine(scales=(0.95, 1.05), degrees=5, translation=shift_vox, image_interpolation="linear",
                                  label_interpolation="nearest", default_pad_value=0, p=0.8))
    if intensity:
        t += [tio.RandomBiasField(coefficients=0.3, p=0.3), tio.RandomGamma(log_gamma=0.3, p=0.3),
              tio.RandomNoise(std=(0, 0.03), p=0.3), tio.RandomBlur(std=(0, 0.5), p=0.3)]
    return tio.Compose(t) if t else None


class IXIDataset(Dataset):
    """flip_mode: 'legacy_noswap' (v1 behaviour, AUDIT K10: flips axis 0 = L-R without swapping labels),
    'swap' (flip and swap the 13 L/R label pairs), 'none'."""
    def __init__(self, pkl_files, augment=False, flip_mode="legacy_noswap", shift_vox=0, intensity=False):
        self.files = pkl_files
        self.augment = augment
        self.flip_mode = flip_mode
        self.tio = _torchio_transforms(shift_vox, intensity) if augment else None
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img, seg = pkload(self.files[idx])
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        seg_idx = np.zeros_like(seg, dtype=np.int64)
        for raw, i in LABEL_TO_IDX.items():
            seg_idx[seg == raw] = i
        if self.augment and self.flip_mode != "none" and torch.rand(1) > 0.5:
            img = img.flip(dims=[1]); seg_idx = seg_idx[::-1].copy()
            if self.flip_mode == "swap":
                seg_idx = SWAP_IDX[seg_idx]
        seg = torch.from_numpy(seg_idx)
        if self.tio is not None:
            import torchio as tio
            subj = self.tio(tio.Subject(image=tio.ScalarImage(tensor=img), seg=tio.LabelMap(tensor=seg[None])))
            img = subj["image"].data.clamp(0, 1).float(); seg = subj["seg"].data[0].long()
        return img, seg

def make_dataloaders(batch_size=1, flip_mode="legacy_noswap", shift_vox=0, intensity=False, limit=None, workers=0, data_dir=None):
    splits = {}
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    for s in ("Train","Val","Test"):
        splits[s] = sorted((data_dir/s).glob("*.pkl")) if (data_dir/s).exists() else []
        if limit:
            splits[s] = splits[s][:limit]   # smoke tests only
    print(f"  Train:{len(splits['Train'])}  Val:{len(splits['Val'])}  Test:{len(splits['Test'])}")
    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    # training loader: CPU augmentation in worker processes overlaps with the GPU step (same samples, same recipe)
    tkw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return (
        DataLoader(IXIDataset(splits["Train"], augment=True, flip_mode=flip_mode, shift_vox=shift_vox, intensity=intensity),
                   batch_size=batch_size, shuffle=True, **tkw),
        DataLoader(IXIDataset(splits["Val"],   augment=False), batch_size=1,          shuffle=False, **kw),
        DataLoader(IXIDataset(splits["Test"],  augment=False), batch_size=1,          shuffle=False, **kw),
    )

# ── 3D U-Net ─────────────────────────────────────────────────────────────────
def conv_block(i, o):
    return nn.Sequential(
        nn.Conv3d(i,o,3,padding=1,bias=False), nn.InstanceNorm3d(o), nn.LeakyReLU(0.2,True),
        nn.Conv3d(o,o,3,padding=1,bias=False), nn.InstanceNorm3d(o), nn.LeakyReLU(0.2,True),
    )

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, num_classes=NUM_CLASSES, base=16):
        super().__init__()
        c = base
        self.e1=conv_block(in_ch,c);   self.e2=conv_block(c,c*2)
        self.e3=conv_block(c*2,c*4);   self.e4=conv_block(c*4,c*8)
        self.bot=conv_block(c*8,c*16)
        self.u4=nn.ConvTranspose3d(c*16,c*8,2,stride=2);  self.d4=conv_block(c*16,c*8)
        self.u3=nn.ConvTranspose3d(c*8, c*4,2,stride=2);  self.d3=conv_block(c*8, c*4)
        self.u2=nn.ConvTranspose3d(c*4, c*2,2,stride=2);  self.d2=conv_block(c*4, c*2)
        self.u1=nn.ConvTranspose3d(c*2, c,  2,stride=2);  self.d1=conv_block(c*2, c)
        self.out=nn.Conv3d(c,num_classes,1)
        self.pool=nn.MaxPool3d(2)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1))
        e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        b=self.bot(self.pool(e4))
        x=self.d4(torch.cat([self.u4(b),e4],1))
        x=self.d3(torch.cat([self.u3(x),e3],1))
        x=self.d2(torch.cat([self.u2(x),e2],1))
        x=self.d1(torch.cat([self.u1(x),e1],1))
        return self.out(x)

# ── Optimized Memory-Efficient Dice Loss ────────────────────────────────────
class FastDiceLoss(nn.Module):
    def forward(self, logits, tgt):
        probs = torch.softmax(logits, dim=1)
        tgt_onehot = torch.zeros_like(probs)
        tgt_onehot.scatter_(1, tgt.unsqueeze(1), 1)
        inter = (probs * tgt_onehot).sum(dim=(0, 2, 3, 4))
        denom = probs.sum(dim=(0, 2, 3, 4)) + tgt_onehot.sum(dim=(0, 2, 3, 4))
        dice = (2.0 * inter + 1.0) / (denom + 1.0)
        return 1.0 - dice[1:].mean()

class Loss(nn.Module):
    def __init__(self, ce_weight=None):
        super().__init__()
        self.dice = FastDiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
    def forward(self, l, t):
        return 0.5 * self.dice(l, t) + 0.5 * self.ce(l, t)

def class_weights(files, cache):
    """CE weights proportional to 1/sqrt(class frequency) over the TRAIN labels, normalised to mean 1."""
    if Path(cache).exists():
        return torch.tensor(json.load(open(cache))["weights"], dtype=torch.float32)
    cnt = np.zeros(NUM_CLASSES)
    for f in files:
        _, seg = pkload(f)
        for raw, i in LABEL_TO_IDX.items():
            cnt[i] += (seg == raw).sum()
    w = 1.0 / np.sqrt(cnt / cnt.sum())
    w = w / w.mean()
    json.dump({"labels": ALL_LABELS, "voxel_counts": cnt.tolist(), "weights": w.tolist(), "n_files": len(files)}, open(cache, "w"), indent=1)
    return torch.tensor(w, dtype=torch.float32)


# ── Dice metric ──────────────────────────────────────────────────────────────
def dice_per_class(pred,true):
    d=np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        i=(pred==c)&(true==c); dn=(pred==c).sum()+(true==c).sum()
        d[c]=(2*i.sum()/dn) if dn>0 else 1.
    return d

# ── Training ─────────────────────────────────────────────────────────────────
def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev} (AMP FP16 Enabled)", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    if args.ce_class_weights and data_dir.resolve() != DATA_DIR.resolve():
        raise SystemExit("--ce-class-weights uses the cached TransMorph Train weights; not valid with --data-dir")
    tdl, vdl, _ = make_dataloaders(args.batch_size, flip_mode=args.flip_mode, shift_vox=args.shift_vox, intensity=args.intensity_aug,
                                   limit=args.limit, workers=args.workers, data_dir=data_dir)
    model = UNet3D(base=args.base_ch).to(dev)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)
    w = class_weights(sorted((DATA_DIR / "Train").glob("*.pkl")), ROOT / "data" / "unet_class_weights.json").to(dev) if args.ce_class_weights else None
    crit = Loss(ce_weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / args.ckpt_name; best = 0.; history = []; start = 0
    last = ckpt.with_name(ckpt.stem + "_last.pth")          # every epoch, for resuming; never used for selection
    hist_f = ckpt_dir / (Path(args.ckpt_name).stem.replace("_best", "") + "_history.json")
    if (ckpt.exists() or last.exists()) and args.reset:
        # Never overwrite a checkpoint. Retraining must use a new --ckpt-name.
        raise SystemExit(f"{ckpt} exists; refusing to overwrite it. Pass a new --ckpt-name for a fresh run.")
    resume = last if last.exists() else ckpt if ckpt.exists() else None
    if resume is not None:
        ck = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start = ck.get("epoch", 0) + 1
        best = ck.get("best_dice", 0.)
        history = json.load(open(hist_f)) if hist_f.exists() else []
        print(f"Resumed from {resume.name} at epoch {start}, best dice {best:.4f}", flush=True)
    elif args.init_ckpt:
        # fine-tuning: weights only from an existing checkpoint (fresh optimizer and schedule); the source is never written
        init = CKPT_DIR / args.init_ckpt
        model.load_state_dict(torch.load(init, map_location=dev, weights_only=False)["model"])
        print(f"Initialised from {init.name} (weights only)", flush=True)

    total_batches = len(tdl)
    for ep in range(start, args.epochs):
        model.train(); loss_sum = 0.; t0 = time.time()
        for batch_idx, (img, seg) in enumerate(tdl):
            img, seg = img.to(dev), seg.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                out = model(img)
                loss = crit(out, seg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                spd = (time.time() - t0) / (batch_idx + 1)
                print(f"  [Ep {ep+1:3d}/{args.epochs}] Batch {batch_idx+1:3d}/{total_batches} - Loss: {loss.item():.4f} ({spd:.2f}s/it)", flush=True)
        sched.step(); avg = loss_sum / len(tdl)
        model.eval(); vd = []
        with torch.no_grad():
            for img, seg in vdl:
                with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                    pred = model(img.to(dev)).argmax(1).cpu().numpy()[0]
                vd.append(dice_per_class(pred, seg.numpy()[0]))
        md = np.mean(vd, 0); fg = md[1:].mean()
        min_cls = float(md[1:].min()); worst = ALL_LABELS[1 + int(np.argmin(md[1:]))]
        rep = "  ".join(f"{n}={md[LABEL_TO_IDX[v]]:.3f}" for n, v in list(REPORT_LABELS.items())[:4])
        print(f"Ep {ep+1:3d}/{args.epochs}  loss={avg:.4f}  val={fg:.4f}  {rep}  {time.time()-t0:.0f}s", flush=True)
        history.append({"epoch": ep+1, "loss": avg, "val_dice": float(fg), "val_min_class_dice": min_cls, "worst_label": worst,
                        "val_dice_per_class": {str(l): float(md[i]) for i, l in enumerate(ALL_LABELS) if l}})
        if min_cls < args.min_class_dice and ep + 1 >= 10:
            print(f"  WARNING: class {worst} val Dice {min_cls:.3f} < {args.min_class_dice}", flush=True)
        if fg > best and min_cls >= args.min_class_dice:
            best=fg
            torch.save({"epoch":ep,"model":model.state_dict(),"optimizer":opt.state_dict(),"best_dice":best,
                        "recipe": {k: getattr(args, k) for k in ("flip_mode","shift_vox","intensity_aug","ce_class_weights","min_class_dice","seed","lr","epochs","data_dir","init_ckpt")}},ckpt)
            print(f"  Saved best (dice={best:.4f})")
        recipe = {k: getattr(args, k) for k in ("flip_mode","shift_vox","intensity_aug","ce_class_weights","min_class_dice","seed","lr","epochs","data_dir","init_ckpt")}
        torch.save({"epoch": ep, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best_dice": best, "recipe": recipe}, last)
        json.dump(history, open(hist_f, "w"), indent=2)  # every epoch
    print(f"Done. Best val Dice: {best:.4f}")

# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(args):
    """In-domain evaluation (E1 core), rewritten in v2 Phase 1.

    All 29 foreground classes; Dice is NaN when a class is absent from both prediction and
    reference (never 1.0); HD95 and ASSD in voxel units (TransMorph volumes are 1 mm isotropic,
    so these are mm under that assumption); per-subject long CSV; bootstrap 95% CIs; provenance.
    Writes results/e1_unet_in_domain/<run>/ and never touches checkpoints/.
    """
    import csv
    from neurosphere.metrics import bootstrap_ci, dice, surface_distances
    from neurosphere.provenance import provenance, sha256_file

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(getattr(args, "ckpt_dir", None) or CKPT_DIR) / getattr(args, "ckpt_name", "unet3d_best.pth")
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}."); return
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model = UNet3D(base=args.base_ch).to(dev); model.load_state_dict(ck["model"]); model.eval()
    split = getattr(args, "split", "Test")
    files = sorted((DATA_DIR / split).glob("*.pkl"))
    limit = getattr(args, "limit", None)
    if limit:
        files = files[:limit]
    use_amp = bool(getattr(args, "amp", False)) and dev.type == "cuda"
    do_surface = not getattr(args, "no_surface", False)
    run = time.strftime("%Y%m%d-%H%M%S") + f"_{split.lower()}" + (f"_limit{limit}" if limit else "")
    # smoke tests (--limit) go to work/, which is not committed; only full runs go to results/
    default_out = (ROOT / "work" / "smoke" / "e1_unet_in_domain" / run) if limit else (RESULTS_DIR / run)
    out = Path(getattr(args, "out_dir", None) or default_out)
    out.mkdir(parents=True, exist_ok=True)
    fg = [l for l in ALL_LABELS if l != 0]
    rows, t_infer = [], []
    with torch.no_grad():
        for f in files:
            img, seg = pkload(f)
            x = torch.from_numpy(img.astype(np.float32))[None, None].to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                p = model(x).argmax(1)[0]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_infer.append(time.perf_counter() - t0)
            pred = np.asarray(ALL_LABELS)[p.cpu().numpy()]
            for l in fg:
                a, b = pred == l, seg == l
                hd95, assd = surface_distances(a, b) if do_surface else (float("nan"), float("nan"))
                rows.append({"file": f.name, "label": l, "dice": dice(a, b), "hd95_vox": hd95, "assd_vox": assd,
                             "n_ref_vox": int(b.sum()), "n_pred_vox": int(a.sum())})
            print(f"  {f.name}: mean fg Dice {np.nanmean([r['dice'] for r in rows[-len(fg):]]):.4f}", flush=True)
    with open(out / "per_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def per_label(key):
        res = {}
        for l in fg:
            v = np.array([r[key] for r in rows if r["label"] == l], float)
            fin = v[np.isfinite(v)]
            res[str(l)] = {"mean": float(fin.mean()) if len(fin) else None, "sd": float(fin.std(ddof=1)) if len(fin) > 1 else None,
                           "ci95": list(bootstrap_ci(v)) if len(fin) else None, "n_finite": int(len(fin)), "n_nan": int(len(v) - len(fin))}
            for k in ("mean", "sd", "ci95"):
                if res[str(l)][k] is None:
                    res[str(l)][k + "_reason"] = "no finite values (class absent from prediction and/or reference)"
        return res
    subj_mean = np.array([np.nanmean([r["dice"] for r in rows if r["file"] == f.name]) for f in files])
    metrics = {
        "split": split, "n_subjects": len(files), "checkpoint": {"file": ckpt.name, "sha256": sha256_file(ckpt), "epoch_index": ck.get("epoch")},
        "precision": "autocast fp16" if use_amp else "fp32",
        "mean_fg_dice_over_subjects": float(subj_mean.mean()), "mean_fg_dice_ci95": list(bootstrap_ci(subj_mean)),
        "per_subject_mean_fg_dice_sd": float(subj_mean.std(ddof=1)) if len(subj_mean) > 1 else None,
        "dice": per_label("dice"), "hd95_vox": per_label("hd95_vox"), "assd_vox": per_label("assd_vox"),
        "n_class_subject_pairs_absent_in_both": int(sum(1 for r in rows if r["n_ref_vox"] == 0 and r["n_pred_vox"] == 0)),
        "inference_s_per_subject": {"median": float(np.median(t_infer)), "iqr": [float(np.percentile(t_infer, 25)), float(np.percentile(t_infer, 75))],
                                    "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"},
        "units": "HD95/ASSD in voxels of the TransMorph grid (1 mm isotropic)",
        "provenance": provenance(seed=None, seed_reason="deterministic inference", data_subset=f"IXI_data/{split}" + (f" first {limit}" if limit else "")),
    }
    if metrics["per_subject_mean_fg_dice_sd"] is None:
        metrics["per_subject_mean_fg_dice_sd_reason"] = "needs at least 2 subjects"
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"\n  Mean fg Dice ({split}, n={len(files)}): {metrics['mean_fg_dice_over_subjects']:.4f}  -> {out}")

# ── Download ─────────────────────────────────────────────────────────────────
def download_dataset():
    import zipfile
    DATA_DIR.mkdir(exist_ok=True)
    zp=DATA_DIR/"IXI_data.zip"
    print("Downloading IXI+FreeSurfer labels (~1.8 GB) from Google Drive...")
    try:
        import gdown; gdown.download(id=GDRIVE_ID,output=str(zp),quiet=False)
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print(f"Download manually: https://drive.google.com/file/d/{GDRIVE_ID}/view")
        print(f"Save as: {zp.resolve()}  then re-run script.")
        return False
    print("Extracting..."); zipfile.ZipFile(zp).extractall(DATA_DIR.parent); zp.unlink()
    print("Done."); return True

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--download-only",action="store_true")
    p.add_argument("--eval-only",    action="store_true")
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--batch-size",type=int,   default=1)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--base-ch",   type=int,   default=16)
    p.add_argument("--reset",     action="store_true")
    p.add_argument("--ckpt-name", default="unet3d_best.pth", help="checkpoint file in checkpoints/ (never overwritten)")
    p.add_argument("--split",     default="Test", choices=["Train","Val","Test"], help="--eval-only: split to evaluate")
    p.add_argument("--limit",     type=int, default=None, help="--eval-only: first N subjects (smoke test)")
    p.add_argument("--amp",       action="store_true", help="--eval-only: autocast fp16 inference")
    p.add_argument("--no-surface",action="store_true", help="--eval-only: skip HD95/ASSD")
    p.add_argument("--out-dir",   default=None, help="--eval-only: output folder (default results/e1_unet_in_domain/<run>)")
    p.add_argument("--flip-mode", default="legacy_noswap", choices=["legacy_noswap","swap","none"])
    p.add_argument("--shift-vox", type=float, default=0, help="max random translation (voxels) via TorchIO RandomAffine")
    p.add_argument("--intensity-aug", action="store_true")
    p.add_argument("--ce-class-weights", action="store_true")
    p.add_argument("--min-class-dice", type=float, default=0.0, help="only save checkpoints whose worst val class Dice >= this")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--workers",   type=int, default=4, help="training DataLoader worker processes (augmentation on CPU)")
    p.add_argument("--ckpt-dir",  default=None, help="checkpoint folder (default checkpoints/; smoke runs use work/smoke/)")
    p.add_argument("--data-dir",  default=None, help="training/selection data in the TransMorph pkl layout (<dir>/Train, <dir>/Val); default IXI_data")
    p.add_argument("--init-ckpt", default=None, help="fine-tune: initialise weights from this checkpoint in checkpoints/ (fresh optimizer)")
    p.add_argument("--recipe",    choices=["v1","v2"], default="v1",
                   help="v2 = Option A (PHASE_1/2): --flip-mode swap --shift-vox 5 --intensity-aug --ce-class-weights --min-class-dice 0.5")
    args=p.parse_args()
    if args.recipe == "v2":
        args.flip_mode, args.shift_vox, args.intensity_aug, args.ce_class_weights, args.min_class_dice = "swap", 5.0, True, True, 0.5
        pass
    if not args.eval_only and not args.download_only and args.ckpt_name == "unet3d_best.pth":
        raise SystemExit("training needs a new --ckpt-name (e.g. unet3d_v2_best.pth); the shipped checkpoint unet3d_best.pth is never overwritten")
    if args.download_only: download_dataset(); sys.exit(0)
    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*.pkl")):
        if not download_dataset(): sys.exit(1)
    if args.eval_only:
        evaluate(args)
    else:
        train(args)
        if args.data_dir:
            # fine-tuning runs never evaluate on TransMorph Test automatically (acceptance rule: Val only)
            print("Fine-tuning run: no automatic Test evaluation. Evaluate on Val (E2-FS) per the acceptance rule.")
        else:
            evaluate(args)
