"""
Small Sample Learning for Defect Classification
Semiconductor Solutions Challenge 2026 — Problem A (Intel)

Two-Stage Classification Pipeline:
  Stage 1: Binary classifier (good vs defect)
  Stage 2: Defect subtype classifier (only on defect images, 8 classes)

Optimized for RTX 5090 (30 GB VRAM):
- ConvNeXt-Small from timm (pretrained ImageNet-21k)
- Stage 1: BCEWithLogitsLoss (plain, no weighting)
- Stage 2: Focal Loss (gamma=2, no alpha, no label smoothing)
- Heavy augmentation via albumentations
- Cosine annealing LR with linear warmup
- Weighted oversampling for class imbalance
- Mixed precision training (AMP)
- Exponential Moving Average (EMA)
- Full-resolution inference benchmarking (1500x2500)
"""

import os, sys, time, json, argparse, math, copy
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.amp import autocast, GradScaler

import torchvision.transforms as T
import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False
    print("WARNING: albumentations not installed, using torchvision transforms")



# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.3):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WaferDefectDataset(Dataset):
    VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}

    def __init__(self, root, transform=None, use_albumentations=False, merge_map=None):
        self.root = Path(root)
        self.transform = transform
        self.use_album = use_albumentations and HAS_ALBUM
        self.samples = []

        raw_classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        # merge_map: dict mapping old class name -> new class name
        # e.g. {"defect3": "rare", "defect4": "rare"}
        if merge_map:
            merged_names = set()
            for c in raw_classes:
                merged_names.add(merge_map.get(c, c))
            self.classes = sorted(merged_names)
        else:
            self.classes = raw_classes
            merge_map = {}

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in raw_classes:
            mapped = merge_map.get(cls, cls)
            cls_path = self.root / cls
            for f in sorted(cls_path.iterdir()):
                if f.suffix.lower() in self.VALID_EXT:
                    self.samples.append((str(f), self.class_to_idx[mapped]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale per problem spec
        img_rgb = Image.merge("RGB", [img, img, img])  # 3-ch for pretrained

        if self.use_album and self.transform is not None:
            img_np = np.array(img_rgb)
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        elif self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = T.ToTensor()(img_rgb)

        return img_tensor, y


# ---------------------------------------------------------------------------
# Two-Stage Dataset Wrappers
# ---------------------------------------------------------------------------
class BinaryWrapperDataset(Dataset):
    """Wraps a WaferDefectDataset to produce binary labels: 0=good, 1=defect."""
    def __init__(self, base_dataset, good_class_idx):
        self.base = base_dataset
        self.good_idx = good_class_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return img, 0 if y == self.good_idx else 1


class DefectRemapDataset(Dataset):
    """Wraps a WaferDefectDataset to remap defect labels to 0..N-1.
    Only use with indices that are defect samples (not 'good')."""
    def __init__(self, base_dataset, label_map):
        self.base = base_dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return img, self.label_map[y]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_album_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.12),
                        hole_width_range=(0.05, 0.12), fill=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_album_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tv_train_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])


def get_tv_val_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(model_name, num_classes, pretrained=True, drop_rate=0.1):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes,
                              drop_rate=drop_rate)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}, params: {n_params:,}, drop: {drop_rate}")
    return model


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, mixup_alpha=0.3):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Mixup
        if mixup_alpha > 0 and np.random.random() < 0.5:
            x, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
            with autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            # For accuracy, use original y
            preds = logits.argmax(1)
            correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        else:
            with autocast("cuda", enabled=use_amp):
                logits = model(x)
                if logits.shape[-1] == 1:
                    loss = criterion(logits.squeeze(1), y.float())
                else:
                    loss = criterion(logits, y)
            if logits.shape[-1] == 1:
                correct += ((logits.squeeze(1) > 0).long() == y).sum().item()
            else:
                correct += (logits.argmax(1) == y).sum().item()

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * y.size(0)
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    correct = total = 0
    all_preds, all_true = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(x)
        if logits.shape[-1] == 1:
            preds = (logits.squeeze(1) > 0).long().cpu()
        else:
            preds = logits.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(y.tolist())
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc, np.array(all_preds), np.array(all_true)


# ---------------------------------------------------------------------------
# Two-Stage Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_two_stage(binary_model, defect_model, loader, device, use_amp,
                       good_class_idx, defect_label_map_inv):
    """Chain binary -> defect subtype classifiers for full 9-class prediction."""
    binary_model.eval()
    defect_model.eval()
    all_preds, all_true = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        batch_size = x.size(0)

        # Stage 1: Binary prediction
        with autocast("cuda", enabled=use_amp):
            binary_logits = binary_model(x)
        if binary_logits.shape[-1] == 1:
            binary_preds = (binary_logits.squeeze(1) > 0).long()
        else:
            binary_preds = binary_logits.argmax(1)  # 0=good, 1=defect

        # Initialize all predictions as good
        final_preds = torch.full((batch_size,), good_class_idx, dtype=torch.long)

        # Stage 2: For defect predictions, get subtype
        defect_mask = binary_preds == 1
        if defect_mask.any():
            defect_x = x[defect_mask]
            with autocast("cuda", enabled=use_amp):
                defect_logits = defect_model(defect_x)
            defect_preds = defect_logits.argmax(1).cpu()
            # Map defect subtype indices back to original class indices
            defect_indices = defect_mask.nonzero(as_tuple=True)[0]
            for i, dp in enumerate(defect_preds):
                final_preds[defect_indices[i]] = defect_label_map_inv[dp.item()]

        all_preds.extend(final_preds.tolist())
        all_true.extend(y.tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    acc = (all_preds == all_true).mean()
    return acc, all_preds, all_true


# ---------------------------------------------------------------------------
# Test Time Augmentation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_tta(model, dataset, indices, device, img_size, use_amp=False, n_augments=5):
    """Test-time augmentation: average predictions over multiple augmented views."""
    model.eval()

    # Build TTA transforms
    if HAS_ALBUM:
        tta_transforms = [
            get_album_val_transform(img_size),  # original
            A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0),
                       A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
            A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0),
                       A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
            A.Compose([A.Resize(img_size, img_size), A.RandomRotate90(p=1.0),
                       A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
            A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0),
                       A.VerticalFlip(p=1.0),
                       A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        ]
    else:
        return None, None, None  # Skip TTA without albumentations

    all_probs = []
    all_true = []
    base_ds = WaferDefectDataset(dataset.root, transform=None, use_albumentations=False)

    for idx in indices:
        path, y = base_ds.samples[idx]
        img = Image.open(path).convert("L")
        img_rgb = Image.merge("RGB", [img, img, img])
        img_np = np.array(img_rgb)
        all_true.append(y)

        probs_sum = None
        for tf in tta_transforms[:n_augments]:
            aug = tf(image=img_np)
            tensor = aug["image"].unsqueeze(0).to(device)
            with autocast("cuda", enabled=use_amp):
                logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu()
            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs

        avg_probs = probs_sum / n_augments
        all_probs.append(avg_probs)

    all_probs = torch.cat(all_probs, dim=0)
    all_preds = all_probs.argmax(1).numpy()
    all_true = np.array(all_true)
    acc = (all_preds == all_true).mean()
    return acc, all_preds, all_true


# ---------------------------------------------------------------------------
# Learning Curve
# ---------------------------------------------------------------------------
def learning_curve_experiment(dataset, val_indices, args, device, num_classes, class_names, merge_map=None):
    print("\n" + "=" * 60)
    print("LEARNING CURVE EXPERIMENT")
    print("=" * 60)

    all_train_indices = [i for i in range(len(dataset)) if i not in set(val_indices)]
    targets = [dataset.samples[i][1] for i in all_train_indices]
    fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    results = []

    use_album = HAS_ALBUM
    val_tf = get_album_val_transform(args.img_size) if use_album else get_tv_val_transform(args.img_size)
    val_ds = WaferDefectDataset(args.data_root, transform=val_tf, use_albumentations=use_album, merge_map=merge_map)
    val_subset = Subset(val_ds, val_indices)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    for frac in fractions:
        n_samples = max(num_classes, int(len(all_train_indices) * frac))
        if frac < 1.0:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=args.seed)
            sub_idx, _ = next(sss.split(all_train_indices, targets))
            train_idx = [all_train_indices[i] for i in sub_idx]
        else:
            train_idx = all_train_indices

        train_tf = get_album_train_transform(args.img_size) if use_album else get_tv_train_transform(args.img_size)
        train_ds = WaferDefectDataset(args.data_root, transform=train_tf, use_albumentations=use_album, merge_map=merge_map)
        train_subset = Subset(train_ds, train_idx)

        sub_targets = [dataset.samples[i][1] for i in train_idx]
        counts = Counter(sub_targets)
        cw = {c: 1.0 / max(counts.get(c, 1), 1) for c in range(num_classes)}
        sw = [cw[t] for t in sub_targets]
        sampler = WeightedRandomSampler(sw, len(sw), replacement=True)
        loader = DataLoader(train_subset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True)

        model = build_model(args.model, num_classes, pretrained=args.pretrained, drop_rate=0.3).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = GradScaler("cuda", enabled=args.amp)
        lc_epochs = max(8, int(args.lc_epochs * max(frac, 0.3)))

        alpha = compute_focal_alpha(dataset, train_idx, num_classes)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        criterion.alpha = criterion.alpha.to(device)

        best_acc = 0.0
        for ep in range(1, lc_epochs + 1):
            train_one_epoch(model, loader, criterion, optimizer, device, scaler, args.amp, mixup_alpha=0.2)
            acc, _, _ = evaluate(model, val_loader, device, args.amp)
            best_acc = max(best_acc, acc)

        print(f"  {frac:>5.0%} ({len(train_idx):>5d} samples, {lc_epochs:>2d} ep) -> Val Acc: {best_acc:.4f}")
        results.append({"fraction": frac, "n_samples": len(train_idx), "val_acc": best_acc})
        del model
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Inference Benchmark
# ---------------------------------------------------------------------------
@torch.no_grad()
def benchmark_inference(model, device, img_size, full_res=(1500, 2500), n_runs=50, use_amp=False):
    model.eval()
    results = {}

    for name, size in [("train_res", (img_size, img_size)), ("full_res", full_res)]:
        dummy = torch.randn(1, 3, size[0], size[1]).to(device)

        for _ in range(10):
            with autocast("cuda", enabled=use_amp):
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast("cuda", enabled=use_amp):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        avg = np.mean(times)
        std = np.std(times)
        p95 = np.percentile(times, 95)
        results[name] = {"avg_ms": round(avg, 1), "std_ms": round(std, 1),
                         "p95_ms": round(p95, 1), "resolution": f"{size[0]}x{size[1]}"}
        status = "PASS" if avg < 1000 else "WARN"
        print(f"  {name:12s} ({size[0]:>4d}x{size[1]:<4d}): {avg:7.1f} ± {std:.1f} ms  p95={p95:.1f} ms [{status}]")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history, out_dir, prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title_prefix = f"{prefix} " if prefix else ""

    axes[0].plot(history["train_loss"], linewidth=2, color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title_prefix}Training Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", linewidth=2, color="#1f77b4")
    axes[1].plot(history["val_acc"], label="Val", linewidth=2, color="#2ca02c")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title_prefix}Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["lr"], linewidth=2, color="#9467bd")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title(f"{title_prefix}LR Schedule"); axes[2].grid(True, alpha=0.3)

    fname = f"{prefix.lower().replace(' ', '_')}_training_curves.png" if prefix else "training_curves.png"
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (Test Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


def plot_accuracy_vs_occurrence(cm, counts, class_names, num_classes, out_dir):
    occurrence = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=float)
    per_class_acc = (cm.diagonal() / np.maximum(cm.sum(axis=1), 1)).astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(occurrence, per_class_acc, c=per_class_acc, cmap="RdYlGn",
                         s=200, edgecolors="black", linewidths=1.5, vmin=0, vmax=1, zorder=5)

    for i, name in enumerate(class_names):
        ax.annotate(name, (occurrence[i], per_class_acc[i]),
                    textcoords="offset points", xytext=(10, 8), fontsize=10, fontweight="bold")

    ax.set_xlabel("Class Occurrence in Training Set", fontsize=12)
    ax.set_ylabel("Per-Class Accuracy on Test Set", fontsize=12)
    ax.set_title("Classification Accuracy vs Defect Class Occurrence", fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="85% target")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, label="Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_occurrence.png"), dpi=200)
    plt.close()

    np.savetxt(os.path.join(out_dir, "acc_vs_occurrence.csv"),
               np.stack([occurrence, per_class_acc], axis=1),
               delimiter=",", header="occurrence,per_class_accuracy", comments="")


def plot_learning_curve(lc_results, out_dir):
    n_samples = [r["n_samples"] for r in lc_results]
    accs = [r["val_acc"] for r in lc_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_samples, accs, "o-", linewidth=2.5, markersize=10, color="#1f77b4", zorder=5)
    for ns, acc in zip(n_samples, accs):
        ax.annotate(f"{acc:.1%}", (ns, acc), textcoords="offset points",
                    xytext=(0, 14), fontsize=11, ha="center", fontweight="bold")

    ax.set_xlabel("Number of Training Samples", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Learning Curve — How Quickly the Model Learns from Limited Data", fontsize=13)
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="85% target")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve.png"), dpi=200)
    plt.close()


def plot_per_class_metrics(report_dict, class_names, out_dir):
    """Bar chart of precision, recall, f1 per class."""
    prec = [report_dict[c]["precision"] for c in class_names]
    rec = [report_dict[c]["recall"] for c in class_names]
    f1 = [report_dict[c]["f1-score"] for c in class_names]

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#1f77b4")
    ax.bar(x, rec, w, label="Recall", color="#2ca02c")
    ax.bar(x + w, f1, w, label="F1-Score", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Per-Class Metrics")
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def compute_focal_alpha(dataset, train_indices, num_classes):
    targets = [dataset.samples[i][1] for i in train_indices]
    counts = Counter(targets)
    total = len(targets)
    alpha = []
    for c in range(num_classes):
        freq = counts.get(c, 1) / total
        alpha.append(1.0 / (num_classes * freq))
    s = sum(alpha)
    alpha = [a * num_classes / s for a in alpha]
    return alpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _train_stage(model, train_loader, val_loader, criterion, device, args, use_amp,
                 stage_name, out_dir):
    """Generic training loop used by both stages."""
    ema = ModelEMA(model, decay=0.9995) if args.use_ema else None
    if ema:
        print(f"  EMA enabled (decay=0.9995)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=args.warmup_epochs,
                                      total_epochs=args.epochs, min_lr=1e-6)
    scaler = GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(out_dir, f"best_{stage_name}_model.pt")
    best_ema_path = os.path.join(out_dir, f"best_{stage_name}_ema_model.pt")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp,
            mixup_alpha=0.0)

        if ema:
            ema.update(model)

        eval_model = ema.ema if ema else model
        val_acc, _, _ = evaluate(eval_model, val_loader, device, use_amp)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            if ema:
                torch.save(ema.state_dict(), best_ema_path)
            patience_counter = 0
            improved = " *BEST*"
        else:
            patience_counter += 1

        print(f"  Ep {epoch:03d}/{args.epochs} | Loss {train_loss:.4f} | "
              f"Train {train_acc:.3f} | Val {val_acc:.3f} | "
              f"LR {current_lr:.2e} | {elapsed:.1f}s{improved}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"  Best {stage_name} Val Acc: {best_val_acc:.4f}")

    # Load best model
    if ema and os.path.exists(best_ema_path):
        ema.load_state_dict(torch.load(best_ema_path, map_location=device, weights_only=True))
        best_model = ema.ema
        print(f"  Using EMA model for {stage_name}")
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        best_model = model

    return best_model, history, best_val_acc


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    use_amp = args.amp and device.type == "cuda"
    print(f"Mixed precision (AMP): {use_amp}")

    # --- Class Merging ---
    merge_map = {}
    if args.merge_classes:
        for pair in args.merge_classes:
            old, new = pair.split(":")
            merge_map[old] = new
        print(f"Class merging: {merge_map}")

    # --- Dataset ---
    base_dataset = WaferDefectDataset(args.data_root, transform=None, merge_map=merge_map)
    num_classes = len(base_dataset.classes)
    class_names = base_dataset.classes
    print(f"\nDataset: {len(base_dataset)} images, {num_classes} classes: {class_names}")

    # Identify good class and defect classes
    assert "good" in class_names, "Expected 'good' class in dataset"
    good_class_idx = class_names.index("good")
    defect_class_indices = sorted([i for i in range(num_classes) if i != good_class_idx])
    defect_class_names = [class_names[i] for i in defect_class_indices]
    num_defect_classes = len(defect_class_names)

    # Defect label mappings (original -> 0..N-1 and back)
    defect_label_map = {orig: new for new, orig in enumerate(defect_class_indices)}
    defect_label_map_inv = {new: orig for orig, new in defect_label_map.items()}

    print(f"Two-Stage Pipeline:")
    print(f"  Stage 1: Binary (good vs defect)")
    print(f"  Stage 2: {num_defect_classes} defect subtypes: {defect_class_names}")

    # --- Stratified Split ---
    all_targets = [s[1] for s in base_dataset.samples]
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_split, random_state=args.seed)
    trainval_idx, test_idx = next(sss_test.split(range(len(base_dataset)), all_targets))

    trainval_targets = [all_targets[i] for i in trainval_idx]
    relative_val = args.val_split / (1 - args.test_split)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=args.seed)
    train_idx_rel, val_idx_rel = next(sss_val.split(range(len(trainval_idx)), trainval_targets))
    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx = [trainval_idx[i] for i in val_idx_rel]

    train_targets = [all_targets[i] for i in train_idx]
    train_counts = Counter(train_targets)

    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    for c in range(num_classes):
        print(f"  {class_names[c]:>10s}: train={train_counts.get(c,0):>5d}")

    with open(os.path.join(args.out_dir, "train_class_counts.json"), "w") as f:
        json.dump({class_names[k]: int(v) for k, v in train_counts.items()}, f, indent=2)

    # --- Transforms ---
    use_album = HAS_ALBUM
    train_tf = get_album_train_transform(args.img_size) if use_album else get_tv_train_transform(args.img_size)
    val_tf = get_album_val_transform(args.img_size) if use_album else get_tv_val_transform(args.img_size)

    # ==================================================================
    # STAGE 1: Binary Classifier (good vs defect)
    # ==================================================================
    print(f"\n{'='*70}")
    print("STAGE 1: Binary Classifier (good vs defect)")
    print(f"{'='*70}")

    bin_train_ds = BinaryWrapperDataset(
        WaferDefectDataset(args.data_root, transform=train_tf, use_albumentations=use_album, merge_map=merge_map),
        good_class_idx)
    bin_val_ds = BinaryWrapperDataset(
        WaferDefectDataset(args.data_root, transform=val_tf, use_albumentations=use_album, merge_map=merge_map),
        good_class_idx)

    bin_train_subset = Subset(bin_train_ds, train_idx)
    bin_val_subset = Subset(bin_val_ds, val_idx)

    # Balanced sampler for binary task
    bin_targets = [0 if all_targets[i] == good_class_idx else 1 for i in train_idx]
    bin_counts = Counter(bin_targets)
    print(f"  Binary: good={bin_counts[0]}, defect={bin_counts[1]}")
    bin_cw = {c: 1.0 / max(bin_counts.get(c, 1), 1) for c in range(2)}
    bin_sw = [bin_cw[t] for t in bin_targets]
    bin_sampler = WeightedRandomSampler(bin_sw, len(bin_sw), replacement=True)

    bin_train_loader = DataLoader(bin_train_subset, batch_size=args.batch_size, sampler=bin_sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                  persistent_workers=args.num_workers > 0)
    bin_val_loader = DataLoader(bin_val_subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=args.num_workers > 0)

    bin_model = build_model(args.model, 1, pretrained=args.pretrained,
                            drop_rate=args.drop_rate).to(device)
    bin_criterion = nn.BCEWithLogitsLoss()

    print(f"  Loss: BCEWithLogitsLoss (no weighting)")

    best_bin_model, bin_history, best_bin_acc = _train_stage(
        bin_model, bin_train_loader, bin_val_loader, bin_criterion, device, args, use_amp,
        "binary", args.out_dir)

    plot_training_curves(bin_history, args.out_dir, prefix="Stage1 Binary")

    # ==================================================================
    # STAGE 2: Defect Subtype Classifier (defect images only)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"STAGE 2: Defect Subtype Classifier ({num_defect_classes} classes)")
    print(f"{'='*70}")

    defect_train_idx = [i for i in train_idx if all_targets[i] != good_class_idx]
    defect_val_idx = [i for i in val_idx if all_targets[i] != good_class_idx]

    defect_train_targets = [defect_label_map[all_targets[i]] for i in defect_train_idx]
    defect_train_counts = Counter(defect_train_targets)

    print(f"  Defect split: train={len(defect_train_idx)}, val={len(defect_val_idx)}")
    for c in range(num_defect_classes):
        cnt = defect_train_counts.get(c, 0)
        warn = " << RARE" if cnt < 15 else ""
        print(f"    {defect_class_names[c]:>10s}: train={cnt:>5d}{warn}")

    def_train_ds = DefectRemapDataset(
        WaferDefectDataset(args.data_root, transform=train_tf, use_albumentations=use_album, merge_map=merge_map),
        defect_label_map)
    def_val_ds = DefectRemapDataset(
        WaferDefectDataset(args.data_root, transform=val_tf, use_albumentations=use_album, merge_map=merge_map),
        defect_label_map)

    def_train_subset = Subset(def_train_ds, defect_train_idx)
    def_val_subset = Subset(def_val_ds, defect_val_idx)

    # Weighted sampler for defect classes (critical for rare classes)
    def_cw = {c: 1.0 / max(defect_train_counts.get(c, 1), 1) for c in range(num_defect_classes)}
    def_sw = [def_cw[t] for t in defect_train_targets]
    def_sampler = WeightedRandomSampler(def_sw, len(def_sw), replacement=True)

    def_train_loader = DataLoader(def_train_subset, batch_size=args.batch_size, sampler=def_sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                  persistent_workers=args.num_workers > 0)
    def_val_loader = DataLoader(def_val_subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=args.num_workers > 0)

    def_model = build_model(args.model, num_defect_classes, pretrained=args.pretrained,
                            drop_rate=args.drop_rate).to(device)
    def_criterion = FocalLoss(alpha=None, gamma=args.focal_gamma, label_smoothing=0.0)

    print(f"  Focal Loss: gamma={args.focal_gamma}, no alpha, no smoothing")

    best_def_model, def_history, best_def_acc = _train_stage(
        def_model, def_train_loader, def_val_loader, def_criterion, device, args, use_amp,
        "defect", args.out_dir)

    plot_training_curves(def_history, args.out_dir, prefix="Stage2 Defect")

    # ==================================================================
    # COMBINED TWO-STAGE EVALUATION
    # ==================================================================
    print(f"\n{'='*70}")
    print("TWO-STAGE TEST EVALUATION")
    print(f"{'='*70}")

    test_ds = WaferDefectDataset(args.data_root, transform=val_tf, use_albumentations=use_album, merge_map=merge_map)
    test_subset = Subset(test_ds, test_idx)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    test_acc, test_preds, test_true = evaluate_two_stage(
        best_bin_model, best_def_model, test_loader, device, use_amp,
        good_class_idx, defect_label_map_inv)
    print(f"Combined Two-Stage Test Accuracy: {test_acc:.4f}")

    # Binary stage accuracy on test set
    bin_test_ds = BinaryWrapperDataset(test_ds, good_class_idx)
    bin_test_loader = DataLoader(Subset(bin_test_ds, test_idx), batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    bin_test_acc, _, _ = evaluate(best_bin_model, bin_test_loader, device, use_amp)
    print(f"Binary Stage Test Accuracy:       {bin_test_acc:.4f}")

    # Defect stage accuracy on defect-only test samples
    defect_test_idx = [i for i in test_idx if all_targets[i] != good_class_idx]
    if defect_test_idx:
        def_test_ds = DefectRemapDataset(test_ds, defect_label_map)
        def_test_loader = DataLoader(Subset(def_test_ds, defect_test_idx),
                                     batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)
        def_test_acc, _, _ = evaluate(best_def_model, def_test_loader, device, use_amp)
        print(f"Defect Stage Test Accuracy:       {def_test_acc:.4f}")
    else:
        def_test_acc = 0.0

    # --- Classification Report ---
    cm = confusion_matrix(test_true, test_preds, labels=list(range(num_classes)))
    report = classification_report(test_true, test_preds, target_names=class_names, digits=4)
    report_dict = classification_report(test_true, test_preds, target_names=class_names,
                                        digits=4, output_dict=True)
    print(f"\n{report}")

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Two-Stage Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Binary Stage Accuracy:   {bin_test_acc:.4f}\n")
        f.write(f"Defect Stage Accuracy:   {def_test_acc:.4f}\n\n")
        f.write(report)
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")

    # --- Plots ---
    plot_confusion_matrix(cm, class_names, args.out_dir)
    plot_accuracy_vs_occurrence(cm, train_counts, class_names, num_classes, args.out_dir)
    plot_per_class_metrics(report_dict, class_names, args.out_dir)

    # --- Inference Benchmark (two-stage pipeline) ---
    print(f"\n{'='*70}")
    print("INFERENCE BENCHMARK")
    print(f"{'='*70}")
    bench = benchmark_inference(best_bin_model, device, args.img_size, use_amp=use_amp)
    with open(os.path.join(args.out_dir, "inference_benchmark.json"), "w") as f:
        json.dump(bench, f, indent=2)

    # --- Summary ---
    summary = {
        "pipeline": "two_stage",
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "stage1_binary": {
            "epochs_trained": len(bin_history["train_loss"]),
            "best_val_acc": round(best_bin_acc, 4),
            "test_acc": round(bin_test_acc, 4),
        },
        "stage2_defect": {
            "num_classes": num_defect_classes,
            "class_names": defect_class_names,
            "epochs_trained": len(def_history["train_loss"]),
            "best_val_acc": round(best_def_acc, 4),
            "test_acc": round(def_test_acc, 4),
        },
        "combined_test_acc": round(test_acc, 4),
        "num_classes": num_classes,
        "class_names": class_names,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_class_counts": {class_names[k]: v for k, v in train_counts.items()},
        "inference_benchmark": bench,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "amp": use_amp,
        "ema": args.use_ema,
        "focal_gamma": args.focal_gamma,
        "mixup_alpha": 0.0,
        "label_smoothing": 0.0,
    }
    with open(os.path.join(args.out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE — All outputs in {args.out_dir}/")
    print(f"Combined Two-Stage Test Accuracy: {test_acc:.4f} | Target: ~85%")
    print(f"{'='*70}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Two-Stage Defect Classification — RTX 5090 Optimized")
    # Data
    p.add_argument("--data_root", default="Data")
    p.add_argument("--out_dir", default="reports")
    # Model — ConvNeXt-Base is the sweet spot for 30GB VRAM
    p.add_argument("--model", default="convnext_small.fb_in22k_ft_in1k",
                   help="timm model (convnext_small, convnext_base, efficientnetv2_m, swin_base_patch4_window12_384, etc.)")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", action="store_false", dest="pretrained")
    p.add_argument("--drop_rate", type=float, default=0.1)
    # Training
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=12)
    # EMA & AMP
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_ema", action="store_false", dest="use_ema")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    # Splits
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    # Misc
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # Class merging: merge rare classes into a single class
    # Format: "old1:new old2:new" e.g. "defect3:rare defect4:rare"
    p.add_argument("--merge_classes", nargs="*", default=["defect3:rare", "defect4:rare"],
                   help='Merge classes, e.g. "defect3:rare defect4:rare"')

    args = p.parse_args()
    main(args)
