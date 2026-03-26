"""
Multi-Task ResNet50 Pipeline: Joint Binary + Defect Subtype Classifier

Three key improvements over baseline:
  1. Edge channels: (grayscale, Sobel X, Sobel Y) — principled directional gradients
  2. Resolution: 256x256 — captures finer defect detail without overfitting
  3. Multi-task: shared backbone with binary + subtype heads, trained jointly

Joint training:
  - Binary head: BCEWithLogitsLoss (pos_weight=1.8) on ALL samples
  - Subtype head: CrossEntropyLoss (class-weighted) on DEFECT samples only
  - Combined loss = binary_loss + 0.5 * subtype_loss

Training schedule:
  - Freeze backbone 5 epochs (lr=1e-3), unfreeze 20 epochs (lr=1e-4)
  - 25 total epochs, 256x256, weight_decay=1e-4
  - Hard negative fine-tuning: 5 extra epochs at lr=5e-5
"""

import os, time, json
import numpy as np
from PIL import Image, ImageFilter
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.amp import autocast, GradScaler

import torchvision.transforms as T
import torchvision.models as models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Edge channel preprocessing
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Sobel kernels (offset=128 to center in uint8 range)
_SOBEL_X = ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1, offset=128)
_SOBEL_Y = ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1, offset=128)


def grayscale_to_edge_channels(img_gray):
    """Convert grayscale PIL image to 3-channel: (grayscale, Sobel X, Sobel Y).

    Ch R: original grayscale intensity
    Ch G: Sobel X (horizontal gradient — detects vertical edges)
    Ch B: Sobel Y (vertical gradient — detects horizontal edges)
    """
    sobel_x = img_gray.filter(_SOBEL_X)
    sobel_y = img_gray.filter(_SOBEL_Y)
    return Image.merge("RGB", [img_gray, sobel_x, sobel_y])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WaferMultiTaskDataset(Dataset):
    """Multi-task dataset returning (image, binary_label, subtype_label).

    binary_label:  0 = good, 1 = defect
    subtype_label: 0..N-1 for defect classes, -1 for good (ignored in subtype loss)
    """
    VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (path, binary_label, subtype_label)

        class_dirs = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        defect_dirs = [d for d in class_dirs if d != "good"]
        self.defect_classes = defect_dirs
        self.defect_to_idx = {c: i for i, c in enumerate(defect_dirs)}

        for cls_name in class_dirs:
            if cls_name == "good":
                binary_label, subtype_label = 0, -1
            else:
                binary_label = 1
                subtype_label = self.defect_to_idx[cls_name]

            cls_path = self.root / cls_name
            for f in sorted(cls_path.iterdir()):
                if f.suffix.lower() in self.VALID_EXT:
                    self.samples.append((str(f), binary_label, subtype_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bin_y, sub_y = self.samples[idx]
        img = Image.open(path).convert("L")
        img_rgb = grayscale_to_edge_channels(img)

        if self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = T.ToTensor()(img_rgb)

        return img_tensor, bin_y, sub_y


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transform(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(7),
        T.ToTensor(),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model — shared backbone, two heads
# ---------------------------------------------------------------------------
class MultiTaskResNet50(nn.Module):
    """ResNet50 with shared backbone and two classification heads."""

    def __init__(self, num_defect_classes):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.binary_head = nn.Linear(2048, 1)
        self.subtype_head = nn.Linear(2048, num_defect_classes)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        bin_logit = self.binary_head(feat).squeeze(1)
        sub_logits = self.subtype_head(feat)
        return bin_logit, sub_logits

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Training — joint multi-task
# ---------------------------------------------------------------------------
def train_one_epoch_multitask(model, loader, criterion_bin, criterion_sub,
                              optimizer, device, scaler, use_amp, subtype_weight=0.5):
    model.train()
    running_loss = 0.0
    bin_correct = bin_total = 0
    sub_correct = sub_total = 0

    for x, y_bin, y_sub in loader:
        x = x.to(device, non_blocking=True)
        y_bin = y_bin.to(device, non_blocking=True).float()
        y_sub = y_sub.to(device, non_blocking=True).long()

        with autocast("cuda", enabled=use_amp):
            bin_logits, sub_logits = model(x)
            loss_bin = criterion_bin(bin_logits, y_bin)

            defect_mask = y_sub >= 0
            if defect_mask.any():
                loss_sub = criterion_sub(sub_logits[defect_mask], y_sub[defect_mask])
            else:
                loss_sub = torch.tensor(0.0, device=device)

            loss = loss_bin + subtype_weight * loss_sub

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

        bin_preds = (bin_logits > 0).long()
        bin_correct += (bin_preds == y_bin.long()).sum().item()
        bin_total += y_bin.size(0)

        if defect_mask.any():
            sub_preds = sub_logits[defect_mask].argmax(dim=1)
            sub_correct += (sub_preds == y_sub[defect_mask]).sum().item()
            sub_total += defect_mask.sum().item()

        running_loss += loss.item() * y_bin.size(0)

    avg_loss = running_loss / bin_total
    bin_acc = bin_correct / bin_total if bin_total > 0 else 0.0
    sub_acc = sub_correct / sub_total if sub_total > 0 else 0.0
    return avg_loss, bin_acc, sub_acc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_binary(model, loader, device, use_amp=False, threshold=0.0, tta=True):
    """Evaluate binary head with optional TTA (flip H + flip V)."""
    model.eval()
    all_logits, all_true = [], []

    for x, y_bin, _y_sub in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            bl1, _ = model(x)
            if tta:
                bl2, _ = model(torch.flip(x, dims=[3]))
                bl3, _ = model(torch.flip(x, dims=[2]))
                bin_logits = (bl1 + bl2 + bl3) / 3
            else:
                bin_logits = bl1

        all_logits.extend(bin_logits.cpu().tolist())
        all_true.extend(y_bin.tolist())

    all_logits = np.array(all_logits)
    all_true = np.array(all_true)
    preds = (all_logits > threshold).astype(int)
    acc = float((preds == all_true).mean())
    return acc, all_logits, all_true


@torch.no_grad()
def evaluate_subtype(model, loader, device, use_amp=False):
    """Evaluate subtype head on defect samples only."""
    model.eval()
    all_preds, all_true = [], []

    for x, _y_bin, y_sub in loader:
        defect_mask = y_sub >= 0
        if not defect_mask.any():
            continue

        x_def = x[defect_mask].to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            _, sub_logits = model(x_def)

        preds = sub_logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(y_sub[defect_mask].tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    acc = float((all_preds == all_true).mean()) if len(all_true) > 0 else 0.0
    return acc, all_preds, all_true


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(history["train_loss"], linewidth=2, color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Joint Training Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["bin_acc"], label="Train (bin)", linewidth=2, color="#1f77b4")
    axes[1].plot(history["val_bin_acc"], label="Val (bin)", linewidth=2, color="#2ca02c")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Binary Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)

    axes[2].plot(history["sub_acc"], label="Train (sub)", linewidth=2, color="#ff7f0e")
    axes[2].plot(history["val_sub_acc"], label="Val (sub)", linewidth=2, color="#9467bd")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Subtype Accuracy"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "multitask_curves.png"), dpi=200)
    plt.close()
    print(f"  Saved training curves to {out_dir}/multitask_curves.png")


def plot_confusion(cm, out_dir, title="Binary Confusion Matrix", fname="binary_confusion.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["good", "defect"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close()


def plot_threshold_sweep(thresholds, f1_scores, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1_scores, linewidth=2, color="#1f77b4")
    best_idx = np.argmax(f1_scores)
    ax.axvline(x=thresholds[best_idx], color="red", linestyle="--", alpha=0.7,
               label=f"Best threshold={thresholds[best_idx]:.3f} (F1={f1_scores[best_idx]:.4f})")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score (Defect)")
    ax.set_title("Threshold Sweep (F1 Defect) on Validation Set")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "threshold_sweep.png"), dpi=200)
    plt.close()


def plot_subtype_confusion(cm, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Defect Subtype Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subtype_confusion.png"), dpi=200)
    plt.close()


def plot_per_class_metrics(report_dict, class_names, out_dir):
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
    ax.set_ylabel("Score"); ax.set_title("Per-Class Defect Subtype Metrics")
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "subtype_per_class_metrics.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Config ---
    DATA_ROOT      = "Data"
    OUT_DIR        = "reports/resnet50_multitask"
    IMG_SIZE       = 256           # improvement #2: up from 224
    BATCH_SIZE     = 128
    FREEZE_EPOCHS  = 5
    TOTAL_EPOCHS   = 25
    LR_FROZEN      = 1e-3
    LR_UNFROZEN    = 1e-4
    WEIGHT_DECAY   = 1e-4
    NUM_WORKERS    = 8
    SEED           = 42
    VAL_SPLIT      = 0.15
    TEST_SPLIT     = 0.15
    SUBTYPE_WEIGHT = 0.5          # improvement #3: joint loss weighting

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True

    use_amp = device.type == "cuda"
    print(f"AMP: {use_amp}")
    print(f"\nImprovements enabled:")
    print(f"  1. Edge channels: (grayscale, Sobel X, Sobel Y)")
    print(f"  2. Resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  3. Multi-task: shared backbone, joint binary+subtype training (weight={SUBTYPE_WEIGHT})")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    full_ds = WaferMultiTaskDataset(DATA_ROOT, transform=None)
    all_binary  = [s[1] for s in full_ds.samples]
    all_subtype = [s[2] for s in full_ds.samples]
    num_defect_classes = len(full_ds.defect_classes)
    defect_class_names = full_ds.defect_classes

    print(f"\nTotal images: {len(full_ds)}")
    bin_counts_all = Counter(all_binary)
    print(f"  good:   {bin_counts_all[0]} ({bin_counts_all[0]/len(full_ds):.1%})")
    print(f"  defect: {bin_counts_all[1]} ({bin_counts_all[1]/len(full_ds):.1%})")
    print(f"  Majority baseline: {max(bin_counts_all.values())/len(full_ds):.1%}")

    sub_counts_all = Counter(s for s in all_subtype if s >= 0)
    print(f"\nDefect classes ({num_defect_classes}): {defect_class_names}")
    for i, name in enumerate(defect_class_names):
        warn = " << RARE" if sub_counts_all.get(i, 0) < 20 else ""
        print(f"  {name:>10s}: {sub_counts_all.get(i, 0):>5d}{warn}")

    # ------------------------------------------------------------------
    # Stratified split (on binary labels)
    # ------------------------------------------------------------------
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
    trainval_idx, test_idx = next(sss_test.split(range(len(full_ds)), all_binary))

    trainval_binary = [all_binary[i] for i in trainval_idx]
    relative_val = VAL_SPLIT / (1 - TEST_SPLIT)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=SEED)
    train_idx_rel, val_idx_rel = next(sss_val.split(range(len(trainval_idx)), trainval_binary))
    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx   = [trainval_idx[i] for i in val_idx_rel]

    train_binary  = [all_binary[i] for i in train_idx]
    val_binary    = [all_binary[i] for i in val_idx]
    test_binary   = [all_binary[i] for i in test_idx]

    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    for split_name, tgts in [("train", train_binary), ("val", val_binary), ("test", test_binary)]:
        c = Counter(tgts)
        print(f"  {split_name:>5s}: good={c[0]}, defect={c[1]}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_ds = WaferMultiTaskDataset(DATA_ROOT, transform=get_train_transform(IMG_SIZE))
    val_ds   = WaferMultiTaskDataset(DATA_ROOT, transform=get_val_transform(IMG_SIZE))

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds, val_idx)
    test_subset  = Subset(val_ds, test_idx)

    bin_counts = Counter(train_binary)
    sample_weights = [1.0 / bin_counts[t] for t in train_binary]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    print(f"\nWeightedRandomSampler: good={1.0/bin_counts[0]:.6f}, defect={1.0/bin_counts[1]:.6f}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MultiTaskResNet50(num_defect_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nMultiTaskResNet50: {n_params:,} params")
    print(f"  binary_head:  Linear(2048, 1)")
    print(f"  subtype_head: Linear(2048, {num_defect_classes})")

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    pos_weight = torch.tensor([1.8]).to(device)
    criterion_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"\nBinary loss:  BCEWithLogitsLoss (pos_weight=1.8)")

    # Subtype class weights from training set
    train_subtype = [all_subtype[i] for i in train_idx if all_subtype[i] >= 0]
    sub_train_counts = Counter(train_subtype)
    class_freqs = torch.tensor(
        [max(sub_train_counts.get(c, 1), 1) for c in range(num_defect_classes)],
        dtype=torch.float32)
    class_weights = 1.0 / class_freqs
    class_weights = class_weights / class_weights.sum() * num_defect_classes
    class_weights = class_weights.to(device)
    criterion_sub = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Subtype loss: CrossEntropyLoss (inverse-freq weighted)")
    for i, name in enumerate(defect_class_names):
        print(f"  {name:>10s}: count={sub_train_counts.get(i,0):>4d}, weight={class_weights[i]:.4f}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    scaler = GradScaler("cuda", enabled=use_amp)
    history = {
        "train_loss": [], "bin_acc": [], "sub_acc": [],
        "val_bin_acc": [], "val_sub_acc": [], "lr": [], "phase": [],
    }
    best_val_bin_acc = 0.0
    best_model_path = os.path.join(OUT_DIR, "best_multitask.pt")

    # === Phase 1: Frozen backbone ===
    print(f"\n{'='*70}")
    print(f"PHASE 1: Frozen backbone, train heads only (epochs 1-{FREEZE_EPOCHS})")
    print(f"  lr={LR_FROZEN}, weight_decay={WEIGHT_DECAY}, subtype_weight={SUBTYPE_WEIGHT}")
    print(f"{'='*70}")

    model.freeze_backbone()
    head_params = list(model.binary_head.parameters()) + list(model.subtype_head.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (frozen phase): {trainable:,}")

    optimizer = optim.AdamW(head_params, lr=LR_FROZEN, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, FREEZE_EPOCHS + 1):
        t0 = time.time()
        train_loss, bin_acc, sub_acc = train_one_epoch_multitask(
            model, train_loader, criterion_bin, criterion_sub,
            optimizer, device, scaler, use_amp, SUBTYPE_WEIGHT)
        val_bin_acc, _, _ = evaluate_binary(model, val_loader, device, use_amp)
        val_sub_acc, _, _ = evaluate_subtype(model, val_loader, device, use_amp)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["bin_acc"].append(bin_acc)
        history["sub_acc"].append(sub_acc)
        history["val_bin_acc"].append(val_bin_acc)
        history["val_sub_acc"].append(val_sub_acc)
        history["lr"].append(LR_FROZEN)
        history["phase"].append("frozen")

        improved = ""
        if val_bin_acc > best_val_bin_acc:
            best_val_bin_acc = val_bin_acc
            torch.save(model.state_dict(), best_model_path)
            improved = " *BEST*"

        print(f"  Ep {epoch:02d}/{TOTAL_EPOCHS} [FROZEN] | Loss {train_loss:.4f} | "
              f"Bin {bin_acc:.4f}/{val_bin_acc:.4f} | "
              f"Sub {sub_acc:.4f}/{val_sub_acc:.4f} | "
              f"LR {LR_FROZEN:.1e} | {elapsed:.1f}s{improved}")

    # === Phase 2: Unfreeze all ===
    print(f"\n{'='*70}")
    print(f"PHASE 2: Unfrozen, full fine-tuning (epochs {FREEZE_EPOCHS+1}-{TOTAL_EPOCHS})")
    print(f"  lr={LR_UNFROZEN}, weight_decay={WEIGHT_DECAY}")
    print(f"{'='*70}")

    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (unfrozen phase): {trainable:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR_UNFROZEN, weight_decay=WEIGHT_DECAY)
    unfrozen_epochs = TOTAL_EPOCHS - FREEZE_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=unfrozen_epochs, eta_min=1e-6)

    for epoch in range(FREEZE_EPOCHS + 1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        train_loss, bin_acc, sub_acc = train_one_epoch_multitask(
            model, train_loader, criterion_bin, criterion_sub,
            optimizer, device, scaler, use_amp, SUBTYPE_WEIGHT)
        val_bin_acc, _, _ = evaluate_binary(model, val_loader, device, use_amp)
        val_sub_acc, _, _ = evaluate_subtype(model, val_loader, device, use_amp)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["bin_acc"].append(bin_acc)
        history["sub_acc"].append(sub_acc)
        history["val_bin_acc"].append(val_bin_acc)
        history["val_sub_acc"].append(val_sub_acc)
        history["lr"].append(current_lr)
        history["phase"].append("unfrozen")

        improved = ""
        if val_bin_acc > best_val_bin_acc:
            best_val_bin_acc = val_bin_acc
            torch.save(model.state_dict(), best_model_path)
            improved = " *BEST*"

        print(f"  Ep {epoch:02d}/{TOTAL_EPOCHS} [UNFROZEN] | Loss {train_loss:.4f} | "
              f"Bin {bin_acc:.4f}/{val_bin_acc:.4f} | "
              f"Sub {sub_acc:.4f}/{val_sub_acc:.4f} | "
              f"LR {current_lr:.1e} | {elapsed:.1f}s{improved}")

    # --- Load best model ---
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"\nLoaded best model (val bin acc: {best_val_bin_acc:.4f})")
    plot_training_curves(history, OUT_DIR)

    # ==================================================================
    # BINARY TEST EVALUATION
    # ==================================================================
    print(f"\n{'='*70}")
    print("BINARY TEST EVALUATION")
    print(f"{'='*70}")

    test_bin_acc, test_logits, test_true = evaluate_binary(model, test_loader, device, use_amp)
    test_preds = (test_logits > 0).astype(int)
    print(f"  Test Accuracy (threshold=0.0): {test_bin_acc:.4f}")

    cm = confusion_matrix(test_true, test_preds, labels=[0, 1])
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              good  defect")
    print(f"  True good   {cm[0,0]:>5d}  {cm[0,1]:>5d}")
    print(f"  True defect {cm[1,0]:>5d}  {cm[1,1]:>5d}")

    report = classification_report(test_true, test_preds, target_names=["good", "defect"], digits=4)
    print(f"\n{report}")
    plot_confusion(cm, OUT_DIR)

    # ==================================================================
    # THRESHOLD SWEEP
    # ==================================================================
    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP (on validation set)")
    print(f"{'='*70}")

    _, val_logits, val_true = evaluate_binary(model, val_loader, device, use_amp)

    thresholds = np.linspace(-3.0, 3.0, 100)
    f1_scores_arr = []
    for t in thresholds:
        preds = (val_logits > t).astype(int)
        f1_scores_arr.append(f1_score(val_true, preds, pos_label=1))

    f1_scores_arr = np.array(f1_scores_arr)
    best_t_idx = np.argmax(f1_scores_arr)
    best_threshold = thresholds[best_t_idx]
    best_thresh_f1 = f1_scores_arr[best_t_idx]

    val_default_f1 = f1_score(val_true, (val_logits > 0).astype(int), pos_label=1)
    print(f"  Default threshold (0.0): Val F1 (defect) = {val_default_f1:.4f}")
    print(f"  Best threshold (F1 defect): {best_threshold:.3f}")
    print(f"  Best F1 (defect): {best_thresh_f1:.4f}")
    print(f"  Improvement: +{(best_thresh_f1 - val_default_f1)*100:.2f}%")

    test_preds_tuned = (test_logits > best_threshold).astype(int)
    test_acc_tuned = float((test_preds_tuned == test_true).mean())
    cm_tuned = confusion_matrix(test_true, test_preds_tuned, labels=[0, 1])

    print(f"\n  Test Accuracy (tuned threshold={best_threshold:.3f}): {test_acc_tuned:.4f}")
    print(f"  Confusion Matrix (tuned):")
    print(f"              Predicted")
    print(f"              good  defect")
    print(f"  True good   {cm_tuned[0,0]:>5d}  {cm_tuned[0,1]:>5d}")
    print(f"  True defect {cm_tuned[1,0]:>5d}  {cm_tuned[1,1]:>5d}")

    plot_threshold_sweep(thresholds, f1_scores_arr, OUT_DIR)

    # ==================================================================
    # HARD NEGATIVE MINING
    # ==================================================================
    print(f"\n{'='*70}")
    print("HARD NEGATIVE MINING")
    print(f"{'='*70}")

    false_negatives = [i for i, (l, t) in enumerate(zip(test_logits, test_true)) if t == 1 and l <= 0]
    false_positives = [i for i, (l, t) in enumerate(zip(test_logits, test_true)) if t == 0 and l > 0]
    print(f"  False negatives (defect predicted as good): {len(false_negatives)}")
    print(f"  False positives (good predicted as defect): {len(false_positives)}")
    print(f"  Total test samples: {len(test_true)}")

    # ==================================================================
    # HARD NEGATIVE FINE-TUNING
    # ==================================================================
    print(f"\n{'='*70}")
    print("HARD NEGATIVE FINE-TUNING (5 epochs, lr=5e-5, 3x oversample)")
    print(f"{'='*70}")

    hn_eval_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
    _, train_logits_hn, train_true_hn = evaluate_binary(
        model, hn_eval_loader, device, use_amp, tta=False)

    hard_train_positions = []
    for i, (logit, true_label) in enumerate(zip(train_logits_hn, train_true_hn)):
        if true_label == 1 and logit <= 0.5:
            hard_train_positions.append(i)
        elif true_label == 0 and logit > -0.5:
            hard_train_positions.append(i)

    hard_dataset_indices = [train_idx[p] for p in hard_train_positions]
    print(f"  Hard samples from training set: {len(hard_dataset_indices)}")

    if len(hard_dataset_indices) > 50:
        hn_indices = train_idx + hard_dataset_indices * 3
        hn_subset = Subset(train_ds, hn_indices)

        hn_binary = [all_binary[i] for i in hn_indices]
        hn_bin_counts = Counter(hn_binary)
        hn_sw = [1.0 / hn_bin_counts[t] for t in hn_binary]
        hn_sampler = WeightedRandomSampler(hn_sw, len(hn_sw), replacement=True)

        hn_loader = DataLoader(hn_subset, batch_size=BATCH_SIZE, sampler=hn_sampler,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                               persistent_workers=NUM_WORKERS > 0)

        HN_LR = 5e-5
        HN_EPOCHS = 5
        optimizer_hn = optim.AdamW(model.parameters(), lr=HN_LR, weight_decay=WEIGHT_DECAY)
        scheduler_hn = optim.lr_scheduler.CosineAnnealingLR(optimizer_hn, T_max=HN_EPOCHS, eta_min=1e-6)

        print(f"  Fine-tuning dataset: {len(hn_indices)} samples "
              f"(original {len(train_idx)} + {len(hard_dataset_indices)*3} hard oversampled)")

        for epoch in range(1, HN_EPOCHS + 1):
            t0 = time.time()
            hn_loss, hn_bin_acc, hn_sub_acc = train_one_epoch_multitask(
                model, hn_loader, criterion_bin, criterion_sub,
                optimizer_hn, device, scaler, use_amp, SUBTYPE_WEIGHT)
            hn_val_bin_acc, _, _ = evaluate_binary(model, val_loader, device, use_amp)
            hn_val_sub_acc, _, _ = evaluate_subtype(model, val_loader, device, use_amp)
            hn_lr = scheduler_hn.get_last_lr()[0]
            scheduler_hn.step()
            elapsed = time.time() - t0

            improved = ""
            if hn_val_bin_acc > best_val_bin_acc:
                best_val_bin_acc = hn_val_bin_acc
                torch.save(model.state_dict(), best_model_path)
                improved = " *BEST*"

            print(f"  HN Ep {epoch:02d}/{HN_EPOCHS} | Loss {hn_loss:.4f} | "
                  f"Bin {hn_bin_acc:.4f}/{hn_val_bin_acc:.4f} | "
                  f"Sub {hn_sub_acc:.4f}/{hn_val_sub_acc:.4f} | "
                  f"LR {hn_lr:.1e} | {elapsed:.1f}s{improved}")

        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"\n  Loaded best model after HN fine-tuning (val bin acc: {best_val_bin_acc:.4f})")

        # Re-evaluate binary after HN
        print(f"\n  --- Post-HN Binary Test Evaluation ---")
        test_bin_acc_hn, test_logits_hn2, test_true_hn2 = evaluate_binary(
            model, test_loader, device, use_amp)
        test_preds_hn = (test_logits_hn2 > 0).astype(int)
        cm_hn = confusion_matrix(test_true_hn2, test_preds_hn, labels=[0, 1])
        print(f"  Test Accuracy (post-HN): {test_bin_acc_hn:.4f}")
        print(f"  Confusion Matrix (post-HN):")
        print(f"              Predicted")
        print(f"              good  defect")
        print(f"  True good   {cm_hn[0,0]:>5d}  {cm_hn[0,1]:>5d}")
        print(f"  True defect {cm_hn[1,0]:>5d}  {cm_hn[1,1]:>5d}")
        defect_recall_hn = cm_hn[1,1] / (cm_hn[1,0] + cm_hn[1,1]) if (cm_hn[1,0] + cm_hn[1,1]) > 0 else 0
        print(f"  Defect recall: {defect_recall_hn:.1%}")

        # Update for final summary
        test_bin_acc = test_bin_acc_hn
        test_logits = test_logits_hn2
        test_true = test_true_hn2
    else:
        print(f"  Too few hard samples ({len(hard_dataset_indices)}), skipping HN fine-tuning.")

    # ==================================================================
    # SUBTYPE TEST EVALUATION
    # ==================================================================
    print(f"\n{'='*70}")
    print("DEFECT SUBTYPE TEST EVALUATION")
    print(f"{'='*70}")

    sub_test_acc, sub_test_preds, sub_test_true = evaluate_subtype(
        model, test_loader, device, use_amp)
    print(f"  Subtype Test Accuracy: {sub_test_acc:.4f}")
    print(f"  Random chance ({num_defect_classes} classes): {1/num_defect_classes:.1%}")

    sub_cm = confusion_matrix(sub_test_true, sub_test_preds, labels=list(range(num_defect_classes)))
    sub_report = classification_report(
        sub_test_true, sub_test_preds, target_names=defect_class_names, digits=4)
    sub_report_dict = classification_report(
        sub_test_true, sub_test_preds, target_names=defect_class_names, digits=4, output_dict=True)
    print(f"\n{sub_report}")

    plot_subtype_confusion(sub_cm, defect_class_names, OUT_DIR)
    plot_per_class_metrics(sub_report_dict, defect_class_names, OUT_DIR)

    # ==================================================================
    # SAVE SUMMARY
    # ==================================================================
    summary = {
        "improvements": [
            "Edge channels: (grayscale, Sobel X, Sobel Y)",
            f"Resolution: {IMG_SIZE}x{IMG_SIZE}",
            f"Multi-task: shared backbone, subtype_weight={SUBTYPE_WEIGHT}",
        ],
        "binary": {
            "model": "MultiTaskResNet50 (shared backbone)",
            "img_size": IMG_SIZE,
            "total_epochs": TOTAL_EPOCHS,
            "freeze_epochs": FREEZE_EPOCHS,
            "lr_frozen": LR_FROZEN,
            "lr_unfrozen": LR_UNFROZEN,
            "weight_decay": WEIGHT_DECAY,
            "subtype_weight": SUBTYPE_WEIGHT,
            "best_val_acc": round(best_val_bin_acc, 4),
            "test_acc_default_threshold": round(float(test_bin_acc), 4),
            "test_acc_tuned_threshold": round(float(test_acc_tuned), 4),
            "best_threshold": round(float(best_threshold), 3),
            "majority_baseline": round(max(bin_counts_all.values()) / len(full_ds), 4),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
        },
        "subtype": {
            "num_classes": num_defect_classes,
            "class_names": defect_class_names,
            "loss": "CrossEntropyLoss (weighted)",
            "class_weights": {defect_class_names[i]: round(class_weights[i].item(), 4)
                              for i in range(num_defect_classes)},
            "test_acc": round(sub_test_acc, 4),
            "random_chance": round(1 / num_defect_classes, 4),
            "train_defect_counts": {defect_class_names[i]: sub_train_counts.get(i, 0)
                                    for i in range(num_defect_classes)},
        },
        "history": {k: [round(v, 4) if isinstance(v, float) else v for v in vals]
                    for k, vals in history.items()},
    }
    with open(os.path.join(OUT_DIR, "multitask_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  --- Binary (good vs defect) ---")
    print(f"  Majority baseline:     {max(bin_counts_all.values())/len(full_ds):.1%}")
    print(f"  Best val accuracy:     {best_val_bin_acc:.4f}")
    print(f"  Test acc (default):    {test_bin_acc:.4f}")
    print(f"  Test acc (tuned):      {test_acc_tuned:.4f}")
    print(f"  Best threshold:        {best_threshold:.3f}")
    print(f"")
    print(f"  --- Subtype ({num_defect_classes} defect classes) ---")
    print(f"  Random chance:         {1/num_defect_classes:.1%}")
    print(f"  Test accuracy:         {sub_test_acc:.4f}")
    print(f"")
    print(f"  All outputs saved to:  {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
