"""
Two-Stage ResNet50 Pipeline: Binary + Defect Subtype Classifier
Clean, minimal script — no EMA, no mixup, no focal, no heavy augmentation.

Stage 1: Binary classifier (good vs defect)
  - BCEWithLogitsLoss (no weighting)
  - Threshold sweep after training

Stage 2: Defect subtype classifier (8 defect classes)
  - CrossEntropyLoss with inverse-frequency class weights
  - Only trains on defect images

Both stages share:
  - ResNet50 (ImageNet pretrained)
  - Freeze backbone 5 epochs (lr=1e-3), unfreeze 15 epochs (lr=1e-4)
  - 20 epochs total, 224x224, weight_decay=1e-4
"""

import os, time, json
import numpy as np
from PIL import Image
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

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class WaferBinaryDataset(Dataset):
    """Binary dataset: 0=good, 1=defect."""
    VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (path, binary_label)

        class_dirs = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        for cls_name in class_dirs:
            label = 0 if cls_name == "good" else 1
            cls_path = self.root / cls_name
            for f in sorted(cls_path.iterdir()):
                if f.suffix.lower() in self.VALID_EXT:
                    self.samples.append((str(f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale per problem spec
        img_rgb = Image.merge("RGB", [img, img, img])  # 3-ch for pretrained

        if self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = T.ToTensor()(img_rgb)

        return img_tensor, y


class WaferDefectDataset(Dataset):
    """Multi-class dataset: defect images only, mapped to 0..N-1."""
    VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (path, defect_label)

        # Collect defect class dirs (everything except 'good')
        class_dirs = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "good"
        ])
        self.classes = class_dirs  # e.g. ['defect1','defect10',...]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_name in class_dirs:
            label = self.class_to_idx[cls_name]
            cls_path = self.root / cls_name
            for f in sorted(cls_path.iterdir()):
                if f.suffix.lower() in self.VALID_EXT:
                    self.samples.append((str(f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")
        img_rgb = Image.merge("RGB", [img, img, img])

        if self.transform is not None:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = T.ToTensor()(img_rgb)

        return img_tensor, y


# ---------------------------------------------------------------------------
# Transforms — minimal, no heavy augmentation
# ---------------------------------------------------------------------------
def get_train_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_resnet50(num_classes=1):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        with autocast("cuda", enabled=use_amp):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

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

        preds = (logits > 0).long()
        correct += (preds == y.long()).sum().item()
        running_loss += loss.item() * y.size(0)
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False, threshold=0.0):
    model.eval()
    correct = total = 0
    all_logits, all_true = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(x).squeeze(1)

        all_logits.extend(logits.cpu().tolist())
        all_true.extend(y.tolist())

        preds = (logits > threshold).long().cpu()
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc, np.array(all_logits), np.array(all_true)


# ---------------------------------------------------------------------------
# Stage 2: Multi-class training / evaluation
# ---------------------------------------------------------------------------
def train_one_epoch_multiclass(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        with autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

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

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        running_loss += loss.item() * y.size(0)
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_multiclass(model, loader, device, use_amp=False):
    model.eval()
    correct = total = 0
    all_preds, all_true = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(x)

        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(y.tolist())

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total if total > 0 else 0.0
    return acc, np.array(all_preds), np.array(all_true)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_curves(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], linewidth=2, color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", linewidth=2, color="#1f77b4")
    axes[1].plot(history["val_acc"], label="Val", linewidth=2, color="#2ca02c")
    axes[1].axhline(y=0.728, color="gray", linestyle="--", alpha=0.5, label="Majority baseline (72.8%)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "resnet50_binary_curves.png"), dpi=200)
    plt.close()
    print(f"  Saved training curves to {out_dir}/resnet50_binary_curves.png")


def plot_confusion(cm, out_dir, title="Binary Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["good", "defect"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "resnet50_binary_confusion.png"), dpi=200)
    plt.close()


def plot_threshold_sweep(thresholds, accuracies, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, accuracies, linewidth=2, color="#1f77b4")
    best_idx = np.argmax(accuracies)
    ax.axvline(x=thresholds[best_idx], color="red", linestyle="--", alpha=0.7,
               label=f"Best threshold={thresholds[best_idx]:.3f} ({accuracies[best_idx]:.4f})")
    ax.axhline(y=0.728, color="gray", linestyle="--", alpha=0.5, label="Majority baseline (72.8%)")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Accuracy")
    ax.set_title("Threshold Sweep on Validation Set")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "resnet50_threshold_sweep.png"), dpi=200)
    plt.close()


def plot_stage2_curves(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], linewidth=2, color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Stage 2 Training Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", linewidth=2, color="#1f77b4")
    axes[1].plot(history["val_acc"], label="Val", linewidth=2, color="#2ca02c")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Stage 2 Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage2_defect_curves.png"), dpi=200)
    plt.close()
    print(f"  Saved Stage 2 curves to {out_dir}/stage2_defect_curves.png")


def plot_stage2_confusion(cm, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Stage 2 Defect Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage2_defect_confusion.png"), dpi=200)
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
    ax.set_ylabel("Score"); ax.set_title("Stage 2 Per-Class Metrics")
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage2_per_class_metrics.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Config ---
    DATA_ROOT   = "Data"
    OUT_DIR     = "reports/resnet50_binary"
    IMG_SIZE    = 224
    BATCH_SIZE  = 128
    FREEZE_EPOCHS = 5
    TOTAL_EPOCHS  = 20
    LR_FROZEN   = 1e-3
    LR_UNFROZEN = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS  = 8
    SEED         = 42
    VAL_SPLIT    = 0.15
    TEST_SPLIT   = 0.15

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

    # --- Dataset ---
    full_ds = WaferBinaryDataset(DATA_ROOT, transform=None)
    print(f"\nTotal images: {len(full_ds)}")
    all_targets = [s[1] for s in full_ds.samples]
    counts = Counter(all_targets)
    print(f"  good:   {counts[0]} ({counts[0]/len(full_ds):.1%})")
    print(f"  defect: {counts[1]} ({counts[1]/len(full_ds):.1%})")
    print(f"  Majority baseline: {max(counts.values())/len(full_ds):.1%}")

    # --- Stratified Split ---
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
    trainval_idx, test_idx = next(sss_test.split(range(len(full_ds)), all_targets))

    trainval_targets = [all_targets[i] for i in trainval_idx]
    relative_val = VAL_SPLIT / (1 - TEST_SPLIT)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=SEED)
    train_idx_rel, val_idx_rel = next(sss_val.split(range(len(trainval_idx)), trainval_targets))
    train_idx = [trainval_idx[i] for i in train_idx_rel]
    val_idx   = [trainval_idx[i] for i in val_idx_rel]

    train_targets = [all_targets[i] for i in train_idx]
    val_targets   = [all_targets[i] for i in val_idx]
    test_targets  = [all_targets[i] for i in test_idx]

    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    for split_name, tgts in [("train", train_targets), ("val", val_targets), ("test", test_targets)]:
        c = Counter(tgts)
        print(f"  {split_name:>5s}: good={c[0]}, defect={c[1]}")

    # --- DataLoaders ---
    train_ds = WaferBinaryDataset(DATA_ROOT, transform=get_train_transform(IMG_SIZE))
    val_ds   = WaferBinaryDataset(DATA_ROOT, transform=get_val_transform(IMG_SIZE))

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(val_ds, val_idx)
    test_subset  = Subset(val_ds, test_idx)  # val transform for test

    # Balanced sampler
    bin_counts = Counter(train_targets)
    cw = {c: 1.0 / max(bin_counts.get(c, 1), 1) for c in range(2)}
    sw = [cw[t] for t in train_targets]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # --- Model ---
    model = build_resnet50().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nResNet50: {n_params:,} params, fc: Linear(2048, 1)")

    # --- Loss ---
    criterion = nn.BCEWithLogitsLoss()
    print(f"Loss: BCEWithLogitsLoss (no weighting)")

    # --- Training ---
    scaler = GradScaler("cuda", enabled=use_amp)
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": [], "phase": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(OUT_DIR, "best_resnet50_binary.pt")

    print(f"\n{'='*70}")
    print(f"PHASE 1: Frozen backbone, train classifier only (epochs 1-{FREEZE_EPOCHS})")
    print(f"  lr={LR_FROZEN}, weight_decay={WEIGHT_DECAY}")
    print(f"{'='*70}")

    # Freeze all except fc
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (frozen phase): {trainable:,}")

    optimizer = optim.AdamW(model.fc.parameters(), lr=LR_FROZEN, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, FREEZE_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                 device, scaler, use_amp)
        val_acc, _, _ = evaluate(model, val_loader, device, use_amp)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(LR_FROZEN)
        history["phase"].append("frozen")

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            improved = " *BEST*"

        print(f"  Ep {epoch:02d}/{TOTAL_EPOCHS} [FROZEN] | Loss {train_loss:.4f} | "
              f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
              f"LR {LR_FROZEN:.1e} | {elapsed:.1f}s{improved}")

    # --- Phase 2: Unfreeze all ---
    print(f"\n{'='*70}")
    print(f"PHASE 2: Unfrozen, full fine-tuning (epochs {FREEZE_EPOCHS+1}-{TOTAL_EPOCHS})")
    print(f"  lr={LR_UNFROZEN}, weight_decay={WEIGHT_DECAY}")
    print(f"{'='*70}")

    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params (unfrozen phase): {trainable:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR_UNFROZEN, weight_decay=WEIGHT_DECAY)

    for epoch in range(FREEZE_EPOCHS + 1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                 device, scaler, use_amp)
        val_acc, _, _ = evaluate(model, val_loader, device, use_amp)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(LR_UNFROZEN)
        history["phase"].append("unfrozen")

        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            improved = " *BEST*"

        print(f"  Ep {epoch:02d}/{TOTAL_EPOCHS} [UNFROZEN] | Loss {train_loss:.4f} | "
              f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
              f"LR {LR_UNFROZEN:.1e} | {elapsed:.1f}s{improved}")

    # --- Load best model ---
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    print(f"\nLoaded best model (val acc: {best_val_acc:.4f})")

    # --- Plot training curves ---
    plot_training_curves(history, OUT_DIR)

    # ==================================================================
    # TEST EVALUATION
    # ==================================================================
    print(f"\n{'='*70}")
    print("TEST EVALUATION")
    print(f"{'='*70}")

    test_acc, test_logits, test_true = evaluate(model, test_loader, device, use_amp)
    test_preds = (test_logits > 0).astype(int)
    print(f"  Test Accuracy (threshold=0.0): {test_acc:.4f}")

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

    val_acc_default, val_logits, val_true = evaluate(model, val_loader, device, use_amp)

    thresholds = np.linspace(-3.0, 3.0, 100)
    accuracies = []
    for t in thresholds:
        preds = (val_logits > t).astype(int)
        acc = (preds == val_true).mean()
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    best_t_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_t_idx]
    best_thresh_acc = accuracies[best_t_idx]

    print(f"  Default threshold (0.0): Val Acc = {val_acc_default:.4f}")
    print(f"  Best threshold ({best_threshold:.3f}): Val Acc = {best_thresh_acc:.4f}")
    print(f"  Improvement: +{(best_thresh_acc - val_acc_default)*100:.2f}%")

    # Apply best threshold to test set
    test_preds_tuned = (test_logits > best_threshold).astype(int)
    test_acc_tuned = (test_preds_tuned == test_true).mean()
    cm_tuned = confusion_matrix(test_true, test_preds_tuned, labels=[0, 1])

    print(f"\n  Test Accuracy (tuned threshold={best_threshold:.3f}): {test_acc_tuned:.4f}")
    print(f"  Confusion Matrix (tuned):")
    print(f"              Predicted")
    print(f"              good  defect")
    print(f"  True good   {cm_tuned[0,0]:>5d}  {cm_tuned[0,1]:>5d}")
    print(f"  True defect {cm_tuned[1,0]:>5d}  {cm_tuned[1,1]:>5d}")

    plot_threshold_sweep(thresholds, accuracies, OUT_DIR)

    # ==================================================================
    # STAGE 2: DEFECT SUBTYPE CLASSIFIER
    # ==================================================================
    S2_OUT_DIR = os.path.join(OUT_DIR, "stage2")
    os.makedirs(S2_OUT_DIR, exist_ok=True)

    print(f"\n\n{'#'*70}")
    print(f"#  STAGE 2: DEFECT SUBTYPE CLASSIFIER")
    print(f"#  CrossEntropyLoss with inverse-frequency class weights")
    print(f"{'#'*70}")

    # --- Stage 2 Dataset ---
    defect_ds_raw = WaferDefectDataset(DATA_ROOT, transform=None)
    num_defect_classes = len(defect_ds_raw.classes)
    defect_class_names = defect_ds_raw.classes
    print(f"\nDefect images: {len(defect_ds_raw)}")
    print(f"Defect classes ({num_defect_classes}): {defect_class_names}")

    all_defect_targets = [s[1] for s in defect_ds_raw.samples]
    defect_counts = Counter(all_defect_targets)
    for i, name in enumerate(defect_class_names):
        warn = " << RARE" if defect_counts[i] < 20 else ""
        print(f"  {name:>10s}: {defect_counts[i]:>5d}{warn}")

    # --- Stratified Split for defect data ---
    sss_def_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
    def_trainval_idx, def_test_idx = next(sss_def_test.split(
        range(len(defect_ds_raw)), all_defect_targets))

    def_trainval_targets = [all_defect_targets[i] for i in def_trainval_idx]
    sss_def_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=SEED)
    def_train_idx_rel, def_val_idx_rel = next(sss_def_val.split(
        range(len(def_trainval_idx)), def_trainval_targets))
    def_train_idx = [def_trainval_idx[i] for i in def_train_idx_rel]
    def_val_idx = [def_trainval_idx[i] for i in def_val_idx_rel]

    def_train_targets = [all_defect_targets[i] for i in def_train_idx]
    def_val_targets = [all_defect_targets[i] for i in def_val_idx]
    def_test_targets = [all_defect_targets[i] for i in def_test_idx]

    print(f"\nStage 2 split: train={len(def_train_idx)}, val={len(def_val_idx)}, test={len(def_test_idx)}")
    for i, name in enumerate(defect_class_names):
        tr = sum(1 for t in def_train_targets if t == i)
        va = sum(1 for t in def_val_targets if t == i)
        te = sum(1 for t in def_test_targets if t == i)
        print(f"  {name:>10s}: train={tr:>4d}, val={va:>3d}, test={te:>3d}")

    # --- Stage 2 DataLoaders ---
    def_train_ds = WaferDefectDataset(DATA_ROOT, transform=get_train_transform(IMG_SIZE))
    def_val_ds = WaferDefectDataset(DATA_ROOT, transform=get_val_transform(IMG_SIZE))

    def_train_subset = Subset(def_train_ds, def_train_idx)
    def_val_subset = Subset(def_val_ds, def_val_idx)
    def_test_subset = Subset(def_val_ds, def_test_idx)

    # Balanced sampler for defect training
    def_bin_counts = Counter(def_train_targets)
    def_cw = {c: 1.0 / max(def_bin_counts.get(c, 1), 1) for c in range(num_defect_classes)}
    def_sw = [def_cw[t] for t in def_train_targets]
    def_sampler = WeightedRandomSampler(def_sw, len(def_sw), replacement=True)

    def_train_loader = DataLoader(def_train_subset, batch_size=BATCH_SIZE, sampler=def_sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                                  persistent_workers=NUM_WORKERS > 0)
    def_val_loader = DataLoader(def_val_subset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True,
                                persistent_workers=NUM_WORKERS > 0)
    def_test_loader = DataLoader(def_test_subset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

    # --- Stage 2 Model ---
    model_s2 = build_resnet50(num_classes=num_defect_classes).to(device)
    print(f"\nStage 2 ResNet50: fc=Linear(2048, {num_defect_classes})")

    # --- Class weights for CrossEntropyLoss ---
    class_freqs = torch.tensor(
        [def_bin_counts.get(c, 1) for c in range(num_defect_classes)], dtype=torch.float32)
    class_weights = 1.0 / class_freqs
    class_weights = class_weights / class_weights.sum() * num_defect_classes  # normalize
    class_weights = class_weights.to(device)

    print(f"\nCrossEntropyLoss class weights:")
    for i, name in enumerate(defect_class_names):
        print(f"  {name:>10s}: count={def_bin_counts.get(i,0):>4d}, weight={class_weights[i]:.4f}")

    criterion_s2 = nn.CrossEntropyLoss(weight=class_weights)

    # --- Stage 2 Training ---
    scaler_s2 = GradScaler("cuda", enabled=use_amp)
    s2_history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": [], "phase": []}
    best_s2_val_acc = 0.0
    best_s2_path = os.path.join(S2_OUT_DIR, "best_resnet50_defect.pt")

    # Phase 1: Frozen backbone
    print(f"\n{'='*70}")
    print(f"STAGE 2 PHASE 1: Frozen backbone (epochs 1-{FREEZE_EPOCHS})")
    print(f"  lr={LR_FROZEN}, weight_decay={WEIGHT_DECAY}")
    print(f"{'='*70}")

    for param in model_s2.parameters():
        param.requires_grad = False
    for param in model_s2.fc.parameters():
        param.requires_grad = True

    trainable_s2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    print(f"  Trainable params (frozen): {trainable_s2:,}")

    optimizer_s2 = optim.AdamW(model_s2.fc.parameters(), lr=LR_FROZEN, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, FREEZE_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch_multiclass(
            model_s2, def_train_loader, criterion_s2, optimizer_s2, device, scaler_s2, use_amp)
        val_acc_s2, _, _ = evaluate_multiclass(model_s2, def_val_loader, device, use_amp)
        elapsed = time.time() - t0

        s2_history["train_loss"].append(train_loss)
        s2_history["train_acc"].append(train_acc)
        s2_history["val_acc"].append(val_acc_s2)
        s2_history["lr"].append(LR_FROZEN)
        s2_history["phase"].append("frozen")

        improved = ""
        if val_acc_s2 > best_s2_val_acc:
            best_s2_val_acc = val_acc_s2
            torch.save(model_s2.state_dict(), best_s2_path)
            improved = " *BEST*"

        print(f"  Ep {epoch:02d}/{TOTAL_EPOCHS} [FROZEN] | Loss {train_loss:.4f} | "
              f"Train {train_acc:.4f} | Val {val_acc_s2:.4f} | "
              f"LR {LR_FROZEN:.1e} | {elapsed:.1f}s{improved}")

    # Phase 2: Unfreeze all
    print(f"\n{'='*70}")
    print(f"STAGE 2 PHASE 2: Unfrozen (epochs {FREEZE_EPOCHS+1}-{TOTAL_EPOCHS})")
    print(f"  lr={LR_UNFROZEN}, weight_decay={WEIGHT_DECAY}")
    print(f"{'='*70}")

    for param in model_s2.parameters():
        param.requires_grad = True

    trainable_s2 = sum(p.numel() for p in model_s2.parameters() if p.requires_grad)
    print(f"  Trainable params (unfrozen): {trainable_s2:,}")

    optimizer_s2 = optim.AdamW(model_s2.parameters(), lr=LR_UNFROZEN, weight_decay=WEIGHT_DECAY)

    for epoch in range(FREEZE_EPOCHS + 1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch_multiclass(
            model_s2, def_train_loader, criterion_s2, optimizer_s2, device, scaler_s2, use_amp)
        val_acc_s2, _, _ = evaluate_multiclass(model_s2, def_val_loader, device, use_amp)
        elapsed = time.time() - t0

        s2_history["train_loss"].append(train_loss)
        s2_history["train_acc"].append(train_acc)
        s2_history["val_acc"].append(val_acc_s2)
        s2_history["lr"].append(LR_UNFROZEN)
        s2_history["phase"].append("unfrozen")

        improved = ""
        if val_acc_s2 > best_s2_val_acc:
            best_s2_val_acc = val_acc_s2
            torch.save(model_s2.state_dict(), best_s2_path)
            improved = " *BEST*"

        print(f"  Ep {epoch:02d}/{TOTAL_EPOCHS} [UNFROZEN] | Loss {train_loss:.4f} | "
              f"Train {train_acc:.4f} | Val {val_acc_s2:.4f} | "
              f"LR {LR_UNFROZEN:.1e} | {elapsed:.1f}s{improved}")

    # --- Load best Stage 2 model ---
    model_s2.load_state_dict(torch.load(best_s2_path, map_location=device, weights_only=True))
    print(f"\nLoaded best Stage 2 model (val acc: {best_s2_val_acc:.4f})")

    # --- Stage 2 plots ---
    plot_stage2_curves(s2_history, S2_OUT_DIR)

    # --- Stage 2 Test Evaluation ---
    print(f"\n{'='*70}")
    print("STAGE 2 TEST EVALUATION")
    print(f"{'='*70}")

    s2_test_acc, s2_test_preds, s2_test_true = evaluate_multiclass(
        model_s2, def_test_loader, device, use_amp)
    print(f"  Stage 2 Test Accuracy: {s2_test_acc:.4f}")
    print(f"  Random chance ({num_defect_classes} classes): {1/num_defect_classes:.1%}")

    s2_cm = confusion_matrix(s2_test_true, s2_test_preds, labels=list(range(num_defect_classes)))
    s2_report = classification_report(
        s2_test_true, s2_test_preds, target_names=defect_class_names, digits=4)
    s2_report_dict = classification_report(
        s2_test_true, s2_test_preds, target_names=defect_class_names, digits=4, output_dict=True)
    print(f"\n{s2_report}")

    plot_stage2_confusion(s2_cm, defect_class_names, S2_OUT_DIR)
    plot_per_class_metrics(s2_report_dict, defect_class_names, S2_OUT_DIR)

    # --- Save combined summary ---
    summary = {
        "stage1_binary": {
            "model": "resnet50",
            "img_size": IMG_SIZE,
            "total_epochs": TOTAL_EPOCHS,
            "freeze_epochs": FREEZE_EPOCHS,
            "lr_frozen": LR_FROZEN,
            "lr_unfrozen": LR_UNFROZEN,
            "weight_decay": WEIGHT_DECAY,
            "best_val_acc": round(best_val_acc, 4),
            "test_acc_default_threshold": round(float(test_acc), 4),
            "test_acc_tuned_threshold": round(float(test_acc_tuned), 4),
            "best_threshold": round(float(best_threshold), 3),
            "majority_baseline": round(max(counts.values()) / len(full_ds), 4),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
        },
        "stage2_defect": {
            "model": "resnet50",
            "num_classes": num_defect_classes,
            "class_names": defect_class_names,
            "loss": "CrossEntropyLoss (weighted)",
            "class_weights": {defect_class_names[i]: round(class_weights[i].item(), 4)
                              for i in range(num_defect_classes)},
            "best_val_acc": round(best_s2_val_acc, 4),
            "test_acc": round(s2_test_acc, 4),
            "random_chance": round(1 / num_defect_classes, 4),
            "train_samples": len(def_train_idx),
            "val_samples": len(def_val_idx),
            "test_samples": len(def_test_idx),
            "train_class_counts": {defect_class_names[i]: def_bin_counts.get(i, 0)
                                   for i in range(num_defect_classes)},
        },
        "history_stage1": {k: [round(v, 4) if isinstance(v, float) else v for v in vals]
                           for k, vals in history.items()},
        "history_stage2": {k: [round(v, 4) if isinstance(v, float) else v for v in vals]
                           for k, vals in s2_history.items()},
    }
    with open(os.path.join(OUT_DIR, "two_stage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  --- Stage 1 (Binary: good vs defect) ---")
    print(f"  Majority baseline:     {max(counts.values())/len(full_ds):.1%}")
    print(f"  Best val accuracy:     {best_val_acc:.4f}")
    print(f"  Test acc (default):    {test_acc:.4f}")
    print(f"  Test acc (tuned):      {test_acc_tuned:.4f}")
    print(f"  Best threshold:        {best_threshold:.3f}")
    print(f"")
    print(f"  --- Stage 2 (Defect subtype: {num_defect_classes} classes) ---")
    print(f"  Random chance:         {1/num_defect_classes:.1%}")
    print(f"  Best val accuracy:     {best_s2_val_acc:.4f}")
    print(f"  Test accuracy:         {s2_test_acc:.4f}")
    print(f"")
    print(f"  All outputs saved to:  {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
